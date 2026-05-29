import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba2 import Mamba2

__all__ = ["WaveUMamba"]


# ------------------------------
# 基础卷积模块：下采样 / 上采样
# ------------------------------
class DownSamplingLayer(nn.Module):
    """
    简单的一维卷积下采样块：
    Conv1d + BN + LeakyReLU
    不包含 stride=2，下采样在主网络里用 ::2 完成，保持与原 WaveUNet0 一致。
    """
    def __init__(self,
                 channel_in: int,
                 channel_out: int,
                 dilation: int = 1,
                 kernel_size: int = 15,
                 stride: int = 1,
                 padding: int = 7):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        return self.main(x)


class UpSamplingLayer(nn.Module):
    """
    升采样后的卷积块：
    先在 WaveUMamba 里用 F.interpolate 上采样，再用本模块做卷积 + BN + 激活。
    """
    def __init__(self,
                 channel_in: int,
                 channel_out: int,
                 kernel_size: int = 5,
                 stride: int = 1,
                 padding: int = 2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        return self.main(x)


# ------------------------------
# 更稳定的 1D Mamba Block
# ------------------------------
class MambaBlock1D(nn.Module):
    """
    1D 信号用的 Mamba2 包装：
    输入 (B, C, L) -> 转成 (B, L, C) 走 Mamba2 -> 再转回 (B, C, L)

    关键改进：
    - LayerNorm 标准化
    - Dropout
    - 可学习残差缩放 gamma（初始值较小，避免一开始把特征“搅太狠”）

    这样对 NMR 弱峰来说，更不容易被 Mamba 直接抹平。
    """
    def __init__(self,
                 channels: int,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 8,
                 dropout: float = 0.1,
                 residual_scale: float = 0.1):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

        # headdim 需要满足 mamba2 内部 d_ssm % headdim == 0
        # 这里默认 headdim=8，配合 channels=24,48,...,288 是安全的
        self.mamba = Mamba2(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        self.dropout = nn.Dropout(dropout)
        # 残差缩放参数，初始给一个比较小的值，避免一开始改动过大
        self.gamma = nn.Parameter(
            torch.tensor(residual_scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        """
        B, C, L = x.shape
        residual = x

        # (B, C, L) -> (B, L, C)
        y = x.transpose(1, 2).contiguous()
        y = self.norm(y)
        y = self.mamba(y)                  # (B, L, C)
        y = self.dropout(y)
        y = y.transpose(1, 2).contiguous() # (B, C, L)

        # 残差连接 + 缩放
        return residual + self.gamma * y


# ------------------------------
# Wave-U-Net + Mamba2 主体
# ------------------------------
class WaveUMamba(nn.Module):
    """
    Wave-U-Net 风格的 1D U-Net + Mamba2
    专门为 NMR 1D 光谱去噪设计.

    默认配置：
    - 输入:  (B, 1, 8192)
    - 输出:  (B, 1, 8192)
    - 每层通道数: 24, 48, 72, ..., 288  (可调 channels_interval)
    - 手工下采样: encoder 里用 x[:, :, ::2] 做 2 倍下采样
    - 上采样: decoder 里用 F.interpolate(scale_factor=2, mode="linear")

    Mamba 只在相对“深”的层启用 (由 mamba_from 控制)，
    避免浅层就过度建模，提高弱峰恢复稳定性。
    """

    def __init__(
        self,
        n_layers: int = 12,
        channels_interval: int = 24,
        mamba_from: int = 4,          # 从第几层开始插 Mamba (0-based)
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 8,
        mamba_dropout: float = 0.1,
        residual: bool = True,        # True: 模型学习 residual = clean - noisy
        out_activation: str = "tanh", # "none" 或 "tanh"
    ):
        super().__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval
        self.mamba_from = mamba_from
        self.residual = residual

        # ---------------- Encoder ----------------
        encoder_in_channels_list = [1] + [
            i * self.channels_interval for i in range(1, self.n_layers)
        ]
        encoder_out_channels_list = [
            i * self.channels_interval for i in range(1, self.n_layers + 1)
        ]

        self.encoder = nn.ModuleList()
        self.encoder_mamba = nn.ModuleList()

        for i in range(self.n_layers):
            cin = encoder_in_channels_list[i]
            cout = encoder_out_channels_list[i]

            # 卷积下采样块
            self.encoder.append(DownSamplingLayer(cin, cout))

            # 深层才上 Mamba，浅层保持纯卷积，避免过拟合 & 过度平滑
            if i >= self.mamba_from:
                self.encoder_mamba.append(
                    MambaBlock1D(
                        cout,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        headdim=headdim,
                        dropout=mamba_dropout,
                        residual_scale=0.1,
                    )
                )
            else:
                self.encoder_mamba.append(nn.Identity())

        # ---------------- Bottleneck ----------------
        mid_channels = self.n_layers * self.channels_interval
        self.middle_conv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, 15, stride=1, padding=7),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.middle_mamba = MambaBlock1D(
            mid_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            dropout=mamba_dropout,
            residual_scale=0.1,
        )

        # ---------------- Decoder ----------------
        decoder_in_channels_list = [
            (2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)
        ] + [2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = nn.ModuleList()
        self.decoder_mamba = nn.ModuleList()

        for i in range(self.n_layers):
            cin = decoder_in_channels_list[i]
            cout = decoder_out_channels_list[i]

            self.decoder.append(UpSamplingLayer(cin, cout))

            # 与 encoder 对应的深层位置同样加 Mamba
            enc_idx = self.n_layers - 1 - i
            if enc_idx >= self.mamba_from:
                self.decoder_mamba.append(
                    MambaBlock1D(
                        cout,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        headdim=headdim,
                        dropout=mamba_dropout,
                        residual_scale=0.1,
                    )
                )
            else:
                self.decoder_mamba.append(nn.Identity())

        # ---------------- Output head ----------------
        out_layers = [nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1)]
        if out_activation.lower() == "tanh":
            out_layers.append(nn.Tanh())
        self.out = nn.Sequential(*out_layers)

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L)    这里 L 默认 8192，但可以更长/更短（只要是 2^n 对齐即可）
        """
        inp = x
        skips = []
        o = x

        # ----- Encoder -----
        for i in range(self.n_layers):
            o = self.encoder[i](o)          # 卷积
            o = self.encoder_mamba[i](o)    # 可能加 Mamba
            skips.append(o)
            o = o[:, :, ::2]                # 手工下采样 /2

        # ----- Bottleneck -----
        o = self.middle_conv(o)
        o = self.middle_mamba(o)

        # ----- Decoder -----
        for i in range(self.n_layers):
            # 先线性插值上采样
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)

            skip = skips[self.n_layers - 1 - i]

            # 长度对齐（插值会有可能差 1 点）
            if o.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - o.shape[-1]
                if diff > 0:
                    o = F.pad(o, (0, diff))
                elif diff < 0:
                    o = o[:, :, :skip.shape[-1]]

            # U-Net 的 skip 连接：concat
            o = torch.cat([o, skip], dim=1)
            o = self.decoder[i](o)
            o = self.decoder_mamba[i](o)

        # 和原始输入对齐长度
        if o.shape[-1] != inp.shape[-1]:
            diff = inp.shape[-1] - o.shape[-1]
            if diff > 0:
                o = F.pad(o, (0, diff))
            elif diff < 0:
                o = o[:, :, :inp.shape[-1]]

        # 拼回原始输入，帮助保持整体形状
        o = torch.cat([o, inp], dim=1)
        out = self.out(o)   # 预测的是“修正量” or “直接输出”

        if self.residual:
            # 模型学习 residual = clean - noisy
            return inp + out
        else:
            return out


