import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_mamba1():
    from mamba_ssm.modules.mamba_simple import Mamba
    return Mamba


def _import_mamba2():
    from mamba_ssm.modules.mamba2 import Mamba2
    return Mamba2


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size, stride, padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return self.main(x)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size, stride, padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class MambaBlock1D(nn.Module):
    """
    (B,C,L) -> (B,L,C) -> Mamba -> (B,C,L)

    mamba_internal_residual:
      - True  (R1): y = x + gamma * Mamba(LN(x))
      - False (R0): y = Mamba(LN(x))
    """
    def __init__(
        self,
        channels: int,
        mamba_version: int = 2,          # 1 or 2
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 8,                # only for Mamba2
        dropout: float = 0.1,
        mamba_internal_residual: bool = True,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.mamba_internal_residual = bool(mamba_internal_residual)

        if int(mamba_version) == 1:
            Mamba = _import_mamba1()
            self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        elif int(mamba_version) == 2:
            Mamba2 = _import_mamba2()
            self.mamba = Mamba2(
                d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim
            )
        else:
            raise ValueError("mamba_version must be 1 or 2")

        self.gamma = (
            nn.Parameter(torch.tensor(residual_scale, dtype=torch.float32))
            if self.mamba_internal_residual
            else None
        )

    def forward(self, x):
        residual = x
        y = x.transpose(1, 2).contiguous()   # (B,L,C)
        y = self.norm(y)
        y = self.mamba(y)
        y = self.dropout(y)
        y = y.transpose(1, 2).contiguous()   # (B,C,L)

        if self.mamba_internal_residual:
            return residual + self.gamma * y
        return y


class WaveUMambaFlex(nn.Module):
    """
    支持：
      - mamba_version: 1/2
      - mamba_internal_residual: R0/R1（指 MambaBlock 内部残差）
      - mamba_from: 从第 k 层开始融合（0-based）

    建议：out_residual 在本轮固定 True（避免变量混杂）。
    """
    def __init__(
        self,
        n_layers: int = 12,
        channels_interval: int = 24,
        mamba_from: int = 0,
        mamba_version: int = 2,
        mamba_internal_residual: bool = True,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 8,
        mamba_dropout: float = 0.1,
        out_residual: bool = True,
        out_activation: str = "none",
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval
        self.mamba_from = int(mamba_from)
        self.out_residual = bool(out_residual)

        encoder_in = [1] + [i * channels_interval for i in range(1, n_layers)]
        encoder_out = [i * channels_interval for i in range(1, n_layers + 1)]

        self.encoder = nn.ModuleList()
        self.encoder_mamba = nn.ModuleList()
        for i in range(n_layers):
            cin, cout = encoder_in[i], encoder_out[i]
            self.encoder.append(DownSamplingLayer(cin, cout))
            if i >= self.mamba_from:
                self.encoder_mamba.append(
                    MambaBlock1D(
                        cout, mamba_version=mamba_version, d_state=d_state, d_conv=d_conv,
                        expand=expand, headdim=headdim, dropout=mamba_dropout,
                        mamba_internal_residual=mamba_internal_residual, residual_scale=gamma_init
                    )
                )
            else:
                self.encoder_mamba.append(nn.Identity())

        mid_channels = n_layers * channels_interval
        self.middle_conv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, 15, stride=1, padding=7),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.middle_mamba = MambaBlock1D(
            mid_channels, mamba_version=mamba_version, d_state=d_state, d_conv=d_conv,
            expand=expand, headdim=headdim, dropout=mamba_dropout,
            mamba_internal_residual=mamba_internal_residual, residual_scale=gamma_init
        )

        decoder_in = [(2 * i + 1) * channels_interval for i in range(1, n_layers)]
        decoder_in += [2 * n_layers * channels_interval]
        decoder_in = decoder_in[::-1]
        decoder_out = encoder_out[::-1]

        self.decoder = nn.ModuleList()
        self.decoder_mamba = nn.ModuleList()
        for i in range(n_layers):
            cin, cout = decoder_in[i], decoder_out[i]
            self.decoder.append(UpSamplingLayer(cin, cout))

            enc_idx = n_layers - 1 - i
            if enc_idx >= self.mamba_from:
                self.decoder_mamba.append(
                    MambaBlock1D(
                        cout, mamba_version=mamba_version, d_state=d_state, d_conv=d_conv,
                        expand=expand, headdim=headdim, dropout=mamba_dropout,
                        mamba_internal_residual=mamba_internal_residual, residual_scale=gamma_init
                    )
                )
            else:
                self.decoder_mamba.append(nn.Identity())

        out_layers = [nn.Conv1d(1 + channels_interval, 1, kernel_size=1, stride=1)]
        if out_activation.lower() == "tanh":
            out_layers.append(nn.Tanh())
        self.out = nn.Sequential(*out_layers)

    def forward(self, x):
        inp = x
        skips = []
        o = x

        for i in range(self.n_layers):
            o = self.encoder[i](o)
            o = self.encoder_mamba[i](o)
            skips.append(o)
            o = o[:, :, ::2]

        o = self.middle_conv(o)
        o = self.middle_mamba(o)

        for i in range(self.n_layers):
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            skip = skips[self.n_layers - 1 - i]

            if o.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - o.shape[-1]
                if diff > 0:
                    o = F.pad(o, (0, diff))
                elif diff < 0:
                    o = o[:, :, :skip.shape[-1]]

            o = torch.cat([o, skip], dim=1)
            o = self.decoder[i](o)
            o = self.decoder_mamba[i](o)

        if o.shape[-1] != inp.shape[-1]:
            diff = inp.shape[-1] - o.shape[-1]
            if diff > 0:
                o = F.pad(o, (0, diff))
            elif diff < 0:
                o = o[:, :, :inp.shape[-1]]

        o = torch.cat([o, inp], dim=1)
        out = self.out(o)
        return inp + out if self.out_residual else out
