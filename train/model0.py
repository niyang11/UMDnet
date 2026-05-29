#师兄的模型
import torch.nn as nn
import torch
from Wave_Unet.model.Wave_Unet_parts import *
import os
import torch.nn.functional as F
from Wave_Unet.model.threshold import  SoftThreshold

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01)

        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self,  channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),

            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

# class UpSamplingLayer(nn.Module):
#     def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2, output_padding=0):
#         super(UpSamplingLayer, self).__init__()
#         self.main = nn.Sequential(
#             # 使用反向卷积代替普通卷积
#             nn.ConvTranspose1d(channel_in, channel_out, kernel_size=kernel_size,
#                                stride=stride, padding=padding, output_padding=output_padding),
#             nn.BatchNorm1d(channel_out),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#         )
#
#     def forward(self, ipt):
#         return self.main(ipt)

class WaveUNet0(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(WaveUNet0, self).__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(encoder_in_channels_list[i],
                                  encoder_out_channels_list[i])
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)


                )


        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]


        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )


    def forward(self, input):
        #input = input.permute(0, 2, 1)
        # print("TransEncoder input shape:", input.shape)
        tmp = []
        o = input

        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            o=o[:,:, ::2]
        o = self.middle(o)

        for i in range(self.n_layers):
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)

            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)
        o = torch.cat([o, input], dim=1)

        o = self.out(o)
        return o

if __name__ == '__main__':

     WaveUNet0 = WaveUNet0(n_layers=12, channels_interval=24)
     print(WaveUNet0)









