import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from torch.nn import init
import numpy as np

fn = 8192

def plot_layers_result(layers_result, save_dir):
    """
    在所有迭代结束后，绘制所有层的输出图像
    :param layers_result: 保存了每层输出的列表
    :param save_dir: 保存图像的目录路径
    """
    plt.figure(figsize=(10, len(layers_result) * 2))  # 设置图像大小
    offset = 0  # 初始偏移量

    for layer_idx, layer_output in enumerate(layers_result):
        # 假设每个输出 tensor 都是二维的(batch_size, feature_dim)，取第一个样本
        if layer_output.dim() == 2:
            layer_output = layer_output[0].cpu().detach().numpy()
            plt.subplot(len(layers_result), 1, layer_idx + 1)  # 每层一个子图
            plt.plot(abs(layer_output + offset))  # 添加垂直偏移
            plt.title(f'Layer {layer_idx} Output')
            plt.xlabel('Feature')
            plt.ylabel('Value')
            plt.grid(True)

        # 如果是三维 tensor，可以处理类似之前的方式
        elif layer_output.dim() == 3:
            layer_output = layer_output[0].cpu().detach().numpy()
            # 显示第一个通道的输出
            layer_output = layer_output[0]  # 选择第一个通道
            plt.subplot(len(layers_result), 1, layer_idx + 1)  # 每层一个子图
            plt.plot(abs(layer_output + offset))  # 添加垂直偏移
            plt.title(f'Layer {layer_idx} Output (First Channel)')
            plt.xlabel('Feature')
            plt.ylabel('Value')
            plt.grid(True)

        # 增加偏移量
        offset += 1

    # 保存所有图像
    plt.tight_layout()  # 调整子图之间的间距
    plt.savefig(os.path.join(save_dir, 'all_layers_output.png'))
    plt.close()  # 关闭当前图像，避免内存溢出

save_dir = '/home/project/ny/pythonProject/ISTA_net/view/fqy_MS80000/plot'
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 15)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 15)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 15)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 15)))

    def forward(self, x, PhiTPhi, PhiTb):

        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        #print("Shape of  x3 output ( x):", x.shape)
        x_com = x + self.lambda_step * PhiTb
        #print("Shape of x_com output ( x):", x_com.shape)
        x_input = x_com.reshape(-1, 1, fn)
        #print("Shape of  x_input after com :", x_input.shape)
        x = F.conv1d(x_input, self.conv1_forward, padding=7)
        x = F.relu(x)
        x_forward = F.conv1d(x, self.conv2_forward, padding=7)
        #print("Shape of   x_forward output:", x_forward.shape)
        x = torch.mul(torch.sign(x_forward), F.relu((torch.abs(x_forward) - self.soft_thr)))
        #print("Shape of    x output ( x_input):", x.shape)
        x = F.conv1d(x, self.conv1_backward, padding=7)
        x = F.relu(x)
        x_backward = F.conv1d(x, self.conv2_backward, padding=7)
        #print("Shape of   x_backward  output :", x_backward.shape)
        x_pred = x_backward.view(-1, fn)
        #print("Shape of   x_pred  output :", x_pred.shape)
        x = F.conv1d(x_forward, self.conv1_backward, padding=7)
        x = F.relu(x)
        x_est = F.conv1d(x, self.conv2_backward, padding=7)
        #print("Shape of   x_est  :", x_est.shape)
        symloss = x_est - x_input

        return [x_pred, symloss]


class ISTNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    # def forward(self, Phix, Phi_ideal, Qinit_ideal):
    #
    #     Phi = Phi_ideal
    #     Qinit = Qinit_ideal
    #     PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
    #
    #     # x = Phix
    #     PhiTb = torch.mm(Phix, Phi)
    #
    #     x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))  # x0
    #
    #     x_ideal = x.cuda()
    #     # for computing symmetric loss
    #
    #     layers_sym = []  # for computing symmetric loss
    #     layers_result = []
    #
    #     for i in range(self.LayerNo):
    #         [x_ideal, layer_sym] = self.fcs[i](x_ideal, PhiTPhi, PhiTb)
    #         layers_sym.append(layer_sym)
    #         layers_result.append(x_ideal.clone())
    #
    #     x_final = x_ideal
    #     #plot_layers_result(layers_result, save_dir)
    #
    #     return [x_final, layers_sym, layers_result]

    def forward(self, Phix, Phi_ideal, Qinit_ideal):
        device = Phix.device

        Phi = Phi_ideal.to(device)
        Qinit = Qinit_ideal.to(device)

        PhiTPhi = torch.mm(Phi.t(), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, Qinit.t())  # x0
        x_ideal = x.to(device)  # 不要再写 x.cuda()

        layers_sym = []
        layers_result = []

        for i in range(self.LayerNo):
            x_ideal, layer_sym = self.fcs[i](x_ideal, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
            layers_result.append(x_ideal.clone())

        x_final = x_ideal
        return [x_final, layers_sym, layers_result]