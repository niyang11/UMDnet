import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from Wave_Unet.model.model0_Prelu import *
from Wave_Unet.model.model0 import *
from Wave_Unet.utils.nmr_dataset import NMSELoss

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 设置保存路径
save_folder = "/home/project/ny/pythonProject/Wave_Unet/best_model/best_model0_20000SS/"
save_folder_loss = '/home/project/ny/pythonProject/Wave_Unet/loss/loss0_20000SS'
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_folder_loss, exist_ok=True)

batch_size=32
Training_data_Name = r"/home/project/ny/matlab/data/train_data/IST_20000SS.mat"
matdata = h5py.File(Training_data_Name)
Training_labels1 = matdata['FFT']
Training_labels1 = np.array(Training_labels1)
Training_labels2 = matdata['FFTN']
Training_labels2 = np.array(Training_labels2)
test_list = np.c_[Training_labels2, Training_labels1]

val_data_Name = r"/home/project/ny/matlab/data/val_data/IST_4000SS.mat"
matdata = h5py.File(val_data_Name)
val_labels1 = matdata['FFT']
val_labels1 = np.array(val_labels1)
val_labels2 = matdata['FFTN']
val_labels2 = np.array(val_labels2)
val_list = np.c_[val_labels2, val_labels1]

class RandomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:8192]
        trg_data = data[8192:]
        # 添加一个维度，假设你想要添加一个"通道"维度，通常是 `unsqueeze(0)`
        # 转换为 PyTorch 张量
        src_data = torch.tensor(src_data)  # 将 src_data 转换为 PyTorch 张量
        trg_data = torch.tensor(trg_data)  # 将 trg_data 转换为 PyTorch 张量
        src_data = src_data.unsqueeze(0)  # 在第0个维度添加一个维度（例如，通道维度）
        trg_data = trg_data.unsqueeze(0)  # 同样为目标数据添加维度
        # print(src_data.shape)
        return src_data, trg_data

    def __len__(self):
        return self.len

val_data = RandomDataset(val_list)
train_data = RandomDataset(test_list)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=4, shuffle=False)

valid_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=4, shuffle=False)

# 创建模型
wave_unet = WaveUNet0(n_layers=12, channels_interval=24).cuda()

# 定义损失函数和优化器
criterion = NMSELoss().cuda()
optimizer = optim.Adam(wave_unet.parameters(), lr=0.0003)

# 准备训练和验证损失收集
losses = []
valid_losses = []
patience = 30
best_loss = float('inf')
best_model_path = os.path.join(save_folder, 'best_model0_20000SS_260225.pth')
no_improve_epochs = 0

# 定义验证函数
def validate_model(model, valid_loader, criterion, device):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for fftn_data, fft_data in valid_loader:
            fftn_data, fft_data = fftn_data.to(device), fft_data.to(device)
            output = model(fftn_data.float())
            #fft_data = fft_data.permute(0, 2, 1)
            loss = criterion(output, fft_data)
            total_loss += loss.item()
    avg_loss = total_loss / len(valid_loader)
    return avg_loss

# 开始训练循环
device = torch.device("cuda")
for epoch in range(999999):
    wave_unet.train()
    running_loss = 0.0

    for fftn_data, fft_data in train_loader:
        fftn_data, fft_data = fftn_data.to(device), fft_data.to(device)
        optimizer.zero_grad()

        output = wave_unet(fftn_data.float())
        #fft_data = fft_data.permute(0, 2, 1)

        loss = criterion(output, fft_data.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)

    valid_loss = validate_model(wave_unet, valid_loader, criterion, device)
    valid_losses.append(valid_loss)

    print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss}, Validation Loss: {valid_loss}")
    torch.save(wave_unet.state_dict(), os.path.join(save_folder, f'model_epoch_{epoch + 1}.pth'))

    if valid_loss < best_loss:
        best_loss = valid_loss
        no_improve_epochs = 0
        torch.save(wave_unet.state_dict(), best_model_path)
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# 保存训练和验证损失数据到HDF5文件中
loss_file_path = os.path.join(save_folder_loss, 'model0_20000SS_260225.h5')
with h5py.File(loss_file_path, 'w') as hf:
    hf.create_dataset("training_losses", data=np.array(losses))
    hf.create_dataset("validation_losses", data=np.array(valid_losses))

print("Finished Training. Losses and model saved.")
