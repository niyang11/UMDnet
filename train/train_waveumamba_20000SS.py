import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import sys
sys.path.append('/home/project/ny/pythonProject')  # 添加模型所在的目录到系统路径
sys.path.append('/home/project/ny/pythonProject/Wave_Unet')  # 添加扩散模型所在的目录到系统路径

from Wave_Unet.model.waveumamba import WaveUMamba
from Wave_Unet.utils.nmr_dataset import NMSELoss

# ---------------- 基础设置 ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_folder = "/home/project/ny/pythonProject/Wave_Unet/best_model/best_waveumamba_20000SS_260225/"
save_folder_loss = "/home/project/ny/pythonProject/Wave_Unet/loss/loss_waveumamba_20000SS_260225"
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_folder_loss, exist_ok=True)

batch_size = 32
max_epochs = 500
patience = 30

# ---------------- 读取 .mat 数据 ----------------
Training_data_Name = r"/home/project/ny/matlab/data/train_data/IST_20000SS.mat"
with h5py.File(Training_data_Name, "r") as f:
    Training_labels1 = np.array(f["FFT"])   # clean
    Training_labels2 = np.array(f["FFTN"])  # noisy
test_list = np.c_[Training_labels2, Training_labels1]  # (N, 16384)

val_data_Name = r"/home/project/ny/matlab/data/val_data/IST_4000SS.mat"
with h5py.File(val_data_Name, "r") as f:
    val_labels1 = np.array(f["FFT"])
    val_labels2 = np.array(f["FFTN"])
val_list = np.c_[val_labels2, val_labels1]


# ---------------- Dataset & DataLoader ----------------
class RandomDataset(Dataset):
    def __init__(self, data):
        # 直接转 float32，后面就不用 .float() 了
        self.data = data.astype(np.float32)
        self.len = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:8192]
        trg_data = data[8192:]

        src_data = torch.from_numpy(src_data).unsqueeze(0)  # (1, 8192)
        trg_data = torch.from_numpy(trg_data).unsqueeze(0)
        return src_data, trg_data

    def __len__(self):
        return self.len


train_data = RandomDataset(test_list)
val_data = RandomDataset(val_list)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,          # ⭐ 训练集打乱
    pin_memory=True,
)

valid_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
)

# ---------------- 模型、损失、优化器 ----------------
wave_unet = WaveUMamba(
    n_layers=12,
    channels_interval=24,
    mamba_from=4,      # 深几层开始用 Mamba，可以按 GPU 情况调
    d_state=64,
    d_conv=4,
    expand=2,
    residual=True,     # 模型输出 = noisy + residual
    out_activation="none",
).to(device)

criterion = NMSELoss().to(device)
optimizer = optim.Adam(wave_unet.parameters(), lr=3e-4)

# 可选：验证集不下降时自动降学习率
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=5
# )

# ---------------- 训练 & 验证循环 ----------------
losses = []
valid_losses = []
best_loss = float("inf")
best_model_path = os.path.join(save_folder, "best_waveumamba_20000SS_260225.pth")
no_improve_epochs = 0


def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fftn_data, fft_data in valid_loader:
            fftn_data = fftn_data.to(device, non_blocking=True)
            fft_data = fft_data.to(device, non_blocking=True)

            output = model(fftn_data)
            loss = criterion(output, fft_data)
            total_loss += loss.item()
    return total_loss / len(valid_loader)


for epoch in range(1, max_epochs + 1):
    wave_unet.train()
    running_loss = 0.0

    for fftn_data, fft_data in train_loader:
        fftn_data = fftn_data.to(device, non_blocking=True)
        fft_data = fft_data.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = wave_unet(fftn_data)
        loss = criterion(output, fft_data)
        loss.backward()

        # 可选：梯度裁剪，Mamba 类模型一般比较受用
        torch.nn.utils.clip_grad_norm_(wave_unet.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)

    valid_loss = validate_model(wave_unet, valid_loader, criterion, device)
    valid_losses.append(valid_loss)
    # scheduler.step(valid_loss)

    print(
        f"Epoch {epoch}, Training Loss: {epoch_loss:.6f}, "
        f"Validation Loss: {valid_loss:.6f}"
    )

    # 只保存最优模型
    if valid_loss < best_loss:
        best_loss = valid_loss
        no_improve_epochs = 0
        torch.save(wave_unet.state_dict(), best_model_path)
        print(f"  → New best model saved (val loss = {best_loss:.6f})")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# ---------------- 保存 loss 曲线 ----------------
loss_file_path = os.path.join(save_folder_loss, "waveumamba_20000SS_260225.h5")
with h5py.File(loss_file_path, "w") as hf:
    hf.create_dataset("training_losses", data=np.array(losses))
    hf.create_dataset("validation_losses", data=np.array(valid_losses))

print("Finished Training. Losses and best model saved.")
