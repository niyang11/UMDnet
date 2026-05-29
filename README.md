# UMD-Net: Mamba-based U-shaped NMR Spectrum Denoising Network

本仓库整理的是毕业论文第三章“基于 Mamba 的 U 型 NMR 波谱去噪网络”相关代码，用于核磁共振（NMR）波谱去噪模型的训练、推理测试与对比实验。第三章提出的 UMD-Net（U-shaped Mamba Denoised Network）以 U 型编码器-解码器为主干，在多尺度卷积特征中融合 Mamba/Mamba2 序列建模模块，用于增强长程谱线相关性建模能力，并比较不同融合起始层位对一维、二维和三维 NMR 去噪结果的影响。

## 研究内容对应关系

- `train/`：模型结构与训练脚本，对应第三章 3.2“去噪网络设计”和 3.3“训练方法与算法流程”。
- `test/test_0401.py`：统一测试、指标统计与结果保存脚本，对应第三章 3.4“实验验证与结果分析”。
- `model/`：本地保存的训练权重文件。由于部分权重超过 GitHub 普通文件大小限制，建议不要直接提交到 GitHub 普通仓库。
- `test/*.mat`：测试数据文件，包含一维噪声谱、二维谱和三维切片数据。部分 `.mat` 文件超过 100 MB，也建议通过网盘、Release、Git LFS 或学校/课题组服务器单独保存。

## 目录结构

```text
.
├── train/
│   ├── train_waveumamba_20000SS.py   # UMD-Net / WaveUMamba 训练脚本
│   ├── train0_20000SS.py             # LD-Net / WaveUNet 基线训练脚本
│   ├── train_dnunet_20000SS.py       # DN-Unet 对比模型训练脚本
│   ├── waveumamba.py                 # Mamba 融合网络结构
│   ├── waveumamba_flex.py            # 可配置融合层位的 Mamba 网络结构
│   ├── unet_basic_dn.py              # DN-Unet 相关结构
│   ├── model0.py                     # U 型卷积基线结构
│   └── ISTA_ljw.py                   # ISTA/ISTNet 相关结构
├── test/
│   ├── test_0401.py                  # 多模型测试、指标计算和结果保存
│   └── *.mat                         # 测试数据，本仓库默认不提交大文件
├── model/
│   └── *.pth                         # 训练权重，本仓库默认不提交大文件
├── requirements.txt
├── .gitignore
└── README.md
```

## 环境依赖

建议使用 Python 3.9 及以上版本，并在具备 CUDA 的服务器或工作站上运行训练脚本。

```bash
pip install -r requirements.txt
```

核心依赖包括：

- PyTorch
- NumPy
- SciPy
- h5py
- pandas
- scikit-learn
- openpyxl
- mamba-ssm（运行 Mamba/Mamba2 模块时需要）

## 数据与权重说明

原始代码来自服务器环境，代码中仍保留了一些服务器绝对路径，例如：

```text
/home/project/ny/pythonProject/...
/home/project/ny/matlab/...
/home/SSD2_4T/...
```

下载到本机后，请将数据和权重按下列相对目录组织，或在脚本中把对应路径改为本机路径：

```text
model/
test/
```

当前本地权重文件包括：

```text
model/DNUnet.pth
model/ISTNet.pth
model/LDNet_20000SS.pth
model/UMD-M1Net.pth
model/UMD-NRNet.pth
model/UMDNet0.pth
model/UMDNet4.pth
model/UMDNet8.pth
model/UMDNet12.pth
```

当前本地测试数据包括：

```text
test/aq_noise.mat
test/danguchun_noise.mat
test/mix_noise.mat
test/test_2d_danbai_0902.mat
test/test_2d_gb1_0902.mat
test/test_2d_titr31.mat
test/test_3d_dataxy.mat
test/test_3d_dataxz.mat
test/test_3d_datayz.mat
```

注意：`DNUnet.pth`、部分 `.mat` 测试数据超过 GitHub 单文件 100 MB 限制，不适合直接通过普通 GitHub 提交上传。若需要公开复现实验，建议使用 Git LFS、GitHub Release、Zenodo、百度网盘或学校服务器保存大文件，并在 README 中补充下载链接。

## 训练

训练脚本位于 `train/` 目录。以 UMD-Net / WaveUMamba 为例：

```bash
python train/train_waveumamba_20000SS.py
```

训练脚本默认读取服务器路径下的训练集：

```text
/home/project/ny/matlab/data/train_data/IST_20000SS.mat
/home/project/ny/matlab/data/val_data/IST_4000SS.mat
```

如果在本机运行，需要把脚本中的 `Training_data_Name`、`val_data_Name`、`save_folder` 和 `save_folder_loss` 改成本机实际路径。

## 测试与指标统计

测试脚本位于：

```bash
python test/test_0401.py
```

该脚本用于加载多个模型权重，在一维、二维和三维测试数据上生成去噪结果，并计算模型大小、参数量、推理时间、R2、Pearson R2、RMSD 和 KSNR 等指标。测试输出默认保存为 `.mat` 和 Excel 结果文件。

本机运行前需要重点检查：

- `sys.path.append(...)` 是否指向当前代码根目录；
- `models` 列表中的权重路径是否对应 `model/` 目录下的实际文件名；
- `load_*` 数据加载函数中的 `.mat` 路径是否对应 `test/` 目录下的实际文件名；
- `Phi_ideal_path`、`Qinit_ideal_path`、`mat_data_path` 是否已替换为本机可访问的 IST/ISTA 辅助矩阵路径。

## 与论文第三章的实验设置

第三章实验主要围绕以下问题展开：

- 不同 Mamba 融合起始层位的对比：UMD-Net0、UMD-Net4、UMD-Net8、UMD-Net12；
- 与 LD-Net、DN-Unet、ISTNet 等方法进行对比；
- 在一维仿真谱、一维实测谱、二维蛋白谱和三维 HNCO 天青蛋白谱上的重建效果分析；
- 通过 RMSD、R2、推理时间和模型大小等指标评价去噪性能；
- 通过 UMD-M1Net、UMD-NRNet 与 UMD-Net4 进行消融实验，验证 Mamba2 模块和残差连接的作用。

论文结论中，UMD-Net4 在重建精度、弱峰保持、伪峰抑制和整体稳定性方面表现较均衡，因此作为第三章主要对比和消融分析的核心配置。

## 上传 GitHub 时的建议

建议优先上传以下内容：

```text
train/
test/test_0401.py
README.md
requirements.txt
.gitignore
```

不建议直接上传：

```text
model/*.pth
test/*.mat
*.h5
*.xlsx
```

如果确实需要把权重和数据也放到 GitHub，请先启用 Git LFS；否则 GitHub 普通上传会因为单文件超过 100 MB 而失败。
