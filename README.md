# CMDnet: Mamba-based NMR Spectrum Denoising Code

This repository contains code for nuclear magnetic resonance (NMR) spectrum denoising experiments based on a U-shaped convolutional network with Mamba/Mamba2 sequence modeling modules. The code includes model definitions, training scripts, and testing/evaluation scripts for 1D, 2D, and 3D NMR spectrum denoising tasks.

## Repository Structure

```text
.
├── train/
│   ├── train_waveumamba_20000SS.py   # CMDnet / WaveUMamba training script
│   ├── train0_20000SS.py             # LD-Net / WaveUNet baseline training script
│   ├── train_dnunet_20000SS.py       # DN-Unet comparison training script
│   ├── waveumamba.py                 # Mamba fusion network
│   ├── waveumamba_flex.py            # Configurable Mamba fusion network
│   ├── unet_basic_dn.py              # DN-Unet-related structure
│   ├── model0.py                     # U-shaped convolutional baseline
│   └── ISTA_ljw.py                   # ISTA/ISTNet-related structure
├── test/
│   ├── test_0401.py                  # Multi-model testing and metric calculation
│   └── *.mat                         # Test data, not committed by default
├── model/
│   └── *.pth                         # Model weights, not committed by default
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment

Python 3.9 or later is recommended. Training is intended for a CUDA-capable GPU environment.

```bash
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- NumPy
- SciPy
- h5py
- pandas
- scikit-learn
- openpyxl
- mamba-ssm

## Data and Model Weights

Large model weights and `.mat` data files are not included in this repository. They should be stored separately and placed in local folders before running the scripts.

Suggested local layout:

```text
model/
test/
```

Expected local model weight names:

```text
model/DNUnet.pth
model/ISTNet.pth
model/LDNet_20000SS.pth
model/CMD-M1Net.pth
model/CMD-NRNet.pth
model/CMDNet0.pth
model/CMDNet4.pth
model/CMDNet8.pth
model/CMDNet12.pth
```

Expected local test data names:

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

Some files exceed GitHub's normal 100 MB file limit, so they should be managed with Git LFS, GitHub Releases, cloud storage, or a separate server.

## Paths That Need to Be Changed

The original scripts contain absolute server paths. Before running the code on a new machine, replace the following path groups with paths that exist in your local environment.

### 1. Python import paths

Files:

```text
test/test_0401.py
train/train_waveumamba_20000SS.py
```

Current examples:

```python
sys.path.append('/home/project/ny/pythonProject')
sys.path.append('/home/project/ny/pythonProject/Wave_Unet')
```

Change them to the local project root or remove them if the package imports already work from your current working directory.

### 2. Training and validation data paths

Files:

```text
train/train0_20000SS.py
train/train_dnunet_20000SS.py
train/train_waveumamba_20000SS.py
```

Variables to edit:

```python
Training_data_Name = r"/home/project/ny/matlab/data/train_data/IST_20000SS.mat"
val_data_Name = r"/home/project/ny/matlab/data/val_data/IST_4000SS.mat"
```

Change them to the local training and validation `.mat` files.

### 3. Model output and loss output paths

Files:

```text
train/train0_20000SS.py
train/train_dnunet_20000SS.py
train/train_waveumamba_20000SS.py
train/ISTA_ljw.py
```

Variables to edit:

```python
save_folder = "/home/project/ny/pythonProject/Wave_Unet/best_model/..."
save_folder_loss = "/home/project/ny/pythonProject/Wave_Unet/loss/..."
save_dir = "/home/project/ny/pythonProject/ISTA_net/view/..."
```

Change them to local folders for checkpoints, loss curves, and figures.

### 4. Test `.mat` data paths

File:

```text
test/test_0401.py
```

Functions to edit:

```text
load_mix_moise
load_aq_moise
load_danguchun_moise
load_2d_danbai_0902
load_2d_gb1_0902
load_2d_titr31
load_hnco_xy
load_hnco_xz
load_hnco_yz
```

Replace paths such as:

```python
'/home/project/ny/matlab/wave_unet/mix_noise.mat'
'/home/project/ny/matlab/wave_unet/2d_data/test_2d_gb1_0902.mat'
'/home/project/ny/matlab/wave_unet/3d_data/test_3d_dataxy.mat'
```

with the corresponding files in your local `test/` directory.

### 5. Model weight paths used during testing

File:

```text
test/test_0401.py
```

Variables and sections to edit:

```python
ABLA_ROOT = "/home/project/ny/pythonProject/Wave_Unet/best_model/ablation_M1M2_k2k4k12/"
models = [...]
```

Change every `.pth` path in the `models` list to the corresponding file in `model/`.

### 6. IST/ISTA auxiliary matrix paths

File:

```text
test/test_0401.py
```

Variables to edit:

```python
Phi_ideal_path = r'/home/project/ny/matlab/istmatrix.mat'
Qinit_ideal_path = r'/home/project/ny/matlab/data/Qinit_20000SS'
mat_data_path = r'/home/SSD2_4T/fangqy/train_data/IST/IST_20000_SS.mat'
```

These paths are required by IST/ISTA-related models. Change them to the local files used to build `Phi` and `Qinit`.

### 7. Prediction result output paths

File:

```text
test/test_0401.py
```

Variable to edit:

```python
save_path = f'/home/project/ny/pythonProject/Wave_Unet/predictions/predictions_all/{display_name}_{test_name}_predictions_{model_number}.mat'
```

Change this to a local output folder, for example:

```python
save_path = f'outputs/predictions/{display_name}_{test_name}_predictions_{model_number}.mat'
```

## Training

Example:

```bash
python train/train_waveumamba_20000SS.py
```

Before training, make sure the training data, validation data, checkpoint output folder, and loss output folder have been changed to valid local paths.

## Testing

Example:

```bash
python test/test_0401.py
```

Before testing, make sure the model weights, test `.mat` files, IST/ISTA auxiliary matrix files, and prediction output folder have been changed to valid local paths.

## Upload Notes

The repository is intended to store source code and documentation. The following files are ignored by default:

```text
model/*.pth
model/*.pt
model/*.ckpt
test/*.mat
*.mat
*.h5
*.xlsx
```

Use Git LFS or an external storage service if the model weights and test data need to be shared.
