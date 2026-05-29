# CMDnet: NMR Spectrum Denoising Code

This repository contains source code, data-generation scripts, selected test data, and trained model weights for NMR spectrum denoising experiments based on a U-shaped convolutional network with Mamba/Mamba2 modules.

## Repository Structure

```text
.
├── data/                 # Test data files
├── dataset/              # MATLAB scripts for generating datasets
├── model/                # Trained model weights under GitHub's 100 MB limit
├── test/                 # Python testing and metric scripts
├── train/                # Python training scripts and model definitions
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment

Python 3.9 or later is recommended. Training is intended for a CUDA-capable GPU environment.

```bash
pip install -r requirements.txt
```

Main Python dependencies:

- PyTorch
- NumPy
- SciPy
- h5py
- pandas
- scikit-learn
- openpyxl
- mamba-ssm

MATLAB is required for generating simulation datasets and splitting large `.mat` files.

## Included Data

The `data/` folder contains test data. Files smaller than GitHub's 100 MB single-file limit are uploaded directly.

Directly uploaded test files:

```text
data/aq_noise.mat
data/danguchun_noise.mat
data/mix_noise.mat
data/test_2d_danbai_0902.mat
data/test_3d_dataxy.mat
data/test_3d_dataxz.mat
data/test_3d_datayz.mat
```

Two original 2D test files are larger than 100 MB:

```text
data/test_2d_gb1_0902.mat
data/test_2d_titr31.mat
```

They are not uploaded directly. Each file is split into three smaller files because the uploaded files correspond to three different noise levels:

```text
data/test_2d_gb1_0902_noise1.mat
data/test_2d_gb1_0902_noise2.mat
data/test_2d_gb1_0902_noise3.mat
data/test_2d_titr31_noise1.mat
data/test_2d_titr31_noise2.mat
data/test_2d_titr31_noise3.mat
```

Each split file contains:

- `real_FFT`: clean/reference spectrum data
- `real_FFTN1`, `real_FFTN2`, or `real_FFTN3`: noisy spectrum data for one noise level
- `noise_level`: noise-level index
- `source_file`: original source filename

The split files can be regenerated from the original large files with:

```matlab
run('dataset/split_2d_noise_files.m')
```

## Included Model Weights

The `model/` folder includes model files that are under GitHub's 100 MB single-file limit:

```text
model/ISTNet.pth
model/LDNet_20000SS.pth
model/CMD-M1Net.pth
model/CMD-NRNet.pth
model/CMDNet0.pth
model/CMDNet4.pth
model/CMDNet8.pth
model/CMDNet12.pth
```

`model/DNUnet.pth` is larger than 100 MB, so it is not uploaded. Use Git LFS, GitHub Releases, cloud storage, or a separate server if this file needs to be shared.

## How to Generate Training and Validation Data

The MATLAB script for generating synthetic 1D NMR training/validation data is:

```text
dataset/dataset_SS.m
```

Typical workflow:

1. Open MATLAB.
2. Set the working directory to the repository root.
3. Open `dataset/dataset_SS.m`.
4. Adjust the dataset size and output path.
5. Run the script to generate `.mat` data containing `FFT` and `FFTN`.

Important parameters in `dataset/dataset_SS.m`:

```matlab
l_t2 = 8192;     % signal length
fn = 8192;       % FFT length
num = 1000;      % number of generated samples
sw = 8000;       % spectral width
```

Output path to change before running:

```matlab
save('/home/project/ny/matlab/data/train_data/IST_1000SS.mat','FFT','FFTN','-v7.3');
```

Change this path to a local path, for example:

```matlab
save('data/IST_1000SS.mat','FFT','FFTN','-v7.3');
```

The script calls `create_f2(...)`. Make sure this helper function is available on the MATLAB path before running the dataset-generation script.

For training and validation, generate separate `.mat` files and point the Python training scripts to them. For example:

```text
data/train_data/IST_20000SS.mat
data/val_data/IST_4000SS.mat
```

## How to Train and Obtain Model Files

Training scripts are in `train/`.

Main CMDnet training script:

```bash
python train/train_waveumamba_20000SS.py
```

Baseline and comparison training scripts:

```bash
python train/train0_20000SS.py
python train/train_dnunet_20000SS.py
```

Before training, change the hard-coded data and output paths in each script.

Training data paths to change:

```python
Training_data_Name = r"/home/project/ny/matlab/data/train_data/IST_20000SS.mat"
val_data_Name = r"/home/project/ny/matlab/data/val_data/IST_4000SS.mat"
```

Model and loss output folders to change:

```python
save_folder = "/home/project/ny/pythonProject/Wave_Unet/best_model/..."
save_folder_loss = "/home/project/ny/pythonProject/Wave_Unet/loss/..."
```

During training, the scripts save the best model when the validation loss improves. The final model file is saved as a `.pth` file in `save_folder`.

Example local setup:

```python
Training_data_Name = r"data/train_data/IST_20000SS.mat"
val_data_Name = r"data/val_data/IST_4000SS.mat"
save_folder = r"model/"
save_folder_loss = r"outputs/loss/"
```

After training, copy or rename the best `.pth` file into `model/`.

## How to Test

Testing script:

```bash
python test/test_0401.py
```

Before testing, change paths in `test/test_0401.py`.

### Python import paths

```python
sys.path.append('/home/project/ny/pythonProject')
sys.path.append('/home/project/ny/pythonProject/improved_diffusion_condition_LD_fre')
```

Change these to the local repository path or remove them if imports work from the repository root.

### Test data paths

Change paths in these functions:

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

For the split 2D files, update `load_2d_gb1_0902` and `load_2d_titr31` to load files such as:

```text
data/test_2d_gb1_0902_noise1.mat
data/test_2d_gb1_0902_noise2.mat
data/test_2d_gb1_0902_noise3.mat
data/test_2d_titr31_noise1.mat
data/test_2d_titr31_noise2.mat
data/test_2d_titr31_noise3.mat
```

### Model weight paths

In `test/test_0401.py`, change every `.pth` path in:

```python
ABLA_ROOT = "/home/project/ny/pythonProject/Wave_Unet/best_model/ablation_M1M2_k2k4k12/"
models = [...]
```

to the corresponding file in `model/`.

### IST/ISTA auxiliary matrix paths

These paths are required by IST/ISTA-related models:

```python
Phi_ideal_path = r'/home/project/ny/matlab/istmatrix.mat'
Qinit_ideal_path = r'/home/project/ny/matlab/data/Qinit_20000SS'
mat_data_path = r'/home/SSD2_4T/fangqy/train_data/IST/IST_20000_SS.mat'
```

Change them to local files before running IST/ISTA tests.

### Prediction output path

Change:

```python
save_path = f'/home/project/ny/pythonProject/Wave_Unet/predictions/predictions_all/{display_name}_{test_name}_predictions_{model_number}.mat'
```

to a local output folder, for example:

```python
save_path = f'outputs/predictions/{display_name}_{test_name}_predictions_{model_number}.mat'
```

## Upload Notes

Files intentionally excluded from Git:

```text
model/DNUnet.pth
data/test_2d_gb1_0902.mat
data/test_2d_titr31.mat
*.h5
*.xlsx
```

The excluded files are too large for normal GitHub upload. Use the split files or an external storage method for sharing them.
