# Alternative Methods to Export FingerNet Weights

Nếu Docker không hoạt động, có các cách khác:

## Method 1: Conda Environment (Khuyến nghị)

### Setup một lần:
```bash
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax

# Tạo environment
conda create -n tf1_env python=3.7 -y
conda activate tf1_env

# Cài TensorFlow 1.15
conda install tensorflow=1.15 keras=2.3.1 -y
pip install numpy scipy opencv-python matplotlib

# Export weights
python export_fingernet_weights_tf1.py

# Deactivate
conda deactivate
```

### Hoặc dùng script tự động:
```bash
./export_with_conda.sh
```

## Method 2: Virtualenv với TF1 (nếu có sẵn TF1)

```bash
# Tạo venv
python3.7 -m venv venv_tf1
source venv_tf1/bin/activate

# Cài TF1 (cần pip version cũ)
pip install tensorflow==1.15.5 keras==2.3.1
pip install numpy scipy opencv-python matplotlib

# Export
python export_fingernet_weights_tf1.py

deactivate
```

## Method 3: Kaggle Notebook

1. Upload `export_fingernet_weights_tf1.py` và `FingerNet/` folder lên Kaggle
2. Tạo notebook với Python 3.7 + TF1 environment
3. Chạy script export
4. Download file `.npz`

## Method 4: Google Colab

1. Upload files lên Google Drive
2. Tạo Colab notebook
3. Install TF1:
   ```python
   !pip install tensorflow==1.15.5 keras==2.3.1
   ```
4. Chạy export script
5. Download `.npz` file

## Method 5: Sử dụng máy khác có TF1

Nếu bạn có access đến máy Linux/Windows đã cài TF1:
1. Copy `export_fingernet_weights_tf1.py` và `FingerNet/` folder
2. Chạy script trên máy đó
3. Copy file `.npz` về máy hiện tại

## Troubleshooting

### "Conda command not found"
- Cài Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Hoặc dùng Method 3/4 (Kaggle/Colab)

### "Python 3.7 not available"
- Conda sẽ tự download Python 3.7
- Hoặc dùng Python 3.6/3.8 (có thể work với TF1)

### "TensorFlow 1.15 installation fails"
- Thử: `pip install tensorflow==1.15.5 --no-cache-dir`
- Hoặc dùng Kaggle/Colab (đã có sẵn TF1)

