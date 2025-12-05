# Hướng dẫn Export Weights KHÔNG dùng Docker

## ✅ Method 1: Conda (Khuyến nghị - Đã có sẵn)

Bạn đã có Conda! Chạy script tự động:

```bash
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
./export_with_conda.sh
```

Script sẽ:
1. Tự động tạo conda environment `tf1_env` với Python 3.7
2. Cài TensorFlow 1.15, Keras 2.3.1, và dependencies
3. Export weights từ `Model.model`
4. Tạo file `fingernet_keras_tf1_weights.npz`

**Thời gian**: ~5-10 phút (lần đầu), ~2-3 phút (lần sau)

---

## Method 2: Manual Conda Setup

Nếu script không work, làm thủ công:

```bash
# 1. Tạo environment
conda create -n tf1_env python=3.7 -y

# 2. Activate
conda activate tf1_env

# 3. Cài TF1 và dependencies
conda install tensorflow=1.15 keras=2.3.1 -y
pip install numpy scipy opencv-python matplotlib

# 4. Export weights
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
python export_fingernet_weights_tf1.py

# 5. Deactivate
conda deactivate

# 6. Convert sang Flax
python convert_from_npz.py
```

---

## Method 3: Kaggle Notebook (Nếu Conda không work)

1. **Upload files lên Kaggle:**
   - `fingernet_jax/export_fingernet_weights_tf1.py`
   - `FingerNet/` folder (toàn bộ)

2. **Tạo notebook:**
   ```python
   # Cell 1: Setup
   import os
   os.chdir('/kaggle/working')
   
   # Cell 2: Run export
   !cd /kaggle/working/IFViT/fingernet_jax && python export_fingernet_weights_tf1.py
   
   # Cell 3: Download
   from IPython.display import FileLink
   FileLink('fingernet_keras_tf1_weights.npz')
   ```

3. **Download file `.npz` về máy**

4. **Convert:**
   ```bash
   cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
   python convert_from_npz.py /path/to/downloaded/file.npz
   ```

---

## Method 4: Google Colab

Tương tự Kaggle, nhưng dùng Google Colab:
- Upload files lên Google Drive
- Mount Drive trong Colab
- Chạy export script
- Download `.npz` file

---

## Troubleshooting

### "Conda environment creation fails"
- Thử: `conda clean --all` rồi chạy lại
- Hoặc dùng Method 3 (Kaggle)

### "TensorFlow 1.15 installation fails"
- Thử: `pip install tensorflow==1.15.5 --no-cache-dir`
- Hoặc dùng Kaggle/Colab (đã có sẵn TF1)

### "Python 3.7 not available"
- Conda sẽ tự download
- Hoặc dùng Python 3.6/3.8

### "Script hangs at model loading"
- `Model.model` có thể bị corrupt
- Thử download lại từ repo gốc
- Hoặc dùng Kaggle/Colab

---

## Sau khi có file .npz

```bash
# Convert sang Flax
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
python convert_from_npz.py

# Test
python deploy_jax.py \
  ../FingerNet/datasets/NIST4/F0001_01.bmp \
  ./test_output_jax \
  ./fingernet_flax_params.npz
```

