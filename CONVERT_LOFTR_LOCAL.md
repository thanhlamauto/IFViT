# Convert LoFTR Checkpoint - Local Guide

## ✅ Có thể convert ở local

Script `convert_loftr_checkpoint.py` chạy được ở cả local và Kaggle, chỉ cần thay đổi đường dẫn.

## 🖥️ Local Development

### Step 1: Đảm bảo có PyTorch

```bash
pip install torch
```

### Step 2: Convert checkpoint

```bash
cd /path/to/IFViT

python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt ./weights/outdoor_ds.ckpt \
    --output ./weights/loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

**Hoặc với đường dẫn tuyệt đối:**

```bash
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt /Users/nguyenthanhlam/SSL_Correspondence/IFViT/weights/outdoor_ds.ckpt \
    --output /Users/nguyenthanhlam/SSL_Correspondence/IFViT/weights/loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

### Step 3: Verify converted checkpoint

```bash
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --verify ./weights/loftr_transformer.npz
```

## 📋 So sánh Local vs Kaggle

| Environment | PyTorch Checkpoint Path | Output Path |
|------------|------------------------|-------------|
| **Local** | `./weights/outdoor_ds.ckpt` | `./weights/loftr_transformer.npz` |
| **Kaggle** | `/kaggle/working/IFViT/weights/outdoor_ds.ckpt` | `/kaggle/working/IFViT/weights/loftr_transformer.npz` |

## 🔧 Requirements

- Python 3.8+
- PyTorch (`pip install torch`)
- NumPy (đã có trong JAX dependencies)

## 📝 Example Commands

### Local (Relative paths)

```bash
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT

python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt weights/outdoor_ds.ckpt \
    --output weights/loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

### Local (Absolute paths)

```bash
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt /Users/nguyenthanhlam/SSL_Correspondence/IFViT/weights/outdoor_ds.ckpt \
    --output /Users/nguyenthanhlam/SSL_Correspondence/IFViT/weights/loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

### Kaggle

```bash
cd /kaggle/working/IFViT

python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt /kaggle/working/IFViT/weights/outdoor_ds.ckpt \
    --output /kaggle/working/IFViT/weights/loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

## ⚠️ Lưu ý

1. **File size**: `outdoor_ds.ckpt` khá lớn (~200-300MB), đảm bảo có đủ disk space
2. **PyTorch version**: Script tương thích với PyTorch 1.8+
3. **Auto-detect prefix**: Nếu không chắc prefix, bỏ `--prefix` để script tự detect

## 🔍 Troubleshooting

### Error: "No module named 'torch'"

**Solution**:
```bash
pip install torch
```

### Error: "Checkpoint not found"

**Solution**: Kiểm tra đường dẫn:
```bash
ls -lh weights/outdoor_ds.ckpt
```

### Error: "No transformer keys found"

**Solution**: Thử bỏ `--prefix` để auto-detect:
```bash
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt weights/outdoor_ds.ckpt \
    --output weights/loftr_transformer.npz
```

## ✅ Sau khi convert

Update config để dùng converted checkpoint:

```python
# ifvit-jax/config.py
DENSE_CONFIG = {
    ...
    "loftr_pretrained_ckpt": "./weights/loftr_transformer.npz",  # Local
    # hoặc
    "loftr_pretrained_ckpt": "/kaggle/working/IFViT/weights/loftr_transformer.npz",  # Kaggle
    ...
}
```

