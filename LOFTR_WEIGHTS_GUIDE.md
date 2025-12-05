# LoFTR Pretrained Weights Guide

## 📋 Overview

Module 1 (Dense Registration) cần load pretrained LoFTR weights để initialize transformer.

**File**: `outdoor_ds.ckpt` (PyTorch format) → Cần convert sang JAX/Flax format (.npz)

## 🔄 Conversion Process

### Step 1: Convert PyTorch → JAX/Flax

```bash
cd /kaggle/working/IFViT
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt /kaggle/working/IFViT/weights/outdoor_ds.ckpt \
    --output /kaggle/working/IFViT/weights/loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

**Note**: Prefix có thể khác tùy checkpoint structure. Nếu không chắc, bỏ `--prefix` để auto-detect.

### Step 2: Verify Converted Checkpoint

```bash
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --verify /kaggle/working/IFViT/weights/loftr_transformer.npz
```

### Step 3: Update Config

Trong `ifvit-jax/config.py` hoặc khi train:

```python
DENSE_CONFIG = {
    ...
    "use_loftr": True,
    "loftr_pretrained_ckpt": "/kaggle/working/IFViT/weights/loftr_transformer.npz",
    ...
}
```

### Step 4: Train với LoFTR Weights

```bash
cd /kaggle/working/IFViT
python ifvit-jax/train_dense_tpu.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

Training script sẽ tự động load LoFTR weights nếu `loftr_pretrained_ckpt` được set trong config.

## 🔍 Checkpoint Structure

### PyTorch Checkpoint (`outdoor_ds.ckpt`)

```
state_dict:
  - loftr_coarse.coarse_transformer.layers.0.linear_q.weight
  - loftr_coarse.coarse_transformer.layers.0.linear_k.weight
  - ...
```

### Converted JAX Checkpoint (`loftr_transformer.npz`)

```
layers.0.0/q_proj/kernel
layers.0.0/k_proj/kernel
layers.0.0/v_proj/kernel
...
```

## ⚙️ Implementation Details

### Conversion Script

`ifvit-jax/ut/convert_loftr_checkpoint.py`:
- Loads PyTorch checkpoint
- Filters transformer weights
- Maps PyTorch keys → Flax keys
- Transposes weight matrices (PyTorch [out, in] → Flax [in, out])
- Saves as .npz

### Loading in Training

`ifvit-jax/train_dense.py` sẽ:
1. Initialize model với random weights
2. Load LoFTR weights từ .npz
3. Merge vào transformer parameters
4. Continue training

## 📝 Example Usage

### Local Development

```bash
# 1. Convert checkpoint
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt ./weights/outdoor_ds.ckpt \
    --output ./weights/loftr_transformer.npz

# 2. Update config.py
# Set: "loftr_pretrained_ckpt": "./weights/loftr_transformer.npz"

# 3. Train
python ifvit-jax/train_dense.py \
    --checkpoint_dir ./checkpoints/dense_reg
```

### Kaggle

```bash
# 1. Convert checkpoint
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt /kaggle/working/IFViT/weights/outdoor_ds.ckpt \
    --output /kaggle/working/IFViT/weights/loftr_transformer.npz

# 2. Update config.py hoặc set trong training script
# Set: "loftr_pretrained_ckpt": "/kaggle/working/IFViT/weights/loftr_transformer.npz"

# 3. Train
python ifvit-jax/train_dense_tpu.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

## ⚠️ Important Notes

1. **Module 1**: Loads LoFTR pretrained weights (initialization)
2. **Module 2**: Loads **trained** Module 1 weights (NOT fresh LoFTR)
   - Module 2 reuses ViT trained in Module 1
   - This follows IFViT paper exactly

3. **Weight Mapping**:
   - PyTorch uses `[out_features, in_features]` for linear layers
   - Flax uses `[in_features, out_features]` for Dense layers
   - Conversion script automatically transposes

4. **Prefix Detection**:
   - Script tries to auto-detect transformer keys
   - Common prefixes: `loftr_coarse.coarse_transformer`, `coarse_transformer`, `transformer`
   - If auto-detect fails, specify `--prefix` manually

## 🔧 Troubleshooting

### Error: "No transformer keys found"

**Solution**: Specify prefix manually:
```bash
python ifvit-jax/ut/convert_loftr_checkpoint.py \
    --pytorch_ckpt outdoor_ds.ckpt \
    --output loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

### Error: "Shape mismatch"

**Solution**: Check if weight matrices need transposition. Conversion script should handle this automatically.

### Error: "Checkpoint not found"

**Solution**: 
1. Verify file path
2. Check if file exists: `ls -lh /kaggle/working/IFViT/weights/loftr_transformer.npz`
3. Update config path

## 📊 Expected Output

After conversion:
```
Loading PyTorch checkpoint: outdoor_ds.ckpt
Total keys in checkpoint: 152109
Auto-detected prefix: 'loftr_coarse.coarse_transformer' (256 keys)

Converting weights to NumPy format...
  ✓ loftr_coarse.coarse_transformer.layers.0.linear_q.weight -> layers.0.0/q_proj/kernel (256, 256)
  ✓ loftr_coarse.coarse_transformer.layers.0.linear_k.weight -> layers.0.0/k_proj/kernel (256, 256)
  ...

Saving to loftr_transformer.npz...
✓ Saved 256 parameters

Summary:
  Total parameters: 8,192,000
  Output file: loftr_transformer.npz
```

