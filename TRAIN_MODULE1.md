# Training Module 1: Dense Registration

## 🚀 Quick Start

### Kaggle TPU v5e-8 (Recommended)

```bash
cd /kaggle/working/IFViT
python ifvit-jax/train_dense_tpu.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

**TPU-optimized features**:
- ✅ Data parallelism với `jax.pmap`
- ✅ Batch sharding across 8 TPU cores
- ✅ Efficient gradient accumulation
- ✅ Multi-device checkpointing

### Local Development (CPU/GPU)

```bash
cd /path/to/IFViT
python ifvit-jax/train_dense.py \
    --checkpoint_dir ./checkpoints/dense_reg
```

### Kaggle Notebooks (CPU/GPU)

```bash
cd /kaggle/working/IFViT
python ifvit-jax/train_dense.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

## 📋 Command Arguments

```bash
python ifvit-jax/train_dense.py \
    [--checkpoint_dir PATH] \
    [--resume_from PATH] \
    [--dataset_root PATH]  # Deprecated: không cần nữa, datasets tự động load từ Kaggle paths
```

### Arguments

- `--checkpoint_dir` (optional): 
  - Override checkpoint directory từ config
  - Default: `./checkpoints/dense_reg` (local) hoặc `/kaggle/working/IFViT/checkpoints/dense_reg` (Kaggle)
  
- `--resume_from` (optional):
  - Resume training từ checkpoint
  - Example: `--resume_from ./checkpoints/dense_reg/dense_reg_epoch_50.pkl`

- `--dataset_root` (deprecated):
  - Không cần thiết nữa
  - Datasets tự động load từ `PaperDatasetRoots()` (auto-detects Kaggle paths)

## 📊 What Happens During Training

### 1. Dataset Loading

Tự động load từ:
- FVC2002: DB1A, DB2A, DB3A
- NIST SD301a: Partitions A, B, C, E, J, K, M, N
- NIST SD302a: Partitions A, B, C, D, E, F, U, V, L, M
- NIST SD300: Replaces MOLF

**Total**: ~25,090 original images

### 2. Data Augmentation

Mỗi image được augment:
- **ONE of 3 noise models**: Perlin noise, Erosion, hoặc Dilation
- **Rotation**: ±60° (applied after corruption)
- **Result**: ~100,360 training images

### 3. Training Process

- **Model**: DenseRegModel (ResNet-18 + Transformer + Matching Head)
- **Loss**: L_D only (dense correspondence loss)
- **Optimizer**: AdamW with warmup cosine decay
- **Batch size**: 128 (configurable)
- **Epochs**: 100 (configurable)
- **Learning rate**: 1e-3 (configurable)

### 4. Checkpoints

- **Periodic**: `dense_reg_epoch_{N}.pkl` (every 5 epochs)
- **Final**: `dense_reg_ckpt.pkl` (sau khi training xong)

## 📁 Output Structure

```
checkpoints/dense_reg/
├── dense_reg_epoch_5.pkl
├── dense_reg_epoch_10.pkl
├── ...
├── dense_reg_epoch_100.pkl
├── dense_reg_ckpt.pkl          # Final checkpoint (dùng cho Module 2)
└── logs/
    ├── train.log
    └── metrics.json
```

## 🔧 Configuration

Tất cả hyperparameters trong `ifvit-jax/config.py` → `DENSE_CONFIG`:

```python
DENSE_CONFIG = {
    "image_size": 128,
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 100,
    "transformer_layers": 4,
    "num_heads": 8,
    "hidden_dim": 256,
    "mlp_dim": 1024,
    "lambda_D": 1.0,  # Only L_D loss
    ...
}
```

## 📝 Example Commands

### TPU Training (Kaggle TPU v5e-8)

```bash
# Kaggle TPU (Recommended - fastest)
cd /kaggle/working/IFViT
python ifvit-jax/train_dense_tpu.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

**TPU Benefits**:
- 8x faster với 8 TPU cores
- Effective batch size = batch_size × 8
- Automatic batch sharding

### Basic Training (CPU/GPU)

```bash
# Local
python ifvit-jax/train_dense.py

# Kaggle (CPU/GPU)
cd /kaggle/working/IFViT
python ifvit-jax/train_dense.py
```

### Custom Checkpoint Directory

```bash
# Local
python ifvit-jax/train_dense.py \
    --checkpoint_dir ./my_checkpoints/dense_reg

# Kaggle
python ifvit-jax/train_dense.py \
    --checkpoint_dir /kaggle/working/IFViT/my_checkpoints/dense_reg
```

### Resume Training

```bash
# Resume from epoch 50
python ifvit-jax/train_dense.py \
    --resume_from ./checkpoints/dense_reg/dense_reg_epoch_50.pkl
```

## ⚠️ Important Notes

1. **TPU vs CPU/GPU**:
   - **TPU**: Dùng `train_dense_tpu.py` (tối ưu cho TPU v5e-8)
   - **CPU/GPU**: Dùng `train_dense.py` (single device)
   - TPU version tự động shard batch across 8 cores

2. **Datasets tự động load**: Không cần chỉ định `--dataset_root`, datasets tự động detect từ Kaggle paths hoặc local paths

3. **Checkpoint path**: Module 2 sẽ tự động load từ `./checkpoints/dense_reg/dense_reg_ckpt.pkl` (hoặc path trong config)

4. **Training time**: 
   - ~100 epochs với ~100k training pairs
   - **TPU v5e-8**: ~2-4 giờ (8 cores parallel)
   - **GPU**: ~8-16 giờ (single GPU)
   - **CPU**: ~2-3 ngày (single CPU)

5. **Memory**: 
   - Batch size 128 với 128×128 images
   - **TPU**: Per-device batch = 128/8 = 16 (efficient)
   - **GPU**: Cần ~8-16GB GPU memory

## 🔍 Monitoring Training

Logs được lưu tại:
- `{checkpoint_dir}/logs/train.log`
- `{checkpoint_dir}/logs/metrics.json`

Checkpoints được lưu tại:
- `{checkpoint_dir}/dense_reg_epoch_{N}.pkl` (periodic)
- `{checkpoint_dir}/dense_reg_ckpt.pkl` (final)

## ✅ After Training

Sau khi training xong, checkpoint final sẽ được lưu tại:
- Local: `./checkpoints/dense_reg/dense_reg_ckpt.pkl`
- Kaggle: `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl`

Checkpoint này sẽ được dùng để train Module 2:

```bash
python ifvit-jax/train_match.py \
    --pretrained_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl \
    --num_classes 100
```

