# IFViT Checkpoint Guide

## 🌐 Environment-Specific Paths

### Local Development
- Base directory: `.` (current directory)
- Checkpoints: `./checkpoints/...`

### Kaggle Notebooks
- Base directory: `/kaggle/working/IFViT`
- Checkpoints: `/kaggle/working/IFViT/checkpoints/...`

**Note**: Scripts tự động detect Kaggle environment và adjust paths.

## 📁 Checkpoint Locations

### Module 1 (Dense Registration)

**Checkpoint Directory**: 
- Local: `./checkpoints/dense_reg/`
- Kaggle: `/kaggle/working/IFViT/checkpoints/dense_reg/`
- Configurable via `DENSE_CONFIG["checkpoint_dir"]`

**Checkpoint Files**:
- **Final checkpoint**: 
  - Local: `./checkpoints/dense_reg/dense_reg_ckpt.pkl`
  - Kaggle: `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl`
  - Lưu sau khi training hoàn tất
  - Chứa: `params`, `opt_state`, `step`, `metadata`
- **Periodic checkpoints**: 
  - Local: `./checkpoints/dense_reg/dense_reg_epoch_{N}.pkl`
  - Kaggle: `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_epoch_{N}.pkl`
  - Lưu mỗi `save_every` epochs (mặc định: 5 epochs)
  - Format: `dense_reg_epoch_5.pkl`, `dense_reg_epoch_10.pkl`, ...

**Cấu trúc checkpoint**:
```python
{
    'state': {
        'params': {...},      # Model parameters
        'opt_state': {...},   # Optimizer state
        'step': int           # Training step
    },
    'metadata': {
        'epoch': int,         # Epoch number
        'config': {...}       # Training config
    },
    'timestamp': str          # ISO timestamp
}
```

### Module 2 (Fingerprint Matcher)

**Checkpoint Directory**: 
- Local: `./checkpoints/matcher/`
- Kaggle: `/kaggle/working/IFViT/checkpoints/matcher/`
- Configurable via `MATCH_CONFIG["checkpoint_dir"]`

**Checkpoint Files**:
- **Final checkpoint**: 
  - Local: `./checkpoints/matcher/matcher_ckpt.pkl`
  - Kaggle: `/kaggle/working/IFViT/checkpoints/matcher/matcher_ckpt.pkl`
- **Periodic checkpoints**: 
  - Local: `./checkpoints/matcher/matcher_epoch_{N}.pkl`
  - Kaggle: `/kaggle/working/IFViT/checkpoints/matcher/matcher_epoch_{N}.pkl`

## 🔄 Module 1 → Module 2 Weight Loading

### Cách hoạt động

1. **Module 1 training**:
   
   **Local**:
   ```bash
   python ifvit-jax/train_dense.py \
       --dataset_root /path/to/data \
       --checkpoint_dir ./checkpoints/dense_reg
   ```
   
   **Kaggle**:
   ```bash
   cd /kaggle/working/IFViT
   python ifvit-jax/train_dense.py \
       --dataset_root /kaggle/input/fvc2002 \
       --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
   ```
   
   Sau khi training xong, checkpoint được lưu tại:
   - Local: `./checkpoints/dense_reg/dense_reg_ckpt.pkl`
   - Kaggle: `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl`

2. **Module 2 config**:
   
   Trong `ifvit-jax/config.py`, `MATCH_CONFIG` đã có sẵn path (relative):
   ```python
   MATCH_CONFIG = {
       ...
       "dense_reg_ckpt": "./checkpoints/dense_reg/dense_reg_ckpt.pkl",
       ...
   }
   ```
   
   **Trên Kaggle**, paths sẽ tự động được resolve thành:
   ```python
   # Tự động: ./checkpoints/... → /kaggle/working/IFViT/checkpoints/...
   ```

3. **Module 2 training** (tự động load weights):
   
   **Local**:
   ```bash
   python ifvit-jax/train_match.py \
       --dataset_root /path/to/data \
       --pretrained_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl \
       --checkpoint_dir ./checkpoints/matcher \
       --num_classes 100
   ```
   
   **Kaggle**:
   ```bash
   cd /kaggle/working/IFViT
   python ifvit-jax/train_match.py \
       --dataset_root /kaggle/input/fvc2002 \
       --pretrained_ckpt /kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl \
       --checkpoint_dir /kaggle/working/IFViT/checkpoints/matcher \
       --num_classes 100
   ```

   Hoặc không cần `--pretrained_ckpt` nếu config đã đúng (paths sẽ tự động resolve):
   ```bash
   # Local
   python ifvit-jax/train_match.py \
       --dataset_root /path/to/data \
       --checkpoint_dir ./checkpoints/matcher \
       --num_classes 100
   
   # Kaggle
   cd /kaggle/working/IFViT
   python ifvit-jax/train_match.py \
       --dataset_root /kaggle/input/fvc2002 \
       --checkpoint_dir /kaggle/working/IFViT/checkpoints/matcher \
       --num_classes 100
   ```
   
   Module 2 sẽ tự động load từ `MATCH_CONFIG["dense_reg_ckpt"]`.

### Weight Loading Logic

Khi Module 2 khởi tạo, trong `create_train_state()`:

1. Tạo model mới (MatcherModel)
2. Initialize parameters (random)
3. **Load Module 1 checkpoint**:
   ```python
   from ut.load_module1_weights import load_module1_transformer_weights
   
   params = load_module1_transformer_weights(
       module1_ckpt_path=pretrained_ckpt,
       module2_params=params,
       share_global_local=True  # Share weights between global & local branches
   )
   ```

4. Chỉ **transformer weights** được copy từ Module 1:
   - `loftr_transformer` hoặc `siamese_transformer`
   - Copy vào cả `loftr_transformer_global` và `loftr_transformer_local`
   - Các layers khác (embedding heads, ArcFace) giữ random initialization

### Verify Checkpoint

Để kiểm tra Module 1 checkpoint có sẵn sàng cho Module 2:

```bash
python -m ifvit-jax.ut.load_module1_weights \
    --module1_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl
```

Hoặc trong Python:
```python
from ifvit_jax.ut.load_module1_weights import verify_module1_loading

verify_module1_loading("./checkpoints/dense_reg/dense_reg_ckpt.pkl")
```

## 📝 Workflow Example

### Step 1: Train Module 1

**Local**:
```bash
python ifvit-jax/train_dense.py \
    --dataset_root /path/to/data \
    --checkpoint_dir ./checkpoints/dense_reg \
    --batch_size 128 \
    --num_epochs 100

# Checkpoint sẽ được lưu tại:
# ./checkpoints/dense_reg/dense_reg_ckpt.pkl
```

**Kaggle**:
```bash
cd /kaggle/working/IFViT
python ifvit-jax/train_dense.py \
    --dataset_root /kaggle/input/fvc2002 \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg \
    --batch_size 128 \
    --num_epochs 100

# Checkpoint sẽ được lưu tại:
# /kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl
```

### Step 2: Verify Module 1 Checkpoint

**Local**:
```bash
python -c "
from ifvit_jax.ut.load_module1_weights import verify_module1_loading
verify_module1_loading('./checkpoints/dense_reg/dense_reg_ckpt.pkl')
"
```

**Kaggle**:
```bash
cd /kaggle/working/IFViT
python -c "
from ifvit_jax.ut.load_module1_weights import verify_module1_loading
verify_module1_loading('/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl')
"
```

### Step 3: Train Module 2 (với Module 1 weights)

**Local**:
```bash
# Option 1: Dùng --pretrained_ckpt
python ifvit-jax/train_match.py \
    --dataset_root /path/to/data \
    --pretrained_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl \
    --checkpoint_dir ./checkpoints/matcher \
    --num_classes 100 \
    --batch_size 128 \
    --num_epochs 70

# Option 2: Dùng config mặc định (đã set sẵn path)
python ifvit-jax/train_match.py \
    --dataset_root /path/to/data \
    --checkpoint_dir ./checkpoints/matcher \
    --num_classes 100
```

**Kaggle**:
```bash
cd /kaggle/working/IFViT

# Option 1: Dùng --pretrained_ckpt
python ifvit-jax/train_match.py \
    --dataset_root /kaggle/input/fvc2002 \
    --pretrained_ckpt /kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/matcher \
    --num_classes 100 \
    --batch_size 128 \
    --num_epochs 70

# Option 2: Dùng config mặc định (đã set sẵn path)
python ifvit-jax/train_match.py \
    --dataset_root /kaggle/input/fvc2002 \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/matcher \
    --num_classes 100
```

### Step 4: Check Logs

Khi Module 2 bắt đầu training, bạn sẽ thấy log:

**Local**:
```
============================================================
Loading Module 1 Transformer Weights
============================================================
Source: ./checkpoints/dense_reg/dense_reg_ckpt.pkl
✓ Found transformer: loftr_transformer
✓ Sharing transformer weights between global and local branches
  ✓ Copied to loftr_transformer_global
  ✓ Copied to loftr_transformer_local (shared weights)
✓ Successfully loaded X,XXX,XXX parameters from Module 1
============================================================
```

**Kaggle**:
```
============================================================
Loading Module 1 Transformer Weights
============================================================
Source: /kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl
✓ Found transformer: loftr_transformer
✓ Sharing transformer weights between global and local branches
  ✓ Copied to loftr_transformer_global
  ✓ Copied to loftr_transformer_local (shared weights)
✓ Successfully loaded X,XXX,XXX parameters from Module 1
============================================================
```

## ⚙️ Configuration

### Thay đổi checkpoint paths

**Option 1: Sửa config file**
```python
# ifvit-jax/config.py
MATCH_CONFIG = {
    ...
    "dense_reg_ckpt": "/custom/path/to/dense_reg_ckpt.pkl",
    ...
}
```

**Option 2: Dùng command line**
```bash
python ifvit-jax/train_match.py \
    --pretrained_ckpt /custom/path/to/dense_reg_ckpt.pkl \
    ...
```

**Option 3: Dùng train_all.py (end-to-end)**
```bash
python ifvit-jax/train_all.py \
    --dense_checkpoint_dir ./checkpoints/dense_reg \
    --matcher_checkpoint_dir ./checkpoints/matcher
```

Script này sẽ tự động:
1. Train Module 1
2. Lấy path của Module 1 final checkpoint
3. Set vào Module 2 config
4. Train Module 2

## 🔍 Troubleshooting

### Module 2 không load được weights

**Lỗi**: `Module 1 checkpoint not found`

**Giải pháp**:
1. Kiểm tra path trong config:
   ```python
   print(MATCH_CONFIG["dense_reg_ckpt"])
   ```

2. Kiểm tra file có tồn tại:
   ```bash
   ls -lh ./checkpoints/dense_reg/dense_reg_ckpt.pkl
   ```

3. Dùng absolute path:
   ```bash
   python ifvit-jax/train_match.py \
       --pretrained_ckpt $(pwd)/checkpoints/dense_reg/dense_reg_ckpt.pkl
   ```

### Checkpoint không có transformer weights

**Lỗi**: `No transformer found in Module 1 checkpoint`

**Nguyên nhân**: Module 1 chưa train xong hoặc checkpoint bị lỗi

**Giải pháp**:
1. Verify checkpoint:
   ```python
   from ifvit_jax.ut.load_module1_weights import verify_module1_loading
   verify_module1_loading("./checkpoints/dense_reg/dense_reg_ckpt.pkl")
   ```

2. Retrain Module 1 nếu cần

### Module 2 không dùng weights từ Module 1

**Kiểm tra**: Xem log khi khởi tạo Module 2

**Nếu thấy**:
```
⚠ Warning: No Module 1 checkpoint provided!
```

**Giải pháp**: Đảm bảo `--pretrained_ckpt` được set hoặc config có `dense_reg_ckpt`

## 📊 Summary

### Local Development
| Component | Checkpoint Path | Loaded By |
|-----------|----------------|-----------|
| Module 1 | `./checkpoints/dense_reg/dense_reg_ckpt.pkl` | - |
| Module 2 | `./checkpoints/matcher/matcher_ckpt.pkl` | - |
| Module 1 → Module 2 | `./checkpoints/dense_reg/dense_reg_ckpt.pkl` | `train_match.py` (automatic) |

### Kaggle Notebooks
| Component | Checkpoint Path | Loaded By |
|-----------|----------------|-----------|
| Module 1 | `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl` | - |
| Module 2 | `/kaggle/working/IFViT/checkpoints/matcher/matcher_ckpt.pkl` | - |
| Module 1 → Module 2 | `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl` | `train_match.py` (automatic) |

**Key Points**:
- ✅ Module 1 lưu checkpoint tại:
  - Local: `./checkpoints/dense_reg/dense_reg_ckpt.pkl`
  - Kaggle: `/kaggle/working/IFViT/checkpoints/dense_reg/dense_reg_ckpt.pkl`
- ✅ Module 2 config đã có sẵn path: `MATCH_CONFIG["dense_reg_ckpt"]` (relative, tự động resolve)
- ✅ Module 2 tự động load transformer weights từ Module 1 khi training
- ✅ Chỉ transformer weights được copy, các layers khác random init
- ✅ **Trên Kaggle**: Luôn dùng absolute paths `/kaggle/working/IFViT/...` để đảm bảo paths đúng

