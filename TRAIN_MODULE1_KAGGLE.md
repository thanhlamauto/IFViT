# Train Module 1 trên Kaggle với LoFTR Pretrained Weights

## 🚀 Lệnh Train

### Option 1: Set trong Config (Recommended)

Tạo hoặc update config trong notebook:

```python
import sys
sys.path.insert(0, '/kaggle/working/IFViT/ifvit-jax')

from config import DENSE_CONFIG

# Update config với LoFTR checkpoint path
DENSE_CONFIG['loftr_pretrained_ckpt'] = '/kaggle/input/pretrained-loftr/jax/default/1/loftr_transformer.npz'
DENSE_CONFIG['checkpoint_dir'] = '/kaggle/working/IFViT/checkpoints/dense_reg'
DENSE_CONFIG['use_loftr'] = True

# Train
from train_dense_tpu import train_dense_reg_tpu
train_dense_reg_tpu(config=DENSE_CONFIG)
```

### Option 2: Command Line (Nếu có script)

```bash
cd /kaggle/working/IFViT

python ifvit-jax/train_dense_tpu.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

**Lưu ý**: Cần set `loftr_pretrained_ckpt` trong config trước.

## 📝 Full Notebook Example

```python
# Cell 1: Setup
import sys
import os
sys.path.insert(0, '/kaggle/working/IFViT')
os.chdir('/kaggle/working/IFViT')

# Cell 2: Update Config
import sys
sys.path.insert(0, '/kaggle/working/IFViT/ifvit-jax')

from config import DENSE_CONFIG

# Set LoFTR pretrained checkpoint
DENSE_CONFIG['loftr_pretrained_ckpt'] = '/kaggle/input/pretrained-loftr/jax/default/1/loftr_transformer.npz'
DENSE_CONFIG['checkpoint_dir'] = '/kaggle/working/IFViT/checkpoints/dense_reg'
DENSE_CONFIG['use_loftr'] = True

print("Config updated:")
print(f"  LoFTR checkpoint: {DENSE_CONFIG['loftr_pretrained_ckpt']}")
print(f"  Checkpoint dir: {DENSE_CONFIG['checkpoint_dir']}")
print(f"  Use LoFTR: {DENSE_CONFIG['use_loftr']}")

# Cell 3: Train
from train_dense_tpu import train_dense_reg_tpu

train_dense_reg_tpu(config=DENSE_CONFIG)
```

## 🔧 Alternative: Update Config File

Nếu muốn update config file trực tiếp:

```python
# Update ifvit-jax/config.py
DENSE_CONFIG = {
    ...
    "use_loftr": True,
    "loftr_pretrained_ckpt": "/kaggle/input/pretrained-loftr/jax/default/1/loftr_transformer.npz",
    "checkpoint_dir": "/kaggle/working/IFViT/checkpoints/dense_reg",
    ...
}
```

Sau đó chạy:

```bash
cd /kaggle/working/IFViT
python ifvit-jax/train_dense_tpu.py
```

## ✅ Verification

Trước khi train, verify checkpoint có thể load:

```python
import sys
sys.path.insert(0, '/kaggle/working/IFViT/ifvit-jax')
from ut.load_loftr_weights import load_loftr_weights

ckpt_path = '/kaggle/input/pretrained-loftr/jax/default/1/loftr_transformer.npz'
weights = load_loftr_weights(ckpt_path)
print(f"✓ Loaded {len(weights)} parameter arrays")
```

## 📊 Expected Output

Khi train, bạn sẽ thấy:

```
============================================================
Loading LoFTR Pretrained Weights
============================================================
Checkpoint: /kaggle/input/pretrained-loftr/jax/default/1/loftr_transformer.npz
============================================================

Loading LoFTR weights from: /kaggle/input/pretrained-loftr/jax/default/1/loftr_transformer.npz
  Loaded 80 parameter arrays
  Total parameters: 5,251,072
  ✓ Merged: layers.0.0/q_proj/kernel (256, 256)
  ...
✓ LoFTR weights loaded successfully
```

