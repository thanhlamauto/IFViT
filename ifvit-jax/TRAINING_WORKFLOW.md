# IFViT Training Workflow - The Correct Way

This document explains the **correct workflow** for training IFViT following the paper exactly.

## ⚠️ Important: Module 1 → Module 2 Flow

**IFViT Paper Quote:**
> "The second module ... employs the ViTs **trained in the first module** with the additional fully connected layer and retrains them ..."

**Key Point:** Module 2 does NOT load LoFTR weights directly. It loads the TRAINED transformer from Module 1!

## 📊 Workflow Diagram

```
LoFTR Pretrained → Module 1 (Train) → Module 2 (Train) → Final Model
    (.npz)         Dense Reg + L_D    Matcher + L_D+L_E+L_A
```

## 🔄 Detailed Steps

### Step 1: Prepare LoFTR Pretrained Weights

```bash
# Download LoFTR checkpoint
wget https://github.com/zju3dv/LoFTR/releases/download/v1.0/outdoor_ds.ckpt

# Convert to JAX format
python convert_loftr_checkpoint.py \
    --pytorch_ckpt outdoor_ds.ckpt \
    --output loftr_transformer.npz
```

### Step 2: Train Module 1 (Dense Registration)

**Purpose:** Learn dense correspondences with LoFTR initialization

**Configuration:**
```python
# config.py - DENSE_CONFIG
{
    "use_loftr": True,  # ✓ Use LoFTR architecture
    "loftr_pretrained_ckpt": "./loftr_transformer.npz",  # ✓ Load LoFTR weights
    "lambda_D": 1.0,  # Only L_D loss
}
```

**Training:**
```bash
python train_dense.py \
    --dataset_root /path/to/fingerprint/data \
    --checkpoint_dir ./checkpoints/dense_reg
```

**Output:** `./checkpoints/dense_reg/dense_reg_ckpt.pkl` (contains TRAINED transformer)

### Step 3: Train Module 2 (Matcher)

**Purpose:** Learn embeddings for verification using Module 1's trained transformer

**Configuration:**
```python
# config.py - MATCH_CONFIG
{
    "use_loftr": True,  # ✓ Use same architecture as Module 1
    "dense_reg_ckpt": "./checkpoints/dense_reg/dense_reg_ckpt.pkl",  # ✓ Load from trained Module 1
    "lambda_D": 0.5,
    "lambda_E": 0.1,
    "lambda_A": 1.0,
}
```

**Training:**
```bash
python train_match.py \
    --dataset_root /path/to/fingerprint/data \
    --num_classes 100 \
    --pretrained_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl \
    --checkpoint_dir ./checkpoints/matcher
```

**What happens internally:**
1. Initialize MatcherModel with random weights
2. Load **Module 1's trained transformer** (NOT LoFTR) into both global and local branches
3. Share transformer weights between global/local (faithful to paper)
4. Train with L_D + L_E + L_A losses

**Output:** `./checkpoints/matcher/matcher_ckpt.pkl` (final model for inference)

## ✅ Correct vs ❌ Incorrect

### ✅ Correct (Following IFViT Paper)

```
Module 1: LoFTR init → train with L_D → save trained transformer
Module 2: Load Module 1's trained transformer → train with L_D+L_E+L_A
```

### ❌ Incorrect (What to avoid)

```
Module 1: LoFTR init → train with L_D → save
Module 2: LoFTR init directly → train with L_D+L_E+L_A  ← WRONG!
```

**Why incorrect?** Module 2 discards Module 1's learning, defeats the purpose of two-stage training.

## 🔍 Verification

Check that Module 2 loaded Module 1 correctly:

```bash
# Verify Module 1 checkpoint
python load_module1_weights.py --module1_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl

# Look for this in training logs:
# "Loading Module 1 Transformer Weights"
# "✓ Found transformer: loftr_transformer"
# "✓ Copied to loftr_transformer_global"
# "✓ Copied to loftr_transformer_local (shared weights)"
```

## 🎯 Key Implementation Details

### Transformer Weight Sharing

Module 2 uses `share_global_local=True` by default:

```python
# In train_match.py
params = load_module1_transformer_weights(
    module1_ckpt_path=pretrained_ckpt,
    module2_params=params,
    share_global_local=True  # ✓ Share Module 1's transformer
)
```

**Why share?**
- IFViT paper uses "the ViTs" (plural = 2 Siamese branches)
- Both branches learn from same Module 1 initialization
- More parameter efficient
- Avoids training separate transformers from scratch

### Module 1 Checkpoint Structure

```
dense_reg_ckpt.pkl/
├── params/
│   ├── ResNet18/          # Backbone weights
│   ├── loftr_transformer/ # ✓ This is what Module 2 loads
│   └── DenseMatchingHead/ # Matching head (not used in Module 2)
└── metadata/
    └── config, epoch, etc.
```

### Module 2 Parameter Loading

```
MatcherModel params (before loading):
├── ResNet18/ (random)          ← Not loaded, will train from scratch
├── loftr_transformer_global/   ← Loaded from Module 1
├── loftr_transformer_local/    ← Loaded from Module 1 (same weights)
├── EmbeddingHead/ (random)     ← New, will train from scratch
└── DenseMatchingHead/ (random) ← For auxiliary L_D loss

After load_module1_transformer_weights():
├── ResNet18/ (still random)
├── loftr_transformer_global/   ← ✓ Module 1's trained weights
├── loftr_transformer_local/    ← ✓ Module 1's trained weights (shared)
├── EmbeddingHead/ (still random)
└── DenseMatchingHead/ (still random)
```

## 📝 Training Logs to Expect

### Module 1 Training
```
Loading LoFTR weights from: ./loftr_transformer.npz
✓ Loaded 48 parameter arrays
✓ Merged 48/48 LoFTR parameters

DenseRegModel Summary
Total parameters: 12,345,678

Starting training...
Epoch 1/50 | L_D: 2.345
...
```

### Module 2 Training
```
============================================================
Loading Module 1 Transformer Weights
============================================================
Module 1 checkpoint: ./checkpoints/dense_reg/dense_reg_ckpt.pkl
This loads the TRAINED transformer from Module 1,
NOT fresh LoFTR weights (as per IFViT paper)
============================================================

✓ Found transformer: loftr_transformer
✓ Sharing transformer weights between global and local branches
  ✓ Copied to loftr_transformer_global
  ✓ Copied to loftr_transformer_local (shared weights)

✓ Successfully loaded 6,543,210 parameters from Module 1
============================================================

MatcherModel Summary
Total parameters: 15,678,901

Starting training...
Epoch 1/40 | L_D: 1.234 | L_E: 0.456 | L_A: 2.345
...
```

## 🚨 Common Mistakes to Avoid

1. **Setting `loftr_pretrained_ckpt` in MATCH_CONFIG**
   - ❌ Module 2 should NOT load LoFTR directly
   - ✓ Only set this in DENSE_CONFIG for Module 1

2. **Not providing `dense_reg_ckpt`**
   - ❌ Module 2 with random transformer defeats two-stage training
   - ✓ Always provide trained Module 1 checkpoint

3. **Using different `use_loftr` settings**
   - ❌ Module 1 with LoFTR, Module 2 with generic → incompatible
   - ✓ Both should use `use_loftr=True` for consistency

4. **Separate global/local transformers**
   - ⚠️ Not wrong, but less parameter efficient
   - ✓ Sharing weights is more faithful to paper

## 📊 Expected Benefits

By following this workflow:

- ✅ **Faster convergence** in Module 2 (builds on Module 1's learning)
- ✅ **Better features** (transformer pretrained on matching task)
- ✅ **Faithful to paper** (exact implementation as described)
- ✅ **Parameter efficient** (shared weights between branches)

## 🎓 Summary

**The Golden Rule:**

> LoFTR → Module 1 → Module 2
> 
> Each arrow is a checkpoint transfer, never skip Module 1!

Following this workflow ensures your implementation matches the IFViT paper exactly. 🎯
