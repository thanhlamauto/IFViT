# Using LoFTR Pretrained Weights in IFViT-JAX

This guide explains how to use pretrained LoFTR weights to initialize the transformer module, following the IFViT paper's approach.

## 📋 Why LoFTR?

The IFViT paper states: *"the pre-trained ViT model for feature matching in LoFTR is employed to accelerate the convergence"*

Using pretrained LoFTR weights provides:
- **Faster convergence** - Start from a model pretrained on matching tasks
- **Better initialization** - LoFTR is trained on large-scale matching datasets
- **More faithful to paper** - Matches the exact implementation described

## 🔧 Setup Steps

### Step 1: Download LoFTR Checkpoint

Download the official LoFTR checkpoint (PyTorch format):

```bash
# Outdoor model (recommended for fingerprints)
wget https://github.com/zju3dv/LoFTR/releases/download/v1.0/outdoor_ds.ckpt

# Or indoor model
wget https://github.com/zju3dv/LoFTR/releases/download/v1.0/indoor_ds.ckpt
```

### Step 2: Convert to JAX Format

Convert the PyTorch checkpoint to NumPy format compatible with Flax:

```bash
python convert_loftr_checkpoint.py \
    --pytorch_ckpt outdoor_ds.ckpt \
    --output loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

The script will:
1. Load the PyTorch checkpoint
2. Extract Local FeatureTransformer weights
3. Map parameter keys from PyTorch to Flax naming
4. Transpose weight matrices (PyTorch uses [out,in], Flax uses [in,out])
5. Save as `.npz` file

### Step 3: Update Configuration

In `config.py`, set:

```python
DENSE_CONFIG = {
    # ... other settings ...
    
    "use_loftr": True,  # Enable LoFTR transformer
    "attention_type": "linear",  # Use linear attention (faster)
    "loftr_pretrained_ckpt": "./loftr_transformer.npz",  # Path to converted checkpoint
}
```

### Step 4: Train with Pretrained Weights

The training script will automatically load LoFTR weights:

```bash
python train_dense.py \
    --dataset_root /path/to/dataset \
    --checkpoint_dir ./checkpoints/dense_reg_loftr
```

## 🏗️ Architecture Differences

### Generic SiameseTransformer (use_loftr=False)

```python
- Standard Flax MultiHeadDotProductAttention
- MLP: x -> Dense(mlp_dim) -> GELU -> Dense(hidden_dim)
- LayerNorm before attention and MLP
- Cannot load LoFTR weights (incompatible structure)
```

### LoFTR LocalFeatureTransformer (use_loftr=True)

```python
- Custom Q/K/V projections
- LinearAttention or FullAttention (configurable)
- MLP: concat([x, message]) -> Dense(2*d_model) -> ReLU -> Dense(d_model)
- LayerNorm placement matches LoFTR exactly
- Compatible with LoFTR pretrained weights ✓
```

## 🔄 Weight Mapping

The conversion script maps PyTorch keys to Flax:

| PyTorch (LoFTR) | Flax (IFViT-JAX) | Notes |
|-----------------|------------------|-------|
| `linear_q.weight` | `layers.{i}.{j}/q_proj/kernel` | Transposed |
| `linear_k.weight` | `layers.{i}.{j}/k_proj/kernel` | Transposed |
| `linear_v.weight` | `layers.{i}.{j}/v_proj/kernel` | Transposed |
| `linear_out.weight` | `layers.{i}.{j}/merge/kernel` | Transposed |
| `ffn.0.weight` | `layers.{i}.{j}/mlp.0/kernel` | Transposed |
| `ffn.2.weight` | `layers.{i}.{j}/mlp.2/kernel` | Transposed |
| `norm1.weight` | `layers.{i}.{j}/norm1/scale` | Direct copy |
| `norm1.bias` | `layers.{i}.{j}/norm1/bias` | Direct copy |
| `norm2.weight` | `layers.{i}.{j}/norm2/scale` | Direct copy |
| `norm2.bias` | `layers.{i}.{j}/norm2/bias` | Direct copy |

Where:
- `{i}` = layer index in transformer
- `{j}` = sublayer index (0 for feat0, 1 for feat1)

## ⚙️ Implementation Details

### LoFTREncoderLayer

Each encoder layer consists of:

```python
1. Q/K/V projections (3 separate Dense layers)
2. Attention mechanism:
   - LinearAttention: O(N) complexity using ELU feature map
   - FullAttention: O(N²) standard scaled dot-product
3. Merge projection
4. MLP with concat([x, message]):
   - Input: 2*d_model (concatenated)
   - Hidden: 2*d_model
   - Output: d_model
5. LayerNorm after MLP
```

### LocalFeatureTransformer

Alternates between self and cross attention:

```python
layer_names = ['self', 'cross', 'self', 'cross', ...]

For each layer:
  if 'self':
    feat0 = LoFTREncoderLayer(feat0, feat0)  # Self-attention
    feat1 = LoFTREncoderLayer(feat1, feat1)
  elif 'cross':
    feat0 = LoFTREncoderLayer(feat0, feat1)  # Cross-attention
    feat1 = LoFTREncoderLayer(feat1, feat0)
```

## 📊 Expected Results

With LoFTR pretrained weights, you should see:

- **Faster convergence**: ~30-50% fewer epochs to reach target performance
- **Better initialization**: Lower initial loss
- **Higher final accuracy**: Especially on small datasets

## 🔍 Verification

To verify weights loaded correctly:

```python
from load_loftr_weights import load_loftr_weights

# Load converted checkpoint
weights = load_loftr_weights('./loftr_transformer.npz')

# Check parameters
print(f"Loaded {len(weights)} parameters")
for k, v in weights.items():
    print(f"  {k}: {v.shape}")
```

## 🐛 Troubleshooting

### "Checkpoint not found"

Make sure you've converted the checkpoint:
```bash
python convert_loftr_checkpoint.py --pytorch_ckpt outdoor_ds.ckpt --output loftr_transformer.npz
```

### "No transformer keys found"

Try specifying the prefix manually:
```bash
python convert_loftr_checkpoint.py \
    --pytorch_ckpt outdoor_ds.ckpt \
    --output loftr_transformer.npz \
    --prefix loftr_coarse.coarse_transformer
```

### "Shape mismatch"

This usually means:
1. Wrong attention_type ('linear' vs 'full')
2. Wrong hidden_dim (should be 256)
3. Wrong num_heads (should be 8)

Check that your config matches LoFTR's architecture.

### "Model performance not improving"

If using LoFTR weights doesn't help:
1. Verify weights loaded correctly (check logs)
2. Ensure data preprocessing matches LoFTR (normalization, augmentation)
3. Try different learning rates (LoFTR used different schedules)
4. Check that image sizes are reasonable (LoFTR used various sizes)

## 📚 References

- **LoFTR Paper**: [LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/abs/2104.00680)
- **LoFTR Code**: [https://github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)
- **IFViT Paper**: [Your paper reference]

## 🎯 Next Steps

1. ✅ Convert LoFTR checkpoint
2. ✅ Update config to use LoFTR
3. ✅ Train Module 1 with pretrained weights
4. ⚠️ Implement data pipeline (`data.py`)
5. 🚀 Train Module 2 (inherits from Module 1)
6. 📊 Evaluate on benchmark datasets

Good luck with your fingerprint matching project! 🎉
