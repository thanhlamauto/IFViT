# IFViT-JAX: Fingerprint Matching with JAX/Flax

Fingerprint matching system using dense correspondence learning and metric learning, implemented in JAX/Flax.

## 📁 Project Structure

```
ifvit-jax/
├── config.py          # Configuration dictionaries
├── data.py            # Data loading and augmentation (TODO: implement with real data)
├── models.py          # Model architectures (ResNet-18, Transformers, Matcher)
├── losses.py          # Loss functions (L_D, L_E, L_A) and score fusion
├── train_dense.py     # Training script for Module 1 (Dense Registration)
├── train_match.py     # Training script for Module 2 (Matcher)
├── train_all.py       # End-to-end training script (Module 1 → Module 2)
├── infer.py           # Inference and verification
└── ut/                # Utility modules
    ├── utils.py           # General utilities (checkpointing, logging, metrics)
    ├── enhance_and_roi.py  # FingerNet enhancement and ROI extraction
    ├── alignment.py       # Image alignment using RANSAC
    └── preprocess_fingernet.py  # Offline preprocessing script
```

## 🏗️ Architecture Overview

### Module 1: Dense Registration
- **Goal**: Learn dense pixel-to-pixel correspondences
- **Model**: ResNet-18 + Siamese Transformer + Dense Matching Head
- **Loss**: L_D (dense correspondence loss)
- **Output**: Matching probability matrix P

### Module 2: Fingerprint Matcher
- **Goal**: Fingerprint verification using global + local features
- **Model**: ResNet-18 + Siamese Transformer + Embedding Heads
- **Losses**:
  - L_D: Dense correspondence loss (optional)
  - L_E: Cosine embedding loss
  - L_A: ArcFace loss
- **Output**: Normalized embeddings for verification

## 🚀 Usage

### Training Module 1 (Dense Registration)

```bash
python train_dense.py \
    --dataset_root /path/to/dataset \
    --checkpoint_dir ./checkpoints/dense_reg
```

**Note**: `data.py` is currently stubbed out. Once you have real data, implement the dataset loading functions.

### Training Module 2 (Matcher)

```bash
python train_match.py \
    --dataset_root /path/to/dataset \
    --num_classes 100 \
    --pretrained_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl \
    --checkpoint_dir ./checkpoints/matcher
```

### Inference/Verification

```bash
python infer.py \
    --img1 /path/to/fingerprint1.png \
    --img2 /path/to/fingerprint2.png \
    --dense_ckpt ./checkpoints/dense_reg/dense_reg_ckpt.pkl \
    --matcher_ckpt ./checkpoints/matcher/matcher_ckpt.pkl
```

## ⚙️ Configuration

### DENSE_CONFIG (Module 1)
```python
{
    "image_size": 128,
    "batch_size": 64,
    "lr": 3e-4,
    "num_epochs": 50,
    "lambda_D": 1.0,  # Only L_D loss
}
```

### MATCH_CONFIG (Module 2)
```python
{
    "image_size": 224,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 40,
    "lambda_D": 0.5,
    "lambda_E": 0.1,
    "lambda_A": 1.0,
    "roi_size": 90,
}
```

## 📊 Model Components

### ResNet-18 Backbone
- Extracts feature maps at 1/8 resolution
- Shape: (B, H/8, W/8, 256)

### Siamese Transformer
- Self-attention on each feature map
- Cross-attention between feature maps
- Refines features for matching

### Dense Matching Head
- Computes correlation matrix
- Applies dual-softmax for soft correspondences
- Output: (B, N, N) matching probability matrix

### Embedding Head
- Projects features to fixed-length embeddings
- L2 normalization
- Output: (B, 256) normalized embeddings

## 🔬 Loss Functions

### L_D: Dense Correspondence Loss
Encourages high probability for ground-truth correspondences.

### L_E: Cosine Embedding Loss
- Genuine pairs: maximize similarity
- Imposter pairs: push below margin

### L_A: ArcFace Loss
Discriminative feature learning with angular margin.

## 📝 TODOs

### Data Pipeline (`data.py`)
All functions in `data.py` are currently stubbed out. When you have real data, implement:

1. **load_image()**: Load fingerprint images
2. **list_pairs()**: Parse dataset structure and create pair lists
3. **random_corrupt_fingerprint()**: Augmentation pipeline
4. **generate_gt_correspondences()**: Create ground-truth matches from known transforms
5. **compute_overlap_and_rois()**: Extract overlapping regions and ROI patches
6. **dense_reg_dataset()**: Batch generator for Module 1
7. **matcher_dataset()**: Batch generator for Module 2

### Dataset Structure
Expected structure (example):
```
dataset/
├── train/
│   ├── subject_001/
│   │   ├── finger1_impression1.png
│   │   ├── finger1_impression2.png
│   │   └── ...
│   ├── subject_002/
│   └── ...
├── val/
└── test/
```

## 🎯 Verification Pipeline

1. **Alignment**: Try multiple rotations to find best alignment
2. **Embedding Extraction**: 
   - Global features from full images
   - Local features from ROI patches
3. **Score Computation**:
   - Global similarity score
   - Local similarity score
   - Fused score (weighted average)
4. **Decision**: Compare fused score against threshold

## 📦 Dependencies

```bash
pip install jax jaxlib flax optax
pip install numpy opencv-python
```

For TPU support:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## 🔧 TPU Deployment

The code is designed to be TPU-ready. To scale to TPU v5e-8:

1. Use `jax.pmap` or `jax.pjit` for data parallelism
2. Shard batches across TPU cores
3. Update training loops with multi-device support

Example:
```python
# Replicate state across devices
state = flax.jax_utils.replicate(state)

# Use pmap for parallel training
train_step_pmap = jax.pmap(train_step, axis_name='batch')
```

## 📄 License

[Your license here]

## 🤝 Citation

If you use this code, please cite:
```
[Your citation]
```

## 📧 Contact

[Your contact information]
