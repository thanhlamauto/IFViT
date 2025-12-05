# Module 1 Training: Data Flow & Processing Pipeline

## 📋 Overview

Khi chạy `train_dense.py`, đây là toàn bộ quy trình từ load data đến training:

```bash
python ifvit-jax/train_dense.py \
    --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
```

## 🔄 Complete Data Flow

### Step 1: Initialize Training

```
train_dense.py
├── Load config (DENSE_CONFIG)
├── Initialize logger
├── Create training state (model + optimizer)
└── Start training loop
```

**Config mặc định**:
- `image_size`: 128×128
- `batch_size`: 128
- `num_epochs`: 100
- `lr`: 1e-3
- `lambda_D`: 1.0 (chỉ dùng L_D loss)

### Step 2: Load Dataset Entries

**Cần load entries trước khi gọi `dense_reg_dataset`**:

```python
from data import PaperDatasetRoots, build_paper_train_entries

# Initialize dataset roots (tự động detect Kaggle paths)
roots = PaperDatasetRoots()

# Build train entries từ tất cả datasets
train_entries = build_paper_train_entries(roots)
# Returns: List[FingerprintEntry]
# Mỗi entry có: path, finger_local_id, impression_id, dataset_name, split, finger_global_id
```

**Datasets được load (theo IFViT paper Section 4.1)**:
- **FVC2002**: DB1A, DB2A, DB3A (training sets)
- **NIST SD301a**: Partitions A, B, C, E, J, K, M, N
- **NIST SD302a**: Partitions A, B, C, D, E, F, U, V, L, M
- **NIST SD300**: Replaces MOLF (DB1, DB2) from paper

**Total**: ~25,090 original fingerprint images (as per paper)

**Paths trên Kaggle**:
- `/kaggle/input/fvc2002/FVC2002`
- `/kaggle/input/fvc2004/FVC2004`
- `/kaggle/input/nist-sd300/NIST SD300`
- `/kaggle/input/nist-sd4/NIST4`
- `/kaggle/input/sd301a/images`
- `/kaggle/input/sd302a/images`

### Step 3: Create Dataset Iterator

```python
from data import dense_reg_dataset

train_dataset = dense_reg_dataset(
    entries=train_entries,  # List[FingerprintEntry]
    config=DENSE_CONFIG,
    split='train',
    shuffle=True
)
```

**Dataset iterator** (`dense_reg_dataset`) sẽ:

1. **Filter entries** theo split (train/val/test)
2. **Shuffle** entries nếu `shuffle=True`
3. **Batch** entries theo `batch_size` (mặc định: 128)

### Step 4: Process Each Entry (Trong mỗi batch)

Với mỗi `FingerprintEntry` trong batch:

#### 4.1. Load Image
```python
img = load_image(entry.path)
# Returns: (H, W) float32 [0, 255]
# - Load bằng OpenCV (cv2.imread)
# - Convert sang float32
```

#### 4.2. Normalize & Resize
```python
img = normalize_image(img, target_size=(128, 128))
# Steps:
# 1. Normalize: img / 255.0 → [0, 1]
# 2. Resize to 128×128 (bilinear interpolation)
# Returns: (128, 128) float32 [0, 1]
```

#### 4.3. Apply Corruption & Augmentation

```python
corrupted, transform = random_corrupt_fingerprint(
    (img * 255).astype(np.float32),  # Convert back to [0, 255]
    rng_key
)
```

**Augmentation pipeline** (`random_corrupt_fingerprint`) - **Theo IFViT paper Section 4.1**:

**3 Noise Models** (as per paper):

1. **Sensor noise** → **Perlin noise** (always applied):
   ```python
   perlin_noise = generate_perlin_noise((H, W), scale=10.0)
   corrupted = corrupted + perlin_noise * noise_scale
   ```
   - Multi-octave noise (1, 2, 4, 8 octaves)
   - Simulates sensor noise

2. **Dryness** → **Erosion** (33% chance):
   ```python
   if random() < 0.33:
       corrupted = cv2.erode(corrupted, kernel=(3,3))
   ```
   - Simulates dry finger conditions

3. **Over-pressurization** → **Dilation** (33% chance):
   ```python
   if random() < 0.33:
       corrupted = cv2.dilate(corrupted, kernel=(3,3))
   ```
   - Simulates excessive pressure

4. **Rotation** (±60°) - **Applied AFTER corruption** (as per paper):
   ```python
   angle = random.uniform(-60, 60)  # degrees
   corrupted = cv2.warpAffine(corrupted, M_rot, (W, H))
   ```

**Result**:
- `corrupted`: (128, 128) float32 [0, 255]
- `transform`: 3×3 rotation matrix (rotation only, applied last)

**Normalize corrupted image**:
```python
corrupted = corrupted / 255.0  # Back to [0, 1]
```

**Note**: Paper states Module 1 uses **ONLY original + corrupted pairs**, 
**NO FingerNet enhancement** (FingerNet is only for Module 2).

#### 4.4. Generate Ground-Truth Correspondences

```python
matches, valid = generate_gt_correspondences(
    img,           # Original (128, 128) [0, 1]
    corrupted,     # Corrupted (128, 128) [0, 1]
    transform,     # 3×3 transformation matrix
    num_points=1000
)
```

**Process**:

1. **Sample random points** trong original image:
   ```python
   y_coords = random.randint(0, H, size=1000)
   x_coords = random.randint(0, W, size=1000)
   pts1 = [x, y, 1]  # Homogeneous coordinates
   ```

2. **Transform points**:
   ```python
   pts2 = transform @ pts1.T  # Apply transformation
   pts2 = pts2[:2] / pts2[2]  # Convert back to 2D
   ```

3. **Check validity** (points trong bounds):
   ```python
   valid = (pts2_x >= 0) & (pts2_x < W) & (pts2_y >= 0) & (pts2_y < H)
   ```

4. **Return**:
   - `matches`: (N, 4) array `[x1, y1, x2, y2]`
   - `valid`: (N,) boolean mask

### Step 5: Batch Assembly

```python
batch_img1 = []      # List of original images
batch_img2 = []      # List of corrupted images
batch_matches = []   # List of correspondence arrays
batch_valid = []     # List of validity masks

for entry in batch_entries:
    # ... process entry ...
    batch_img1.append(img[..., None])        # (128, 128, 1)
    batch_img2.append(corrupted[..., None])  # (128, 128, 1)
    batch_matches.append(matches)             # (N, 4)
    batch_valid.append(valid)                # (N,)
```

**Padding** (để tất cả samples có cùng số correspondences):

```python
max_matches = max(len(m) for m in batch_matches)

# Pad shorter arrays
for m, v in zip(batch_matches, batch_valid):
    if len(m) < max_matches:
        pad = zeros((max_matches - len(m), 4))
        m = vstack([m, pad])
        v = concatenate([v, zeros(...)])
```

**Final batch**:

```python
batch = {
    'img1': jnp.array(...),      # (B, 128, 128, 1) float32 [0, 1]
    'img2': jnp.array(...),      # (B, 128, 128, 1) float32 [0, 1]
    'matches': jnp.array(...),   # (B, K, 4) float32 [x1, y1, x2, y2]
    'valid_mask': jnp.array(...) # (B, K) bool
}
```

### Step 6: Preprocess Batch

```python
batch = preprocess_batch(batch)
# Hiện tại chỉ pass-through (images đã normalize)
# Có thể thêm: mean subtraction, data augmentation, etc.
```

### Step 7: Training Step

```python
@jax.jit
def train_step(state, batch, rng, config):
    # Forward pass
    P, matches, feat1, feat2 = model.apply(
        params, batch['img1'], batch['img2'], train=True
    )
    # P: (B, 64, 64) matching probability matrix
    # matches: (B, 100, 2) top matches
    # feat1, feat2: (B, 8, 8, 256) refined features
    
    # Compute loss
    losses = total_loss_dense(
        P=P,
        gt_matches=batch['matches'],
        valid_mask=batch['valid_mask'],
        lambda_D=1.0
    )
    # L_D: Dense correspondence loss
    
    # Backward pass
    grads = grad(loss_fn)(params)
    
    # Update parameters
    state = state.apply_gradients(grads)
    
    return state, metrics
```

**Model forward pass**:

1. **ResNet-18 backbone**:
   - Input: `img1`, `img2` (B, 128, 128, 1)
   - Output: `feat1`, `feat2` (B, 8, 8, 256)
   - Scale: 128 → 8 (16× downsampling)

2. **Siamese Transformer**:
   - Input: `feat1`, `feat2` (B, 8, 8, 256)
   - Process: Self-attention + Cross-attention (4 layers)
   - Output: `refined_feat1`, `refined_feat2` (B, 8, 8, 256)

3. **Dense Matching Head**:
   - Input: `refined_feat1`, `refined_feat2`
   - Compute correlation matrix: (B, 64, 64)
   - Apply dual-softmax
   - Output: `P` (B, 64, 64) matching probability matrix

**Loss computation**:

```python
L_D = dense_reg_loss(
    P,                    # (B, 64, 64) matching probabilities
    gt_matches,           # (B, K, 4) [x1, y1, x2, y2]
    valid_mask,           # (B, K) boolean
    feature_shape=(8, 8)  # Feature map size
)

# Process:
# 1. Convert GT matches to feature map coordinates (scale by 8)
# 2. Convert to flat indices
# 3. Gather probabilities from P
# 4. Compute negative log likelihood
# 5. Average over valid matches
```

### Step 8: Logging & Checkpointing

```python
# Log metrics every N steps
if global_step % log_every == 0:
    logger.log_metrics(step, metrics)

# Save checkpoint every N epochs
if (epoch + 1) % save_every == 0:
    save_checkpoint(f"dense_reg_epoch_{epoch+1}.pkl", ...)

# Save final checkpoint
save_checkpoint("dense_reg_ckpt.pkl", ...)
```

## 📊 Data Statistics

**Per batch**:
- Batch size: 128 images
- Image size: 128×128×1
- Correspondences per image: ~1000 (sau padding: max trong batch)
- Feature map size: 8×8×256
- Matching matrix: 64×64 (8×8 flattened)

**Augmentation parameters** (theo IFViT paper):
- **3 Noise models**: Perlin noise (always) + Erosion (33%) + Dilation (33%)
- **Rotation**: ±60° (applied after corruption)
- **Result**: ~4× augmentation (25,090 original → ~100,360 training images)

**Training pairs** (theo paper):
- **100,000 pairs total**:
  - 75,000 genuine pairs (same finger: original ↔ corrupted)
  - 25,000 imposter pairs (different fingers, also with GT "no match")

## 🔍 Key Points

1. **Self-supervised learning**: 
   - Không cần manual annotations
   - GT correspondences được generate từ known transformation
   - **Module 1 KHÔNG dùng FingerNet** (chỉ dùng original + corrupted)

2. **Data augmentation** (theo IFViT paper):
   - **3 noise models**: Perlin noise (sensor) + Erosion (dryness) + Dilation (over-pressure)
   - Rotation ±60° (applied after corruption)
   - Mỗi epoch có augmentation khác nhau (random)
   - **Result**: ~4× augmentation (25k → 100k images)

3. **Training pairs**:
   - **75,000 genuine pairs**: same finger (original ↔ corrupted)
   - **25,000 imposter pairs**: different fingers (also with GT "no match")
   - **Total**: 100,000 pairs

4. **Ground-truth generation**:
   - Dựa trên transformation matrix (rotation only)
   - Chỉ valid points (trong bounds) được dùng

5. **Training objective**:
   - Maximize matching probability cho GT correspondences
   - Loss: L_D only (dense correspondence loss)

## 🐛 Current Implementation Note

**Hiện tại `train_dense.py` cần được update** để load entries:

```python
# Cần thêm vào train_dense.py:
from data import PaperDatasetRoots, build_paper_train_entries

# Trong train_dense_reg():
roots = PaperDatasetRoots()
train_entries = build_paper_train_entries(roots)

train_dataset = dense_reg_dataset(
    entries=train_entries,  # Thay vì dataset_root
    config=config,
    split='train',
    shuffle=True
)
```

## 📝 Summary Flow Diagram

```
Dataset Roots (Kaggle paths)
    ↓
build_paper_train_entries()
    ↓
List[FingerprintEntry] (train split)
    ↓
dense_reg_dataset()
    ↓
For each entry:
    load_image() → normalize_image() → random_corrupt_fingerprint() → generate_gt_correspondences()
    ↓
Batch: {img1, img2, matches, valid_mask}
    ↓
preprocess_batch()
    ↓
train_step()
    ├── Forward: ResNet → Transformer → Matching Head
    ├── Loss: L_D (dense correspondence)
    └── Backward: Update parameters
    ↓
Logging & Checkpointing
```

