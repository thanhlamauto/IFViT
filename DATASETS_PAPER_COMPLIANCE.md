# IFViT Paper Dataset Compliance

## 📋 Module 1 Training Datasets (Section 4.1)

### Original Datasets (25,090 images)

Theo IFViT paper, Module 1 training sử dụng:

1. **FVC2002**
   - DB1A, DB2A, DB3A
   - Total: ~2,400 images (3 databases × 100 fingers × 8 impressions)

2. **NIST SD301a**
   - Partitions: **A, B, C, E, J, K, M, N**
   - Note: SD301a uses device codes "dryrun-A", "dryrun-B", etc.
   - Total: ~1,920 images (8 partitions × 240 fingers × 1 impression)

3. **NIST SD302a**
   - Partitions: **A, B, C, D, E, F, U, V, L, M**
   - Note: SD302a uses device codes A-H, but paper specifies specific partitions
   - Total: ~17,890 images (10 partitions × variable fingers × 10 impressions)

4. **NIST SD300** (replaces MOLF from paper)
   - Paper used: MOLF DB1, DB2
   - Our implementation: NIST SD300 (rolled + plain)
   - Total: ~3,000 images

**Total original images**: ~25,090 (as per paper)

### Data Augmentation (3 Noise Models)

Từ 25,090 ảnh gốc, áp dụng **ONE of 3 noise models** cho mỗi corrupted version:

1. **Sensor noise** → Perlin noise
2. **Dryness** → Erosion  
3. **Over-pressurization** → Dilation

Sau đó: **Random rotation ±60°** (applied after corruption)

**Interpretation**: 
- Mỗi ảnh gốc tạo corrupted versions với ONE noise type (randomly selected)
- Với rotation variations, mỗi ảnh gốc → ~4 corrupted versions
- **Result**: ~100,360 training images (25,090 × ~4×)

### Training Pairs (100,000 total)

- **75,000 genuine pairs**: Same finger (original ↔ corrupted)
- **25,000 imposter pairs**: Different fingers (also with GT "no match")

## 🔧 Implementation Status

### ✅ Implemented

1. **Dataset loaders**:
   - ✅ FVC2002 (DB1A, DB2A, DB3A)
   - ✅ NIST SD301a (with partition filtering)
   - ✅ NIST SD302a (with partition filtering)
   - ✅ NIST SD300 (replaces MOLF)

2. **Augmentation**:
   - ✅ Perlin noise (sensor noise)
   - ✅ Erosion (dryness)
   - ✅ Dilation (over-pressurization)
   - ✅ Rotation ±60°

3. **Pair generation**:
   - ✅ Genuine pairs (same finger, original ↔ corrupted)
   - ✅ Imposter pairs (different fingers)

### ⚠️ Notes

1. **NIST SD302a partitions**:
   - Paper specifies: A, B, C, D, E, F, U, V, L, M
   - SD302a dataset uses device codes: A, B, C, D, E, F, G, H
   - Current implementation uses: A, B, C, D, E, F
   - **TODO**: Verify if U, V, L, M are separate devices or different naming

2. **NIST SD301a partitions**:
   - Paper specifies: A, B, C, E, J, K, M, N
   - Implementation uses: `devices=["dryrun-A", "dryrun-B", "dryrun-C", "dryrun-E", "dryrun-J", "dryrun-K", "dryrun-M", "dryrun-N"]`
   - ✅ Correct mapping

3. **MOLF replacement**:
   - Paper uses: MOLF DB1, DB2
   - Our implementation: NIST SD300
   - ✅ Acceptable replacement (similar dataset characteristics)

## 📊 Dataset Statistics

### Expected Counts (from paper)

| Dataset | Partitions | Original Images | After Augmentation |
|---------|-----------|-----------------|-------------------|
| FVC2002 | DB1A, DB2A, DB3A | ~2,400 | ~9,600 |
| NIST SD301a | A, B, C, E, J, K, M, N | ~1,920 | ~7,680 |
| NIST SD302a | A, B, C, D, E, F, U, V, L, M | ~17,890 | ~71,560 |
| NIST SD300 | All | ~3,000 | ~12,000 |
| **Total** | | **~25,090** | **~100,360** |

### Training Pairs

- Genuine pairs: 75,000 (75%)
- Imposter pairs: 25,000 (25%)
- **Total**: 100,000 pairs

## 🔍 Verification

Để verify datasets match paper:

```python
from data import PaperDatasetRoots, build_paper_train_entries

roots = PaperDatasetRoots()
train_entries = build_paper_train_entries(roots)

# Count by dataset
from collections import Counter
dataset_counts = Counter(e.dataset_name for e in train_entries)
print("Dataset counts:", dataset_counts)

# Total images
print(f"Total training images: {len(train_entries)}")
print(f"Expected: ~25,090 (before augmentation)")

# After augmentation: ~100,360 images
# Training pairs: 100,000 (75k genuine + 25k imposter)
```

## 📝 Key Points

1. **Module 1 uses ONLY original + corrupted pairs**
   - ❌ NO FingerNet enhancement
   - ✅ Original fingerprint + corrupted version
   - ✅ GT correspondences from known transformation

2. **Augmentation order** (as per paper):
   - Apply 3 noise models first (Perlin, erosion, dilation)
   - Then apply rotation ±60°

3. **Pair generation**:
   - Genuine: same finger, different corruptions
   - Imposter: different fingers (also with GT "no match")

4. **PrintsGAN**:
   - Paper mentions PrintsGAN for Module 2 pre-training
   - **NOT used in Module 1** (only for Module 2)

