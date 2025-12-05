"""
Image loading and dataset iterators for training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Dict, Iterator, Optional
from pathlib import Path
import cv2
import random
import warnings

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base import FingerprintEntry
from .augmentation import random_corrupt_fingerprint, generate_gt_correspondences
from .pairs import _generate_pairs


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load a fingerprint image from path with fallback for corrupt files.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array (H, W) float32 in range [0, 255], or None if failed
        
    Note:
        Returns None instead of raising to allow graceful handling of corrupt files.
    """
    # Try OpenCV first (faster)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img.astype(np.float32)
    
    # Fallback to PIL if OpenCV fails (handles some corrupt PNGs better)
    if PIL_AVAILABLE:
        try:
            img = np.array(Image.open(str(path)).convert('L'), dtype=np.float32)
            if img.size > 0:
                return img
        except Exception as e:
            warnings.warn(f"Failed to load image with PIL: {path} - {e}")
    
    # Both methods failed
    return None


def normalize_image(img: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Normalize image to [0, 1] and optionally resize.
    
    Args:
        img: (H, W) float32 in [0, 255]
        target_size: Optional (H, W) to resize to
        
    Returns:
        (H, W) or (target_H, target_W) float32 in [0, 1]
    """
    img = img / 255.0
    
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return img.astype(np.float32)


def _generate_dense_reg_pairs(
    entries: List[FingerprintEntry],
    imposter_ratio: float = 0.25
) -> List[Tuple[FingerprintEntry, Optional[FingerprintEntry], bool]]:
    """
    Generate pairs for dense registration training.
    
    According to IFViT paper:
    - Genuine pairs: same finger (original ↔ corrupted)
    - Imposter pairs: different fingers (GT "no match")
    
    Args:
        entries: List of FingerprintEntry
        imposter_ratio: Ratio of imposter pairs (default 0.25 = 25%)
        
    Returns:
        List of (entry1, entry2, is_genuine) tuples
        - Genuine: (entry, None, True) - entry will be paired with corrupted version
        - Imposter: (entry1, entry2, False) - different fingers, no correspondences
    """
    pairs = []
    
    # Genuine pairs: each entry → corrupted version (same finger)
    for entry in entries:
        pairs.append((entry, None, True))
    
    # Imposter pairs: different fingers
    finger_to_entries = {}
    for entry in entries:
        fid = entry.finger_global_id
        if fid is not None:
            if fid not in finger_to_entries:
                finger_to_entries[fid] = []
            finger_to_entries[fid].append(entry)
    
    n_genuine = len(pairs)
    n_imposter = int(n_genuine * imposter_ratio / (1 - imposter_ratio))  # To get 25% of total
    
    # Sample imposter pairs
    unique_fingers = list(finger_to_entries.keys())
    if len(unique_fingers) >= 2:
        for _ in range(n_imposter):
            if len(unique_fingers) < 2:
                break
            fid1, fid2 = random.sample(unique_fingers, 2)
            entry1 = random.choice(finger_to_entries[fid1])
            entry2 = random.choice(finger_to_entries[fid2])
            pairs.append((entry1, entry2, False))
    
    return pairs


def dense_reg_dataset(
    entries: List[FingerprintEntry],
    config: Dict,
    split: str = "train",
    shuffle: bool = True
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Generate batches for dense registration training.
    
    According to IFViT paper (Section 4.1):
    - 75,000 genuine pairs: same finger (original ↔ corrupted)
    - 25,000 imposter pairs: different fingers (GT "no match")
    - Total: 100,000 pairs
    
    Args:
        entries: List of FingerprintEntry
        config: DENSE_CONFIG dictionary
        split: "train", "val", or "test"
        shuffle: Whether to shuffle
        
    Yields:
        Batch dictionary with:
        - img1: [B, H, W, 1] original images
        - img2: [B, H, W, 1] corrupted/different images
        - matches: [B, K, 4] ground-truth correspondences
        - valid_mask: [B, K] validity mask (all False for imposter pairs)
    """
    # Filter by split
    split_entries = [e for e in entries if e.split == split]
    
    # Generate pairs (genuine + imposter)
    imposter_ratio = config.get("imposter_ratio", 0.25)  # 25% imposter
    pairs = _generate_dense_reg_pairs(split_entries, imposter_ratio=imposter_ratio)
    
    if shuffle:
        random.Random(42).shuffle(pairs)
    
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    num_points = config.get("num_correspondence_points", 1000)
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        
        batch_img1 = []
        batch_img2 = []
        batch_matches = []
        batch_valid = []
        
        rng = jax.random.PRNGKey(42 + i // batch_size)
        
        for entry1, entry2, is_genuine in batch_pairs:
            try:
                # Load first image
                img1 = load_image(entry1.path)
                if img1 is None:
                    warnings.warn(f"Skipping corrupt image: {entry1.path}")
                    continue
                img1 = normalize_image(img1, target_size=(image_size, image_size))
                
                if is_genuine:
                    # Genuine pair: original ↔ corrupted (same finger)
                    rng, corrupt_rng = jax.random.split(rng)
                    corrupted, transform = random_corrupt_fingerprint(
                            (img1 * 255).astype(np.float32), 
                        corrupt_rng
                    )
                    corrupted = corrupted / 255.0
                    
                    # Generate GT correspondences
                    matches, valid = generate_gt_correspondences(
                            img1, corrupted, transform, num_points=num_points
                    )
                    
                    batch_img1.append(img1[..., None])
                    batch_img2.append(corrupted[..., None])
                    batch_matches.append(matches)
                    batch_valid.append(valid)
                else:
                    # Imposter pair: different fingers (GT "no match")
                    img2 = load_image(entry2.path)
                    if img2 is None:
                        warnings.warn(f"Skipping corrupt image: {entry2.path}")
                        continue
                    img2 = normalize_image(img2, target_size=(image_size, image_size))
                    
                    # No valid correspondences (all zeros, all invalid)
                    matches = np.zeros((num_points, 4), dtype=np.float32)
                    valid = np.zeros(num_points, dtype=bool)
                    
                    batch_img1.append(img1[..., None])
                    batch_img2.append(img2[..., None])
                    batch_matches.append(matches)
                    batch_valid.append(valid)
            except Exception as e:
                warnings.warn(f"Skipping pair due to error: {entry1.path} / {entry2.path} - {e}")
                continue
        
        # Skip empty batches (all pairs failed to load)
        if len(batch_img1) == 0:
            continue
        
        # Pad to same length
        max_matches = max(len(m) for m in batch_matches)
        padded_matches = []
        padded_valid = []
        
        for m, v in zip(batch_matches, batch_valid):
            if len(m) < max_matches:
                pad = np.zeros((max_matches - len(m), 4))
                padded_matches.append(np.vstack([m, pad]))
                padded_valid.append(np.concatenate([v, np.zeros(max_matches - len(v), dtype=bool)]))
            else:
                padded_matches.append(m)
                padded_valid.append(v)
        
        yield {
            'img1': jnp.array(np.stack(batch_img1)),
            'img2': jnp.array(np.stack(batch_img2)),
            'matches': jnp.array(padded_matches),
            'valid_mask': jnp.array(padded_valid)
        }


def matcher_dataset(
    entries: List[FingerprintEntry],
    config: Dict,
    split: str = "train",
    shuffle: bool = True,
    num_classes: int = None,
    preprocessed_dir: Optional[str] = None
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Generate batches for matcher training.
    
    Args:
        entries: List of FingerprintEntry
        config: MATCH_CONFIG dictionary
        split: "train", "val", or "test"
        shuffle: Whether to shuffle
        num_classes: Total number of classes (for ArcFace)
        preprocessed_dir: Optional directory with preprocessed .npz files
        
    Yields:
        Batch dictionary with:
        - img1, img2: [B, H, W, 1] images
        - roi1, roi2: [B, 90, 90, 1] ROIs
        - label_pair: [B] 1 for genuine, -1 for imposter
        - class_id1, class_id2: [B] global finger IDs
    """
    # Filter by split
    split_entries = [e for e in entries if e.split == split]
    
    # Generate pairs
    pairs = _generate_pairs(split_entries, config.get("imposter_ratio", 0.25))
    
    if shuffle:
        random.Random(42).shuffle(pairs)
    
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    roi_size = config["roi_size"]
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        
        batch_img1 = []
        batch_img2 = []
        batch_roi1 = []
        batch_roi2 = []
        batch_global1 = []
        batch_global2 = []
        batch_labels = []
        batch_class1 = []
        batch_class2 = []
        
        for entry1, entry2, is_genuine in batch_pairs:
            try:
                if preprocessed_dir:
                    # Load from preprocessed .npz
                    pair_idx = hash((entry1.path, entry2.path)) % 1000000
                    npz_path = Path(preprocessed_dir) / f"pair_{pair_idx:06d}.npz"
                    
                    if npz_path.exists():
                        data = np.load(npz_path)
                        batch_global1.append(data['global1'])
                        batch_global2.append(data['global2'])
                        batch_roi1.append(data['roi1'])
                        batch_roi2.append(data['roi2'])
                    else:
                        # Fallback to loading raw images
                        img1 = load_image(entry1.path)
                        img2 = load_image(entry2.path)
                        if img1 is None or img2 is None:
                            warnings.warn(f"Skipping corrupt image pair: {entry1.path} / {entry2.path}")
                            continue
                        img1 = normalize_image(img1, target_size=(image_size, image_size))
                        img2 = normalize_image(img2, target_size=(image_size, image_size))
                        batch_img1.append(img1[..., None])
                        batch_img2.append(img2[..., None])
                        # Placeholder ROIs
                        batch_roi1.append(np.zeros((roi_size, roi_size, 1), dtype=np.float32))
                        batch_roi2.append(np.zeros((roi_size, roi_size, 1), dtype=np.float32))
                        batch_global1.append(img1[..., None])
                        batch_global2.append(img2[..., None])
                else:
                    # Load raw images (will need FingerNet processing in training loop)
                    img1 = load_image(entry1.path)
                    img2 = load_image(entry2.path)
                    if img1 is None or img2 is None:
                        warnings.warn(f"Skipping corrupt image pair: {entry1.path} / {entry2.path}")
                        continue
                    img1 = normalize_image(img1, target_size=(image_size, image_size))
                    img2 = normalize_image(img2, target_size=(image_size, image_size))
                    batch_img1.append(img1[..., None])
                    batch_img2.append(img2[..., None])
                    # Placeholder - will be filled by FingerNet + ROI extraction
                    batch_roi1.append(np.zeros((roi_size, roi_size, 1), dtype=np.float32))
                    batch_roi2.append(np.zeros((roi_size, roi_size, 1), dtype=np.float32))
                    batch_global1.append(img1[..., None])
                    batch_global2.append(img2[..., None])
                
                batch_labels.append(1 if is_genuine else -1)
                batch_class1.append(entry1.finger_global_id)
                batch_class2.append(entry2.finger_global_id)
            except Exception as e:
                warnings.warn(f"Skipping pair due to error: {entry1.path} / {entry2.path} - {e}")
                continue
        
        # Skip empty batches (all pairs failed to load)
        if len(batch_labels) == 0:
            continue
        
        yield {
            'img1': jnp.array(np.stack(batch_img1)) if batch_img1 else None,
            'img2': jnp.array(np.stack(batch_img2)) if batch_img2 else None,
            'roi1': jnp.array(np.stack(batch_roi1)),
            'roi2': jnp.array(np.stack(batch_roi2)),
            'global1': jnp.array(np.stack(batch_global1)),
            'global2': jnp.array(np.stack(batch_global2)),
            'label_pair': jnp.array(batch_labels),
            'class_id1': jnp.array(batch_class1),
            'class_id2': jnp.array(batch_class2)
        }


def preprocess_batch(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Preprocess batch for training (normalization, etc.).
    
    Args:
        batch: Raw batch dictionary
        
    Returns:
        Preprocessed batch
    """
    # Images should already be in [0, 1] range
    # Additional preprocessing can be added here
    return batch


def compute_overlap_and_rois(
    img1: np.ndarray,
    img2: np.ndarray,
    roi_size: int = 90
) -> Dict[str, np.ndarray]:
    """
    Compute overlapped region and extract ROI patches.
    
    NOTE: This is a placeholder. In practice, use FingerNet enhancement
    and the functions in ut.enhance_and_roi for proper implementation.
    
    Args:
        img1: First fingerprint image (H, W)
        img2: Second fingerprint image (H, W)
        roi_size: Size of ROI patch (default 90)
        
    Returns:
        Dictionary with overlap_mask, roi1, roi2, etc.
    """
    # Placeholder implementation
    H, W = img1.shape[:2]
    
    # Simple center crop for ROI
    cy, cx = H // 2, W // 2
    half = roi_size // 2
    roi1 = img1[cy-half:cy+half, cx-half:cx+half]
    roi2 = img2[cy-half:cy+half, cx-half:cx+half]
    
    # Pad if needed
    if roi1.shape[0] < roi_size or roi1.shape[1] < roi_size:
        roi1 = cv2.resize(roi1, (roi_size, roi_size))
        roi2 = cv2.resize(roi2, (roi_size, roi_size))
    
    return {
        "overlap_mask": np.ones((H, W), dtype=np.uint8),
        "roi1": roi1[..., None],
        "roi2": roi2[..., None],
        "overlap_enhanced1": img1[..., None],
        "overlap_enhanced2": img2[..., None],
    }

