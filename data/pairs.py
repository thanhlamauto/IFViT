"""
Pair generation for training.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import random
from .base import FingerprintEntry


@dataclass
class DenseRegPair:
    """Pair for dense registration training."""
    img1: np.ndarray  # Original image [H, W]
    img2: np.ndarray  # Corrupted/transformed image [H, W]
    transform_matrix: np.ndarray  # 3x3 transformation matrix
    matches: Optional[np.ndarray] = None  # Ground-truth matches [N, 4] (x1, y1, x2, y2)
    valid_mask: Optional[np.ndarray] = None  # [N] boolean mask


@dataclass
class MatcherPair:
    """Pair for matcher training."""
    img1: np.ndarray  # [H, W]
    img2: np.ndarray  # [H, W]
    roi1: np.ndarray  # [90, 90, 1] local ROI
    roi2: np.ndarray  # [90, 90, 1] local ROI
    global1: np.ndarray  # [H, W, 1] overlapped enhanced region
    global2: np.ndarray  # [H, W, 1] overlapped enhanced region
    label_pair: int  # 1 for genuine, -1 for imposter
    class_id1: int  # Global finger ID for ArcFace
    class_id2: int  # Global finger ID for ArcFace


def _generate_pairs(
    entries: List[FingerprintEntry],
    imposter_ratio: float = 0.25
) -> List[Tuple[FingerprintEntry, FingerprintEntry, bool]]:
    """
    Generate genuine and imposter pairs from entries.
    
    Args:
        entries: List of FingerprintEntry
        imposter_ratio: Ratio of imposter pairs to genuine pairs
        
    Returns:
        List of (entry1, entry2, is_genuine) tuples
    """
    pairs = []
    
    # Group by finger ID
    finger_to_entries = {}
    for entry in entries:
        fid = entry.finger_global_id
        if fid not in finger_to_entries:
            finger_to_entries[fid] = []
        finger_to_entries[fid].append(entry)
    
    # Genuine pairs (same finger, different impressions)
    for fid, finger_entries in finger_to_entries.items():
        for i in range(len(finger_entries)):
            for j in range(i + 1, len(finger_entries)):
                pairs.append((finger_entries[i], finger_entries[j], True))
    
    # Imposter pairs (different fingers)
    # Paper: use only first impression for imposters
    first_impressions = []
    for fid, finger_entries in finger_to_entries.items():
        # Get first impression (lowest impression_id)
        first_imp = min(finger_entries, key=lambda e: e.impression_id)
        first_impressions.append(first_imp)
    
    n_imposter = int(len(pairs) * imposter_ratio)
    
    for _ in range(n_imposter):
        i, j = random.sample(range(len(first_impressions)), 2)
        pairs.append((first_impressions[i], first_impressions[j], False))
    
    return pairs

