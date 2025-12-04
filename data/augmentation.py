"""
Augmentation and synthetic distortion for Module 1 training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict
import cv2


def random_corrupt_fingerprint(
    img: np.ndarray, 
    rng_key: jax.random.PRNGKey,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random corruptions and transformations to fingerprint image.
    
    Args:
        img: Input fingerprint image (H, W) float32 [0, 255]
        rng_key: JAX random key
        config: Augmentation config dict (from AUGMENT_CONFIG)
        
    Returns:
        corrupted_img: Transformed/corrupted image (H, W) float32
        transform_matrix: 3x3 affine transformation matrix
    """
    if config is None:
        import sys
        from pathlib import Path
        # Add ifvit-jax to path
        ifvit_jax_path = Path(__file__).parent.parent / "ifvit-jax"
        if str(ifvit_jax_path) not in sys.path:
            sys.path.insert(0, str(ifvit_jax_path))
        from config import AUGMENT_CONFIG
        config = AUGMENT_CONFIG
    
    H, W = img.shape
    rng_key, *subkeys = jax.random.split(rng_key, 6)
    
    # Rotation (±60°)
    angle = jax.random.uniform(
        subkeys[0], 
        minval=config["rotation_range"][0], 
        maxval=config["rotation_range"][1]
    )
    center = (W / 2, H / 2)
    M_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    
    # Translation (optional, small)
    tx = jax.random.uniform(subkeys[1], minval=-W*0.1, maxval=W*0.1)
    ty = jax.random.uniform(subkeys[2], minval=-H*0.1, maxval=H*0.1)
    M_trans = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    
    # Combine rotation + translation
    M = M_rot.copy()
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply transformation
    corrupted = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    
    # Add noise (Perlin-like or Gaussian)
    noise = jax.random.normal(subkeys[3], img.shape) * config.get("noise_std", 0.02) * 255
    corrupted = np.clip(corrupted + noise, 0, 255)
    
    # Morphological operations
    corrupted_uint8 = corrupted.astype(np.uint8)
    
    if jax.random.uniform(subkeys[4]) < config.get("erosion_prob", 0.3):
        kernel = np.ones(config.get("morph_kernel_size", (3, 3)), np.uint8)
        corrupted_uint8 = cv2.erode(corrupted_uint8, kernel, iterations=1)
    
    if jax.random.uniform(subkeys[5]) < config.get("dilation_prob", 0.3):
        kernel = np.ones(config.get("morph_kernel_size", (3, 3)), np.uint8)
        corrupted_uint8 = cv2.dilate(corrupted_uint8, kernel, iterations=1)
    
    corrupted = corrupted_uint8.astype(np.float32)
    
    # Convert M to 3x3 homogeneous transform
    transform_matrix = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    
    return corrupted, transform_matrix


def generate_gt_correspondences(
    img1: np.ndarray,
    img2: np.ndarray, 
    transform_matrix: np.ndarray,
    num_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ground-truth pixel correspondences given known transformation.
    
    Args:
        img1: Original image (H, W)
        img2: Transformed image (H, W)
        transform_matrix: 3x3 transformation matrix from img1 to img2
        num_points: Number of correspondence points to generate
        
    Returns:
        matches: (N, 4) array of [x1, y1, x2, y2] correspondences
        valid_mask: (N,) boolean mask of valid correspondences (within bounds)
    """
    H, W = img1.shape[:2]
    
    # Sample random points in img1
    y_coords = np.random.randint(0, H, size=num_points)
    x_coords = np.random.randint(0, W, size=num_points)
    
    # Transform to homogeneous coordinates
    pts1 = np.stack([x_coords, y_coords, np.ones(num_points)], axis=1)  # (N, 3)
    
    # Apply transformation
    pts2 = (transform_matrix @ pts1.T).T  # (N, 3)
    pts2 = pts2[:, :2] / (pts2[:, 2:3] + 1e-8)  # Convert back to 2D
    
    # Check valid bounds
    valid_mask = (
        (pts2[:, 0] >= 0) & (pts2[:, 0] < W) &
        (pts2[:, 1] >= 0) & (pts2[:, 1] < H)
    )
    
    # Combine into matches array
    matches = np.concatenate([pts1[:, :2], pts2], axis=1)  # (N, 4)
    
    return matches, valid_mask

