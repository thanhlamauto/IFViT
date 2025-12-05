"""
Augmentation and synthetic distortion for Module 1 training.

According to IFViT paper (Section 4.1):
- 3 noise models: Sensor noise (Perlin noise), Dryness (erosion), Over-pressurization (dilation)
- Random rotation ±60° applied to corrupted image
- Module 1 uses ONLY original + corrupted pairs (NO FingerNet enhancement)
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict
import cv2


def generate_perlin_noise(shape: Tuple[int, int], scale: float = 10.0, rng_key: jax.random.PRNGKey = None) -> np.ndarray:
    """
    Generate Perlin-like noise for sensor noise simulation.
    
    Simplified Perlin noise using multiple octaves of random noise.
    This approximates sensor noise as described in IFViT paper.
    
    Args:
        shape: (H, W) output shape
        scale: Noise scale (higher = smoother noise)
        rng_key: JAX random key
        
    Returns:
        Noise array (H, W) in range [-1, 1]
    """
    H, W = shape
    noise = np.zeros((H, W), dtype=np.float32)
    
    # Multi-octave noise (simplified Perlin)
    octaves = [1, 2, 4, 8]
    for octave in octaves:
        h, w = max(1, H // octave), max(1, W // octave)
        if rng_key is not None:
            rng_key, subkey = jax.random.split(rng_key)
            octave_noise = jax.random.normal(subkey, (h, w))
        else:
            octave_noise = np.random.randn(h, w).astype(np.float32)
        
        # Upsample to original size
        if h != H or w != W:
            octave_noise = cv2.resize(octave_noise, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Weight by octave
        noise += octave_noise / (octave * scale)
    
    # Normalize to [-1, 1]
    noise = noise / (np.abs(noise).max() + 1e-8)
    return noise


def random_corrupt_fingerprint(
    img: np.ndarray, 
    rng_key: jax.random.PRNGKey,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random corruptions and transformations to fingerprint image.
    
    According to IFViT paper (Section 4.1):
    - 3 noise models: Sensor noise (Perlin noise), Dryness (erosion), Over-pressurization (dilation)
    - Random rotation ±60° applied AFTER corruption
    
    Args:
        img: Input fingerprint image (H, W) float32 [0, 255]
        rng_key: JAX random key
        config: Augmentation config dict (from AUGMENT_CONFIG)
        
    Returns:
        corrupted_img: Transformed/corrupted image (H, W) float32 [0, 255]
        transform_matrix: 3x3 affine transformation matrix (rotation only, applied last)
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
    # Split into 6 keys: 1 for rng_key update + 5 for operations
    rng_key, *subkeys = jax.random.split(rng_key, 6)
    
    # Start with original image
    corrupted = img.copy()
    corrupted_uint8 = corrupted.astype(np.uint8)
    
    # Apply ONE of 3 noise models (as per paper):
    # Paper: "Áp dụng 3 loại noise / biến dạng cho mỗi ảnh"
    # Interpretation: Each image gets ONE of the 3 noise types
    noise_type = jax.random.randint(subkeys[0], shape=(), minval=0, maxval=3)
    
    if noise_type == 0:
        # 1. Sensor noise → Perlin noise
        perlin_noise = generate_perlin_noise((H, W), scale=10.0, rng_key=subkeys[1])
        noise_scale = config.get("perlin_noise_scale", 0.1) * 255
        corrupted = corrupted + perlin_noise * noise_scale
        corrupted_uint8 = np.clip(corrupted, 0, 255).astype(np.uint8)
    elif noise_type == 1:
        # 2. Dryness → Erosion
        kernel = np.ones(config.get("morph_kernel_size", (3, 3)), np.uint8)
        corrupted_uint8 = cv2.erode(corrupted_uint8, kernel, iterations=1)
    else:  # noise_type == 2
        # 3. Over-pressurization → Dilation
        kernel = np.ones(config.get("morph_kernel_size", (3, 3)), np.uint8)
        corrupted_uint8 = cv2.dilate(corrupted_uint8, kernel, iterations=1)
    
    corrupted = corrupted_uint8.astype(np.float32)
    
    # Apply rotation ±60° (as per paper, applied after corruption)
    angle = jax.random.uniform(
        subkeys[2], 
        minval=config["rotation_range"][0], 
        maxval=config["rotation_range"][1]
    )
    center = (W / 2, H / 2)
    M_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    
    # Apply rotation transformation
    corrupted = cv2.warpAffine(corrupted, M_rot, (W, H), flags=cv2.INTER_LINEAR)
    
    # Clip to valid range
    corrupted = np.clip(corrupted, 0, 255)
    
    # Convert rotation matrix to 3x3 homogeneous transform
    transform_matrix = np.vstack([M_rot, [0, 0, 1]]).astype(np.float32)
    
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

