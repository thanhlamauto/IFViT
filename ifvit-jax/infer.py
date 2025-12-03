"""
Inference script for fingerprint alignment and verification.

Provides functions to:
1. Load trained models
2. Align fingerprint pairs
3. Extract embeddings
4. Compute verification scores
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List, Optional
import cv2
from pathlib import Path

from config import INFER_CONFIG, DENSE_CONFIG, MATCH_CONFIG
from models import DenseRegModel, MatcherModel
from losses import compute_matching_score
from data import compute_overlap_and_rois, normalize_image
from utils import load_checkpoint


# ============================================================================
# Model Loading
# ============================================================================

def load_dense_model(checkpoint_path: str, config: Dict = None) -> Tuple[callable, Dict]:
    """
    Load trained DenseRegModel from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional config dict (defaults to DENSE_CONFIG)
        
    Returns:
        apply_fn: Model apply function
        params: Model parameters
    """
    if config is None:
        config = DENSE_CONFIG
    
    # Load checkpoint
    state_dict, metadata = load_checkpoint(checkpoint_path)
    params = state_dict['params']
    
    # Create model
    model = DenseRegModel(
        image_size=config["image_size"],
        num_transformer_layers=config["transformer_layers"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=config["dropout_rate"]
    )
    
    print(f"✓ Loaded DenseRegModel from {checkpoint_path}")
    
    return model.apply, params


def load_matcher_model(checkpoint_path: str, config: Dict = None) -> Tuple[callable, Dict]:
    """
    Load trained MatcherModel from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional config dict (defaults to MATCH_CONFIG)
        
    Returns:
        apply_fn: Model apply function
        params: Model parameters
    """
    if config is None:
        config = MATCH_CONFIG
    
    # Load checkpoint
    state_dict, metadata = load_checkpoint(checkpoint_path)
    params = state_dict['params']
    
    # Create model
    model = MatcherModel(
        image_size=config["image_size"],
        roi_size=config["roi_size"],
        num_transformer_layers=config["transformer_layers"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=config["dropout_rate"],
        embedding_dim=config["embedding_dim"]
    )
    
    print(f"✓ Loaded MatcherModel from {checkpoint_path}")
    
    return model.apply, params


# ============================================================================
# Fingerprint Alignment
# ============================================================================

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by given angle.
    
    Args:
        img: Input image (H, W) or (H, W, C)
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return rotated


def count_matches(P: jnp.ndarray, threshold: float = 0.1) -> int:
    """
    Count number of confident matches from probability matrix.
    
    Args:
        P: (N, N) matching probability matrix
        threshold: Confidence threshold
        
    Returns:
        Number of matches above threshold
    """
    max_probs = jnp.max(P, axis=1)
    num_matches = jnp.sum(max_probs > threshold)
    return int(num_matches)


def align_fingerprints(
    apply_fn: callable,
    params: Dict,
    img1: np.ndarray,
    img2: np.ndarray,
    rotation_angles: List[float] = None,
    min_matches: int = 10
) -> Tuple[np.ndarray, float, int, jnp.ndarray]:
    """
    Align two fingerprint images by trying different rotations.
    
    Args:
        apply_fn: DenseRegModel apply function
        params: Model parameters
        img1: First fingerprint image (H, W)
        img2: Second fingerprint image (H, W)
        rotation_angles: List of angles to try (default: [-60, -30, 0, 30, 60])
        min_matches: Minimum matches to consider alignment valid
        
    Returns:
        aligned_img2: Best aligned version of img2
        best_angle: Best rotation angle
        num_matches: Number of matches at best angle
        P: Matching probability matrix at best angle
    """
    if rotation_angles is None:
        rotation_angles = INFER_CONFIG["rotation_angles"]
    
    # Normalize images
    img1_norm = normalize_image(img1)
    
    # Add batch and channel dimensions
    img1_batch = jnp.expand_dims(jnp.expand_dims(img1_norm, 0), -1)  # (1, H, W, 1)
    
    best_angle = 0.0
    best_matches = 0
    best_P = None
    best_img2_rotated = img2
    
    for angle in rotation_angles:
        # Rotate img2
        img2_rotated = rotate_image(img2, angle)
        img2_norm = normalize_image(img2_rotated)
        img2_batch = jnp.expand_dims(jnp.expand_dims(img2_norm, 0), -1)
        
        # Forward pass
        P, matches, feat1, feat2 = apply_fn(
            {'params': params},
            img1_batch,
            img2_batch,
            train=False
        )
        
        # Count matches
        num_matches = count_matches(P[0])
        
        if num_matches > best_matches:
            best_matches = num_matches
            best_angle = angle
            best_P = P[0]
            best_img2_rotated = img2_rotated
    
    print(f"  Best alignment: angle={best_angle}°, matches={best_matches}")
    
    if best_matches < min_matches:
        print(f"  ⚠ Warning: Only {best_matches} matches found (min: {min_matches})")
    
    return best_img2_rotated, best_angle, best_matches, best_P


# ============================================================================
# Embedding Extraction
# ============================================================================

def extract_embeddings(
    apply_fn: callable,
    params: Dict,
    img1: np.ndarray,
    img2: np.ndarray,
    roi_size: int = 90
) -> Dict[str, jnp.ndarray]:
    """
    Extract global and local embeddings from fingerprint pair.
    
    Args:
        apply_fn: MatcherModel apply function
        params: Model parameters
        img1: First fingerprint image (H, W)
        img2: Second fingerprint image (H, W)
        roi_size: Size of ROI patches
        
    Returns:
        Dictionary with:
        - emb_g1, emb_g2: Global embeddings
        - emb_l1, emb_l2: Local embeddings
    """
    # Compute ROIs (if data.py is not implemented, use simple center crop)
    try:
        overlap_data = compute_overlap_and_rois(img1, img2, roi_size)
        roi1 = overlap_data['roi1']
        roi2 = overlap_data['roi2']
    except NotImplementedError:
        # Fallback: center crop
        h, w = img1.shape[:2]
        half_roi = roi_size // 2
        cy, cx = h // 2, w // 2
        
        roi1 = img1[cy-half_roi:cy+half_roi, cx-half_roi:cx+half_roi]
        roi2 = img2[cy-half_roi:cy+half_roi, cx-half_roi:cx+half_roi]
        
        # Pad if needed
        if roi1.shape[0] < roi_size or roi1.shape[1] < roi_size:
            pad_h = max(0, roi_size - roi1.shape[0])
            pad_w = max(0, roi_size - roi1.shape[1])
            roi1 = np.pad(roi1, ((0, pad_h), (0, pad_w)), mode='constant')
            roi2 = np.pad(roi2, ((0, pad_h), (0, pad_w)), mode='constant')
    
    # Normalize and add batch/channel dims
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    roi1_norm = normalize_image(roi1)
    roi2_norm = normalize_image(roi2)
    
    img1_batch = jnp.expand_dims(jnp.expand_dims(img1_norm, 0), -1)
    img2_batch = jnp.expand_dims(jnp.expand_dims(img2_norm, 0), -1)
    roi1_batch = jnp.expand_dims(jnp.expand_dims(roi1_norm, 0), -1)
    roi2_batch = jnp.expand_dims(jnp.expand_dims(roi2_norm, 0), -1)
    
    # Forward pass
    emb_g1, emb_g2, emb_l1, emb_l2, P, matches = apply_fn(
        {'params': params},
        img1_batch, img2_batch,
        roi1_batch, roi2_batch,
        train=False
    )
    
    return {
        'emb_g1': emb_g1[0],  # Remove batch dim
        'emb_g2': emb_g2[0],
        'emb_l1': emb_l1[0],
        'emb_l2': emb_l2[0]
    }


# ============================================================================
# Verification Pipeline
# ============================================================================

def verify_pair(
    dense_apply_fn: callable,
    dense_params: Dict,
    matcher_apply_fn: callable,
    matcher_params: Dict,
    img1: np.ndarray,
    img2: np.ndarray,
    do_alignment: bool = True,
    config: Dict = None
) -> Dict[str, any]:
    """
    Complete verification pipeline for a fingerprint pair.
    
    Args:
        dense_apply_fn: DenseRegModel apply function
        dense_params: DenseRegModel parameters
        matcher_apply_fn: MatcherModel apply function
        matcher_params: MatcherModel parameters
        img1: First fingerprint image (H, W)
        img2: Second fingerprint image (H, W)
        do_alignment: Whether to perform alignment
        config: Optional config dict (defaults to INFER_CONFIG)
        
    Returns:
        Dictionary with:
        - aligned_img2: Aligned version of img2
        - rotation_angle: Best rotation angle
        - num_matches: Number of dense matches
        - score_global: Global similarity score
        - score_local: Local similarity score
        - score_fused: Fused similarity score
        - decision: Match/non-match decision
    """
    if config is None:
        config = INFER_CONFIG
    
    result = {}
    
    # Step 1: Alignment (optional)
    if do_alignment:
        print("Aligning fingerprints...")
        aligned_img2, angle, num_matches, P = align_fingerprints(
            dense_apply_fn,
            dense_params,
            img1,
            img2,
            rotation_angles=config.get("rotation_angles"),
            min_matches=config.get("min_matches", 10)
        )
        result['aligned_img2'] = aligned_img2
        result['rotation_angle'] = angle
        result['num_matches'] = num_matches
    else:
        aligned_img2 = img2
        result['aligned_img2'] = img2
        result['rotation_angle'] = 0.0
        result['num_matches'] = None
    
    # Step 2: Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(
        matcher_apply_fn,
        matcher_params,
        img1,
        aligned_img2,
        roi_size=MATCH_CONFIG.get("roi_size", 90)
    )
    
    # Step 3: Compute scores
    print("Computing similarity scores...")
    scores = compute_matching_score(
        emb_g1=embeddings['emb_g1'],
        emb_g2=embeddings['emb_g2'],
        emb_l1=embeddings['emb_l1'],
        emb_l2=embeddings['emb_l2'],
        alpha_global=config.get("alpha_global", 0.6),
        alpha_local=config.get("alpha_local", 0.4)
    )
    
    result['score_global'] = float(scores['global'])
    result['score_local'] = float(scores['local'])
    result['score_fused'] = float(scores['fused'])
    
    # Step 4: Decision
    threshold = config.get("threshold", 0.5)
    result['decision'] = 'Match' if scores['fused'] > threshold else 'Non-match'
    result['threshold'] = threshold
    
    # Print summary
    print("\n" + "="*60)
    print("Verification Result")
    print("="*60)
    print(f"  Global score:  {result['score_global']:.4f}")
    print(f"  Local score:   {result['score_local']:.4f}")
    print(f"  Fused score:   {result['score_fused']:.4f}")
    print(f"  Threshold:     {threshold:.4f}")
    print(f"  Decision:      {result['decision']}")
    print("="*60 + "\n")
    
    return result


# ============================================================================
# Batch Verification
# ============================================================================

def verify_batch(
    dense_model: Tuple[callable, Dict],
    matcher_model: Tuple[callable, Dict],
    pairs: List[Tuple[str, str, int]],
    do_alignment: bool = True
) -> List[Dict]:
    """
    Verify a batch of fingerprint pairs.
    
    Args:
        dense_model: (apply_fn, params) for DenseRegModel
        matcher_model: (apply_fn, params) for MatcherModel
        pairs: List of (path1, path2, label) tuples
        do_alignment: Whether to perform alignment
        
    Returns:
        List of verification results
    """
    dense_apply_fn, dense_params = dense_model
    matcher_apply_fn, matcher_params = matcher_model
    
    results = []
    
    for i, (path1, path2, label) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Verifying pair:")
        print(f"  Image 1: {path1}")
        print(f"  Image 2: {path2}")
        print(f"  GT Label: {'Genuine' if label == 1 else 'Imposter'}")
        
        # Load images
        try:
            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                print(f"  ✗ Failed to load images")
                continue
            
            # Verify
            result = verify_pair(
                dense_apply_fn, dense_params,
                matcher_apply_fn, matcher_params,
                img1, img2,
                do_alignment=do_alignment
            )
            
            result['path1'] = path1
            result['path2'] = path2
            result['gt_label'] = label
            
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fingerprint Verification")
    parser.add_argument('--img1', type=str, required=True, help='Path to first fingerprint')
    parser.add_argument('--img2', type=str, required=True, help='Path to second fingerprint')
    parser.add_argument('--dense_ckpt', type=str, help='DenseRegModel checkpoint')
    parser.add_argument('--matcher_ckpt', type=str, help='MatcherModel checkpoint')
    parser.add_argument('--no_alignment', action='store_true', help='Skip alignment')
    
    args = parser.parse_args()
    
    # Use default checkpoints if not specified
    dense_ckpt = args.dense_ckpt or INFER_CONFIG["dense_model_ckpt"]
    matcher_ckpt = args.matcher_ckpt or INFER_CONFIG["matcher_model_ckpt"]
    
    # Check if checkpoints exist
    if not Path(dense_ckpt).exists():
        print(f"✗ DenseRegModel checkpoint not found: {dense_ckpt}")
        return
    
    if not Path(matcher_ckpt).exists():
        print(f"✗ MatcherModel checkpoint not found: {matcher_ckpt}")
        return
    
    # Load models
    print("Loading models...")
    dense_apply_fn, dense_params = load_dense_model(dense_ckpt)
    matcher_apply_fn, matcher_params = load_matcher_model(matcher_ckpt)
    
    # Load images
    print(f"\nLoading images...")
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None:
        print(f"✗ Failed to load image: {args.img1}")
        return
    if img2 is None:
        print(f"✗ Failed to load image: {args.img2}")
        return
    
    print(f"  Image 1: {img1.shape}")
    print(f"  Image 2: {img2.shape}")
    
    # Verify
    result = verify_pair(
        dense_apply_fn, dense_params,
        matcher_apply_fn, matcher_params,
        img1, img2,
        do_alignment=not args.no_alignment
    )


if __name__ == '__main__':
    main()
