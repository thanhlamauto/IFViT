"""
Alignment utilities for IFViT Module 1.

Use dense matches from the DenseRegModel to estimate a homography
between two fingerprints and warp image 1 into the frame of image 2.

This roughly corresponds to the “rotation search + RANSAC + homography”
step in Fig. 2 of the paper.
"""

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def _flatten_index_to_xy(flat_idx: np.ndarray, h_feat: int, w_feat: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flattened index to (x, y) on a feature map of size (h_feat, w_feat)."""
    y = flat_idx // w_feat
    x = flat_idx % w_feat
    return x, y


def _feat_to_img_coords(
    x: np.ndarray,
    y: np.ndarray,
    img_h: int,
    img_w: int,
    h_feat: int,
    w_feat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale feature-map coordinates back to original image coordinates (assumes uniform stride)."""
    scale_y = img_h / float(h_feat)
    scale_x = img_w / float(w_feat)
    return x * scale_x, y * scale_y


def matches_to_points(
    matches: np.ndarray,
    img_h: int,
    img_w: int,
    h_feat: int,
    w_feat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dense matches (flattened indices on feature maps) to image-space point coordinates.

    Args:
        matches: (K, 2) integer array, flattened indices (i, j) in [0, H_feat*W_feat)
        img_h, img_w: height/width of the original images
        h_feat, w_feat: height/width of the feature maps (DenseRegModel output)

    Returns:
        pts1, pts2: (K, 2) float32 arrays of (x, y) image coordinates.
    """
    if matches.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    i_idx = matches[:, 0]
    j_idx = matches[:, 1]

    x1_f, y1_f = _flatten_index_to_xy(i_idx, h_feat, w_feat)
    x2_f, y2_f = _flatten_index_to_xy(j_idx, h_feat, w_feat)

    x1, y1 = _feat_to_img_coords(x1_f, y1_f, img_h, img_w, h_feat, w_feat)
    x2, y2 = _feat_to_img_coords(x2_f, y2_f, img_h, img_w, h_feat, w_feat)

    pts1 = np.stack([x1, y1], axis=-1).astype(np.float32)
    pts2 = np.stack([x2, y2], axis=-1).astype(np.float32)
    return pts1, pts2


def estimate_homography_from_matches(
    matches: np.ndarray,
    img1_shape: Sequence[int],
    img2_shape: Sequence[int],
    h_feat: int,
    w_feat: int,
    min_matches: int = 30,
    ransac_thresh: float = 3.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate homography H (3x3) from dense matches using RANSAC.

    Args:
        matches: (K, 2) integer array of flattened feature-map indices
        img1_shape, img2_shape: shapes of original images (H, W) or (H, W, C)
        h_feat, w_feat: feature-map resolution
        min_matches: minimum matches required to attempt homography
        ransac_thresh: RANSAC reprojection threshold (in pixels)

    Returns:
        H: (3, 3) homography or None
        mask_inliers: (K, 1) RANSAC inlier mask or None
    """
    if matches.shape[0] < min_matches:
        return None, None

    img_h, img_w = img1_shape[0], img1_shape[1]
    pts1, pts2 = matches_to_points(matches, img_h, img_w, h_feat, w_feat)

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask


def warp_image(img: np.ndarray, H: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Warp an image with a homography.

    Args:
        img: [H, W] or [H, W, 1] array
        H: (3, 3) homography matrix
        out_shape: (out_h, out_w)

    Returns:
        warped: warped image with same number of channels as input.
    """
    out_h, out_w = out_shape
    if img.ndim == 2:
        return cv2.warpPerspective(img, H, (out_w, out_h))
    else:
        warped = cv2.warpPerspective(img[..., 0], H, (out_w, out_h))
        return warped[..., None]


def align_with_best_rotation(
    img1: np.ndarray,
    img2: np.ndarray,
    model_apply_fn,
    params,
    angles: Iterable[float] = (-60, -30, 0, 30, 60),
    image_size: int = 128,
    min_matches: int = 30,
    ransac_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float, Optional[np.ndarray]]:
    """
    Run DenseRegModel at multiple rotations and pick the transform
    with the best match count, then estimate H and warp img1 into img2.

    Args:
        img1, img2: original grayscale images as np.float32, [H, W] or [H, W, 1],
                    already resized to (image_size, image_size)
        model_apply_fn: callable like apply(params, img1_batch, img2_batch, train)
                        returning (P, matches, feat1, feat2) or similar
        params: parameters of the DenseRegModel
        angles: list of rotation angles in degrees to try for img1
        image_size: assumed input size of DenseRegModel
        min_matches: minimum matches for homography estimation
        ransac_thresh: RANSAC reprojection threshold

    Returns:
        aligned1: warped version of img1 (aligned to img2 frame)
        img2: unchanged img2
        best_H: best homography or None
        best_angle: angle that produced best_H (0 if none)
        best_matches: (K, 2) matches used for best_H, or None
    """
    if img1.ndim == 3:
        base1 = img1[..., 0]
    else:
        base1 = img1
    if img2.ndim == 3:
        base2 = img2[..., 0]
    else:
        base2 = img2

    h_feat = image_size // 16
    w_feat = image_size // 16

    best_score = -1
    best_H = None
    best_angle = 0.0
    best_matches = None

    center = (image_size / 2.0, image_size / 2.0)

    for a in angles:
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rot1 = cv2.warpAffine(base1, M, (image_size, image_size))
        rot1_in = rot1[..., None].astype(np.float32)[None, ...]
        img2_in = base2[..., None].astype(np.float32)[None, ...]

        # Expect model_apply_fn to return (P, matches, *_)
        out = model_apply_fn(params, rot1_in, img2_in, False)
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            _, matches_b = out[0], out[1]
        else:
            raise ValueError("model_apply_fn must return at least (P, matches, ...)")

        matches_b = np.array(matches_b[0])
        num_matches = matches_b.shape[0]
        if num_matches < min_matches:
            continue

        H, mask = estimate_homography_from_matches(
            matches_b,
            img1_shape=base1.shape,
            img2_shape=base2.shape,
            h_feat=h_feat,
            w_feat=w_feat,
            min_matches=min_matches,
            ransac_thresh=ransac_thresh,
        )
        if H is None:
            continue

        if num_matches > best_score:
            best_score = num_matches
            best_H = H
            best_angle = float(a)
            best_matches = matches_b

    if best_H is None:
        # Fallback: no alignment found
        aligned1 = img1
        return aligned1, img2, None, 0.0, None

    aligned1 = warp_image(base1, best_H, out_shape=base2.shape[:2])
    if img1.ndim == 3:
        aligned1 = aligned1[..., None]
    return aligned1.astype(np.float32), img2.astype(np.float32), best_H, best_angle, best_matches


