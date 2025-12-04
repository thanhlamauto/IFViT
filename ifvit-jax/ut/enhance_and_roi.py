"""
FingerNet-based enhancement, overlapped-region extraction, and ROI building
for IFViT Module 2.

This module wires:
- FingerNet (or any enhancement model) as a callable on single images.
- Effective ridge-region masks via Sobel + box filter + threshold.
- Overlapped regions between two enhanced fingerprints.
- 90x90 ROIs on the aligned original fingerprints for the local branch.
"""

from typing import Tuple

import cv2
import numpy as np


class FingerNetEnhancer:
    """
    Thin wrapper around a FingerNet PyTorch model.

    You should instantiate this with a real model:

        model = ...  # torch.nn.Module with weights loaded
        enhancer = FingerNetEnhancer(model)

    Notes:
        - Input images are automatically converted to float32.
        - If max(img) > 1.0, the image is scaled to [0, 1] by dividing by 255.
          This matches the original FingerNet pipeline, which uses img/255.0.
    """

    def __init__(self, model):
        self.model = model  # torch.nn.Module or compatible callable

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance a single fingerprint image.

        Args:
            img: [H, W] or [H, W, 1], uint8 or float32.
                 If values are in [0, 255], they will be scaled to [0, 1].

        Returns:
            enh: np.float32 [H, W] enhanced image
        """
        import torch

        if img.ndim == 3:
            img = img[..., 0]

        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        with torch.no_grad():
            y = self.model(x)

        enh = y.squeeze().detach().cpu().numpy().astype(np.float32)
        return enh


def compute_effective_mask(
    enh_img: np.ndarray,
    ksize: int = 25,
    thresh_ratio: float = 0.15,
) -> np.ndarray:
    """
    Compute an effective ridge-region mask from an enhanced fingerprint.

    Implements Sobel gradient magnitude -> box filter -> threshold T2, then
    morphological clean-up and largest connected-component selection.

    Args:
        enh_img: [H, W] float32 enhanced image
        ksize: kernel size for box filter
        thresh_ratio: T2 = thresh_ratio * max(integral)

    Returns:
        mask: [H, W] uint8, values {0, 1}
    """
    enh = enh_img.astype(np.float32)

    gx = cv2.Sobel(enh, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enh, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx ** 2 + gy ** 2)

    integral = cv2.blur(g, (ksize, ksize))
    t2 = float(thresh_ratio) * float(np.max(integral) + 1e-8)

    mask = (integral >= t2).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = 1 + np.argmax(areas)
    final_mask = (labels == largest_id).astype(np.uint8)
    return final_mask


def extract_overlapped_region(
    enh1: np.ndarray,
    enh2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract overlapped ridge region from two enhanced fingerprints.

    Args:
        enh1, enh2: [H, W] float32 enhanced images

    Returns:
        I_oe1, I_oe2: [H, W, 1] float32 overlapped regions (background = 0)
        overlap_mask: [H, W] uint8 mask (0/1)
    """
    mask1 = compute_effective_mask(enh1)
    mask2 = compute_effective_mask(enh2)

    overlap = ((mask1 > 0) & (mask2 > 0)).astype(np.uint8)

    I_oe1 = np.where(overlap[..., None].astype(bool), enh1[..., None], 0.0).astype(np.float32)
    I_oe2 = np.where(overlap[..., None].astype(bool), enh2[..., None], 0.0).astype(np.float32)

    return I_oe1, I_oe2, overlap


def extract_roi_from_original(
    orig1: np.ndarray,
    orig2: np.ndarray,
    overlap_mask: np.ndarray,
    roi_size: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 90x90 ROIs from aligned original fingerprints based on overlapped mask.

    Args:
        orig1, orig2: [H, W] or [H, W, 1] aligned original images
        overlap_mask: [H, W] uint8, 0 or 1
        roi_size: patch size (default 90)

    Returns:
        roi1, roi2: [roi_size, roi_size, 1] float32 ROIs
    """
    H, W = overlap_mask.shape
    ys, xs = np.where(overlap_mask > 0)

    if len(xs) == 0:
        cx, cy = W // 2, H // 2
    else:
        cx = int(xs.mean())
        cy = int(ys.mean())

    half = roi_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    pad_left = max(0, half - cx)
    pad_top = max(0, half - cy)
    pad_right = max(0, cx + half - W)
    pad_bottom = max(0, cy + half - H)

    def _crop_and_pad(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img2d = img[..., 0]
        else:
            img2d = img

        crop = img2d[y1:y2, x1:x2]
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        crop = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
        return crop[..., None].astype(np.float32)

    roi1 = _crop_and_pad(orig1)
    roi2 = _crop_and_pad(orig2)
    return roi1, roi2


def build_matcher_inputs(
    aligned1: np.ndarray,
    img2: np.ndarray,
    fingernet_enhancer: FingerNetEnhancer,
    roi_size: int = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level helper to prepare MatcherModel inputs.

    Args:
        aligned1: aligned version of original image 1 [H, W] or [H, W, 1]
        img2: original image 2 in the target frame [H, W] or [H, W, 1]
        fingernet_enhancer: FingerNetEnhancer instance
        roi_size: ROI patch size (default 90)

    Returns:
        global1, global2: overlapped enhanced regions [H, W, 1]
        roi1, roi2: local ROIs [roi_size, roi_size, 1]
    """
    if aligned1.ndim == 3:
        base1 = aligned1[..., 0]
    else:
        base1 = aligned1

    if img2.ndim == 3:
        base2 = img2[..., 0]
    else:
        base2 = img2

    enh1 = fingernet_enhancer(base1)
    enh2 = fingernet_enhancer(base2)

    I_oe1, I_oe2, overlap_mask = extract_overlapped_region(enh1, enh2)
    roi1, roi2 = extract_roi_from_original(base1, base2, overlap_mask, roi_size=roi_size)

    return I_oe1, I_oe2, roi1, roi2


