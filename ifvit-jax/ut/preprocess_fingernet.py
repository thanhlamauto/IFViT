"""
Offline preprocessing script for FingerNet enhancement + overlap + ROI.

Goal:
    - Run ONCE (CPU/GPU, Keras/TF) to prepare data for IFViT (JAX/TPU).
    - For each fingerprint pair (img1, img2):
        * Load and resize images to a fixed size.
        * Run FingerNet to get enhanced images.
        * Compute overlapped ridge region.
        * Extract 90x90 ROIs on aligned originals (here we assume img1,img2
          are already spatially aligned; alignment module is handled separately).
        * Save everything to disk as .npz for fast loading in JAX.

Usage example:

    python ifvit-jax/ut/preprocess_fingernet.py \\
        --pairs_csv /path/to/pairs.csv \\
        --finger_root FingerNet \\
        --output_dir /path/to/preprocessed \\
        --image_size 512 \\
        --roi_size 90

where `pairs.csv` has at least two columns (with header):

    img1,img2
    /abs/path/to/image1_1.bmp,/abs/path/to/image1_2.bmp
    /abs/path/to/image2_1.bmp,/abs/path/to/image2_2.bmp
    ...

The script will create one .npz file per row, named `pair_{index:06d}.npz`
containing:

    - img1_path, img2_path (strings)
    - enh1, enh2: enhanced images [H,W] float32
    - overlap_mask: [H,W] uint8 (0/1)
    - roi1, roi2: [roi_size,roi_size,1] float32

You can then build a JAX dataloader that reads these .npz files directly.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _add_fingernet_to_path(finger_root: str) -> None:
    """Ensure FingerNet/src is on sys.path so we can import get_main_net."""
    import sys

    src_path = os.path.join(finger_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)


def build_fingernet_model(
    finger_root: str,
    image_size: int,
):
    """
    Build the FingerNet Keras model in "deploy" mode with released weights.

    This uses the original Keras/TF1 implementation from `FingerNet/src`.
    """
    _add_fingernet_to_path(finger_root)

    # Delayed import so this module can still be imported without TF1 installed.
    from train_test_deploy import get_main_net  # type: ignore

    weights_path = os.path.join(
        finger_root, "models", "released_version", "Model.model"
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"FingerNet weights not found at: {weights_path}\n"
            "Make sure you have `FingerNet/models/released_version/Model.model`."
        )

    # Input shape (H,W,1); allow variable H,W by using None if you don't want resize.
    input_shape = (image_size, image_size, 1)
    model = get_main_net(input_shape=input_shape, weights_path=weights_path)
    return model


def load_and_resize_grayscale(path: str, image_size: int) -> np.ndarray:
    """
    Load grayscale image with OpenCV and resize to (image_size, image_size).

    Returns:
        img: float32 [H,W] in range [0,255]
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32)


def enhance_with_fingernet(model, img: np.ndarray) -> np.ndarray:
    """
    Run FingerNet to obtain enhanced fingerprint image.

    Args:
        model: Keras model from FingerNet (get_main_net with deploy=True).
        img: [H,W] float32 in range [0,255] or [0,1]

    Returns:
        enh: [H,W] float32 enhanced image (phase-based enhancement).
    """
    from keras import backend as K  # type: ignore

    # Normalize to [0,1] as in original code (train_test_deploy.py uses img/255.0).
    x = img.copy()
    if x.max() > 1.0:
        x = x / 255.0

    x = x[..., None]          # [H,W,1]
    x = x[None, ...]          # [1,H,W,1]

    # We only need enh_img_real from deploy mode outputs.
    outputs = model.predict(x, batch_size=1)
    # In deploy mode: [enh_img_real, ori_out_1, ori_out_2, seg_out, ...]
    enh_img_real = outputs[0]
    enh = enh_img_real[0, ..., 0]

    # Ensure float32 and release any TF graph/session resources if needed.
    return enh.astype(np.float32)


def compute_effective_mask(
    enh_img: np.ndarray,
    ksize: int = 25,
    thresh_ratio: float = 0.15,
) -> np.ndarray:
    """
    Same mask computation as in `enhance_and_roi.py`, duplicated here
    to keep this script self-contained w.r.t. dependencies.

    Returns:
        mask: [H,W] uint8, values {0,1}
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
    Overlapped ridge region between two enhanced images.

    Returns:
        I_oe1, I_oe2: [H,W,1] float32
        overlap_mask: [H,W] uint8 {0,1}
    """
    mask1 = compute_effective_mask(enh1)
    mask2 = compute_effective_mask(enh2)

    overlap = ((mask1 > 0) & (mask2 > 0)).astype(np.uint8)

    I_oe1 = np.where(overlap[..., None].astype(bool), enh1[..., None], 0.0).astype(
        np.float32
    )
    I_oe2 = np.where(overlap[..., None].astype(bool), enh2[..., None], 0.0).astype(
        np.float32
    )
    return I_oe1, I_oe2, overlap


def extract_roi_from_original(
    orig1: np.ndarray,
    orig2: np.ndarray,
    overlap_mask: np.ndarray,
    roi_size: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same ROI logic as `enhance_and_roi.extract_roi_from_original`.

    Args:
        orig1, orig2: [H,W] float32 original (assumed already aligned)
        overlap_mask: [H,W] uint8 {0,1}

    Returns:
        roi1, roi2: [roi_size,roi_size,1] float32
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
    pad_bottom = max(0, cx + half - W)

    def _crop_and_pad(img: np.ndarray) -> np.ndarray:
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


def preprocess_pairs(
    pairs_csv: str,
    finger_root: str,
    output_dir: str,
    image_size: int = 512,
    roi_size: int = 90,
    max_pairs: int = -1,
) -> None:
    """Main preprocessing loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading FingerNet model from: {finger_root}")
    model = build_fingernet_model(finger_root, image_size=image_size)
    print("FingerNet model loaded.")

    with open(pairs_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "img1" not in reader.fieldnames or "img2" not in reader.fieldnames:
            raise ValueError(
                "pairs_csv must have at least two columns named 'img1' and 'img2'."
            )

        for idx, row in enumerate(reader):
            if max_pairs > 0 and idx >= max_pairs:
                break

            img1_path = row["img1"]
            img2_path = row["img2"]

            print(f"[{idx}] Processing pair:")
            print(f"    img1: {img1_path}")
            print(f"    img2: {img2_path}")

            img1 = load_and_resize_grayscale(img1_path, image_size=image_size)
            img2 = load_and_resize_grayscale(img2_path, image_size=image_size)

            # Here we assume img1,img2 are already roughly aligned.
            # If you use the JAX alignment module, you can either:
            #   - run it before this script and save aligned images, or
            #   - run it later inside the JAX pipeline and only use
            #     enh1,enh2 here as enhanced originals.
            enh1 = enhance_with_fingernet(model, img1)
            enh2 = enhance_with_fingernet(model, img2)

            I_oe1, I_oe2, overlap_mask = extract_overlapped_region(enh1, enh2)
            roi1, roi2 = extract_roi_from_original(img1, img2, overlap_mask, roi_size=roi_size)

            out_file = output_path / f"pair_{idx:06d}.npz"
            np.savez_compressed(
                out_file,
                img1_path=img1_path,
                img2_path=img2_path,
                enh1=enh1,
                enh2=enh2,
                overlap_mask=overlap_mask,
                global1=I_oe1,
                global2=I_oe2,
                roi1=roi1,
                roi2=roi2,
            )
            print(f"    Saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline preprocessing with FingerNet enhancement + overlap + ROI."
    )
    parser.add_argument(
        "--pairs_csv",
        type=str,
        required=True,
        help="CSV file with at least columns 'img1,img2'.",
    )
    parser.add_argument(
        "--finger_root",
        type=str,
        default="FingerNet",
        help="Path to FingerNet root directory (containing 'src/' and 'models/').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed .npz files.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize images to this square size before enhancement.",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=90,
        help="Size of ROI patches.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=-1,
        help="Optionally limit the number of pairs processed (for debugging).",
    )

    args = parser.parse_args()

    preprocess_pairs(
        pairs_csv=args.pairs_csv,
        finger_root=args.finger_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        roi_size=args.roi_size,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()


