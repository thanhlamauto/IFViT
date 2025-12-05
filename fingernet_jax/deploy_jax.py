# deploy_jax.py
import os
import sys
from time import time
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze
from scipy import ndimage, io
import cv2

from utils_jax import (
    mkdir,
    init_log,
    get_files_in_folder,
    mnt_writer,
    draw_ori_on_img,
    draw_minutiae,
    label2mnt,
    nms,
)
from fingernet_flax import FingerNet, ori_highest_peak_jax


PRETRAIN_PARAMS = "./fingernet_flax_params.npz"  # file npz chứa params & batch_stats đã convert
OUTPUT_ROOT = "../output_jax/"


def load_params_from_npz(path):
    """Load Flax parameters from npz file."""
    data = np.load(path, allow_pickle=True)
    params = freeze(data["params"].item())
    batch_stats = freeze(data["batch_stats"].item()) if "batch_stats" in data else freeze({})
    variables = {"params": params}
    if batch_stats:
        variables["batch_stats"] = batch_stats
    return variables




def build_model_and_state(img_size, params_path=None):
    """
    Build model and load weights.
    
    Args:
        img_size: (H, W) tuple
        params_path: Path to converted weights npz file
        
    Returns:
        model, variables
    """
    model = FingerNet()
    
    # Load weights if available
    if params_path and os.path.exists(params_path):
        print(f"✓ Loading weights from: {params_path}")
        variables = load_params_from_npz(params_path)
    else:
        print("⚠️  No weights file found. Using random initialization.")
        print("   NOTE: Results will be random. Run convert_keras_to_flax.py first!")
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, img_size[0], img_size[1], 1), dtype=jnp.float32)
        variables = model.init(rng, dummy, train=False)

    return model, variables


def forward(model, variables, image_batch):
    """Forward pass - model is static, so we can't jit it directly."""
    return model.apply(variables, image_batch, train=False)


def deploy(deploy_set, set_name=None):
    if set_name is None:
        set_name = os.path.basename(os.path.normpath(deploy_set))

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(OUTPUT_ROOT, timestamp)
    mkdir(output_dir)
    log = init_log(output_dir)
    mkdir(os.path.join(output_dir, set_name))

    log.info(f"Predicting {set_name}:")
    _, img_name = get_files_in_folder(deploy_set, '.bmp')
    if len(img_name) == 0:
        deploy_set = os.path.join(deploy_set, 'images/')
        _, img_name = get_files_in_folder(deploy_set, '.bmp')

    # đọc 1 ảnh để xác định size (và làm cho chia hết cho 8)
    img0 = cv2.imread(os.path.join(deploy_set, img_name[0] + '.bmp'), cv2.IMREAD_GRAYSCALE)
    img_size = np.array(img0.shape, dtype=np.int32) // 8 * 8

    model, variables = build_model_and_state(img_size, params_path=PRETRAIN_PARAMS)

    time_c = []
    for i, name in enumerate(img_name):
        log.info("%s %d / %d: %s", set_name, i + 1, len(img_name), name)
        t0 = time()

        image = cv2.imread(os.path.join(deploy_set, name + '.bmp'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        image = image[:img_size[0], :img_size[1]]
        image_batch = image[None, ..., None]  # (1,H,W,1)
        image_batch_jax = jnp.array(image_batch)

        outputs = forward(model, variables, image_batch_jax)
        t1 = time()

        enh_img_real = np.array(outputs["enh_img_real"])
        ori_out_1 = np.array(outputs["ori_out_1"])
        seg_out = np.array(outputs["seg_out"])
        mnt_o_out = np.array(outputs["mnt_o_out"])
        mnt_w_out = np.array(outputs["mnt_w_out"])
        mnt_h_out = np.array(outputs["mnt_h_out"])
        mnt_s_out = np.array(outputs["mnt_s_out"])

        round_seg = np.round(np.squeeze(seg_out))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out_post = cv2.morphologyEx(round_seg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        mnt = label2mnt(
            np.squeeze(mnt_s_out) * np.round(seg_out_post),
            mnt_w_out,
            mnt_h_out,
            mnt_o_out,
            thresh=0.5
        )
        mnt_nms = nms(mnt)

        # orientation từ ori_out_1
        angval = ori_highest_peak_jax(jnp.array(ori_out_1))
        angval = np.array(angval)
        ori = (np.argmax(angval, axis=-1) * 2 - 90) / 180.0 * np.pi

        t2 = time()

        set_dir = os.path.join(output_dir, set_name)
        mnt_writer(mnt_nms, name, img_size, os.path.join(set_dir, f"{name}.mnt"))
        draw_ori_on_img(image_batch, ori, np.ones_like(seg_out_post), os.path.join(set_dir, f"{name}_ori.png"))
        draw_minutiae(image_batch, mnt_nms[:, :3], os.path.join(set_dir, f"{name}_mnt.png"))

        enh_save = np.squeeze(enh_img_real) * ndimage.zoom(np.round(seg_out_post), [8, 8], order=0)
        cv2.imwrite(os.path.join(set_dir, f"{name}_enh.png"), (enh_save * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(set_dir, f"{name}_seg.png"),
                    (ndimage.zoom(np.round(seg_out_post), [8, 8], order=0) * 255).astype(np.uint8))
        io.savemat(os.path.join(set_dir, f"{name}.mat"),
                   {'orientation': ori, 'orientation_distribution_map': ori_out_1})

        t3 = time()

        time_c.append([t1 - t0, t2 - t1, t3 - t2])
        log.info("load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f",
                 time_c[-1][0], time_c[-1][1], time_c[-1][2])

    time_c = np.mean(np.array(time_c), axis=0)
    log.info("Average: load+conv: %.3fs, ori-select+seg-post+nms: %.3f, draw: %.3f",
             time_c[0], time_c[1], time_c[2])


def test_single_image(image_path, output_dir="./test_output", params_path=None):
    """
    Test FingerNet JAX on a single image.
    
    Args:
        image_path: Path to input fingerprint image (.bmp)
        output_dir: Directory to save outputs
        params_path: Path to converted weights (default: PRETRAIN_PARAMS)
    """
    if params_path is None:
        params_path = PRETRAIN_PARAMS
    
    mkdir(output_dir)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Image shape: {img.shape}")
    
    # Preprocess: make size multiple of 8
    img_size = np.array(img.shape, dtype=np.int32) // 8 * 8
    img = img[:img_size[0], :img_size[1]]
    
    # Normalize to [0, 1]
    image = img.astype(np.float32) / 255.0
    image_batch = image[None, ..., None]  # (1, H, W, 1)
    image_batch_jax = jnp.array(image_batch)
    
    # Build model and load weights
    print("Building model...")
    model, variables = build_model_and_state(img_size, params_path=params_path)
    
    # Forward pass
    print("Running inference...")
    t0 = time()
    outputs = forward(model, variables, image_batch_jax)
    t1 = time()
    print(f"Inference time: {t1-t0:.3f}s")
    
    # Extract outputs
    enh_img_real = np.array(outputs["enh_img_real"])
    ori_out_1 = np.array(outputs["ori_out_1"])
    seg_out = np.array(outputs["seg_out"])
    mnt_o_out = np.array(outputs["mnt_o_out"])
    mnt_w_out = np.array(outputs["mnt_w_out"])
    mnt_h_out = np.array(outputs["mnt_h_out"])
    mnt_s_out = np.array(outputs["mnt_s_out"])
    
    # Postprocess
    print("Postprocessing...")
    round_seg = np.round(np.squeeze(seg_out))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    seg_out_post = cv2.morphologyEx(round_seg.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Extract minutiae
    mnt = label2mnt(
        np.squeeze(mnt_s_out) * np.round(seg_out_post),
        mnt_w_out,
        mnt_h_out,
        mnt_o_out,
        thresh=0.5
    )
    mnt_nms = nms(mnt)
    
    # Orientation
    angval = ori_highest_peak_jax(jnp.array(ori_out_1))
    angval = np.array(angval)
    ori = (np.argmax(angval, axis=-1) * 2 - 90) / 180.0 * np.pi
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Enhanced image
    enh_path = os.path.join(output_dir, f"{base_name}_enh.png")
    enh_img = np.squeeze(enh_img_real) * ndimage.zoom(np.round(seg_out_post), [8, 8], order=0)
    cv2.imwrite(enh_path, (enh_img * 255).astype(np.uint8))
    print(f"✓ Saved enhanced image: {enh_path}")
    
    # Segmentation
    seg_path = os.path.join(output_dir, f"{base_name}_seg.png")
    seg_full = ndimage.zoom(np.round(seg_out_post), [8, 8], order=0)
    cv2.imwrite(seg_path, (seg_full * 255).astype(np.uint8))
    print(f"✓ Saved segmentation: {seg_path}")
    
    # Orientation visualization
    ori_path = os.path.join(output_dir, f"{base_name}_ori.png")
    draw_ori_on_img(image_batch, ori, np.ones_like(seg_out_post), ori_path)
    print(f"✓ Saved orientation: {ori_path}")
    
    # Minutiae visualization
    mnt_path = os.path.join(output_dir, f"{base_name}_mnt.png")
    draw_minutiae(image_batch, mnt_nms[:, :3], mnt_path)
    print(f"✓ Saved minutiae: {mnt_path}")
    
    # Minutiae file
    mnt_file = os.path.join(output_dir, f"{base_name}.mnt")
    mnt_writer(mnt_nms, base_name, img_size, mnt_file)
    print(f"✓ Saved minutiae file: {mnt_file}")
    
    # Orientation mat file
    mat_path = os.path.join(output_dir, f"{base_name}.mat")
    io.savemat(mat_path, {'orientation': ori, 'orientation_distribution_map': ori_out_1})
    print(f"✓ Saved orientation mat: {mat_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  Minutiae found: {len(mnt_nms)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    if len(sys.argv) > 1:
        # Test single image
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"
        params_path = sys.argv[3] if len(sys.argv) > 3 else None
        test_single_image(image_path, output_dir, params_path)
    else:
        # Batch deploy
        deploy_set = "../FingerNet/datasets/NIST4/"
        deploy(deploy_set, set_name="NIST4_jax")


if __name__ == "__main__":
    main()
