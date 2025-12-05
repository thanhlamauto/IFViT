#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test FingerNet on a single image.
Wrapper script to handle Keras 3 compatibility issues.
"""

import os
import sys
import numpy as np
import cv2
from scipy import misc, ndimage

# Fake keras module for compatibility
import tensorflow as tf
if "keras" in sys.modules:
    del sys.modules["keras"]
sys.modules["keras"] = tf.keras

# Create fake keras.layers submodules
keras = sys.modules["keras"]
keras.layers = tf.keras.layers
keras.layers.core = tf.keras.layers
keras.layers.convolutional = tf.keras.layers
keras.layers.normalization = tf.keras.layers
keras.layers.advanced_activations = tf.keras.layers
keras.models = tf.keras.models
keras.regularizers = tf.keras.regularizers
keras.optimizers = tf.keras.optimizers
keras.utils = tf.keras.utils
keras.callbacks = tf.keras.callbacks
keras.backend = tf.keras.backend
keras.backend.tf = tf.compat.v1
keras.backend.set_session = lambda sess: None

# Fix Lambda layer for Keras 3
class LegacyLambda(tf.keras.layers.Layer):
    def __init__(self, function, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.function = function
        self._output_shape = output_shape
    
    def call(self, inputs):
        return self.function(inputs)
    
    def compute_output_shape(self, input_shape):
        if self._output_shape is not None:
            return self._output_shape
        # Return input shape as default
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

# Replace Lambda in keras.layers
keras.layers.Lambda = LegacyLambda

# Now import the original code
sys.path.insert(0, os.path.dirname(__file__))
import train_test_deploy as ttd

def test_single_image(image_path, output_dir="./test_output"):
    """
    Test FingerNet on a single image.
    
    Args:
        image_path: Path to input fingerprint image (.bmp)
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = misc.imread(image_path, mode='L')
    print(f"Image shape: {img.shape}")
    
    # Preprocess: make size multiple of 8
    img_size = np.array(img.shape, dtype=np.int32) // 8 * 8
    img = img[:img_size[0], :img_size[1]]
    
    # Normalize to [0, 1]
    image = img.astype(np.float32) / 255.0
    image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
    
    # Load model
    model_path = "../models/released_version/Model.model"
    print(f"Loading model from: {model_path}")
    model = ttd.get_main_net((img_size[0], img_size[1], 1), model_path)
    
    # Predict
    print("Running inference...")
    enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = model.predict(image)
    
    # Postprocess
    print("Postprocessing...")
    round_seg = np.round(np.squeeze(seg_out))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    seg_out_clean = cv2.morphologyEx(round_seg, cv2.MORPH_OPEN, kernel)
    
    # Extract minutiae
    mnt = ttd.label2mnt(
        np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out_clean)),
        mnt_w_out, mnt_h_out, mnt_o_out,
        thresh=0.5
    )
    mnt_nms = ttd.nms(mnt)
    
    # Orientation
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    ori_peak = sess.run(ttd.ori_highest_peak(ori_out_1))
    ori = (np.argmax(ori_peak, axis=-1) * 2 - 90) / 180.0 * np.pi
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save enhanced image
    enh_path = os.path.join(output_dir, f"{base_name}_enh.png")
    enh_img = np.squeeze(enhance_img) * ndimage.zoom(np.round(np.squeeze(seg_out_clean)), [8, 8], order=0)
    misc.imsave(enh_path, enh_img)
    print(f"Saved enhanced image: {enh_path}")
    
    # Save segmentation
    seg_path = os.path.join(output_dir, f"{base_name}_seg.png")
    seg_full = ndimage.zoom(np.round(np.squeeze(seg_out_clean)), [8, 8], order=0)
    misc.imsave(seg_path, seg_full)
    print(f"Saved segmentation: {seg_path}")
    
    # Save orientation visualization
    ori_path = os.path.join(output_dir, f"{base_name}_ori.png")
    ttd.draw_ori_on_img(image, ori, np.ones_like(seg_out_clean), ori_path)
    print(f"Saved orientation: {ori_path}")
    
    # Save minutiae
    mnt_path = os.path.join(output_dir, f"{base_name}_mnt.png")
    ttd.draw_minutiae(image, mnt_nms[:, :3], mnt_path)
    print(f"Saved minutiae: {mnt_path}")
    
    # Save minutiae file
    mnt_file = os.path.join(output_dir, f"{base_name}.mnt")
    ttd.mnt_writer(mnt_nms, base_name, img_size, mnt_file)
    print(f"Saved minutiae file: {mnt_file}")
    
    print(f"\n✓ Processing complete!")
    print(f"  Minutiae found: {len(mnt_nms)}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path> [output_dir]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"
    
    test_single_image(image_path, output_dir)

