#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export FingerNet weights from Model.model to .npz format.
This script MUST be run in a TensorFlow 1.x / Keras 2.x environment.

Usage:
    # In TF1/Keras2 environment (e.g., Docker with TF1):
    python export_fingernet_weights_tf1.py
    
    # Or with conda environment:
    conda activate tf1_env
    python export_fingernet_weights_tf1.py
"""

import os
import sys
import numpy as np

# Add FingerNet src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINGERNET_SRC_DIR = os.path.join(SCRIPT_DIR, "..", "FingerNet", "src")
FINGERNET_SRC_DIR = os.path.abspath(FINGERNET_SRC_DIR)

if FINGERNET_SRC_DIR not in sys.path:
    sys.path.insert(0, FINGERNET_SRC_DIR)

# Change to FingerNet src directory for imports
original_cwd = os.getcwd()
print(f"Changing to: {FINGERNET_SRC_DIR}")
os.chdir(FINGERNET_SRC_DIR)

# Set minimal argv for train_test_deploy.py
original_argv = sys.argv
sys.argv = ["train_test_deploy.py", "0", "deploy"]

try:
    import train_test_deploy as ttd
finally:
    os.chdir(original_cwd)
    sys.argv = original_argv

# Paths
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "FingerNet", "models", "released_version", "Model.model"
)
OUTPUT_NPZ = os.path.join(
    os.path.dirname(__file__), "fingernet_keras_tf1_weights.npz"
)


def export_weights():
    """
    Load Keras model and export all weights to .npz file.
    """
    import sys
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Output will be saved to: {OUTPUT_NPZ}")
    sys.stdout.flush()
    
    # Build model and load weights
    print("Building model architecture...")
    sys.stdout.flush()
    
    try:
        print("  → Creating model graph (this may take 1-2 minutes)...")
        sys.stdout.flush()
        model = ttd.get_main_net((512, 512, 1), MODEL_PATH)
        print("✓ Model loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Extract all weights
    weights_dict = {}
    total_layers = len([l for l in model.layers if len(l.get_weights()) > 0])
    current_layer = 0
    
    print(f"\nExporting weights from {total_layers} layers...")
    sys.stdout.flush()
    
    for layer in model.layers:
        layer_name = layer.name
        layer_weights = layer.get_weights()
        
        if len(layer_weights) == 0:
            continue
        
        current_layer += 1
        progress = int(100 * current_layer / total_layers)
        print(f"[{progress:3d}%] Exporting layer: {layer_name} ({len(layer_weights)} weight arrays)")
        sys.stdout.flush()
        
        # Store weights with layer name as prefix
        for i, w in enumerate(layer_weights):
            key = f"{layer_name}/weight_{i}"
            weights_dict[key] = w
        
        # Also store layer class name for reference
        weights_dict[f"{layer_name}/__class__"] = layer.__class__.__name__
    
    # Save to .npz
    print("\nSaving to .npz file...")
    sys.stdout.flush()
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ, **weights_dict)
    
    file_size_mb = os.path.getsize(OUTPUT_NPZ) / (1024*1024)
    print(f"\n✓ Successfully exported {len(weights_dict)} weight arrays to: {OUTPUT_NPZ}")
    print(f"  File size: {file_size_mb:.2f} MB")
    sys.stdout.flush()
    
    return OUTPUT_NPZ


if __name__ == "__main__":
    try:
        export_weights()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNOTE: This script requires TensorFlow 1.x / Keras 2.x environment.")
        print("If you're using TF2/Keras3, you need to:")
        print("  1. Set up a TF1 environment (Docker/Conda)")
        print("  2. Run this script there")
        print("  3. Then use the exported .npz file with convert_keras_to_flax.py")
        sys.exit(1)

