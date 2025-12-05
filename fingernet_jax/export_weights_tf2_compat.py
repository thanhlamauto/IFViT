#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attempt to export FingerNet weights using TensorFlow 2.x with TF1 compatibility mode.
This is a fallback if Docker/Conda TF1 environment is not available.

Usage:
    python export_weights_tf2_compat.py
"""

import os
import sys
import numpy as np

# Try to use TF1 compatibility mode
import tensorflow as tf

# Enable TF1 compatibility
tf.compat.v1.disable_v2_behavior()

# Set up TF1-style session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Add FingerNet src to path
FINGERNET_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "FingerNet", "src")
if FINGERNET_SRC_DIR not in sys.path:
    sys.path.insert(0, FINGERNET_SRC_DIR)

# Change to FingerNet src directory for imports
original_cwd = os.getcwd()
os.chdir(FINGERNET_SRC_DIR)

# Set minimal argv for train_test_deploy.py
original_argv = sys.argv
sys.argv = ["train_test_deploy.py", "0", "deploy"]

try:
    # Monkey-patch keras imports to use tf.keras
    import types
    
    # Create fake keras module
    keras_root = types.ModuleType("keras")
    keras_root.backend = tf.compat.v1.keras.backend
    keras_root.models = tf.keras.models
    keras_root.layers = tf.keras.layers
    keras_root.regularizers = tf.keras.regularizers
    keras_root.optimizers = tf.keras.optimizers
    keras_root.utils = tf.keras.utils
    keras_root.callbacks = tf.keras.callbacks
    
    # Create submodules
    keras_layers = types.ModuleType("keras.layers")
    keras_layers.core = tf.keras.layers
    keras_layers.convolutional = tf.keras.layers
    keras_layers.normalization = tf.keras.layers
    keras_layers.advanced_activations = tf.keras.layers
    
    keras_root.layers = keras_layers
    
    sys.modules["keras"] = keras_root
    sys.modules["keras.backend"] = keras_root.backend
    sys.modules["keras.models"] = keras_root.models
    sys.modules["keras.layers"] = keras_layers
    sys.modules["keras.layers.core"] = keras_layers.core
    sys.modules["keras.layers.convolutional"] = keras_layers.convolutional
    sys.modules["keras.layers.normalization"] = keras_layers.normalization
    sys.modules["keras.layers.advanced_activations"] = keras_layers.advanced_activations
    sys.modules["keras.regularizers"] = keras_root.regularizers
    sys.modules["keras.optimizers"] = keras_root.optimizers
    sys.modules["keras.utils"] = keras_root.utils
    sys.modules["keras.callbacks"] = keras_root.callbacks
    
    # Patch utils.py copy_file
    import importlib.util
    utils_path = os.path.join(FINGERNET_SRC_DIR, "utils.py")
    spec = importlib.util.spec_from_file_location("utils", utils_path)
    if spec and spec.loader:
        utils_mod = importlib.util.module_from_spec(spec)
        sys.modules["utils"] = utils_mod
        spec.loader.exec_module(utils_mod)
        utils_mod.copy_file = lambda a, b: None  # no-op
    
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
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Output will be saved to: {OUTPUT_NPZ}")
    print(f"Using TensorFlow {tf.__version__} with TF1 compatibility mode")
    
    try:
        # Build model and load weights
        model = ttd.get_main_net((512, 512, 1), MODEL_PATH)
        
        # Extract all weights
        weights_dict = {}
        for layer in model.layers:
            layer_name = layer.name
            layer_weights = layer.get_weights()
            
            if len(layer_weights) == 0:
                continue
                
            print(f"Exporting layer: {layer_name} ({len(layer_weights)} weight arrays)")
            
            # Store weights with layer name as prefix
            for i, w in enumerate(layer_weights):
                key = f"{layer_name}/weight_{i}"
                weights_dict[key] = w
                print(f"  {key}: shape={w.shape}, dtype={w.dtype}")
            
            # Also store layer class name for reference
            weights_dict[f"{layer_name}/__class__"] = layer.__class__.__name__
        
        # Save to .npz
        os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
        np.savez(OUTPUT_NPZ, **weights_dict)
        
        print(f"\n✓ Successfully exported {len(weights_dict)} weight arrays to: {OUTPUT_NPZ}")
        print(f"  File size: {os.path.getsize(OUTPUT_NPZ) / (1024*1024):.2f} MB")
        
        return OUTPUT_NPZ
    
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nThis approach may not work with Model.model format.")
        print("Please use Docker/Conda with TensorFlow 1.x instead:")
        print("  docker run -it --rm -v $(pwd)/..:/workspace tensorflow/tensorflow:1.15.5-gpu-py3 bash")
        raise


if __name__ == "__main__":
    try:
        export_weights()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

