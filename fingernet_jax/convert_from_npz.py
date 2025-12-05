#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert FingerNet weights from exported .npz file to Flax format.
This script loads weights from a .npz file (exported by export_fingernet_weights_tf1.py)
and maps them to the Flax FingerNet model.

Usage:
    python convert_from_npz.py [path_to_exported_weights.npz]
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.serialization import to_state_dict

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from fingernet_flax import FingerNet

# Default paths
DEFAULT_NPZ_PATH = os.path.join(
    os.path.dirname(__file__), "fingernet_keras_tf1_weights.npz"
)
OUTPUT_NPZ_PATH = os.path.join(
    os.path.dirname(__file__), "fingernet_flax_params.npz"
)


def load_weights_from_npz(npz_path):
    """
    Load weights from exported .npz file.
    
    Returns:
        dict: {layer_name: {weight_index: array, ...}, ...}
    """
    print(f"Loading weights from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Organize weights by layer
    weights_by_layer = {}
    for key in data.files:
        if key.endswith("/__class__"):
            continue  # Skip class name markers
        
        # Parse: "layer_name/weight_0" -> ("layer_name", 0)
        parts = key.rsplit("/", 1)
        if len(parts) != 2:
            continue
        
        layer_name, weight_key = parts
        if not weight_key.startswith("weight_"):
            continue
        
        weight_idx = int(weight_key.split("_")[1])
        
        if layer_name not in weights_by_layer:
            weights_by_layer[layer_name] = {}
        weights_by_layer[layer_name][weight_idx] = data[key]
    
    print(f"Loaded weights for {len(weights_by_layer)} layers")
    return weights_by_layer


def init_flax_variables():
    """
    Initialize Flax FingerNet variables.
    """
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 512, 512, 1), dtype=jnp.float32)
    variables = FingerNet().init(rng, dummy, train=False)
    return variables


def _assign_param(tree, module_name, param_name, value):
    """
    Recursively search and assign parameter in Flax tree.
    """
    if not isinstance(tree, dict):
        return False
    
    if module_name in tree and isinstance(tree[module_name], dict) and param_name in tree[module_name]:
        tree[module_name][param_name] = jnp.array(value)
        return True
    
    for v in tree.values():
        if isinstance(v, dict) and _assign_param(v, module_name, param_name, value):
            return True
    
    return False


def map_layer_name_keras_to_flax(keras_name):
    """
    Map Keras layer name to Flax module name.
    
    Examples:
        "conv1_1" -> "conv-1_1"
        "bn-1_1" -> "bn-1_1" (unchanged)
        "prelu-1_1" -> "prelu-1_1" (unchanged)
        "atrousconv4_2" -> "atrousconv4_2" (unchanged)
    """
    # Most names are already correct, but conv layers need "-" prefix
    if keras_name.startswith("conv") and not keras_name.startswith("conv-"):
        # Check if it's a conv layer (not atrousconv)
        if not keras_name.startswith("atrousconv"):
            # Add "-" after "conv" if not present
            if keras_name[4] != "-":
                return "conv-" + keras_name[4:]
    return keras_name


def convert_weights(npz_path):
    """
    Convert weights from .npz to Flax format.
    """
    # Load exported weights
    keras_weights = load_weights_from_npz(npz_path)
    
    # Initialize Flax model
    print("Initializing Flax model...")
    variables = init_flax_variables()
    params = unfreeze(variables["params"])
    batch_stats = unfreeze(variables.get("batch_stats", {}))
    
    # Map weights
    mapped_count = 0
    skipped_count = 0
    
    for keras_name, weights_dict in keras_weights.items():
        # Get layer class from weights (if stored)
        weight_indices = sorted(weights_dict.keys())
        num_weights = len(weight_indices)
        
        # Map layer name
        flax_name = map_layer_name_keras_to_flax(keras_name)
        
        # Determine layer type from name pattern
        if keras_name.startswith("conv") or keras_name.startswith("atrousconv"):
            # Conv2D: weights[0] = kernel, weights[1] = bias (optional)
            if num_weights >= 1:
                kernel = weights_dict[0]
                bias = weights_dict[1] if num_weights >= 2 else np.zeros((kernel.shape[-1],), dtype=kernel.dtype)
                
                # Skip Gabor convs (they're fixed in Flax)
                if keras_name in ("enh_img_real_1", "enh_img_imag_1"):
                    print(f"  [SKIP] {keras_name} (Gabor conv - fixed in Flax)")
                    skipped_count += 1
                    continue
                
                ok_w = _assign_param(params, flax_name, "kernel", kernel)
                ok_b = _assign_param(params, flax_name, "bias", bias)
                if ok_w and ok_b:
                    print(f"  [OK] {keras_name} -> {flax_name} (Conv2D)")
                    mapped_count += 1
                else:
                    print(f"  [WARN] {keras_name} -> {flax_name} (Conv2D) - mapping failed")
                    skipped_count += 1
        
        elif keras_name.startswith("bn-"):
            # BatchNormalization: weights[0]=gamma, [1]=beta, [2]=mean, [3]=var
            if num_weights == 4:
                gamma = weights_dict[0]
                beta = weights_dict[1]
                mean = weights_dict[2]
                var = weights_dict[3]
                
                ok_scale = _assign_param(params, flax_name, "scale", gamma)
                ok_bias = _assign_param(params, flax_name, "bias", beta)
                ok_mean = _assign_param(batch_stats, flax_name, "mean", mean)
                ok_var = _assign_param(batch_stats, flax_name, "var", var)
                
                if ok_scale and ok_bias and ok_mean and ok_var:
                    print(f"  [OK] {keras_name} -> {flax_name} (BatchNorm)")
                    mapped_count += 1
                else:
                    print(f"  [WARN] {keras_name} -> {flax_name} (BatchNorm) - mapping failed")
                    skipped_count += 1
            else:
                print(f"  [WARN] {keras_name} - expected 4 weights, got {num_weights}")
                skipped_count += 1
        
        elif keras_name.startswith("prelu-"):
            # PReLU: weights[0] = alpha (often shape (1,1,C))
            if num_weights >= 1:
                alpha = weights_dict[0]
                alpha = np.squeeze(alpha)  # (1,1,C) -> (C,)
                
                ok_alpha = _assign_param(params, flax_name, "alpha", alpha)
                if ok_alpha:
                    print(f"  [OK] {keras_name} -> {flax_name} (PReLU)")
                    mapped_count += 1
                else:
                    print(f"  [WARN] {keras_name} -> {flax_name} (PReLU) - mapping failed")
                    skipped_count += 1
            else:
                print(f"  [WARN] {keras_name} - expected 1 weight, got {num_weights}")
                skipped_count += 1
        
        else:
            # Other layers (e.g., Activation, Lambda) - no weights
            print(f"  [SKIP] {keras_name} (no trainable weights)")
            skipped_count += 1
    
    # Save converted weights
    print(f"\nMapping complete: {mapped_count} mapped, {skipped_count} skipped")
    print(f"Saving to: {OUTPUT_NPZ_PATH}")
    
    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez(
        OUTPUT_NPZ_PATH,
        params=to_state_dict(freeze(params)),
        batch_stats=to_state_dict(freeze(batch_stats)),
    )
    
    print(f"✓ Saved Flax parameters to: {OUTPUT_NPZ_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_NPZ_PATH) / (1024*1024):.2f} MB")
    
    return OUTPUT_NPZ_PATH


if __name__ == "__main__":
    import jax
    
    npz_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ_PATH
    
    if not os.path.exists(npz_path):
        print(f"✗ Error: Weight file not found: {npz_path}")
        print(f"\nUsage: python convert_from_npz.py [path_to_weights.npz]")
        print(f"\nFirst, export weights from TF1/Keras2 environment:")
        print(f"  1. Set up TF1/Keras2 environment (Docker/Conda)")
        print(f"  2. Run: python export_fingernet_weights_tf1.py")
        print(f"  3. Copy fingernet_keras_tf1_weights.npz to this directory")
        print(f"  4. Run: python convert_from_npz.py")
        sys.exit(1)
    
    try:
        convert_weights(npz_path)
    except Exception as e:
        import traceback
        print(f"\n✗ Error during conversion:")
        traceback.print_exc()
        sys.exit(1)

