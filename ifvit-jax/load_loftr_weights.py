"""
Utility to load pretrained LoFTR weights into JAX model.

This module provides functions to:
1. Load converted LoFTR checkpoint (.npz)
2. Map weights to Flax model structure
3. Initialize model with pretrained weights
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Any
import pickle


def load_loftr_weights(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load converted LoFTR weights from .npz file.
    
    Args:
        npz_path: Path to converted LoFTR checkpoint (.npz)
        
    Returns:
        Dictionary of parameter arrays
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"LoFTR checkpoint not found: {npz_path}")
    
    print(f"Loading LoFTR weights from: {npz_path}")
    data = np.load(npz_path)
    
    weights = {k: data[k] for k in data.files}
    print(f"  Loaded {len(weights)} parameter arrays")
    
    total_params = sum(v.size for v in weights.values())
    print(f"  Total parameters: {total_params:,}")
    
    return weights


def merge_params(base_params: Dict, loftr_weights: Dict[str, np.ndarray], prefix: str = 'loftr_transformer') -> Dict:
    """
    Merge LoFTR weights into base parameter tree.
    
    Args:
        base_params: Base Flax parameter tree (from model.init())
        loftr_weights: LoFTR weights from .npz file
        prefix: Prefix path in parameter tree where LoFTR weights should go
        
    Returns:
        Updated parameter tree with LoFTR weights merged in
    """
    import copy
    params = copy.deepcopy(base_params)
    
    # Navigate to the right location in parameter tree
    if prefix:
        parts = prefix.split('/')
        target = params
        for part in parts:
            if part not in target:
                print(f"⚠ Warning: prefix '{prefix}' not found in parameter tree")
                return params
            target = target[part]
    else:
        target = params
    
    # Merge weights
    num_merged = 0
    for loftr_key, loftr_value in loftr_weights.items():
        # Parse the Flax-style key (e.g., 'layers.0.0/q_proj/kernel')
        parts = loftr_key.split('/')
        
        # Navigate to the right location
        current = target
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                print(f"  ⚠ Skipping {loftr_key}: path not found")
                break
            current = current[part]
        else:
            # Set the parameter
            param_name = parts[-1]
            if param_name in current:
                # Check shape compatibility
                expected_shape = current[param_name].shape
                if loftr_value.shape == expected_shape:
                    current[param_name] = jnp.array(loftr_value)
                    num_merged += 1
                    print(f"  ✓ Merged: {loftr_key} {loftr_value.shape}")
                else:
                    print(f"  ⚠ Shape mismatch for {loftr_key}: " 
                          f"expected {expected_shape}, got {loftr_value.shape}")
            else:
                print(f"  ⚠ Parameter not found: {loftr_key}")
    
    print(f"\n✓ Merged {num_merged}/{len(loftr_weights)} LoFTR parameters")
    
    return params


def initialize_with_loftr(
    model_class,
    model_kwargs: Dict[str, Any],
    loftr_ckpt_path: str,
    rng: jax.random.PRNGKey,
    dummy_inputs: Dict[str, jnp.ndarray]
) -> Dict:
    """
    Initialize a model with LoFTR pretrained weights.
    
    Args:
        model_class: Model class (e.g., DenseRegModel)
        model_kwargs: Keyword arguments for model constructor
        loftr_ckpt_path: Path to converted LoFTR checkpoint (.npz)
        rng: Random key for initialization
        dummy_inputs: Dictionary of dummy inputs for model.init()
        
    Returns:
        Parameters with LoFTR weights loaded
    """
    # Create model
    model = model_class(**model_kwargs)
    
    # Initialize with random weights
    print("Initializing model...")
    variables = model.init(rng, **dummy_inputs)
    params = variables['params']
    
    # Load LoFTR weights
    if loftr_ckpt_path and Path(loftr_ckpt_path).exists():
        loftr_weights = load_loftr_weights(loftr_ckpt_path)
        
        # Merge into params
        # The exact prefix depends on model structure
        # For DenseRegModel with use_loftr=True, it should be at 'loftr_transformer'
        params = merge_params(params, loftr_weights, prefix='loftr_transformer')
    else:
        if loftr_ckpt_path:
            print(f"⚠ Warning: LoFTR checkpoint not found: {loftr_ckpt_path}")
        print("  Using random initialization for all parameters")
    
    return params


def save_params_with_loftr(filepath: str, params: Dict, metadata: Dict = None):
    """
    Save parameters (including LoFTR weights) to file.
    
    Args:
        filepath: Output file path
        params: Parameter dictionary
        metadata: Optional metadata
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'params': params,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"✓ Saved parameters to {filepath}")


# ============================================================================
# Example Usage
# ============================================================================

def example_load_loftr_for_dense_model():
    """
    Example: Load LoFTR weights for DenseRegModel.
    """
    from models import DenseRegModel
    from config import DENSE_CONFIG
    
    # Model configuration
    config = DENSE_CONFIG.copy()
    config['use_loftr'] = True
    config['loftr_pretrained_ckpt'] = './loftr_transformer.npz'
    
    # Create dummy inputs
    rng = jax.random.PRNGKey(0)
    dummy_img = jnp.zeros((1, config['image_size'], config['image_size'], 1))
    dummy_inputs = {
        'img1': dummy_img,
        'img2': dummy_img,
        'train': False
    }
    
    # Initialize model with LoFTR weights
    params = initialize_with_loftr(
        model_class=DenseRegModel,
        model_kwargs={
            'image_size': config['image_size'],
            'num_transformer_layers': config['transformer_layers'],
            'num_heads': config['num_heads'],
            'hidden_dim': config['hidden_dim'],
            'mlp_dim': config['mlp_dim'],
            'dropout_rate': config['dropout_rate'],
            'use_loftr': config['use_loftr'],
            'attention_type': config['attention_type']
        },
        loftr_ckpt_path=config['loftr_pretrained_ckpt'],
        rng=rng,
        dummy_inputs=dummy_inputs
    )
    
    print(f"\n✓ Model initialized with LoFTR pretrained weights")
    return params


if __name__ == '__main__':
    # Example usage
    print("="*60)
    print("LoFTR Weight Loading Example")
    print("="*60)
    
    try:
        params = example_load_loftr_for_dense_model()
    except Exception as e:
        print(f"\n⚠ Example failed: {e}")
        print("\nTo use this:")
        print("1. Convert LoFTR checkpoint:")
        print("   python convert_loftr_checkpoint.py --pytorch_ckpt outdoor_ds.ckpt --output loftr_transformer.npz")
        print("\n2. Set loftr_pretrained_ckpt in config.py")
        print("\n3. Run training with use_loftr=True")
