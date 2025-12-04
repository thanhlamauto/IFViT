"""
Load Module 1 weights into Module 2.

This follows IFViT paper: Module 2 initializes from Module 1's TRAINED transformer,
not from LoFTR directly.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any
from pathlib import Path
import copy

from .utils import load_checkpoint


def load_module1_transformer_weights(
    module1_ckpt_path: str,
    module2_params: Dict[str, Any],
    share_global_local: bool = True
) -> Dict[str, Any]:
    """
    Load trained transformer weights from Module 1 into Module 2.
    
    Following IFViT paper:
    - Module 1 trains ViT with LoFTR init + L_D loss
    - Module 2 reuses Module 1's TRAINED ViT, NOT fresh LoFTR weights
    
    Args:
        module1_ckpt_path: Path to trained DenseRegModel checkpoint
        module2_params: MatcherModel parameter tree (from model.init())
        share_global_local: If True, use same transformer weights for global & local
                            If False, copy Module 1 weights to both branches separately
        
    Returns:
        Updated module2_params with Module 1's transformer weights loaded
    """
    if not Path(module1_ckpt_path).exists():
        raise FileNotFoundError(f"Module 1 checkpoint not found: {module1_ckpt_path}")
    
    print(f"\n{'='*60}")
    print("Loading Module 1 Transformer Weights into Module 2")
    print(f"{'='*60}")
    print(f"Source: {module1_ckpt_path}")
    
    # Load Module 1 checkpoint
    state_dict, metadata = load_checkpoint(module1_ckpt_path)
    module1_params = state_dict['params']
    
    # Extract transformer weights from Module 1
    # Need to find the transformer path
    transformer_params = None
    transformer_key = None
    
    # Try different possible keys
    possible_keys = [
        'loftr_transformer',  # If use_loftr=True
        'siamese_transformer'  # If use_loftr=False
    ]
    
    for key in possible_keys:
        if key in module1_params:
            transformer_params = module1_params[key]
            transformer_key = key
            print(f"✓ Found transformer: {key}")
            break
    
    if transformer_params is None:
        print("⚠ Warning: No transformer found in Module 1 checkpoint")
        print("  Available keys:", list(module1_params.keys()))
        print("  Returning Module 2 params unchanged")
        return module2_params
    
    # Copy to Module 2
    params = copy.deepcopy(module2_params)
    num_params_copied = 0
    
    if share_global_local:
        # Strategy 1: Share transformer between global and local
        # Both branches use exact same weights
        print("\n✓ Sharing transformer weights between global and local branches")
        
        # Copy to global branch
        if 'loftr_transformer_global' in params:
            params['loftr_transformer_global'] = copy.deepcopy(transformer_params)
            num_params_copied += count_params(transformer_params)
            print(f"  ✓ Copied to loftr_transformer_global")
        elif 'siamese_transformer' in params:
            # If using generic transformer
            params['siamese_transformer'] = copy.deepcopy(transformer_params)
            num_params_copied += count_params(transformer_params)
            print(f"  ✓ Copied to siamese_transformer")
        
        # Copy to local branch (same weights)
        if 'loftr_transformer_local' in params:
            params['loftr_transformer_local'] = copy.deepcopy(transformer_params)
            print(f"  ✓ Copied to loftr_transformer_local (shared weights)")
    
    else:
        # Strategy 2: Independent copies for global and local
        print("\n✓ Creating independent transformer copies for global and local")
        
        # Copy to global
        if 'loftr_transformer_global' in params:
            params['loftr_transformer_global'] = copy.deepcopy(transformer_params)
            num_params_copied += count_params(transformer_params)
            print(f"  ✓ Copied to loftr_transformer_global")
        
        # Copy to local (independent copy)
        if 'loftr_transformer_local' in params:
            params['loftr_transformer_local'] = copy.deepcopy(transformer_params)
            num_params_copied += count_params(transformer_params)
            print(f"  ✓ Copied to loftr_transformer_local")
    
    print(f"\n✓ Successfully loaded {num_params_copied:,} parameters from Module 1")
    print(f"{'='*60}\n")
    
    return params


def count_params(params_dict: Dict) -> int:
    """Recursively count parameters in nested dict."""
    total = 0
    for v in params_dict.values():
        if isinstance(v, dict):
            total += count_params(v)
        elif hasattr(v, 'size'):
            total += v.size
    return total


def verify_module1_loading(module1_ckpt_path: str):
    """
    Verify that Module 1 checkpoint can be loaded and contains transformer weights.
    
    Args:
        module1_ckpt_path: Path to Module 1 checkpoint
    """
    if not Path(module1_ckpt_path).exists():
        print(f"✗ Module 1 checkpoint not found: {module1_ckpt_path}")
        return False
    
    try:
        state_dict, metadata = load_checkpoint(module1_ckpt_path)
        params = state_dict['params']
        
        print(f"\n{'='*60}")
        print("Module 1 Checkpoint Verification")
        print(f"{'='*60}")
        print(f"Checkpoint: {module1_ckpt_path}")
        print(f"\nTop-level keys:")
        for key in params.keys():
            if isinstance(params[key], dict):
                num_params = count_params(params[key])
                print(f"  {key}: {num_params:,} parameters")
            else:
                print(f"  {key}: {params[key].shape if hasattr(params[key], 'shape') else type(params[key])}")
        
        # Check for transformer
        has_transformer = ('loftr_transformer' in params) or ('siamese_transformer' in params)
        
        if has_transformer:
            print(f"\n✓ Transformer found - ready for Module 2")
        else:
            print(f"\n⚠ No transformer found - Module 2 may not load correctly")
        
        print(f"{'='*60}\n")
        
        return has_transformer
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Module 1 checkpoint for Module 2")
    parser.add_argument(
        '--module1_ckpt',
        type=str,
        required=True,
        help='Path to Module 1 (DenseRegModel) checkpoint'
    )
    
    args = parser.parse_args()
    
    verify_module1_loading(args.module1_ckpt)
