"""
Convert LoFTR PyTorch checkpoint to JAX/NumPy format.

This script loads a LoFTR checkpoint and converts the LocalFeatureTransformer
weights to a format compatible with our Flax implementation.

Usage:
    python convert_loftr_checkpoint.py \
        --pytorch_ckpt outdoor_ds.ckpt \
        --output loftr_transformer.npz
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict


def map_pytorch_key_to_flax(pytorch_key: str, layer_idx: int, sublayer_idx: int) -> str:
    """
    Map PyTorch parameter key to Flax parameter path.
    
    Args:
        pytorch_key: PyTorch state dict key (e.g., "loftr_coarse.layers.0.q_proj.weight")
        layer_idx: Layer index in the transformer
        sublayer_idx: Sublayer index (0 for feat0, 1 for feat1 in self-attn)
        
    Returns:
        Flax parameter path (e.g., "layers.0.0/q_proj/kernel")
    """
    # Remove prefix (e.g., "loftr_coarse." or "loftr_coarse.coarse_transformer.")
    # Extract the actual parameter name
    
    # Example keys:
    # "loftr_coarse.layers.0.q_proj.weight" -> "layers.0.0/q_proj/kernel"
    # "loftr_coarse.layers.0.norm1.weight" -> "layers.0.0/norm1/scale"
    
    parts = pytorch_key.split('.')
    
    # Find the parameter name (last part)
    param_name = parts[-1]  # e.g., "weight", "bias"
    param_type = parts[-2] if len(parts) >= 2 else None  # e.g., "q_proj", "norm1", "0" (for mlp.0)
    
    # Special handling for MLP: "loftr_coarse.layers.0.mlp.0.weight" -> "layers.0.0/mlp.0/kernel"
    if 'mlp' in pytorch_key:
        # Find mlp index (e.g., "0" or "2" in "mlp.0.weight")
        mlp_idx = None
        for i, part in enumerate(parts):
            if part == 'mlp' and i + 1 < len(parts):
                mlp_idx = parts[i + 1]  # e.g., "0" or "2"
                break
        
        if mlp_idx is not None:
            if param_name == 'weight':
                return f'layers.{layer_idx}.{sublayer_idx}/mlp.{mlp_idx}/kernel'
            elif param_name == 'bias':
                return f'layers.{layer_idx}.{sublayer_idx}/mlp.{mlp_idx}/bias'
    
    # Map other parameter names
    if param_name == 'weight':
        if param_type in ['q_proj', 'k_proj', 'v_proj', 'merge']:
            # Linear layer weights: transpose needed
            return f'layers.{layer_idx}.{sublayer_idx}/{param_type}/kernel'
        elif param_type in ['norm1', 'norm2']:
            # LayerNorm weights: no transpose
            return f'layers.{layer_idx}.{sublayer_idx}/{param_type}/scale'
    elif param_name == 'bias':
        if param_type in ['q_proj', 'k_proj', 'v_proj', 'merge']:
            return f'layers.{layer_idx}.{sublayer_idx}/{param_type}/bias'
        elif param_type in ['norm1', 'norm2']:
            return f'layers.{layer_idx}.{sublayer_idx}/{param_type}/bias'
    
    return None


def convert_checkpoint(pytorch_ckpt_path: str, output_path: str, prefix: str = None):
    """
    Convert PyTorch LoFTR checkpoint to NumPy format for Flax.
    
    Args:
        pytorch_ckpt_path: Path to PyTorch checkpoint (.ckpt)
        output_path: Path to save NumPy checkpoint (.npz)
        prefix: Optional prefix to filter keys (e.g., 'loftr_coarse.coarse_transformer')
    """
    print(f"Loading PyTorch checkpoint: {pytorch_ckpt_path}")
    
    # Load checkpoint
    ckpt = torch.load(pytorch_ckpt_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    print(f"Total keys in checkpoint: {len(state_dict)}")
    
    # Filter for transformer keys
    if prefix:
        transformer_keys = {k: v for k, v in state_dict.items() if prefix in k}
        print(f"Filtered to {len(transformer_keys)} keys with prefix '{prefix}'")
        
        if len(transformer_keys) == 0:
            print(f"\n⚠ Warning: No keys found with prefix '{prefix}'")
            print("  Available keys (first 20):")
            for i, k in enumerate(list(state_dict.keys())[:20]):
                print(f"    {k}")
            print("\n  Trying auto-detect...")
            prefix = None  # Fall back to auto-detect
    else:
        # Try to auto-detect transformer keys
        possible_prefixes = [
            'loftr_coarse',  # Most common format
            'loftr_coarse.coarse_transformer',
            'coarse_transformer',
            'local_transformer',
            'transformer'
        ]
        
        transformer_keys = {}
        for p in possible_prefixes:
            transformer_keys = {k: v for k, v in state_dict.items() if p in k}
            if transformer_keys:
                print(f"Auto-detected prefix: '{p}' ({len(transformer_keys)} keys)")
                break
    
    # If still no keys found, show all keys and let user choose
    if not transformer_keys:
        print("\n⚠ Warning: No transformer keys found with common prefixes")
        print("  Available keys (first 30):")
        for i, k in enumerate(list(state_dict.keys())[:30]):
            print(f"    {k}")
        if len(state_dict) > 30:
            print(f"  ... and {len(state_dict) - 30} more keys")
        print("\n  Please specify --prefix manually based on the keys above")
        return
    
    # Convert to NumPy and map keys
    print("\nConverting weights to NumPy format...")
    np_params = OrderedDict()
    
    for pytorch_key, pytorch_tensor in transformer_keys.items():
        # Extract layer information from key
        # This is checkpoint-specific and may need adjustment
        
        # Example key: "loftr_coarse.coarse_transformer.layers.0.linear_q.weight"
        parts = pytorch_key.split('.')
        
        try:
            # Find layer index
            layer_idx = None
            sublayer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    if i + 2 < len(parts) and parts[i + 2].isdigit():
                        sublayer_idx = int(parts[i + 2])
                    break
            
            if layer_idx is None:
                print(f"  ⚠ Skipping key (no layer index): {pytorch_key}")
                continue
            
            # Default sublayer index
            if sublayer_idx is None:
                sublayer_idx = 0
            
            # Map key
            flax_key = map_pytorch_key_to_flax(pytorch_key, layer_idx, sublayer_idx)
            
            if flax_key is None:
                print(f"  ⚠ No mapping for: {pytorch_key}")
                continue
            
            # Convert tensor to numpy
            np_weight = pytorch_tensor.detach().cpu().numpy()
            
            # Transpose weight matrices (PyTorch uses [out, in], Flax uses [in, out])
            # Only transpose for linear layer kernels (not biases, not LayerNorm)
            if 'kernel' in flax_key and len(np_weight.shape) == 2:
                np_weight = np_weight.T
            
            np_params[flax_key] = np_weight
            print(f"  ✓ {pytorch_key} -> {flax_key} {np_weight.shape}")
            
        except Exception as e:
            print(f"  ✗ Error processing {pytorch_key}: {e}")
            continue
    
    # Save to npz
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, **np_params)
    print(f"✓ Saved {len(np_params)} parameters")
    
    # Print summary
    total_params = sum(v.size for v in np_params.values())
    print(f"\nSummary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Output file: {output_path}")
    
    return np_params


def load_converted_checkpoint(npz_path: str):
    """
    Load converted checkpoint and display info.
    
    Args:
        npz_path: Path to .npz file
    """
    print(f"Loading converted checkpoint: {npz_path}")
    data = np.load(npz_path)
    
    print(f"\nLoaded {len(data.files)} parameters:")
    for key in sorted(data.files):
        print(f"  {key}: {data[key].shape}")
    
    total_params = sum(data[k].size for k in data.files)
    print(f"\nTotal parameters: {total_params:,}")
    
    return dict(data)


def main():
    parser = argparse.ArgumentParser(description="Convert LoFTR checkpoint to JAX format")
    parser.add_argument(
        '--pytorch_ckpt',
        type=str,
        required=True,
        help='Path to PyTorch LoFTR checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='loftr_transformer.npz',
        help='Output path for NumPy checkpoint (.npz)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default=None,
        help='Key prefix to filter (e.g., loftr_coarse.coarse_transformer)'
    )
    parser.add_argument(
        '--verify',
        type=str,
        default=None,
        help='Verify a converted checkpoint by loading it'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        load_converted_checkpoint(args.verify)
    else:
        if not Path(args.pytorch_ckpt).exists():
            print(f"✗ PyTorch checkpoint not found: {args.pytorch_ckpt}")
            print("\nTo download LoFTR checkpoints:")
            print("  wget https://github.com/zju3dv/LoFTR/releases/download/v1.0/outdoor_ds.ckpt")
            print("  wget https://github.com/zju3dv/LoFTR/releases/download/v1.0/indoor_ds.ckpt")
            return
        
        convert_checkpoint(args.pytorch_ckpt, args.output, args.prefix)


if __name__ == '__main__':
    main()
