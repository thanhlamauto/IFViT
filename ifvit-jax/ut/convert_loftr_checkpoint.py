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
        pytorch_key: PyTorch state dict key
        layer_idx: Layer index in the transformer
        sublayer_idx: Sublayer index (0 for feat0, 1 for feat1 in self-attn)
        
    Returns:
        Flax parameter path
    """
    # Remove prefix (e.g., "loftr_coarse.coarse_transformer.")
    # This depends on the exact checkpoint structure
    
    # Example mapping:
    # PyTorch: backbone.layer2.0.conv1.weight
    # Flax: LocalFeatureTransformer/layers.0.0/q_proj/kernel
    
    mapping = {
        'linear_q.weight': f'layers.{layer_idx}.{sublayer_idx}/q_proj/kernel',
        'linear_q.bias': f'layers.{layer_idx}.{sublayer_idx}/q_proj/bias',
        'linear_k.weight': f'layers.{layer_idx}.{sublayer_idx}/k_proj/kernel',
        'linear_k.bias': f'layers.{layer_idx}.{sublayer_idx}/k_proj/bias',
        'linear_v.weight': f'layers.{layer_idx}.{sublayer_idx}/v_proj/kernel',
        'linear_v.bias': f'layers.{layer_idx}.{sublayer_idx}/v_proj/bias',
        'linear_out.weight': f'layers.{layer_idx}.{sublayer_idx}/merge/kernel',
        'linear_out.bias': f'layers.{layer_idx}.{sublayer_idx}/merge/bias',
        'ffn.0.weight': f'layers.{layer_idx}.{sublayer_idx}/mlp.0/kernel',
        'ffn.0.bias': f'layers.{layer_idx}.{sublayer_idx}/mlp.0/bias',
        'ffn.2.weight': f'layers.{layer_idx}.{sublayer_idx}/mlp.2/kernel',
        'ffn.2.bias': f'layers.{layer_idx}.{sublayer_idx}/mlp.2/bias',
        'norm1.weight': f'layers.{layer_idx}.{sublayer_idx}/norm1/scale',
        'norm1.bias': f'layers.{layer_idx}.{sublayer_idx}/norm1/bias',
        'norm2.weight': f'layers.{layer_idx}.{sublayer_idx}/norm2/scale',
        'norm2.bias': f'layers.{layer_idx}.{sublayer_idx}/norm2/bias',
    }
    
    for pt_suffix, flax_path in mapping.items():
        if pytorch_key.endswith(pt_suffix):
            return flax_path
    
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
    else:
        # Try to auto-detect transformer keys
        possible_prefixes = [
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
        
        if not transformer_keys:
            print("⚠ Warning: No transformer keys found with common prefixes")
            print("  Available keys sample:")
            for i, k in enumerate(list(state_dict.keys())[:10]):
                print(f"    {k}")
            print("  Please specify --prefix manually")
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
            flax_key = map_pytorch_key_to_flax('.'.join(parts), layer_idx, sublayer_idx)
            
            if flax_key is None:
                print(f"  ⚠ No mapping for: {pytorch_key}")
                continue
            
            # Convert tensor to numpy
            np_weight = pytorch_tensor.numpy()
            
            # Transpose weight matrices (PyTorch uses [out, in], Flax uses [in, out])
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
