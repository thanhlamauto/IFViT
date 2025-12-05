"""
Utility functions for path handling in different environments (Local vs Kaggle).
"""

import os
from pathlib import Path
from typing import Union


def is_kaggle() -> bool:
    """Check if running in Kaggle environment."""
    return os.path.exists("/kaggle/working") and "KAGGLE" in os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").upper()


def get_base_dir() -> Path:
    """Get base directory for the project."""
    if is_kaggle():
        return Path("/kaggle/working/IFViT")
    else:
        # Try to find project root
        current = Path(__file__).resolve()
        # Go up from ifvit-jax/ut/path_utils.py to project root
        while current.name != "IFViT" and current.parent != current:
            current = current.parent
        if current.name == "IFViT":
            return current
        # Fallback to current working directory
        return Path.cwd()


def resolve_path(path: Union[str, Path], base_dir: Path = None) -> Path:
    """
    Resolve a path, converting relative paths to absolute based on environment.
    
    Args:
        path: Path string or Path object (can be relative or absolute)
        base_dir: Base directory (defaults to project root)
        
    Returns:
        Resolved absolute Path
        
    Examples:
        # Local
        resolve_path("./checkpoints/dense_reg") 
        # → /home/user/IFViT/checkpoints/dense_reg
        
        # Kaggle
        resolve_path("./checkpoints/dense_reg")
        # → /kaggle/working/IFViT/checkpoints/dense_reg
        
        # Absolute paths are returned as-is
        resolve_path("/kaggle/working/IFViT/checkpoints/dense_reg")
        # → /kaggle/working/IFViT/checkpoints/dense_reg
    """
    path = Path(path)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # Resolve relative to base directory
    if base_dir is None:
        base_dir = get_base_dir()
    
    return (base_dir / path).resolve()


def get_checkpoint_path(module: str = "dense_reg", base_dir: Path = None) -> Path:
    """
    Get checkpoint path for a module.
    
    Args:
        module: "dense_reg" or "matcher"
        base_dir: Base directory (defaults to project root)
        
    Returns:
        Path to checkpoint directory
    """
    if base_dir is None:
        base_dir = get_base_dir()
    
    return base_dir / "checkpoints" / module


def get_module1_checkpoint(base_dir: Path = None) -> Path:
    """Get Module 1 final checkpoint path."""
    checkpoint_dir = get_checkpoint_path("dense_reg", base_dir)
    return checkpoint_dir / "dense_reg_ckpt.pkl"


def get_module2_checkpoint(base_dir: Path = None) -> Path:
    """Get Module 2 final checkpoint path."""
    checkpoint_dir = get_checkpoint_path("matcher", base_dir)
    return checkpoint_dir / "matcher_ckpt.pkl"


# Example usage
if __name__ == "__main__":
    print(f"Running in Kaggle: {is_kaggle()}")
    print(f"Base directory: {get_base_dir()}")
    print(f"Module 1 checkpoint: {get_module1_checkpoint()}")
    print(f"Module 2 checkpoint: {get_module2_checkpoint()}")
    
    # Test path resolution
    test_paths = [
        "./checkpoints/dense_reg",
        "/kaggle/working/IFViT/checkpoints/dense_reg",
        "checkpoints/matcher"
    ]
    
    print("\nPath resolution examples:")
    for p in test_paths:
        resolved = resolve_path(p)
        print(f"  {p:50s} → {resolved}")

