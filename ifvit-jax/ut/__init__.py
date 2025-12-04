"""
Utility modules for IFViT-JAX.

This package contains:
- utils: General utilities (checkpointing, logging, etc.)
- enhance_and_roi: FingerNet enhancement and ROI extraction
- alignment: Image alignment using RANSAC
- preprocess_fingernet: Offline preprocessing script
- load_module1_weights: Load Module 1 weights into Module 2
- load_loftr_weights: Load LoFTR pretrained weights
- convert_loftr_checkpoint: Convert LoFTR PyTorch checkpoint to JAX
"""

from . import utils
from . import enhance_and_roi
from . import alignment
from . import load_module1_weights
from . import load_loftr_weights
from . import convert_loftr_checkpoint

# Re-export commonly used utilities
from .utils import (
    save_checkpoint,
    load_checkpoint,
    Logger,
    print_model_summary,
    count_parameters,
    load_pretrained_weights
)

from .enhance_and_roi import (
    FingerNetEnhancer,
    compute_effective_mask,
    extract_overlapped_region,
    extract_roi_from_original,
    build_matcher_inputs
)

from .alignment import (
    matches_to_points,
    estimate_homography_from_matches,
    warp_image,
    align_with_best_rotation
)

from .load_module1_weights import (
    load_module1_transformer_weights,
    verify_module1_loading
)

from .load_loftr_weights import (
    load_loftr_weights,
    merge_params,
    initialize_with_loftr
)

from .convert_loftr_checkpoint import (
    convert_checkpoint,
    load_converted_checkpoint
)

__all__ = [
    # Modules
    'utils',
    'enhance_and_roi',
    'alignment',
    'load_module1_weights',
    'load_loftr_weights',
    'convert_loftr_checkpoint',
    
    # Utils functions
    'save_checkpoint',
    'load_checkpoint',
    'Logger',
    'print_model_summary',
    'count_parameters',
    'load_pretrained_weights',
    
    # Enhancement & ROI
    'FingerNetEnhancer',
    'compute_effective_mask',
    'extract_overlapped_region',
    'extract_roi_from_original',
    'build_matcher_inputs',
    
    # Alignment
    'matches_to_points',
    'estimate_homography_from_matches',
    'warp_image',
    'align_with_best_rotation',
    
    # Module 1 weight loading
    'load_module1_transformer_weights',
    'verify_module1_loading',
    
    # LoFTR weight loading
    'load_loftr_weights',
    'merge_params',
    'initialize_with_loftr',
    
    # LoFTR checkpoint conversion
    'convert_checkpoint',
    'load_converted_checkpoint',
]

