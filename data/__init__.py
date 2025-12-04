"""
Data loading, augmentation, and preprocessing for IFViT.

Modular dataset structure with separate files for each dataset type.
"""

from .base import FingerprintEntry, FingerprintDataset
from .fvc2002 import FVC2002Dataset
from .fvc2004 import FVC2004Dataset
from .nist_sd300 import NISTSD300Dataset
from .nist_sd301a import NISTSD301aDataset
from .nist_sd302a import NISTSD302aDataset
from .nist_sd4 import NISTSD4Dataset
from .pairs import DenseRegPair, MatcherPair
from .augmentation import random_corrupt_fingerprint, generate_gt_correspondences
from .loaders import (
    load_image, normalize_image,
    dense_reg_dataset, matcher_dataset,
    preprocess_batch, compute_overlap_and_rois,
)
from .utils import combine_datasets, create_global_finger_id_mapping
from .paper_splits import (
    PaperDatasetRoots,
    build_paper_train_entries,
    build_paper_val_entries,
    build_paper_test_entries,
)

__all__ = [
    # Base classes
    'FingerprintEntry',
    'FingerprintDataset',
    # Dataset classes
    'FVC2002Dataset',
    'FVC2004Dataset',
    'NISTSD300Dataset',
    'NISTSD301aDataset',
    'NISTSD302aDataset',
    'NISTSD4Dataset',
    # High-level paper splits
    'PaperDatasetRoots',
    'build_paper_train_entries',
    'build_paper_val_entries',
    'build_paper_test_entries',
    # Pair classes
    'DenseRegPair',
    'MatcherPair',
    # Augmentation
    'random_corrupt_fingerprint',
    'generate_gt_correspondences',
    # Loaders
    'load_image',
    'normalize_image',
    'dense_reg_dataset',
    'matcher_dataset',
    'preprocess_batch',
    'compute_overlap_and_rois',
    # Utilities
    'combine_datasets',
    'create_global_finger_id_mapping',
]

