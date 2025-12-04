"""
High-level helpers to build training and testing datasets following IFViT paper.

We only use real datasets that you actually have:
    - FVC2002 DB1A/DB2A/DB3A  → Training
    - NIST SD301a             → Training
    - NIST SD302a             → Training / Validation / Testing
    - FVC2004 DB1A/DB2A/DB3A  → Testing
    - NIST SD4                → Testing

MOLF and PrintsGAN are NOT used here because they are not available.
You can extend this file later to plug in FPGAN (as a replacement for PrintsGAN)
once you have that dataset.
"""

from dataclasses import dataclass
from typing import List

from .base import FingerprintEntry
from .fvc2002 import FVC2002Dataset
from .fvc2004 import FVC2004Dataset
from .nist_sd301a import NISTSD301aDataset
from .nist_sd302a import NISTSD302aDataset
from .nist_sd4 import NISTSD4Dataset
from .utils import combine_datasets


@dataclass
class PaperDatasetRoots:
    """
    Root paths for all datasets used in IFViT paper (real datasets only).

    Adjust these paths to match your local folder layout.
    """

    fvc2002: str = "datasets/FVC2002"
    fvc2004: str = "datasets/FVC2004"
    nist_sd301a: str = "datasets/NIST_SD301a"
    nist_sd302a: str = "datasets/NIST_SD302a"
    nist_sd4: str = "datasets/NIST_SD4"


def build_paper_train_entries(roots: PaperDatasetRoots) -> List[FingerprintEntry]:
    """
    Build combined training entries following IFViT paper (without MOLF / PrintsGAN).

    Includes:
        - FVC2002 DB1A, DB2A, DB3A  (Db1_a, Db2_a, Db3_a)
        - NIST SD301a               (all devices / captures by default)
        - NIST SD302a               (train split)
    """
    datasets = []

    # FVC2002: Db1_a, Db2_a, Db3_a (training)
    for db_name in ["Db1_a", "Db2_a", "Db3_a"]:
        datasets.append(FVC2002Dataset(root_dir=roots.fvc2002, db_name=db_name, split="train"))

    # NIST SD301a: use as-is (you can narrow devices/captures later if needed)
    datasets.append(
        NISTSD301aDataset(
            root_dir=roots.nist_sd301a,
            split="train",
            # Example: only 500 PPI if you want to be strict
            resolutions=["500"],
        )
    )

    # NIST SD302a: train split
    datasets.append(
        NISTSD302aDataset(
            root_dir=roots.nist_sd302a,
            split="train",
        )
    )

    # Combine and re-assign global finger IDs across all training datasets
    entries = combine_datasets(datasets, assign_global_ids=True)
    return entries


def build_paper_val_entries(roots: PaperDatasetRoots) -> List[FingerprintEntry]:
    """
    Build validation entries following IFViT paper (only NIST SD302a has explicit val split).
    """
    datasets = [
        NISTSD302aDataset(
            root_dir=roots.nist_sd302a,
            split="val",
        )
    ]
    entries = combine_datasets(datasets, assign_global_ids=True)
    return entries


def build_paper_test_entries(roots: PaperDatasetRoots) -> List[FingerprintEntry]:
    """
    Build combined testing entries following IFViT paper (without MOLF / PrintsGAN).

    Includes:
        - FVC2004 DB1A, DB2A, DB3A  (DB1_A, DB2_A, DB3_A)
        - NIST SD4                  (all splits)
        - NIST SD302a               (test split)
    """
    datasets = []

    # FVC2004: DB1_A, DB2_A, DB3_A (testing)
    for db_name in ["DB1_A", "DB2_A", "DB3_A"]:
        datasets.append(FVC2004Dataset(root_dir=roots.fvc2004, db_name=db_name, split="test"))

    # NIST SD4: all figs_0 and figs_1 (default in loader), used for testing
    datasets.append(
        NISTSD4Dataset(
            root_dir=roots.nist_sd4,
            split="test",
        )
    )

    # NIST SD302a: test split
    datasets.append(
        NISTSD302aDataset(
            root_dir=roots.nist_sd302a,
            split="test",
        )
    )

    # Combine and re-assign global finger IDs across all test datasets
    entries = combine_datasets(datasets, assign_global_ids=True)
    return entries


