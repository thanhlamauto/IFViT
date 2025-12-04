"""
Dataset loader for FPGAN synthetic fingerprints.

Directory structure (from the user):

    root/
        ID_XXXXXXXX/
            ID_XXXXXXXX_im000_instanceXXXXXXX.png
            ID_XXXXXXXX_im001_instanceXXXXXXX.png
            ...
            ID_XXXXXXXX_im009_instanceXXXXXXX.png

- Level 1 directory name (ID_XXXXXXXX) is the subject / identity label.
- Inside each ID directory, each PNG is one impression.

This dataset is intended ONLY for pre-training the Matching / fixed-length
representation module (Module 2), similar to PrintsGAN in the paper.
It is NOT used in the "paper splits" helpers (which are for real datasets).
"""

import re
from pathlib import Path
from typing import List

from .base import FingerprintEntry, FingerprintDataset


ID_DIR_PATTERN = re.compile(r"^ID_(\d{7,8})$")  # e.g. ID_0000003, ID_0001000
FILE_PATTERN = re.compile(
    r"^ID_(\d{7,8})_im(\d{3})_instance(\d{7})\.png$", re.IGNORECASE
)


class FPGANDataset(FingerprintDataset):
    """
    Loader for FPGAN synthetic fingerprints.

    We treat:
        - folder name ID_XXXXXXXX  → finger_local_id
        - imXXX                    → impression_id (int from 0..999)

    `split` is user-defined: because FPGAN as provided has no official
    train/val/test split, you typically set split="train" and use this
    only for pre-training.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        finger_global_id_offset: int = 0,
        dataset_name: str = "FPGAN",
    ):
        self.dataset_name = dataset_name
        super().__init__(root_dir=root_dir, split=split, finger_global_id_offset=finger_global_id_offset)

    def _load_entries(self) -> List[FingerprintEntry]:
        root = self.root_dir
        if not root.exists():
            raise FileNotFoundError(f"FPGAN root directory does not exist: {root}")

        entries: List[FingerprintEntry] = []

        # Each subdirectory under root is one ID_XXXXXXXX folder
        for id_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            m_id = ID_DIR_PATTERN.match(id_dir.name)
            if not m_id:
                # Skip unexpected folders
                continue

            # Use numeric part of ID_XXXXXXXX as finger_local_id
            local_id = int(m_id.group(1))

            # Inside each ID folder: many PNG files
            for img_path in sorted(id_dir.glob("*.png")):
                m_file = FILE_PATTERN.match(img_path.name)
                if not m_file:
                    # Skip files not matching naming convention
                    continue

                # Sanity-check ID consistency between folder and filename
                file_id_str, im_str, _instance_str = m_file.groups()
                if file_id_str != m_id.group(1):
                    # Inconsistent naming → skip to be safe
                    continue

                impression_id = int(im_str)  # im000 → 0, im001 → 1, ...

                entries.append(
                    FingerprintEntry(
                        path=str(img_path),
                        finger_local_id=local_id,
                        impression_id=impression_id,
                        dataset_name=self.dataset_name,
                        split=self.split,
                    )
                )

        return entries


