"""
Base classes for fingerprint datasets.
"""

from typing import List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FingerprintEntry:
    """Standardized entry for a fingerprint image."""
    path: str
    finger_local_id: int  # Local ID within dataset (e.g., finger 1, 2, 3...)
    impression_id: int    # Impression number (1, 2, 3...)
    dataset_name: str     # Dataset identifier (e.g., "FVC2002_DB1", "NIST_SD302a")
    split: str            # "train", "val", or "test"
    finger_global_id: Optional[int] = None  # Global ID across all datasets (for ArcFace)


class FingerprintDataset(ABC):
    """
    Abstract base class for fingerprint datasets.
    
    Each dataset (FVC2002, NIST SD301a, etc.) should inherit this and implement:
    - _load_entries(): Parse dataset structure and return list of FingerprintEntry
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        finger_global_id_offset: int = 0
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            split: "train", "val", or "test"
            finger_global_id_offset: Starting offset for global finger IDs
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.finger_global_id_offset = finger_global_id_offset
        
        # Load entries
        self.entries = self._load_entries()
        
        # Assign global IDs
        self._assign_global_ids()
    
    @abstractmethod
    def _load_entries(self) -> List[FingerprintEntry]:
        """
        Parse dataset structure and return list of FingerprintEntry.
        
        Must be implemented by each dataset class.
        """
        pass
    
    def _assign_global_ids(self):
        """Assign global finger IDs to entries."""
        finger_to_global = {}
        for entry in self.entries:
            key = (entry.dataset_name, entry.finger_local_id)
            if key not in finger_to_global:
                finger_to_global[key] = len(finger_to_global) + self.finger_global_id_offset
            entry.finger_global_id = finger_to_global[key]
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> FingerprintEntry:
        return self.entries[idx]
    
    def get_all_entries(self) -> List[FingerprintEntry]:
        """Get all entries in this dataset."""
        return self.entries
    
    def get_finger_ids(self) -> List[int]:
        """Get list of unique global finger IDs."""
        return sorted(set(e.finger_global_id for e in self.entries))

