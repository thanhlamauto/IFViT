"""
NIST SD4 dataset loader.

Structure:
    datasets/NIST_SD4/figs_0/  # Split 0 (500 PNG images)
    datasets/NIST_SD4/figs_1/  # Split 1 (500 PNG images)
    
    File naming: {PREFIX}{SUBJECT_ID}_{FINGER_POSITION}.png
    - PREFIX: 'f' or 's' (capture type)
    - SUBJECT_ID: Participant ID (e.g., 0001, 0002, ..., 0250 in figs_0; 0251+ in figs_1)
    - FINGER_POSITION: Finger position code (01-10) - used as impression_id
    
    Example: f0001_01.png
             -> prefix = 'f'
             -> finger_local_id = 1 (from SUBJECT_ID)
             -> impression_id = 1 (from FINGER_POSITION)
    
    Example: s0001_02.png
             -> prefix = 's'
             -> finger_local_id = 1 (same subject, different prefix)
             -> impression_id = 2
    
    According to IFViT paper:
    - NIST SD4 is used for testing (2000 fingers, 2 impressions per finger - rolled)
"""

from typing import List, Optional
from pathlib import Path
from .base import FingerprintDataset, FingerprintEntry


class NISTSD4Dataset(FingerprintDataset):
    """
    NIST SD4 dataset loader.
    
    Structure: 2000 fingers, 2 impressions per finger (rolled).
    Usually used for testing only.
    
    Dataset splits:
    - figs_0: 500 PNG images (250 'f', 250 's')
    - figs_1: 500 PNG images (250 'f', 250 's')
    - Total: 1000 fingerprint images
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "test",  # Default to test as per paper
        finger_global_id_offset: int = 0,
        dataset_splits: Optional[List[str]] = None,  # ["figs_0"], ["figs_1"], or ["figs_0", "figs_1"]
        prefixes: Optional[List[str]] = None  # ["f"], ["s"], or ["f", "s"]
    ):
        """
        Args:
            root_dir: Root directory (e.g., "/path/to/datasets/NIST_SD4")
            split: "train", "val", or "test" (default: "test" as per paper)
            finger_global_id_offset: Starting offset for global finger IDs
            dataset_splits: List of dataset splits to load (e.g., ["figs_0", "figs_1"]).
                           If None, loads all available splits
            prefixes: List of prefixes to load (e.g., ["f"], ["s"], or ["f", "s"]).
                     If None, loads all prefixes
        """
        self.dataset_splits = dataset_splits if dataset_splits is not None else ["figs_0", "figs_1"]
        self.prefixes = prefixes if prefixes is not None else ["f", "s"]
        super().__init__(root_dir, split, finger_global_id_offset)
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load NIST SD4 entries."""
        entries = []
        
        # Create mapping from (prefix, subject_id) to finger_local_id for consistency
        subject_to_id = {}
        next_finger_id = 0
        
        # Iterate through each dataset split (figs_0, figs_1)
        for dataset_split in self.dataset_splits:
            split_path = self.root_dir / dataset_split
            
            if not split_path.exists():
                print(f"⚠ Warning: Dataset split {dataset_split} not found: {split_path}")
                continue
            
            # Get all PNG files
            image_files = sorted(split_path.glob("*.png")) + sorted(split_path.glob("*.PNG"))
            
            for img_path in image_files:
                # Parse filename: {PREFIX}{SUBJECT_ID}_{FINGER_POSITION}.png
                # Example: f0001_01.png or s0001_02.png
                stem = img_path.stem
                
                # Check if filename starts with one of the requested prefixes
                prefix = stem[0] if len(stem) > 0 else None
                if prefix not in self.prefixes:
                    continue
                
                # Parse: remove prefix, then split by '_'
                # f0001_01 -> ['0001', '01']
                # s0001_02 -> ['0001', '02']
                remaining = stem[1:]  # Remove prefix
                parts = remaining.split('_')
                
                if len(parts) >= 2:
                    try:
                        subject_str = parts[0]  # e.g., "0001"
                        finger_pos_str = parts[1]  # e.g., "01"
                        
                        # Convert SUBJECT_ID to integer
                        subject_id = int(subject_str)
                        
                        # Create unique finger key combining prefix and subject
                        # This ensures 'f' and 's' prefixes for same subject are treated as different fingers
                        # (or same finger depending on interpretation - adjust if needed)
                        finger_key = f"{prefix}_{subject_id}"
                        
                        # Assign consistent finger_local_id
                        if finger_key not in subject_to_id:
                            subject_to_id[finger_key] = next_finger_id
                            next_finger_id += 1
                        
                        finger_local_id = subject_to_id[finger_key]
                        impression_id = int(finger_pos_str)
                        
                        entries.append(FingerprintEntry(
                            path=str(img_path),
                            finger_local_id=finger_local_id,
                            impression_id=impression_id,
                            dataset_name=f"NIST_SD4_{dataset_split}_{prefix}",
                            split=self.split
                        ))
                    except (ValueError, IndexError):
                        # Skip files that don't match naming pattern
                        continue
        
        if len(entries) == 0:
            raise ValueError(
                f"No valid entries found in {self.root_dir}. "
                f"Expected naming pattern: {{PREFIX}}{{SUBJECT_ID}}_{{FINGER_POSITION}}.png"
            )
        
        return entries
