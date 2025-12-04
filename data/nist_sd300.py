"""
NIST SD300 dataset loader.

Structure:
    datasets/NIST_SD300/images/500/png/
        NIST300_rolled/     # Rolled impressions (8,871 images)
        plain/              # Plain impressions (10,564 images)
    
    File naming: {SUBJECT}_{IMPRESSION}_{PPI}_{FRGP}.png
    - SUBJECT: Participant ID (e.g., 00001000) - used as finger_local_id
    - IMPRESSION: "roll" or "plain" - impression type
    - PPI: 500 (resolution)
    - FRGP: Fingerprint position code (01-10 for roll, 02-14 for plain) - used as impression_id
    
    Example: 00001000_roll_500_01.png
             -> finger_local_id = 00001000 (or hash to int)
             -> impression_id = 1
             -> impression_type = "roll"
    
    Example: 00001000_plain_500_02.png
             -> finger_local_id = 00001000
             -> impression_id = 2
             -> impression_type = "plain"
"""

from typing import List, Optional
from pathlib import Path
from .base import FingerprintDataset, FingerprintEntry


class NISTSD300Dataset(FingerprintDataset):
    """
    NIST SD300 dataset loader.
    
    Structure: Multiple subjects with rolled and plain impressions.
    - Rolled: 8,871 images (FRGP 01-10)
    - Plain: 10,564 images (FRGP 02-14)
    
    According to IFViT paper:
    - NIST SD301a is used for training (240 fingers, 8 impressions)
    - SD300 may be used as additional training data
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        finger_global_id_offset: int = 0,
        impression_types: Optional[List[str]] = None,  # ["roll"], ["plain"], or ["roll", "plain"]
        ppi: int = 500  # Resolution (default 500 PPI)
    ):
        """
        Args:
            root_dir: Root directory (e.g., "/path/to/datasets/NIST_SD300")
            split: "train", "val", or "test"
            finger_global_id_offset: Starting offset for global finger IDs
            impression_types: List of impression types to load. 
                              Options: ["roll"], ["plain"], or ["roll", "plain"]
                              If None, loads both types
            ppi: Resolution in PPI (default 500)
        """
        self.impression_types = impression_types if impression_types is not None else ["roll", "plain"]
        self.ppi = ppi
        super().__init__(root_dir, split, finger_global_id_offset)
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load NIST SD300 entries."""
        entries = []
        
        # Path structure: root_dir/images/{ppi}/png/
        images_path = self.root_dir / "images" / str(self.ppi) / "png"
        
        if not images_path.exists():
            raise FileNotFoundError(
                f"NIST SD300 images directory not found: {images_path}\n"
                f"Expected structure: {self.root_dir}/images/{self.ppi}/png/"
            )
        
        # Create mapping from subject to finger_local_id for consistency
        subject_to_id = {}
        next_finger_id = 0
        
        # Load rolled impressions
        if "roll" in self.impression_types:
            rolled_path = images_path / "NIST300_rolled"
            if rolled_path.exists():
                image_files = sorted(rolled_path.glob("*.png")) + sorted(rolled_path.glob("*.PNG"))
                
                for img_path in image_files:
                    # Parse filename: {SUBJECT}_roll_{PPI}_{FRGP}.png
                    # Example: 00001000_roll_500_01.png
                    stem = img_path.stem
                    parts = stem.split('_')
                    
                    if len(parts) >= 4 and parts[1] == "roll" and parts[2] == str(self.ppi):
                        try:
                            subject_str = parts[0]  # e.g., "00001000"
                            frgp_str = parts[3]     # e.g., "01"
                            
                            # Assign consistent finger_local_id
                            if subject_str not in subject_to_id:
                                subject_to_id[subject_str] = next_finger_id
                                next_finger_id += 1
                            
                            finger_local_id = subject_to_id[subject_str]
                            impression_id = int(frgp_str)
                            
                            entries.append(FingerprintEntry(
                                path=str(img_path),
                                finger_local_id=finger_local_id,
                                impression_id=impression_id,
                                dataset_name="NIST_SD300",
                                split=self.split
                            ))
                        except (ValueError, IndexError):
                            continue
            else:
                print(f"⚠ Warning: Rolled impressions path not found: {rolled_path}")
        
        # Load plain impressions
        if "plain" in self.impression_types:
            plain_path = images_path / "plain"
            if plain_path.exists():
                image_files = sorted(plain_path.glob("*.png")) + sorted(plain_path.glob("*.PNG"))
                
                for img_path in image_files:
                    # Parse filename: {SUBJECT}_plain_{PPI}_{FRGP}.png
                    # Example: 00001000_plain_500_02.png
                    stem = img_path.stem
                    parts = stem.split('_')
                    
                    if len(parts) >= 4 and parts[1] == "plain" and parts[2] == str(self.ppi):
                        try:
                            subject_str = parts[0]  # e.g., "00001000"
                            frgp_str = parts[3]     # e.g., "02"
                            
                            # Use same mapping as rolled (same subject = same finger)
                            # If subject not seen before, assign new ID
                            if subject_str not in subject_to_id:
                                subject_to_id[subject_str] = next_finger_id
                                next_finger_id += 1
                            
                            finger_local_id = subject_to_id[subject_str]
                            impression_id = int(frgp_str)
                            
                            entries.append(FingerprintEntry(
                                path=str(img_path),
                                finger_local_id=finger_local_id,
                                impression_id=impression_id,
                                dataset_name="NIST_SD300",
                                split=self.split
                            ))
                        except (ValueError, IndexError):
                            continue
            else:
                print(f"⚠ Warning: Plain impressions path not found: {plain_path}")
        
        if len(entries) == 0:
            raise ValueError(
                f"No valid entries found in {images_path}. "
                f"Expected naming pattern: {{SUBJECT}}_{{roll|plain}}_{{PPI}}_{{FRGP}}.png"
            )
        
        return entries

