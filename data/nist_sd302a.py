"""
NIST SD302a dataset loader.

Structure:
    datasets/NIST_SD302a/images/challengers/{DEVICE}/roll/png/
    
    Where DEVICE is one of: A, B, C, D, E, F, G, H (8 devices)
    
    File naming: {SUBJECT}_{DEVICE}_roll_{FRGP}.png
    - SUBJECT: Participant ID (e.g., 00002303) - used as finger_local_id
    - DEVICE: Device code (A-H) - used for grouping
    - FRGP: Fingerprint position code (01-10) - used as impression_id
    
    Example: 00002303_A_roll_01.png
             -> finger_local_id = 00002303 (or hash to int)
             -> impression_id = 1
             -> device = A
"""

from typing import List, Tuple, Optional
from pathlib import Path
import random
from .base import FingerprintDataset, FingerprintEntry


class NISTSD302aDataset(FingerprintDataset):
    """
    NIST SD302a dataset loader.
    
    Structure: 1789 fingers, 10 impressions per finger (varies by device).
    Can be split into train/val/test.
    
    According to IFViT paper:
    - Training: 193,047 pairs
    - Validation: 18,316 pairs
    - Testing: 28,900 pairs
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        finger_global_id_offset: int = 0,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),  # train, val, test
        devices: Optional[List[str]] = None  # If None, use all devices (A-H)
    ):
        """
        Args:
            root_dir: Root directory (e.g., "/path/to/datasets/NIST_SD302a")
            split: "train", "val", or "test"
            finger_global_id_offset: Starting offset for global finger IDs
            split_ratio: (train_ratio, val_ratio, test_ratio) for splitting by finger
            devices: List of device codes to use (e.g., ["A", "B", "C"]). If None, use all A-H
        """
        self.split_ratio = split_ratio
        self.devices = devices if devices is not None else ["A", "B", "C", "D", "E", "F", "G", "H"]
        super().__init__(root_dir, split, finger_global_id_offset)
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load NIST SD302a entries."""
        all_entries = []
        
        # Path structure: root_dir/images/challengers/{DEVICE}/roll/png/
        images_path = self.root_dir / "images" / "challengers"
        
        if not images_path.exists():
            raise FileNotFoundError(
                f"NIST SD302a images directory not found: {images_path}\n"
                f"Expected structure: {self.root_dir}/images/challengers/{{DEVICE}}/roll/png/"
            )
        
        # Create mapping from (device, subject) to finger_local_id for consistency
        finger_key_to_id = {}
        next_finger_id = 0
        
        # Iterate through each device
        for device in self.devices:
            device_path = images_path / device / "roll" / "png"
            
            if not device_path.exists():
                print(f"⚠ Warning: Device {device} path not found: {device_path}")
                continue
            
            # Get all PNG files
            image_files = sorted(device_path.glob("*.png")) + sorted(device_path.glob("*.PNG"))
            
            for img_path in image_files:
                # Parse filename: {SUBJECT}_{DEVICE}_roll_{FRGP}.png
                # Example: 00002303_A_roll_01.png
                stem = img_path.stem
                parts = stem.split('_')
                
                if len(parts) >= 4 and parts[1] == device and parts[2] == "roll":
                    try:
                        subject_str = parts[0]  # e.g., "00002303"
                        frgp_str = parts[3]     # e.g., "01"
                        
                        # Create unique finger key combining device and subject
                        # This ensures different devices with same subject ID are treated as different fingers
                        # Format: device_subject (e.g., "A_00002303")
                        finger_key = f"{device}_{subject_str}"
                        
                        # Assign consistent finger_local_id
                        if finger_key not in finger_key_to_id:
                            finger_key_to_id[finger_key] = next_finger_id
                            next_finger_id += 1
                        
                        finger_local_id = finger_key_to_id[finger_key]
                        
                        # Convert FRGP to impression_id
                        impression_id = int(frgp_str)
                        
                        all_entries.append(FingerprintEntry(
                            path=str(img_path),
                            finger_local_id=finger_local_id,
                            impression_id=impression_id,
                            dataset_name=f"NIST_SD302a_{device}",
                            split="train"  # Will be split below
                        ))
                    except (ValueError, IndexError) as e:
                        # Skip files that don't match naming pattern
                        continue
        
        if len(all_entries) == 0:
            raise ValueError(
                f"No valid entries found in {images_path}. "
                f"Expected naming pattern: {{SUBJECT}}_{{DEVICE}}_roll_{{FRGP}}.png"
            )
        
        # Group entries by finger (finger_local_id) for splitting
        # Split into train/val/test by finger (not by image)
        finger_to_entries = {}
        for entry in all_entries:
            fid = entry.finger_local_id
            if fid not in finger_to_entries:
                finger_to_entries[fid] = []
            finger_to_entries[fid].append(entry)
        
        unique_fingers = sorted(finger_to_entries.keys())
        random.Random(42).shuffle(unique_fingers)  # Fixed seed for reproducibility
        
        n_train = int(len(unique_fingers) * self.split_ratio[0])
        n_val = int(len(unique_fingers) * self.split_ratio[1])
        
        train_fingers = set(unique_fingers[:n_train])
        val_fingers = set(unique_fingers[n_train:n_train+n_val])
        test_fingers = set(unique_fingers[n_train+n_val:])
        
        # Assign split to entries
        for entry in all_entries:
            if entry.finger_local_id in train_fingers:
                entry.split = "train"
            elif entry.finger_local_id in val_fingers:
                entry.split = "val"
            else:
                entry.split = "test"
        
        # Return only entries for requested split
        return [e for e in all_entries if e.split == self.split]
