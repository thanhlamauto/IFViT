"""
NIST SD301a dataset loader.

Structure:
    datasets/NIST_SD301a/images/friction-ridge/{DEVICE}/{RESOLUTION}/{CAPTURE}/png/
    
    Where:
    - DEVICE: dryrun-A, dryrun-B, ..., dryrun-P (14 devices)
    - RESOLUTION: 500, 1000, or variable
    - CAPTURE: roll, plain, slap, palm, slap-segmented, palm-segmented
    
    File naming: {SUBJECT}_{ENCOUNTER}_{DEVICE}_{RESOLUTION}_{CAPTURE}_{FRGP}.png
    - SUBJECT: Participant ID (e.g., 00002223) - used as finger_local_id
    - ENCOUNTER: Encounter number (usually 01)
    - DEVICE: Device code (dryrun-A, dryrun-B, etc.)
    - RESOLUTION: 500, 1000, or variable
    - CAPTURE: roll, plain, slap, palm, slap-segmented, palm-segmented
    - FRGP: Fingerprint position code (01-14) - used as impression_id
    
    Example: 00002223_01_dryrun-A_500_roll_01.png
             -> finger_local_id = 00002223 (or hash to int)
             -> impression_id = 1
             -> device = dryrun-A
             -> capture_type = roll
"""

from typing import List, Optional
from pathlib import Path
from .base import FingerprintDataset, FingerprintEntry


class NISTSD301aDataset(FingerprintDataset):
    """
    NIST SD301a dataset loader.
    
    Structure: Multiple devices with different capture types (roll, plain, slap, palm, etc.).
    - dryrun-A, B: roll (240 images each)
    - dryrun-C, F, G, K, P: slap + slap-segmented
    - dryrun-D, E, J, L, M, N: plain
    - dryrun-H: palm + palm-segmented + slap + slap-segmented (500 and 1000 PPI)
    
    According to IFViT paper:
    - NIST SD301a is used for training (240 fingers, 8 impressions)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        finger_global_id_offset: int = 0,
        devices: Optional[List[str]] = None,  # If None, use all devices
        capture_types: Optional[List[str]] = None,  # ["roll"], ["plain"], ["slap"], etc.
        resolutions: Optional[List[str]] = None,  # ["500"], ["1000"], ["variable"]
        use_segmented: bool = True  # Include segmented images (slap-segmented, palm-segmented)
    ):
        """
        Args:
            root_dir: Root directory (e.g., "/path/to/datasets/NIST_SD301a")
            split: "train", "val", or "test"
            finger_global_id_offset: Starting offset for global finger IDs
            devices: List of device codes to use (e.g., ["dryrun-A", "dryrun-B"]). 
                     If None, use all available devices
            capture_types: List of capture types to load.
                           Options: ["roll"], ["plain"], ["slap"], ["palm"], 
                                   ["slap-segmented"], ["palm-segmented"]
                           If None, loads all available capture types
            resolutions: List of resolutions to use (e.g., ["500"], ["1000"]).
                         If None, uses all available resolutions
            use_segmented: If True, includes segmented images (slap-segmented, palm-segmented)
        """
        self.devices = devices
        self.capture_types = capture_types
        self.resolutions = resolutions
        self.use_segmented = use_segmented
        super().__init__(root_dir, split, finger_global_id_offset)
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load NIST SD301a entries."""
        entries = []
        
        # Path structure: root_dir/images/friction-ridge/{DEVICE}/{RESOLUTION}/{CAPTURE}/png/
        images_path = self.root_dir / "images" / "friction-ridge"
        
        if not images_path.exists():
            raise FileNotFoundError(
                f"NIST SD301a images directory not found: {images_path}\n"
                f"Expected structure: {self.root_dir}/images/friction-ridge/{{DEVICE}}/{{RESOLUTION}}/{{CAPTURE}}/png/"
            )
        
        # Get available devices
        available_devices = sorted([d.name for d in images_path.iterdir() if d.is_dir()])
        
        if self.devices is None:
            devices_to_use = available_devices
        else:
            devices_to_use = [d for d in self.devices if d in available_devices]
            if len(devices_to_use) == 0:
                raise ValueError(
                    f"None of the specified devices {self.devices} found. "
                    f"Available devices: {available_devices}"
                )
        
        # Create mapping from (device, subject) to finger_local_id for consistency
        subject_to_id = {}
        next_finger_id = 0
        
        # Iterate through each device
        for device in devices_to_use:
            device_path = images_path / device
            
            if not device_path.exists():
                print(f"⚠ Warning: Device {device} path not found: {device_path}")
                continue
            
            # Get available resolutions for this device
            available_resolutions = sorted([d.name for d in device_path.iterdir() if d.is_dir()])
            
            if self.resolutions is None:
                resolutions_to_use = available_resolutions
            else:
                resolutions_to_use = [r for r in self.resolutions if r in available_resolutions]
            
            # Iterate through each resolution
            for resolution in resolutions_to_use:
                resolution_path = device_path / resolution
                
                if not resolution_path.exists():
                    continue
                
                # Get available capture types for this resolution
                available_captures = sorted([d.name for d in resolution_path.iterdir() if d.is_dir()])
                
                # Filter capture types
                captures_to_load = []
                for capture in available_captures:
                    # Skip segmented if not requested
                    if not self.use_segmented and capture in ["slap-segmented", "palm-segmented"]:
                        continue
                    
                    # Filter by capture_types if specified
                    if self.capture_types is not None:
                        if capture not in self.capture_types:
                            continue
                    
                    captures_to_load.append(capture)
                
                # Load images from each capture type
                for capture_type in captures_to_load:
                    capture_path = resolution_path / capture_type / "png"
                    
                    if not capture_path.exists():
                        continue
                    
                    # Get all PNG files
                    image_files = sorted(capture_path.glob("*.png")) + sorted(capture_path.glob("*.PNG"))
                    
                    for img_path in image_files:
                        # Parse filename: {SUBJECT}_{ENCOUNTER}_{DEVICE}_{RESOLUTION}_{CAPTURE}_{FRGP}.png
                        # Example: 00002223_01_dryrun-A_500_roll_01.png
                        stem = img_path.stem
                        parts = stem.split('_')
                        
                        # Expected format: SUBJECT_ENCOUNTER_DEVICE_RESOLUTION_CAPTURE_FRGP
                        # Minimum 6 parts
                        if len(parts) >= 6:
                            try:
                                subject_str = parts[0]  # e.g., "00002223"
                                encounter_str = parts[1]  # e.g., "01"
                                device_str = parts[2]  # e.g., "dryrun-A"
                                resolution_str = parts[3]  # e.g., "500"
                                capture_str = parts[4]  # e.g., "roll"
                                frgp_str = parts[5]  # e.g., "01"
                                
                                # Verify device and capture match path
                                if device_str != device or capture_str != capture_type:
                                    continue
                                
                                # Create unique finger key combining device and subject
                                # This ensures different devices with same subject ID are treated as different fingers
                                finger_key = f"{device}_{subject_str}"
                                
                                # Assign consistent finger_local_id
                                if finger_key not in subject_to_id:
                                    subject_to_id[finger_key] = next_finger_id
                                    next_finger_id += 1
                                
                                finger_local_id = subject_to_id[finger_key]
                                impression_id = int(frgp_str)
                                
                                entries.append(FingerprintEntry(
                                    path=str(img_path),
                                    finger_local_id=finger_local_id,
                                    impression_id=impression_id,
                                    dataset_name=f"NIST_SD301a_{device}_{capture_type}",
                                    split=self.split
                                ))
                            except (ValueError, IndexError):
                                # Skip files that don't match naming pattern
                                continue
        
        if len(entries) == 0:
            raise ValueError(
                f"No valid entries found in {images_path}. "
                f"Expected naming pattern: {{SUBJECT}}_{{ENCOUNTER}}_{{DEVICE}}_{{RESOLUTION}}_{{CAPTURE}}_{{FRGP}}.png"
            )
        
        return entries
