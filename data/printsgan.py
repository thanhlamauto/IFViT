"""
PrintsGAN dataset loader.
"""

from typing import List
from pathlib import Path
from .base import FingerprintDataset, FingerprintEntry


class PrintsGANDataset(FingerprintDataset):
    """
    PrintsGAN synthetic dataset.
    
    Structure: 2500 fingers, 15 impressions per finger (synthetic).
    Used for pre-training Module 2.
    """
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load PrintsGAN entries."""
        entries = []
        
        # PrintsGAN structure: usually subject folders or flat with naming
        # Adjust based on actual structure
        image_files = sorted(self.root_dir.rglob("*.png")) + sorted(self.root_dir.rglob("*.jpg"))
        
        # Placeholder parsing - implement based on actual PrintsGAN structure
        for idx, img_path in enumerate(image_files):
            finger_local_id = idx // 15 + 1
            impression_id = (idx % 15) + 1
            
            entries.append(FingerprintEntry(
                path=str(img_path),
                finger_local_id=finger_local_id,
                impression_id=impression_id,
                dataset_name="PrintsGAN",
                split=self.split
            ))
        
        return entries

