"""
FVC2004 dataset loader.

According to IFViT paper: Only DB1A, DB2A, DB3A are used for Testing (7,750 pairs each).
"""

from typing import List, Optional
from pathlib import Path
from .base import FingerprintDataset, FingerprintEntry


class FVC2004Dataset(FingerprintDataset):
    """
    FVC2004 dataset loader.
    
    Structure:
        datasets/FVC2004/Dbs/DB1_A/  (testing: 100 fingers × 8 impressions = 800 files)
        datasets/FVC2004/Dbs/DB2_A/  (testing: 100 fingers × 8 impressions = 800 files)
        datasets/FVC2004/Dbs/DB3_A/  (testing: 100 fingers × 8 impressions = 800 files)
    
    File format: .tif
    Naming: {finger_id}_{impression_id}.tif (e.g., 101_1.tif, 101_2.tif, ..., 110_8.tif)
    
    According to IFViT paper: Only DB1A, DB2A, DB3A are used (for testing).
    """
    
    def __init__(
        self,
        root_dir: str,
        db_name: str = "DB1_A",  # "DB1_A", "DB2_A", or "DB3_A" (default: DB1_A as per paper)
        split: Optional[str] = None,  # Auto-detect from folder name if None
        finger_global_id_offset: int = 0
    ):
        """
        Args:
            root_dir: Root directory (e.g., "/path/to/datasets/FVC2004")
            db_name: Database folder name. Default "DB1_A" (as per IFViT paper).
                     Options: "DB1_A", "DB2_A", "DB3_A"
            split: "train" or "test". If None, auto-detect from folder name (_A = test for FVC2004)
            finger_global_id_offset: Starting offset for global finger IDs
        """
        self.db_name = db_name
        
        # Auto-detect split from folder name if not provided
        # FVC2004 _A folders are used for testing according to paper
        if split is None:
            if db_name.upper().endswith('_A'):
                split = "test"  # FVC2004 _A folders are for testing
            elif db_name.upper().endswith('_B'):
                split = "test"  # _B folders also exist but not used in paper
            else:
                split = "test"  # Default to test for FVC2004
        
        super().__init__(root_dir, split, finger_global_id_offset)
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load FVC2004 entries."""
        entries = []
        
        # Construct path: root_dir/Dbs/db_name/
        dbs_path = self.root_dir / "Dbs"
        if not dbs_path.exists():
            # Try alternative: maybe root_dir is already pointing to Dbs/
            dbs_path = self.root_dir
            db_path = dbs_path / self.db_name
        else:
            db_path = dbs_path / self.db_name
        
        if not db_path.exists():
            raise FileNotFoundError(
                f"FVC2004 dataset folder not found: {db_path}\n"
                f"Expected structure: {self.root_dir}/Dbs/{self.db_name}/\n"
                f"Available folders: {list(dbs_path.iterdir()) if dbs_path.exists() else 'N/A'}"
            )
        
        # FVC2004 naming: {finger_id}_{impression_id}.tif
        # e.g., 101_1.tif, 101_2.tif, ..., 110_8.tif
        image_files = sorted(db_path.glob("*.tif")) + sorted(db_path.glob("*.TIF"))
        
        if len(image_files) == 0:
            raise ValueError(f"No .tif files found in {db_path}")
        
        for img_path in image_files:
            # Parse filename: e.g., "101_1.tif" -> finger=101, impression=1
            stem = img_path.stem
            parts = stem.split('_')
            
            if len(parts) >= 2:
                try:
                    finger_local_id = int(parts[0])
                    impression_id = int(parts[1])
                    
                    dataset_name = f"FVC2004_{self.db_name}"
                    
                    entries.append(FingerprintEntry(
                        path=str(img_path),
                        finger_local_id=finger_local_id,
                        impression_id=impression_id,
                        dataset_name=dataset_name,
                        split=self.split
                    ))
                except ValueError:
                    # Skip files that don't match naming pattern
                    continue
        
        if len(entries) == 0:
            raise ValueError(
                f"No valid entries found in {db_path}. "
                f"Expected naming pattern: {{finger_id}}_{{impression_id}}.tif"
            )
        
        return entries

