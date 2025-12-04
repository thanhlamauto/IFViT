"""
FVC2002 dataset loader.

According to IFViT paper: Only DB1A, DB2A, DB3A are used for Training (7,750 pairs each).
"""

from typing import List, Optional
from pathlib import Path
from .base import FingerprintDataset, FingerprintEntry


class FVC2002Dataset(FingerprintDataset):
    """
    FVC2002 dataset loader.
    
    Structure:
        datasets/FVC2002/Dbs/Db1_a/  (training: 100 fingers × 8 impressions = 800 files)
        datasets/FVC2002/Dbs/Db2_a/  (training: 100 fingers × 8 impressions = 800 files)
        datasets/FVC2002/Dbs/Db3_a/  (training: 100 fingers × 8 impressions = 800 files)
        datasets/FVC2002/Dbs/Db1_b/  (testing: 10 fingers × 8 impressions = 80 files)
        datasets/FVC2002/Dbs/Db2_b/  (testing: 10 fingers × 8 impressions = 80 files)
        datasets/FVC2002/Dbs/Db3_b/  (testing: 10 fingers × 8 impressions = 80 files)
    
    File format: .tif
    Naming: {finger_id}_{impression_id}.tif (e.g., 101_1.tif, 101_2.tif, ..., 110_8.tif)
    
    According to IFViT paper: Only DB1A, DB2A, DB3A are used (for training).
    """
    
    def __init__(
        self,
        root_dir: str,
        db_name: str = "Db1_a",  # "Db1_a", "Db2_a", or "Db3_a" (default: Db1_a as per paper)
        split: Optional[str] = None,  # Auto-detect from folder name if None
        finger_global_id_offset: int = 0
    ):
        """
        Args:
            root_dir: Root directory (e.g., "/path/to/datasets/FVC2002")
            db_name: Database folder name. Default "Db1_a" (as per IFViT paper).
                     Options: "Db1_a", "Db2_a", "Db3_a" (training sets used in paper)
            split: "train" or "test". If None, auto-detect from folder name (_a = train, _b = test)
            finger_global_id_offset: Starting offset for global finger IDs
        """
        self.db_name = db_name
        
        # Auto-detect split from folder name if not provided
        if split is None:
            if db_name.lower().endswith('_a'):
                split = "train"  # _a folders are for training (used in paper)
            elif db_name.lower().endswith('_b'):
                split = "test"  # _b folders are for testing (not used in paper)
            else:
                split = "train"  # Default to train for FVC2002
        
        super().__init__(root_dir, split, finger_global_id_offset)
    
    def _load_entries(self) -> List[FingerprintEntry]:
        """Load FVC2002 entries."""
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
                f"FVC2002 dataset folder not found: {db_path}\n"
                f"Expected structure: {self.root_dir}/Dbs/{self.db_name}/\n"
                f"Available folders: {list(dbs_path.iterdir()) if dbs_path.exists() else 'N/A'}"
            )
        
        # FVC2002 naming: {finger_id}_{impression_id}.tif
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
                    
                    dataset_name = f"FVC2002_{self.db_name}"
                    
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
