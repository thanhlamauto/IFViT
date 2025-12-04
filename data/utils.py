"""
Dataset combination utilities.
"""

from typing import List, Dict, Tuple
from .base import FingerprintDataset, FingerprintEntry


def combine_datasets(
    datasets: List[FingerprintDataset],
    assign_global_ids: bool = True
) -> List[FingerprintEntry]:
    """
    Combine multiple datasets into a single list of entries.
    
    Args:
        datasets: List of FingerprintDataset instances
        assign_global_ids: If True, reassign global IDs across all datasets
        
    Returns:
        Combined list of FingerprintEntry
    """
    all_entries = []
    
    if assign_global_ids:
        # Reassign global IDs across all datasets
        finger_to_global = {}
        global_id_counter = 0
        
        for dataset in datasets:
            for entry in dataset.get_all_entries():
                key = (entry.dataset_name, entry.finger_local_id)
                if key not in finger_to_global:
                    finger_to_global[key] = global_id_counter
                    global_id_counter += 1
                entry.finger_global_id = finger_to_global[key]
                all_entries.append(entry)
    else:
        # Use existing global IDs
        for dataset in datasets:
            all_entries.extend(dataset.get_all_entries())
    
    return all_entries


def create_global_finger_id_mapping(
    entries: List[FingerprintEntry]
) -> Dict[Tuple[str, int], int]:
    """
    Create mapping from (dataset_name, finger_local_id) to global_id.
    
    Args:
        entries: List of FingerprintEntry
        
    Returns:
        Dictionary mapping (dataset_name, finger_local_id) -> global_id
    """
    mapping = {}
    for entry in entries:
        key = (entry.dataset_name, entry.finger_local_id)
        if key not in mapping:
            mapping[key] = entry.finger_global_id
    return mapping

