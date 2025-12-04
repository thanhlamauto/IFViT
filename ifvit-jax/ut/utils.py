"""
Utility functions for IFViT training and inference.

Includes:
- Random key management
- Checkpoint saving/loading
- Logging utilities
- Metric computation
"""

import jax
import jax.numpy as jnp
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
from datetime import datetime


# ============================================================================
# Random Key Management
# ============================================================================

def make_rngs(master_key: jax.random.PRNGKey, n: int) -> List[jax.random.PRNGKey]:
    """
    Split a master random key into n sub-keys.
    
    Args:
        master_key: Master PRNGKey
        n: Number of sub-keys to generate
        
    Returns:
        List of n PRNGKeys
    """
    keys = jax.random.split(master_key, n + 1)
    return list(keys[1:])


def create_rng_dict(master_key: jax.random.PRNGKey, names: List[str]) -> Dict[str, jax.random.PRNGKey]:
    """
    Create a dictionary of random keys for different purposes.
    
    Args:
        master_key: Master PRNGKey
        names: List of key names
        
    Returns:
        Dictionary mapping names to PRNGKeys
    """
    keys = make_rngs(master_key, len(names))
    return dict(zip(names, keys))


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(filepath: str, state_dict: Dict[str, Any], metadata: Dict[str, Any] = None):
    """
    Save training state to checkpoint file.
    
    Args:
        filepath: Path to save checkpoint
        state_dict: Dictionary containing model state (params, optimizer state, etc.)
        metadata: Optional metadata (epoch, step, metrics, etc.)
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'state': state_dict,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"✓ Saved checkpoint to {filepath}")


def load_checkpoint(filepath: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load training state from checkpoint file.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        state_dict: Model state dictionary
        metadata: Checkpoint metadata
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Loaded checkpoint from {filepath}")
    
    return checkpoint['state'], checkpoint.get('metadata', {})


def load_pretrained_weights(
    checkpoint_path: str,
    target_params: Dict,
    prefix: str = None,
    strict: bool = False
) -> Dict:
    """
    Load pretrained weights into target parameters.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint
        target_params: Target parameter dictionary to update
        prefix: Optional prefix to filter loaded parameters
        strict: If True, raise error on missing/extra keys
        
    Returns:
        Updated target_params
    """
    state_dict, _ = load_checkpoint(checkpoint_path)
    
    if 'params' in state_dict:
        pretrained_params = state_dict['params']
    else:
        pretrained_params = state_dict
    
    # Filter by prefix if specified
    if prefix:
        pretrained_params = {
            k: v for k, v in pretrained_params.items()
            if k.startswith(prefix)
        }
    
    # Update target params
    for key, value in pretrained_params.items():
        if key in target_params:
            target_params[key] = value
        elif strict:
            raise KeyError(f"Key {key} not found in target params")
    
    if strict:
        for key in target_params:
            if key not in pretrained_params:
                raise KeyError(f"Target key {key} not found in pretrained params")
    
    print(f"✓ Loaded {len(pretrained_params)} pretrained parameters")
    
    return target_params


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Simple training logger."""
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        
        self.metrics_history = []
        self.start_time = time.time()
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== {experiment_name} ===\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n\n")
    
    def log(self, message: str, print_to_console: bool = True):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if print_to_console:
            print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def log_metrics(self, step: int, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics for a training step."""
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.log(f"{prefix}Step {step} | {metrics_str}")
        
        # Save to history
        self.metrics_history.append({
            'step': step,
            'metrics': metrics,
            'timestamp': time.time() - self.start_time
        })
        
        # Save metrics to JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch summary."""
        self.log(f"\n{'='*60}")
        self.log(f"Epoch {epoch} Summary")
        self.log(f"{'='*60}")
        for key, value in metrics.items():
            self.log(f"  {key}: {value:.6f}")
        self.log(f"{'='*60}\n")


def log_metrics(step: int, metrics: Dict[str, float], prefix: str = ""):
    """
    Simple metric logging function.
    
    Args:
        step: Training step
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for log message
    """
    metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    print(f"{prefix}Step {step} | {metrics_str}")


# ============================================================================
# Metric Computation
# ============================================================================

def compute_accuracy(predictions: jnp.ndarray, labels: jnp.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: (B,) predicted labels
        labels: (B,) ground-truth labels
        
    Returns:
        Accuracy as float in [0, 1]
    """
    correct = jnp.sum(predictions == labels)
    total = labels.shape[0]
    return float(correct / total)


def compute_matching_metrics(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute matching metrics (accuracy, precision, recall, etc.).
    
    Args:
        scores: (N,) similarity scores
        labels: (N,) ground-truth labels (1 for genuine, -1 for imposter)
        threshold: Decision threshold
        
    Returns:
        Dictionary with metrics
    """
    # Convert labels to binary (1 for genuine, 0 for imposter)
    labels_binary = (labels > 0).astype(jnp.int32)
    
    # Predictions based on threshold
    predictions = (scores > threshold).astype(jnp.int32)
    
    # True positives, false positives, etc.
    tp = jnp.sum((predictions == 1) & (labels_binary == 1))
    fp = jnp.sum((predictions == 1) & (labels_binary == 0))
    tn = jnp.sum((predictions == 0) & (labels_binary == 0))
    fn = jnp.sum((predictions == 0) & (labels_binary == 1))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # False accept rate (FAR) and False reject rate (FRR)
    far = fp / (fp + tn + 1e-8)
    frr = fn / (fn + tp + 1e-8)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'far': float(far),
        'frr': float(frr),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def compute_eer(scores_genuine: jnp.ndarray, scores_imposter: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        scores_genuine: (N,) scores for genuine pairs
        scores_imposter: (M,) scores for imposter pairs
        
    Returns:
        eer: Equal error rate
        threshold: Threshold at EER
    """
    # Combine scores and labels
    scores = jnp.concatenate([scores_genuine, scores_imposter])
    labels = jnp.concatenate([
        jnp.ones(len(scores_genuine)),
        jnp.zeros(len(scores_imposter))
    ])
    
    # Sort by scores
    sorted_indices = jnp.argsort(scores)
    scores_sorted = scores[sorted_indices]
    labels_sorted = labels[sorted_indices]
    
    # Compute FAR and FRR at different thresholds
    fars = []
    frrs = []
    
    for i in range(len(scores_sorted)):
        threshold = scores_sorted[i]
        
        # Predictions
        predictions = (scores >= threshold).astype(jnp.int32)
        
        # FAR and FRR
        fp = jnp.sum((predictions == 1) & (labels == 0))
        tn = jnp.sum((predictions == 0) & (labels == 0))
        fn = jnp.sum((predictions == 0) & (labels == 1))
        tp = jnp.sum((predictions == 1) & (labels == 1))
        
        far = fp / (fp + tn + 1e-8)
        frr = fn / (fn + tp + 1e-8)
        
        fars.append(float(far))
        frrs.append(float(frr))
    
    fars = jnp.array(fars)
    frrs = jnp.array(frrs)
    
    # Find EER (where FAR ≈ FRR)
    diff = jnp.abs(fars - frrs)
    eer_idx = jnp.argmin(diff)
    
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2.0
    threshold = scores_sorted[eer_idx]
    
    return float(eer), float(threshold)


# ============================================================================
# Miscellaneous Utilities
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def count_parameters(params: Dict) -> int:
    """Count total number of parameters in param dict."""
    def _count_nested(d):
        total = 0
        for v in d.values():
            if isinstance(v, dict):
                total += _count_nested(v)
            elif hasattr(v, 'shape'):
                total += jnp.prod(jnp.array(v.shape))
        return total
    
    return int(_count_nested(params))


def print_model_summary(params: Dict, model_name: str = "Model"):
    """Print summary of model parameters."""
    total_params = count_parameters(params)
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"{'='*60}\n")


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    print(f"✓ Created experiment directory: {exp_dir}")
    
    return exp_dir
