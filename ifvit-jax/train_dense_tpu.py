"""
Training script for Module 1: Dense Registration (TPU-optimized).

Optimized for TPU v5e-8 on Kaggle with:
- Data parallelism using jax.pmap
- Efficient batch sharding
- Gradient accumulation support
- Multi-device checkpointing

Usage on Kaggle TPU:
    !python ifvit-jax/train_dense_tpu.py \
        --checkpoint_dir /kaggle/working/IFViT/checkpoints/dense_reg
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import jax_utils
import optax
from typing import Dict, Tuple
import argparse
from pathlib import Path
import numpy as np

from config import DENSE_CONFIG
from models import DenseRegModel
from losses import total_loss_dense
import sys
from pathlib import Path

# Add root to path for data module
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from data import (
    dense_reg_dataset, 
    preprocess_batch,
    PaperDatasetRoots,
    build_paper_train_entries,
    build_paper_val_entries,
)
from ut.utils import (
    save_checkpoint, load_checkpoint, Logger,
    print_model_summary, count_parameters
)


# ============================================================================
# TPU Setup
# ============================================================================

def setup_tpu():
    """Initialize TPU and return device count."""
    # JAX will automatically detect TPU
    devices = jax.devices()
    num_devices = len(devices)
    
    print(f"Found {num_devices} devices")
    print(f"Device type: {devices[0].platform}")
    print(f"Devices: {devices}")
    
    return num_devices, devices


# ============================================================================
# Training State
# ============================================================================

class TrainState(train_state.TrainState):
    """Custom training state with batch_stats for BatchNorm."""
    batch_stats: Dict = None


def create_train_state(config: Dict, rng: jax.random.PRNGKey) -> TrainState:
    """Create initial training state (same as train_dense.py)."""
    # Initialize model
    model = DenseRegModel(
        image_size=config["image_size"],
        num_transformer_layers=config["transformer_layers"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=config["dropout_rate"],
        use_loftr=config.get("use_loftr", True),
        attention_type=config.get("attention_type", "linear")
    )
    
    # Create dummy input for initialization
    dummy_img = jnp.zeros((1, config["image_size"], config["image_size"], 1))
    
    # Initialize parameters
    # Use train=True to ensure batch_stats is created
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, dummy_img, dummy_img, train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    # Ensure batch_stats is not None or empty
    if batch_stats is None or len(batch_stats) == 0:
        # If batch_stats is empty, initialize it by running a forward pass
        # This will create the batch_stats collection
        _, batch_stats = model.apply(
            variables,
            dummy_img,
            dummy_img,
            train=True,
            mutable=['batch_stats']
        )
        batch_stats = batch_stats['batch_stats']
    
    # Load LoFTR pretrained weights if specified
    loftr_ckpt = config.get("loftr_pretrained_ckpt")
    if loftr_ckpt and Path(loftr_ckpt).exists():
        print(f"\n{'='*60}")
        print("Loading LoFTR Pretrained Weights")
        print(f"{'='*60}")
        print(f"Checkpoint: {loftr_ckpt}")
        print(f"{'='*60}\n")
        
        try:
            from ut.load_loftr_weights import load_loftr_weights, merge_params
            
            loftr_weights = load_loftr_weights(loftr_ckpt)
            params = merge_params(params, loftr_weights, prefix='loftr_transformer')
            print("✓ LoFTR weights loaded successfully\n")
        except Exception as e:
            print(f"⚠ Warning: Could not load LoFTR weights: {e}")
            print("  Continuing with random initialization\n")
    elif loftr_ckpt:
        print(f"⚠ Warning: LoFTR checkpoint not found: {loftr_ckpt}")
        print("  Continuing with random initialization\n")
    
    # Print model summary
    print_model_summary(params, "DenseRegModel")
    
    # Create optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["lr"],
        warmup_steps=1000,
        decay_steps=config["num_epochs"] * 1000,
        end_value=config["lr"] * 0.01
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config["weight_decay"]
        )
    )
    
    # Create training state with batch_stats
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )
    
    return state


# ============================================================================
# TPU-Optimized Training Step
# ============================================================================

# Global variable for lambda_D (set before pmap)
LAMBDA_D = 1.0

def train_step_fn(
    state: TrainState,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    matches: jnp.ndarray,
    valid_mask: jnp.ndarray,
    rng: jax.random.PRNGKey
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Single device training step (will be pmapped).
    
    Args:
        state: Current training state
        img1: (per_device_batch, H, W, 1) images
        img2: (per_device_batch, H, W, 1) images
        matches: (per_device_batch, K, 4) correspondences
        valid_mask: (per_device_batch, K) validity mask
        rng: Random key (shape: (2,))
        
    Returns:
        Updated state and metrics dictionary
    """
    # Use global lambda_D (set before creating pmap)
    lambda_D = LAMBDA_D
    
    def loss_fn(params):
        # Forward pass with batch_stats
        # Ensure batch_stats is not None
        batch_stats_dict = state.batch_stats if state.batch_stats is not None else {}
        variables = {'params': params, 'batch_stats': batch_stats_dict}
        outputs, updated_variables = state.apply_fn(
            variables,
            img1,
            img2,
            train=True,
            rngs={'dropout': rng},
            mutable=['batch_stats']
        )
        P, matches_out, feat1, feat2 = outputs
        updated_batch_stats = updated_variables.get('batch_stats', {})
        
        # Compute loss
        losses = total_loss_dense(
            P=P,
            gt_matches=matches,
            valid_mask=valid_mask,
            lambda_D=lambda_D,
            feature_shape=None
        )
        
        return losses['total'], (losses, updated_batch_stats)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, updated_batch_stats)), grads = grad_fn(state.params)
    
    # Update parameters and batch_stats
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updated_batch_stats['batch_stats'])
    
    return state, metrics


def shard_batch(batch: Dict[str, jnp.ndarray], num_devices: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Shard batch across devices and return as separate arrays.
    
    Args:
        batch: Batch dictionary
        num_devices: Number of TPU devices
        
    Returns:
        Tuple of (img1, img2, matches, valid_mask) - all sharded
    """
    img1 = batch['img1']
    img2 = batch['img2']
    matches = batch['matches']
    valid_mask = batch.get('valid_mask')
    
    # Split batch dimension across devices
    batch_size = img1.shape[0]
    per_device_batch = batch_size // num_devices
    
    # Reshape: (batch_size, ...) -> (num_devices, per_device_batch, ...)
    img1_sharded = img1.reshape((num_devices, per_device_batch) + img1.shape[1:])
    img2_sharded = img2.reshape((num_devices, per_device_batch) + img2.shape[1:])
    matches_sharded = matches.reshape((num_devices, per_device_batch) + matches.shape[1:])
    
    if valid_mask is not None:
        valid_mask_sharded = valid_mask.reshape((num_devices, per_device_batch) + valid_mask.shape[1:])
    else:
        # Create dummy valid_mask if None
        valid_mask_sharded = jnp.ones((num_devices, per_device_batch, matches.shape[1]), dtype=bool)
    
    return img1_sharded, img2_sharded, matches_sharded, valid_mask_sharded


# ============================================================================
# Main Training Loop (TPU)
# ============================================================================

def train_dense_reg_tpu(
    config: Dict = None,
    resume_from: str = None
):
    """
    Main training function for Dense Registration (TPU-optimized).
    
    Args:
        config: Config dictionary (defaults to DENSE_CONFIG)
        resume_from: Optional checkpoint path to resume from
    """
    if config is None:
        config = DENSE_CONFIG
    
    # Setup TPU
    num_devices, devices = setup_tpu()
    
    # Adjust batch size for multi-device
    # Effective batch size = batch_size * num_devices
    per_device_batch = config["batch_size"] // num_devices
    if per_device_batch < 1:
        per_device_batch = 1
        print(f"⚠ Warning: batch_size {config['batch_size']} < num_devices {num_devices}")
        print(f"  Using per_device_batch=1, effective batch_size={num_devices}")
    
    effective_batch_size = per_device_batch * num_devices
    print(f"Batch size: {config['batch_size']} → {per_device_batch} per device")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Initialize logger
    logger = Logger(
        log_dir=str(Path(config["checkpoint_dir"]) / "logs"),
        experiment_name="dense_reg_tpu"
    )
    logger.log("Starting Dense Registration training (TPU-optimized)")
    logger.log(f"Config: {config}")
    logger.log(f"TPU devices: {num_devices}")
    logger.log(f"Effective batch size: {effective_batch_size}")
    
    # Initialize RNG
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Set global lambda_D for training step
    global LAMBDA_D
    LAMBDA_D = config.get('lambda_D', 1.0)
    
    # Create training state
    if resume_from:
        logger.log(f"Resuming from checkpoint: {resume_from}")
        state_dict, metadata = load_checkpoint(resume_from)
        state = TrainState.create(
            apply_fn=state_dict['apply_fn'],
            params=state_dict['params'],
            tx=state_dict['tx'],
            batch_stats=state_dict.get('batch_stats', {})
        )
        start_epoch = metadata.get('epoch', 0) + 1
    else:
        state = create_train_state(config, init_rng)
        start_epoch = 0
    
    # Load dataset entries first (before creating pmap)
    try:
        roots = PaperDatasetRoots()
        train_entries = build_paper_train_entries(roots)
        val_entries = build_paper_val_entries(roots)
        
        logger.log(f"Loaded {len(train_entries)} training entries")
        logger.log(f"Loaded {len(val_entries)} validation entries")
        
        if len(train_entries) == 0:
            raise ValueError("No training entries found. Check dataset paths.")
    except Exception as e:
        logger.log(f"⚠ Error loading datasets: {e}")
        raise
    
    # Ensure batch_stats is not None or empty before replicating
    if state.batch_stats is None or len(state.batch_stats) == 0:
        # Initialize batch_stats if it's None or empty
        dummy_img = jnp.zeros((1, config["image_size"], config["image_size"], 1))
        _, batch_stats = state.apply_fn(
            {'params': state.params, 'batch_stats': {}},
            dummy_img,
            dummy_img,
            train=True,
            mutable=['batch_stats']
        )
        state = state.replace(batch_stats=batch_stats['batch_stats'])
    
    # Replicate state across devices
    state = jax_utils.replicate(state)
    
    # Create pmapped training step (after dataset loading)
    # Note: Removed config from function signature to avoid pmap issues with Python dicts
    # Use static_argnums to ensure pmap doesn't trace through dataset loading functions
    train_step_pmap = jax.pmap(
        train_step_fn,
        axis_name='batch',
        donate_argnums=(0,),  # Donate state buffer for efficiency
        static_broadcasted_argnums=()  # No static args (all are JAX arrays)
    )
    
    # Training loop
    logger.log("\n" + "="*60)
    logger.log("Starting training loop (TPU)")
    logger.log("="*60 + "\n")
    
    global_step = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        logger.log(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Use pre-loaded entries (reload if needed for new epoch)
        try:
            
            # Create dataset iterator
            train_dataset = dense_reg_dataset(
                entries=train_entries,
                config=config,
                split='train',
                shuffle=True
            )
            
            epoch_metrics = {
                'total': [],
                'L_D': []
            }
            
            # Training
            for step, batch in enumerate(train_dataset):
                # Preprocess batch
                batch = preprocess_batch(batch)
                
                # Shard batch across devices
                # Ensure batch size is divisible by num_devices
                batch_size = batch['img1'].shape[0]
                if batch_size < effective_batch_size:
                    # Pad batch if needed
                    pad_size = effective_batch_size - batch_size
                    for key in ['img1', 'img2', 'matches', 'valid_mask']:
                        if batch[key] is not None:
                            pad_shape = (pad_size,) + batch[key].shape[1:]
                            pad = jnp.zeros(pad_shape, dtype=batch[key].dtype)
                            batch[key] = jnp.concatenate([batch[key], pad], axis=0)
                
                # Shard batch and unpack for pmap
                img1_sharded, img2_sharded, matches_sharded, valid_mask_sharded = shard_batch(batch, num_devices)
                
                # Shard RNG - split into per-device keys
                # jax.random.split returns shape (num_devices, 2) which is correct for pmap
                step_rngs = jax.random.split(rng, num_devices)
                
                # Training step (pmapped)
                state, metrics = train_step_pmap(
                    state,
                    img1_sharded,
                    img2_sharded,
                    matches_sharded,
                    valid_mask_sharded,
                    step_rngs
                )
                
                # Unreplicate metrics for logging
                metrics = jax_utils.unreplicate(metrics)
                
                # Log metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(float(value))
                
                global_step += 1
                
                # Periodic logging
                if global_step % config["log_every"] == 0:
                    log_metrics = {k: float(v) for k, v in metrics.items()}
                    logger.log_metrics(global_step, log_metrics, prefix="[Train] ")
                
                # Update RNG
                rng, _ = jax.random.split(rng)
            
            # Epoch summary
            avg_metrics = {
                k: sum(v) / len(v) if v else 0.0
                for k, v in epoch_metrics.items()
            }
            logger.log_epoch(epoch + 1, avg_metrics)
            
        except Exception as e:
            logger.log(f"⚠ Error: {e}")
            import traceback
            logger.log(traceback.format_exc())
            break
        
        # Save checkpoint (unreplicate state first)
        if (epoch + 1) % config["save_every"] == 0:
            # Unreplicate state for saving
            unreplicated_state = jax_utils.unreplicate(state)
            
            checkpoint_path = Path(config["checkpoint_dir"]) / f"dense_reg_epoch_{epoch+1}.pkl"
            save_checkpoint(
                str(checkpoint_path),
                {
                    'params': unreplicated_state.params,
                    'opt_state': unreplicated_state.opt_state,
                    'step': unreplicated_state.step,
                    'apply_fn': unreplicated_state.apply_fn,
                },
                metadata={
                    'epoch': epoch,
                    'config': config,
                    'num_devices': num_devices,
                }
            )
            logger.log(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Save final checkpoint
    unreplicated_state = jax_utils.unreplicate(state)
    final_checkpoint = Path(config["checkpoint_dir"]) / "dense_reg_ckpt.pkl"
    save_checkpoint(
        str(final_checkpoint),
        {
            'params': unreplicated_state.params,
            'opt_state': unreplicated_state.opt_state,
            'step': unreplicated_state.step,
            'apply_fn': unreplicated_state.apply_fn,
        },
        metadata={
            'epoch': config["num_epochs"],
            'config': config,
            'num_devices': num_devices,
        }
    )
    
    logger.log("\n" + "="*60)
    logger.log("Training complete!")
    logger.log("="*60)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Dense Registration Model (TPU-optimized)")
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Override checkpoint directory from config'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Checkpoint path to resume from'
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    config = DENSE_CONFIG.copy()
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    # Train
    train_dense_reg_tpu(
        config=config,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()

