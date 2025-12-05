"""
Training script for Module 1: Dense Registration.

Trains the DenseRegModel to learn dense correspondences using only L_D loss.
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from typing import Dict, Tuple
import argparse
from pathlib import Path

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
# Training State
# ============================================================================

class TrainState(train_state.TrainState):
    """Custom training state with additional fields."""
    pass


def create_train_state(config: Dict, rng: jax.random.PRNGKey) -> TrainState:
    """
    Create initial training state.
    
    Args:
        config: DENSE_CONFIG dictionary
        rng: Random key
        
    Returns:
        Initialized TrainState
    """
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
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, dummy_img, dummy_img, train=False)
    params = variables['params']
    
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
        decay_steps=config["num_epochs"] * 1000,  # Approximate steps per epoch
        end_value=config["lr"] * 0.01
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config["weight_decay"]
        )
    )
    
    # Create training state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state


# ============================================================================
# Training Step
# ============================================================================

@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    config: Dict
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Perform a single training step.
    
    Args:
        state: Current training state
        batch: Batch of data
        rng: Random key
        config: Config dictionary
        
    Returns:
        Updated state and metrics dictionary
    """
    def loss_fn(params):
        # Forward pass
        P, matches, feat1, feat2 = state.apply_fn(
            {'params': params},
            batch['img1'],
            batch['img2'],
            train=True,
            rngs={'dropout': rng}
        )
        
        # Compute loss
        losses = total_loss_dense(
            P=P,
            gt_matches=batch['matches'],
            valid_mask=batch.get('valid_mask'),
            lambda_D=config['lambda_D'],
            feature_shape=None  # Will be inferred from feature maps
        )
        
        return losses['total'], losses
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


# ============================================================================
# Validation
# ============================================================================

@jax.jit
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    config: Dict
) -> Dict[str, jnp.ndarray]:
    """
    Perform a single evaluation step.
    
    Args:
        state: Current training state
        batch: Batch of data
        rng: Random key
        config: Config dictionary
        
    Returns:
        Metrics dictionary
    """
    # Forward pass (no dropout in eval mode)
    P, matches, feat1, feat2 = state.apply_fn(
        {'params': state.params},
        batch['img1'],
        batch['img2'],
        train=False
    )
    
    # Compute loss
    losses = total_loss_dense(
        P=P,
        gt_matches=batch['matches'],
        valid_mask=batch.get('valid_mask'),
        lambda_D=config['lambda_D'],
        feature_shape=None
    )
    
    return losses


# ============================================================================
# Main Training Loop
# ============================================================================

def train_dense_reg(
    dataset_root: str,
    config: Dict = None,
    resume_from: str = None
):
    """
    Main training function for Dense Registration.
    
    Args:
        dataset_root: Path to dataset
        config: Config dictionary (defaults to DENSE_CONFIG)
        resume_from: Optional checkpoint path to resume from
    """
    if config is None:
        config = DENSE_CONFIG
    
    # Initialize logger
    logger = Logger(
        log_dir=str(Path(config["checkpoint_dir"]) / "logs"),
        experiment_name="dense_reg"
    )
    logger.log("Starting Dense Registration training")
    logger.log(f"Config: {config}")
    
    # Initialize RNG
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Create training state
    if resume_from:
        logger.log(f"Resuming from checkpoint: {resume_from}")
        state_dict, metadata = load_checkpoint(resume_from)
        state = TrainState.create(
            apply_fn=state_dict['apply_fn'],
            params=state_dict['params'],
            tx=state_dict['tx']
        )
        start_epoch = metadata.get('epoch', 0) + 1
    else:
        state = create_train_state(config, init_rng)
        start_epoch = 0
    
    # Training loop
    logger.log("\n" + "="*60)
    logger.log("Starting training loop")
    logger.log("="*60 + "\n")
    
    global_step = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        logger.log(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Load dataset entries
        try:
            # Initialize dataset roots (auto-detects Kaggle paths)
            roots = PaperDatasetRoots()
            
            # Build train entries from all datasets
            train_entries = build_paper_train_entries(roots)
            val_entries = build_paper_val_entries(roots)
            
            logger.log(f"Loaded {len(train_entries)} training entries")
            logger.log(f"Loaded {len(val_entries)} validation entries")
            
            if len(train_entries) == 0:
                raise ValueError("No training entries found. Check dataset paths.")
            
            # Create dataset iterators
            train_dataset = dense_reg_dataset(
                entries=train_entries,
                config=config,
                split='train',
                shuffle=True
            )
            
            val_dataset = dense_reg_dataset(
                entries=val_entries,
                config=config,
                split='val',
                shuffle=False
            )
            
            epoch_metrics = {
                'total': [],
                'L_D': []
            }
            
            # Training
            for step, batch in enumerate(train_dataset):
                # Preprocess batch
                batch = preprocess_batch(batch)
                
                # Training step
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, batch, step_rng, config)
                
                # Log metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(float(value))
                
                global_step += 1
                
                # Periodic logging
                if global_step % config["log_every"] == 0:
                    log_metrics = {k: float(v) for k, v in metrics.items()}
                    logger.log_metrics(global_step, log_metrics, prefix="[Train] ")
                
                # Break after some steps (for demonstration)
                # Remove this when real data is available
                if step >= 10:
                    break
            
            # Epoch summary
            avg_metrics = {
                k: sum(v) / len(v) if v else 0.0
                for k, v in epoch_metrics.items()
            }
            logger.log_epoch(epoch + 1, avg_metrics)
            
        except NotImplementedError as e:
            logger.log(f"⚠ Dataset not implemented yet: {e}")
            logger.log("  Skipping training for now - waiting for real data")
            logger.log("  Training state has been initialized successfully")
            break
        
        # Save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            checkpoint_path = Path(config["checkpoint_dir"]) / f"dense_reg_epoch_{epoch+1}.pkl"
            save_checkpoint(
                str(checkpoint_path),
                {
                    'params': state.params,
                    'opt_state': state.opt_state,
                    'step': state.step
                },
                metadata={
                    'epoch': epoch,
                    'config': config
                }
            )
    
    # Save final checkpoint
    final_checkpoint = Path(config["checkpoint_dir"]) / "dense_reg_ckpt.pkl"
    save_checkpoint(
        str(final_checkpoint),
        {
            'params': state.params,
            'opt_state': state.opt_state,
            'step': state.step
        },
        metadata={
            'epoch': config["num_epochs"],
            'config': config
        }
    )
    
    logger.log("\n" + "="*60)
    logger.log("Training complete!")
    logger.log("="*60)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Dense Registration Model")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./data',
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Checkpoint path to resume from'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Override checkpoint directory from config'
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    config = DENSE_CONFIG.copy()
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    # Train
    train_dense_reg(
        dataset_root=args.dataset_root,
        config=config,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
