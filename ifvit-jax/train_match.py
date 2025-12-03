"""
Training script for Module 2: Fingerprint Matcher.

Trains the MatcherModel using L_D, L_E, and L_A losses.
Loads pretrained weights from Module 1 (DenseRegModel).
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from typing import Dict, Tuple
import argparse
from pathlib import Path

from config import MATCH_CONFIG
from models import MatcherModel
from losses import total_loss_matcher
from data import matcher_dataset, preprocess_batch
from utils import (
    save_checkpoint, load_checkpoint, load_pretrained_weights,
    Logger, print_model_summary, count_parameters
)


# ============================================================================
# Training State
# ============================================================================

class TrainState(train_state.TrainState):
    """Custom training state with ArcFace parameters."""
    arcface_params: Dict = None


def create_train_state(
    config: Dict,
    rng: jax.random.PRNGKey,
    pretrained_ckpt: str = None,
    num_classes: int = None
) -> TrainState:
    """
    Create initial training state.
    
    Args:
        config: MATCH_CONFIG dictionary
        rng: Random key
        pretrained_ckpt: Path to pretrained DenseRegModel checkpoint
        num_classes: Number of subject classes
        
    Returns:
        Initialized TrainState
    """
    # Initialize model
    model = MatcherModel(
        image_size=config["image_size"],
        roi_size=config["roi_size"],
        num_transformer_layers=config["transformer_layers"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=config["dropout_rate"],
        embedding_dim=config["embedding_dim"]
    )
    
    # Create dummy input for initialization
    dummy_img = jnp.zeros((1, config["image_size"], config["image_size"], 1))
    dummy_roi = jnp.zeros((1, config["roi_size"], config["roi_size"], 1))
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    variables = model.init(
        init_rng,
        dummy_img, dummy_img,
        dummy_roi, dummy_roi,
        train=False
    )
    params = variables['params']
    
    # IMPORTANT: Load transformer weights from Module 1 (NOT from LoFTR directly!)
    # Following IFViT paper: "employs the ViTs trained in the first module"
    if pretrained_ckpt and Path(pretrained_ckpt).exists():
        print(f"\n{'='*60}")
        print("Loading Module 1 Transformer Weights")
        print(f"{'='*60}")
        print(f"Module 1 checkpoint: {pretrained_ckpt}")
        print("This loads the TRAINED transformer from Module 1,")
        print("NOT fresh LoFTR weights (as per IFViT paper)")
        print(f"{'='*60}\n")
        
        try:
            from load_module1_weights import load_module1_transformer_weights
            
            # Share transformer weights between global and local branches
            # This is more faithful to paper: single ViT trained in Module 1
            params = load_module1_transformer_weights(
                module1_ckpt_path=pretrained_ckpt,
                module2_params=params,
                share_global_local=True  # Share weights as per IFViT
            )
        except Exception as e:
            print(f"⚠ Warning: Could not load Module 1 weights: {e}")
            print("  Continuing with random initialization")
            print("  This may not match IFViT paper's approach!")
    else:
        print(f"\n⚠ Warning: No Module 1 checkpoint provided!")
        print(f"  Module 2 should load from trained Module 1, not from scratch")
        print(f"  Set dense_reg_ckpt in config or use --pretrained_ckpt")
        print(f"  Current value: {pretrained_ckpt}\n")
    
    # Print model summary
    print_model_summary(params, "MatcherModel")
    
    # Create optimizer with warmup
    warmup_steps = config.get("warmup_epochs", 5) * 1000  # Approximate
    total_steps = config["num_epochs"] * 1000
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["lr"],
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
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
        tx=tx,
        arcface_params={}  # Will be initialized during first forward pass
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
    config: Dict,
    num_classes: int
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Perform a single training step.
    
    Args:
        state: Current training state
        batch: Batch of data
        rng: Random key
        config: Config dictionary
        num_classes: Total number of subject classes
        
    Returns:
        Updated state and metrics dictionary
    """
    def loss_fn(params):
        # Forward pass
        emb_g1, emb_g2, emb_l1, emb_l2, P, matches = state.apply_fn(
            {'params': params},
            batch['img1'], batch['img2'],
            batch['roi1'], batch['roi2'],
            train=True,
            rngs={'dropout': rng}
        )
        
        # Compute losses
        losses, updated_arcface_params = total_loss_matcher(
            emb_g1=emb_g1,
            emb_g2=emb_g2,
            emb_l1=emb_l1,
            emb_l2=emb_l2,
            labels_pair=batch['label_pair'],
            class_id1=batch['class_id1'],
            class_id2=batch['class_id2'],
            num_classes=num_classes,
            arcface_params=state.arcface_params,
            P=P if 'matches' in batch else None,
            gt_matches=batch.get('matches'),
            valid_mask=batch.get('valid_mask'),
            lambda_D=config['lambda_D'],
            lambda_E=config['lambda_E'],
            lambda_A=config['lambda_A'],
            embedding_margin=config['embedding_margin'],
            arcface_scale=config['arcface_scale'],
            arcface_margin=config['arcface_margin']
        )
        
        return losses['total'], (losses, updated_arcface_params)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, updated_arcface_params)), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(arcface_params=updated_arcface_params)
    
    return state, metrics


# ============================================================================
# Validation
# ============================================================================

@jax.jit
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    config: Dict,
    num_classes: int
) -> Dict[str, jnp.ndarray]:
    """
    Perform a single evaluation step.
    
    Args:
        state: Current training state
        batch: Batch of data
        rng: Random key
        config: Config dictionary
        num_classes: Total number of classes
        
    Returns:
        Metrics dictionary
    """
    # Forward pass (no dropout)
    emb_g1, emb_g2, emb_l1, emb_l2, P, matches = state.apply_fn(
        {'params': state.params},
        batch['img1'], batch['img2'],
        batch['roi1'], batch['roi2'],
        train=False
    )
    
    # Compute losses
    losses, _ = total_loss_matcher(
        emb_g1=emb_g1,
        emb_g2=emb_g2,
        emb_l1=emb_l1,
        emb_l2=emb_l2,
        labels_pair=batch['label_pair'],
        class_id1=batch['class_id1'],
        class_id2=batch['class_id2'],
        num_classes=num_classes,
        arcface_params=state.arcface_params,
        P=P if 'matches' in batch else None,
        gt_matches=batch.get('matches'),
        valid_mask=batch.get('valid_mask'),
        lambda_D=config['lambda_D'],
        lambda_E=config['lambda_E'],
        lambda_A=config['lambda_A'],
        embedding_margin=config['embedding_margin'],
        arcface_scale=config['arcface_scale'],
        arcface_margin=config['arcface_margin']
    )
    
    return losses


# ============================================================================
# Main Training Loop
# ============================================================================

def train_matcher(
    dataset_root: str,
    num_classes: int,
    config: Dict = None,
    resume_from: str = None
):
    """
    Main training function for Matcher.
    
    Args:
        dataset_root: Path to dataset
        num_classes: Number of subject classes in dataset
        config: Config dictionary (defaults to MATCH_CONFIG)
        resume_from: Optional checkpoint path to resume from
    """
    if config is None:
        config = MATCH_CONFIG
    
    # Initialize logger
    logger = Logger(
        log_dir=str(Path(config["checkpoint_dir"]) / "logs"),
        experiment_name="matcher"
    )
    logger.log("Starting Matcher training")
    logger.log(f"Config: {config}")
    logger.log(f"Number of classes: {num_classes}")
    
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
            tx=state_dict['tx'],
            arcface_params=state_dict.get('arcface_params', {})
        )
        start_epoch = metadata.get('epoch', 0) + 1
    else:
        # Load pretrained Module 1 weights
        pretrained_ckpt = config.get("dense_reg_ckpt")
        state = create_train_state(config, init_rng, pretrained_ckpt, num_classes)
        start_epoch = 0
    
    # Training loop
    logger.log("\n" + "="*60)
    logger.log("Starting training loop")
    logger.log("="*60 + "\n")
    
    global_step = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        logger.log(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Note: matcher_dataset is currently not implemented
        # This is a placeholder for when data is available
        try:
            train_dataset = matcher_dataset(
                dataset_root,
                config,
                split='train',
                shuffle=True,
                num_classes=num_classes
            )
            
            epoch_metrics = {
                'total': [],
                'L_D': [],
                'L_E': [],
                'L_A': []
            }
            
            # Training
            for step, batch in enumerate(train_dataset):
                # Preprocess batch
                batch = preprocess_batch(batch)
                
                # Training step
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, batch, step_rng, config, num_classes)
                
                # Log metrics
                for key, value in metrics.items():
                    if key in epoch_metrics:
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
            checkpoint_path = Path(config["checkpoint_dir"]) / f"matcher_epoch_{epoch+1}.pkl"
            save_checkpoint(
                str(checkpoint_path),
                {
                    'params': state.params,
                    'opt_state': state.opt_state,
                    'step': state.step,
                    'arcface_params': state.arcface_params
                },
                metadata={
                    'epoch': epoch,
                    'config': config,
                    'num_classes': num_classes
                }
            )
    
    # Save final checkpoint
    final_checkpoint = Path(config["checkpoint_dir"]) / "matcher_ckpt.pkl"
    save_checkpoint(
        str(final_checkpoint),
        {
            'params': state.params,
            'opt_state': state.opt_state,
            'step': state.step,
            'arcface_params': state.arcface_params
        },
        metadata={
            'epoch': config["num_epochs"],
            'config': config,
            'num_classes': num_classes
        }
    )
    
    logger.log("\n" + "="*60)
    logger.log("Training complete!")
    logger.log("="*60)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Matcher Model")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./data',
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help='Number of subject classes in dataset'
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
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default=None,
        help='Path to pretrained DenseRegModel checkpoint'
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    config = MATCH_CONFIG.copy()
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.pretrained_ckpt:
        config['dense_reg_ckpt'] = args.pretrained_ckpt
    
    # Train
    train_matcher(
        dataset_root=args.dataset_root,
        num_classes=args.num_classes,
        config=config,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
