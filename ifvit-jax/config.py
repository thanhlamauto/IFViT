"""
Configuration dictionaries for IFViT-JAX training.

Module 1 (DenseReg): Dense correspondence registration using only L_D loss.
Module 2 (Matcher): Fingerprint matching using L_D, L_E, and L_A losses.
"""

# ============================================================================
# Module 1: Dense Registration Configuration
# ============================================================================
DENSE_CONFIG = {
    # Image settings
    "image_size": 128,  # Paper: 128x128
    
    # Training hyperparameters (from paper 2404.08237v1)
    "batch_size": 128,  # Paper: 128
    "lr": 1e-3,  # Paper: 1e-3
    "num_epochs": 100,  # Paper: 100
    "weight_decay": 2e-4,  # Paper: 2e-4
    
    # Loss weights
    "lambda_D": 1.0,  # Only use L_D for dense registration
    
    # Model architecture
    "backbone": "resnet18",
    "transformer_layers": 4,
    "num_heads": 8,
    "hidden_dim": 256,
    "mlp_dim": 1024,
    "dropout_rate": 0.1,
    
    # LoFTR settings
    "use_loftr": True,  # Use LoFTR LocalFeatureTransformer (for loading pretrained weights)
    "attention_type": "linear",  # 'linear' or 'full'
    "loftr_pretrained_ckpt": None,  # Path to converted LoFTR checkpoint (.npz)
    
    # Checkpointing
    "checkpoint_dir": "./checkpoints/dense_reg",
    "save_every": 5,  # Save every N epochs
    
    # Logging
    "log_every": 10,  # Log every N steps
}


# ============================================================================
# Module 2: Matcher Configuration
# ============================================================================
MATCH_CONFIG = {
    # Image settings
    "image_size": 224,  # Note: Paper doesn't specify, but typically larger for Module 2
    "roi_size": 90,  # Paper: 90x90 ROIs
    
    # Training hyperparameters (from paper 2404.08237v1)
    # Fine-tuning on real fingerprints: 70 epochs, LR=1e-4
    "batch_size": 128,  # Paper: 128
    "lr": 1e-4,  # Paper: 1e-4 (fine-tuning), 1e-3 (pre-training on PrintsGAN)
    "num_epochs": 70,  # Paper: 70 (fine-tuning)
    "weight_decay": 2e-4,  # Paper: 2e-4
    "warmup_epochs": 5,  # Not in paper, but common practice
    
    # Loss weights (from paper: λ1=0.5, λ2=0.1, λ3=1.0)
    "lambda_D": 0.5,   # Paper: 0.5
    "lambda_E": 0.1,   # Paper: 0.1
    "lambda_A": 1.0,   # Paper: 1.0
    
    # Model architecture
    "backbone": "resnet18",
    "transformer_layers": 4,
    "num_heads": 8,
    "hidden_dim": 256,
    "mlp_dim": 1024,
    "dropout_rate": 0.1,
    "embedding_dim": 256,
    
    # LoFTR settings (inherited from Module 1)
    "use_loftr": True,  # Use LoFTR LocalFeatureTransformer
    "attention_type": "linear",  # 'linear' or 'full'
    
    # ArcFace settings (from paper: m=0.4, s=64)
    "arcface_scale": 64.0,  # Paper: s=64
    "arcface_margin": 0.4,  # Paper: m=0.4
    
    # Embedding loss settings (from paper: margin m=0.4)
    "embedding_margin": 0.4,  # Paper: m=0.4 (cosine embedding loss)
    
    # Score fusion (for inference)
    "alpha_global": 0.6,
    "alpha_local": 0.4,
    
    # Checkpointing
    "checkpoint_dir": "./checkpoints/matcher",
    # IMPORTANT: dense_reg_ckpt should point to TRAINED Module 1 checkpoint
    # Module 2 loads Module 1's trained transformer (NOT fresh LoFTR weights)
    # This follows IFViT paper: "employs the ViTs trained in the first module"
    "dense_reg_ckpt": "./checkpoints/dense_reg/dense_reg_ckpt.pkl",  # Trained Module 1
    "save_every": 5,
    
    # Logging
    "log_every": 10,
}


# ============================================================================
# Inference Configuration
# ============================================================================
INFER_CONFIG = {
    # Model checkpoints
    "dense_model_ckpt": "./checkpoints/dense_reg/dense_reg_ckpt.pkl",
    "matcher_model_ckpt": "./checkpoints/matcher/matcher_ckpt.pkl",
    
    # Alignment settings
    "rotation_angles": [-60, -30, 0, 30, 60],  # Degrees to try for alignment
    "min_matches": 10,  # Minimum matches to consider alignment valid
    
    # Score fusion
    "alpha_global": 0.6,
    "alpha_local": 0.4,
    
    # Verification threshold
    "threshold": 0.5,  # Score threshold for match/non-match decision
}


# ============================================================================
# Data Augmentation Configuration
# ============================================================================
AUGMENT_CONFIG = {
    # Rotation
    "rotation_range": (-60, 60),  # Degrees
    
    # Noise
    "noise_std": 0.02,
    
    # Morphological operations
    "erosion_prob": 0.3,
    "dilation_prob": 0.3,
    "morph_kernel_size": (3, 3),
    
    # Brightness/Contrast
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    
    # Blur
    "gaussian_blur_prob": 0.2,
    "blur_sigma_range": (0.5, 1.5),
}
