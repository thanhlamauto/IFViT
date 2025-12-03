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
    "image_size": 128,
    
    # Training hyperparameters
    "batch_size": 64,
    "lr": 3e-4,
    "num_epochs": 50,
    "weight_decay": 1e-4,
    
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
    "image_size": 224,
    "roi_size": 90,  # ROI patch size for local features
    
    # Training hyperparameters
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 40,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    
    # Loss weights
    "lambda_D": 0.5,   # Dense correspondence loss
    "lambda_E": 0.1,   # Cosine embedding loss
    "lambda_A": 1.0,   # ArcFace loss
    
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
    
    # ArcFace settings
    "arcface_scale": 30.0,  # s parameter
    "arcface_margin": 0.5,  # m parameter
    
    # Embedding loss settings
    "embedding_margin": 0.2,
    
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
