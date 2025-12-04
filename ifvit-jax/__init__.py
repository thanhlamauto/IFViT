"""
IFViT-JAX: Fingerprint Matching with JAX/Flax

A fingerprint verification system using dense correspondence learning
and metric learning.
"""

__version__ = "0.1.0"

from . import config
from . import models
from . import losses
from . import ut

# Make key classes/functions easily accessible
from .models import DenseRegModel, MatcherModel, ResNet18
from .losses import (
    dense_reg_loss,
    cosine_embedding_loss,
    arcface_loss,
    compute_matching_score
)
from .ut.utils import (
    save_checkpoint,
    load_checkpoint,
    Logger
)

__all__ = [
    # Models
    'DenseRegModel',
    'MatcherModel',
    'ResNet18',
    
    # Losses
    'dense_reg_loss',
    'cosine_embedding_loss',
    'arcface_loss',
    'compute_matching_score',
    
    # Utils
    'save_checkpoint',
    'load_checkpoint',
    'Logger',
    
    # Modules
    'config',
    'models',
    'losses',
    'ut',
]
