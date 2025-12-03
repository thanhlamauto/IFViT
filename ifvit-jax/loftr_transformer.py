"""
LoFTR LocalFeatureTransformer implementation in Flax.

Re-implements the exact architecture from LoFTR paper for loading pretrained weights.

Reference: https://github.com/zju3dv/LoFTR
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional
import math


# ============================================================================
# Linear Attention (efficient approximation)
# ============================================================================

class LinearAttention(nn.Module):
    """
    Linear attention mechanism (O(N) complexity).
    
    Uses feature map φ(q) and φ(k) to approximate softmax attention.
    """
    
    @staticmethod
    def elu_feature_map(x):
        """ELU + 1 feature map for linear attention."""
        return nn.elu(x) + 1.0
    
    def __call__(self, q, k, v, q_mask=None, kv_mask=None):
        """
        Args:
            q: (B, N, H, D) queries
            k: (B, M, H, D) keys
            v: (B, M, H, Dv) values
            q_mask: Optional (B, N) query mask
            kv_mask: Optional (B, M) key/value mask
            
        Returns:
            out: (B, N, H, Dv) attention output
        """
        # Apply feature map
        q = self.elu_feature_map(q)
        k = self.elu_feature_map(k)
        
        # Apply masks if provided
        if kv_mask is not None:
            # kv_mask: (B, M) -> (B, M, 1, 1)
            kv_mask = kv_mask[:, :, None, None]
            k = k * kv_mask
            v = v * kv_mask
        
        # Linear attention: O(N) complexity
        # KV = sum_m k_m^T v_m: (B, H, D, Dv)
        kv = jnp.einsum('bmhd,bmhe->bhde', k, v)
        
        # Z = sum_m k_m: (B, H, D)
        z = jnp.sum(k, axis=1)
        
        # out = (q * KV) / (q * Z): (B, N, H, Dv)
        out = jnp.einsum('bnhd,bhde->bnhe', q, kv)
        normalizer = jnp.einsum('bnhd,bhd->bnh', q, z)
        out = out / (normalizer[..., None] + 1e-6)
        
        return out


# ============================================================================
# Full Attention (standard scaled dot-product)
# ============================================================================

class FullAttention(nn.Module):
    """
    Standard scaled dot-product attention.
    """
    
    def __call__(self, q, k, v, q_mask=None, kv_mask=None):
        """
        Args:
            q: (B, N, H, D) queries
            k: (B, M, H, D) keys
            v: (B, M, H, Dv) values
            q_mask: Optional (B, N) query mask
            kv_mask: Optional (B, M) key/value mask
            
        Returns:
            out: (B, N, H, Dv) attention output
        """
        # Compute attention scores
        # (B, N, H, D) x (B, M, H, D) -> (B, H, N, M)
        d = q.shape[-1]
        scores = jnp.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(d)
        
        # Apply key mask
        if kv_mask is not None:
            # kv_mask: (B, M) -> (B, 1, 1, M)
            scores = jnp.where(kv_mask[:, None, None, :], scores, -1e9)
        
        # Softmax
        attn = jax.nn.softmax(scores, axis=-1)
        
        # Apply query mask
        if q_mask is not None:
            # q_mask: (B, N) -> (B, 1, N, 1)
            attn = attn * q_mask[:, None, :, None]
        
        # Compute output
        # (B, H, N, M) x (B, M, H, Dv) -> (B, N, H, Dv)
        out = jnp.einsum('bhnm,bmhd->bnhd', attn, v)
        
        return out


# ============================================================================
# LoFTR Encoder Layer
# ============================================================================

class LoFTREncoderLayer(nn.Module):
    """
    Single LoFTR encoder layer with custom attention + MLP.
    
    Args:
        d_model: Model dimension (typically 256)
        nhead: Number of attention heads (typically 8)
        attention_type: 'linear' or 'full'
    """
    d_model: int
    nhead: int
    attention_type: str = 'linear'
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        source: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
        source_mask: Optional[jnp.ndarray] = None,
        train: bool = True
    ):
        """
        Args:
            x: (B, N, d_model) query features
            source: (B, M, d_model) key/value features
            x_mask: (B, N) mask for x
            source_mask: (B, M) mask for source
            train: Training mode
            
        Returns:
            out: (B, N, d_model) updated features
        """
        assert self.d_model % self.nhead == 0
        dim_per_head = self.d_model // self.nhead
        
        # ====== Multi-head Attention ======
        # Project to Q, K, V
        q = nn.Dense(self.d_model, name='q_proj')(x)
        k = nn.Dense(self.d_model, name='k_proj')(source)
        v = nn.Dense(self.d_model, name='v_proj')(source)
        
        # Reshape to multi-head: (B, N, d_model) -> (B, N, nhead, dim_per_head)
        B, N = x.shape[:2]
        M = source.shape[1]
        
        q = q.reshape(B, N, self.nhead, dim_per_head)
        k = k.reshape(B, M, self.nhead, dim_per_head)
        v = v.reshape(B, M, self.nhead, dim_per_head)
        
        # Apply attention
        if self.attention_type == 'linear':
            message = LinearAttention()(q, k, v, x_mask, source_mask)
        else:  # 'full'
            message = FullAttention()(q, k, v, x_mask, source_mask)
        
        # Reshape back: (B, N, nhead, dim_per_head) -> (B, N, d_model)
        message = message.reshape(B, N, self.d_model)
        
        # Merge projection
        message = nn.Dense(self.d_model, name='merge')(message)
        
        # ====== Feed-forward with concat ======
        # LoFTR uses concat([x, message]) for MLP input (size 2*d_model)
        # This is different from standard Transformer
        
        # Normalization before MLP
        normed = nn.LayerNorm(name='norm1')(x)
        
        # Concatenate x with message
        mlp_input = jnp.concatenate([normed, message], axis=-1)  # (B, N, 2*d_model)
        
        # MLP: 2*d_model -> 2*d_model -> d_model
        mlp_out = nn.Dense(2 * self.d_model, name='mlp.0')(mlp_input)
        mlp_out = nn.relu(mlp_out)
        mlp_out = nn.Dense(self.d_model, name='mlp.2')(mlp_out)
        
        # Residual + normalization
        out = x + mlp_out
        out = nn.LayerNorm(name='norm2')(out)
        
        return out


# ============================================================================
# Local Feature Transformer
# ============================================================================

class LocalFeatureTransformer(nn.Module):
    """
    LoFTR's LocalFeatureTransformer module.
    
    Alternates between self-attention and cross-attention layers.
    
    Args:
        d_model: Model dimension (typically 256)
        nhead: Number of attention heads (typically 8)
        layer_names: Sequence of 'self' or 'cross' (e.g., ['self', 'cross', 'self', 'cross'])
        attention_type: 'linear' or 'full'
    """
    d_model: int = 256
    nhead: int = 8
    layer_names: Sequence[str] = ('self', 'cross', 'self', 'cross')
    attention_type: str = 'linear'
    
    @nn.compact
    def __call__(
        self,
        feat0: jnp.ndarray,
        feat1: jnp.ndarray,
        mask0: Optional[jnp.ndarray] = None,
        mask1: Optional[jnp.ndarray] = None,
        train: bool = True
    ):
        """
        Args:
            feat0: (B, N0, d_model) features from image 0
            feat1: (B, N1, d_model) features from image 1
            mask0: Optional (B, N0) mask
            mask1: Optional (B, N1) mask
            train: Training mode
            
        Returns:
            feat0_out: (B, N0, d_model) refined features
            feat1_out: (B, N1, d_model) refined features
        """
        assert feat0.shape[-1] == self.d_model
        assert feat1.shape[-1] == self.d_model
        
        for i, layer_type in enumerate(self.layer_names):
            if layer_type == 'self':
                # Self-attention on both features
                feat0 = LoFTREncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    attention_type=self.attention_type,
                    name=f'layers.{i}.0'  # Layer for feat0
                )(feat0, feat0, mask0, mask0, train)
                
                feat1 = LoFTREncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    attention_type=self.attention_type,
                    name=f'layers.{i}.1'  # Layer for feat1
                )(feat1, feat1, mask1, mask1, train)
                
            elif layer_type == 'cross':
                # Cross-attention between features
                feat0_new = LoFTREncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    attention_type=self.attention_type,
                    name=f'layers.{i}.0'  # Query from feat0, KV from feat1
                )(feat0, feat1, mask0, mask1, train)
                
                feat1_new = LoFTREncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    attention_type=self.attention_type,
                    name=f'layers.{i}.1'  # Query from feat1, KV from feat0
                )(feat1, feat0, mask1, mask0, train)
                
                feat0 = feat0_new
                feat1 = feat1_new
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        return feat0, feat1


# ============================================================================
# Positional Encoding
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, height: int, width: int) -> jnp.ndarray:
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension (must be even)
        height: Grid height
        width: Grid width
        
    Returns:
        pos_embed: (H*W, embed_dim) positional embeddings
    """
    assert embed_dim % 2 == 0
    
    # Generate grid
    grid_h = jnp.arange(height, dtype=jnp.float32)
    grid_w = jnp.arange(width, dtype=jnp.float32)
    grid_h, grid_w = jnp.meshgrid(grid_h, grid_w, indexing='ij')
    
    # Flatten
    grid_h = grid_h.reshape(-1)
    grid_w = grid_w.reshape(-1)
    
    # Compute positional embeddings
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    
    # Height embeddings
    pos_h = jnp.outer(grid_h, omega)  # (H*W, embed_dim//2)
    pos_h = jnp.concatenate([jnp.sin(pos_h), jnp.cos(pos_h)], axis=1)
    
    # Width embeddings
    pos_w = jnp.outer(grid_w, omega)
    pos_w = jnp.concatenate([jnp.sin(pos_w), jnp.cos(pos_w)], axis=1)
    
    # Combine (can also use addition instead of concat)
    pos_embed = jnp.concatenate([pos_h[:, :embed_dim//2], pos_w[:, :embed_dim//2]], axis=1)
    
    return pos_embed


class PositionalEncoding2D(nn.Module):
    """
    Learnable or fixed 2D positional encoding.
    """
    d_model: int
    height: int
    width: int
    learnable: bool = False
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, H, W, C) feature map
            
        Returns:
            x_with_pos: (B, H*W, C) features with positional encoding added
        """
        B, H, W, C = x.shape
        assert H == self.height and W == self.width
        assert C == self.d_model
        
        if self.learnable:
            # Learnable positional embedding
            pos_embed = self.param(
                'pos_embed',
                nn.initializers.normal(stddev=0.02),
                (H * W, self.d_model)
            )
        else:
            # Fixed sinusoidal positional embedding
            pos_embed = get_2d_sincos_pos_embed(self.d_model, H, W)
            # Make it a parameter so it's tracked, but not trainable
            pos_embed = self.variable(
                'pos_encoding',
                'pos_embed',
                lambda: pos_embed
            ).value
        
        # Flatten spatial dimensions
        x_flat = x.reshape(B, H * W, C)
        
        # Add positional encoding
        x_with_pos = x_flat + pos_embed[None, :, :]
        
        return x_with_pos
