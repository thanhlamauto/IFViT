"""
Model architectures for IFViT-JAX.

Includes:
- ResNet-18 backbone
- Transformer blocks (self-attention & cross-attention)
- Siamese Transformer encoder (generic)
- LoFTR LocalFeatureTransformer (for loading pretrained weights)
- Dense matching head
- Embedding heads for Module 2
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional, Callable
import functools

# Import LoFTR transformer implementation
from loftr_transformer import LocalFeatureTransformer, PositionalEncoding2D


# ============================================================================
# ResNet-18 Backbone
# ============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet."""
    filters: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        # First conv
        y = nn.Conv(self.filters, (3, 3), strides=self.stride, padding='SAME', 
                    use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        
        # Second conv
        y = nn.Conv(self.filters, (3, 3), strides=1, padding='SAME',
                    use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        
        # Shortcut
        if self.stride != 1 or x.shape[-1] != self.filters:
            residual = nn.Conv(self.filters, (1, 1), strides=self.stride,
                             use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)
        
        return nn.relu(y + residual)


class ResNet18(nn.Module):
    """
    ResNet-18 backbone for feature extraction.
    
    Returns feature map of size (H/8, W/8, 256) or configurable.
    """
    num_classes: Optional[int] = None  # None for feature extraction only
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Initial conv
        x = nn.Conv(64, (7, 7), strides=2, padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=2, padding='SAME')
        
        # Layer 1: 64 filters, 2 blocks
        for _ in range(2):
            x = ResidualBlock(64)(x, train)
        
        # Layer 2: 128 filters, 2 blocks
        x = ResidualBlock(128, stride=2)(x, train)
        x = ResidualBlock(128)(x, train)
        
        # Layer 3: 256 filters, 2 blocks
        x = ResidualBlock(256, stride=2)(x, train)
        x = ResidualBlock(256)(x, train)
        
        # Layer 4: 512 filters, 2 blocks (optional, can stop at 256)
        # For dense matching, we typically use earlier features
        # Uncomment if you want deeper features:
        # x = ResidualBlock(512, stride=2)(x, train)
        # x = ResidualBlock(512)(x, train)
        
        if self.num_classes is not None:
            # Classification head
            x = jnp.mean(x, axis=(1, 2))  # Global average pooling
            x = nn.Dense(self.num_classes)(x)
            return x
        
        # Return feature map for dense matching
        return x  # Shape: (B, H/16, W/16, 256)


# ============================================================================
# Transformer Blocks
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Self-attention transformer block.
    
    Features:
    - Multi-head self-attention
    - Feed-forward MLP
    - Layer normalization
    - Residual connections
    """
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train
        )(y, y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        x = x + y
        
        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        x = x + y
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention transformer block.
    
    Query from A, Key/Value from B.
    """
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, q_input, kv_input, train: bool = True):
        # Cross-attention: Q from q_input, K/V from kv_input
        y = nn.LayerNorm()(q_input)
        kv = nn.LayerNorm()(kv_input)
        
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train
        )(y, kv)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        x = q_input + y
        
        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        x = x + y
        
        return x


# ============================================================================
# Siamese Transformer Encoder
# ============================================================================

class SiameseTransformer(nn.Module):
    """
    Siamese transformer that processes two feature maps with:
    - Self-attention on each
    - Cross-attention between them
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        mlp_dim: MLP dimension
        dropout_rate: Dropout rate
    """
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, feat1, feat2, train: bool = True):
        """
        Args:
            feat1: (B, H, W, C) feature map from image 1
            feat2: (B, H, W, C) feature map from image 2
            train: Training mode
            
        Returns:
            refined_feat1: (B, H, W, C)
            refined_feat2: (B, H, W, C)
        """
        B, H, W, C = feat1.shape
        
        # Project to hidden_dim if needed
        if C != self.hidden_dim:
            feat1 = nn.Dense(self.hidden_dim)(feat1)
            feat2 = nn.Dense(self.hidden_dim)(feat2)
        
        # Flatten spatial dimensions for transformer
        # (B, H, W, C) -> (B, H*W, C)
        feat1_flat = feat1.reshape(B, H * W, -1)
        feat2_flat = feat2.reshape(B, H * W, -1)
        
        # Add positional encoding
        feat1_flat = feat1_flat + self.param(
            'pos_encoding_1',
            nn.initializers.normal(stddev=0.02),
            (1, H * W, self.hidden_dim)
        )
        feat2_flat = feat2_flat + self.param(
            'pos_encoding_2',
            nn.initializers.normal(stddev=0.02),
            (1, H * W, self.hidden_dim)
        )
        
        # Transformer layers
        for i in range(self.num_layers):
            # Self-attention on feat1
            feat1_flat = TransformerBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'self_attn_1_layer_{i}'
            )(feat1_flat, train)
            
            # Self-attention on feat2
            feat2_flat = TransformerBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'self_attn_2_layer_{i}'
            )(feat2_flat, train)
            
            # Cross-attention: feat1 ← feat2
            feat1_flat = CrossAttentionBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'cross_attn_1_layer_{i}'
            )(feat1_flat, feat2_flat, train)
            
            # Cross-attention: feat2 ← feat1
            feat2_flat = CrossAttentionBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'cross_attn_2_layer_{i}'
            )(feat2_flat, feat1_flat, train)
        
        # Reshape back to spatial
        feat1_out = feat1_flat.reshape(B, H, W, self.hidden_dim)
        feat2_out = feat2_flat.reshape(B, H, W, self.hidden_dim)
        
        return feat1_out, feat2_out


# ============================================================================
# Dense Matching Head (Dual-Softmax)
# ============================================================================

class DenseMatchingHead(nn.Module):
    """
    Dense matching head using dual-softmax correlation.
    
    Computes correlation matrix between two feature maps and applies
    dual-softmax to get soft correspondences.
    """
    temperature: float = 0.1
    
    def correlation_matrix(self, feat1, feat2):
        """
        Compute correlation matrix between flattened features.
        
        Args:
            feat1: (B, H, W, C)
            feat2: (B, H, W, C)
            
        Returns:
            corr: (B, H*W, H*W) correlation matrix
        """
        B, H, W, C = feat1.shape
        
        # Flatten and normalize
        feat1_flat = feat1.reshape(B, H * W, C)
        feat2_flat = feat2.reshape(B, H * W, C)
        
        feat1_norm = feat1_flat / (jnp.linalg.norm(feat1_flat, axis=-1, keepdims=True) + 1e-8)
        feat2_norm = feat2_flat / (jnp.linalg.norm(feat2_flat, axis=-1, keepdims=True) + 1e-8)
        
        # Correlation: (B, H*W, H*W)
        corr = jnp.einsum('bic,bjc->bij', feat1_norm, feat2_norm)
        
        return corr / self.temperature
    
    def dual_softmax(self, corr):
        """
        Apply dual-softmax to correlation matrix.
        
        Args:
            corr: (B, N, M) correlation scores
            
        Returns:
            P: (B, N, M) matching probability matrix
        """
        # Softmax over dimension 2 (for each point in img1, distribution over img2)
        P1 = jax.nn.softmax(corr, axis=2)
        
        # Softmax over dimension 1 (for each point in img2, distribution over img1)
        P2 = jax.nn.softmax(corr, axis=1)
        
        # Combine (geometric mean or product)
        P = jnp.sqrt(P1 * P2)
        
        return P
    
    @nn.compact
    def __call__(self, feat1, feat2):
        """
        Args:
            feat1: (B, H, W, C)
            feat2: (B, H, W, C)
            
        Returns:
            P: (B, H*W, H*W) matching probability matrix
            matches: (B, K, 2) top-K matches as indices (i, j)
        """
        # Compute correlation
        corr = self.correlation_matrix(feat1, feat2)
        
        # Dual-softmax
        P = self.dual_softmax(corr)
        
        # Extract top matches (for visualization/eval)
        # Take max along axis 2 for each point in img1
        max_vals = jnp.max(P, axis=2)  # (B, H*W)
        
        # Top-K matches
        K = 100  # Number of top matches to return
        top_k_indices = jnp.argsort(-max_vals, axis=1)[:, :K]  # (B, K)
        
        # For each top-k index in img1, find corresponding index in img2
        batch_indices = jnp.arange(P.shape[0])[:, None]
        matches_j = jnp.argmax(P[batch_indices, top_k_indices], axis=2)  # (B, K)
        
        # Stack into (B, K, 2) where matches[b, k] = [i, j]
        matches = jnp.stack([top_k_indices, matches_j], axis=2)
        
        return P, matches


# ============================================================================
# Module 1: Dense Registration Model
# ============================================================================

class DenseRegModel(nn.Module):
    """
    Complete model for Module 1: Dense Registration.
    
    Pipeline:
    1. ResNet-18 feature extraction
    2. Transformer refinement (LoFTR or generic Siamese)
    3. Dense matching head
    """
    image_size: int = 128
    num_transformer_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    mlp_dim: int = 1024
    dropout_rate: float = 0.1
    use_loftr: bool = False  # Use LoFTR LocalFeatureTransformer (for pretrained weights)
    attention_type: str = 'linear'  # 'linear' or 'full' (for LoFTR)
    
    @nn.compact
    def __call__(self, img1, img2, train: bool = True):
        """
        Args:
            img1: (B, H, W, 1)
            img2: (B, H, W, 1)
            train: Training mode
            
        Returns:
            P: (B, H'*W', H'*W') matching probability matrix
            matches: (B, K, 2) top matches
            feat1: (B, H', W', C) refined features
            feat2: (B, H', W', C) refined features
        """
        # Backbone
        backbone = ResNet18()
        feat1 = backbone(img1, train)  # (B, H/16, W/16, 256)
        feat2 = backbone(img2, train)
        
        B, H, W, C = feat1.shape
        
        if self.use_loftr:
            # Use LoFTR LocalFeatureTransformer
            # Project to hidden_dim if needed
            if C != self.hidden_dim:
                feat1 = nn.Dense(self.hidden_dim, name='feat_proj')(feat1)
                feat2 = nn.Dense(self.hidden_dim, name='feat_proj')(feat2)
            
            # Add positional encoding
            pos_enc = PositionalEncoding2D(
                d_model=self.hidden_dim,
                height=H,
                width=W,
                learnable=False,
                name='pos_encoding'
            )
            feat1_flat = pos_enc(feat1)  # (B, H*W, hidden_dim)
            feat2_flat = pos_enc(feat2)
            
            # Determine layer sequence (e.g., ['self', 'cross', 'self', 'cross'])
            # For num_layers=4, we use 2 self+cross pairs
            layer_names = []
            for i in range(self.num_transformer_layers // 2):
                layer_names.extend(['self', 'cross'])
            
            # LoFTR transformer
            transformer = LocalFeatureTransformer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                layer_names=tuple(layer_names),
                attention_type=self.attention_type,
                name='loftr_transformer'
            )
            feat1_flat, feat2_flat = transformer(feat1_flat, feat2_flat, train=train)
            
            # Reshape back to spatial
            feat1 = feat1_flat.reshape(B, H, W, self.hidden_dim)
            feat2 = feat2_flat.reshape(B, H, W, self.hidden_dim)
            
        else:
            # Use generic SiameseTransformer
            transformer = SiameseTransformer(
                num_layers=self.num_transformer_layers,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name='siamese_transformer'
            )
            feat1, feat2 = transformer(feat1, feat2, train)
        
        # Dense matching
        matching_head = DenseMatchingHead()
        P, matches = matching_head(feat1, feat2)
        
        return P, matches, feat1, feat2


# ============================================================================
# Module 2: Embedding & Matcher Model
# ============================================================================

class EmbeddingHead(nn.Module):
    """
    Embedding head for converting features to normalized embeddings.
    """
    embedding_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, H, W, C) or (B, C) features
            
        Returns:
            emb: (B, embedding_dim) L2-normalized embedding
        """
        # Global average pooling if spatial
        if len(x.shape) == 4:
            x = jnp.mean(x, axis=(1, 2))
        
        # Dense projection
        x = nn.Dense(self.embedding_dim)(x)
        
        # L2 normalize
        emb = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        return emb


class MatcherModel(nn.Module):
    """
    Complete model for Module 2: Fingerprint Matching.
    
    Processes global (overlapped) and local (ROI) features.
    Outputs embeddings for metric learning.
    """
    image_size: int = 224
    roi_size: int = 90
    num_transformer_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    mlp_dim: int = 1024
    dropout_rate: float = 0.1
    embedding_dim: int = 256
    use_loftr: bool = False  # Use LoFTR LocalFeatureTransformer
    attention_type: str = 'linear'  # 'linear' or 'full' (for LoFTR)
    
    @nn.compact
    def __call__(self, img1, img2, roi1, roi2, train: bool = True):
        """
        Args:
            img1: (B, H, W, 1) full image 1
            img2: (B, H, W, 1) full image 2
            roi1: (B, roi_size, roi_size, 1) ROI patch 1
            roi2: (B, roi_size, roi_size, 1) ROI patch 2
            train: Training mode
            
        Returns:
            emb_g1: (B, embedding_dim) global embedding 1
            emb_g2: (B, embedding_dim) global embedding 2
            emb_l1: (B, embedding_dim) local embedding 1
            emb_l2: (B, embedding_dim) local embedding 2
            P: (B, N, N) optional matching matrix (for L_D)
            matches: (B, K, 2) optional matches
        """
        backbone = ResNet18()
        
        # Create transformer (LoFTR or generic)
        if self.use_loftr:
            # Layer sequence for LoFTR
            layer_names = []
            for i in range(self.num_transformer_layers // 2):
                layer_names.extend(['self', 'cross'])
            
            def create_loftr_transformer(feat, name_suffix):
                """Helper to create LoFTR transformer with positional encoding."""
                B, H, W, C = feat.shape
                
                # Project if needed
                if C != self.hidden_dim:
                    feat_proj = nn.Dense(self.hidden_dim, name=f'feat_proj_{name_suffix}')(feat)
                else:
                    feat_proj = feat
                
                # Add positional encoding
                pos_enc = PositionalEncoding2D(
                    d_model=self.hidden_dim,
                    height=H,
                    width=W,
                    learnable=False,
                    name=f'pos_encoding_{name_suffix}'
                )
                return pos_enc(feat_proj), H, W
        else:
            transformer = SiameseTransformer(
                num_layers=self.num_transformer_layers,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name='siamese_transformer'
            )
        
        # Global branch (full images)
        feat_g1 = backbone(img1, train)
        feat_g2 = backbone(img2, train)
        
        if self.use_loftr:
            feat_g1_flat, H_g, W_g = create_loftr_transformer(feat_g1, 'global')
            feat_g2_flat, _, _ = create_loftr_transformer(feat_g2, 'global')
            
            loftr_global = LocalFeatureTransformer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                layer_names=tuple(layer_names),
                attention_type=self.attention_type,
                name='loftr_transformer_global'
            )
            feat_g1_flat, feat_g2_flat = loftr_global(feat_g1_flat, feat_g2_flat, train=train)
            
            # Reshape back
            B = feat_g1.shape[0]
            feat_g1 = feat_g1_flat.reshape(B, H_g, W_g, self.hidden_dim)
            feat_g2 = feat_g2_flat.reshape(B, H_g, W_g, self.hidden_dim)
        else:
            feat_g1, feat_g2 = transformer(feat_g1, feat_g2, train)
        
        # Local branch (ROI patches)
        feat_l1 = backbone(roi1, train)
        feat_l2 = backbone(roi2, train)
        
        if self.use_loftr:
            feat_l1_flat, H_l, W_l = create_loftr_transformer(feat_l1, 'local')
            feat_l2_flat, _, _ = create_loftr_transformer(feat_l2, 'local')
            
            loftr_local = LocalFeatureTransformer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                layer_names=tuple(layer_names),
                attention_type=self.attention_type,
                name='loftr_transformer_local'
            )
            feat_l1_flat, feat_l2_flat = loftr_local(feat_l1_flat, feat_l2_flat, train=train)
            
            # Reshape back
            feat_l1 = feat_l1_flat.reshape(B, H_l, W_l, self.hidden_dim)
            feat_l2 = feat_l2_flat.reshape(B, H_l, W_l, self.hidden_dim)
        else:
            feat_l1, feat_l2 = transformer(feat_l1, feat_l2, train)
        
        # Embedding heads
        emb_head = EmbeddingHead(embedding_dim=self.embedding_dim)
        emb_g1 = emb_head(feat_g1)
        emb_g2 = emb_head(feat_g2)
        emb_l1 = emb_head(feat_l1)
        emb_l2 = emb_head(feat_l2)
        
        # Optional: Dense matching (for L_D auxiliary loss)
        matching_head = DenseMatchingHead()
        P, matches = matching_head(feat_g1, feat_g2)
        
        return emb_g1, emb_g2, emb_l1, emb_l2, P, matches
