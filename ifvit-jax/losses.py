"""
Loss functions and score fusion for IFViT.

Includes:
- Dense registration loss (L_D)
- Cosine embedding loss (L_E)
- ArcFace loss (L_A)
- Combined losses for Module 1 and Module 2
- Score fusion for verification
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict
import flax.linen as nn


# ============================================================================
# Dense Registration Loss (L_D)
# ============================================================================

def dense_reg_loss(
    P: jnp.ndarray,
    gt_matches: jnp.ndarray,
    valid_mask: Optional[jnp.ndarray] = None,
    feature_shape: Tuple[int, int] = None
) -> jnp.ndarray:
    """
    Dense correspondence loss using dual-softmax matching probability.
    
    Args:
        P: (B, N, M) matching probability matrix from dual-softmax
           where N = H*W (flattened feature map size)
        gt_matches: (B, K, 4) ground-truth matches as [x1, y1, x2, y2]
                    OR (B, N, M) dense GT probability matrix
        valid_mask: Optional (B, K) or (B, N, M) mask for valid matches
        feature_shape: (H, W) shape of feature map for coordinate conversion
        
    Returns:
        loss: Scalar loss value
        
    The loss encourages P[i, j] to be high for ground-truth correspondences.
    """
    B = P.shape[0]
    N = P.shape[1]
    
    if len(gt_matches.shape) == 3 and gt_matches.shape[-1] == 4:
        # gt_matches is (B, K, 4) format - convert to indices
        # feature_shape must be provided to avoid concretization errors
        if feature_shape is None:
            # Fallback: compute from N, but use JAX operations only
            # This should rarely happen if feature_shape is passed correctly
            # Ensure N is a JAX array before sqrt
            N_array = jnp.array(N, dtype=jnp.float32)
            sqrt_N = jnp.sqrt(N_array)
            H = W = sqrt_N.astype(jnp.int32)
        else:
            # Convert to JAX arrays if they're not already
            if isinstance(feature_shape, (tuple, list)):
                H = jnp.array(feature_shape[0], dtype=jnp.int32)
                W = jnp.array(feature_shape[1], dtype=jnp.int32)
            else:
                # Already JAX arrays or single value
                H, W = feature_shape
        
        # Extract coordinates
        x1 = gt_matches[..., 0]  # (B, K)
        y1 = gt_matches[..., 1]
        x2 = gt_matches[..., 2]
        y2 = gt_matches[..., 3]
        
        # Convert to feature map coordinates (scale down by 8 for ResNet)
        scale = 8  # Assuming H_feat = H_img / 8
        x1_feat = (x1 / scale).astype(jnp.int32)
        y1_feat = (y1 / scale).astype(jnp.int32)
        x2_feat = (x2 / scale).astype(jnp.int32)
        y2_feat = (y2 / scale).astype(jnp.int32)
        
        # Clamp to valid range
        # Use JAX arrays for bounds to avoid concretization errors
        W_int = W.astype(jnp.int32) if isinstance(W, jnp.ndarray) else jnp.array(W, dtype=jnp.int32)
        H_int = H.astype(jnp.int32) if isinstance(H, jnp.ndarray) else jnp.array(H, dtype=jnp.int32)
        x1_feat = jnp.clip(x1_feat, 0, W_int - 1)
        y1_feat = jnp.clip(y1_feat, 0, H_int - 1)
        x2_feat = jnp.clip(x2_feat, 0, W_int - 1)
        y2_feat = jnp.clip(y2_feat, 0, H_int - 1)
        
        # Convert to flat indices
        # Use JAX arrays for W
        idx1 = y1_feat * W_int + x1_feat  # (B, K)
        idx2 = y2_feat * W_int + x2_feat  # (B, K)
        
        # Gather probabilities
        batch_indices = jnp.arange(B)[:, None]  # (B, 1)
        probs = P[batch_indices, idx1, idx2]  # (B, K)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            probs = probs * valid_mask
            num_valid = jnp.sum(valid_mask, axis=1, keepdims=True) + 1e-8
        else:
            num_valid = gt_matches.shape[1]
        
        # Negative log likelihood
        loss = -jnp.log(probs + 1e-8)
        loss = jnp.sum(loss, axis=1) / num_valid
        loss = jnp.mean(loss)
        
    else:
        # gt_matches is already a probability matrix (B, N, M)
        # Use cross-entropy loss
        if valid_mask is not None:
            loss = -jnp.sum(gt_matches * jnp.log(P + 1e-8) * valid_mask) / jnp.sum(valid_mask)
        else:
            loss = -jnp.mean(gt_matches * jnp.log(P + 1e-8))
    
    return loss


# ============================================================================
# Cosine Embedding Loss (L_E)
# ============================================================================

def cosine_embedding_loss(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    labels: jnp.ndarray,
    margin: float = 0.2
) -> jnp.ndarray:
    """
    Cosine embedding loss for metric learning.
    
    Args:
        emb1: (B, D) embeddings from image 1
        emb2: (B, D) embeddings from image 2
        labels: (B,) labels, 1 for genuine pairs, -1 for imposter pairs
        margin: Margin for negative pairs
        
    Returns:
        loss: Scalar loss value
        
    For genuine pairs (label=1): minimize 1 - cosine_sim
    For imposter pairs (label=-1): minimize max(0, cosine_sim - margin)
    """
    # Compute cosine similarity
    # Embeddings should already be L2-normalized
    cosine_sim = jnp.sum(emb1 * emb2, axis=-1)  # (B,)
    
    # Loss for genuine pairs: 1 - similarity
    genuine_loss = 1.0 - cosine_sim
    
    # Loss for imposter pairs: max(0, similarity - margin)
    imposter_loss = jnp.maximum(0.0, cosine_sim - margin)
    
    # Select based on label
    # labels: 1 for genuine, -1 for imposter
    loss = jnp.where(labels > 0, genuine_loss, imposter_loss)
    
    return jnp.mean(loss)


# ============================================================================
# ArcFace Loss (L_A)
# ============================================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for discriminative feature learning.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of embeddings
        scale: Scale parameter (s)
        margin: Angular margin (m)
    """
    num_classes: int
    embedding_dim: int
    scale: float = 30.0
    margin: float = 0.5
    
    @nn.compact
    def __call__(self, embeddings: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) class labels
            
        Returns:
            loss: Scalar cross-entropy loss with angular margin
        """
        # Weight matrix (D, num_classes)
        W = self.param(
            'weight',
            nn.initializers.normal(stddev=0.01),
            (self.embedding_dim, self.num_classes)
        )
        
        # L2 normalize weights
        W_norm = W / (jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
        
        # Cosine similarity
        logits = jnp.matmul(embeddings, W_norm)  # (B, num_classes)
        
        # Get target logits
        batch_indices = jnp.arange(embeddings.shape[0])
        target_logits = logits[batch_indices, labels]  # (B,)
        
        # Compute angles
        theta = jnp.arccos(jnp.clip(target_logits, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add margin
        target_logits_margin = jnp.cos(theta + self.margin)
        
        # Replace target logits with margin-added version
        one_hot = jax.nn.one_hot(labels, self.num_classes)
        logits_margin = logits * (1 - one_hot) + target_logits_margin[:, None] * one_hot
        
        # Scale
        logits_margin = logits_margin * self.scale
        
        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(logits_margin, axis=-1)
        loss = -jnp.sum(one_hot * log_probs, axis=-1)
        
        return jnp.mean(loss)


def arcface_loss(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    params: Dict,
    scale: float = 30.0,
    margin: float = 0.5
) -> Tuple[jnp.ndarray, Dict]:
    """
    Functional interface for ArcFace loss.
    
    Args:
        embeddings: (B, D) embeddings
        labels: (B,) class IDs
        num_classes: Total number of classes
        params: Model parameters dict (contains 'arcface' subdict)
        scale: Scale parameter (s)
        margin: Angular margin (m)
        
    Returns:
        loss: Scalar loss
        updated_params: Updated parameters dict
    """
    embedding_dim = embeddings.shape[-1]
    
    # Initialize ArcFace module
    arcface_module = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        scale=scale,
        margin=margin
    )
    
    # Get or initialize params
    if 'arcface' not in params:
        # Initialize
        init_rng = jax.random.PRNGKey(0)
        arcface_params = arcface_module.init(init_rng, embeddings, labels)
        params['arcface'] = arcface_params
    
    # Apply
    loss = arcface_module.apply(params['arcface'], embeddings, labels)
    
    return loss, params


# ============================================================================
# Combined Losses
# ============================================================================

def total_loss_dense(
    P: jnp.ndarray,
    gt_matches: jnp.ndarray,
    valid_mask: Optional[jnp.ndarray] = None,
    lambda_D: float = 1.0,
    feature_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, jnp.ndarray]:
    """
    Total loss for Module 1 (Dense Registration).
    
    Args:
        P: Matching probability matrix
        gt_matches: Ground-truth matches
        valid_mask: Valid match mask
        lambda_D: Weight for dense loss
        feature_shape: Feature map shape
        
    Returns:
        Dictionary with 'total', 'L_D'
    """
    L_D = dense_reg_loss(P, gt_matches, valid_mask, feature_shape)
    total = lambda_D * L_D
    
    return {
        'total': total,
        'L_D': L_D
    }


def total_loss_matcher(
    emb_g1: jnp.ndarray,
    emb_g2: jnp.ndarray,
    emb_l1: jnp.ndarray,
    emb_l2: jnp.ndarray,
    labels_pair: jnp.ndarray,
    class_id1: jnp.ndarray,
    class_id2: jnp.ndarray,
    num_classes: int,
    arcface_params: Dict,
    P: Optional[jnp.ndarray] = None,
    gt_matches: Optional[jnp.ndarray] = None,
    valid_mask: Optional[jnp.ndarray] = None,
    lambda_D: float = 0.5,
    lambda_E: float = 0.1,
    lambda_A: float = 1.0,
    embedding_margin: float = 0.2,
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.5,
    feature_shape: Optional[Tuple[int, int]] = None
) -> Tuple[Dict[str, jnp.ndarray], Dict]:
    """
    Total loss for Module 2 (Matcher).
    
    Args:
        emb_g1, emb_g2: Global embeddings
        emb_l1, emb_l2: Local embeddings
        labels_pair: Pair labels (1 genuine, -1 imposter)
        class_id1, class_id2: Class IDs for ArcFace
        num_classes: Total number of classes
        arcface_params: ArcFace parameters
        P: Optional matching probability (for L_D)
        gt_matches: Optional GT matches (for L_D)
        valid_mask: Optional validity mask
        lambda_D, lambda_E, lambda_A: Loss weights
        embedding_margin: Margin for embedding loss
        arcface_scale, arcface_margin: ArcFace hyperparameters
        feature_shape: Feature map shape
        
    Returns:
        losses: Dictionary with 'total', 'L_D', 'L_E', 'L_A'
        updated_arcface_params: Updated ArcFace parameters
    """
    losses = {}
    
    # L_D: Dense correspondence loss (optional)
    if P is not None and gt_matches is not None:
        L_D = dense_reg_loss(P, gt_matches, valid_mask, feature_shape)
        losses['L_D'] = L_D
    else:
        L_D = 0.0
        losses['L_D'] = jnp.array(0.0)
    
    # L_E: Cosine embedding loss (global + local)
    L_E_global = cosine_embedding_loss(emb_g1, emb_g2, labels_pair, embedding_margin)
    L_E_local = cosine_embedding_loss(emb_l1, emb_l2, labels_pair, embedding_margin)
    L_E = (L_E_global + L_E_local) / 2.0
    losses['L_E'] = L_E
    losses['L_E_global'] = L_E_global
    losses['L_E_local'] = L_E_local
    
    # L_A: ArcFace loss (combine both embeddings)
    # Concatenate embeddings and labels
    all_embeddings = jnp.concatenate([emb_g1, emb_g2, emb_l1, emb_l2], axis=0)
    all_labels = jnp.concatenate([class_id1, class_id2, class_id1, class_id2], axis=0)
    
    L_A, updated_params = arcface_loss(
        all_embeddings,
        all_labels,
        num_classes,
        arcface_params,
        scale=arcface_scale,
        margin=arcface_margin
    )
    losses['L_A'] = L_A
    
    # Total loss
    total = lambda_D * L_D + lambda_E * L_E + lambda_A * L_A
    losses['total'] = total
    
    return losses, updated_params


# ============================================================================
# Score Computation and Fusion
# ============================================================================

def cosine_score(emb1: jnp.ndarray, emb2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cosine similarity score between embeddings.
    
    Args:
        emb1: (D,) or (B, D) embedding
        emb2: (D,) or (B, D) embedding
        
    Returns:
        score: Scalar or (B,) similarity score in [-1, 1]
    """
    # Embeddings should be L2-normalized
    score = jnp.sum(emb1 * emb2, axis=-1)
    return score


def fuse_scores(
    score_global: jnp.ndarray,
    score_local: jnp.ndarray,
    alpha_global: float = 0.6,
    alpha_local: float = 0.4
) -> jnp.ndarray:
    """
    Fuse global and local similarity scores.
    
    Args:
        score_global: Global similarity score
        score_local: Local similarity score
        alpha_global: Weight for global score
        alpha_local: Weight for local score
        
    Returns:
        fused_score: Weighted average of scores
    """
    # Ensure weights sum to 1
    total_weight = alpha_global + alpha_local
    alpha_global = alpha_global / total_weight
    alpha_local = alpha_local / total_weight
    
    fused = alpha_global * score_global + alpha_local * score_local
    return fused


def compute_matching_score(
    emb_g1: jnp.ndarray,
    emb_g2: jnp.ndarray,
    emb_l1: jnp.ndarray,
    emb_l2: jnp.ndarray,
    alpha_global: float = 0.6,
    alpha_local: float = 0.4
) -> Dict[str, jnp.ndarray]:
    """
    Compute all matching scores from embeddings.
    
    Args:
        emb_g1, emb_g2: Global embeddings
        emb_l1, emb_l2: Local embeddings
        alpha_global, alpha_local: Fusion weights
        
    Returns:
        Dictionary with 'global', 'local', and 'fused' scores
    """
    score_global = cosine_score(emb_g1, emb_g2)
    score_local = cosine_score(emb_l1, emb_l2)
    score_fused = fuse_scores(score_global, score_local, alpha_global, alpha_local)
    
    return {
        'global': score_global,
        'local': score_local,
        'fused': score_fused
    }
