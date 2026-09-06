#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-modal attention mechanism for CMAT.

Implements multi-head self-attention across modalities followed by 
learned modality-wise weighting for fusion.
"""

from typing import Mapping, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modality_encoder import ModalitySpecificEncoders


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention with learned modality weighting.
    
    Architecture:
      1. Multi-head self-attention across modalities
      2. Residual connection + LayerNorm
      3. Learned attention weights for modality fusion
      4. Weighted aggregation
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_modalities: int = 3,
        n_heads: int = 4,
    ):
        """
        Args:
            d_model: Dimension of modality embeddings
            n_modalities: Number of modalities (e.g., RNA, ATAC, MET)
            n_heads: Number of attention heads
        """
        super().__init__()

        if n_modalities < 1:
            raise ValueError("n_modalities must be at least 1")

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.attention_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, n_modalities),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        modalities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modalities: Stacked modality embeddings, 
                       shape (batch_size, n_modalities, d_model)
        
        Returns:
            fused: Fused representation, shape (batch_size, d_model)
            weights: Modality attention weights, shape (batch_size, n_modalities)
        """
        # Multi-head self-attention across modalities
        attention_output, _ = self.self_attention(
            modalities, modalities, modalities
        )

        # Residual connection + LayerNorm
        modalities = self.norm1(modalities + attention_output)

        # Compute modality-wise attention weights
        context = modalities.mean(dim=1)  # (batch_size, d_model)
        scores = self.attention_net(context)  # (batch_size, n_modalities)
        weights = F.softmax(scores, dim=-1)

        # Weighted fusion
        fused = (modalities * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.norm2(fused)

        return fused, weights


class CMATAttentionEncoder(nn.Module):
    """
    Complete CMAT encoder with cross-modal attention.
    
    Pipeline:
      1. Modality-specific encoders (RNA/ATAC/MET -> hidden_dim)
      2. Cross-modal attention fusion
      3. Projection to low-dimensional embedding
      4. L2 normalization
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        n_heads: int = 4,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality names to input dimensions
                          e.g., {'RNA': 2000, 'ATAC': 50000, 'MET': 10000}
            hidden_dim: Hidden dimension for encoders and attention
            embedding_dim: Final embedding dimension
            n_heads: Number of attention heads
        """
        super().__init__()

        if len(modality_dims) == 0:
            raise ValueError("modality_dims must contain at least one modality")

        self.modality_order = list(modality_dims.keys())

        # Modality-specific encoders
        self.modality_encoders = ModalitySpecificEncoders(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            dropout=0.15,
        )

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            d_model=hidden_dim,
            n_modalities=len(self.modality_order),
            n_heads=n_heads,
        )

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(
        self,
        modalities: Mapping[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            modalities: Dictionary of modality tensors
                       e.g., {'RNA': tensor(batch, 2000), 'ATAC': tensor(batch, 50000)}
            return_attention: If True, also return attention weights
        
        Returns:
            embedding: L2-normalized embedding, shape (batch_size, embedding_dim)
            weights: (optional) Modality attention weights, shape (batch_size, n_modalities)
        """
        # Encode each modality
        encoded_modalities = self.modality_encoders(modalities)

        # Stack modalities (batch_size, n_modalities, hidden_dim)
        stacked_modalities = torch.stack(
            [encoded_modalities[name] for name in self.modality_order],
            dim=1,
        )

        # Cross-modal attention fusion
        fused, weights = self.cross_modal_attention(stacked_modalities)

        # Project to embedding space
        embedding = self.projection(fused)

        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)

        if return_attention:
            return embedding, weights

        return embedding
