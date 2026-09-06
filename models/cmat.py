#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMAT: Cross-Modal Attention-based Trajectory inference model
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .modality_encoder import ModalitySpecificEncoders
from .cmat_attention import CMATAttentionEncoder
from .trajectory_generator import TrajectoryGenerator


class CMAT(nn.Module):
    """
    Cross-Modal Attention Framework for developmental trajectory inference.
    
    Combines modality-specific encoders, cross-modal attention, and
    trajectory generation for multimodal single-cell data.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        latent_dim: int = 128,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.latent_dim = latent_dim
        
        # Modality-specific encoders
        self.encoders = ModalitySpecificEncoders(
            modality_dims=modality_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        
        # Cross-modal attention
        self.attention = CMATAttentionEncoder(
            d_model=latent_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers,
            dropout=dropout,
        )
        
        # Trajectory generator
        self.generator = TrajectoryGenerator(
            latent_dim=latent_dim,
            output_dim=latent_dim,
        )
    
    def forward(
        self,
        modality_data: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ):
        """
        Forward pass through CMAT.
        
        Args:
            modality_data: Dict mapping modality names to tensors [batch, features]
            return_attention: Whether to return attention weights
            
        Returns:
            embeddings: Integrated multimodal embeddings [batch, latent_dim]
            trajectory: Generated trajectory coordinates [batch, latent_dim]
            attention_weights: (optional) Cross-modal attention weights
        """
        # Encode each modality
        encoded = self.encoders(modality_data)  # Dict[str, Tensor]
        
        # Stack modality embeddings
        modality_names = sorted(encoded.keys())
        embeddings = torch.stack([encoded[name] for name in modality_names], dim=1)
        # Shape: [batch, num_modalities, latent_dim]
        
        # Apply cross-modal attention
        if return_attention:
            integrated, attention_weights = self.attention(embeddings, return_attention=True)
            # integrated: [batch, num_modalities, latent_dim]
        else:
            integrated = self.attention(embeddings)
            attention_weights = None
        
        # Aggregate across modalities (mean pooling)
        aggregated = integrated.mean(dim=1)  # [batch, latent_dim]
        
        # Generate trajectory
        trajectory = self.generator(aggregated)  # [batch, latent_dim]
        
        if return_attention:
            return aggregated, trajectory, attention_weights
        else:
            return aggregated, trajectory
    
    def encode(self, modality_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode data to latent space without trajectory generation."""
        embeddings, _ = self.forward(modality_data, return_attention=False)
        return embeddings
    
    def generate_trajectory(self, embedding: torch.Tensor) -> torch.Tensor:
        """Generate trajectory from latent embedding."""
        return self.generator(embedding)
