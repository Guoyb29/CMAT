#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modality-specific encoders for CMAT.

Each modality (RNA, ATAC, MET) has its own encoder that projects
high-dimensional input to a shared hidden space.
"""

from typing import Dict, Mapping

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """
    Single-modality encoder with BatchNorm and Dropout.
    
    Architecture: Linear -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        """
        Args:
            input_dim: Input dimension (e.g., number of genes, peaks, CpGs)
            hidden_dim: Output hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, input_dim)
        
        Returns:
            Encoded representation, shape (batch_size, hidden_dim)
        """
        return self.encoder(x)


class ModalitySpecificEncoders(nn.Module):
    """
    Container for multiple modality-specific encoders.
    
    Manages separate encoders for each modality (RNA, ATAC, MET)
    and ensures all inputs are present during forward pass.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality names to input dimensions
                          e.g., {'RNA': 2000, 'ATAC': 50000, 'MET': 10000}
            hidden_dim: Shared hidden dimension for all modalities
            dropout: Dropout probability
        """
        super().__init__()

        if len(modality_dims) == 0:
            raise ValueError("modality_dims must contain at least one modality")

        self.modality_order = list(modality_dims.keys())

        # Create separate encoder for each modality
        self.encoders = nn.ModuleDict(
            {
                modality_name: ModalityEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for modality_name, input_dim in modality_dims.items()
            }
        )

    def forward(
        self,
        modalities: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all modalities.
        
        Args:
            modalities: Dictionary of input tensors
                       e.g., {'RNA': tensor(batch, 2000), 'ATAC': tensor(batch, 50000)}
        
        Returns:
            Dictionary of encoded representations
            e.g., {'RNA': tensor(batch, 256), 'ATAC': tensor(batch, 256)}
        
        Raises:
            KeyError: If any expected modality is missing from input
        """
        # Check for missing modalities
        missing_modalities = [
            name for name in self.modality_order
            if name not in modalities
        ]

        if missing_modalities:
            raise KeyError(f"Missing modality inputs: {missing_modalities}")

        # Encode each modality
        encoded_modalities = {
            modality_name: self.encoders[modality_name](modalities[modality_name])
            for modality_name in self.modality_order
        }

        return encoded_modalities
