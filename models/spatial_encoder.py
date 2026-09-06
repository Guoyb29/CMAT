#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    
    def __init__(
        self,
        rna_dim: int,
        spatial_dim: int = 2,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
    ):
        super().__init__()
        
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        fusion_input_dim = hidden_dim // 2 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(self, rna: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        rna_emb = self.rna_encoder(rna)
        spatial_emb = self.spatial_encoder(spatial)
        
        fused = torch.cat([rna_emb, spatial_emb], dim=1)
        embedding = self.fusion(fused)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
