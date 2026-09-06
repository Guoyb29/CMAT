#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TrajectoryGenerator(nn.Module):
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 1024,
        alpha: float = 0.0,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.alpha = alpha
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_genes * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
        )
        
        # Gating network (per-gene modulation)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_genes),
            nn.Sigmoid(),
        )
        
        # Residual network (per-gene perturbation)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_genes),
            nn.Tanh(),
        )
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        pseudotime: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate inputs
        x = torch.cat([source, target, pseudotime], dim=1)
        
        # Extract features
        features = self.backbone(x)
        
        # Compute gate and residual
        gate_weights = self.gate(features)
        residual_values = self.residual(features)
        
        # Linear interpolation baseline
        baseline = (1 - pseudotime) * source + pseudotime * target
        
        # Modulated output (α=0 -> pure baseline)
        output = baseline + self.alpha * (gate_weights * residual_values)
        
        return output, gate_weights


class TrajectoryGeneratorLoss(nn.Module):
    
    def __init__(
        self,
        source_marker_indices: list,
        target_marker_indices: list,
    ):
        super().__init__()
        
        self.source_indices = source_marker_indices
        self.target_indices = target_marker_indices
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pseudotime: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:

        # 1. Reconstruction loss
        baseline = (1 - pseudotime) * source + pseudotime * target
        loss_recon = self.mse(pred, baseline)
        
        # 2. Pearson correlation loss (enforce directionality)
        loss_pearson = 0.0
        
        if len(self.source_indices) > 0:
            source_markers = pred[:, self.source_indices]
            loss_pearson += self._pearson_loss(source_markers, pseudotime, direction='down')
        
        if len(self.target_indices) > 0:
            target_markers = pred[:, self.target_indices]
            loss_pearson += self._pearson_loss(target_markers, pseudotime, direction='up')
        
        # 3. Monotonicity loss (penalize violations)
        loss_mono = 0.0
        
        if len(self.source_indices) > 0:
            for idx in self.source_indices:
                loss_mono += self._monotonic_loss(pred[:, idx:idx+1], pseudotime, direction='down')
        
        if len(self.target_indices) > 0:
            for idx in self.target_indices:
                loss_mono += self._monotonic_loss(pred[:, idx:idx+1], pseudotime, direction='up')
        
        # 4. Gate regularization (encourage sparsity)
        loss_gate = gate.mean()
        
        # Combined loss
        total_loss = (
            1.0 * loss_recon +
            300.0 * loss_pearson +
            120.0 * loss_mono +
            0.01 * loss_gate
        )
        
        return total_loss
    
    def _pearson_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        # Center variables
        x_centered = x - x.mean(dim=0, keepdim=True)
        t_centered = t - t.mean()
        
        # Compute Pearson correlation per marker
        numerator = (x_centered * t_centered).sum(dim=0)
        denominator = torch.sqrt((x_centered ** 2).sum(dim=0)) * torch.sqrt((t_centered ** 2).sum()) + 1e-8
        corr = numerator / denominator
        
        # Penalty based on direction
        if direction == 'down':
            penalty = (1.0 + corr).mean()  # Want corr -> -1
        else:
            penalty = (1.0 - corr).mean()  # Want corr -> +1
        
        return penalty
    
    def _monotonic_loss(
        self,
        pred: torch.Tensor,
        t: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        # Sort by pseudotime
        sorted_idx = torch.argsort(t.squeeze())
        pred_sorted = pred[sorted_idx]
        
        # Compute differences
        diff = pred_sorted[1:] - pred_sorted[:-1]
        
        # Penalize violations
        if direction == 'down':
            violation = torch.relu(diff)  # Penalize increases
        else:
            violation = torch.relu(-diff)  # Penalize decreases
        
        return violation.mean()
