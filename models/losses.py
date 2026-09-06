#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for CMAT model training.

Includes:
  - Lineage alignment loss (contrastive learning)
  - Trajectory generator loss (with marker constraints)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


def lineage_alignment_loss(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    correspondence_matrix: Union[np.ndarray, torch.Tensor],
    temperature: float = 0.15,
    positive_threshold: float = 0.25,
) -> torch.Tensor:
    """
    Bidirectional contrastive loss for cross-stage lineage alignment.
    
    Uses a prior correspondence matrix to define positive pairs across
    developmental stages, encouraging embeddings from corresponding lineages
    to be close in the latent space.
    
    Args:
        source_emb: Source stage embeddings, shape (batch_size, embedding_dim)
        target_emb: Target stage embeddings, shape (batch_size, embedding_dim)
        source_labels: Source cell type labels, shape (batch_size,)
        target_labels: Target cell type labels, shape (batch_size,)
        correspondence_matrix: Prior lineage correspondence probabilities,
                              shape (n_source_types, n_target_types)
        temperature: Temperature scaling for contrastive loss
        positive_threshold: Minimum correspondence probability to consider
                           a pair as positive
    
    Returns:
        Scalar loss value (average of source->target and target->source)
    """
    device = source_emb.device
    batch_size = source_emb.size(0)

    if target_emb.size(0) != batch_size:
        raise ValueError(
            f"Batch size mismatch: source_emb {source_emb.size(0)} "
            f"vs target_emb {target_emb.size(0)}"
        )

    # Convert correspondence matrix to tensor
    correspondence_matrix = torch.as_tensor(
        correspondence_matrix, dtype=torch.float32, device=device
    )

    # L2-normalize embeddings for cosine similarity
    source_emb = F.normalize(source_emb, p=2, dim=1)
    target_emb = F.normalize(target_emb, p=2, dim=1)

    # Compute temperature-scaled similarity matrix
    similarity = torch.matmul(source_emb, target_emb.T) / temperature

    # Build positive pair mask from correspondence matrix
    positive_mask = torch.zeros(batch_size, batch_size, device=device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            source_state = int(source_labels[i].item())
            target_state = int(target_labels[j].item())
            
            if correspondence_matrix[source_state, target_state] > positive_threshold:
                positive_mask[i, j] = 1.0

    # Fallback: if no positive pairs, use weak uniform prior
    if positive_mask.sum().item() == 0:
        positive_mask = torch.ones_like(positive_mask) * 0.15

    def compute_side_loss(sim: torch.Tensor, mask: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """
        Compute one-sided contrastive loss (source->target or target->source).
        
        For each anchor, compute InfoNCE loss:
            -log( exp(sim_positive) / sum(exp(sim_all)) )
        
        Args:
            sim: Similarity matrix (batch_size, batch_size)
            mask: Positive pair mask (batch_size, batch_size)
            axis: 0 for source->target (row-wise), 1 for target->source (column-wise)
        
        Returns:
            Average contrastive loss over valid anchors
        """
        loss = 0.0
        n_valid = 0

        for i in range(sim.shape[axis]):
            # Extract logits and positive mask for this anchor
            if axis == 0:
                pos_mask = mask[i]
                logits = sim[i]
            else:
                pos_mask = mask[:, i]
                logits = sim[:, i]

            # Skip if no positive pairs
            if pos_mask.sum() == 0:
                continue

            # Indices of positive samples
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            pos_logits = logits[pos_indices]

            # InfoNCE loss components
            log_pos = torch.logsumexp(pos_logits, dim=0) - np.log(len(pos_indices))
            log_all = torch.logsumexp(logits, dim=0)

            loss += -(log_pos - log_all)
            n_valid += 1

        if n_valid > 0:
            return loss / n_valid
        
        return torch.tensor(0.0, device=device)

    # Bidirectional contrastive loss
    source_to_target_loss = compute_side_loss(similarity, positive_mask, axis=0)
    target_to_source_loss = compute_side_loss(similarity, positive_mask, axis=1)

    total_loss = (source_to_target_loss + target_to_source_loss) / 2.0

    return total_loss


class TrajectoryGeneratorLoss(nn.Module):
    """
    Multi-objective loss for trajectory generator training.
    
    Combines:
      1. Reconstruction loss (MSE to linear interpolation baseline)
      2. Pearson correlation loss (enforce marker directionality)
      3. Monotonicity loss (penalize non-monotonic marker changes)
      4. Gate regularization (encourage sparsity)
    """

    def __init__(
        self,
        source_marker_indices: list,
        target_marker_indices: list,
        weight_recon: float = 1.0,
        weight_pearson: float = 300.0,
        weight_mono: float = 120.0,
        weight_gate: float = 0.01,
    ):
        """
        Args:
            source_marker_indices: Indices of genes expected to decrease over time
            target_marker_indices: Indices of genes expected to increase over time
            weight_recon: Weight for reconstruction loss
            weight_pearson: Weight for Pearson correlation loss
            weight_mono: Weight for monotonicity loss
            weight_gate: Weight for gate regularization
        """
        super().__init__()

        self.source_indices = source_marker_indices
        self.target_indices = target_marker_indices
        
        self.weight_recon = weight_recon
        self.weight_pearson = weight_pearson
        self.weight_mono = weight_mono
        self.weight_gate = weight_gate
        
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pseudotime: torch.Tensor,
        gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted gene expression, shape (batch_size, n_genes)
            source: Source endpoint expression, shape (batch_size, n_genes)
            target: Target endpoint expression, shape (batch_size, n_genes)
            pseudotime: Pseudotime values, shape (batch_size, 1)
            gate: Gate weights from trajectory generator, shape (batch_size, n_genes)
        
        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Dictionary of individual loss values (for logging)
        """
        # 1. Reconstruction loss (MSE to linear baseline)
        baseline = (1 - pseudotime) * source + pseudotime * target
        loss_recon = self.mse(pred, baseline)

        # 2. Pearson correlation loss (directionality)
        loss_pearson = 0.0
        
        if len(self.source_indices) > 0:
            source_markers = pred[:, self.source_indices]
            loss_pearson += self._pearson_loss(
                source_markers, pseudotime, direction='down'
            )

        if len(self.target_indices) > 0:
            target_markers = pred[:, self.target_indices]
            loss_pearson += self._pearson_loss(
                target_markers, pseudotime, direction='up'
            )

        # 3. Monotonicity loss (penalize violations)
        loss_mono = 0.0

        if len(self.source_indices) > 0:
            for idx in self.source_indices:
                loss_mono += self._monotonic_loss(
                    pred[:, idx:idx+1], pseudotime, direction='down'
                )

        if len(self.target_indices) > 0:
            for idx in self.target_indices:
                loss_mono += self._monotonic_loss(
                    pred[:, idx:idx+1], pseudotime, direction='up'
                )

        # Normalize monotonicity loss by number of markers
        n_markers = len(self.source_indices) + len(self.target_indices)
        if n_markers > 0:
            loss_mono = loss_mono / n_markers

        # 4. Gate regularization (sparsity)
        loss_gate = gate.mean()

        # Combined loss
        total_loss = (
            self.weight_recon * loss_recon +
            self.weight_pearson * loss_pearson +
            self.weight_mono * loss_mono +
            self.weight_gate * loss_gate
        )

        # Loss dictionary for logging
        loss_dict = {
            'recon': loss_recon.item(),
            'pearson': loss_pearson if isinstance(loss_pearson, float) else loss_pearson.item(),
            'mono': loss_mono.item() if isinstance(loss_mono, torch.Tensor) else loss_mono,
            'gate': loss_gate.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict

    def _pearson_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        """
        Compute Pearson correlation penalty for marker directionality.
        
        Args:
            x: Marker expression, shape (batch_size, n_markers)
            t: Pseudotime, shape (batch_size, 1)
            direction: 'up' (positive correlation) or 'down' (negative correlation)
        
        Returns:
            Penalty encouraging correct correlation sign
        """
        # Center variables (zero mean)
        x_centered = x - x.mean(dim=0, keepdim=True)
        t_centered = t - t.mean()

        # Compute Pearson correlation per marker
        numerator = (x_centered * t_centered).sum(dim=0)
        x_std = torch.sqrt((x_centered ** 2).sum(dim=0)) + 1e-8
        t_std = torch.sqrt((t_centered ** 2).sum()) + 1e-8
        
        corr = numerator / (x_std * t_std)

        # Penalty based on desired direction
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
        """
        Penalize non-monotonic changes in marker expression.
        
        Args:
            pred: Single marker predictions, shape (batch_size, 1)
            t: Pseudotime, shape (batch_size, 1)
            direction: 'up' (should increase) or 'down' (should decrease)
        
        Returns:
            Average violation magnitude
        """
        # Sort predictions by pseudotime
        sorted_idx = torch.argsort(t.squeeze())
        pred_sorted = pred[sorted_idx]

        # Compute consecutive differences
        diff = pred_sorted[1:] - pred_sorted[:-1]

        # Penalize violations
        if direction == 'down':
            violation = torch.relu(diff)  # Penalize increases
        else:
            violation = torch.relu(-diff)  # Penalize decreases

        return violation.mean()
