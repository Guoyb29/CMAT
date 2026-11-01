#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.trajectory_generator import TrajectoryGenerator, TrajectoryGeneratorLoss


# ==============================================================================
# Marker Genes
# ==============================================================================

FGO_CORE_MARKERS = [
    "GDF9", "BMP15", "ZP1", "ZP2", "ZP3", "NOBOX", "LHX8", "FIGLA",
]

MII_CORE_MARKERS = [
    "TUBB8", "MOS", "CCNB1", "CDK1", "CDC20", "AURKC", "REC8", "SMC1B", "CENPE",
]


# ==============================================================================
# Dataset
# ==============================================================================

class CellPairDataset(Dataset):
    """Dataset of matched FGO-MII cell pairs."""
    
    def __init__(self, fgo_expr: np.ndarray, mii_expr: np.ndarray):
        self.fgo_expr = torch.FloatTensor(fgo_expr)
        self.mii_expr = torch.FloatTensor(mii_expr)
        
    def __len__(self):
        return len(self.fgo_expr)
    
    def __getitem__(self, idx):
        return {
            'source': self.fgo_expr[idx],
            'target': self.mii_expr[idx],
        }


# ==============================================================================
# Training
# ==============================================================================

def train_generator(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str,
    n_epochs: int,
    n_steps: int,
    lr: float = 1e-3,
) -> nn.Module:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    
    model.train()
    
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            batch_size = source.size(0)
            
            # Sample random pseudotimes
            pseudotime = torch.rand(batch_size, 1, device=device)
            
            # Generate predictions
            pred, gate = model(source, target, pseudotime)
            
            # Compute loss
            loss = criterion(pred, source, target, pseudotime, gate)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f}")
    
    return model


# ==============================================================================
# Trajectory Generation
# ==============================================================================

def generate_trajectories(
    model: nn.Module,
    fgo_expression: np.ndarray,
    mii_expression: np.ndarray,
    device: str,
    n_steps: int,
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    
    n_pairs = len(fgo_expression)
    n_genes = fgo_expression.shape[1]
    
    pseudotime_values = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
    
    trajectories = np.zeros((n_pairs, n_steps, n_genes), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(0, n_pairs, batch_size):
            end_idx = min(i + batch_size, n_pairs)
            
            source_batch = torch.FloatTensor(fgo_expression[i:end_idx]).to(device)
            target_batch = torch.FloatTensor(mii_expression[i:end_idx]).to(device)
            
            batch_size_actual = source_batch.size(0)
            
            # Generate trajectory for each pseudotime
            batch_trajectory = []
            for t in pseudotime_values:
                t_batch = t.expand(batch_size_actual, 1)
                pred, _ = model(source_batch, target_batch, t_batch)
                batch_trajectory.append(pred.cpu().numpy())
            
            trajectories[i:end_idx] = np.array(batch_trajectory).transpose(1, 0, 2)
    
    return trajectories


# ==============================================================================
# Matching
# ==============================================================================

def match_cells_by_embedding(
    fgo_embedding: np.ndarray,
    mii_embedding: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nbrs.fit(mii_embedding)
    
    distances, indices = nbrs.kneighbors(fgo_embedding)
    
    return indices[:, 0]


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Generate FGO->MII trajectories')
    
    parser.add_argument('--input_h5ad', type=str, required=True,
                        help='Path to input h5ad with FGO/MII expression')
    parser.add_argument('--fgo_embedding', type=str, required=True,
                        help='Path to FGO embedding .npy file')
    parser.add_argument('--mii_embedding', type=str, required=True,
                        help='Path to MII embedding .npy file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    parser.add_argument('--n_steps_train', type=int, default=100,
                        help='Number of pseudotime steps for training')
    parser.add_argument('--n_steps_generate', type=int, default=100,
                        help='Number of pseudotime steps for generation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='Hidden dimension')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Residual modulation strength')
    
    parser.add_argument('--lambda_pearson', type=float, default=300.0,
                        help='Pearson correlation loss weight')
    parser.add_argument('--lambda_mono', type=float, default=120.0,
                        help='Monotonicity loss weight')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("CMAT Trajectory Generation: FGO -> MII")
    print("="*70)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    adata = sc.read_h5ad(args.input_h5ad)
    
    fgo_mask = adata.obs['stage'] == 'FGO'
    mii_mask = adata.obs['stage'] == 'MII'
    
    fgo_adata = adata[fgo_mask].copy()
    mii_adata = adata[mii_mask].copy()
    
    fgo_expr = fgo_adata.X.toarray() if hasattr(fgo_adata.X, 'toarray') else fgo_adata.X
    mii_expr = mii_adata.X.toarray() if hasattr(mii_adata.X, 'toarray') else mii_adata.X
    
    gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)
    
    print(f"  FGO: {fgo_expr.shape[0]} cells, {n_genes} genes")
    print(f"  MII: {mii_expr.shape[0]} cells, {n_genes} genes")
    
    # Load embeddings
    print("\n[2/5] Loading embeddings and matching cells...")
    fgo_emb = np.load(args.fgo_embedding)
    mii_emb = np.load(args.mii_embedding)
    
    print(f"  FGO embedding: {fgo_emb.shape}")
    print(f"  MII embedding: {mii_emb.shape}")
    
    # Match cells
    matched_indices = match_cells_by_embedding(fgo_emb, mii_emb, k=1)
    mii_expr_matched = mii_expr[matched_indices]
    
    print(f"  Matched {len(matched_indices)} FGO-MII pairs")
    
    # Find marker indices
    print("\n[3/5] Finding marker genes...")
    
    def find_marker_indices(markers: List[str], gene_names: List[str]) -> List[int]:
        indices = []
        for marker in markers:
            if marker in gene_names:
                indices.append(gene_names.index(marker))
        return indices
    
    fgo_marker_idx = find_marker_indices(FGO_CORE_MARKERS, gene_names)
    mii_marker_idx = find_marker_indices(MII_CORE_MARKERS, gene_names)
    
    print(f"  FGO core markers found: {len(fgo_marker_idx)}/{len(FGO_CORE_MARKERS)}")
    print(f"  MII core markers found: {len(mii_marker_idx)}/{len(MII_CORE_MARKERS)}")
    
    # Create dataset
    dataset = CellPairDataset(fgo_expr, mii_expr_matched)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    print("\n[4/5] Training trajectory generator...")
    
    model = TrajectoryGenerator(
        n_genes=n_genes,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
    ).to(device)
    
    criterion = TrajectoryGeneratorLoss(
        source_marker_indices=fgo_marker_idx,
        target_marker_indices=mii_marker_idx,
    )
    
    model = train_generator(
        model=model,
        criterion=criterion,
        dataloader=dataloader,
        device=device,
        n_epochs=args.epochs,
        n_steps=args.n_steps_train,
        lr=1e-3,
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'generator_model.pt'))
    
    # Generate trajectories
    print("\n[5/5] Generating final trajectories...")
    
    trajectories = generate_trajectories(
        model=model,
        fgo_expression=fgo_expr,
        mii_expression=mii_expr_matched,
        device=device,
        n_steps=args.n_steps_generate,
        batch_size=args.batch_size,
    )
    
    # Create trajectory AnnData
    print("\n[Saving] Creating trajectory AnnData...")
    
    n_pairs = len(fgo_expr)
    n_steps = args.n_steps_generate
    
    # Reshape to [n_pairs * n_steps, n_genes]
    traj_matrix = trajectories.reshape(-1, n_genes)
    
    # Create obs
    obs = pd.DataFrame({
        'trajectory_id': np.repeat(np.arange(n_pairs), n_steps),
        'trajectory_step': np.tile(np.arange(n_steps), n_pairs),
        'pseudotime': np.tile(np.linspace(0, 1, n_steps), n_pairs),
        'stage': np.tile(
            np.where(np.arange(n_steps) == 0, 'FGO',
                     np.where(np.arange(n_steps) == n_steps - 1, 'MII', 'generated')),
            n_pairs,
        ),
        'path': 'FGO_to_MII',
        'stage_from': 'FGO',
        'stage_to': 'MII',
        'source_fgo_cell': np.repeat(fgo_adata.obs_names, n_steps),
        'matched_mii_cell': np.repeat(mii_adata.obs_names[matched_indices], n_steps),
    })
    
    # Create var
    var = pd.DataFrame(index=gene_names)
    
    # Create AnnData
    traj_adata = sc.AnnData(X=traj_matrix, obs=obs, var=var)
    
    # Add metadata
    traj_adata.uns['trajectory_direction'] = 'FGO_to_MII'
    traj_adata.uns['n_steps_generate'] = n_steps
    traj_adata.uns['n_steps_train'] = args.n_steps_train
    traj_adata.uns['alpha'] = args.alpha
    traj_adata.uns['seed'] = args.seed
    traj_adata.uns['fgo_core_markers'] = FGO_CORE_MARKERS
    traj_adata.uns['mii_core_markers'] = MII_CORE_MARKERS
    traj_adata.uns['lambda_pearson'] = args.lambda_pearson
    traj_adata.uns['lambda_mono'] = args.lambda_mono
    
    # Save
    output_path = os.path.join(args.output_dir, 'FGO_to_MII_trajectories.h5ad')
    traj_adata.write_h5ad(output_path)
    
    print(f"\n  Saved trajectory AnnData:")
    print(f"    Shape: {traj_adata.shape}")
    print(f"    Trajectories: {n_pairs}")
    print(f"    Steps per trajectory: {n_steps}")
    print(f"    Output: {output_path}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
