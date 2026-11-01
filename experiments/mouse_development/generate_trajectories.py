#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# Import CMAT components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.trajectory_generator import TrajectoryGenerator, TrajectoryGeneratorLoss


# ==============================================================================
# Configuration
# ==============================================================================

MARKER_GENES = {
    'Epiblast': ['Pou5f1', 'Sox2', 'Nanog', 'Utf1', 'Klf4'],
    'Nascent mesoderm': ['T', 'Mesp1', 'Tbx6', 'Eomes', 'Mixl1'],
    'Mesenchyme': ['Twist1', 'Snai2', 'Vim', 'Pdgfra', 'Hand2'],
    'ExE endoderm': ['Sox17', 'Foxa2', 'Gata6', 'Gata4', 'Hhex'],
    'Brain': ['Sox2', 'Pax6', 'Neurod1', 'Tubb3', 'Rbfox3', 'Dcx'],
    'Heart': ['Tnnt2', 'Myh6', 'Myh7', 'Nkx2-5', 'Gata4', 'Mef2c'],
    'Liver': ['Alb', 'Afp', 'Apoa1', 'Hnf4a', 'Apob', 'Ttr'],
    'Cartilage primordium': ['Sox9', 'Col2a1', 'Acan', 'Prrx1', 'Col9a1'],
}

DEVELOPMENTAL_PATHS = [
    {
        'name': 'Epiblast_to_Brain',
        'source': 'Epiblast',
        'target': 'Brain',
        'source_markers': MARKER_GENES['Epiblast'],
        'target_markers': MARKER_GENES['Brain'],
    },
    {
        'name': 'Nascent_mesoderm_to_Heart',
        'source': 'Nascent mesoderm',
        'target': 'Heart',
        'source_markers': MARKER_GENES['Nascent mesoderm'],
        'target_markers': MARKER_GENES['Heart'],
    },
    {
        'name': 'ExE_endoderm_to_Liver',
        'source': 'ExE endoderm',
        'target': 'Liver',
        'source_markers': MARKER_GENES['ExE endoderm'],
        'target_markers': MARKER_GENES['Liver'],
    },
    {
        'name': 'Mesenchyme_to_Cartilage',
        'source': 'Mesenchyme',
        'target': 'Cartilage primordium',
        'source_markers': MARKER_GENES['Mesenchyme'],
        'target_markers': MARKER_GENES['Cartilage primordium'],
    },
]


# ==============================================================================
# Endpoint Pairing
# ==============================================================================

def pair_endpoints_knn(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    source_expression: np.ndarray,
    target_expression: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(target_embeddings)
    
    distances, indices = knn.kneighbors(source_embeddings)
    
    # For each source cell, randomly select one of top-k targets
    target_paired = []
    for i in range(len(source_embeddings)):
        selected_idx = np.random.choice(indices[i])
        target_paired.append(target_expression[selected_idx])
    
    target_paired = np.array(target_paired)
    
    return source_expression, target_paired


# ==============================================================================
# Dataset
# ==============================================================================

class CellPairDataset(Dataset):
    """Dataset of paired source-target cells."""
    
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = torch.FloatTensor(source)
        self.target = torch.FloatTensor(target)
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


# ==============================================================================
# Training
# ==============================================================================

def train_generator(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    n_epochs: int,
    n_steps: int,
    device: torch.device,
) -> None:

    model.train()
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for source_batch, target_batch in dataloader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            batch_size = source_batch.size(0)
            
            # Replicate for multiple pseudotime steps
            source_rep = source_batch.repeat_interleave(n_steps, dim=0)
            target_rep = target_batch.repeat_interleave(n_steps, dim=0)
            
            # Generate pseudotime sequence
            pseudotime = torch.linspace(0, 1, n_steps).repeat(batch_size).view(-1, 1).to(device)
            
            # Forward pass
            pred, gate = model(source_rep, target_rep, pseudotime)
            
            # Compute loss
            loss = criterion(pred, source_rep, target_rep, pseudotime, gate)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")


# ==============================================================================
# Trajectory Generation
# ==============================================================================

def generate_trajectories(
    model: nn.Module,
    source: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    
    n_pairs = source.shape[0]
    n_genes = source.shape[1]
    
    trajectories = np.zeros((n_pairs, n_steps, n_genes))
    
    with torch.no_grad():
        for i in range(n_pairs):
            source_cell = torch.FloatTensor(source[i:i+1]).to(device)
            target_cell = torch.FloatTensor(target[i:i+1]).to(device)
            
            # Replicate for all time steps
            source_rep = source_cell.repeat(n_steps, 1)
            target_rep = target_cell.repeat(n_steps, 1)
            
            # Generate pseudotime
            pseudotime = torch.linspace(0, 1, n_steps).view(-1, 1).to(device)
            
            # Generate trajectory
            traj, _ = model(source_rep, target_rep, pseudotime)
            
            trajectories[i] = traj.cpu().numpy()
    
    return trajectories


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate CMAT developmental trajectories')
    
    parser.add_argument('--e75_path', type=str, required=True,
                        help='Path to E7.5 h5ad file')
    parser.add_argument('--e125_path', type=str, required=True,
                        help='Path to E12.5 h5ad file')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory containing trained embeddings')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trajectories')
    parser.add_argument('--seed', type=int, default=120,
                        help='Random seed')
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--n_steps', type=int, default=120,
                        help='Number of trajectory steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--knn_k', type=int, default=10,
                        help='K for endpoint pairing')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print(f"CMAT Trajectory Generation (seed={args.seed})")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    e75 = sc.read_h5ad(args.e75_path)
    e125 = sc.read_h5ad(args.e125_path)
    
    e75_rna = e75.layers['rna'].toarray() if hasattr(e75.layers['rna'], 'toarray') else e75.layers['rna']
    e75_celltype = e75.obs['celltype'].values
    
    e125_rna = e125.X.toarray() if hasattr(e125.X, 'toarray') else e125.X
    e125_celltype = e125.obs['annotation'].values
    
    gene_names = e75.var_names.tolist()
    n_genes = len(gene_names)
    
    # Load embeddings
    print(f"\n[2/5] Loading embeddings from {args.embedding_dir}...")
    e75_embeddings = np.load(f"{args.embedding_dir}/e75_embeddings_s{args.seed}.npy")
    e125_embeddings = np.load(f"{args.embedding_dir}/e125_embeddings_s{args.seed}.npy")
    
    print(f"  E7.5 embeddings: {e75_embeddings.shape}")
    print(f"  E12.5 embeddings: {e125_embeddings.shape}")
    
    # Generate trajectories for each path
    print("\n[3/5] Generating trajectories for each developmental path...")
    
    for path in DEVELOPMENTAL_PATHS:
        path_name = path['name']
        source_type = path['source']
        target_type = path['target']
        
        print(f"\n  [{path_name}]")
        print(f"    Source: {source_type} -> Target: {target_type}")
        
        # Filter cells
        source_mask = e75_celltype == source_type
        target_mask = e125_celltype == target_type
        
        source_expr = e75_rna[source_mask]
        target_expr = e125_rna[target_mask]
        source_emb = e75_embeddings[source_mask]
        target_emb = e125_embeddings[target_mask]
        
        print(f"    Source cells: {source_expr.shape[0]}, Target cells: {target_expr.shape[0]}")
        
        # Pair endpoints
        print("    Pairing endpoints with KNN...")
        source_paired, target_paired = pair_endpoints_knn(
            source_emb, target_emb, source_expr, target_expr, k=args.knn_k
        )
        
        # Prepare dataset
        dataset = CellPairDataset(source_paired, target_paired)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Get marker indices
        source_indices = [gene_names.index(g) for g in path['source_markers'] if g in gene_names]
        target_indices = [gene_names.index(g) for g in path['target_markers'] if g in gene_names]
        
        print(f"    Source markers found: {len(source_indices)}/{len(path['source_markers'])}")
        print(f"    Target markers found: {len(target_indices)}/{len(path['target_markers'])}")
        
        # Initialize model
        model = TrajectoryGenerator(
            n_genes=n_genes,
            hidden_dim=1024,
            alpha=0.0,  # Pure baseline
        ).to(device)
        
        criterion = TrajectoryGeneratorLoss(
            source_marker_indices=source_indices,
            target_marker_indices=target_indices,
        )
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # Train
        print(f"    Training generator...")
        train_generator(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=dataloader,
            n_epochs=args.n_epochs,
            n_steps=50,
            device=device,
        )
        
        # Generate trajectories
        print(f"    Generating {args.n_steps}-step trajectories...")
        trajectories = generate_trajectories(
            model=model,
            source=source_paired,
            target=target_paired,
            n_steps=args.n_steps,
            device=device,
        )
        
        # Save trajectories
        traj_path = f"{args.output_dir}/{path_name}_trajectory_s{args.seed}.npy"
        np.save(traj_path, trajectories)
        print(f"    Saved: {traj_path}")
        # The validation and temporal-evaluation scripts consume AnnData.
        n_pairs = trajectories.shape[0]
        steps = np.tile(np.arange(args.n_steps), n_pairs)
        obs = pd.DataFrame({
            'trajectory_id': np.repeat(np.arange(n_pairs), args.n_steps),
            'trajectory_step': steps,
            'pseudotime': np.tile(np.linspace(0, 1, args.n_steps), n_pairs),
            'stage': np.where(steps == 0, 'E7.5',
                              np.where(steps == args.n_steps - 1, 'E12.5', 'generated')),
        })
        traj_adata = sc.AnnData(
            X=trajectories.reshape(-1, n_genes),
            obs=obs,
            var=pd.DataFrame(index=gene_names),
        )
        traj_adata.write_h5ad(
            os.path.join(args.output_dir, f'{path_name}_trajectory_s{args.seed}.h5ad')
        )
        
        # Save model
        model_path = f"{args.output_dir}/{path_name}_model_s{args.seed}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"    Saved model: {model_path}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
