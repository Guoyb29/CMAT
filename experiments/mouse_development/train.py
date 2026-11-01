#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import warnings
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.cmat_attention import CMATAttentionEncoder
from models.spatial_encoder import SpatialEncoder
from models.losses import lineage_alignment_loss


# ==============================================================================
# Data Loading
# ==============================================================================

def load_mouse_data(e75_path: str, e125_path: str) -> Dict:
    print("[1/3] Loading E7.5 scNMT-seq data...")
    e75 = sc.read_h5ad(e75_path)
    
    e75_rna = e75.layers['rna'].toarray() if hasattr(e75.layers['rna'], 'toarray') else e75.layers['rna']
    e75_atac = e75.layers['atac'].toarray() if hasattr(e75.layers['atac'], 'toarray') else e75.layers['atac']
    e75_met = e75.layers['met'].toarray() if hasattr(e75.layers['met'], 'toarray') else e75.layers['met']
    
    e75_celltype_names = e75.obs['celltype'].values
    e75_labels = e75.obs['celltype'].astype('category').cat.codes.values
    
    print(f"  E7.5: {e75_rna.shape[0]} cells, {e75_rna.shape[1]} genes")
    print(f"        {e75_atac.shape[1]} ATAC regions, {e75_met.shape[1]} MET regions")
    print(f"        {len(np.unique(e75_labels))} cell types")
    
    print("\n[2/3] Loading E12.5 spatial transcriptomics...")
    e125 = sc.read_h5ad(e125_path)
    
    e125_rna = e125.X.toarray() if hasattr(e125.X, 'toarray') else e125.X
    e125_spatial = e125.obsm['spatial']
    
    e125_celltype_names = e125.obs['annotation'].values
    e125_labels = e125.obs['annotation'].astype('category').cat.codes.values
    
    print(f"  E12.5: {e125_rna.shape[0]} cells, {e125_rna.shape[1]} genes")
    print(f"         Spatial dimensions: {e125_spatial.shape[1]}")
    print(f"         {len(np.unique(e125_labels))} tissue types")
    
    print("\n[3/3] Constructing lineage correspondence matrix...")
    lineage_matrix = construct_lineage_matrix(e75_celltype_names, e125_celltype_names)
    
    return {
        'e75_rna': e75_rna,
        'e75_atac': e75_atac,
        'e75_met': e75_met,
        'e75_labels': e75_labels,
        'e75_celltype_names': e75_celltype_names,
        'e125_rna': e125_rna,
        'e125_spatial': e125_spatial,
        'e125_labels': e125_labels,
        'e125_celltype_names': e125_celltype_names,
        'lineage_matrix': lineage_matrix,
    }


def construct_lineage_matrix(e75_types: np.ndarray, e125_types: np.ndarray) -> np.ndarray:
    e75_unique = sorted(np.unique(e75_types))
    e125_unique = sorted(np.unique(e125_types))
    
    n_e75 = len(e75_unique)
    n_e125 = len(e125_unique)
    
    lineage_map = {
        'Epiblast': ['Brain'],
        'Nascent mesoderm': ['Heart'],
        'Mesenchyme': ['Cartilage primordium'],
        'ExE endoderm': ['Liver'],
    }
    
    matrix = np.zeros((n_e75, n_e125), dtype=np.float32)
    
    for i, e75_type in enumerate(e75_unique):
        if e75_type in lineage_map:
            for e125_type in lineage_map[e75_type]:
                if e125_type in e125_unique:
                    j = e125_unique.index(e125_type)
                    matrix[i, j] = 1.0
    
    print(f"  Lineage matrix: {n_e75} E7.5 types x {n_e125} E12.5 types")
    print(f"  Valid connections: {int(matrix.sum())}")
    
    return matrix


# ==============================================================================
# Preprocessing
# ==============================================================================

class MultimodalPreprocessor:
    
    def __init__(self, rna_pca_dim: int = 2000, atac_pca_dim: int = 5000, 
                 met_pca_dim: int = 1000, random_state: int = 42):
        self.rna_pca_dim = rna_pca_dim
        self.atac_pca_dim = atac_pca_dim
        self.met_pca_dim = met_pca_dim
        self.random_state = random_state
        
        self.rna_scaler = StandardScaler()
        self.atac_scaler = StandardScaler()
        self.met_scaler = StandardScaler()
        self.spatial_scaler = StandardScaler()
        
        self.rna_pca = None
        self.atac_pca = None
        self.met_pca = None
    
    def fit_transform_e75(self, rna: np.ndarray, atac: np.ndarray, met: np.ndarray) -> Dict:
        print("  Preprocessing E7.5 (fit + transform)...")
        
        # RNA
        rna_dim = min(self.rna_pca_dim, rna.shape[1], rna.shape[0])
        rna_scaled = self.rna_scaler.fit_transform(rna)
        self.rna_pca = PCA(n_components=rna_dim, random_state=self.random_state)
        rna_pca = self.rna_pca.fit_transform(rna_scaled)
        
        # ATAC
        atac_dim = min(self.atac_pca_dim, atac.shape[1], atac.shape[0])
        atac_scaled = self.atac_scaler.fit_transform(atac)
        self.atac_pca = PCA(n_components=atac_dim, random_state=self.random_state)
        atac_pca = self.atac_pca.fit_transform(atac_scaled)
        
        # MET
        met_dim = min(self.met_pca_dim, met.shape[1], met.shape[0])
        met_scaled = self.met_scaler.fit_transform(met)
        self.met_pca = PCA(n_components=met_dim, random_state=self.random_state)
        met_pca = self.met_pca.fit_transform(met_scaled)
        
        print(f"    RNA: {rna.shape[1]} -> {rna_dim} (PCA)")
        print(f"    ATAC: {atac.shape[1]} -> {atac_dim} (PCA)")
        print(f"    MET: {met.shape[1]} -> {met_dim} (PCA)")
        
        return {'rna': rna_pca, 'atac': atac_pca, 'met': met_pca}
    
    def transform_e125(self, rna: np.ndarray, spatial: np.ndarray) -> Dict:
        print("  Preprocessing E12.5 (transform only)...")
        
        rna_scaled = self.rna_scaler.transform(rna)
        rna_pca = self.rna_pca.transform(rna_scaled)
        
        spatial_scaled = self.spatial_scaler.fit_transform(spatial)
        
        print(f"    RNA: {rna.shape[1]} -> {rna_pca.shape[1]} (PCA)")
        print(f"    Spatial: {spatial.shape[1]}-D coordinates (scaled)")
        
        return {'rna': rna_pca, 'spatial': spatial_scaled}


# ==============================================================================
# Dataset
# ==============================================================================

class CMATDataset(Dataset):
    
    def __init__(
        self,
        e75_data: Dict,
        e125_data: Dict,
        e75_labels: np.ndarray,
        e125_labels: np.ndarray,
        samples_per_epoch: int = 1000,
        cells_per_sample: int = 64,
    ):
        self.e75_data = e75_data
        self.e125_data = e125_data
        self.e75_labels = e75_labels
        self.e125_labels = e125_labels
        self.samples_per_epoch = samples_per_epoch
        self.cells_per_sample = cells_per_sample
        
        self.n_e75_types = len(np.unique(e75_labels))
        self.n_e125_types = len(np.unique(e125_labels))
        
        self._prepare_indices()
    
    def _prepare_indices(self):
        self.e75_indices = {i: np.where(self.e75_labels == i)[0] 
                           for i in range(self.n_e75_types)}
        self.e125_indices = {i: np.where(self.e125_labels == i)[0] 
                            for i in range(self.n_e125_types)}
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Random cell types
        e75_type_ids = np.random.randint(0, self.n_e75_types, self.cells_per_sample)
        e125_type_ids = np.random.randint(0, self.n_e125_types, self.cells_per_sample)
        
        # Sample cells
        e75_cell_indices = [np.random.choice(self.e75_indices[t]) for t in e75_type_ids]
        e125_cell_indices = [np.random.choice(self.e125_indices[t]) for t in e125_type_ids]
        
        # E7.5 data
        e75_rna = np.array([self.e75_data['rna'][i] for i in e75_cell_indices])
        e75_atac = np.array([self.e75_data['atac'][i] for i in e75_cell_indices])
        e75_met = np.array([self.e75_data['met'][i] for i in e75_cell_indices])
        
        # E12.5 data
        e125_rna = np.array([self.e125_data['rna'][i] for i in e125_cell_indices])
        e125_spatial = np.array([self.e125_data['spatial'][i] for i in e125_cell_indices])
        
        return {
            'e75_rna': torch.FloatTensor(e75_rna),
            'e75_atac': torch.FloatTensor(e75_atac),
            'e75_met': torch.FloatTensor(e75_met),
            'e75_labels': torch.LongTensor(e75_type_ids),
            'e125_rna': torch.FloatTensor(e125_rna),
            'e125_spatial': torch.FloatTensor(e125_spatial),
            'e125_labels': torch.LongTensor(e125_type_ids),
        }


def collate_fn(batch):
    return batch[0]


# ==============================================================================
# Model
# ==============================================================================

class CMATModel(nn.Module):
    
    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        met_dim: int,
        spatial_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.e75_encoder = CMATAttentionEncoder(
            modality_dims={'rna': rna_dim, 'atac': atac_dim, 'met': met_dim},
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            embedding_dim=embedding_dim,
        )
    
        self.e125_encoder = SpatialEncoder(
            rna_dim=rna_dim,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
    
    def forward_e75(self, rna, atac, met, return_attention=False):
        return self.e75_encoder(
            {'rna': rna, 'atac': atac, 'met': met},
            return_attention=return_attention,
        )
    
    def forward_e125(self, rna, spatial):
        return self.e125_encoder(rna, spatial)
    
    def forward(self, batch):
        e75_emb = self.forward_e75(batch['e75_rna'], batch['e75_atac'], batch['e75_met'])
        e125_emb = self.forward_e125(batch['e125_rna'], batch['e125_spatial'])
        return e75_emb, e125_emb


# ==============================================================================
# Trainer
# ==============================================================================

class CMATTrainer:
    """Training loop for CMAT model."""
    
    def __init__(
        self,
        model: nn.Module,
        lineage_matrix: np.ndarray,
        train_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        temperature: float = 0.07,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.lineage_matrix = torch.FloatTensor(lineage_matrix).to(device)
        self.train_loader = train_loader
        self.device = device
        self.temperature = temperature
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6,
        )
        
        self.train_losses = []
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward
            e75_emb, e125_emb = self.model(batch)
            
            # Loss
            loss = lineage_alignment_loss(
                source_emb=e75_emb,
                target_emb=e125_emb,
                source_labels=batch['e75_labels'],
                target_labels=batch['e125_labels'],
                correspondence_matrix=self.lineage_matrix,
                temperature=self.temperature,
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        return avg_loss
    
    def train(self, num_epochs: int = 50):
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs...")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            loss = self.train_epoch()
            self.train_losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
            
            self.scheduler.step()
        
        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def get_embeddings(self, e75_data: Dict, e125_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        
        batch_size = 512
        
        # E7.5 embeddings
        e75_rna_tensor = torch.FloatTensor(e75_data['rna']).to(self.device)
        e75_atac_tensor = torch.FloatTensor(e75_data['atac']).to(self.device)
        e75_met_tensor = torch.FloatTensor(e75_data['met']).to(self.device)
        
        e75_embeddings = []
        for i in range(0, len(e75_rna_tensor), batch_size):
            batch_rna = e75_rna_tensor[i:i+batch_size]
            batch_atac = e75_atac_tensor[i:i+batch_size]
            batch_met = e75_met_tensor[i:i+batch_size]
            
            emb = self.model.forward_e75(batch_rna, batch_atac, batch_met)
            e75_embeddings.append(emb.cpu().numpy())
        
        e75_embeddings = np.concatenate(e75_embeddings, axis=0)
        
        # E12.5 embeddings
        e125_rna_tensor = torch.FloatTensor(e125_data['rna']).to(self.device)
        e125_spatial_tensor = torch.FloatTensor(e125_data['spatial']).to(self.device)
        
        e125_embeddings = []
        for i in range(0, len(e125_rna_tensor), batch_size):
            batch_rna = e125_rna_tensor[i:i+batch_size]
            batch_spatial = e125_spatial_tensor[i:i+batch_size]
            
            emb = self.model.forward_e125(batch_rna, batch_spatial)
            e125_embeddings.append(emb.cpu().numpy())
        
        e125_embeddings = np.concatenate(e125_embeddings, axis=0)
        
        return e75_embeddings, e125_embeddings


# ==============================================================================
# Evaluation
# ==============================================================================

def compute_alignment_accuracy(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    lineage_matrix: np.ndarray,
    k: int = 5,
) -> float:
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(target_emb)
    
    _, indices = knn.kneighbors(source_emb)
    
    correct = 0
    total = 0
    
    for i, src_label in enumerate(source_labels):
        neighbor_labels = target_labels[indices[i]]
        admissible = np.where(lineage_matrix[src_label] == 1)[0]
        
        if len(admissible) > 0:
            if np.any(np.isin(neighbor_labels, admissible)):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return silhouette_score(embeddings, labels, metric='euclidean')


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CMAT on mouse E7.5 -> E12.5')
    parser.add_argument('--e75_path', type=str, required=True, help='Path to E7.5 h5ad file')
    parser.add_argument('--e125_path', type=str, required=True, help='Path to E12.5 h5ad file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Cells per batch')
    parser.add_argument('--temperature', type=float, default=0.07, help='Contrastive temperature')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Random seed: {args.seed}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("="*70)
    print("Loading data...")
    print("="*70 + "\n")
    data = load_mouse_data(args.e75_path, args.e125_path)
    
    # Preprocess
    print("\n" + "="*70)
    print("Preprocessing...")
    print("="*70 + "\n")
    preprocessor = MultimodalPreprocessor(random_state=args.seed)
    e75_processed = preprocessor.fit_transform_e75(
        data['e75_rna'], data['e75_atac'], data['e75_met']
    )
    e125_processed = preprocessor.transform_e125(
        data['e125_rna'], data['e125_spatial']
    )
    
    # Create dataset
    dataset = CMATDataset(
        e75_processed, e125_processed,
        data['e75_labels'], data['e125_labels'],
        cells_per_sample=args.batch_size,
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == 'cuda'),
        collate_fn=collate_fn,
    )
    
    # Build model
    model = CMATModel(
        rna_dim=e75_processed['rna'].shape[1],
        atac_dim=e75_processed['atac'].shape[1],
        met_dim=e75_processed['met'].shape[1],
        spatial_dim=e125_processed['spatial'].shape[1],
    )
    
    # Train
    trainer = CMATTrainer(
        model=model,
        lineage_matrix=data['lineage_matrix'],
        train_loader=train_loader,
        lr=args.lr,
        temperature=args.temperature,
        device=device,
    )
    
    trainer.train(num_epochs=args.num_epochs)
    
    # Evaluate
    print("="*70)
    print("Evaluating...")
    print("="*70 + "\n")
    
    e75_emb, e125_emb = trainer.get_embeddings(e75_processed, e125_processed)
    np.save(os.path.join(args.output_dir, f'e75_embeddings_s{args.seed}.npy'), e75_emb)
    np.save(os.path.join(args.output_dir, f'e125_embeddings_s{args.seed}.npy'), e125_emb)
    
    alignment_acc = compute_alignment_accuracy(
        e75_emb, e125_emb,
        data['e75_labels'], data['e125_labels'],
        data['lineage_matrix'],
        k=5,
    )
    
    e75_sil = compute_silhouette(e75_emb, data['e75_labels'])
    e125_sil = compute_silhouette(e125_emb, data['e125_labels'])
    
    print(f"Alignment accuracy (k=5): {alignment_acc:.2%}")
    print(f"E7.5 silhouette score: {e75_sil:.3f}")
    print(f"E12.5 silhouette score: {e125_sil:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f'cmat_mouse_seed{args.seed}_{timestamp}.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'e75_embeddings': e75_emb,
        'e125_embeddings': e125_emb,
        'train_losses': trainer.train_losses,
        'alignment_accuracy': alignment_acc,
        'e75_silhouette': e75_sil,
        'e125_silhouette': e125_sil,
        'args': vars(args),
    }, output_path)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
