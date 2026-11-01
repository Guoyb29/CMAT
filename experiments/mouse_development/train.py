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
