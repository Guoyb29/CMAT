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
