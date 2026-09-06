#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding alignment visualization for CMAT.
"""

import os
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import scanpy as sc

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})


def plot_embedding_alignment(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    output_path: str,
    source_name: str = "Source",
    target_name: str = "Target",
    title: str = "CMAT Embedding Alignment",
    figsize: tuple = (16, 7),
    dpi: int = 300,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.3,
    random_state: int = 42,
) -> None:
    """
    Visualize alignment between source and target embeddings using UMAP.

    Parameters
    ----------
    source_embeddings : np.ndarray
        Source embeddings of shape (n_source, embedding_dim).
    target_embeddings : np.ndarray
        Target embeddings of shape (n_target, embedding_dim).
    source_labels : np.ndarray
        Cell type labels for source cells.
    target_labels : np.ndarray
        Cell type labels for target cells.
    output_path : str
        Path to save the figure.
    source_name : str, optional
        Name for source data (e.g., "E7.5").
    target_name : str, optional
        Name for target data (e.g., "E12.5").
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size (width, height).
    dpi : int, optional
        Figure DPI for saving.
    umap_n_neighbors : int, optional
        UMAP n_neighbors parameter.
    umap_min_dist : float, optional
        UMAP min_dist parameter.
    random_state : int, optional
        Random state for UMAP.
    """
    print("Running UMAP on combined embeddings...")

    # Combine embeddings
    combined_embeddings = np.vstack([source_embeddings, target_embeddings])
    stage_labels = np.array(
        [source_name] * len(source_embeddings)
        + [target_name] * len(target_embeddings)
    )
    celltype_labels = np.concatenate([source_labels, target_labels])

    # Compute UMAP
    reducer = UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric='cosine',
        random_state=random_state,
    )
    umap_coords = reducer.fit_transform(combined_embeddings)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: Color by stage
    ax = axes[0]

    stage_colors = {source_name: 'coral', target_name: 'steelblue'}

    for stage, color in stage_colors.items():
        mask = stage_labels == stage
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=color,
            label=stage,
            alpha=0.6,
            s=10,
            edgecolors='none',
        )

    ax.set_xlabel('UMAP 1', fontweight='bold')
    ax.set_ylabel('UMAP 2', fontweight='bold')
    ax.set_title('(A) Colored by Stage', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, loc='best', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Color by cell type
    ax = axes[1]

    unique_types = np.unique(celltype_labels)
    palette = sns.color_palette('tab20', len(unique_types))

    for i, celltype in enumerate(unique_types):
        mask = celltype_labels == celltype
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=[palette[i]],
            label=celltype,
            alpha=0.6,
            s=10,
            edgecolors='none',
        )

    ax.set_xlabel('UMAP 1', fontweight='bold')
    ax.set_ylabel('UMAP 2', fontweight='bold')
    ax.set_title('(B) Colored by Cell Type', fontweight='bold', fontsize=14)
    ax.legend(
        frameon=True,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        ncol=1,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle(title, fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    """Command-line interface for embedding visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize CMAT embedding alignment'
    )

    parser.add_argument(
        '--embedding-dir',
        type=str,
        required=True,
        help='Directory containing embedding .npy files',
    )
    parser.add_argument(
        '--source-h5ad',
        type=str,
        required=True,
        help='Path to source h5ad file for cell type labels',
    )
    parser.add_argument(
        '--target-h5ad',
        type=str,
        required=True,
        help='Path to target h5ad file for cell type labels',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for plots',
    )
    parser.add_argument(
        '--source-name',
        type=str,
        default='E7.5',
        help='Name for source stage',
    )
    parser.add_argument(
        '--target-name',
        type=str,
        default='E12.5',
        help='Name for target stage',
    )
    parser.add_argument(
        '--source-celltype-key',
        type=str,
        default='celltype',
        help='Key for cell type labels in source h5ad',
    )
    parser.add_argument(
        '--target-celltype-key',
        type=str,
        default='annotation',
        help='Key for cell type labels in target h5ad',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=120,
        help='Random seed used in embedding generation',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CMAT Embedding Visualization")
    print("=" * 70)

    # Load embeddings
    print("\n[1/3] Loading embeddings...")
    source_embeddings = np.load(
        os.path.join(
            args.embedding_dir,
            f"{args.source_name.lower().replace('.', '')}_embeddings_s{args.seed}.npy",
        )
    )
    target_embeddings = np.load(
        os.path.join(
            args.embedding_dir,
            f"{args.target_name.lower().replace('.', '')}_embeddings_s{args.seed}.npy",
        )
    )

    # Load cell type labels
    print("\n[2/3] Loading cell type labels...")
    source_adata = sc.read_h5ad(args.source_h5ad)
    target_adata = sc.read_h5ad(args.target_h5ad)

    source_labels = source_adata.obs[args.source_celltype_key].values
    target_labels = target_adata.obs[args.target_celltype_key].values

    # Create visualization
    print("\n[3/3] Creating UMAP visualization...")
    plot_embedding_alignment(
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        source_labels=source_labels,
        target_labels=target_labels,
        output_path=os.path.join(
            args.output_dir,
            f"embedding_alignment_s{args.seed}.png",
        ),
        source_name=args.source_name,
        target_name=args.target_name,
        title=f"CMAT Embedding Alignment (seed={args.seed})",
        dpi=args.dpi,
    )

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
