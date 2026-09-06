#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory visualization for CMAT developmental paths.
"""

import os
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})


# Default marker genes for mouse development
DEFAULT_MARKER_GENES = {
    'Epiblast': ['Pou5f1', 'Sox2', 'Nanog', 'Utf1', 'Klf4'],
    'Nascent mesoderm': ['T', 'Mesp1', 'Tbx6', 'Eomes', 'Mixl1'],
    'Mesenchyme': ['Twist1', 'Snai2', 'Vim', 'Pdgfra', 'Hand2'],
    'ExE endoderm': ['Sox17', 'Foxa2', 'Gata6', 'Gata4', 'Hhex'],
    'Brain': ['Sox2', 'Pax6', 'Neurod1', 'Tubb3', 'Rbfox3', 'Dcx'],
    'Heart': ['Tnnt2', 'Myh6', 'Myh7', 'Nkx2-5', 'Gata4', 'Mef2c'],
    'Liver': ['Alb', 'Afp', 'Apoa1', 'Hnf4a', 'Apob', 'Ttr'],
    'Cartilage primordium': ['Sox9', 'Col2a1', 'Acan', 'Prrx1', 'Col9a1'],
}


def plot_trajectory_markers(
    trajectory: np.ndarray,
    gene_names: List[str],
    source_markers: List[str],
    target_markers: List[str],
    path_name: str,
    output_path: str,
    figsize: tuple = (14, 5),
    dpi: int = 300,
) -> None:
    """
    Plot marker gene dynamics along a developmental trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory array of shape (n_steps, n_genes).
    gene_names : list of str
        List of gene names corresponding to columns in trajectory.
    source_markers : list of str
        Source (early) marker genes expected to decrease.
    target_markers : list of str
        Target (late) marker genes expected to increase.
    path_name : str
        Name of the developmental path (for title).
    output_path : str
        Path to save the figure.
    figsize : tuple, optional
        Figure size (width, height).
    dpi : int, optional
        Figure DPI for saving.
    """
    n_steps = trajectory.shape[0]
    pseudotime = np.linspace(0, 1, n_steps)

    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: Source markers (should decrease)
    ax = axes[0]

    for marker in source_markers:
        if marker in gene_to_idx:
            idx = gene_to_idx[marker]
            expr = trajectory[:, idx]

            # Compute correlation
            corr, _ = spearmanr(pseudotime, expr)

            # Plot with color based on expected direction
            color = 'green' if corr < 0 else 'red'
            ax.plot(
                pseudotime,
                expr,
                label=f'{marker} (rho={corr:.2f})',
                linewidth=2,
                alpha=0.8,
                color=color,
            )

    ax.set_xlabel('Pseudotime', fontweight='bold')
    ax.set_ylabel('Normalized Expression', fontweight='bold')
    ax.set_title('(A) Source Markers (should decrease)', fontweight='bold')
    ax.legend(frameon=True, loc='best', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Target markers (should increase)
    ax = axes[1]

    for marker in target_markers:
        if marker in gene_to_idx:
            idx = gene_to_idx[marker]
            expr = trajectory[:, idx]

            # Compute correlation
            corr, _ = spearmanr(pseudotime, expr)

            # Plot with color based on expected direction
            color = 'green' if corr > 0 else 'red'
            ax.plot(
                pseudotime,
                expr,
                label=f'{marker} (rho={corr:.2f})',
                linewidth=2,
                alpha=0.8,
                color=color,
            )

    ax.set_xlabel('Pseudotime', fontweight='bold')
    ax.set_ylabel('Normalized Expression', fontweight='bold')
    ax.set_title('(B) Target Markers (should increase)', fontweight='bold')
    ax.legend(frameon=True, loc='best', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(
        f'{path_name.replace("_", " -> ")}',
        fontweight='bold',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    """Command-line interface for trajectory visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize CMAT developmental trajectories'
    )

    parser.add_argument(
        '--trajectory-dir',
        type=str,
        required=True,
        help='Directory containing trajectory .npy files',
    )
    parser.add_argument(
        '--source-h5ad',
        type=str,
        required=True,
        help='Path to source (E7.5) h5ad file for gene names',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for plots',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=120,
        help='Random seed used in trajectory generation',
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
    print("CMAT Trajectory Visualization")
    print("=" * 70)

    # Load gene names
    import scanpy as sc

    print(f"\nLoading gene names from {args.source_h5ad}...")
    source_adata = sc.read_h5ad(args.source_h5ad)
    gene_names = source_adata.var_names.tolist()

    # Define developmental paths
    paths = [
        {
            'name': 'Epiblast_to_Brain',
            'source': 'Epiblast',
            'target': 'Brain',
        },
        {
            'name': 'Nascent_mesoderm_to_Heart',
            'source': 'Nascent mesoderm',
            'target': 'Heart',
        },
        {
            'name': 'ExE_endoderm_to_Liver',
            'source': 'ExE endoderm',
            'target': 'Liver',
        },
        {
            'name': 'Mesenchyme_to_Cartilage',
            'source': 'Mesenchyme',
            'target': 'Cartilage primordium',
        },
    ]

    # Plot each path
    for path in paths:
        path_name = path['name']

        print(f"\nProcessing {path_name}...")

        # Load trajectory
        traj_file = os.path.join(
            args.trajectory_dir,
            f"{path_name}_trajectory_s{args.seed}.npy",
        )

        if not os.path.exists(traj_file):
            print(f"  Warning: {traj_file} not found, skipping.")
            continue

        trajectories = np.load(traj_file)  # [n_pairs, n_steps, n_genes]

        # Compute mean trajectory across all cell pairs
        mean_trajectory = trajectories.mean(axis=0)

        # Plot
        plot_trajectory_markers(
            trajectory=mean_trajectory,
            gene_names=gene_names,
            source_markers=DEFAULT_MARKER_GENES[path['source']],
            target_markers=DEFAULT_MARKER_GENES[path['target']],
            path_name=path_name,
            output_path=os.path.join(
                args.output_dir,
                f"{path_name}_markers_s{args.seed}.png",
            ),
            dpi=args.dpi,
        )

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
