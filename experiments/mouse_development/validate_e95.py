#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9.5 validation for mouse E7.5 -> E12.5 trajectory.

Validates a pre-generated trajectory against independent E9.5 cells
that were not used during training.

Corresponds to intermediate-stage validation in the paper.
"""

import argparse
import os
from pathlib import Path

import scanpy as sc

from evaluation.metrics import validate_intermediate_stage


# =============================================================================
# Default marker genes for E9.5
# =============================================================================

DEFAULT_E95_MARKERS = {
    "Heart": ["Nkx2-5", "Gata4", "Hand1", "Mef2c", "Myh6", "Tnnt2"],
    "Brain": ["Pax6", "Sox1", "Nestin", "Sox2"],
    "Liver": ["Foxa2", "Sox17", "Hnf4a", "Afp"],
    "Blood": ["Tal1", "Gata1", "Gfi1b"],
}


# =============================================================================
# Main validation
# =============================================================================

def validate_e95(
    trajectory_path: str,
    e95_path: str,
    output_dir: str,
    expected_pseudotime: float = 0.5,
    markers: dict = None,
    pseudotime_key: str = "pseudotime",
    verbose: bool = True,
):
    """
    Validate trajectory against E9.5 intermediate stage.

    Parameters
    ----------
    trajectory_path : str
        Path to generated trajectory h5ad file.
    e95_path : str
        Path to E9.5 real cells h5ad file.
    output_dir : str
        Output directory for results.
    expected_pseudotime : float
        Expected pseudotime for E9.5 (default 0.5, halfway).
    markers : dict, optional
        Marker genes for validation.
    pseudotime_key : str
        Key for pseudotime in trajectory obs.
    verbose : bool
        Print detailed progress.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("E9.5 Validation")
        print("=" * 70)

    # Load data
    if verbose:
        print("\nLoading data...")
    
    adata_traj = sc.read_h5ad(trajectory_path)
    adata_e95 = sc.read_h5ad(e95_path)

    if verbose:
        print(f"  Trajectory: {adata_traj.n_obs} cells, {adata_traj.n_vars} genes")
        print(f"  E9.5 real: {adata_e95.n_obs} cells, {adata_e95.n_vars} genes")

    # Run validation
    result = validate_intermediate_stage(
        adata_trajectory=adata_traj,
        adata_intermediate=adata_e95,
        stage_name="E9.5",
        expected_pseudotime=expected_pseudotime,
        markers=markers,
        pseudotime_key=pseudotime_key,
        verbose=verbose,
    )

    # Save results
    if verbose:
        print("\nSaving results...")

    output_path = Path(output_dir)
    
    result["search_results"].to_csv(
        output_path / "pseudotime_search.csv",
        index=False,
    )

    if result["marker_results"] is not None:
        result["marker_results"].to_csv(
            output_path / "marker_validation.csv",
            index=False,
        )

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Stage: {result['stage_name']}")
        print(f"Best pseudotime: {result['best_pseudotime']:.3f}")
        print(f"Expected pseudotime: {result['expected_pseudotime']:.3f}")
        print(f"Temporal error: {result['temporal_error']:.3f}")
        print(f"\nFeature-level distance:")
        print(f"  Pearson: {result['feature_distance']['pearson']:.4f}")
        print(f"  Spearman: {result['feature_distance']['spearman']:.4f}")
        print(f"  Cosine: {result['feature_distance']['cosine']:.4f}")
        print(f"\nLatent space distance: {result['latent_distance']:.4f}")
        print(f"Wasserstein distance: {result['wasserstein_distance']:.4f}")

        if result["marker_results"] is not None:
            good_pct = (result["marker_results"]["rel_error_pct"] < 20).sum()
            good_pct = good_pct / len(result["marker_results"]) * 100
            print(f"\nMarker gene accuracy: {good_pct:.1f}% good (<20% error)")

        print(f"\nResults saved to: {output_dir}")
        print("=" * 70)

    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate E7.5->E12.5 trajectory against E9.5 intermediate stage'
    )

    parser.add_argument(
        '--trajectory',
        type=str,
        required=True,
        help='Path to generated trajectory h5ad file',
    )
    parser.add_argument(
        '--e95',
        type=str,
        required=True,
        help='Path to E9.5 intermediate stage h5ad file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./validation_e95',
        help='Output directory for validation results',
    )
    parser.add_argument(
        '--expected-pseudotime',
        type=float,
        default=0.5,
        help='Expected pseudotime for E9.5 (default: 0.5, halfway between E7.5 and E12.5)',
    )
    parser.add_argument(
        '--pseudotime-key',
        type=str,
        default='pseudotime',
        help='Key for pseudotime in trajectory obs',
    )
    parser.add_argument(
        '--no-markers',
        action='store_true',
        help='Skip marker gene validation',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Run validation
    validate_e95(
        trajectory_path=args.trajectory,
        e95_path=args.e95,
        output_dir=args.output_dir,
        expected_pseudotime=args.expected_pseudotime,
        markers=None if args.no_markers else DEFAULT_E95_MARKERS,
        pseudotime_key=args.pseudotime_key,
        verbose=True,
    )


if __name__ == "__main__":
    main()
