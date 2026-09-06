#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Endpoint-only validation for human FGO -> MII trajectory.

Validates a trajectory trained using only FGO and MII endpoints
against independent GO1, GO2, and MI intermediate-stage cells.

Corresponds to Figure 6 in the paper.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import scanpy as sc

from evaluation.metrics import validate_intermediate_stage


# =============================================================================
# Default marker genes for human oocyte maturation
# =============================================================================

DEFAULT_MARKERS = {
    "Oocyte": ["GDF9", "ZP3", "FIGLA", "NOBOX"],
    "Meiosis": ["SYCP3", "MLH1", "REC8", "SMC1B"],
    "Maturation": ["MOS", "CCNB1", "CDK1", "CDC20"],
}


# =============================================================================
# Validation
# =============================================================================

def validate_intermediate_stages(
    trajectory_path: str,
    intermediate_stages: list,
    output_dir: str,
    markers: Optional[Dict] = None,
    pseudotime_key: str = "pseudotime",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Validate trajectory against multiple intermediate stages.

    Parameters
    ----------
    trajectory_path : str
        Path to generated trajectory h5ad file.
    intermediate_stages : list of tuples
        List of (stage_name, h5ad_path, expected_pseudotime).
    output_dir : str
        Output directory for results.
    markers : dict, optional
        Marker genes for validation.
    pseudotime_key : str
        Key for pseudotime in trajectory obs.
    verbose : bool
        Print detailed progress.

    Returns
    -------
    pd.DataFrame
        Summary of all stage validations.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Endpoint-only Validation: FGO -> MII")
        print("=" * 70)

    # Load trajectory
    if verbose:
        print("\nLoading trajectory...")
    adata_traj = sc.read_h5ad(trajectory_path)
    if verbose:
        print(f"  Trajectory: {adata_traj.n_obs} cells, {adata_traj.n_vars} genes")

    # Validate each stage
    all_results = []

    for stage_name, stage_path, expected_t in intermediate_stages:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Validating {stage_name}")
            print(f"{'=' * 70}")

        # Load intermediate cells
        if not Path(stage_path).exists():
            print(f"  WARNING: {stage_path} not found, skipping.")
            continue

        adata_stage = sc.read_h5ad(stage_path)
        if verbose:
            print(f"  {stage_name} cells: {adata_stage.n_obs}")

        # Run validation
        result = validate_intermediate_stage(
            adata_trajectory=adata_traj,
            adata_intermediate=adata_stage,
            stage_name=stage_name,
            expected_pseudotime=expected_t,
            markers=markers,
            pseudotime_key=pseudotime_key,
            verbose=verbose,
        )

        # Save stage-specific results
        stage_dir = Path(output_dir) / stage_name.lower()
        stage_dir.mkdir(parents=True, exist_ok=True)

        result["search_results"].to_csv(
            stage_dir / "pseudotime_search.csv",
            index=False,
        )

        if result["marker_results"] is not None:
            result["marker_results"].to_csv(
                stage_dir / "marker_validation.csv",
                index=False,
            )

        # Collect summary
        all_results.append({
            "Stage": stage_name,
            "Expected_t": expected_t,
            "Best_t": result["best_pseudotime"],
            "Temporal_Error": result["temporal_error"],
            "Pearson": result["feature_distance"]["pearson"],
            "Spearman": result["feature_distance"]["spearman"],
            "Cosine": result["feature_distance"]["cosine"],
            "Latent_Distance": result["latent_distance"],
            "Wasserstein": result["wasserstein_distance"],
            "N_Cells_Real": result["n_cells_real"],
            "N_Cells_Gen": result["n_cells_generated"],
            "N_Common_Genes": result["n_common_genes"],
        })

    # Create summary DataFrame
    df_summary = pd.DataFrame(all_results)

    # Save overall summary
    df_summary.to_csv(Path(output_dir) / "all_stages_summary.csv", index=False)

    # Print final summary
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(df_summary.to_string(index=False))
        print(f"\nResults saved to: {output_dir}")
        print("=" * 70)

    return df_summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate FGO->MII trajectory against intermediate stages'
    )

    parser.add_argument(
        '--trajectory',
        type=str,
        required=True,
        help='Path to generated trajectory h5ad file',
    )
    parser.add_argument(
        '--go1',
        type=str,
        required=True,
        help='Path to GO1 intermediate stage h5ad',
    )
    parser.add_argument(
        '--go2',
        type=str,
        required=True,
        help='Path to GO2 intermediate stage h5ad',
    )
    parser.add_argument(
        '--mi',
        type=str,
        required=True,
        help='Path to MI intermediate stage h5ad',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./validation_endpoint',
        help='Output directory for validation results',
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

    # Define intermediate stages
    stages = [
        ("GO1", args.go1, 0.25),
        ("GO2", args.go2, 0.50),
        ("MI", args.mi, 0.75),
    ]

    # Run validation
    validate_intermediate_stages(
        trajectory_path=args.trajectory,
        intermediate_stages=stages,
        output_dir=args.output_dir,
        markers=None if args.no_markers else DEFAULT_MARKERS,
        pseudotime_key=args.pseudotime_key,
        verbose=True,
    )


if __name__ == "__main__":
    main()
