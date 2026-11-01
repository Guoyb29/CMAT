#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start script for human oocyte maturation experiment (FGO -> MII).

This script runs the complete CMAT pipeline for human oocyte maturation.

Usage:
    python run_human_experiment.py
"""

import subprocess
import sys
from pathlib import Path

import yaml


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd, cwd=Path(__file__).resolve().parent, capture_output=False, text=True
    )
    
    if result.returncode != 0:
        print(f"\n? Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n? {description} completed successfully.")
    return result


def main():
    print("=" * 80)
    print("CMAT: Human Oocyte Maturation Experiment (FGO -> MII)")
    print("=" * 80)

    # Project paths
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "human.yaml"
    
    if not config_path.exists():
        print(f"? Config file not found: {config_path}")
        sys.exit(1)

    # Load config to get paths
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Output directories
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('save_dir', output_config.get('results_dir', './results/human')))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    alignment_dir = output_dir / "alignment"
    trajectory_dir = output_dir / "trajectories"
    validation_dir = output_dir / "validation"
    report_dir = validation_dir / "biology_report"

    # Create output directories
    for dir_path in [output_dir, alignment_dir, trajectory_dir, validation_dir, report_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Data paths (placeholders - users should update these)
    data_h5ad = config.get('data', {}).get('trajectory_h5ad', 'data/human/scChaRM_FGO_MII.h5ad')
    go1_h5ad = config.get('data', {}).get('go1_path', 'data/human/GO1_cells.h5ad')
    go2_h5ad = config.get('data', {}).get('go2_path', 'data/human/GO2_cells.h5ad')
    mi_h5ad = config.get('data', {}).get('mi_path', 'data/human/MI_cells.h5ad')

    # Step 1: Train alignment model
    run_command(
        [
            sys.executable,
            "-m", "experiments.human_FGO_MII.train",
            "--trajectory-h5ad", str(data_h5ad),
            "--output-dir", str(alignment_dir),
            "--epochs", "50",
        ],
        "[1/5] Training CMAT alignment model"
    )

    # Step 2: Generate trajectories
    fgo_emb = alignment_dir / "fgo_embedding.npy"
    mii_emb = alignment_dir / "mii_embedding.npy"
    
    run_command(
        [
            sys.executable,
            "-m", "experiments.human_FGO_MII.generate_trajectories",
            "--input_h5ad", str(data_h5ad),
            "--fgo_embedding", str(fgo_emb),
