#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start script for mouse embryo development experiment (E7.5 -> E12.5).

This script runs the complete CMAT pipeline for mouse embryo development.

Usage:
    python run_mouse_experiment.py
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
    print("CMAT: Mouse Embryo Development Experiment (E7.5 -> E12.5)")
    print("=" * 80)

    # Project paths
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "mouse.yaml"
    
    if not config_path.exists():
        print(f"? Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Output directories
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('save_dir', output_config.get('results_dir', './results/mouse')))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    alignment_dir = output_dir / "alignment"
    trajectory_dir = output_dir / "trajectories"
    validation_dir = output_dir / "validation"

    # Create directories
    for dir_path in [output_dir, alignment_dir, trajectory_dir, validation_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Data paths
    e75_path = config.get('data', {}).get('source_path', 'data/mouse/E75_scNMT.h5ad')
    e125_path = config.get('data', {}).get('target_path', 'data/mouse/E125_spatial.h5ad')
    e95_path = config.get('data', {}).get('validation_path', 'data/mouse/E95_real_converted.h5ad')

    # Step 1: Train alignment
    run_command(
        [
            sys.executable,
            "-m", "experiments.mouse_development.train",
            "--e75_path", str(e75_path),
            "--e125_path", str(e125_path),
            "--output_dir", str(alignment_dir),
            "--num_epochs", "50",
            "--seed", "120",
        ],
        "[1/4] Training CMAT alignment model"
    )

    # Step 2: Generate trajectories (4 developmental paths)
    embedding_dir = alignment_dir
    
    run_command(
        [
            sys.executable,
            "-m", "experiments.mouse_development.generate_trajectories",
            "--e75_path", str(e75_path),
            "--e125_path", str(e125_path),
            "--embedding_dir", str(embedding_dir),
