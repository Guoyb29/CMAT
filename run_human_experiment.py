    
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
            "--mii_embedding", str(mii_emb),
            "--output_dir", str(trajectory_dir),
            "--n_steps_generate", "120",
        ],
        "[2/5] Generating continuous trajectories"
    )

    trajectory_file = trajectory_dir / "FGO_to_MII_trajectories.h5ad"

    # Step 3: Validate endpoint
    run_command(
        [
            sys.executable,
            "-m", "experiments.human_FGO_MII.validate_endpoint",
            "--trajectory", str(trajectory_file),
            "--go1", str(go1_h5ad),
            "--go2", str(go2_h5ad),
            "--mi", str(mi_h5ad),
            "--output-dir", str(validation_dir),
        ],
        "[3/5] Validating against intermediate stages (GO1/GO2/MI)"
    )

    # Step 4: Optional report from precomputed biological-analysis tables.
    if (validation_dir / "01_observed_geometry.csv").exists():
        run_command(
            [
                sys.executable,
                "-m", "experiments.human_FGO_MII.biology_evidence_analysis",
                "--out", str(validation_dir),
                "--report-dir", str(report_dir),
            ],
            "[4/5] Biological evidence analysis (Figure 6e)"
        )
    else:
        print("[4/5] Skipping optional biology report: precomputed tables are not present.")

    # Step 5: Temporal alignment metrics
    metrics_dir = output_dir / "temporal_metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    run_command(
        [
            sys.executable,
            "-m", "evaluation.temporal_alignment",
            "--trajectory-files", f"CMAT:{trajectory_file}",
            "--early-batch", "FGO",
            "--late-batch", "MII",
            "--output-dir", str(metrics_dir),
        ],
        "[5/5] Computing temporal alignment metrics"
    )

    # Summary
    print("\n" + "=" * 80)
    print("? EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults directory: {output_dir}")
    print("\nKey outputs:")
    print(f"  ? Trained model: {alignment_dir}/")
    print(f"  ? Trajectory: {trajectory_file}")
    print(f"  ? Validation: {validation_dir}/")
    if (validation_dir / "01_observed_geometry.csv").exists():
        print(f"  ? Biology report: {report_dir}/biological_evidence_report.md")
