    
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
            "--output_dir", str(trajectory_dir),
            "--n_epochs", "500",
            "--n_steps", "120",
            "--seed", "120",
        ],
        "[2/4] Generating 4 developmental trajectories"
    )

    # Step 3: Validate against E9.5
    # Validate Brain trajectory
    brain_trajectory = trajectory_dir / "Epiblast_to_Brain_trajectory_s120.h5ad"
    
    run_command(
        [
            sys.executable,
            "-m", "experiments.mouse_development.validate_e95",
            "--trajectory", str(brain_trajectory),
            "--e95", str(e95_path),
            "--output-dir", str(validation_dir / "e95_brain"),
            "--expected-pseudotime", "0.5",
        ],
        "[3/4] Validating against E9.5 intermediate stage"
    )

    # Step 4: Temporal alignment metrics
    metrics_dir = output_dir / "temporal_metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    run_command(
        [
            sys.executable,
            "-m", "evaluation.temporal_alignment",
            "--trajectory-files", f"CMAT:{brain_trajectory}",
            "--output-dir", str(metrics_dir),
        ],
        "[4/4] Computing temporal alignment metrics"
    )

    # Summary
    print("\n" + "=" * 80)
    print("? EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults directory: {output_dir}")
    print("\nKey outputs:")
    print(f"  ? Trained model: {alignment_dir}/")
    print(f"  ? Trajectories: {trajectory_dir}/")
    print(f"    - Epiblast_to_Brain_trajectory_s120.h5ad")
    print(f"    - Nascent_mesoderm_to_Heart_trajectory_s120.h5ad")
    print(f"    - ExE_endoderm_to_Liver_trajectory_s120.h5ad")
    print(f"    - Mesenchyme_to_Cartilage_trajectory_s120.h5ad")
    print(f"  ? E9.5 validation: {validation_dir}/")
    print(f"  ? Metrics: {metrics_dir}/")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
