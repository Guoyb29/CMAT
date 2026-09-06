#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory validation metrics for CMAT.

Includes:
- Marker-direction validation (validate_trajectory_markers)
- Marker monotonicity (compute_marker_monotonicity)
- Trajectory smoothness (compute_trajectory_smoothness)
- Intermediate-stage validation (validate_intermediate_stage)
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# =============================================================================
# Utility functions
# =============================================================================

def to_dense_vector(x):
    """Convert sparse or dense matrix to flat NumPy array."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).flatten()


def to_dense(data) -> np.ndarray:
    """Convert sparse or dense matrix to dense NumPy array."""
    if hasattr(data, "toarray"):
        return np.asarray(data.toarray())
    return np.asarray(data)


def safe_extract(data) -> np.ndarray:
    """Convert sparse or dense matrix to flat NumPy array."""
    if hasattr(data, "toarray"):
        return np.asarray(data.toarray()).flatten()
    if hasattr(data, "A"):
        return np.asarray(data.A).flatten()
    return np.asarray(data).flatten()


def safe_spearman(x, y):
    """Spearman correlation with protection against constant vectors."""
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return np.nan

    try:
        corr, _ = spearmanr(x, y)
        if np.isnan(corr):
            return np.nan
        return corr
    except Exception:
        return np.nan


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with protection against constant vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < 2 or y.size < 2:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    try:
        value, _ = pearsonr(x, y)
        return float(value)
    except Exception:
        return np.nan


# =============================================================================
# Marker validation
# =============================================================================

def validate_trajectory_markers(
    adata,
    start_markers: List[str],
    end_markers: List[str],
    path_key: str = "path",
    pseudotime_key: str = "pseudotime",
    threshold: float = 0.0,
) -> Dict:
    """
    Validate trajectory by checking marker gene directions.

    Parameters
    ----------
    adata : AnnData
        Trajectory data with pseudotime.
    start_markers : list of str
        Genes expected to decrease along trajectory.
    end_markers : list of str
        Genes expected to increase along trajectory.
    path_key : str
        Key in obs for path/branch identification.
    pseudotime_key : str
        Key in obs for pseudotime values.
    threshold : float
        Correlation threshold (default 0.0).
        Start markers pass if corr < -threshold.
        End markers pass if corr > threshold.

    Returns
    -------
    dict
        Validation results including accuracy and details.
    """
    if path_key in adata.obs.columns:
        paths = adata.obs[path_key].astype(str).unique().tolist()
        if len(paths) == 1:
            trajectory = adata[adata.obs[path_key].astype(str) == paths[0]].copy()
            path_name = paths[0]
        else:
            print(f"WARNING: Multiple paths found: {paths}, using first")
            trajectory = adata[adata.obs[path_key].astype(str) == paths[0]].copy()
            path_name = paths[0]
    else:
        trajectory = adata.copy()
        path_name = "default"

    if trajectory.n_obs < 50:
        print(f"WARNING: {path_name}: Too few cells ({trajectory.n_obs})")
        return None

    # Sort by pseudotime
    trajectory = trajectory[trajectory.obs[pseudotime_key].argsort()].copy()

    time = trajectory.obs[pseudotime_key].astype(float).values
    time_bins = np.unique(time)

    marker_records = []

    # Validate start markers (expected to decrease: corr < -threshold)
    start_correct = 0
    start_total = 0
    start_corrs = []

    for gene in start_markers:
        if gene not in trajectory.var_names:
            marker_records.append({
                "gene": gene,
                "marker_type": "start",
                "expected_direction": "down",
                "spearman_corr": np.nan,
                "passed": False,
                "status": "missing",
            })
            continue

        expr = to_dense_vector(trajectory[:, gene].X)
        expr_mean = np.array([expr[time == t].mean() for t in time_bins])

        corr = safe_spearman(time_bins, expr_mean)

        if np.isnan(corr):
            passed = False
            status = "constant"
        else:
            passed = corr < -threshold
            status = "OK" if passed else "X"
            start_total += 1
            start_corrs.append(corr)
            if passed:
                start_correct += 1

        marker_records.append({
            "gene": gene,
            "marker_type": "start",
            "expected_direction": "down",
            "spearman_corr": corr,
            "passed": passed,
            "status": status,
        })

    # Validate end markers (expected to increase: corr > threshold)
    end_correct = 0
    end_total = 0
    end_corrs = []

    for gene in end_markers:
        if gene not in trajectory.var_names:
            marker_records.append({
                "gene": gene,
                "marker_type": "end",
                "expected_direction": "up",
                "spearman_corr": np.nan,
                "passed": False,
                "status": "missing",
            })
            continue

        expr = to_dense_vector(trajectory[:, gene].X)
        expr_mean = np.array([expr[time == t].mean() for t in time_bins])

        corr = safe_spearman(time_bins, expr_mean)

        if np.isnan(corr):
            passed = False
            status = "constant"
        else:
            passed = corr > threshold
            status = "OK" if passed else "X"
            end_total += 1
            end_corrs.append(corr)
            if passed:
                end_correct += 1

        marker_records.append({
            "gene": gene,
            "marker_type": "end",
            "expected_direction": "up",
            "spearman_corr": corr,
            "passed": passed,
            "status": status,
        })

    # Compute accuracy
    start_acc = start_correct / start_total if start_total > 0 else 0.0
    end_acc = end_correct / end_total if end_total > 0 else 0.0

    total_correct = start_correct + end_correct
    total_markers = start_total + end_total
    overall_acc = total_correct / total_markers if total_markers > 0 else 0.0

    return {
        "path": path_name,
        "start_accuracy": start_acc,
        "end_accuracy": end_acc,
        "overall_accuracy": overall_acc,
        "start_correct": start_correct,
        "start_total": start_total,
        "end_correct": end_correct,
        "end_total": end_total,
        "total_correct": total_correct,
        "total_markers": total_markers,
        "start_correlations": start_corrs,
        "end_correlations": end_corrs,
        "marker_details": pd.DataFrame(marker_records),
        "n_cells": trajectory.n_obs,
        "n_time_points": len(time_bins),
    }


# =============================================================================
# Marker monotonicity
# =============================================================================

def compute_marker_monotonicity(
    adata,
    markers: List[str],
    expected_direction: str = "up",
    pseudotime_key: str = "pseudotime",
    n_bins: int = 50,
) -> Dict:
    """
    Compute monotonicity score for marker genes along pseudotime.

    Measures how consistently markers increase/decrease with pseudotime
    by computing the fraction of monotonic transitions between consecutive bins.

    Parameters
    ----------
    adata : AnnData
        Trajectory data with pseudotime.
    markers : list of str
        Marker genes to evaluate.
    expected_direction : str
        Expected trend: 'up' for increasing, 'down' for decreasing.
    pseudotime_key : str
        Key in obs for pseudotime values.
    n_bins : int
        Number of bins for pseudotime discretization.

    Returns
    -------
    dict
        Monotonicity scores per gene and overall statistics.
    """
    trajectory = adata.copy()
    trajectory = trajectory[trajectory.obs[pseudotime_key].argsort()].copy()

    time = trajectory.obs[pseudotime_key].astype(float).values

    # Bin pseudotime
    bins = np.linspace(time.min(), time.max(), n_bins + 1)
    bin_indices = np.digitize(time, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    marker_results = []

    for gene in markers:
        if gene not in trajectory.var_names:
            marker_results.append({
                "gene": gene,
                "monotonicity_score": np.nan,
                "correlation": np.nan,
                "status": "missing",
            })
            continue

        expr = to_dense_vector(trajectory[:, gene].X)

        # Compute binned mean expression
        binned_expr = np.array([expr[bin_indices == i].mean() for i in range(n_bins)])

        # Skip constant genes
        if np.std(binned_expr) < 1e-8:
            marker_results.append({
                "gene": gene,
                "monotonicity_score": np.nan,
                "correlation": np.nan,
                "status": "constant",
            })
            continue

        # Compute monotonicity score
        diffs = np.diff(binned_expr)

        if expected_direction == "up":
            monotonic_transitions = (diffs >= 0).sum()
        else:  # down
            monotonic_transitions = (diffs <= 0).sum()

        monotonicity_score = monotonic_transitions / len(diffs)

        # Also compute correlation
        time_bins = np.arange(n_bins)
        corr = safe_spearman(time_bins, binned_expr)

        marker_results.append({
            "gene": gene,
            "monotonicity_score": monotonicity_score,
            "correlation": corr,
            "status": "OK",
        })

    df_markers = pd.DataFrame(marker_results)

    # Overall statistics
    valid = df_markers[df_markers["status"] == "OK"]

    if len(valid) > 0:
        mean_monotonicity = valid["monotonicity_score"].mean()
        mean_correlation = valid["correlation"].mean()
    else:
        mean_monotonicity = np.nan
        mean_correlation = np.nan

    return {
        "mean_monotonicity": mean_monotonicity,
        "mean_correlation": mean_correlation,
        "n_markers": len(markers),
        "n_valid": len(valid),
        "marker_details": df_markers,
    }


# =============================================================================
# Trajectory smoothness
# =============================================================================

def compute_trajectory_smoothness(
    adata,
    pseudotime_key: str = "pseudotime",
    n_bins: int = 50,
) -> float:
    """
    Compute trajectory smoothness using autocorrelation.

    Parameters
    ----------
    adata : AnnData
        Trajectory data.
    pseudotime_key : str
        Key in obs for pseudotime.
    n_bins : int
        Number of bins for pseudotime discretization.

    Returns
    -------
    float
        Mean autocorrelation across genes (higher = smoother).
    """
    trajectory = adata.copy()
    trajectory = trajectory[trajectory.obs[pseudotime_key].argsort()].copy()

    time = trajectory.obs[pseudotime_key].astype(float).values

    # Bin pseudotime
    bins = np.linspace(time.min(), time.max(), n_bins + 1)
    bin_indices = np.digitize(time, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    X = trajectory.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    smoothness_scores = []

    for gene_idx in range(X.shape[1]):
        expr = X[:, gene_idx]

        # Compute binned mean expression
        binned_expr = np.array([expr[bin_indices == i].mean() for i in range(n_bins)])

        # Skip constant genes
        if np.std(binned_expr) < 1e-6:
            continue

        # Compute first-order autocorrelation
        if len(binned_expr) > 1:
            autocorr = np.corrcoef(binned_expr[:-1], binned_expr[1:])[0, 1]
            if not np.isnan(autocorr):
                smoothness_scores.append(autocorr)

    return np.mean(smoothness_scores) if len(smoothness_scores) > 0 else 0.0


# =============================================================================
# Overall trajectory quality
# =============================================================================

def assess_trajectory_quality(
    adata,
    start_markers: List[str],
    end_markers: List[str],
    pseudotime_key: str = "pseudotime",
) -> Dict:
    """
    Comprehensive trajectory quality assessment.

    Parameters
    ----------
    adata : AnnData
        Trajectory data.
    start_markers : list of str
        Start-stage markers.
    end_markers : list of str
        End-stage markers.
    pseudotime_key : str
        Key in obs for pseudotime.

    Returns
    -------
    dict
        Quality metrics including marker accuracy and smoothness.
    """
    # Marker validation
    marker_result = validate_trajectory_markers(
        adata=adata,
        start_markers=start_markers,
        end_markers=end_markers,
        pseudotime_key=pseudotime_key,
        threshold=0.0,
    )

    if marker_result is None:
        return None

    # Smoothness
    smoothness = compute_trajectory_smoothness(
        adata=adata,
        pseudotime_key=pseudotime_key,
    )

    return {
        "marker_accuracy": marker_result["overall_accuracy"],
        "start_accuracy": marker_result["start_accuracy"],
        "end_accuracy": marker_result["end_accuracy"],
        "smoothness": smoothness,
        "n_cells": marker_result["n_cells"],
        "n_time_points": marker_result["n_time_points"],
        "marker_details": marker_result["marker_details"],
    }


# =============================================================================
# Intermediate-stage validation
# =============================================================================

def linear_scaling(
    generated_data: np.ndarray,
    real_data: np.ndarray,
) -> np.ndarray:
    """
    Gene-wise linear scaling to match real data distribution.

    Each gene in generated data is standardized using its mean and std,
    then transformed to match the corresponding real data mean and std.

    Parameters
    ----------
    generated_data : np.ndarray
        Generated expression matrix (cells x genes).
    real_data : np.ndarray
        Real expression matrix (cells x genes).

    Returns
    -------
    np.ndarray
        Scaled generated data with matched distribution.
    """
    real_mean = real_data.mean(axis=0)
    real_std = real_data.std(axis=0)

    gen_mean = generated_data.mean(axis=0)
    gen_std = generated_data.std(axis=0)

    # Protect against zero std
    gen_std_safe = np.where(gen_std < 1e-8, 1.0, gen_std)
    real_std_safe = np.where(real_std < 1e-8, 0.0, real_std)

    scaled = (generated_data - gen_mean) / gen_std_safe * real_std_safe + real_mean

    return np.maximum(scaled, 0.0)


def compute_feature_level_distance(
    real_mean: np.ndarray,
    gen_mean: np.ndarray,
) -> Dict[str, float]:
    """
    Compute feature-level similarity metrics.

    Parameters
    ----------
    real_mean : np.ndarray
        Mean expression of real cells (genes,).
    gen_mean : np.ndarray
        Mean expression of generated cells (genes,).

    Returns
    -------
    dict
        Pearson, Spearman, and Cosine similarity.
    """
    pearson_r = safe_pearson(real_mean, gen_mean)
    spearman_r = safe_spearman(real_mean, gen_mean)

    try:
        cosine = float(cosine_similarity([real_mean], [gen_mean])[0, 0])
    except Exception:
        cosine = np.nan

    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "cosine": cosine,
    }


def compute_latent_space_distance(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    n_components: int = 50,
) -> float:
    """
    Compute cosine distance in PCA latent space.

    Parameters
    ----------
    real_data : np.ndarray
        Real expression matrix (cells x genes).
    gen_data : np.ndarray
        Generated expression matrix (cells x genes).
    n_components : int
        Number of PCA components.

    Returns
    -------
    float
        Cosine distance (1 - cosine similarity) between mean embeddings.
    """
    try:
        # Fit PCA on real data
        n_comp = min(n_components, real_data.shape[0], real_data.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(real_data)

        # Transform both datasets
        real_latent = pca.transform(real_data)
        gen_latent = pca.transform(gen_data)

        # Compute mean embeddings
        real_mean_emb = real_latent.mean(axis=0)
        gen_mean_emb = gen_latent.mean(axis=0)

        # Cosine distance
        cosine_sim = cosine_similarity([real_mean_emb], [gen_mean_emb])[0, 0]

        return 1.0 - float(cosine_sim)

    except Exception:
        return np.nan


def compute_wasserstein_distance_multi(
    real_data: np.ndarray,
    gen_data: np.ndarray,
) -> float:
    """
    Compute mean Wasserstein distance across all genes.

    Parameters
    ----------
    real_data : np.ndarray
        Real expression matrix (cells x genes).
    gen_data : np.ndarray
        Generated expression matrix (cells x genes).

    Returns
    -------
    float
        Mean Wasserstein distance across genes.
    """
    try:
        n_genes = real_data.shape[1]
        distances = []

        for i in range(n_genes):
            real_expr = real_data[:, i]
            gen_expr = gen_data[:, i]

            # Skip if constant
            if np.std(real_expr) < 1e-12 or np.std(gen_expr) < 1e-12:
                continue

            wass_dist = wasserstein_distance(real_expr, gen_expr)
            distances.append(wass_dist)

        return float(np.mean(distances)) if distances else np.nan

    except Exception:
        return np.nan


def find_best_pseudotime(
    adata_trajectory,
    adata_intermediate,
    common_genes: List[str],
    pseudotime_key: str = "pseudotime",
    search_min: float = 0.0,
    search_max: float = 1.0,
    search_step: float = 0.01,
    window: float = 0.05,
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    Automatically find the pseudotime that best matches intermediate cells.

    Parameters
    ----------
    adata_trajectory : AnnData
        Generated trajectory with pseudotime in obs.
    adata_intermediate : AnnData
        Real intermediate-stage cells.
    common_genes : list of str
        Genes to use for comparison.
    pseudotime_key : str
        Key in adata_trajectory.obs for pseudotime.
    search_min, search_max : float
        Pseudotime search range.
    search_step : float
        Search step size.
    window : float
        Window size around each pseudotime for cell extraction.

    Returns
    -------
    best_t : float
        Best matching pseudotime.
    best_corr : float
        Pearson correlation at best_t.
    best_n_cells : int
        Number of cells at best_t.
    df_search : pd.DataFrame
        Full search results.
    """
    # Extract real intermediate expression
    real_X = to_dense(adata_intermediate[:, common_genes].X)
    real_mean = real_X.mean(axis=0)

    # Get trajectory pseudotime
    trajectory_time = adata_trajectory.obs[pseudotime_key].values

    # Search grid
    search_times = np.arange(search_min, search_max + search_step, search_step)

    results = []

    for t in search_times:
        # Extract cells within window
        mask = np.abs(trajectory_time - t) <= window
        n_cells = mask.sum()

        if n_cells < 10:
            results.append({
                "pseudotime": t,
                "n_cells": n_cells,
                "pearson": np.nan,
            })
            continue

        # Extract generated expression
        gen_X = to_dense(adata_trajectory[mask, :][:, common_genes].X)
        gen_mean = gen_X.mean(axis=0)

        # Compute correlation
        corr = safe_pearson(real_mean, gen_mean)

        results.append({
            "pseudotime": t,
            "n_cells": n_cells,
            "pearson": corr,
        })

    df_search = pd.DataFrame(results)

    # Find best pseudotime
    valid = df_search.dropna(subset=["pearson"])

    if len(valid) == 0:
        return np.nan, np.nan, 0, df_search

    best_row = valid.loc[valid["pearson"].idxmax()]

    return (
        float(best_row["pseudotime"]),
        float(best_row["pearson"]),
        int(best_row["n_cells"]),
        df_search,
    )


def validate_intermediate_stage(
    adata_trajectory,
    adata_intermediate,
    stage_name: str = "intermediate",
    expected_pseudotime: Optional[float] = None,
    markers: Optional[Dict[str, List[str]]] = None,
    pseudotime_key: str = "pseudotime",
    search_window: float = 0.05,
    use_linear_scaling: bool = True,
    n_pca_components: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Validate trajectory against independent intermediate-stage cells.

    This function:
    1. Finds common genes between trajectory and real cells
    2. Searches for the best-matching pseudotime
    3. Computes 3-level distance metrics:
       - Feature-level (Pearson, Spearman, Cosine)
       - Latent space (PCA + cosine distance)
       - Distributional (Wasserstein distance)
    4. Optionally validates marker genes

    Parameters
    ----------
    adata_trajectory : AnnData
        Generated trajectory with pseudotime in obs.
    adata_intermediate : AnnData
        Real intermediate-stage cells (held-out).
    stage_name : str
        Name of the intermediate stage (e.g., 'E9.5', 'GO1').
    expected_pseudotime : float, optional
        Expected pseudotime for this stage (if known).
    markers : dict, optional
        Marker genes for validation.
        Format: {'Lineage1': ['gene1', 'gene2'], 'Lineage2': [...]}
    pseudotime_key : str
        Key in adata_trajectory.obs for pseudotime values.
    search_window : float
        Window size for pseudotime search (±window).
    use_linear_scaling : bool
        Whether to apply linear scaling to match distributions.
    n_pca_components : int
        Number of PCA components for latent space distance.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Validation results including:
        - best_pseudotime: Best matching pseudotime
        - feature_distance: Pearson, Spearman, Cosine
        - latent_distance: PCA cosine distance
        - wasserstein_distance: Mean Wasserstein distance
        - temporal_error: |expected_t - best_t| (if expected_pseudotime provided)
        - marker_results: Marker validation (if markers provided)
        - search_results: Full pseudotime search curve
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Validating intermediate stage: {stage_name}")
        print(f"{'='*70}")
        print(f"Real cells: {adata_intermediate.n_obs}")
        print(f"Trajectory cells: {adata_trajectory.n_obs}")

    # 1. Find common genes
    common_genes = list(
        set(adata_trajectory.var_names) & set(adata_intermediate.var_names)
    )

    if len(common_genes) < 100:
        raise ValueError(
            f"Too few common genes: {len(common_genes)}. "
            "Need at least 100 for robust validation."
        )

    if verbose:
        print(f"Common genes: {len(common_genes)}")

    # 2. Find best-matching pseudotime
    if verbose:
        print("\nSearching for best pseudotime match...")

    best_t, best_corr, best_n_cells, df_search = find_best_pseudotime(
        adata_trajectory=adata_trajectory,
        adata_intermediate=adata_intermediate,
        common_genes=common_genes,
        pseudotime_key=pseudotime_key,
        window=search_window,
    )

    if np.isnan(best_t):
        raise ValueError("Could not find valid pseudotime match.")

    if verbose:
        print(f"Best pseudotime: {best_t:.3f}")
        print(f"  Correlation: {best_corr:.4f}")
        print(f"  N cells: {best_n_cells}")

    # 3. Extract cells at best pseudotime
    trajectory_time = adata_trajectory.obs[pseudotime_key].values
    mask = np.abs(trajectory_time - best_t) <= search_window

    adata_real = adata_intermediate[:, common_genes].copy()
    adata_gen = adata_trajectory[mask, :][:, common_genes].copy()

    # Convert to dense
    real_X = to_dense(adata_real.X)
    gen_X = to_dense(adata_gen.X)

    # Apply linear scaling if requested
    if use_linear_scaling:
        gen_X_scaled = linear_scaling(gen_X, real_X)
    else:
        gen_X_scaled = gen_X

    # Compute mean expressions
    real_mean = real_X.mean(axis=0)
    gen_mean = gen_X_scaled.mean(axis=0)

    # 4. Feature-level distance
    if verbose:
        print("\nComputing feature-level distance...")

    feature_dist = compute_feature_level_distance(real_mean, gen_mean)

    if verbose:
        print(f"  Pearson: {feature_dist['pearson']:.4f}")
        print(f"  Spearman: {feature_dist['spearman']:.4f}")
        print(f"  Cosine: {feature_dist['cosine']:.4f}")

    # 5. Latent space distance
    if verbose:
        print("\nComputing latent space distance...")

    latent_dist = compute_latent_space_distance(
        real_X, gen_X_scaled, n_components=n_pca_components
    )

    if verbose:
        print(f"  PCA cosine distance: {latent_dist:.4f}")

    # 6. Wasserstein distance
    if verbose:
        print("\nComputing Wasserstein distance...")

    wass_dist = compute_wasserstein_distance_multi(real_X, gen_X_scaled)

    if verbose:
        print(f"  Mean Wasserstein: {wass_dist:.4f}")

    # 7. Temporal error
    temporal_error = None
    if expected_pseudotime is not None:
        temporal_error = abs(best_t - expected_pseudotime)
        if verbose:
            print(f"\nTemporal error: {temporal_error:.3f}")
            print(f"  Expected: {expected_pseudotime:.3f}")
            print(f"  Observed: {best_t:.3f}")

    # 8. Marker gene validation
    marker_results = None
    if markers is not None:
        if verbose:
            print("\nValidating marker genes...")

        marker_records = []

        for lineage, gene_list in markers.items():
            available_markers = [g for g in gene_list if g in common_genes]

            if len(available_markers) == 0:
                continue

            for gene in available_markers:
                gene_idx = common_genes.index(gene)

                real_expr = real_X[:, gene_idx]
                gen_expr = gen_X_scaled[:, gene_idx]

                real_expr_mean = real_expr.mean()
                gen_expr_mean = gen_expr.mean()

                abs_error = abs(gen_expr_mean - real_expr_mean)

                if abs(real_expr_mean) > 1e-8:
                    rel_error = 100.0 * abs_error / abs(real_expr_mean)
                else:
                    rel_error = np.nan

                corr = safe_pearson(real_expr, gen_expr)

                marker_records.append({
                    "lineage": lineage,
                    "gene": gene,
                    "real_mean": real_expr_mean,
                    "gen_mean": gen_expr_mean,
                    "abs_error": abs_error,
                    "rel_error_pct": rel_error,
                    "correlation": corr,
                })

        marker_results = pd.DataFrame(marker_records)

        if len(marker_results) > 0:
            good_pct = (
                (marker_results["rel_error_pct"] < 20).sum()
                / len(marker_results)
                * 100
            )
            if verbose:
                print(f"  Marker accuracy: {good_pct:.1f}% good (<20% error)")

    # Compile results
    results = {
        "stage_name": stage_name,
        "best_pseudotime": best_t,
        "best_correlation": best_corr,
        "n_cells_real": adata_intermediate.n_obs,
        "n_cells_generated": best_n_cells,
        "n_common_genes": len(common_genes),
        "feature_distance": feature_dist,
        "latent_distance": latent_dist,
        "wasserstein_distance": wass_dist,
        "temporal_error": temporal_error,
        "expected_pseudotime": expected_pseudotime,
        "marker_results": marker_results,
        "search_results": df_search,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Validation complete: {stage_name}")
        print(f"{'='*70}\n")

    return results
