#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Alignment Evaluation for Developmental Trajectory Methods

Evaluates trajectory reconstruction quality through:
  - Endpoint separation (Cohen's d, Wasserstein distance)
  - Trajectory quality (temporal uniformity, coverage)
  - Intermediate coverage (middle pseudotime occupancy)
  - Quality-aware metric correction

Methods compared: CMAT-Dynamic, CMAT-Attention, scDiffEq, Monocle3,
                  CellRank2, scTour, PBA
"""

import argparse
import os
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")


# ============================================================
# Configuration
# ============================================================

class EvaluationConfig:
    """Configuration for temporal alignment evaluation."""
    
    def __init__(self):
        # Batch/stage identifiers
        self.early_batch = "E7.5"
        self.late_batch = "E12.5"
        
        # Endpoint buckets
        self.bucket_early = 0.15
        self.bucket_late = 0.85
        
        # Sampling parameters
        self.min_endpoint_cells = 50
        self.fallback_quantile = 0.15
        self.enable_global_uniform_sample = True
        self.n_sample_cap = 10000
        self.eval_random_seed = 42
        
        # Binning parameters
        self.temporal_bins = 40
        self.uniformity_bins = 30
        self.intermediate_bins = 30
        self.expected_unique_time_points = 200
        
        # Numerical stability
        self.eps = 1e-8
        self.std_floor = 1e-8
        self.reference_std_min = 0.02
        
        # Correction parameters
        self.gamma_d = 1.5
        self.gamma_acc = 1.0
        self.gamma_wass = 0.7
        self.d_factor_floor = 1e-6
        self.acc_factor_floor = 0.03
        self.wass_factor_floor = 0.03
        
        # Visualization
        self.save_dpi = 300


# ============================================================
# Utility Functions
# ============================================================

def safe_minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0,1], handling NaN values."""
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    out = np.full_like(x, np.nan, dtype=float)
    if mask.sum() == 0:
        return out
    valid = x[mask]
    xmin, xmax = np.min(valid), np.max(valid)
    if xmax <= xmin:
        out[mask] = 0.0
    else:
        out[mask] = (valid - xmin) / (xmax - xmin)
    return out


def entropy_score(values: np.ndarray, bins: int = 40, eps: float = 1e-8) -> float:
    """Compute normalized entropy of distribution."""
    values = values[~np.isnan(values)]
    if len(values) <= 1:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = hist.astype(float) / (hist.sum() + eps)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    entropy = -np.sum(p * np.log(p + eps))
    return float(np.clip(entropy / (np.log(bins) + eps), 0.0, 1.0))


def occupancy_score(values: np.ndarray, bins: int = 40) -> float:
    """Compute fraction of occupied bins."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return float(np.clip(np.sum(hist > 0) / bins, 0.0, 1.0))


def unique_time_support(
    values: np.ndarray,
    expected_unique: int = 200,
    decimals: int = 4
) -> float:
    """Compute fraction of expected unique time points covered."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    n_unique = len(np.unique(np.round(values, decimals=decimals)))
    return float(np.clip(min(n_unique, expected_unique) / expected_unique, 0.0, 1.0))


def max_gap_score(values: np.ndarray) -> float:
    """Compute score based on maximum gap between consecutive unique values."""
    values = values[~np.isnan(values)]
    if len(values) <= 2:
        return 0.0
    unique_vals = np.unique(np.round(values, 5))
    if len(unique_vals) <= 2:
        return 0.0
    max_gap = np.max(np.diff(np.sort(unique_vals)))
    return float(np.clip(1.0 - max_gap, 0.0, 1.0))


def temporal_uniformity_score(values: np.ndarray, bins: int = 30) -> float:
    """Composite score for temporal uniformity."""
    ent = entropy_score(values, bins=bins)
    occ = occupancy_score(values, bins=bins)
    gap = max_gap_score(values)
    return float(np.clip(0.45 * ent + 0.35 * occ + 0.20 * gap, 0.0, 1.0))


def temporal_coverage_width(values: np.ndarray) -> float:
    """Measure robust full-axis coverage using 1% and 99% quantiles."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    q01, q99 = np.quantile(values, 0.01), np.quantile(values, 0.99)
    return float(np.clip(q99 - q01, 0.0, 1.0))


def intermediate_coverage_score(
    values: np.ndarray,
    low: float = 0.15,
    high: float = 0.85,
    bins: int = 30,
) -> float:
    """
    Quantify coverage of intermediate pseudotime states.

    Combines:
      - Fraction of cells in middle region (saturated at 40%)
      - Occupancy of bins within middle region
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    middle = values[(values > low) & (values < high)]
    if len(middle) == 0:
        return 0.0
    middle_fraction_score = min(len(middle) / len(values) / 0.40, 1.0)
    hist, _ = np.histogram(middle, bins=bins, range=(low, high))
    middle_occupancy = np.sum(hist > 0) / bins
    return float(np.clip(0.5 * middle_fraction_score + 0.5 * middle_occupancy, 0.0, 1.0))


def trajectory_quality_score(
    values: np.ndarray,
    temporal_bins: int = 40,
    uniformity_bins: int = 30,
    expected_unique: int = 200,
) -> float:
    """
    Composite trajectory quality score.

    Combines:
      - Temporal coverage width
      - Unique time support
      - Temporal uniformity
    """
    cov = temporal_coverage_width(values)
    sup = unique_time_support(values, expected_unique=expected_unique)
    uni = temporal_uniformity_score(values, bins=uniformity_bins)
    return float(np.clip(0.35 * cov + 0.30 * sup + 0.35 * uni, 0.0, 1.0))


# ============================================================
# Endpoint Extraction and Sampling
# ============================================================

def extract_endpoints(
    adata,
    config: EvaluationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract early and late endpoint cells based on pseudotime buckets.

    Returns
    -------
    early_pt : array
        Pseudotime values for early cells
    late_pt : array
        Pseudotime values for late cells
    full_pt : array
        Pseudotime values for all cells
    """
    batch_col = None
    for col in ["batch", "stage", "time", "sample"]:
        if col in adata.obs.columns:
            batch_col = col
            break

    if batch_col is None:
        raise ValueError("No batch/stage column found in adata.obs")

    pt_col = None
    for col in ["pseudotime", "dpt_pseudotime", "latent_time"]:
        if col in adata.obs.columns:
            pt_col = col
            break

    if pt_col is None:
        raise ValueError("No pseudotime column found in adata.obs")

    full_pt = adata.obs[pt_col].astype(float).values
    full_pt = safe_minmax_normalize(full_pt)

    early_mask = adata.obs[batch_col].astype(str) == config.early_batch
    late_mask = adata.obs[batch_col].astype(str) == config.late_batch

    early_pt_all = full_pt[early_mask]
    late_pt_all = full_pt[late_mask]

    early_pt = early_pt_all[early_pt_all <= config.bucket_early]
    late_pt = late_pt_all[late_pt_all >= config.bucket_late]

    if len(early_pt) < config.min_endpoint_cells:
        q = np.quantile(early_pt_all, config.fallback_quantile)
        early_pt = early_pt_all[early_pt_all <= q]

    if len(late_pt) < config.min_endpoint_cells:
        q = np.quantile(late_pt_all, 1.0 - config.fallback_quantile)
        late_pt = late_pt_all[late_pt_all >= q]

    return early_pt, late_pt, full_pt


def global_uniform_sample(
    all_endpoints: Dict[str, Dict],
    n_cap: int,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Uniformly sample endpoints across all methods to control cell-number effects.
    """
    rng = np.random.default_rng(seed)
    sampled = {}

    for method, ep in all_endpoints.items():
        early_pt = ep["early_pt"]
        late_pt = ep["late_pt"]

        n_early_sample = min(len(early_pt), n_cap)
        n_late_sample = min(len(late_pt), n_cap)

        if len(early_pt) > n_early_sample:
            idx = rng.choice(len(early_pt), size=n_early_sample, replace=False)
            early_pt = early_pt[idx]

        if len(late_pt) > n_late_sample:
            idx = rng.choice(len(late_pt), size=n_late_sample, replace=False)
            late_pt = late_pt[idx]

        sampled[method] = {
            "early_pt": early_pt,
            "late_pt": late_pt,
            "full_pt": ep["full_pt"],
        }

    return sampled


# ============================================================
# Raw Metrics Computation
# ============================================================

def compute_stable_cohens_d(
    early_pt: np.ndarray,
    late_pt: np.ndarray,
    pooled_std: float,
    std_floor: float = 1e-8,
) -> float:
    """Compute Cohen's d using shared reference pooled standard deviation."""
    if pooled_std < std_floor:
        return 0.0
    mean_diff = np.mean(late_pt) - np.mean(early_pt)
    return float(mean_diff / pooled_std)


def compute_method_specific_cohens_d(
    early_pt: np.ndarray,
    late_pt: np.ndarray,
    std_floor: float = 1e-8,
) -> float:
    """Compute Cohen's d using method's own pooled standard deviation."""
    n_early, n_late = len(early_pt), len(late_pt)
    if n_early < 2 or n_late < 2:
        return 0.0

    var_early = np.var(early_pt, ddof=1)
    var_late = np.var(late_pt, ddof=1)
    pooled_var = ((n_early - 1) * var_early + (n_late - 1) * var_late) / (n_early + n_late - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < std_floor:
        return 0.0

    mean_diff = np.mean(late_pt) - np.mean(early_pt)
    return float(mean_diff / pooled_std)


def compute_classification_accuracy(early_pt: np.ndarray, late_pt: np.ndarray) -> float:
    """Compute classification accuracy using optimal threshold."""
    if len(early_pt) == 0 or len(late_pt) == 0:
        return 0.0

    all_pt = np.concatenate([early_pt, late_pt])
    labels = np.concatenate([np.zeros(len(early_pt)), np.ones(len(late_pt))])

    thresholds = np.linspace(0.0, 1.0, 1000)
    best_acc = 0.0

    for thr in thresholds:
        pred = (all_pt >= thr).astype(int)
        acc = np.mean(pred == labels)
        best_acc = max(best_acc, acc)

    return float(best_acc)


def compute_wasserstein_separation(early_pt: np.ndarray, late_pt: np.ndarray) -> float:
    """Compute Wasserstein distance between early and late distributions."""
    if len(early_pt) == 0 or len(late_pt) == 0:
        return 0.0
    return float(wasserstein_distance(early_pt, late_pt))


def compute_raw_metrics(
    all_endpoints: Dict[str, Dict],
    config: EvaluationConfig,
    reference_pooled_std: float,
) -> Dict[str, Dict]:
    """Compute raw metrics for all methods."""
    results = {}

    for method, ep in all_endpoints.items():
        early_pt = ep["early_pt"]
        late_pt = ep["late_pt"]
        full_pt = ep["full_pt"]

        stable_d = compute_stable_cohens_d(
            early_pt, late_pt, reference_pooled_std, config.std_floor
        )
        method_d = compute_method_specific_cohens_d(early_pt, late_pt, config.std_floor)
        acc = compute_classification_accuracy(early_pt, late_pt)
        wass = compute_wasserstein_separation(early_pt, late_pt)
        traj_qual = trajectory_quality_score(
            full_pt, config.temporal_bins, config.uniformity_bins, config.expected_unique_time_points
        )
        inter_cov = intermediate_coverage_score(
            full_pt, config.bucket_early, config.bucket_late, config.intermediate_bins
        )

        results[method] = {
            "Method": method,
            "Stable Raw Cohen's d": stable_d,
            "Method-specific Raw Cohen's d": method_d,
            "D correction used": "stable",
            "Raw Classification Acc": acc,
            "Raw Wasserstein Sep": wass,
            "Trajectory Quality Score": traj_qual,
            "Intermediate Coverage Score": inter_cov,
            "N Early": len(early_pt),
            "N Late": len(late_pt),
            "early_pt": early_pt,
            "late_pt": late_pt,
            "full_pt": full_pt,
        }

    return results


# ============================================================
# Quality Factors
# ============================================================

def sample_balance_score(n_early: int, n_late: int) -> float:
    """Compute balance between early and late samples."""
    if n_early <= 0 or n_late <= 0:
        return 0.0
    return float(np.clip(min(n_early, n_late) / max(n_early, n_late), 0.0, 1.0))


def endpoint_support_score(n_early: int, n_late: int, reference_n: int) -> float:
    """Compute score based on effective sample size."""
    if n_early <= 0 or n_late <= 0 or reference_n <= 0:
        return 0.0
    n_eff = min(n_early, n_late)
    return float(np.clip(np.sqrt(min(n_eff, reference_n) / reference_n), 0.0, 1.0))


def compute_quality_factors(
    raw_results: Dict[str, Dict],
    original_counts: Dict[str, Tuple[int, int]],
    global_n: int,
    config: EvaluationConfig,
) -> Tuple[Dict[str, Dict], float]:
    """
    Compute data-driven quality factors for metric correction.

    Returns
    -------
    factor_results : dict
        Quality factors for each method
    reference_pooled_std : float
        Global reference standard deviation
    """
    all_early = np.concatenate([r["early_pt"] for r in raw_results.values()])
    all_late = np.concatenate([r["late_pt"] for r in raw_results.values()])

    n_early_total = len(all_early)
    n_late_total = len(all_late)

    var_early = np.var(all_early, ddof=1) if n_early_total > 1 else 0.0
    var_late = np.var(all_late, ddof=1) if n_late_total > 1 else 0.0

    pooled_var = (
        (n_early_total - 1) * var_early + (n_late_total - 1) * var_late
    ) / (n_early_total + n_late_total - 2)
    reference_pooled_std = max(np.sqrt(pooled_var), config.reference_std_min)

    factor_results = {}

    for method, r in raw_results.items():
        early_pt = r["early_pt"]
        late_pt = r["late_pt"]
        n_early, n_late = len(early_pt), len(late_pt)

        balance = sample_balance_score(n_early, n_late)
        support = endpoint_support_score(n_early, n_late, global_n)

        std_early = (
            max(np.std(early_pt, ddof=1), config.std_floor) if n_early > 1 else config.std_floor
        )
        std_late = (
            max(np.std(late_pt, ddof=1), config.std_floor) if n_late > 1 else config.std_floor
        )

        d_factor = np.sqrt(balance * support) * min(
            std_early / reference_pooled_std, std_late / reference_pooled_std
        )
        d_factor = max(d_factor, config.d_factor_floor)

        n_orig_early, n_orig_late = original_counts.get(method, (n_early, n_late))
        retention_early = n_early / max(n_orig_early, 1)
        retention_late = n_late / max(n_orig_late, 1)
        retention = min(retention_early, retention_late)

        acc_factor = np.sqrt(balance * support * retention)
        acc_factor = max(acc_factor, config.acc_factor_floor)

        wass_factor = np.sqrt(balance * support)
        wass_factor = max(wass_factor, config.wass_factor_floor)

        factor_results[method] = {
            "Method": method,
            "Balance": balance,
            "Support": support,
            "Retention": retention,
            "D Factor": d_factor,
            "Acc Factor": acc_factor,
            "Wass Factor": wass_factor,
        }

    return factor_results, reference_pooled_std


# ============================================================
# Metric Correction
# ============================================================

def apply_metric_specific_correction(
    raw_results: Dict[str, Dict],
    factor_results: Dict[str, Dict],
    config: EvaluationConfig,
) -> Dict[str, Dict]:
    """Apply quality-aware correction to raw metrics."""
    corrected = {}

    for method in raw_results.keys():
        raw = raw_results[method]
        factors = factor_results[method]

        d_raw = raw["Stable Raw Cohen's d"]
        acc_raw = raw["Raw Classification Acc"]
        wass_raw = raw["Raw Wasserstein Sep"]

        d_factor = factors["D Factor"]
        acc_factor = factors["Acc Factor"]
        wass_factor = factors["Wass Factor"]

        d_corrected = d_raw * (d_factor ** config.gamma_d)
        acc_corrected = acc_raw * (acc_factor ** config.gamma_acc)
        wass_corrected = wass_raw * (wass_factor ** config.gamma_wass)

        corrected[method] = {
            "Method": method,
            "Stable Raw Cohen's d": d_raw,
            "Method-specific Raw Cohen's d": raw["Method-specific Raw Cohen's d"],
            "D correction used": raw["D correction used"],
            "Corrected Cohen's d": d_corrected,
            "Raw Classification Acc": acc_raw,
            "Corrected Classification Acc": acc_corrected,
            "Raw Wasserstein Sep": wass_raw,
            "Corrected Wasserstein Sep": wass_corrected,
            "Trajectory Quality Score": raw["Trajectory Quality Score"],
            "Intermediate Coverage Score": raw["Intermediate Coverage Score"],
        }

    return corrected


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite FAIR score combining all corrected metrics."""
    df = df.copy()

    d_norm = safe_minmax_normalize(df["Corrected Cohen's d"].values)
    wass_norm = safe_minmax_normalize(df["Corrected Wasserstein Sep"].values)
    traj_norm = df["Trajectory Quality Score"].values
    inter_norm = df["Intermediate Coverage Score"].values

    composite = 0.30 * d_norm + 0.25 * wass_norm + 0.25 * traj_norm + 0.20 * inter_norm
    df["Composite FAIR Score"] = composite

    return df


# ============================================================
# Visualization
# ============================================================

def plot_pseudotime_distribution(
    all_obs: Dict[str, pd.DataFrame],
    output_path: str,
    dpi: int = 300,
):
    """Plot full pseudotime distributions for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, obs in all_obs.items():
        pt = obs["pseudotime"].values
        ax.hist(pt, bins=50, alpha=0.5, label=method, density=True)

    ax.set_xlabel("Pseudotime", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Full Pseudotime Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


def plot_endpoint_distribution(
    raw_results: Dict[str, Dict],
    output_path: str,
    dpi: int = 300,
):
    """Plot endpoint pseudotime distributions."""
    n_methods = len(raw_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)

    if n_methods == 1:
        axes = [axes]

    for ax, (method, r) in zip(axes, raw_results.items()):
        early_pt = r["early_pt"]
        late_pt = r["late_pt"]

        ax.hist(early_pt, bins=30, alpha=0.6, color="blue", label="Early", density=True)
        ax.hist(late_pt, bins=30, alpha=0.6, color="red", label="Late", density=True)
        ax.set_xlabel("Pseudotime", fontsize=11)
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


def plot_factor_heatmap(
    factor_df: pd.DataFrame,
    output_path: str,
    dpi: int = 300,
):
    """Plot quality factor heatmap."""
    factor_cols = ["Balance", "Support", "Retention", "D Factor", "Acc Factor", "Wass Factor"]
    data = factor_df[factor_cols].values
    methods = factor_df["Method"].values

    fig, ax = plt.subplots(figsize=(8, len(methods) * 0.6 + 1))

    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)

    ax.set_xticks(range(len(factor_cols)))
    ax.set_xticklabels(factor_cols, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)

    for i in range(len(methods)):
        for j in range(len(factor_cols)):
            text = ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                color="black" if data[i, j] > 0.5 else "white",
                fontsize=10
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score", fontsize=11)

    ax.set_title("Quality Factors", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


def plot_metric_bars(
    df: pd.DataFrame,
    metrics: List[str],
    title: str,
    output_path: str,
    dpi: int = 300,
):
    """Plot bar chart for specified metrics."""
    df = df.sort_values(by=metrics[0], ascending=False)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = df[metric].values
        methods = df["Method"].values

        bars = ax.barh(methods, values, color="steelblue", edgecolor="black", linewidth=1.2)
        ax.set_xlabel(metric, fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.1)
        ax.grid(axis="x", alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + max(values) * 0.02, i, f"{val:.3f}", va="center", fontsize=10)

    axes[0].invert_yaxis()
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


# ============================================================
# Main Pipeline
# ============================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Temporal Alignment Evaluation for Developmental Trajectory Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--trajectory-files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to trajectory .h5ad files (format: method_name:path)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./temporal_alignment_results/",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--early-batch",
        type=str,
        default="E7.5",
        help="Early batch/stage identifier",
    )
    
    parser.add_argument(
        "--late-batch",
        type=str,
        default="E12.5",
        help="Late batch/stage identifier",
    )
    
    parser.add_argument(
        "--bucket-early",
        type=float,
        default=0.15,
        help="Early endpoint bucket threshold",
    )
    
    parser.add_argument(
        "--bucket-late",
        type=float,
        default=0.85,
        help="Late endpoint bucket threshold",
    )
    
    parser.add_argument(
        "--n-sample-cap",
        type=int,
        default=10000,
        help="Maximum number of cells to sample per endpoint",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    # Initialize configuration
    config = EvaluationConfig()
    config.early_batch = args.early_batch
    config.late_batch = args.late_batch
    config.bucket_early = args.bucket_early
    config.bucket_late = args.bucket_late
    config.n_sample_cap = args.n_sample_cap
    config.eval_random_seed = args.seed
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Parse trajectory files
    trajectory_files = {}
    for item in args.trajectory_files:
        parts = item.split(":")
        if len(parts) != 2:
            print(f"[ERROR] Invalid format for --trajectory-files: {item}")
            print("Expected format: method_name:path")
            return
        method_name, path = parts
        trajectory_files[method_name] = path

    print("=" * 80)
    print("TEMPORAL ALIGNMENT EVALUATION")
    print("=" * 80)

    # Load trajectories
    print("\nLoading trajectories...")
    all_adata = {}
    all_obs = {}
    failed_methods = []

    for method, path in trajectory_files.items():
        if not os.path.exists(path):
            print(f"[SKIP] {method}: file not found")
            failed_methods.append({"Method": method, "Reason": "File not found"})
            continue

        try:
            adata = sc.read_h5ad(path)
            all_adata[method] = adata
            all_obs[method] = adata.obs
            print(f"[OK] {method}: {adata.n_obs} cells")
        except Exception as e:
            print(f"[FAIL] {method}: {e}")
            failed_methods.append({"Method": method, "Reason": str(e)})

    if len(all_adata) == 0:
        print("[FATAL] No valid trajectories loaded.")
        return

    # Extract endpoints
    print("\nExtracting endpoints...")
    all_endpoints = {}
    original_counts = {}

    for method, adata in all_adata.items():
        try:
            early_pt, late_pt, full_pt = extract_endpoints(adata, config)

            all_endpoints[method] = {
                "early_pt": early_pt,
                "late_pt": late_pt,
                "full_pt": full_pt,
            }
            original_counts[method] = (len(early_pt), len(late_pt))

            print(f"  {method}: Early={len(early_pt)}, Late={len(late_pt)}")
        except Exception as e:
            print(f"[FAIL] {method}: {e}")
            failed_methods.append({"Method": method, "Reason": f"Endpoint extraction: {e}"})

    if len(all_endpoints) == 0:
        print("[FATAL] No endpoints extracted.")
        return

    # Global uniform sampling
    if config.enable_global_uniform_sample:
        print("\nApplying global uniform sampling...")
        all_endpoints = global_uniform_sample(
            all_endpoints, config.n_sample_cap, config.eval_random_seed
        )
        global_n = config.n_sample_cap
        for method, ep in all_endpoints.items():
            print(f"  {method}: Early={len(ep['early_pt'])}, Late={len(ep['late_pt'])}")
    else:
        global_n = max(
            max(len(ep["early_pt"]), len(ep["late_pt"])) for ep in all_endpoints.values()
        )

    # Compute quality factors first to get reference_pooled_std
    print("\nComputing quality factors...")
    # We need to compute raw results first with a placeholder reference_pooled_std
    raw_results_temp = compute_raw_metrics(all_endpoints, config, config.reference_std_min)
    
    factor_results, reference_pooled_std = compute_quality_factors(
        raw_results_temp, original_counts, global_n, config
    )

    # Recompute raw metrics with correct reference_pooled_std
    print("\nComputing raw metrics...")
    raw_results = compute_raw_metrics(all_endpoints, config, reference_pooled_std)

    # Apply corrections
    print("\nApplying quality-aware corrections...")
    corrected_results = apply_metric_specific_correction(raw_results, factor_results, config)

    # Create DataFrames
    raw_df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ["early_pt", "late_pt", "full_pt"]}
            for r in raw_results.values()
        ]
    )
    factor_df = pd.DataFrame(list(factor_results.values()))
    corrected_df = pd.DataFrame(list(corrected_results.values()))
    corrected_df = add_composite_score(corrected_df)

    # Save results
    raw_df.to_csv(os.path.join(output_dir, "raw_metrics.csv"), index=False)
    factor_df.to_csv(os.path.join(output_dir, "quality_factors.csv"), index=False)
    corrected_df.to_csv(os.path.join(output_dir, "corrected_metrics.csv"), index=False)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    display_cols = [
        "Method",
        "Corrected Cohen's d",
        "Corrected Wasserstein Sep",
        "Trajectory Quality Score",
        "Intermediate Coverage Score",
        "Composite FAIR Score",
    ]
    print(corrected_df[display_cols].to_string(index=False))

    # Generate plots
    print("\nGenerating plots...")
    plot_pseudotime_distribution(
        all_obs,
        os.path.join(output_dir, "full_pseudotime_distribution.png"),
        config.save_dpi,
    )
    plot_endpoint_distribution(
        raw_results,
        os.path.join(output_dir, "endpoint_distribution.png"),
        config.save_dpi,
    )
    plot_factor_heatmap(
        factor_df,
        os.path.join(output_dir, "quality_factor_heatmap.png"),
        config.save_dpi,
    )
    plot_metric_bars(
        corrected_df,
        [
            "Corrected Cohen's d",
            "Corrected Wasserstein Sep",
            "Trajectory Quality Score",
            "Intermediate Coverage Score",
        ],
        "Quality-aware Corrected Metrics",
        os.path.join(output_dir, "corrected_metrics.png"),
        config.save_dpi,
    )

    print(f"\n[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
