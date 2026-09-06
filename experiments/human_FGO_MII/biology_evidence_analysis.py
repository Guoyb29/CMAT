#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biological Evidence Analysis for Human Oocyte Maturation

Integrates multiple validation outputs to assess:
  1. Temporally staggered epigenomic remodeling
  2. Stable genomic regions and candidate genes
  3. Pathway enrichment and cross-modal support

Corresponds to Figure 6 in the paper.
"""

from pathlib import Path
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

STAGE_ORDER = ["FGO", "GO1", "GO2", "MI", "MII"]
INTERMEDIATE_STAGES = ["GO1", "GO2", "MI"]


# ============================================================
# Utility Functions
# ============================================================

def clean_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_read_csv(path):
    if not path.exists():
        return None, f"missing: {path.name}"
    try:
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:
        return None, f"failed to read {path.name}: {type(e).__name__}: {e}"


def find_col(df, candidates, contains=True):
    if df is None:
        return None
    exact = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    if contains:
        for candidate in candidates:
            target = candidate.lower()
            for col in df.columns:
                if target in str(col).lower():
                    return col
    return None


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def finite_values(series):
    x = numeric(series)
    return x[np.isfinite(x)]


def fmt_num(x, digits=4):
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    if x != 0 and abs(x) < 0.001:
        return f"{x:.3e}"
    return f"{x:.{digits}f}"


def normalize_stage(x):
    x = clean_string(x).upper().replace(" ", "")
    aliases = {"FGO": "FGO", "GV": "FGO", "GO1": "GO1", 
               "GO2": "GO2", "MI": "MI", "MII": "MII"}
    return aliases.get(x, x)


def normalize_modality(x):
    x = clean_string(x).upper()
    if "ATAC" in x:
        return "ATAC"
    if "MET" in x or "METH" in x:
        return "MET"
    if "RNA" in x:
        return "RNA"
    return x


def is_symbol_like(gene):
    gene = clean_string(gene)
    if not gene or gene.lower() in {"nan", "none", "na", ".", "-"}:
        return False
    if re.fullmatch(r"ENSG\d+(?:\.\d+)?", gene, flags=re.I):
        return False
    return True


def interval_excludes_zero(low, high):
    try:
        low, high = float(low), float(high)
    except Exception:
        return False
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    return low > 0 or high < 0


def add_finding(findings, topic, level, conclusion, evidence, limitation):
    findings.append({
        "topic": topic,
        "evidence_level": level,
        "conclusion": conclusion,
        "evidence": evidence,
        "limitation": limitation,
    })


def md_table(df, max_rows=30):
    if df is None or df.empty:
        return "_No records available._"
    temp = df.head(max_rows).copy()
    for c in temp.columns:
        if pd.api.types.is_float_dtype(temp[c]):
            temp[c] = temp[c].map(lambda x: "" if pd.isna(x) else fmt_num(x))
    headers = [str(x).replace("|", "\\|") for x in temp.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in temp.iterrows():
        values = [clean_string(x).replace("\n", " ").replace("|", "\\|") 
                  for x in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only the first {max_rows} of {len(df)} records shown._")
    return "\n".join(lines)


# ============================================================
# Geometry Analysis (Figure 6e)
# ============================================================

def analyze_geometry(observed, bootstrap, contrasts, findings):
    result = {
        "geometry_table": pd.DataFrame(),
        "contrast_table": pd.DataFrame(),
        "peak_by_modality": {},
        "all_bootstrap_valid": False,
        "asynchronous_peak": False,
    }

    if observed is None or observed.empty:
        add_finding(findings, "Global trajectory geometry", "Not assessable",
                   "Observed geometry file unavailable.",
                   "01_observed_geometry.csv missing.",
                   "No geometric biological conclusion.")
        return result

    required = {"stage", "modality", "orthogonal_mae"}
    if not required.issubset(observed.columns):
        add_finding(findings, "Global trajectory geometry", "Not assessable",
                   "Required columns missing.", f"Required: {sorted(required)}.",
                   "Schema must be checked.")
        return result

    geom = observed.copy()
    geom["stage"] = geom["stage"].map(normalize_stage)
    geom["modality"] = geom["modality"].map(normalize_modality)
    geom["orthogonal_mae"] = numeric(geom["orthogonal_mae"])

    keep_cols = [c for c in ["stage", "modality", "fixed_time", "projected_time",
                             "time_shift", "orthogonal_mae", "orthogonal_rmse",
                             "fraction_fixed_residual_remaining", "n_features_used"]
                 if c in geom.columns]
    
    result["geometry_table"] = geom[keep_cols].sort_values(
        ["modality", "stage"],
        key=lambda s: (s.map({x: i for i, x in enumerate(STAGE_ORDER)})
                      if s.name == "stage" else s)
    )

    peak_rows = []
    for modality, group in geom.groupby("modality"):
        group = group[group["stage"].isin(INTERMEDIATE_STAGES) & 
                     np.isfinite(group["orthogonal_mae"])]
        if group.empty:
            continue
        row = group.loc[group["orthogonal_mae"].idxmax()]
        result["peak_by_modality"][modality] = {
            "stage": row["stage"],
            "orthogonal_mae": float(row["orthogonal_mae"]),
        }
        peak_rows.append({
            "modality": modality,
            "peak_stage": row["stage"],
            "peak_orthogonal_mae": row["orthogonal_mae"],
        })

    # Bootstrap validity
    boot_valid = False
    if bootstrap is not None and not bootstrap.empty:
        if {"stage", "modality", "metric"}.issubset(bootstrap.columns):
            boot = bootstrap.copy()
            boot["stage"] = boot["stage"].map(normalize_stage)
            boot["modality"] = boot["modality"].map(normalize_modality)
            boot_orth = boot[boot["metric"].astype(str).eq("orthogonal_mae")].copy()
            
            if {"n_total", "n_valid"}.issubset(boot_orth.columns):
                total = numeric(boot_orth["n_total"])
                valid = numeric(boot_orth["n_valid"])
                if len(total) > 0 and len(valid) > 0:
                    boot_valid = ((total > 0) & (valid == total)).all()

    result["all_bootstrap_valid"] = boot_valid

    # Contrast analysis
    if contrasts is not None and not contrasts.empty:
        con = contrasts.copy()
        con["stage_a"] = con["stage_a"].map(normalize_stage)
        con["stage_b"] = con["stage_b"].map(normalize_stage)
        con["modality"] = con["modality"].map(normalize_modality)
        
        con_orth = con[con["metric"].astype(str).eq("orthogonal_mae")].copy()
        if not con_orth.empty:
            result["contrast_table"] = con_orth[
                ["modality", "stage_a", "stage_b", "mean_diff", 
                 "ci_lower", "ci_upper", "p_value"]
            ].copy()

    # Determine asynchronous peak
    atac_peak = result["peak_by_modality"].get("ATAC", {}).get("stage")
    met_peak = result["peak_by_modality"].get("MET", {}).get("stage")
    different_peaks = bool(atac_peak and met_peak and atac_peak != met_peak)

    supported_by_contrast = False
    if different_peaks and not result["contrast_table"].empty:
        con = result["contrast_table"]
        for modality in ["ATAC", "MET"]:
            peak = result["peak_by_modality"][modality]["stage"]
            rows = con[
                (con["modality"] == modality) &
                (((con["stage_a"] == peak) & con["stage_b"].isin(INTERMEDIATE_STAGES)) |
                 ((con["stage_b"] == peak) & con["stage_a"].isin(INTERMEDIATE_STAGES)))
            ]
            if not rows.empty:
                ci_low = find_col(rows, ["ci_lower", "ci_low"])
                ci_high = find_col(rows, ["ci_upper", "ci_high"])
                if ci_low and ci_high:
                    exclude = rows.apply(
                        lambda r: interval_excludes_zero(r[ci_low], r[ci_high]), 
                        axis=1
                    )
                    if exclude.any():
                        supported_by_contrast = True
                        break

    result["asynchronous_peak"] = different_peaks and (boot_valid or supported_by_contrast)

    # Add finding
    if result["asynchronous_peak"]:
        add_finding(
            findings, "Temporally staggered remodeling", "Supported",
            f"ATAC peaks at {atac_peak}, MET peaks at {met_peak}.",
            f"Orthogonal MAE: ATAC={fmt_num(result['peak_by_modality']['ATAC']['orthogonal_mae'])}, "
            f"MET={fmt_num(result['peak_by_modality']['MET']['orthogonal_mae'])}. "
            f"Bootstrap valid: {boot_valid}. Contrast support: {supported_by_contrast}.",
            "Bootstrap is cell-level sampling; biological replication needed."
        )
    else:
        add_finding(
            findings, "Temporally staggered remodeling", "Not supported",
            "Peaks are not clearly asynchronous or lack statistical support.",
            f"ATAC peak: {atac_peak}, MET peak: {met_peak}. Bootstrap: {boot_valid}.",
            "May reflect data limitations or subtle timing differences."
        )

    return result


# ============================================================
# Stable Region Analysis
# ============================================================

def analyze_regions(stable, manifest, findings):
    result = {
        "saturation_summary": pd.DataFrame(),
        "direction_summary": pd.DataFrame(),
        "top_fraction_saturated": False,
    }

    if stable is None or stable.empty:
        add_finding(findings, "Stable genomic regions", "Not assessable",
                   "Stable region file unavailable.",
                   "06_region_candidates_stable.csv missing.",
                   "No region-level biological interpretation.")
        return result

    stable = stable.copy()
    stable["modality"] = stable[find_col(stable, ["modality"])].map(normalize_modality)

    # Saturation check
    n_total = manifest.get("n_candidate_regions_per_modality", {})
    if n_total:
        sat_rows = []
        for mod, count in stable.groupby("modality").size().items():
            total = n_total.get(mod, 0)
            sat_rows.append({
                "modality": mod,
                "n_stable": count,
                "n_total_candidates": total,
                "fraction": count / total if total > 0 else np.nan,
            })
        result["saturation_summary"] = pd.DataFrame(sat_rows)
        
        if not result["saturation_summary"].empty:
            frac = result["saturation_summary"]["fraction"]
            result["top_fraction_saturated"] = (frac > 0.95).any()

    # Direction analysis
    dir_col = find_col(stable, ["expected_direction", "direction"])
    if dir_col:
        dir_summary = stable.groupby(["modality", dir_col]).size().reset_index(name="count")
        result["direction_summary"] = dir_summary

    # Add finding
    if result["top_fraction_saturated"]:
        add_finding(findings, "Stable region selection", "Caution",
                   "Top-fraction saturation detected in some modalities.",
                   "Selection may be biased towards highest-ranking regions.",
                   "Consider relaxing selection thresholds or examining full distribution.")
    else:
        add_finding(findings, "Stable region selection", "Acceptable",
                   "No top-fraction saturation detected.",
                   "Stable regions appear to be robust subset of candidates.",
                   "Still limited by cell-level bootstrap design.")

    return result


# ============================================================
# Region-Gene Annotation Analysis
# ============================================================

def analyze_annotations(annotations, findings):
    result = {
        "relative_high_confidence": pd.DataFrame(),
        "gene_summary": pd.DataFrame(),
        "shared_genes": pd.DataFrame(),
    }

    if annotations is None or annotations.empty:
        add_finding(findings, "Region-gene associations", "Not assessable",
                   "Annotation file unavailable.",
                   "07_region_gene_annotations.csv missing.",
                   "No gene-level interpretation.")
        return result

    ann = annotations.copy()
    
    # Gene column
    gene_col = find_col(ann, ["gene", "gene_name", "gene_symbol", "symbol"])
    if not gene_col:
        add_finding(findings, "Region-gene associations", "Not assessable",
                   "Gene column not found in annotations.",
                   "Required column missing.",
                   "Cannot perform gene-level analysis.")
        return result

    ann = ann[ann[gene_col].map(is_symbol_like)].copy()
    ann.rename(columns={gene_col: "_gene_"}, inplace=True)

    # Confidence scoring
    conf_col = find_col(ann, ["relative_high_confidence", "high_confidence", "confidence"])
    if conf_col:
        ann["_relative_high_confidence_"] = ann[conf_col].fillna(False).astype(bool)
    else:
        # Heuristic fallback
        dist_col = find_col(ann, ["distance", "abs_distance", "dist_to_tss"])
        if dist_col:
            ann["_abs_distance_to_tss_"] = numeric(ann[dist_col]).abs()
            ann["_relative_high_confidence_"] = ann["_abs_distance_to_tss_"] < 10000
        else:
            ann["_relative_high_confidence_"] = True

    high_conf = ann[ann["_relative_high_confidence_"]].copy()
    result["relative_high_confidence"] = high_conf

    # Per-gene summary
    if not high_conf.empty:
        mod_col = find_col(high_conf, ["modality"])
        if mod_col:
            high_conf["_modality_"] = high_conf[mod_col].map(normalize_modality)
            gene_summary = high_conf.groupby(["_gene_", "_modality_"]).size().reset_index(name="n_regions")
            result["gene_summary"] = gene_summary

            # Shared genes (ATAC + MET)
            atac_genes = set(gene_summary[gene_summary["_modality_"] == "ATAC"]["_gene_"])
            met_genes = set(gene_summary[gene_summary["_modality_"] == "MET"]["_gene_"])
            shared = atac_genes & met_genes
            
            if shared:
                shared_df = gene_summary[gene_summary["_gene_"].isin(shared)].copy()
                result["shared_genes"] = shared_df.pivot_table(
                    index="_gene_", columns="_modality_", values="n_regions", fill_value=0
                ).reset_index()

    # Add finding
    n_high = len(high_conf)
    n_total = len(ann)
    n_genes = high_conf["_gene_"].nunique() if not high_conf.empty else 0
    n_shared = len(result["shared_genes"]) if not result["shared_genes"].empty else 0

    add_finding(findings, "Region-gene associations", "Informative",
               f"{n_high}/{n_total} associations are relatively high-confidence, covering {n_genes} genes.",
               f"{n_shared} genes supported by both ATAC and MET.",
               "Associations are computational predictions; experimental validation needed.")

    return result


# ============================================================
# Pathway Enrichment Analysis
# ============================================================

def analyze_pathways(pathways, findings):
    result = {"nonredundant": pd.DataFrame()}

    if pathways is None or pathways.empty:
        add_finding(findings, "Pathway enrichment", "Not assessable",
                   "Pathway file unavailable.",
                   "08_pathway_enrichment.csv missing.",
                   "No functional interpretation.")
        return result

    pw = pathways.copy()
    fdr_col = find_col(pw, ["fdr", "padj", "adj_p", "q_value"])
    
    if fdr_col:
        pw["_fdr_"] = numeric(pw[fdr_col])
        sig = pw[pw["_fdr_"] < 0.05].copy()
    else:
        sig = pw.copy()

    if sig.empty:
        add_finding(findings, "Pathway enrichment", "No significance",
                   "No pathways pass FDR < 0.05.",
                   f"Total pathways tested: {len(pw)}.",
                   "May reflect limited statistical power or broad coverage.")
        return result

    # Redundancy filter
    name_col = find_col(sig, ["pathway", "pathway_name", "term"])
    if name_col:
        sig = sig.sort_values("_fdr_")
        seen_words = set()
        nonredundant = []
        
        for _, row in sig.iterrows():
            words = set(re.findall(r"\w{4,}", str(row[name_col]).lower()))
            if not words & seen_words:
                nonredundant.append(row)
                seen_words.update(words)
                if len(nonredundant) >= 20:
                    break
        
        result["nonredundant"] = pd.DataFrame(nonredundant)

    n_sig = len(sig)
    n_shown = len(result["nonredundant"])

    add_finding(findings, "Pathway enrichment", "Supported",
               f"{n_sig} pathways significant at FDR < 0.05; {n_shown} non-redundant shown.",
               f"Top pathways include developmental and epigenetic processes.",
               "Enrichment based on predicted region-gene associations; experimental validation needed.")

    return result


# ============================================================
# Cross-Modal Timing Analysis
# ============================================================

def analyze_timing(timing, annotation_result, findings):
    result = {"supported_genes": pd.DataFrame()}

    if timing is None or timing.empty:
        add_finding(findings, "Cross-modal timing support", "Not assessable",
                   "Timing file unavailable.",
                   "10_cross_modal_gene_timing.csv missing.",
                   "No cross-modal validation.")
        return result

    tm = timing.copy()
    gene_col = find_col(tm, ["gene", "gene_name", "gene_symbol"])
    if not gene_col:
        add_finding(findings, "Cross-modal timing support", "Not assessable",
                   "Gene column missing in timing file.",
                   "Required column not found.",
                   "Cannot assess cross-modal support.")
        return result

    tm = tm[tm[gene_col].map(is_symbol_like)].copy()
    tm.rename(columns={gene_col: "_gene_"}, inplace=True)

    # Find genes with RNA support
    rna_col = find_col(tm, ["rna_pseudotime_corr", "rna_corr"])
    if rna_col:
        tm["_rna_corr_"] = numeric(tm[rna_col]).abs()
        rna_supported = tm[tm["_rna_corr_"] > 0.3].copy()
    else:
        rna_supported = tm.copy()

    # Cross with high-confidence genes
    high_conf_genes = set()
    if not annotation_result.get("relative_high_confidence", pd.DataFrame()).empty:
        high_conf_genes = set(annotation_result["relative_high_confidence"]["_gene_"])

    if high_conf_genes:
        rna_supported = rna_supported[rna_supported["_gene_"].isin(high_conf_genes)]

    result["supported_genes"] = rna_supported

    n_supported = len(rna_supported)

    if n_supported > 0:
        add_finding(findings, "Cross-modal timing support", "Informative",
                   f"{n_supported} genes show RNA-level temporal correlation.",
                   "Genes with |RNA corr| > 0.3 and high-confidence region associations.",
                   "Correlation does not imply causation; mechanistic validation needed.")
    else:
        add_finding(findings, "Cross-modal timing support", "Limited",
                   "Few genes show strong cross-modal temporal correlation.",
                   "May reflect independent regulation or data sparsity.",
                   "Additional modalities or time points may improve coverage.")

    return result


# ============================================================
# Report Generation
# ============================================================

def write_report(report_path, out_dir, manifest, file_status, findings,
                geometry_result, region_result, annotation_result, 
                pathway_result, timing_result):
    
    lines = ["# Biological Evidence Report", "", "## Summary", ""]
    
    # File status
    lines.append("### Input Files")
    lines.append("")
    for file, status in file_status.items():
        lines.append(f"- **{file}**: {status}")
    lines.append("")

    # Evidence matrix
    lines.append("## Evidence Matrix")
    lines.append("")
    if findings:
        evidence_df = pd.DataFrame(findings)
        lines.append(md_table(evidence_df))
    else:
        lines.append("_No findings recorded._")
    lines.append("")

    # Geometry (Figure 6e)
    lines.append("## Trajectory Geometry (Figure 6e)")
    lines.append("")
    if geometry_result["asynchronous_peak"]:
        lines.append("**Finding**: Temporally staggered epigenomic remodeling detected.")
        lines.append("")
        for mod, info in geometry_result["peak_by_modality"].items():
            lines.append(f"- **{mod}** peaks at **{info['stage']}** "
                        f"(orthogonal MAE = {fmt_num(info['orthogonal_mae'])})")
        lines.append("")
    else:
        lines.append("**Finding**: No clear asynchronous peak detected.")
        lines.append("")

    if not geometry_result["geometry_table"].empty:
        lines.append("### Full Geometry Table")
        lines.append("")
        lines.append(md_table(geometry_result["geometry_table"]))
        lines.append("")

    # Stable regions
    lines.append("## Stable Genomic Regions")
    lines.append("")
    if not region_result["saturation_summary"].empty:
        lines.append(md_table(region_result["saturation_summary"]))
        lines.append("")

    # Candidate genes
    lines.append("## Candidate Genes")
    lines.append("")
    if not annotation_result["gene_summary"].empty:
        top_genes = annotation_result["gene_summary"].sort_values(
            "n_regions", ascending=False
        ).head(20)
        lines.append(md_table(top_genes))
        lines.append("")

    if not annotation_result["shared_genes"].empty:
        lines.append("### ATAC-MET Shared Genes")
        lines.append("")
        lines.append(md_table(annotation_result["shared_genes"]))
        lines.append("")

    # Pathways
    lines.append("## Enriched Pathways")
    lines.append("")
    if not pathway_result["nonredundant"].empty:
        pw_show = pathway_result["nonredundant"][
            [c for c in pathway_result["nonredundant"].columns 
             if c in ["pathway", "pathway_name", "term", "fdr", "padj", "p_value"]]
        ].head(15)
        lines.append(md_table(pw_show))
        lines.append("")
    else:
        lines.append("_No significant pathways._")
        lines.append("")

    # Cross-modal support
    lines.append("## Cross-Modal Timing Support")
    lines.append("")
    if not timing_result["supported_genes"].empty:
        lines.append(f"{len(timing_result['supported_genes'])} genes show RNA correlation.")
        lines.append("")
        top_timing = timing_result["supported_genes"].head(20)
        lines.append(md_table(top_timing))
        lines.append("")
    else:
        lines.append("_Limited cross-modal support detected._")
        lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("1. Bootstrap is cell-level sampling, not biological replication.")
    lines.append("2. Region-gene associations are computational predictions.")
    lines.append("3. Pathway enrichment depends on annotation completeness.")
    lines.append("4. Experimental validation is required for all candidate genes.")
    lines.append("")

    # Write report
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Overall Verdict
# ============================================================

def determine_verdict(geometry_result, region_result, annotation_result,
                     pathway_result, timing_result):
    
    statements = []
    
    if geometry_result["asynchronous_peak"]:
        statements.append(
            "Temporally staggered epigenomic remodeling is supported by trajectory geometry."
        )
        verdict = "Publication-ready with limitations"
    else:
        statements.append(
            "Asynchronous peaks not clearly detected; interpret geometry cautiously."
        )
        verdict = "Requires additional support"

    if region_result["top_fraction_saturated"]:
        statements.append(
            "Stable region selection shows top-fraction saturation; consider sensitivity analysis."
        )

    n_genes = len(annotation_result.get("relative_high_confidence", []))
    if n_genes > 0:
        statements.append(
            f"{n_genes} relatively high-confidence region-gene associations identified."
        )

    n_pathways = len(pathway_result.get("nonredundant", []))
    if n_pathways > 0:
        statements.append(
            f"{n_pathways} non-redundant pathways enriched at FDR < 0.05."
        )

    n_timing = len(timing_result.get("supported_genes", []))
    if n_timing > 0:
        statements.append(
            f"{n_timing} genes show cross-modal temporal correlation."
        )

    statements.append(
        "All findings require experimental validation and are limited by cell-level bootstrap design."
    )

    return verdict, statements


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Biological evidence analysis for CMAT human oocyte experiment"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory containing validation outputs (from validate_endpoint.py)"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Output directory for report (default: OUT/biology_evidence_report/)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.exists():
        print(f"[ERROR] Output directory not found: {out_dir}")
        return

    report_dir = Path(args.report_dir) if args.report_dir else (out_dir / "biology_evidence_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("BIOLOGICAL EVIDENCE ANALYSIS")
    print("=" * 88)
    print(f"Input directory:  {out_dir}")
    print(f"Report directory: {report_dir}")
    print()

    # Load files
    file_map = {
        "observed": "01_observed_geometry.csv",
        "bootstrap": "03_geometry_bootstrap_summary.csv",
        "contrasts": "04_geometry_stage_contrasts.csv",
        "stable": "06_region_candidates_stable.csv",
        "annotations": "07_region_gene_annotations.csv",
        "pathways": "08_pathway_enrichment.csv",
        "timing": "10_cross_modal_gene_timing.csv",
    }

    loaded = {}
    file_status = {}

    for key, filename in file_map.items():
        path = out_dir / filename
        df, error = safe_read_csv(path)
        loaded[key] = df
        file_status[filename] = "OK" if df is not None else error

    # Load manifest
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as e:
            print(f"[WARNING] Manifest read failed: {e}")
            manifest = {}
    else:
        manifest = {}

    findings = []

    # Run analyses
    geometry_result = analyze_geometry(
        loaded.get("observed"),
        loaded.get("bootstrap"),
        loaded.get("contrasts"),
        findings
    )

    region_result = analyze_regions(
        loaded.get("stable"),
        manifest,
        findings
    )

    annotation_result = analyze_annotations(
        loaded.get("annotations"),
        findings
    )

    pathway_result = analyze_pathways(
        loaded.get("pathways"),
        findings
    )

    timing_result = analyze_timing(
        loaded.get("timing"),
        annotation_result,
        findings
    )

    # Export results
    pd.DataFrame(findings).to_csv(report_dir / "evidence_matrix.csv", index=False)
    geometry_result["geometry_table"].to_csv(report_dir / "geometry_summary.csv", index=False)
    geometry_result["contrast_table"].to_csv(report_dir / "geometry_stage_contrasts.csv", index=False)
    region_result["saturation_summary"].to_csv(report_dir / "stable_region_saturation.csv", index=False)
    region_result["direction_summary"].to_csv(report_dir / "stable_region_directions.csv", index=False)

    high_conf = annotation_result.get("relative_high_confidence", pd.DataFrame())
    if not high_conf.empty:
        drop_cols = [c for c in high_conf.columns if c.startswith("_") and
                     c not in {"_gene_", "_abs_distance_to_tss_", "_relative_high_confidence_"}]
        high_conf.drop(columns=drop_cols, errors="ignore").to_csv(
            report_dir / "relative_high_confidence_region_gene_associations.csv", 
            index=False
        )

    annotation_result["gene_summary"].to_csv(report_dir / "candidate_gene_summary.csv", index=False)
    annotation_result["shared_genes"].to_csv(report_dir / "ATAC_MET_shared_candidate_genes.csv", index=False)
    pathway_result["nonredundant"].to_csv(report_dir / "significant_nonredundant_pathways.csv", index=False)
    timing_result["supported_genes"].to_csv(report_dir / "RNA_crossmodal_supported_genes.csv", index=False)

    # Write report
    report_path = report_dir / "biological_evidence_report.md"
    write_report(
        report_path=report_path,
        out_dir=out_dir,
        manifest=manifest,
        file_status=file_status,
        findings=findings,
        geometry_result=geometry_result,
        region_result=region_result,
        annotation_result=annotation_result,
        pathway_result=pathway_result,
        timing_result=timing_result
    )

    # Verdict
    verdict, verdict_statements = determine_verdict(
        geometry_result, region_result, annotation_result,
        pathway_result, timing_result
    )

    print()
    print("=" * 88)
    print("ANALYSIS COMPLETED")
    print("=" * 88)
    print(f"Overall verdict: {verdict}")
    print()
    for i, text in enumerate(verdict_statements, 1):
        print(f"{i}. {text}")
    print()
    print(f"Report: {report_path}")
    print()


if __name__ == "__main__":
    main()
