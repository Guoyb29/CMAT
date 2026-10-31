# CMAT: Cross-Modal Attention Framework for Multimodal Developmental Trajectory Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **CMAT** for reconstructing continuous molecular-state trajectories from discrete single-cell observations across incompatible measurement platforms.

> **Yongbo Guo¹, Yang Gao¹, Fangyuan Shi²,³**  
> ¹ School of Information Engineering, Ningxia University, Yinchuan 750021, China  
> ² School of Medical Technology, North Minzu University, Yinchuan 750021, China  
> ³ Key Laboratory of Fertility Preservation and Maintenance of Ministry of Education, Ningxia Medical University  
> *Correspondence: shifangyuan@nun.edu.cn*

---

## Overview

CMAT addresses the challenge of **reconstructing developmental trajectories when different stages are measured on incompatible platforms** (e.g., scNMT-seq at early stage, spatial transcriptomics at late stage). Traditional methods fail when molecular modalities differ between endpoints.

### Key Innovation

1. **Cross-modal attention** - Adaptively integrates RNA, chromatin accessibility, and DNA methylation based on cell state
2. **Developmentally-informed alignment** - Learns source-target cell correspondence under lineage constraints
3. **Endpoint-conditioned generation** - Reconstructs continuous molecular states between matched cell pairs

### Results

**Mouse Embryogenesis (E7.5 → E12.5)**
- E9.5 validation: Pearson correlation **0.85** for held-out intermediate stage
- Marker directionality: **89%** recovery (vs. 68% baseline)

**Human Oocyte Maturation (FGO → MII)**
- Temporal alignment: **13.2%** error for GO1/GO2/MI validation
- Revealed **modality-specific epigenomic remodeling**:
  - DNA methylation peaks at GO1 (38% deviation)
  - Chromatin accessibility peaks at GO2 (42% deviation)
  - RNA expression remains stable (<15%)

---

## Installation

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/CMAT.git
cd CMAT

# Create conda environment
conda env create -f environment.yml
conda activate cmat

# Install package
pip install -e .
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA ≥ 11.0 (recommended)

---

## Quick Start

### Human Oocyte Maturation (FGO → MII)

```bash
# Download data (see Data Preparation below)
mkdir -p data/human
# Place scChaRM_FGO_MII.h5ad, GO1_cells.h5ad, GO2_cells.h5ad, MI_cells.h5ad in data/human/

# Run complete pipeline
python run_human_experiment.py

# Outputs:
# - results/human/alignment/          # Trained model & embeddings
# - results/human/trajectories/       # FGO→MII trajectories
# - results/human/validation/         # GO1/GO2/MI validation
# - results/human/validation/biology_report/  # Figure 6e analysis
```

### Mouse Embryo Development (E7.5 → E12.5)

```bash
# Download data
bash scripts/download_mouse_data.sh

# Run complete pipeline
python run_mouse_experiment.py

# Outputs:
# - results/mouse/trajectories/  # 4 developmental paths (Epiblast→Brain, etc.)
# - results/mouse/validation/    # E9.5 intermediate validation
```

---

## Data Preparation

### Human Oocyte Maturation

**Dataset**: scChaRM-seq (Yan et al., *Cell Stem Cell*, 2021)
**GEO**: [GSE154762](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154762)

```bash
mkdir -p data/human
# Download and convert to h5ad format:
# - scChaRM_FGO_MII.h5ad (training: FGO + MII cells)
# - GO1_cells.h5ad, GO2_cells.h5ad, MI_cells.h5ad (validation)
```

### Mouse Embryo Development

**Datasets**:
- E7.5 trimodal: [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) (Argelaguet et al., *Nature* 2019)
- E9.5 validation: [GSE119945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119945) (Cao et al., *Nature* 2019)
- E12.5 spatial: [STDS0000058](https://db.cngb.org/stomics/datasets/STDS0000058) (Chen et al., *Cell* 2022)

```bash
bash scripts/download_mouse_data.sh
# Or manually download and place in data/mouse/
```

---

## Project Structure

```
CMAT/
├── models/
│   ├── cmat.py                  # Cross-modal attention encoder
│   ├── trajectory_generator.py  # Gated residual generator
│   └── losses.py               # Contrastive + boundary + smoothness losses
├── experiments/
│   ├── human_FGO_MII/
│   │   ├── train_alignment.py          # Step 1: Learn embeddings
│   │   ├── generate_trajectory.py      # Step 2: Generate trajectories
│   │   ├── validate_endpoint.py        # Step 3: Validate GO1/GO2/MI
│   │   └── biology_evidence_analysis.py # Step 4: Figure 6e analysis
│   └── mouse_e75_e125/
│       ├── train_alignment.py
│       ├── generate_trajectory.py
│       └── validate_e95.py
├── evaluation/
│   ├── metrics.py              # Pearson, Spearman, cosine, etc.
│   └── temporal_alignment.py   # Temporal error & smoothness
├── preprocessing/
│   └── preprocess.py           # Data normalization & QC
├── configs/
│   ├── human.yaml              # Human experiment config
│   └── mouse.yaml              # Mouse experiment config
├── run_human_experiment.py     # One-click human pipeline
├── run_mouse_experiment.py     # One-click mouse pipeline
└── README.md
```

---

## Reproducing Paper Figures

### Main Results

| Figure | Description | Command |
|--------|-------------|----------|
| **Figure 2** | Mouse trajectories (4 lineages) | `python run_mouse_experiment.py` |
| **Figure 3** | E9.5 validation | Included in mouse pipeline output |
| **Figure 5** | Human FGO→MII trajectories | `python run_human_experiment.py` |
| **Figure 6e** | Modality-specific deviations | Included in human pipeline output |

### Detailed Execution

For step-by-step manual execution, see individual scripts in `experiments/`:

```bash
# Human experiment (manual steps)
python experiments/human_FGO_MII/train_alignment.py --trajectory-h5ad data/human/scChaRM_FGO_MII.h5ad --output-dir results/human/alignment
python experiments/human_FGO_MII/generate_trajectory.py --input-h5ad data/human/scChaRM_FGO_MII.h5ad --output-dir results/human/trajectories
python experiments/human_FGO_MII/validate_endpoint.py --trajectory results/human/trajectories/FGO_to_MII_trajectories.h5ad --output-dir results/human/validation
python experiments/human_FGO_MII/biology_evidence_analysis.py --out results/human/validation

# Mouse experiment (manual steps)
python experiments/mouse_e75_e125/train_alignment.py --e75-path data/mouse/E75_scNMT.h5ad --e125-path data/mouse/E125_spatial.h5ad --output-dir results/mouse/alignment
python experiments/mouse_e75_e125/generate_trajectory.py --embedding-dir results/mouse/alignment --output-dir results/mouse/trajectories
python experiments/mouse_e75_e125/validate_e95.py --trajectory results/mouse/trajectories/Epiblast_to_Brain_trajectory_s120.h5ad --output-dir results/mouse/validation
```

---

## Method Overview

**Stage 1: Multimodal Integration**
- Cross-modal attention fuses RNA/ATAC/Methylation → unified cell embeddings

**Stage 2: Developmentally-Informed Alignment**
- Contrastive learning with lineage constraints → source-target cell pairs

**Stage 3: Trajectory Generation**
- Gated residual generator with boundary conditions → continuous pseudotime states

**Validation**
- Feature-level: Pearson/Spearman correlation with held-out stages
- Latent-level: Euclidean distance, Wasserstein distance
- Temporal: Pseudotime alignment error, developmental ordering

For detailed methodology and pseudocode, see the paper.

---

## Citation

If you use CMAT in your research, please cite:

```bibtex
@article{guo2026cmat,
  title={CMAT: Cross-Modal Attention Framework for Multimodal Developmental Trajectory Inference},
  author={Guo, Yongbo and Gao, Yang and Shi, Fangyuan},
  journal={Nature Computational Science},
  year={2026},
  note={In submission}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Yongbo Guo**: guoyongbo@nxu.edu.cn (First author)
- **Fangyuan Shi**: shifangyuan@nun.edu.cn (Corresponding author)

**Issues**: Please report bugs or feature requests via [GitHub Issues](https://github.com/yourusername/CMAT/issues).

---

## Acknowledgments

We thank the authors of the following datasets for making their data publicly available:
- Argelaguet et al. (2019) - Mouse scNMT-seq data
- Cao et al. (2019) - Mouse sci-RNA-seq3 data  
- Chen et al. (2022) - Mouse Stereo-seq data
- Yan et al. (2021) - Human scChaRM-seq data
