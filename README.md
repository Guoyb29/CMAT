# CMAT: Cross-Modal Attention Framework for Multimodal Developmental Trajectory Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code associated with the CMAT manuscript.

**Authors:** Yong-Bo Guo¹, Yang Gao², Yu-Jian Kang³, Fang-Yuan Shi¹˒⁴˒⁵*

1. School of Information Engineering, Ningxia University, Yinchuan, Ningxia 750021, China.
2. School of Medical Technology, North Minzu University, Yinchuan, Ningxia 750021, China.
3. Chongqing Key Laboratory of Intelligent Oncology for Breast Cancer, Cancer Hospital, School of Medicine, Chongqing University, Chongqing 400030, China.
4. Collaborative Innovation Center for Ningxia Big Data and Artificial Intelligence Co-founded by Ningxia Municipality and Ministry of Education, Ningxia University, Yinchuan, Ningxia 750021, China.
5. Ningxia Key Laboratory of Artificial Intelligence and Information Security for Channeling Computing Resources from the East to the West, Ningxia University, Yinchuan, Ningxia 750021, China.

**Correspondence:** Fang-Yuan Shi, [shify@nxu.edu.cn](mailto:shify@nxu.edu.cn).

## Overview

CMAT reconstructs molecular-state trajectories between sparsely sampled developmental stages with partially shared measurement modalities. The framework described in the manuscript combines modality-specific encoders, adaptive cross-modal attention, developmentally informed endpoint correspondence, and endpoint-conditioned residual generation.

The manuscript evaluates mouse embryogenesis and human oocyte maturation. The principal mouse analysis links trimodal E7.5 cells to an E12.5 spatial RNA reference and evaluates reconstructed RNA states against independently observed E9.5 populations. Human analyses distinguish endpoint-only RNA reconstruction from stage-calibrated epigenomic reconstruction, including held-out-cell and leave-one-stage-out evaluations.

In the scNMT-seq and scChaRM-seq datasets, accessibility is inferred from GpC methyltransferase labeling. The label `ATAC` in the scripts denotes the accessibility modality.

## Source code

```bash
git clone https://github.com/Guoyb29/CMAT.git
cd CMAT
git rev-parse HEAD
```

Source files can also be downloaded using **Code > Download ZIP**. Record the commit identifier used for each analysis.

## Software environment

The manuscript reports the following environment for the principal Python analyses:

| Software | Version |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.9.1 |
| CUDA | 12.8 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| Scanpy | 1.11.5 |
| AnnData | 0.11.4 |
| statsmodels | 0.14.5 |

Plotting and configuration utilities additionally use matplotlib, seaborn, umap-learn, and PyYAML. Separate environments for optimal-transport and Monocle 3 analyses are described in the supplementary methods.

## Data sources

All experimental datasets used in the study were previously published.

| Analysis | Dataset source | Accession |
| --- | --- | --- |
| Mouse E7.5 multimodal endpoint | Argelaguet et al., Nature, 2019 | [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) |
| Mouse E12.5 spatial endpoint | Chen et al., Cell, 2022 | STOmicsDB: `STDS0000058`; `CNP0001543` |
| Mouse E9.5 reference and E9.5–E13.5 analysis | Cao et al., Nature, 2019 | [GSE119945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119945) |
| Mouse E6.5–E8.5 analysis | Pijuan-Sala et al., Nature, 2019 | ArrayExpress: `E-MTAB-6967` |
| Human oocyte maturation, scChaRM-seq | Yan et al., Cell Stem Cell, 2021 | [GSE154762](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154762) |

The human scChaRM-seq reference is:

Yan, R. et al. **Decoding dynamic epigenetic landscapes in human oocytes using single-cell multi-omics sequencing.** Cell Stem Cell 28, 1641–1656.e7 (2021).

## Data preparation

The experiment scripts use preprocessed `.h5ad` files. Preparation includes stage and cell-type annotation, alignment of RNA features by gene identity, and modality-specific processing of accessibility and methylation features. The manuscript uses 100-kb genomic regions for the human epigenomic analyses.

Cell identifiers and feature ordering must remain consistent across expression matrices, embeddings, and matched endpoint pairs. Preprocessing and calibration must use only observations permitted by each training and validation design.

The original data repositories provide the published datasets; analysis-ready files require the preprocessing described in the manuscript and supplementary methods.

## Repository layout

| Path | Contents or intended role |
| --- | --- |
| `models/` | Modality encoders, cross-modal attention, spatial encoder, trajectory generator, and losses |
| `preprocessing/` | Preprocessing utilities and paired multimodal sampling |
| `experiments/human_FGO_MII/train.py` | Human endpoint representation training |
| `experiments/human_FGO_MII/generate_trajectories.py` | Human RNA trajectory generation |
| `experiments/human_FGO_MII/validate_endpoint.py` | GO1, GO2, and MI intermediate-stage evaluation |
| `experiments/human_FGO_MII/biology_evidence_analysis.py` | Report generation from precomputed biological-analysis tables |
| `experiments/mouse_development/train.py` | Mouse endpoint representation training |
| `experiments/mouse_development/generate_trajectories.py` | Mouse lineage-restricted trajectory generation |
| `experiments/mouse_development/validate_e95.py` | E9.5 intermediate-stage evaluation |
| `evaluation/` | Molecular and temporal evaluation utilities |
| `visualization/` | Embedding and trajectory plotting utilities |
| `configs/` | Human and mouse configuration files |
| `run_human_experiment.py`, `run_mouse_experiment.py` | Top-level workflow scripts |
| `download_mouse_data.sh` | Mouse data download helper |

## Execution and release scope

The current public snapshot requires corrections to workflow integration and evaluation before end-to-end reproduction of the manuscript. Run instructions should be tied to a tested code version, its processed inputs, and the corresponding configurations.

The archived `v1.0.0` code snapshot is available at [https://doi.org/10.5281/zenodo.22546359](https://doi.org/10.5281/zenodo.22546359). This archive contains source code, configurations, and documentation; it does not contain separate processed matrices, model checkpoints, or complete manuscript result tables.

GitHub updates do not automatically replace this version-specific archive. Identify the code commit and archive version used when reporting an analysis.

## Citation

The manuscript is unpublished.

```bibtex
@unpublished{guo2026cmat,
  title={CMAT: Cross-Modal Attention Framework for Multimodal Developmental Trajectory Inference},
  author={Guo, Yong-Bo and Gao, Yang and Kang, Yu-Jian and Shi, Fang-Yuan},
  year={2026},
  note={Unpublished manuscript}
}
```

The Zenodo DOI identifies the archived code snapshot.

## License

Source code is distributed under the [MIT License](LICENSE). Refer to the original dataset records for data access and reuse conditions.

## Contact

- **Corresponding author:** Fang-Yuan Shi, [shify@nxu.edu.cn](mailto:shify@nxu.edu.cn)
- **First author:** Yong-Bo Guo, [12025130913@stu.nxu.edu.cn](mailto:12025130913@stu.nxu.edu.cn)
- **Code issues:** [GitHub Issues](https://github.com/Guoyb29/CMAT/issues)

## Acknowledgments

We thank the researchers who generated and publicly shared the datasets used in this study, including Argelaguet et al. (2019), Cao et al. (2019), Chen et al. (2022), Pijuan-Sala et al. (2019), and Yan et al. (2021).
