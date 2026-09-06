#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.cmat_attention import (
    CMATAttentionEncoder,
)

from models.losses import (
    lineage_alignment_loss,
)

from preprocessing.dataset import (
    PairedMultimodalDataset,
)

from preprocessing.preprocess import (
    MultimodalPreprocessor,
)


class HumanFGOMIIModel(nn.Module):

    def __init__(
        self,
        modality_dims,
        hidden_dim=256,
        embedding_dim=64,
        n_heads=4,
    ):
        super().__init__()

        self.fgo_encoder = (
            CMATAttentionEncoder(
                modality_dims=(
                    modality_dims
                ),
                hidden_dim=hidden_dim,
                embedding_dim=(
                    embedding_dim
                ),
                n_heads=n_heads,
            )
        )

        self.mii_encoder = (
            CMATAttentionEncoder(
                modality_dims=(
                    modality_dims
                ),
                hidden_dim=hidden_dim,
                embedding_dim=(
                    embedding_dim
                ),
                n_heads=n_heads,
            )
        )

    def forward(
        self,
        batch,
        return_attention=False,
    ):

        fgo_output = (
            self.fgo_encoder(
                batch["source"],
                return_attention=(
                    return_attention
                ),
            )
        )

        mii_output = (
            self.mii_encoder(
                batch["target"],
                return_attention=(
                    return_attention
                ),
            )
        )

        if return_attention:

            return (
                fgo_output[0],
                mii_output[0],
                fgo_output[1],
            )

        return (
            fgo_output,
            mii_output,
        )


class HumanFGOMIITrainer:

    def __init__(
        self,
        model,
        lineage_matrix,
        train_loader,
        device="cuda",
        lr=5e-4,
        temperature=0.15,
        positive_threshold=0.25,
    ):

        self.model = model.to(
            device
        )

        self.lineage_matrix = (
            lineage_matrix
        )

        self.train_loader = (
            train_loader
        )

        self.device = device

        self.temperature = (
            temperature
        )

        self.positive_threshold = (
            positive_threshold
        )

        self.optimizer = (
            torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=1e-5,
            )
        )

        self.scheduler = (
            torch.optim.lr_scheduler
            .CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=1e-6,
            )
        )

    @staticmethod
    def _modalities_to_device(
        modalities,
        device,
    ):

        return {
            modality_name:
                tensor.to(device)
            for modality_name, tensor
            in modalities.items()
        }

    def train(
        self,
        num_epochs=50,
    ):

        for epoch in range(
            1,
            num_epochs + 1,
        ):

            self.model.train()

            epoch_loss = 0.0

            for batch in (
                self.train_loader
            ):

                batch["source"] = (
                    self._modalities_to_device(
                        batch["source"],
                        self.device,
                    )
                )

                batch["target"] = (
                    self._modalities_to_device(
                        batch["target"],
                        self.device,
                    )
                )

                batch[
                    "source_labels"
                ] = (
                    batch[
                        "source_labels"
                    ].to(
                        self.device
                    )
                )

                batch[
                    "target_labels"
                ] = (
                    batch[
                        "target_labels"
                    ].to(
                        self.device
                    )
                )

                (
                    fgo_embedding,
                    mii_embedding,
                    _,
                ) = self.model(
                    batch,
                    return_attention=True,
                )

                loss = (
                    lineage_alignment_loss(
                        source_emb=(
                            fgo_embedding
                        ),
                        target_emb=(
                            mii_embedding
                        ),
                        source_labels=(
                            batch[
                                "source_labels"
                            ]
                        ),
                        target_labels=(
                            batch[
                                "target_labels"
                            ]
                        ),
                        correspondence_matrix=(
                            self.lineage_matrix
                        ),
                        temperature=(
                            self.temperature
                        ),
                        positive_threshold=(
                            self.positive_threshold
                        ),
                    )
                )

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                )

                self.optimizer.step()

                epoch_loss += (
                    loss.item()
                )

            average_loss = (
                epoch_loss
                / len(
                    self.train_loader
                )
            )

            self.scheduler.step()

            if (
                epoch == 1
                or epoch % 10 == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch:3d}/"
                    f"{num_epochs} | "
                    f"Loss: "
                    f"{average_loss:.4f}"
                )


def _to_dense_array(
    matrix,
):

    if hasattr(
        matrix,
        "toarray",
    ):
        return matrix.toarray()

    return np.asarray(
        matrix
    )


def _make_contiguous_labels(
    labels,
):

    unique_labels = np.unique(
        labels
    )

    label_map = {
        old_label: new_label
        for new_label, old_label
        in enumerate(
            unique_labels
        )
    }

    mapped_labels = np.array(
        [
            label_map[label]
            for label in labels
        ],
        dtype=int,
    )

    return (
        mapped_labels,
        unique_labels,
    )


def load_sccharm_data(
    trajectory_h5ad,
):

    adata = sc.read_h5ad(
        trajectory_h5ad
    )

    if (
        "subcluster"
        not in adata.obs.columns
    ):

        print(
            "subcluster is missing; "
            "running Leiden clustering."
        )

        sc.pp.pca(
            adata,
            n_comps=30,
        )

        sc.pp.neighbors(
            adata,
            n_neighbors=10,
        )

        sc.tl.leiden(
            adata,
            resolution=0.5,
            key_added="subcluster",
        )

    required_columns = {
        "stage",
        "subcluster",
        "monocle3_pseudotime",
    }

    missing_columns = (
        required_columns
        - set(
            adata.obs.columns
        )
    )

    if missing_columns:

        raise KeyError(
            "Missing required "
            "adata.obs columns: "
            f"{sorted(missing_columns)}"
        )

    fgo_mask = (
        adata.obs["stage"]
        == "FGO"
    )

    mii_mask = (
        adata.obs["stage"]
        == "MII"
    )

    fgo_adata = (
        adata[
            fgo_mask
        ].copy()
    )

    mii_adata = (
        adata[
            mii_mask
        ].copy()
    )

    if fgo_adata.n_obs == 0:
        raise ValueError(
            "No FGO cells were found."
        )

    if mii_adata.n_obs == 0:
        raise ValueError(
            "No MII cells were found."
        )

    original_fgo_labels = (
        fgo_adata.obs[
            "subcluster"
        ].values
    )

    original_mii_labels = (
        mii_adata.obs[
            "subcluster"
        ].values
    )

    (
        fgo_labels,
        fgo_states,
    ) = _make_contiguous_labels(
        original_fgo_labels
    )

    (
        mii_labels,
        mii_states,
    ) = _make_contiguous_labels(
        original_mii_labels
    )

    n_fgo = len(
        fgo_states
    )

    n_mii = len(
        mii_states
    )

    lineage_matrix = np.zeros(
        (
            n_fgo,
            n_mii,
        ),
        dtype=float,
    )

    fgo_pseudotime_means = []

    for state in fgo_states:

        state_mask = (
            fgo_adata.obs[
                "subcluster"
            ].values
            == state
        )

        mean_pseudotime = (
            fgo_adata.obs.loc[
                state_mask,
                "monocle3_pseudotime",
            ].mean()
        )

        fgo_pseudotime_means.append(
            mean_pseudotime
        )

    mii_pseudotime_means = []

    for state in mii_states:

        state_mask = (
            mii_adata.obs[
                "subcluster"
            ].values
            == state
        )

        mean_pseudotime = (
            mii_adata.obs.loc[
                state_mask,
                "monocle3_pseudotime",
            ].mean()
        )

        mii_pseudotime_means.append(
            mean_pseudotime
        )

    for i in range(
        n_fgo
    ):

        for j in range(
            n_mii
        ):

            pseudotime_distance = abs(
                fgo_pseudotime_means[i]
                - mii_pseudotime_means[j]
            )

            lineage_matrix[
                i,
                j,
            ] = max(
                0.1,
                0.85
                - pseudotime_distance
                * 0.8,
            )

    def get_modality(
        adata_part,
        key,
        fallback_dimension,
    ):

        if key in adata_part.obsm:

            return _to_dense_array(
                adata_part.obsm[
                    key
                ]
            )

        expression_matrix = (
            _to_dense_array(
                adata_part.X
            )
        )

        return expression_matrix[
            :,
            :fallback_dimension,
        ]

    return {
        "fgo": {
            "rna": get_modality(
                fgo_adata,
                "rna",
                2000,
            ),
            "atac": get_modality(
                fgo_adata,
                "atac",
                5000,
            ),
            "met": get_modality(
                fgo_adata,
                "met",
                1000,
            ),
        },

        "mii": {
            "rna": get_modality(
                mii_adata,
                "rna",
                2000,
            ),
            "atac": get_modality(
                mii_adata,
                "atac",
                5000,
            ),
            "met": get_modality(
                mii_adata,
                "met",
                1000,
            ),
        },

        "fgo_labels":
            fgo_labels,

        "mii_labels":
            mii_labels,

        "lineage_matrix":
            lineage_matrix,
    }


def parse_arguments():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Human FGO-to-MII "
                "CMAT-Attn validation"
            )
        )
    )

    parser.add_argument(
        "--trajectory-h5ad",
        required=True,
        help=(
            "Path to the Monocle3 "
            "FGO/MII h5ad file."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "human_FGO_MII"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = (
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(f"Device: {device}")

    data = load_sccharm_data(
        args.trajectory_h5ad
    )

    preprocessor = (
        MultimodalPreprocessor(
            pca_dims={
                "rna": 50,
                "atac": 100,
                "met": 50,
            },
            random_state=42,
        )
    )

    fgo_processed = (
        preprocessor.fit_transform(
            data["fgo"]
        )
    )

    mii_processed = (
        preprocessor.transform(
            data["mii"]
        )
    )

    dataset = (
        PairedMultimodalDataset(
            source_data=(
                fgo_processed
            ),
            target_data=(
                mii_processed
            ),
            source_labels=(
                data[
                    "fgo_labels"
                ]
            ),
            target_labels=(
                data[
                    "mii_labels"
                ]
            ),
            correspondence_matrix=(
                data[
                    "lineage_matrix"
                ]
            ),
            samples_per_epoch=1000,
            cells_per_sample=24,
            positive_threshold=0.25,
            positive_sampling_probability=0.8,
            min_positive_fraction=0.4,
        )
    )

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=(
            lambda samples:
                samples[0]
        ),
    )

    modality_dims = {
        modality_name:
            modality_values.shape[1]
        for (
            modality_name,
            modality_values,
        ) in fgo_processed.items()
    }

    model = HumanFGOMIIModel(
        modality_dims=(
            modality_dims
        ),
        hidden_dim=256,
        embedding_dim=64,
        n_heads=4,
    )

    trainer = HumanFGOMIITrainer(
        model=model,
        lineage_matrix=(
            data[
                "lineage_matrix"
            ]
        ),
        train_loader=(
            train_loader
        ),
        device=device,
        lr=5e-4,
        temperature=0.15,
        positive_threshold=0.25,
    )

    trainer.train(
        num_epochs=(
            args.epochs
        )
    )

    output_directory = Path(
        args.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    with torch.no_grad():

        fgo_tensors = {
            modality_name:
                torch.as_tensor(
                    modality_values,
                    dtype=torch.float32,
                    device=device,
                )
            for (
                modality_name,
                modality_values,
            ) in fgo_processed.items()
        }

        mii_tensors = {
            modality_name:
                torch.as_tensor(
                    modality_values,
                    dtype=torch.float32,
                    device=device,
                )
            for (
                modality_name,
                modality_values,
            ) in mii_processed.items()
        }

        fgo_embedding = (
            model.fgo_encoder(
                fgo_tensors,
                return_attention=False,
            )
        )

        mii_embedding = (
            model.mii_encoder(
                mii_tensors,
                return_attention=False,
            )
        )

    # ? 修复：使用正确的文件名
    fgo_path = (
        output_directory
        / "fgo_embedding.npy"
    )

    mii_path = (
        output_directory
        / "mii_embedding.npy"
    )

    np.save(
        fgo_path,
        fgo_embedding
        .cpu()
        .numpy(),
    )

    np.save(
        mii_path,
        mii_embedding
        .cpu()
        .numpy(),
    )

    print(
        "Saved:",
        fgo_path,
        fgo_embedding.shape,
    )

    print(
        "Saved:",
        mii_path,
        mii_embedding.shape,
    )


if __name__ == "__main__":
    main()