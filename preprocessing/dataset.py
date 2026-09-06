#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Mapping

import numpy as np
import torch

from torch.utils.data import Dataset


class PairedMultimodalDataset(Dataset):

    def __init__(
        self,
        source_data: Mapping[
            str,
            np.ndarray,
        ],
        target_data: Mapping[
            str,
            np.ndarray,
        ],
        source_labels,
        target_labels,
        correspondence_matrix,
        samples_per_epoch: int = 1000,
        cells_per_sample: int = 24,
        positive_threshold: float = 0.25,
        positive_sampling_probability: float = 0.8,
        min_positive_fraction: float = 0.4,
    ):

        self.source_data = {
            name: np.asarray(values)
            for name, values
            in source_data.items()
        }

        self.target_data = {
            name: np.asarray(values)
            for name, values
            in target_data.items()
        }

        self.source_labels = np.asarray(
            source_labels
        )

        self.target_labels = np.asarray(
            target_labels
        )

        self.correspondence_matrix = (
            np.asarray(
                correspondence_matrix
            )
        )

        self.samples_per_epoch = (
            samples_per_epoch
        )

        self.cells_per_sample = (
            cells_per_sample
        )

        self.positive_threshold = (
            positive_threshold
        )

        self.positive_sampling_probability = (
            positive_sampling_probability
        )

        self.min_positive_fraction = (
            min_positive_fraction
        )

        self.n_source = len(
            np.unique(
                self.source_labels
            )
        )

        self.n_target = len(
            np.unique(
                self.target_labels
            )
        )

        expected_shape = (
            self.n_source,
            self.n_target,
        )

        if (
            self.correspondence_matrix.shape
            != expected_shape
        ):
            raise ValueError(
                "correspondence_matrix must "
                f"have shape {expected_shape}; "
                "received "
                f"{self.correspondence_matrix.shape}."
            )

        n_source_cells = len(
            self.source_labels
        )

        n_target_cells = len(
            self.target_labels
        )

        for (
            modality_name,
            values,
        ) in self.source_data.items():

            if (
                values.shape[0]
                != n_source_cells
            ):
                raise ValueError(
                    "Source modality "
                    f"{modality_name} contains "
                    f"{values.shape[0]} cells, "
                    "but source_labels contains "
                    f"{n_source_cells}."
                )

        for (
            modality_name,
            values,
        ) in self.target_data.items():

            if (
                values.shape[0]
                != n_target_cells
            ):
                raise ValueError(
                    "Target modality "
                    f"{modality_name} contains "
                    f"{values.shape[0]} cells, "
                    "but target_labels contains "
                    f"{n_target_cells}."
                )

    def __len__(
        self,
    ):

        return (
            self.samples_per_epoch
        )

    def __getitem__(
        self,
        index,
    ):

        # Stochastic sampling is used.
        del index

        source_state_ids = (
            np.random.randint(
                0,
                self.n_source,
                self.cells_per_sample,
            )
        )

        target_state_ids = []

        positive_count = 0

        minimum_positive_count = int(
            self.cells_per_sample
            * self.min_positive_fraction
        )

        for source_state in (
            source_state_ids
        ):

            positive_indices = np.where(
                self.correspondence_matrix[
                    source_state
                ]
                > self.positive_threshold
            )[0]

            select_positive = (
                len(
                    positive_indices
                )
                > 0
                and (
                    np.random.rand()
                    < self.positive_sampling_probability
                    or positive_count
                    < minimum_positive_count
                )
            )

            if select_positive:

                target_state = (
                    np.random.choice(
                        positive_indices
                    )
                )

                positive_count += 1

            else:

                negative_indices = (
                    np.where(
                        self.correspondence_matrix[
                            source_state
                        ]
                        <= self.positive_threshold
                    )[0]
                )

                if (
                    len(
                        negative_indices
                    )
                    > 0
                ):

                    target_state = (
                        np.random.choice(
                            negative_indices
                        )
                    )

                else:

                    target_state = (
                        np.random.randint(
                            0,
                            self.n_target,
                        )
                    )

            target_state_ids.append(
                target_state
            )

        source_cell_indices = [
            np.random.choice(
                np.where(
                    self.source_labels
                    == state_id
                )[0]
            )
            for state_id
            in source_state_ids
        ]

        target_cell_indices = [
            np.random.choice(
                np.where(
                    self.target_labels
                    == state_id
                )[0]
            )
            for state_id
            in target_state_ids
        ]

        source_modalities = {
            modality_name:
                torch.as_tensor(
                    values[
                        source_cell_indices
                    ],
                    dtype=torch.float32,
                )
            for modality_name, values
            in self.source_data.items()
        }

        target_modalities = {
            modality_name:
                torch.as_tensor(
                    values[
                        target_cell_indices
                    ],
                    dtype=torch.float32,
                )
            for modality_name, values
            in self.target_data.items()
        }

        return {
            "source": source_modalities,
            "target": target_modalities,

            "source_labels":
                torch.as_tensor(
                    source_state_ids,
                    dtype=torch.long,
                ),

            "target_labels":
                torch.as_tensor(
                    target_state_ids,
                    dtype=torch.long,
                ),
        }