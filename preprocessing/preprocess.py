#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Mapping

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MultimodalPreprocessor:

    def __init__(
        self,
        pca_dims: Mapping[str, int],
        random_state: int = 42,
    ):

        if len(pca_dims) == 0:
            raise ValueError(
                "pca_dims must contain "
                "at least one modality."
            )

        self.pca_dims = {
            modality_name: int(
                dimension
            )
            for modality_name, dimension
            in pca_dims.items()
        }

        self.random_state = (
            random_state
        )

        self.scalers = {
            modality_name:
                StandardScaler()
            for modality_name
            in self.pca_dims
        }

        self.pca_models: Dict[
            str,
            PCA,
        ] = {}

        self.input_dims = {}

    @staticmethod
    def _get_safe_n_components(
        data: np.ndarray,
        target_dim: int,
    ) -> int:

        n_samples, n_features = (
            data.shape
        )

        safe_dimension = min(
            target_dim,
            max(
                1,
                n_samples - 1,
            ),
            n_features,
        )

        return int(
            safe_dimension
        )

    def fit_transform(
        self,
        data: Mapping[
            str,
            np.ndarray,
        ],
    ) -> Dict[
        str,
        np.ndarray,
    ]:

        processed = {}

        for (
            modality_name,
            target_dim,
        ) in self.pca_dims.items():

            if (
                modality_name
                not in data
            ):
                raise KeyError(
                    "Missing source modality: "
                    f"{modality_name}"
                )

            values = np.asarray(
                data[modality_name]
            )

            if values.ndim != 2:
                raise ValueError(
                    f"{modality_name} must "
                    "be a 2D matrix."
                )

            self.input_dims[
                modality_name
            ] = values.shape[1]

            scaled = (
                self.scalers[
                    modality_name
                ].fit_transform(
                    values
                )
            )

            n_components = (
                self._get_safe_n_components(
                    values,
                    target_dim,
                )
            )

            self.pca_models[
                modality_name
            ] = PCA(
                n_components=n_components,
                random_state=(
                    self.random_state
                ),
            )

            processed[
                modality_name
            ] = (
                self.pca_models[
                    modality_name
                ].fit_transform(
                    scaled
                )
            )

        return processed

    def transform(
        self,
        data: Mapping[
            str,
            np.ndarray,
        ],
    ) -> Dict[
        str,
        np.ndarray,
    ]:

        if (
            len(self.pca_models)
            == 0
        ):
            raise RuntimeError(
                "The preprocessor must "
                "be fitted before transform()."
            )

        processed = {}

        for modality_name in (
            self.pca_dims
        ):

            if (
                modality_name
                not in data
            ):
                raise KeyError(
                    "Missing target modality: "
                    f"{modality_name}"
                )

            values = np.asarray(
                data[modality_name]
            )

            if values.ndim != 2:
                raise ValueError(
                    f"{modality_name} must "
                    "be a 2D matrix."
                )

            expected_dim = (
                self.input_dims[
                    modality_name
                ]
            )

            if (
                values.shape[1]
                != expected_dim
            ):
                raise ValueError(
                    f"Feature dimension mismatch "
                    f"for {modality_name}: "
                    f"source={expected_dim}, "
                    f"target={values.shape[1]}."
                )

            scaled = (
                self.scalers[
                    modality_name
                ].transform(
                    values
                )
            )

            processed[
                modality_name
            ] = (
                self.pca_models[
                    modality_name
                ].transform(
                    scaled
                )
            )

        return processed