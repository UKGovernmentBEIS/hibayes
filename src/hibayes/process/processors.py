from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from jax import numpy as jnp

from ._process import DataProcessor, process
from .utils import infer_jax_dtype

if TYPE_CHECKING:
    from ..analysis import AnalysisState
    from ..ui import ModellingDisplay

Features = Dict[str, float | int | jnp.ndarray | np.ndarray]
Coords = Dict[
    str, List[Any]
]  # arviz information on how to map indexes to names. Here we store both the coords and the dims values
Dims = Dict[str, List[str]]


@process
def extract_observed_feature(
    feature_name: str = "score",
) -> DataProcessor:
    """
    Extract a single feature from the data, e.g. the score column.

    Args:
        feature_name: The name of the feature to extract.
    """

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if (
            feature_name not in state.processed_data.columns
        ):  # in the processor interface I check to make sure the processed data is not None how to ignore all the type checkers?
            raise ValueError(
                f"Processed data must contain '{feature_name}' column. Either write a custom processor, or use map_columns processor to map the columns to this name."
            )

        dtype = infer_jax_dtype(state.processed_data[feature_name])

        features: Features = {
            "obs": jnp.array(state.processed_data[feature_name].values, dtype=dtype)
        }

        if state.features:
            state.features.update(features)
        else:
            state.features = features

        if display:
            display.logger.info(
                f"Extracted '{feature_name}' -> 'obs' with dtype: {dtype}"
            )

        return state

    return process


@process
def extract_features(
    feature_names: List[str] = ["score"],
):
    """
    Simply extract features from the dataset. Each feature should be a column in the dataset. No indexing or factorization is done here.
    """

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if not all(col in state.processed_data.columns for col in feature_names):
            raise ValueError(
                f"Processed data must contain {feature_names} columns but only contains {state.processed_data.columns.tolist()}. Either write a custom processor, or use map_columns processor to map the columns to these names."
            )

        features: Features = {}
        for feature_name in feature_names:
            dtype = infer_jax_dtype(state.processed_data[feature_name])
            features[feature_name] = jnp.array(
                state.processed_data[feature_name].values, dtype=dtype
            )
            if display:
                display.logger.info(f"Extracted '{feature_name}' with dtype: {dtype}")

        if state.features:
            state.features.update(features)
        else:
            state.features = features

        return state

    return process


@process
def extract_predictors(
    predictor_names: List[str] = ["model", "task"],
    interactions: bool = False,
    effect_coding_for_main_effects: bool = False,
) -> DataProcessor:
    """
    Extract predictors from the data used in the GLM. For each predictor codes and
    indexes are extracted, along with coordinates for both constrained and full effects.

    Args:
        predictor_names: The names of the features to extract from the data.
        interactions: Whether to extract all possible pairwise interactions - currently just supports two-way interactions.
        effect_coding_for_main_effects: Whether to create coords for effect coding
    """

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if not all(col in state.processed_data.columns for col in predictor_names):
            raise ValueError(
                f"Processed data must contain {predictor_names} columns but only contains {state.processed_data.columns.tolist()}."
            )

        features: Features = {}
        coords: Coords = {}
        dims: Dims = {}

        for predictor_name in predictor_names:
            series = state.processed_data[predictor_name]
            feature_code, feature_index = pd.factorize(series, sort=True)

            # Basic features
            features[predictor_name + "_index"] = jnp.asarray(
                feature_code,
                dtype=jnp.int32,
            )
            features["num_" + predictor_name] = len(feature_index)

            # Coordinates for the predictor levels
            coords[predictor_name] = feature_index.tolist()

            if effect_coding_for_main_effects and len(feature_index) > 1:
                # Dimensions for full effects (all n parameters)
                dims[predictor_name + "_effects"] = [predictor_name]

                # Dimensions for constrained effects (n-1 parameters)
                dims[predictor_name + "_effects_constrained"] = [
                    f"{predictor_name}_constrained"
                ]
                coords[f"{predictor_name}_constrained"] = feature_index[
                    :-1
                ].tolist()  # All but last
            else:
                # Standard dummy coding - use all levels
                dims[predictor_name + "_effects"] = [predictor_name]

            if display:
                display.logger.info(
                    f"Extracted '{predictor_name}' -> '{predictor_name}_index'"
                )
                if effect_coding_for_main_effects and len(feature_index) > 1:
                    display.logger.info(
                        f"  - Full effects: {len(feature_index)} params"
                    )
                    display.logger.info(
                        f"  - Constrained effects: {len(feature_index) - 1} params"
                    )

        # Handle interactions - generate all pairwise combinations
        if (
            interactions and len(predictor_names) >= 2
        ):  # currently just two-way interactions
            all_interactions = list(combinations(predictor_names, 2))

            for pred1, pred2 in all_interactions:
                interaction_name = f"{pred1}_{pred2}"
                n1 = features[f"num_{pred1}"]
                n2 = features[f"num_{pred2}"]

                # Full interaction effects n1*n2 parameters
                dims[f"{interaction_name}_effects"] = [pred1, pred2]

                if display:
                    display.logger.info(
                        f"Added interaction dims for '{interaction_name}'"
                    )
                    if effect_coding_for_main_effects and n1 > 1 and n2 > 1:
                        display.logger.info(f"  - Full: ({n1}, {n2}) params")
                        display.logger.info(
                            f"  - Constrained: ({n1 - 1}, {n2 - 1}) params"
                        )

        # Update state
        if state.features:
            state.features.update(features)
        else:
            state.features = features

        if state.coords:
            state.coords.update(coords)
        else:
            state.coords = coords

        if state.dims:
            state.dims.update(dims)
        else:
            state.dims = dims

        return state

    return process


@process
def drop_rows_with_missing_features(
    feature_names: List[str] = ["model", "task"],
) -> DataProcessor:
    """
    Drop rows with missing features from the data.

    Args:
        feature_names: The names of the features to check for missing values.
    """

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if display:
            display.logger.info(f"Dropping rows with missing features: {feature_names}")

        state.processed_data.dropna(subset=feature_names, inplace=True)

        return state

    return process


@process
def map_columns(
    column_mapping: Optional[Dict[str, str]] = None,
) -> DataProcessor:
    """
    Map columns in the data to new names.

    Args:
        column_mapping: A dictionary mapping old column names to new column names.
    """

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if column_mapping is None:
            return state
        if display:
            display.logger.info(f"Mapping columns: {column_mapping}")

        state.processed_data.rename(columns=column_mapping, inplace=True)

        return state

    return process


@process
def groupby(
    groupby_columns: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> DataProcessor:
    """
    Group the data by specified columns and aggregate scores for binomial models.
    adds n_correct and n_total columsn to the dataset.

    Args:
        groupby_columns: A list of columns to group by.
        *args: Additional positional arguments to pass to the `groupby` method.
        **kwargs: Additional keyword arguments to pass to the `groupby` method.
    """

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if groupby_columns is None:
            return state

        if not all(
            col in state.processed_data.columns for col in groupby_columns + ["score"]
        ):
            raise ValueError(
                f"Processed data must both contain {groupby_columns} and score columns. Either write a custom processor, or use map_columns processor to map the columns to these names."
            )

        if display:
            display.logger.info(f"Grouping data by: {groupby_columns}")

        state.processed_data = (
            state.processed_data.groupby(groupby_columns, *args, **kwargs)
            .agg(
                n_correct=("score", "sum"),
                n_total=("score", "count"),
            )
            .reset_index()
        )

        return state

    return process
