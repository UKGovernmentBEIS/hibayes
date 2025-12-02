from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from jax import numpy as jnp

from ._process import DataProcessor, process
from .utils import get_category_order, infer_jax_dtype

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
    categorical_features: list[str] | None = None,
    continuous_features: list[str] | None = None,
    *,
    interactions: bool = False,
    effect_coding_for_main_effects: bool = False,
    standardise: bool = False,
    reference_categories: dict[str, Any] | None = None,
    category_order: dict[str, list[Any]] | None = None,
):
    """
    Extract inputs for GLMs:
      - Categorical: factorised indices, counts, coords, dims (+ optional constrained coords)
      - Continuous: raw arrays (optionally standardised)
      - Interactions:
          * categorical × categorical: dims [c1, c2]
          * continuous × categorical: feature '{x}_{g}_effects'

    Args:
        categorical_features: List of categorical column names to extract.
        continuous_features: List of continuous column names to extract.
        interactions: Whether to extract pairwise interactions.
        effect_coding_for_main_effects: Whether to create constrained coords for effect coding.
        standardise: Whether to standardise continuous features.
        reference_categories: Dict mapping categorical feature names to the category value
            that should be treated as the reference (placed first in the ordering, so it
            becomes the zero/baseline in dummy coding). If a feature is not in this dict,
            the default alphabetical ordering is used.
        category_order: Dict mapping categorical feature names to the full list of category
            values in the desired order. The first category in the list becomes the reference.
            Takes precedence over reference_categories if both are specified for a feature.
    """
    categorical_features = categorical_features or []
    continuous_features = continuous_features or []
    reference_categories = reference_categories or {}
    category_order = category_order or {}

    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        all_needed = categorical_features + continuous_features
        missing = [c for c in all_needed if c not in state.processed_data.columns]
        if missing:
            raise ValueError(
                f"Processed data missing required columns: {missing}. "
                f"Available: {state.processed_data.columns.tolist()}"
            )
        overlap = set(categorical_features).intersection(continuous_features)
        if overlap:
            raise ValueError(
                f"Columns cannot be both categorical and continuous: {sorted(overlap)}"
            )

        features: Features = {}
        coords: Coords = {}
        dims: Dims = {}

        # continuous
        for feat in continuous_features:
            series = state.processed_data[feat]
            dtype = infer_jax_dtype(series)
            if standardise:
                mean = float(series.mean())
                std = float(series.std())
                state.processed_data[feat] = (series - mean) / (std + 1e-8)
                if display:
                    display.logger.info(
                        f"Standardised continuous '{feat}' (mean: {mean}, std: {std})"
                    )
            features[feat] = jnp.asarray(state.processed_data[feat].values, dtype=dtype)
            if display:
                display.logger.info(
                    f"Extracted continuous '{feat}' with dtype: {dtype}"
                )

        # categorical
        for cat in categorical_features:
            series = state.processed_data[cat]

            # Determine category ordering
            index = get_category_order(
                series,
                feature_name=cat,
                reference_category=reference_categories.get(cat),
                custom_order=category_order.get(cat),
            )

            # Create mapping from category to index
            cat_to_idx = {cat_val: i for i, cat_val in enumerate(index)}
            code = series.map(cat_to_idx).values

            features[f"{cat}_index"] = jnp.asarray(code, dtype=jnp.int32)
            features[f"num_{cat}"] = len(index)
            coords[cat] = index.tolist()
            dims[f"{cat}_effects"] = [cat]
            if effect_coding_for_main_effects and len(index) > 1:
                dims[f"{cat}_effects_constrained"] = [f"{cat}_constrained"]
                coords[f"{cat}_constrained"] = index[:-1].tolist()
            if display:
                display.logger.info(f"Extracted categorical '{cat}' -> '{cat}_index'")
                display.logger.info(f"  - Category order: {index.tolist()} (reference: {index[0]})")
                if effect_coding_for_main_effects and len(index) > 1:
                    display.logger.info(
                        f"  - Full effects: {len(index)}; Constrained: {len(index) - 1}"
                    )

        # interactions
        if interactions:
            # categorical × categorical dims
            if len(categorical_features) >= 2:
                for c1, c2 in combinations(categorical_features, 2):
                    inter_name = f"{c1}_{c2}"
                    n1 = int(features[f"num_{c1}"])
                    n2 = int(features[f"num_{c2}"])
                    dims[f"{inter_name}_effects"] = [c1, c2]
                    if display:
                        display.logger.info(
                            f"Added interaction dims for '{inter_name}' (full: {n1}×{n2})"
                        )
                        if effect_coding_for_main_effects and n1 > 1 and n2 > 1:
                            display.logger.info(
                                f"  - Constrained (if used in model): {(n1 - 1)}×{(n2 - 1)}"
                            )

            # continuous × categorical: add feature + dims for per-category slopes
            for x in continuous_features:
                x_dtype = infer_jax_dtype(state.processed_data[x])
                x_values = jnp.asarray(state.processed_data[x].values, dtype=x_dtype)
                for g in categorical_features:
                    # per-observation regressor reused with category-index in the model
                    features[f"{g}_{x}"] = x_values
                    features[f"{x}_{g}"] = x_values  # allow either order
                    # effects: one slope per category level (optionally constrained)
                    dims[f"{g}_{x}_effects"] = [g]
                    dims[f"{x}_{g}_effects"] = [g]  # allow either order
                    if display:
                        display.logger.info(
                            f"Added cont×cat interaction {g}_{x}; "
                            f"effects dims '{g}_{x}_effects'=[{g}]"
                        )

        if state.features:
            state.features.update(features)
        else:
            state.features = features

        if coords:
            if state.coords:
                state.coords.update(coords)
            else:
                state.coords = coords

        if dims:
            if state.dims:
                state.dims.update(dims)
            else:
                state.dims = dims

        return state

    return process


@process
def extract_features_deprecated(
    feature_names: List[str] = ["score"],
    standardise: bool = False,
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
            if standardise:
                mean = state.processed_data[feature_name].mean()
                std = state.processed_data[feature_name].std()
                state.processed_data[feature_name] = (
                    state.processed_data[feature_name] - mean
                ) / (std + 1e-8)

                features[feature_name] = jnp.array(
                    state.processed_data[feature_name].values, dtype=dtype
                )
                if display:
                    display.logger.info(
                        f"Standardised '{feature_name}' (mean: {mean}, std: {std})"
                    )
            else:
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
def extract_predictors_deprecated(
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
