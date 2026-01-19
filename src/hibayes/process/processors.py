from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from jax import numpy as jnp
import json

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
    test: Optional[str] = None,
) -> DataProcessor:
    """
    Extract a single feature from the data, e.g. the score column.

    Args:
        feature_name: The name of the feature to extract.
        test: Whether to also extract the feature for test data.
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

        if test:

            features: Features = {
                "obs": jnp.array(state.processed_data[~state.processed_data[test]][feature_name].values, dtype=dtype)
            }
            test_features: Features = {
                "obs": jnp.array(state.processed_data[state.processed_data[test]][feature_name].values, dtype=dtype)
            }
            if state.test_features:
                state.test_features.update(test_features)
            else:
                state.test_features = test_features

        else:

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
    test: Optional[str] = None,
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
        test: Column name indicating test data for train/test split
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
        test_features: Features = {} if test else None
        coords: Coords = {}
        dims: Dims = {}

        # Determine train/test split if test column provided
        train_mask = None
        test_mask = None
        if test and test in state.processed_data.columns:
            test_mask = state.processed_data[test].astype(bool)
            train_mask = ~test_mask
            if display:
                n_train = train_mask.sum()
                n_test = test_mask.sum()
                display.logger.info(f"Processing with train/test split - Train: {n_train}, Test: {n_test}")

        # continuous
        for feat in continuous_features:
            series = state.processed_data[feat]
            dtype = infer_jax_dtype(series)
            if standardise:
                # Standardise using train data statistics only
                if train_mask is not None:
                    train_series = series[train_mask]
                    mean = float(train_series.mean())
                    std = float(train_series.std())
                else:
                    mean = float(series.mean())
                    std = float(series.std())
                state.processed_data[feat] = (series - mean) / (std + 1e-8)
                if display:
                    display.logger.info(
                        f"Standardised continuous '{feat}' (mean: {mean}, std: {std})"
                    )

            if test_mask is not None:
                features[feat] = jnp.asarray(state.processed_data.loc[train_mask, feat].values, dtype=dtype)
                test_features[feat] = jnp.asarray(state.processed_data.loc[test_mask, feat].values, dtype=dtype)
            else:
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


            if test_mask is not None:
                train_code = code[train_mask]
                test_code = code[test_mask]
                features[f"{cat}_index"] = jnp.asarray(train_code, dtype=jnp.int32)
                test_features[f"{cat}_index"] = jnp.asarray(test_code, dtype=jnp.int32)
                features[f"num_{cat}"] = len(index)
                test_features[f"num_{cat}"] = len(index)
            else:
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
                if test_mask is not None:
                    x_train_values = jnp.asarray(state.processed_data.loc[train_mask, x].values, dtype=x_dtype)
                    x_test_values = jnp.asarray(state.processed_data.loc[test_mask, x].values, dtype=x_dtype)
                else:
                    x_values = jnp.asarray(state.processed_data[x].values, dtype=x_dtype)

                for g in categorical_features:
                    # per-observation regressor reused with category-index in the model
                    if test_mask is not None:
                        features[f"{g}_{x}"] = x_train_values
                        features[f"{x}_{g}"] = x_train_values  # allow either order
                        test_features[f"{g}_{x}"] = x_test_values
                        test_features[f"{x}_{g}"] = x_test_values  # allow either order
                    else:
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

        # Update test_features if test split was used
        if test_features:
            if state.test_features:
                state.test_features.update(test_features)
            else:
                state.test_features = test_features

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
    adds n_correct and n_total columns to the dataset.

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

@process
def merge_scout_results(
    scout_scan_path: str,
    scanner_name: str,
    join_on_left: str = "transcript_id",
    join_on_right: str = "transcript_id",
    columns_to_keep: list[str] | None = None,
    prefix: str | None = None
) -> DataProcessor:
    """
    Merge Scout scan results into the processed data.
    
    Requires: inspect-scout (install with: pip install inspect-scout)
    
    Parameters:
    - scout_scan_path: Path to Scout scan directory
    - scanner_name: Name of the scanner
    - join_on_left: Column in processed data to join on (default: "transcript_id")
    - join_on_right: Column in Scout results to join on (default: "transcript_id")
                     If not found as a column, will auto-extract from transcript_metadata
    - columns_to_keep: Scout columns to merge (default: ['value', 'explanation', 'scan_model_usage'])
    - prefix: Prefix for ALL merged columns from Scout (default: scanner_name)
    
    Example:
    - merge_scout_results:
        scout_scan_path: "scans/scan_id=abc123"
        scanner_name: "is_looping"
        join_on_left: "id"
        join_on_right: "transcript_id"
        prefix: "from_scout_loop"
    """
    try:
        from inspect_scout import scan_results_df
    except ImportError:
        raise ImportError(
            "inspect-scout is required for merge_scout_results. "
            "Install it with: pip install inspect-scout"
        ) from None

    def process(state: AnalysisState, display: ModellingDisplay | None = None) -> AnalysisState:
        if state.processed_data is None or state.processed_data.empty:
            return state

        # Validate join column exists in processed data
        if join_on_left not in state.processed_data.columns:
            raise ValueError(
                f"Column '{join_on_left}' not found in processed data. "
                f"Available columns: {state.processed_data.columns.tolist()}"
            )

        # Set defaults - include scan_model_usage for the grader model
        cols_to_keep = columns_to_keep or ['value', 'explanation', 'scan_model_usage']
        final_prefix = prefix or scanner_name
        
        # Load Scout results
        scout_results = scan_results_df(scout_scan_path)

        # Validate scanner exists
        if scanner_name not in scout_results.scanners:
            available = list(scout_results.scanners.keys())
            raise ValueError(
                f"Scanner '{scanner_name}' not found in Scout results. "
                f"Available scanners: {available}"
            )

        scout_df = scout_results.scanners[scanner_name].copy()
        
        # Extract join column from metadata if it doesn't exist
        if join_on_right not in scout_df.columns:
            def extract(meta):
                try:
                    data = json.loads(meta) if isinstance(meta, str) else meta
                    return data.get(join_on_right) if isinstance(data, dict) else None
                except (json.JSONDecodeError, TypeError, AttributeError):
                    return None
            
            scout_df[join_on_right] = scout_df['transcript_metadata'].apply(extract)
            
            if display:
                display.logger.info(f"Extracted '{join_on_right}' from metadata: {scout_df[join_on_right].head(3).tolist()}")
        
        # Filter to requested columns
        cols_to_keep = [c for c in cols_to_keep if c in scout_df.columns]
        if not cols_to_keep:
            if display:
                display.logger.warning("No valid columns to merge")
            return state
        
        # Select columns (including join column)
        scout_df = scout_df[[join_on_right] + cols_to_keep].copy()
        
        # Rename ALL scout columns with prefix (including join column to avoid conflicts)
        rename_map = {c: f"{final_prefix}_{c}" for c in cols_to_keep}
        if join_on_left != join_on_right:
            rename_map[join_on_right] = f"{final_prefix}_{join_on_right}"
        
        scout_df = scout_df.rename(columns=rename_map)
        
        # Merge
        before = len(state.processed_data)
        merged = pd.merge(
            state.processed_data, 
            scout_df, 
            left_on=join_on_left, 
            right_on=f"{final_prefix}_{join_on_right}" if join_on_left != join_on_right else join_on_right,
            how='left'
        )
        
        # Drop the renamed join column after merge (we don't need it)
        join_col_to_drop = f"{final_prefix}_{join_on_right}"
        if join_col_to_drop in merged.columns:
            merged = merged.drop(columns=[join_col_to_drop])
        
        # Report
        if display:
            matched = merged[f"{final_prefix}_{cols_to_keep[0]}"].notna().sum()
            display.logger.info(f"Merged '{scanner_name}': {matched}/{before} rows matched on {join_on_left}←{join_on_right}")
            display.logger.info(f"  Added columns: {[f'{final_prefix}_{c}' for c in cols_to_keep]}")
        
        state.processed_data = merged
        return state

    return process
