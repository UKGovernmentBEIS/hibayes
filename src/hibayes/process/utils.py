from typing import Any

import jax.numpy as jnp
import pandas as pd


def get_category_order(
    series: pd.Series,
    feature_name: str,
    reference_category: Any | None = None,
    custom_order: list[Any] | None = None,
) -> pd.Index:
    """
    Determine the ordering of categories for a categorical feature.

    Args:
        series: The pandas Series containing the categorical data.
        feature_name: Name of the feature (used in error messages).
        reference_category: If provided, this category will be placed first (index 0),
            making it the reference/baseline in dummy coding.
        custom_order: If provided, specifies the full ordering of categories.
            The first category becomes the reference. Takes precedence over
            reference_category if both are specified.

    Returns:
        pd.Index with the ordered category values.

    Raises:
        ValueError: If custom_order is missing categories found in the data,
            or if reference_category is not found in the data.
    """
    if custom_order is not None:
        # User specified full ordering
        unique_in_data = set(series.unique())
        unique_in_order = set(custom_order)

        # Check for categories in data but not in order
        missing_from_order = unique_in_data - unique_in_order
        if missing_from_order:
            raise ValueError(
                f"category_order for '{feature_name}' is missing categories found in data: "
                f"{sorted(missing_from_order)}"
            )

        # Filter to only categories present in data (in specified order)
        return pd.Index([c for c in custom_order if c in unique_in_data])

    if reference_category is not None:
        # User specified just the reference category
        _, default_index = pd.factorize(series, sort=True)

        if reference_category not in default_index:
            raise ValueError(
                f"reference_categories['{feature_name}'] = {reference_category!r} not found in data. "
                f"Available categories: {default_index.tolist()}"
            )

        # Put reference category first, keep others in sorted order
        others = [c for c in default_index if c != reference_category]
        return pd.Index([reference_category] + others)

    # Default: alphabetical sorting
    _, index = pd.factorize(series, sort=True)
    return index


def infer_jax_dtype(pandas_series: pd.Series) -> jnp.dtype:
    """Infer appropriate JAX dtype from pandas series."""
    pandas_dtype = pandas_series.dtype

    # int types
    if pandas_dtype.kind == "i":  # signed integer
        return jnp.int32
    elif pandas_dtype.kind == "u":  # unsigned integer
        if pandas_dtype.itemsize <= 4:
            return jnp.uint32
        else:
            return jnp.uint64
    # flt types
    elif pandas_dtype.kind == "f":  # floating point
        return jnp.float32  # no float64 for GPU performance

    # bool type
    elif pandas_dtype.kind == "b":  # boolean
        return jnp.bool_
    raise ValueError(
        f"Unsupported pandas dtype '{pandas_dtype}' for conversion to JAX dtype."
    )
