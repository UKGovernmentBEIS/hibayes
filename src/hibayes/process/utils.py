import jax.numpy as jnp
import pandas as pd


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
        if pandas_dtype.itemsize <= 4:
            return jnp.float32
        else:
            return jnp.float64
    raise ValueError(
        f"Unsupported pandas dtype '{pandas_dtype}' for conversion to JAX dtype."
    )
