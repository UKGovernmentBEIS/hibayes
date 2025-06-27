from typing import Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm


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


def logit_to_prob(x):
    return 1 / (1 + np.exp(-x))


def probit_to_prob(x):
    return norm.cdf(x)


def cloglog_to_prob(x):
    """log‑log link."""
    return 1.0 - np.exp(-np.exp(x))


def link_to_key(fn: Callable | str, mapping: dict[str, Callable]) -> str:
    """Return the key in LINK_FUNCTION_MAP that maps to `fn`."""
    if isinstance(fn, str):
        return fn
    for k, v in mapping.items():
        if v is fn:
            return k
    raise ValueError(f"Unknown link function {fn!r}")
