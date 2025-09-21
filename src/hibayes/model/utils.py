from typing import Callable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from scipy.stats import norm

from ..process import Features


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


def create_interaction_effects(
    name1: str, name2: str, features: Features, prior: dist.Distribution
) -> jnp.ndarray:
    """Create interaction effects matrix with sum-to-zero constraints."""
    n1 = features[f"num_{name1}"]
    n2 = features[f"num_{name2}"]

    if n1 == 1 or n2 == 1:
        return jnp.zeros((n1, n2))

    # Sample free parameters (excluding last row and column)
    raw = numpyro.sample(
        f"{name1}_{name2}_effects_constrained", prior.expand([(n1 - 1) * (n2 - 1)])
    ).reshape((n1 - 1, n2 - 1))

    # Initialize full matrix
    b_full = jnp.zeros((n1, n2))

    # Fill in the free parameters
    b_full = b_full.at[: n1 - 1, : n2 - 1].set(raw)

    # Set last row to satisfy row sum-to-zero constraint
    b_full = b_full.at[n1 - 1, : n2 - 1].set(
        -jnp.sum(b_full[: n1 - 1, : n2 - 1], axis=0)
    )

    # Set last column to satisfy column sum-to-zero constraint
    b_full = b_full.at[:, n2 - 1].set(-jnp.sum(b_full[:, : n2 - 1], axis=1))

    numpyro.deterministic(f"{name1}_{name2}_effects", b_full)
    return b_full
