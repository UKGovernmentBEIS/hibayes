import copy
from collections.abc import Sequence
from typing import Callable, Type, TypeVar

import numpy as np
from scipy.stats import norm

T = TypeVar("T")


def logit_to_prob(x):
    return 1 / (1 + np.exp(-x))


def probit_to_prob(x):
    return norm.cdf(x)


def cloglog_to_prob(x):
    """log‑log link."""
    return 1.0 - np.exp(-np.exp(x))


def merge_sequence(
    base_seq: Sequence, update_seq: Sequence, field_type: Type
) -> Sequence:
    """Merge two sequences, with update_seq taking precedence"""
    # For lists, we'll use the update sequence by default
    # This behavior could be customized based on your specific needs
    return copy.deepcopy(update_seq)


def _link_to_key(fn: Callable | str, mapping: dict[str, Callable]) -> str:
    """Return the key in LINK_FUNCTION_MAP that maps to `fn`."""
    if isinstance(fn, str):
        return fn
    for k, v in mapping.items():
        if v is fn:
            return k
    raise ValueError(f"Unknown link function {fn!r}")
