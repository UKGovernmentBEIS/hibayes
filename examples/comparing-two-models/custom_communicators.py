"""Plots for the comparing-two-models example.

The model (``accuracy_comparison``), the results table
(``comparison_summary``) and the fit check (``agreement_check``) ship with
hibayes. These communicators are example-specific plots, and show how to
write your own: each is a function decorated with ``@communicate`` that reads
the fitted models from the analysis state and adds figures to it.
"""

import re
from typing import List, Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
from scipy.special import expit

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay


# ---------------------------------------------------------------------------
# Plots. Colours and styling follow one light-mode palette so the figures
# read as a set: blue for the model / the reference AI model, orange for the
# other AI model, greys for "the models agree".
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#6b6a63"
GRID = "#e4e3dd"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AGREE_RIGHT = "#c9c8c1"
AGREE_WRONG = "#8f8e86"
STYLE = {
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


def _tidy(ax, left: bool = False) -> None:
    for side in ("top", "right") + (() if left else ("left",)):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=6)


def _model_label(name: str) -> str:
    """Model ids like 'claude-3-5-sonnet-20240620' -> 'claude-3-5-sonnet'."""
    return re.sub(r"-\d{8}$", "", str(name))


def _short(name: str) -> str:
    """Dataset ids like 'ehovy/race' -> 'race'."""
    return str(name).split("/")[-1]


def _groups(post) -> List:
    """Benchmarks if the fit has a benchmark dimension, else one group."""
    return list(post["benchmark"].values) if "benchmark" in post.dims else [None]


def _sel(da, group):
    return da.sel(benchmark=group) if group is not None else da


def _group_label(state: AnalysisState, group) -> str:
    if group is not None:
        return _short(group)
    data = state.processed_data
    if "dataset" in data and data["dataset"].nunique() == 1:
        return _short(data["dataset"].iloc[0])
    return "all items"


def _answer_pairs(state: AnalysisState) -> pd.DataFrame | None:
    """
    One row per (group, item, epoch) with each AI model's 0/1 score as a
    column, or None if the data has been aggregated (no item column).
    """
    data = state.processed_data
    if data is None or not {"item", "model", "score"} <= set(data.columns):
        return None
    group = "benchmark" if "benchmark" in data else None
    index = [c for c in (group, "item", "epoch") if c and c in data]
    pairs = data.pivot_table(index=index, columns="model", values="score").dropna()
    if group is None:
        pairs = pd.concat({"": pairs}, names=["benchmark"])
    return pairs


def _paired_fits(state: AnalysisState):
    """Fitted models that have item offsets (the paired models)."""
    for model_analysis in state.models:
        if model_analysis.is_fitted and "item_sigma" in (
            model_analysis.inference_data.posterior
        ):
            yield model_analysis


def _thinned_posterior(model_analysis, n: int):
    post = model_analysis.inference_data.posterior.stack(sample=("chain", "draw"))
    keep = np.linspace(0, post.sizes["sample"] - 1, n).astype(int)
    return post.isel(sample=keep)


def _prior_label(site: dict) -> str:
    fn = site["fn"]
    while hasattr(fn, "base_dist"):  # unwrap plates / expand
        fn = fn.base_dist
    params = [
        np.ravel(np.asarray(getattr(fn, p)))[0]
        for p in ("loc", "scale")
        if hasattr(fn, p)
    ]
    return f"{type(fn).__name__}({', '.join(f'{v:g}' for v in params)})"


@communicate
def prior_plot(num_samples: int = 4000, seed_value: int = 0) -> Communicator:
    """
    What each comparison model's priors imply on the accuracy scale.

    Runs the configured model with no data (prior predictive), so the plot
    reflects whatever prior settings the config passed. Left: the reference
    model's accuracy. Right: the accuracy gap between the two AI models.
    Panel titles show the priors on ``baseline`` and ``difference``.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        reference, other = state.coords["model"][:2]
        ref_label, oth_label = _model_label(reference), _model_label(other)
        made = 0
        for model_analysis in state.models:
            features = model_analysis.features
            sites = trace(seed(model_analysis.model, seed_value)).get_trace(features)
            if "accuracy_difference" not in sites:
                continue
            draws = Predictive(model_analysis.model, num_samples=num_samples)(
                jax.random.PRNGKey(seed_value), features
            )
            accuracy = np.asarray(draws["accuracy"])[..., 0].ravel()
            gap = np.asarray(draws["accuracy_difference"]).ravel()

            panels = [
                (
                    accuracy,
                    np.linspace(0, 1, 26),
                    f"baseline ~ {_prior_label(sites['baseline'])}",
                    f"{ref_label} accuracy implied by the prior",
                    "{:.2f}",
                ),
                (
                    gap,
                    np.linspace(-1, 1, 41),
                    f"difference ~ {_prior_label(sites['difference'])}",
                    f"Accuracy gap ({oth_label} minus {ref_label}) implied by the prior",
                    "{:+.2f}",
                ),
            ]
            with plt.rc_context(STYLE):
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
                for ax, (values, bins, title, xlabel, fmt) in zip(axes, panels):
                    ax.hist(
                        values,
                        bins=bins,
                        density=True,
                        color=BLUE,
                        edgecolor=SURFACE,
                        linewidth=2,
                    )
                    lo, hi = np.quantile(values, [0.025, 0.975])
                    ax.set_title(
                        title, loc="left", fontsize=13, fontweight="bold", pad=24
                    )
                    ax.text(
                        0.0,
                        1.02,
                        f"95% of prior mass between {fmt.format(lo)} and {fmt.format(hi)}",
                        transform=ax.transAxes,
                        va="bottom",
                        fontsize=10,
                        color=MUTED,
                    )
                    ax.set_xlabel(xlabel)
                    ax.set_yticks([])
                    _tidy(ax)
                fig.tight_layout(w_pad=4)
            state.add_plot(fig, f"model_{model_analysis.model_name}_priors")
            made += 1
        return state, "pass" if made else "NA"

    return communicate


@communicate
def agreement_plot() -> Communicator:
    """
    How often the two AI models agree, per benchmark.

    Pairs each item's answers from both models (and each epoch, if items
    were answered repeatedly) into four patterns. Only the two
    "only one model right" patterns carry information about which model is
    better; those segments are coloured and labelled with their counts.
    Needs one row per answer (no ``groupby``).
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        pairs = _answer_pairs(state)
        if pairs is None:
            if display:
                display.logger.warning(
                    "agreement_plot needs one row per answer with item, model and "
                    "score columns; skipping."
                )
            return state, "NA"
        reference, other = state.coords["model"][:2]
        ref_label, oth_label = _model_label(reference), _model_label(other)
        ref, oth = pairs[reference], pairs[other]
        kinds = {
            "both right": (ref == 1) & (oth == 1),
            f"only {ref_label} right": (ref == 1) & (oth == 0),
            f"only {oth_label} right": (ref == 0) & (oth == 1),
            "both wrong": (ref == 0) & (oth == 0),
        }
        colours = [AGREE_RIGHT, BLUE, ORANGE, AGREE_WRONG]
        counts = pd.DataFrame(
            {k: v.groupby(level="benchmark").sum() for k, v in kinds.items()}
        ).astype(int)
        shares = counts.div(counts.sum(axis=1), axis=0)
        groups = list(counts.index)

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 1.3 + 0.75 * len(groups)))
            y = np.arange(len(groups))[::-1]
            left = np.zeros(len(groups))
            for kind, colour in zip(kinds, colours):
                ax.barh(
                    y,
                    shares[kind],
                    left=left,
                    height=0.6,
                    color=colour,
                    edgecolor=SURFACE,
                    linewidth=2,
                    label=kind,
                )
                if colour in (BLUE, ORANGE):  # direct-label the informative segments
                    for yi, start, w, n in zip(y, left, shares[kind], counts[kind]):
                        ax.text(
                            start + w / 2,
                            yi - 0.42,
                            f"{n}",
                            ha="center",
                            va="top",
                            fontsize=9,
                        )
                left += shares[kind]
            labels = [
                f"{_group_label(state, g or None)}  ({counts.loc[g].sum():,} pairs)"
                for g in groups
            ]
            ax.set_yticks(y, labels)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.9, len(groups) - 0.4)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_xlabel("Share of answer pairs (same item, same epoch)")
            ax.legend(ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
            _tidy(ax)
            fig.tight_layout()
        state.add_plot(fig, "agreement")
        return state, "pass"

    return communicate


@communicate
def item_difficulty_plot(num_draws: int = 2000, seed_value: int = 0) -> Communicator:
    """
    The fitted spread of item difficulty, and the accuracy gap it implies.

    Top row: the chance the reference model answers a *new* item correctly,
    drawn from each benchmark's fitted Normal(baseline, item_sigma) on the
    log-odds scale. This is the population distribution, not per-item
    estimates: with one or a few answers per item, an individual item's
    difficulty mostly reflects its own outcome pattern.

    Bottom row: the accuracy gap a constant log-odds difference implies at
    each difficulty, with a 95% interval. It is largest for medium-difficulty
    items and shrinks towards zero for very easy or very hard ones.
    """

    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        reference, other = state.coords["model"][:2]
        ref_label, oth_label = _model_label(reference), _model_label(other)
        rng = np.random.default_rng(seed_value)
        grid = np.linspace(0.005, 0.995, 199)
        made = 0
        for model_analysis in _paired_fits(state):
            post = _thinned_posterior(model_analysis, num_draws)
            groups = _groups(post)
            with plt.rc_context(STYLE):
                fig, axes = plt.subplots(
                    2,
                    len(groups),
                    figsize=(3 * len(groups) + 1, 5.4),
                    sharex=True,
                    squeeze=False,
                    gridspec_kw={"height_ratios": [1, 1.2]},
                )
                y_lo = y_hi = 0.0
                for col, group in enumerate(groups):
                    base, sigma = (
                        np.asarray(_sel(post[v], group))
                        for v in ("baseline", "item_sigma")
                    )
                    diff = np.asarray(
                        _sel(post["difference"], group).isel(model_contrast=0)
                    )
                    z = rng.standard_normal((base.size, 200))
                    p_new = expit(base[:, None] + sigma[:, None] * z).ravel()
                    ax = axes[0, col]
                    ax.hist(
                        p_new,
                        bins=np.linspace(0, 1, 21),
                        color=BLUE,
                        edgecolor=SURFACE,
                        linewidth=1.5,
                        weights=np.full(p_new.size, 1 / p_new.size),
                    )
                    ax.set_title(
                        _group_label(state, group),
                        loc="left",
                        fontweight="bold",
                        fontsize=12,
                        pad=18,
                    )
                    ax.text(
                        0.0,
                        1.02,
                        f"σ_item ≈ {sigma.mean():.1f} log-odds",
                        transform=ax.transAxes,
                        va="bottom",
                        fontsize=9,
                        color=MUTED,
                    )
                    ax.set_yticks([])
                    _tidy(ax)

                    logit = np.log(grid / (1 - grid))
                    gap = expit(logit[None, :] + diff[:, None]) - grid[None, :]
                    lo, mid, hi = np.quantile(gap, [0.025, 0.5, 0.975], axis=0)
                    y_lo, y_hi = min(y_lo, lo.min()), max(y_hi, hi.max())
                    ax = axes[1, col]
                    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
                    ax.fill_between(grid, lo, hi, color=BLUE, alpha=0.2, linewidth=0)
                    ax.plot(grid, mid, color=BLUE, linewidth=2)
                    ax.set_xlim(0, 1)
                    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
                    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
                    _tidy(ax, left=True)
                    if col:
                        ax.tick_params(labelleft=False)
                pad = 0.05 * (y_hi - y_lo)
                for ax in axes[1]:
                    ax.set_ylim(y_lo - pad, y_hi + pad)
                axes[0, 0].set_ylabel("Share of items")
                axes[1, 0].set_ylabel(f"{oth_label} minus\n{ref_label} accuracy")
                fig.supxlabel(
                    f"Item difficulty: chance {ref_label} answers an item correctly",
                    fontsize=11,
                )
                fig.tight_layout()
            state.add_plot(fig, f"model_{model_analysis.model_name}_item_difficulty")
            made += 1
        return state, "pass" if made else "NA"

    return communicate
