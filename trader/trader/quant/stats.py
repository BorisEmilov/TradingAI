"""prompt-fase-cuantitativa.md: statistical harness for the quant-factor battery
-- doesn't exist anywhere else in this codebase (no scipy in requirements.txt,
no bootstrap/permutation/multiple-comparison module). Mirrors the method this
project's earlier ML research line used successfully (moving-block bootstrap +
block-permutation null, multiple block sizes for robustness, BH-FDR as the
primary multiple-comparison correction with Bonferroni reported alongside as
the strict bound) -- see the parent project's memory of that line for context;
none of its code carries over, this is a fresh implementation.

Two different block-size units, on purpose, matching what each function
actually resamples:
- `block_bootstrap_mean_ci`'s `block_size` is in units of EVENTS -- the caller
  passes only the event-time return series (e.g. `full_returns[event_mask]`),
  so "10 consecutive events" is comparable in spirit whether the underlying
  timeframe is M15 or D1.
- The permutation functions' `block_size` is in units of BARS of the
  underlying (trimmed) return series -- it has to be, since what's being
  reshuffled there is the raw bar-by-bar return sequence itself, to preserve
  its own autocorrelation structure. Callers should scale this to each
  factor's own characteristic horizon (e.g. a multiple of its lookback N),
  not reuse one fixed bar count across M15/H1/D1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BLOCK_SIZES = (10, 20, 40)


@dataclass(frozen=True)
class BootstrapResult:
    block_size: int
    mean: float
    ci_lo: float
    ci_hi: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_lo > 0.0 or self.ci_hi < 0.0


@dataclass(frozen=True)
class PermutationResult:
    block_size: int
    observed: float
    p_value: float


def _moving_blocks(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block-bootstrap resample of length n."""
    if block_size >= n:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n]


def block_bootstrap_mean_ci(
    x: np.ndarray, block_size: int, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> BootstrapResult:
    """Moving-block bootstrap CI on the mean of `x` (already the event-aligned
    series, e.g. forward returns at the times the factor fired, in
    chronological order -- NOT the full bar-by-bar series)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = _moving_blocks(n, block_size, rng)
        means[b] = x[idx].mean()
    alpha = 1.0 - ci
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(block_size=block_size, mean=float(x.mean()), ci_lo=float(lo), ci_hi=float(hi))


def _fixed_blocks(arr: np.ndarray, block_size: int) -> list[np.ndarray]:
    """Fixed (non-overlapping) contiguous blocks of `arr` -- the last one is
    naturally ragged (shorter) if block_size doesn't evenly divide len(arr)
    rather than padded/repeated, so concatenating all blocks in ANY order
    always reconstructs exactly len(arr) elements, no truncation needed."""
    n = len(arr)
    return [arr[i : i + block_size] for i in range(0, n, block_size)]


def block_permutation_pvalue_sparse_event(
    full_returns: np.ndarray, event_mask: np.ndarray, block_size: int, n_perm: int = 2000, seed: int = 0
) -> PermutationResult:
    """For a SPARSE event mask (a minority of bars -- mean-reversion z>2 spikes,
    session/weekday buckets). Null: shuffle the FULL forward-return series in
    contiguous blocks (breaks the specific alignment between signal timing and
    realized returns while preserving the return series' own autocorrelation/
    marginal distribution), keep `event_mask` fixed at its original bar
    positions, recompute the event-conditional mean each time. Two-sided
    p-value against |observed|.

    Do NOT use this for a factor that's "on" at nearly every bar (e.g. plain
    momentum, where direction is defined almost everywhere) -- the sum of a
    reordered array over almost-the-whole array is invariant to the
    reordering, so the null collapses to ~the observed value and the test is
    degenerate. Use `block_permutation_pvalue_dense_signal` for those.
    """
    full_returns = np.asarray(full_returns, dtype=float)
    event_mask = np.asarray(event_mask, dtype=bool)
    observed = float(full_returns[event_mask].mean())

    blocks = _fixed_blocks(full_returns, block_size)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for p in range(n_perm):
        block_order = rng.permutation(n_blocks)
        shuffled = np.concatenate([blocks[b] for b in block_order])
        null_stats[p] = shuffled[event_mask].mean()

    p_value = float(np.mean(np.abs(null_stats) >= abs(observed)))
    return PermutationResult(block_size=block_size, observed=observed, p_value=p_value)


def block_permutation_pvalue_dense_signal(
    direction: np.ndarray, forward_returns: np.ndarray, block_size: int, n_perm: int = 2000, seed: int = 0
) -> PermutationResult:
    """For a DENSE, always-on signal (e.g. momentum direction defined at nearly
    every bar). Null: keep `forward_returns` fixed, shuffle `direction` in
    contiguous blocks -- breaks the alignment between which way the signal
    pointed and which future return actually followed, while preserving each
    series' own autocorrelation. Two-sided p-value against |observed| mean of
    direction * forward_returns.
    """
    direction = np.asarray(direction, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    observed = float((direction * forward_returns).mean())

    blocks = _fixed_blocks(direction, block_size)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for p in range(n_perm):
        block_order = rng.permutation(n_blocks)
        shuffled_dir = np.concatenate([blocks[b] for b in block_order])
        null_stats[p] = (shuffled_dir * forward_returns).mean()

    p_value = float(np.mean(np.abs(null_stats) >= abs(observed)))
    return PermutationResult(block_size=block_size, observed=observed, p_value=p_value)


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Standard BH step-up procedure. Returns a boolean mask (same order as
    input) of which hypotheses are rejected (survive FDR control at level q)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    below = sorted_p <= thresholds
    if not below.any():
        return [False] * n
    max_rank = np.max(np.where(below)[0])
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[: max_rank + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    if n == 0:
        return []
    threshold = alpha / n
    return [p <= threshold for p in p_values]
