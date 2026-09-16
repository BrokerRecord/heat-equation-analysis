"""Shared utilities: error norms, plotting, saving."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------
def linf_error(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """Maximum absolute difference."""
    return float(np.max(np.abs(u - v)))


def l2_error(
    u: NDArray[np.float64], v: NDArray[np.float64], dx: float
) -> float:
    """Discrete L2 norm: sqrt(dx * Σ (u_i - v_i)²)."""
    return float(np.sqrt(dx * np.sum((u - v) ** 2)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_solution(
    x: NDArray[np.float64],
    u_num: NDArray[np.float64],
    u_exact: NDArray[np.float64] | None = None,
    title: str = "Solution",
    ax: plt.Axes | None = None,
):
    """Plot numerical and (optional) exact solution."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if u_exact is not None:
        ax.plot(x, u_exact, "b-", lw=2, label="Analytical")
    ax.plot(x, u_num, "r--", lw=2, label="Numerical")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """Save figure and ensure directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")