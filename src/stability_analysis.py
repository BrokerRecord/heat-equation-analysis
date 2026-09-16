"""
Von Neumann stability sweep for the FTCS scheme.

The amplification factor is
    G(θ) = 1 - 4r sin²(θ/2),   r = α Δt / Δx²
so |G| ≤ 1 for all θ  ⟺  r ≤ 1/2.

This script numerically confirms the threshold by running the solver for
several values of r and plotting the resulting max-amplitude growth.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from heat_equation import ftcs_solver, initial_condition_sine
from utils import save_figure


def run_stability_sweep(
    r_values=(0.25, 0.49, 0.5, 0.51, 0.6, 0.9),
    alpha: float = 0.01,
    L: float = 1.0,
    Nx: int = 101,
    n_steps: int = 200,
):
    """Return dict {r: max|u| over time}."""
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    u0 = initial_condition_sine(x, L)

    results = {}
    for r in r_values:
        dt = r * dx**2 / alpha
        try:
            hist = ftcs_solver(u0, alpha, dx, dt, n_steps)
            # max over interior points (drop BC nodes)
            results[r] = np.max(np.abs(hist[:, 1:-1]))
        except ValueError as e:
            print(f"[skip] r={r}: {e}")
    return results


def main():
    results = run_stability_sweep()

    fig, ax = plt.subplots(figsize=(8, 5))
    rs = list(results.keys())
    amps = [results[r] for r in rs]
    colors = ["tab:green" if r <= 0.5 else "tab:red" for r in rs]
    ax.bar([str(r) for r in rs], amps, color=colors, edgecolor="k")
    ax.axvline(x=1.5, color="k", ls="--", lw=1, label="r = 0.5 (stability limit)")
    ax.set_xlabel("r = α Δt / Δx²")
    ax.set_ylabel("max |u| after 200 steps")
    ax.set_title("FTCS stability: amplitude growth vs r")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, "figures/stability_comparison.png")

    for r, a in results.items():
        print(f"r = {r:>4}  →  max|u| = {a:.3e}")


if __name__ == "__main__":
    main()