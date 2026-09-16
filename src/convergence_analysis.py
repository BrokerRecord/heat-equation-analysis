"""
Convergence study for the FTCS scheme.

Refines Δx (and Δt proportionally, keeping r fixed) and computes L∞ and L2
errors against the analytical solution. Verifies the expected orders:
    - O(Δt)  temporal
    - O(Δx²) spatial
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from heat_equation import (
    analytical_solution,
    ftcs_solver,
    initial_condition_sine,
)
from utils import l2_error, linf_error, save_figure


def run_spatial_refinement(
    Nx_list=(21, 41, 81, 161, 321),
    alpha: float = 0.01,
    L: float = 1.0,
    T: float = 0.1,
    r_fixed: float = 0.4,
    n_modes: int = 400,
):
    """Refine Nx while keeping r fixed (so Δt ~ Δx²)."""
    dx_list, dt_list, e_inf, e_l2 = [], [], [], []
    u0_ref = None
    x_ref = None

    for Nx in Nx_list:
        x = np.linspace(0.0, L, Nx)
        dx = x[1] - x[0]
        dt = r_fixed * dx**2 / alpha
        n_steps = int(round(T / dt))
        dt = T / n_steps  # adjust to land exactly on T

        u0 = initial_condition_sine(x, L)
        u_num = ftcs_solver(u0, alpha, dx, dt, n_steps)[-1]
        u_exact = analytical_solution(x, T, alpha, L, u0, n_modes=n_modes)

        dx_list.append(dx)
        dt_list.append(dt)
        e_inf.append(linf_error(u_num, u_exact))
        e_l2.append(l2_error(u_num, u_exact, dx))

    return np.array(dx_list), np.array(dt_list), np.array(e_inf), np.array(e_l2)


def estimate_order(h, err):
    """Least-squares slope on log-log data."""
    p = np.polyfit(np.log(h), np.log(err), 1)
    return p[0]


def main():
    dx, dt, e_inf, e_l2 = run_spatial_refinement()

    print("Spatial refinement study")
    print(f"{'Δx':>10} {'Δt':>12} {'L∞ error':>14} {'L2 error':>14}")
    for d, t, ei, el in zip(dx, dt, e_inf, e_l2):
        print(f"{d:10.5f} {t:12.3e} {ei:14.3e} {el:14.3e}")

    p_inf = estimate_order(dx, e_inf)
    p_l2 = estimate_order(dx, e_l2)
    print(f"\nEstimated order (L∞): {p_inf:.2f}")
    print(f"Estimated order (L2): {p_l2:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dx, e_inf, "o-", label=f"L∞ (slope ≈ {p_inf:.2f})")
    ax.loglog(dx, e_l2, "s-", label=f"L2 (slope ≈ {p_l2:.2f})")
    # reference O(Δx²) line
    ref = e_l2[-1] * (dx / dx[-1]) ** 2
    ax.loglog(dx, ref, "k--", lw=1, label="O(Δx²) reference")
    ax.set_xlabel("Δx")
    ax.set_ylabel("Error")
    ax.set_title("Convergence of FTCS vs analytical solution")
    ax.legend(), ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "figures/convergence_plot.png")


if __name__ == "__main__":
    main()