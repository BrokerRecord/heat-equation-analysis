"""
Core solvers for the 1D heat equation.

PDE:      ∂u/∂t = α ∂²u/∂x²,   x ∈ (0, L),  t ∈ (0, T]
BCs:      u(0,t) = u(L,t) = 0
IC:       u(x,0) = u0(x)

Analytical solution (separation of variables):
    u(x,t) = Σ_{n=1}^{∞} B_n sin(nπx/L) exp(-α (nπ/L)² t)
    B_n    = (2/L) ∫_0^L u0(x) sin(nπx/L) dx

Numerical scheme (FTCS):
    u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2 u_i^n + u_{i-1}^n),  r = α Δt / Δx²
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def initial_condition_sine(
    x: NDArray[np.float64], L: float = 1.0, amplitude: float = 1.0
) -> NDArray[np.float64]:
    """Single-mode sine: u0(x) = A sin(πx/L)."""
    return amplitude * np.sin(np.pi * x / L)


def initial_condition_step(
    x: NDArray[np.float64], L: float = 1.0, height: float = 1.0
) -> NDArray[np.float64]:
    """Square pulse on the middle 50% of the domain, zero at the boundaries."""
    u0 = np.zeros_like(x)
    u0[(x > 0.25 * L) & (x < 0.75 * L)] = height
    return u0


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------
def _fourier_coefficients(
    u0: NDArray[np.float64],
    x: NDArray[np.float64],
    L: float,
    n_modes: int,
) -> NDArray[np.float64]:
    """Numerically integrate B_n = (2/L) ∫ u0(x) sin(nπx/L) dx via trapezoid."""
    B = np.zeros(n_modes)
    for n in range(1, n_modes + 1):
        integrand = u0 * np.sin(n * np.pi * x / L)
        B[n - 1] = (2.0 / L) * np.trapezoid(integrand, x)
    return B


def analytical_solution(
    x: NDArray[np.float64],
    t: float,
    alpha: float,
    L: float,
    u0: NDArray[np.float64],
    n_modes: int = 200,
) -> NDArray[np.float64]:
    """
    Fourier-series analytical solution evaluated at time t.

    Parameters
    ----------
    x : spatial grid (must match the grid used to sample u0)
    t : evaluation time
    alpha : thermal diffusivity
    L : domain length
    u0 : initial temperature on x
    n_modes : number of Fourier modes retained

    Returns
    -------
    u(x, t) as an array with the same shape as x.
    """
    B = _fourier_coefficients(u0, x, L, n_modes)
    u = np.zeros_like(x)
    for n in range(1, n_modes + 1):
        lam = (n * np.pi / L) ** 2
        u += B[n - 1] * np.sin(n * np.pi * x / L) * np.exp(-alpha * lam * t)
    return u


# ---------------------------------------------------------------------------
# Numerical FTCS solver
# ---------------------------------------------------------------------------
def ftcs_solver(
    u0: NDArray[np.float64],
    alpha: float,
    dx: float,
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """
    Explicit Forward-Time Centered-Space solver.

    Parameters
    ----------
    u0 : initial condition on the interior grid (boundaries excluded)
    alpha : thermal diffusivity
    dx : spatial step
    dt : time step
    n_steps : number of time steps to take

    Returns
    -------
    History array of shape (n_steps + 1, len(u0)).
    """
    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError(
            f"FTCS is unstable for r = {r:.3f} > 0.5. Reduce dt or increase dx."
        )

    u = u0.copy()
    history = np.empty((n_steps + 1, u.size), dtype=np.float64)
    history[0] = u

    for n in range(n_steps):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        # Dirichlet BC: boundaries stay at 0
        u[0] = 0.0
        u[-1] = 0.0
        history[n + 1] = u

    return history


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    alpha, L, T = 0.01, 1.0, 0.5
    Nx = 101
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]

    # Stable time step: r = 0.4
    r_target = 0.4
    dt = r_target * dx**2 / alpha
    n_steps = int(round(T / dt))

    u0 = initial_condition_sine(x, L)

    # Numerical
    hist = ftcs_solver(u0, alpha, dx, dt, n_steps)
    u_num = hist[-1]

    # Analytical
    u_exact = analytical_solution(x, T, alpha, L, u0, n_modes=200)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, u0, "k--", label="IC (t=0)")
    plt.plot(x, u_exact, "b-", lw=2, label="Analytical")
    plt.plot(x, u_num, "r--", lw=2, label="FTCS numerical")
    plt.xlabel("x"), plt.ylabel("u(x, t)")
    plt.title(f"Analytical vs Numerical @ t={T}")
    plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
    plt.savefig("figures/analytical_vs_numerical.png", dpi=300)
    plt.show()

    print(f"L∞ error = {np.max(np.abs(u_num - u_exact)):.3e}")