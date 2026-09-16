"""Unit tests for the heat-equation solvers."""
import numpy as np
import pytest

from src.heat_equation import (
    analytical_solution,
    ftcs_solver,
    initial_condition_sine,
)
from src.utils import linf_error


def test_single_mode_decay():
    """For u0 = sin(πx/L), analytical decay is exp(-α π² t / L²)."""
    alpha, L, T = 0.01, 1.0, 0.2
    Nx = 101
    x = np.linspace(0.0, L, Nx)
    u0 = initial_condition_sine(x, L)
    u_exact = analytical_solution(x, T, alpha, L, u0, n_modes=50)
    expected = np.sin(np.pi * x / L) * np.exp(-alpha * np.pi**2 * T / L**2)
    assert linf_error(u_exact, expected) < 1e-10


def test_ftcs_matches_analytical_sine():
    """FTCS at stable r should match analytical solution to ~1e-4."""
    alpha, L, T = 0.01, 1.0, 0.1
    Nx = 101
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    r = 0.4
    dt = r * dx**2 / alpha
    n_steps = int(round(T / dt))
    dt = T / n_steps

    u0 = initial_condition_sine(x, L)
    u_num = ftcs_solver(u0, alpha, dx, dt, n_steps)[-1]
    u_exact = analytical_solution(x, T, alpha, L, u0, n_modes=200)

    assert linf_error(u_num, u_exact) < 1e-4


def test_ftcs_raises_when_unstable():
    """r > 0.5 must raise."""
    x = np.linspace(0, 1, 11)
    u0 = np.sin(np.pi * x)
    with pytest.raises(ValueError):
        ftcs_solver(u0, alpha=1.0, dx=0.1, dt=0.01, n_steps=1)


def test_boundary_conditions_preserved():
    """Dirichlet BCs remain zero throughout the simulation."""
    alpha, L = 0.01, 1.0
    Nx = 51
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    dt = 0.4 * dx**2 / alpha
    u0 = initial_condition_sine(x, L)
    hist = ftcs_solver(u0, alpha, dx, dt, n_steps=50)
    assert np.allclose(hist[:, 0], 0.0)
    assert np.allclose(hist[:, -1], 0.0)