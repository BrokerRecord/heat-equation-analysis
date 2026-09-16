"""
Boundary-condition helpers for the 1D heat equation.
Currently supports Dirichlet (fixed value). Neumann/periodic can be added.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_dirichlet(
    u: NDArray[np.float64], left: float = 0.0, right: float = 0.0
) -> None:
    """Enforce u[0] = left and u[-1] = right in-place."""
    u[0] = left
    u[-1] = right


def apply_neumann(
    u: NDArray[np.float64], dx: float, flux_left: float = 0.0, flux_right: float = 0.0
) -> None:
    """
    Zero-flux (insulated) Neumann BC via ghost-node mirroring:
        u[0]  = u[1]  - dx * flux_left
        u[-1] = u[-2] + dx * flux_right
    """
    u[0] = u[1] - dx * flux_left
    u[-1] = u[-2] + dx * flux_right