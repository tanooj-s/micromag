import numpy as np

from ._core import LLG, Demag, Grid, Problem, anisotropy_field, demag_tensor, exchange_field, has_cuda, mu0, scale, zeeman_field

__all__ = ["LLG", "Demag", "Grid", "Problem", "anisotropy_field", "demag_tensor", "energy", "exchange_field", "has_cuda", "mu0", "scale", "zeeman_field"]


def energy(grid, m, Ms, H, c):
    """Energy (J) of one term from its field: E = -c·µ0·dV·Σ Ms (m·H).

    c = 1/2 for terms quadratic in m (exchange, anisotropy), 1 for linear ones (Zeeman). See D10.
    """
    # ponytail: numpy reduction on the host; move into C++ when the C++ minimizer needs energy
    return -c * mu0 * grid.dx * grid.dy * grid.dz * np.sum(Ms * np.sum(m * H, axis=0))
