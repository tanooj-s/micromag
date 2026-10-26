import numpy as np

from ._core import LLG, Demag, Grid, Minimizer, Problem, anisotropy_field, demag_tensor, exchange_field, has_cuda, kB, mu0, philox, scale, zeeman_field

__all__ = ["LLG", "Demag", "Grid", "Minimizer", "Problem", "anisotropy_field", "demag_tensor", "energy", "exchange_field", "has_cuda", "hysteresis", "kB", "mu0", "philox", "run", "scale", "zeeman_field"]


def energy(grid, m, Ms, H, c):
    """Energy (J) of one term from its field: E = -c·µ0·dV·Σ Ms (m·H).

    c = 1/2 for terms quadratic in m (exchange, anisotropy), 1 for linear ones (Zeeman). See D10.
    """
    # ponytail: numpy reduction on the host; move into C++ when the C++ minimizer needs energy
    return -c * mu0 * grid.dx * grid.dy * grid.dz * np.sum(Ms * np.sum(m * H, axis=0))


def hysteresis(problem, m, fields, minimizer):
    """Minimize at each applied field in turn, starting from m (D22).

    fields: (n, 3) in A/m. Returns a flat dict of arrays, ready for np.savez: H (n, 3), m (n, 3) = <m> over
    magnetic cells, the four partial energies in J (n,), and iterations (n,). The final state is left in m.
    """
    fields = np.asarray(fields, dtype=float)
    magnetic = problem.Ms > 0
    mean_m, energies, iterations = [], [], []
    for H in fields:
        problem.H_ext = H
        iterations.append(minimizer.minimize(m))
        mean_m.append(m[:, magnetic].mean(axis=1))
        energies.append(problem.energies(m))
    return {
        "H": fields.copy(),
        "m": np.array(mean_m),
        **{term: np.array([E[term] for E in energies]) for term in energies[0]},
        "iterations": np.array(iterations),
    }


from .run import run  # noqa: E402  (last: run.py imports the names above)
