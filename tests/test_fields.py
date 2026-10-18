import numpy as np
import pytest

import micromag as mm

H_EXT = np.array([1e5, -2e5, 3e5])
C = {"exchange": 0.5, "anisotropy": 0.5, "zeeman": 1.0, "demag": 0.5}  # energy prefactor per term (D10)


def random_problem(seed=0):
    """Small grid with unequal cell sizes, mixed materials and some vacuum cells."""
    rng = np.random.default_rng(seed)
    g = mm.Grid(5, 4, 3, 2e-9, 3e-9, 4e-9)
    s = (g.nz, g.ny, g.nx)
    m = rng.normal(size=(3, *s))
    m /= np.linalg.norm(m, axis=0)
    k = rng.normal(size=(3, *s))
    k /= np.linalg.norm(k, axis=0)
    Ms = rng.uniform(0.5e6, 1.5e6, s)
    Ms[rng.random(s) < 0.2] = 0.0
    assert (Ms == 0).any()
    A = rng.uniform(5e-12, 15e-12, s)
    K = rng.uniform(1e6, 5e6, s)
    return dict(grid=g, m=m, Ms=Ms, A=A, K=K, k=k, demag=mm.Demag(g))


def field(term, p, m):
    out = np.empty_like(m)
    if term == "exchange":
        mm.exchange_field(p["grid"], m, p["Ms"], p["A"], out)
    elif term == "anisotropy":
        mm.anisotropy_field(p["grid"], m, p["Ms"], p["K"], p["k"], out)
    elif term == "demag":
        p["demag"].field(m, p["Ms"], out)
    else:
        mm.zeeman_field(p["grid"], p["Ms"], H_EXT, out)
    return out


def energy(term, p, m):
    return mm.energy(p["grid"], m, p["Ms"], field(term, p, m), C[term])


def test_uniform_state_energies():
    p = random_problem()
    g, Ms = p["grid"], p["Ms"]
    dV = g.dx * g.dy * g.dz
    u = np.empty_like(p["m"])
    u[:] = np.array([0.6, 0.0, 0.8])[:, None, None, None]
    # atol=0 everywhere: energies here are ~1e-20 J, far below numpy's default atol
    assert energy("exchange", p, u) == 0.0  # m_j - m_i is exactly 0, even with mixed A
    assert np.isclose(energy("anisotropy", p, p["k"].copy()), -np.sum(p["K"][Ms > 0]) * dV, rtol=1e-12, atol=0)
    assert np.isclose(energy("zeeman", p, u), -mm.mu0 * dV * np.sum(Ms) * (u[:, 0, 0, 0] @ H_EXT), rtol=1e-12, atol=0)


@pytest.mark.parametrize("term", C)
def test_field_is_minus_energy_gradient(term):
    # H = -1/(µ0 Ms dV) ∂E/∂m. E is quadratic (exchange, anisotropy) or linear (Zeeman) in m,
    # so central differences are exact up to rounding. Passing also proves the exchange operator is symmetric.
    p = random_problem()
    g, m = p["grid"], p["m"]
    h = 1e-3
    grad = np.empty_like(m)
    for i in np.ndindex(m.shape):
        mp, mn = m.copy(), m.copy()
        mp[i] += h
        mn[i] -= h
        grad[i] = (energy(term, p, mp) - energy(term, p, mn)) / (2 * h)
    expected = -mm.mu0 * p["Ms"] * g.dx * g.dy * g.dz * field(term, p, m)
    assert np.linalg.norm(expected) > 0  # an all-zero field would otherwise pass trivially
    assert np.linalg.norm(grad - expected) <= 1e-7 * np.linalg.norm(expected)


@pytest.mark.parametrize("term", C)
def test_vacuum_cells_have_zero_field(term):
    p = random_problem()
    H = field(term, p, p["m"])
    assert np.all(H[:, p["Ms"] == 0] == 0)


A_W, K_W, MS_W = 7.7e-12, 4.3e6, 1.28e6  # NdFeB-like
DELTA = np.sqrt(A_W / K_W)  # wall width is π·δ0


def bloch_errors(cells_per_delta):
    """Place the exact Bloch wall on a 40·δ0 line; return (max torque / H_K, relative wall-energy error)."""
    dx = DELTA / cells_per_delta
    nx = 40 * cells_per_delta
    g = mm.Grid(nx, 1, 1, dx, dx, dx)
    x = (np.arange(nx) + 0.5 - nx / 2) * dx
    theta = 2 * np.arctan(np.exp(x / DELTA))
    m = np.zeros((3, 1, 1, nx))
    m[1, 0, 0], m[2, 0, 0] = np.sin(theta), np.cos(theta)  # rotates in the y-z plane, from +z to -z
    Ms, A, K = (np.full((1, 1, nx), v) for v in (MS_W, A_W, K_W))
    k = np.zeros_like(m)
    k[2] = 1.0
    H_ex, H_an = np.empty_like(m), np.empty_like(m)
    mm.exchange_field(g, m, Ms, A, H_ex)
    mm.anisotropy_field(g, m, Ms, K, k, H_an)

    H_K = 2 * K_W / (mm.mu0 * MS_W)
    torque = np.linalg.norm(np.cross(m, H_ex + H_an, axis=0), axis=0).max() / H_K
    area = dx * dx
    E = mm.energy(g, m, Ms, H_ex, 0.5) + mm.energy(g, m, Ms, H_an, 0.5)
    sigma = (E + K_W * nx * dx * area) / area  # relative to the uniform state, energy -K·V
    return torque, abs(sigma / (4 * np.sqrt(A_W * K_W)) - 1)


def test_bloch_wall_converges_at_second_order():
    # D13: torque -> 0 checks the width π√(A/K); σ -> 4√(AK) checks the energy.
    errs = np.array([bloch_errors(n) for n in (4, 8, 16)])  # rows: resolution; cols: torque, energy
    ratios = errs[:-1] / errs[1:]
    assert np.all((3.5 < ratios) & (ratios < 4.5)), ratios
    assert np.all(errs[-1] < 2e-3), errs


def test_binding_rejects_mismatched_arrays():
    p = random_problem()
    g = p["grid"]
    out = np.empty_like(p["m"])
    with pytest.raises(ValueError):  # axes in (x, y, z) order instead of (z, y, x)
        mm.exchange_field(g, np.empty((3, g.nx, g.ny, g.nz)), p["Ms"], p["A"], out)
    with pytest.raises(ValueError):
        p["demag"].field(np.empty((3, g.nx, g.ny, g.nz)), p["Ms"], out)
    with pytest.raises(TypeError):  # implicit conversion would write H into a hidden copy
        mm.exchange_field(g, p["m"], p["Ms"], p["A"], out.astype(np.float32))
