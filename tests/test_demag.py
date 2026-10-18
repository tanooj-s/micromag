import itertools
import math

import mpmath as mp
import numpy as np
import pytest

import micromag as mm

COMPONENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]  # order of demag_tensor's first axis


def full_tensor(N6):
    """(6, ...) -> (3, 3, ...), using N's symmetry."""
    N = np.empty((3, 3, *N6.shape[1:]))
    for c, (a, b) in enumerate(COMPONENTS):
        N[a, b] = N[b, a] = N6[c]
    return N


def dipole(r, V):
    """Point-dipole tensor with the convention H = -N M."""
    R = np.linalg.norm(r)
    return -(3 * np.outer(r, r) / R**2 - np.eye(3)) * V / (4 * np.pi * R**3)


def aharoni_z(a, b, c):
    """Demag factor along z of a uniformly magnetised prism with half-sides a, b, c (Aharoni, JAP 83, 3432 (1998))."""
    r = math.sqrt(a * a + b * b + c * c)
    ab, bc, ac = math.hypot(a, b), math.hypot(b, c), math.hypot(a, c)
    return (
        (b * b - c * c) / (2 * b * c) * math.log((r - a) / (r + a))
        + (a * a - c * c) / (2 * a * c) * math.log((r - b) / (r + b))
        + b / (2 * c) * math.log((ab + a) / (ab - a))
        + a / (2 * c) * math.log((ab + b) / (ab - b))
        + c / (2 * a) * math.log((bc - b) / (bc + b))
        + c / (2 * b) * math.log((ac - a) / (ac + a))
        + 2 * math.atan(a * b / (c * r))
        + (a**3 + b**3 - 2 * c**3) / (3 * a * b * c)
        + (a * a + b * b - 2 * c * c) / (3 * a * b * c) * r
        + c / (a * b) * (ac + bc)
        - (ab**3 + bc**3 + ac**3) / (3 * a * b * c)
    ) / math.pi


@pytest.mark.parametrize(
    "n, d",
    [((8, 8, 8), (1, 1, 1)), ((1, 1, 1), (1, 2, 3)), ((32, 32, 1), (1, 1, 1))],
    ids=["cube", "single-1:2:3-cell", "thin-film"],
)
def test_uniform_prism_matches_aharoni(n, d):
    g = mm.Grid(*n, *(1e-9 * np.array(d)))
    half = [ni * di / 2 for ni, di in zip(n, d)]
    expected = [aharoni_z(half[1], half[2], half[0]), aharoni_z(half[2], half[0], half[1]), aharoni_z(*half)]
    demag, Ms = mm.Demag(g), np.full((g.nz, g.ny, g.nx), 8e5)
    for a in range(3):
        m = np.zeros((3, g.nz, g.ny, g.nx))
        m[a] = 1.0
        H = np.empty_like(m)
        demag.field(m, Ms, H)
        assert np.isclose(-H[a].mean() / 8e5, expected[a], rtol=1e-10, atol=0)
        for b in set(range(3)) - {a}:  # a symmetric prism has no off-diagonal body tensor
            assert abs(H[b].mean()) <= 1e-12 * 8e5


def test_far_field_approaches_point_dipole():
    # Non-cubic cells: for cubes the leading correction cancels and the error falls like 1/r⁴, not 1/r² (D17).
    d = 1e-9 * np.array([1.0, 1.5, 2.0])
    g = mm.Grid(32, 32, 32, *d)
    N = full_tensor(mm.demag_tensor(g))
    for sign in itertools.product([1, -1], repeat=3):  # every octant: checks the mirroring signs
        errs = []
        for off in [(10, 7, 3), (20, 14, 6)]:
            i, j, k = np.multiply(sign, off)
            exact = dipole(np.array([i, j, k]) * d, np.prod(d))
            errs.append(np.linalg.norm(N[:, :, k, j, i] - exact) / np.linalg.norm(exact))
        assert 3.5 < errs[0] / errs[1] < 4.5, errs


def newell_mp(X, Y, Z, dx, dy, dz):
    """All 6 Newell components at 50 digits: the same formulas as the C++ near field, without the cancellation."""
    mp.mp.dps = 50

    def f(x, y, z):
        R, r = mp.sqrt(x * x + y * y + z * z), mp.mpf(0)
        if y and z * z != x * x:
            r += y / 2 * (z * z - x * x) * mp.asinh(y / mp.sqrt(x * x + z * z))
        if z and y * y != x * x:
            r += z / 2 * (y * y - x * x) * mp.asinh(z / mp.sqrt(x * x + y * y))
        if x and y and z:
            r -= x * y * z * mp.atan(y * z / (x * R))
        return r + (2 * x * x - y * y - z * z) * R / 6

    def g(x, y, z):
        R, r = mp.sqrt(x * x + y * y + z * z), mp.mpf(0)
        if x and y and z:
            r += x * y * z * mp.asinh(z / mp.sqrt(x * x + y * y))
        if y and 3 * z * z != y * y:
            r += y / 6 * (3 * z * z - y * y) * mp.asinh(x / mp.sqrt(y * y + z * z))
        if x and 3 * z * z != x * x:
            r += x / 6 * (3 * z * z - x * x) * mp.asinh(y / mp.sqrt(x * x + z * z))
        if z:
            r -= z**3 / 6 * mp.atan(x * y / (z * R))
        if z and y:
            r -= z * y * y / 2 * mp.atan(x * z / (y * R))
        if z and x:
            r -= z * x * x / 2 * mp.atan(y * z / (x * R))
        return r - x * y * R / 3

    def newell(F, X, Y, Z, dx, dy, dz):
        c = {-1: -1, 0: 2, 1: -1}
        s = sum(c[i] * c[j] * c[k] * F(X + i * dx, Y + j * dy, Z + k * dz) for i in c for j in c for k in c)
        return s / (4 * mp.pi * dx * dy * dz)

    X, Y, Z, dx, dy, dz = map(mp.mpf, (X, Y, Z, dx, dy, dz))
    return np.array([float(v) for v in (
        newell(f, X, Y, Z, dx, dy, dz), newell(f, Y, X, Z, dy, dx, dz), newell(f, Z, Y, X, dz, dy, dx),
        newell(g, X, Y, Z, dx, dy, dz), newell(g, X, Z, Y, dx, dz, dy), newell(g, Y, Z, X, dy, dz, dx),
    )])


@pytest.mark.parametrize("d", [(1, 1, 1), (1, 1.5, 2), (1, 1, 0.25)], ids=["cubic", "1:1.5:2", "1:1:0.25"])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 0, 1), (1, 0.7, 0.3)], ids=["x", "z", "oblique"])
def test_tensor_matches_mpmath(d, u):
    # D14: Newell below 20 cells, Gauss-averaged dipole beyond; 1e-6 relative to the largest entry at each offset.
    radii = [0, 1, 2, 5, 10, 19, 21, 30, 60, 127]
    offs = [np.round(R * np.array(u) / np.linalg.norm(u)).astype(int) for R in radii]
    i_max, j_max, k_max = offs[-1]
    g = mm.Grid(i_max + 1, j_max + 1, k_max + 1, *d)  # unitless cells: N is scale-free
    N = mm.demag_tensor(g)
    for i, j, k in offs:
        exact = newell_mp(i * d[0], j * d[1], k * d[2], *d)
        err = np.abs(N[:, k, j, i] - exact).max() / np.abs(exact).max()
        assert err <= 1e-6, ((i, j, k), err)


@pytest.mark.parametrize("n", [(5, 4, 3), (6, 5, 1)], ids=["odd-3d", "single-layer"])
def test_fft_matches_direct_sum(n):
    rng = np.random.default_rng(1)
    g = mm.Grid(*n, 2e-9, 3e-9, 4e-9)
    s = (g.nz, g.ny, g.nx)
    m = rng.normal(size=(3, *s))
    Ms = rng.uniform(0.5e6, 1.5e6, s)
    Ms[rng.random(s) < 0.2] = 0.0

    # eq. 21 as an O(N²) sum: H_i = -Σ_j N(r_i - r_j) Ms_j m_j. Negative offsets index the wraparound layout directly.
    z, y, x = (a.ravel() for a in np.indices(s))
    N = full_tensor(mm.demag_tensor(g))[:, :, z[:, None] - z, y[:, None] - y, x[:, None] - x]
    direct = -np.einsum("abij,bj->ai", N, (Ms * m).reshape(3, -1)).reshape(m.shape)
    direct[:, Ms == 0] = 0.0
    assert np.linalg.norm(direct) > 0  # an all-zero field would otherwise pass trivially

    H = np.empty_like(m)
    mm.Demag(g).field(m, Ms, H)
    assert np.linalg.norm(H - direct) <= 1e-12 * np.linalg.norm(direct)
