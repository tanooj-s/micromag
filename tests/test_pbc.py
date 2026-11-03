"""Iteration-8 checks (D47): periodic boundaries in exchange, demag and grains, and a periodic film's switching field."""

import json

import h5py
import numpy as np
import pytest

import micromag as mm
from micromag.regions import voronoi
from micromag.run import run
from test_demag import demag_field, full_tensor


# ---- exchange ----

def exchange(g, m, Ms, A, device):
    out = np.empty_like(m)
    if device == "cpu":
        mm.exchange_field(g, m, Ms, A, out)
    else:
        zero = np.zeros_like(Ms)
        mm.Problem(g, Ms, A, zero, np.zeros_like(m), (0, 0, 0), device=device, demag=False).field("exchange", m, out)
    return out


@pytest.mark.parametrize("n, pbc, axis, q", [
    ((8, 3, 2), (1, 0, 0), 0, 1),
    ((8, 3, 2), (1, 0, 0), 0, 3),
    ((3, 6, 2), (0, 2, 0), 1, 1),
    ((2, 3, 5), (0, 0, 1), 2, 2),
    ((2, 3, 1), (1, 1, 1), 0, 1),   # n = 2 along the spiral: both neighbours are the same cell; nz = 1 periodic too
], ids=["x", "x-q3", "y", "z", "n2"])
def test_wrapped_exchange_is_exact_on_a_spin_spiral(n, pbc, axis, q, device):
    # m = (cos kx, sin kx, 0) with q whole turns along a periodic axis is an eigenvector of the 6-neighbour Laplacian:
    # H = 2A/(µ0 Ms) · 2 (cos kΔ - 1)/Δ² · m in every cell, edges included (eq. 8, D45).
    d = (2e-9, 3e-9, 4e-9)
    g = mm.Grid(*n, *d, pbc=pbc)
    s = (g.nz, g.ny, g.nx)
    Ms, A = np.full(s, 8e5), np.full(s, 1.3e-11)
    k = 2 * np.pi * q / (n[axis] * d[axis])
    phase = k * (np.indices(s)[2 - axis] + 0.5) * d[axis]
    m = np.stack([np.cos(phase), np.sin(phase), np.zeros(s)])
    expected = 2 * 1.3e-11 / (mm.mu0 * 8e5) * 2 * (np.cos(k * d[axis]) - 1) / d[axis] ** 2 * m
    assert np.linalg.norm(expected) > 0
    H = exchange(g, m, Ms, A, device)
    assert np.abs(H - expected).max() <= 1e-12 * np.abs(expected).max()


# ---- demag ----

def window(n, P):
    """(n, n, 2W+1) weights of the offsets -W..W between target i and source j on one axis (D45): on a periodic axis,
    every d ≡ i - j (mod n) with |d| ≤ W = Pn + n/2, the two ends at weight ½ when n is even; on a free axis, d = i - j."""
    i = np.arange(n)
    if P == 0:
        W = n - 1
        w = np.zeros((n, n, 2 * W + 1))
        w[i[:, None], i[None, :], (i[:, None] - i[None, :]) + W] = 1
        return w, W
    W = P * n + n // 2
    d = np.arange(-W, W + 1)
    w = ((i[:, None, None] - i[None, :, None] - d) % n == 0).astype(float)
    if n % 2 == 0:
        w[..., [0, -1]] *= 0.5
    return w, W


def window_sum(g, m, Ms):
    """eq. 21 as a direct sum over each cell's window of copies, with N(offset) from a free grid large enough to hold
    every offset (validated in D17). Independent of the FFT, the wrap and the periodic tensor."""
    n, P = (g.nx, g.ny, g.nz), g.pbc
    (wx, Wx), (wy, Wy), (wz, Wz) = (window(k, p) for k, p in zip(n, P))
    big = mm.Grid(Wx + 1, Wy + 1, Wz + 1, g.dx, g.dy, g.dz)
    N = full_tensor(mm.demag_tensor(big))  # (3, 3, 2Wz+2, 2Wy+2, 2Wx+2) in wraparound order; a 1-cell axis has size 1
    take = lambda a, W, axis: np.take(a, np.arange(-W, W + 1) % a.shape[axis], axis=axis)
    N = take(take(take(N, Wz, 2), Wy, 3), Wx, 4)
    K = np.einsum("zZo,yYp,xXq,abopq->abzyxZYX", wz, wy, wx, N, optimize=True)
    H = -np.einsum("abzyxZYX,bZYX->azyx", K, Ms * m)
    H[:, Ms == 0] = 0.0
    return H


def random_state(g, seed=1):
    rng = np.random.default_rng(seed)
    s = (g.nz, g.ny, g.nx)
    Ms = rng.uniform(0.5e6, 1.5e6, s)
    Ms[rng.random(s) < 0.2] = 0.0
    return rng.normal(size=(3, *s)), Ms


@pytest.mark.parametrize("n, d, pbc", [
    ((5, 4, 3), (2e-9, 3e-9, 4e-9), (2, 0, 0)),
    ((6, 4, 3), (2e-9, 3e-9, 4e-9), (1, 2, 0)),
    ((5, 4, 3), (2e-9, 2e-9, 2e-9), (1, 1, 1)),
    ((6, 5, 1), (2e-9, 3e-9, 4e-9), (3, 3, 0)),
    ((4, 6, 2), (2e-9, 3e-9, 4e-9), (1, 1, 2)),
], ids=["x-odd", "xy-even", "xyz-cubic", "film", "xyz-even"])
def test_periodic_fft_matches_a_window_direct_sum(n, d, pbc, device):
    g = mm.Grid(*n, *d, pbc=pbc)
    m, Ms = random_state(g)
    direct = window_sum(g, m, Ms)
    free = demag_field(mm.Grid(*n, *d), m, Ms, "cpu")
    assert np.linalg.norm(direct - free) > 1e-3 * np.linalg.norm(free)  # guard: the copies change the field
    H = demag_field(g, m, Ms, device)
    assert np.linalg.norm(H - direct) <= 1e-12 * np.linalg.norm(direct)


def demag_factors(n, d, pbc, device="cpu"):
    """-⟨H⟩/Ms for m uniform along x, y and z: (3,) diagonal factors, and the field spread across the tile."""
    g = mm.Grid(*n, *d, pbc=pbc)
    Ms = np.full((g.nz, g.ny, g.nx), 8e5)
    N, spread = np.empty(3), 0.0
    for a in range(3):
        m = np.zeros((3, *Ms.shape))
        m[a] = 1
        H = demag_field(g, m, Ms, device)[a] / -8e5
        N[a], spread = H.mean(), max(spread, H.max() - H.min())
    return N, spread


def test_periodic_rod_tends_to_half_at_second_order():
    # A square rod periodic along x: N_yy = N_zz -> 1/2 and N_xx -> 0. The copies beyond P act like the far ends of
    # the rod, a field ~ 1/(P L)², so the error falls about 4× per doubling of P.
    Ns = [demag_factors((4, 4, 4), (2e-9,) * 3, (P, 0, 0))[0] for P in (4, 8, 16)]
    for N in Ns:
        assert abs(N[1] - N[2]) <= 1e-12 and abs(N.sum() - 1) <= 1e-5
    err = np.array([0.5 - N[1] for N in Ns])
    assert np.all(err > 0) and np.all((3.3 < err[:-1] / err[1:]) & (err[:-1] / err[1:] < 4.3)), err
    assert Ns[-1][0] < 3e-3


def test_periodic_film_tends_to_one_at_first_order():
    # A film periodic in x and y: N_zz -> 1, N_xx = N_yy -> 0. The copies end at a square plate of side (2P+1) L,
    # whose edge field at the centre is ~ t / ((2P+1) L), so the error falls about 2× per doubling of P.
    Ns = [demag_factors((4, 4, 1), (2e-9,) * 3, (P, P, 0))[0] for P in (4, 8, 16)]
    err = np.array([1 - N[2] for N in Ns])
    assert np.all(err > 0) and np.all((1.7 < err[:-1] / err[1:]) & (err[:-1] / err[1:] < 2.2)), err
    for N in Ns:
        assert abs(N[0] - N[1]) <= 1e-12 and abs(N.sum() - 1) <= 1e-5


def test_periodic_cube_is_a_third_with_a_uniform_field(device):
    # Copies in all three axes fill a cube around every cell: N = 1/3 by symmetry and trace 1, and the field is the
    # same in every cell. A free cube also averages 1/3, but its field is far from uniform.
    N, spread = demag_factors((4, 4, 4), (2e-9,) * 3, (2, 2, 2), device)
    assert np.abs(N - 1 / 3).max() <= 1e-5 and spread <= 1e-10
    _, free_spread = demag_factors((4, 4, 4), (2e-9,) * 3, (0, 0, 0), device)
    assert free_spread > 1e-2


def test_only_periodic_axes_drop_their_padding():
    N = mm.demag_tensor(mm.Grid(5, 4, 3, 2e-9, 3e-9, 4e-9, pbc=(2, 0, 1)))
    assert N.shape == (6, 3, 8, 5)  # z periodic (3), y free (2·4), x periodic (5)


# ---- grains ----

def brute_force_voronoi(points, seeds, period):
    """Owner and boundary distance with the seeds copied over 5^k boxes of the periodic axes: more than enough."""
    shifts = np.stack(np.meshgrid(*[np.arange(-2, 3) if L else [0] for L in period], indexing="ij"), -1).reshape(-1, 3)
    L = np.array([L or 0.0 for L in period])
    images = (seeds[None] + shifts[:, None] * L).reshape(-1, 3)
    d2 = ((points[:, None] - images[None]) ** 2).sum(2)
    near = d2.argmin(1)
    own = images[near]
    plane = (d2 - d2[np.arange(len(points)), near][:, None]) / (2 * np.linalg.norm(images[None] - own[:, None], axis=2) + 1e-300)
    plane[np.arange(len(points)), near] = np.inf
    return near % len(seeds), plane.min(1)


@pytest.mark.parametrize("period", [(16e-9, 0, 0), (16e-9, 12e-9, 0), (16e-9, 12e-9, 8e-9)], ids=["x", "xy", "xyz"])
def test_periodic_voronoi_matches_a_brute_force_oracle(period):
    rng = np.random.default_rng(4)
    box = np.array([16e-9, 12e-9, 8e-9])
    points, seeds = rng.uniform(0, box, (2000, 3)), rng.uniform(0, box, (7, 3))
    owner, dist = voronoi(points, seeds, period=period)
    o, dd = brute_force_voronoi(points, seeds, period)
    assert not np.array_equal(voronoi(points, seeds)[0], o)  # guard: some points belong to a seed across the wrap
    assert np.array_equal(owner, o)
    assert np.allclose(dist, dd, rtol=1e-12, atol=1e-24)


def test_two_seeds_in_a_periodic_slab_have_two_boundary_planes():
    # Seeds at x = 5 and 11 nm in a 16 nm box periodic in x: boundaries at x = 8 nm and across the wrap at x = 0 ≡ 16 nm.
    x = (np.arange(16) + 0.5) * 1e-9
    pts = np.stack(np.broadcast_arrays(x[:, None], 1.5e-9, 1.5e-9), -1).reshape(-1, 3)
    seeds = np.array([[5e-9, 1.3e-9, 2.9e-9], [11e-9, 1.3e-9, 2.9e-9]])
    owner, dist = voronoi(pts, seeds, period=(16e-9, 0, 0))
    assert np.array_equal(owner, (pts[:, 0] > 8e-9).astype(int))
    expected = np.minimum.reduce([np.abs(pts[:, 0] - p) for p in (0, 8e-9, 16e-9)])
    assert np.allclose(dist, expected, rtol=0, atol=1e-21)
    assert np.count_nonzero(dist < 2e-9) == 8  # a 4 nm boundary: 4 layers at x = 8 nm and 4 across the wrap


def test_periodic_grains_follow_their_seeds_around_the_box():
    # Shifting every seed by k cells along a periodic axis (mod L) rolls the grain map by k cells.
    n, dx = (16, 12, 4), 1e-9
    L = np.array(n) * dx
    x, y, z = (a.ravel() for a in np.meshgrid(*[(np.arange(k) + 0.5) * dx for k in n], indexing="ij"))
    pts = np.stack([x, y, z], 1)
    seeds = np.random.default_rng(2).uniform(0, L, (6, 3))
    period = (L[0], L[1], 0)
    owner, dist = (a.reshape(n) for a in voronoi(pts, seeds, period=period))
    shifted = seeds + [5 * dx, 0, 0]
    shifted[:, 0] %= L[0]
    o2, d2 = (a.reshape(n) for a in voronoi(pts, shifted, period=period))
    assert np.array_equal(o2, np.roll(owner, 5, axis=0))
    assert np.allclose(d2, np.roll(dist, 5, axis=0), rtol=0, atol=1e-21)


# ---- the card and the results folder ----

GRAINS_CARD = """
[grid]
n = [12, 10, 4]
cell = [1e-9, 1e-9, 1e-9]
pbc = [2, 1, 0]

[material.hard]
Ms = 1.2e6
A = 8e-12
K = 4e6

[grains]
count = 5
first_id = 1
material = "hard"
seed = 3

[initial]
m = [0, 0, 1]

[[stage]]
type = "minimize"
"""


def test_periodic_card_runs_and_records_its_boundaries(tmp_path):
    card = tmp_path / "card.toml"
    card.write_text(GRAINS_CARD)
    run(card, tmp_path / "run")
    prov = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert prov["pbc"] == [2, 1, 0]
    with h5py.File(tmp_path / "run" / "trajectory.h5") as f:
        assert list(f.attrs["pbc"]) == [2, 1, 0]
        regions = f["regions"][()]
    # the painted grains are the periodic Voronoi cells of the recorded seeds
    x, y, z = np.meshgrid(*[(np.arange(k) + 0.5) * 1e-9 for k in (12, 10, 4)], indexing="ij")
    from micromag.card import load
    from micromag.regions import paint
    seeds = paint(load(card)).seeds
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    o, _ = brute_force_voronoi(pts, seeds, (12e-9, 10e-9, 0))
    assert np.array_equal(regions.transpose(2, 1, 0).ravel(), 1 + o)
    # a restart may change the boundaries, like the cell size
    again = tmp_path / "again.toml"
    again.write_text(GRAINS_CARD.replace("pbc = [2, 1, 0]", "").replace('m = [0, 0, 1]', 'file = "run/trajectory.h5"'))
    run(again, tmp_path / "again")
    assert json.loads((tmp_path / "again" / "provenance.json").read_text())["pbc"] == [0, 0, 0]


# ---- the physics check ----

FILM = """
[grid]
n = [16, 16, 2]
cell = [2e-9, 2e-9, 2e-9]
pbc = {pbc}

[material.perp]
Ms = 8e5
A = 1.3e-11
K = {K!r}

[initial]
m = [0, 0, 1]

[[stage]]
type = "hysteresis"
direction = [{s!r}, 0.0, {c!r}]
mu0_H_from = 0.0
mu0_H_to = -1.5
mu0_H_step = 0.05
refine_to = 1e-4
dump = true
"""


def test_periodic_film_switches_as_a_stoner_wohlfarth_particle(tmp_path):
    # A film periodic in x and y with perpendicular anisotropy (Q = 2K/(µ0 Ms²) = 2.5) is uniform until it switches,
    # so it is one Stoner–Wohlfarth particle with shape anisotropy: H_K,eff = 2K/(µ0 Ms) - (N_zz - N_xx) Ms.
    Ms, K, psi = 8e5, 2.5 * mm.mu0 * 8e5**2 / 2, np.radians(30)
    N, _ = demag_factors((16, 16, 2), (2e-9,) * 3, (10, 10, 0))
    assert N[2] > 0.99 and abs(N[0] - N[1]) <= 1e-12
    BK = mm.mu0 * (2 * K / (mm.mu0 * Ms) - (N[2] - N[0]) * Ms)  # µ0 H_K,eff in T
    h_sw = (np.cos(psi) ** (2 / 3) + np.sin(psi) ** (2 / 3)) ** -1.5

    switch = {}
    for pbc in ("[10, 10, 0]", "[0, 0, 0]"):
        card = tmp_path / f"film{pbc[1]}.toml"
        card.write_text(FILM.format(pbc=pbc, K=K, s=float(np.sin(psi)), c=float(np.cos(psi))))
        out = tmp_path / card.stem
        run(card, out)
        switch[pbc] = json.loads((out / "provenance.json").read_text())["stages"][0]["summary"]["switch_T"]
        if pbc == "[10, 10, 0]":
            with h5py.File(out / "trajectory.h5") as f:
                m, H = f["stage1/m"][()], f["stage1/mu0_H"][()]
            before = H @ [np.sin(psi), 0, np.cos(psi)] > switch[pbc]  # fields not yet past the switch
            dev = np.abs(m[before] - m[before].mean(axis=(2, 3, 4), keepdims=True)).max()
            assert before.sum() > 10 and dev < 1e-3, dev
    assert abs(switch["[10, 10, 0]"] + h_sw * BK) <= 2e-4, (switch, -h_sw * BK)
    assert abs(switch["[0, 0, 0]"] - switch["[10, 10, 0]"]) > 0.05  # free edges switch somewhere else
