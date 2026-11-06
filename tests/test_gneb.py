"""Iteration-9 checks (D51): energy barriers by geodesic NEB with a climbing image."""

import csv
import json

import h5py
import numpy as np
import pytest

import micromag as mm
from micromag.gneb import distance, gneb, great_circle, tangents
from micromag.run import run

Ms, K, A = 8e5, 5e5, 1e-11
HK = 2 * K / (mm.mu0 * Ms)  # anisotropy field, A/m


def macrospin(device, H=(0.0, 0.0, 0.0), d=5e-9):
    g = mm.Grid(1, 1, 1, d, d, d)
    s = (1, 1, 1)
    k = np.zeros((3, *s))
    k[2] = 1
    return g, mm.Problem(g, np.full(s, Ms), np.full(s, A), np.full(s, K), k, H, device=device, demag=False)


def spin(theta, phi=0.0):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).reshape(3, 1, 1, 1)


def spins(*thetas):
    return np.concatenate([spin(t) for t in thetas], axis=3)


# ---- geometry ----

def test_distance_is_the_ms_weighted_rms_great_circle_angle():
    # Two cells rotated by 0.3 and 0.9 rad, the second with three times the moment: L² = ¼·0.3² + ¾·0.9².
    a, b = spins(0.0, 0.2), spins(0.3, 1.1)
    w = np.array([0.25, 0.75]).reshape(1, 1, 2)
    assert np.isclose(distance(a, b, w), np.sqrt(0.25 * 0.3**2 + 0.75 * 0.9**2), rtol=1e-12)
    assert np.isclose(distance(spin(0.0), spin(np.pi), np.ones((1, 1, 1))), np.pi, rtol=1e-12)  # the far end stays exact


def test_great_circle_is_evenly_spaced_on_the_sphere():
    a, b = spin(0.4, 0.3), spin(2.0, 1.7)
    path = great_circle(a, b, 7)
    w = np.ones((1, 1, 1))
    gaps = [distance(path[i], path[i + 1], w) for i in range(6)]
    assert np.allclose(path[0], a) and np.allclose(path[-1], b)
    assert np.allclose(np.linalg.norm(path, axis=1), 1, rtol=1e-14, atol=0)
    assert gaps[0] > 0 and np.allclose(gaps, distance(a, b, w) / 6, rtol=1e-12)


def test_great_circle_between_antiparallel_states_is_still_a_half_turn():
    a = np.zeros((3, 1, 1, 2))
    a[2] = 1
    a[0, 0, 0, 1], a[2, 0, 0, 1] = 1, 0  # also a cell along x, where the fixed axis choice must change
    path = great_circle(a, -a, 5)
    assert np.all(np.isfinite(path))
    assert np.allclose(np.linalg.norm(path, axis=1), 1, rtol=1e-14, atol=0)
    mid = path[2]
    assert np.allclose((mid * a).sum(axis=0), 0, atol=1e-12)  # a quarter turn away from each end


def test_tangent_points_to_the_higher_neighbour_and_blends_at_a_peak():
    # Uneven spacing, so the two differences have different directions after projection.
    th = np.array([0.0, 0.2, 0.9, 1.1, 2.0])
    m = np.stack([spin(t) for t in th])
    w = np.ones((1, 1, 1))

    def expected(i, forward, back):
        v = forward * (m[i + 1] - m[i]) + back * (m[i] - m[i - 1])
        v = v - (v * m[i]).sum(axis=0) * m[i]
        return v / np.linalg.norm(v)

    rising = tangents(m, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), w)
    falling = tangents(m, np.array([4.0, 3.0, 2.0, 1.0, 0.0]), w)
    peak = tangents(m, np.array([0.0, 1.0, 5.0, 2.0, 0.0]), w)   # at i = 2: ΔE₊ = 3, ΔE₋ = 4, the left neighbour is higher
    assert np.allclose(rising[2], expected(2, 1, 0)) and np.allclose(falling[2], expected(2, 0, 1))
    assert np.allclose(peak[2], expected(2, 3, 4))
    assert np.allclose((rising[1:-1] * m[1:-1]).sum(axis=1), 0, atol=1e-14)  # in the tangent plane
    assert not rising[0].any() and not rising[-1].any()


def test_tangent_of_two_cells_is_normalised_in_the_weighted_metric():
    th = np.array([[0.0, 0.1], [0.5, 0.7], [1.0, 1.4]])
    m = np.stack([spins(*row) for row in th])
    w = np.array([0.2, 0.8]).reshape(1, 1, 2)
    t = tangents(m, np.array([0.0, 1.0, 2.0]), w)[1]
    assert np.isclose((w * t * t).sum(), 1, rtol=1e-12)


# ---- macrospin ----

@pytest.mark.parametrize("h", [0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("images", [3, 6])
def test_macrospin_barrier_is_exact_on_the_axis(h, images, device):
    # E/V = K sin²θ + 2 K h cosθ with H against m: the top is at cosθ = h, E_B = KV(1 - h)². Few images and no
    # interpolation, so only a climbing image can hit this.
    g, p = macrospin(device, (0.0, 0.0, -h * HK))
    path = gneb(p, g, spin(0.0), spin(np.pi), images=images, torque_tol=1e-9)
    expected = K * g.dx**3 * (1 - h) ** 2
    assert expected > 0
    assert abs(path.barrier / expected - 1) <= 1e-9, (path.barrier, expected)
    top = path.m[path.climbing]
    assert abs(top[2, 0, 0, 0] - h) <= 1e-7




def test_macrospin_barrier_at_a_tilted_field_matches_a_scan(device):
    # Field h HK at 150° from +z in the xz plane, against m. The start minimum lies near +z and the end near -z.
    h, psi, d = 0.3, np.radians(150), 5e-9
    H = h * HK * np.array([np.sin(psi), 0.0, np.cos(psi)])
    g, p = macrospin(device, tuple(H), d)
    a, b = spin(0.1), spin(np.pi - 0.1)
    mn = mm.Minimizer(p)
    mn.minimize(a, 1e-10, 10000)
    mn.minimize(b, 1e-10, 10000)
    path = gneb(p, g, a, b, images=5, torque_tol=1e-9)

    phi = np.linspace(-np.pi, 2 * np.pi, 3_000_001)  # polar angle in the xz plane, unwrapped
    E = (K * np.sin(phi) ** 2 - mm.mu0 * Ms * h * HK * np.cos(phi - psi)) * d**3
    t0, t1 = np.arctan2(a[0, 0, 0, 0], a[2, 0, 0, 0]), np.arctan2(b[0, 0, 0, 0], b[2, 0, 0, 0])
    between = (phi >= t0) & (phi <= t1)
    expected = E[between].max() - (K * np.sin(t0) ** 2 - mm.mu0 * Ms * h * HK * np.cos(t0 - psi)) * d**3
    assert expected > 0
    assert abs(path.barrier / expected - 1) <= 1e-6, (path.barrier, expected)


# ---- wire ----

WIRE_K, WIRE_A, WIRE_MS, WIRE_H = 5e4, 1e-11, 8e5, 0.1


def wire(device, dx, length=240e-9):
    n = round(length / dx)
    g = mm.Grid(n, 1, 1, dx, 4e-9, 4e-9)
    s = (1, 1, n)
    k = np.zeros((3, *s))
    k[2] = 1
    H = (0, 0, -WIRE_H * 2 * WIRE_K / (mm.mu0 * WIRE_MS))
    p = mm.Problem(g, np.full(s, WIRE_MS), np.full(s, WIRE_A), np.full(s, WIRE_K), k, H, device=device, demag=False)
    up = np.zeros((3, *s))
    up[2] = 1
    return g, p, up


def wall_entry_barrier(h, S):
    """A wall entering a free end against a field h H_K. With e(θ) = K(1 - cosθ)(1 + cosθ - 2h) above the start, A θ'² = e
    along the wall and θ' = 0 at the end, so cosθ₀ = 2h - 1 and E_B = 2 S ∫₀^θ₀ √(A e) dθ, in closed form with a = 2h.
    At h = 0 this is 4√(AK) S, one Bloch wall (D51 Correction: at H = 0 the wall has no preferred place, so no single saddle)."""
    a = 2 * h
    I = np.sqrt(2 * (2 - a)) - a * np.log((np.sqrt(2) + np.sqrt(2 - a)) / np.sqrt(a))
    return 2 * np.sqrt(WIRE_A * WIRE_K) * I * S


def wire_barrier(device, dx, seed=0):
    g, p, up = wire(device, dx)
    return gneb(p, g, up, -up, images=21, torque_tol=1e-6, seed=seed), g


def test_wall_entry_formula_tends_to_one_bloch_wall():
    assert np.isclose(wall_entry_barrier(1e-9, 1.0), 4 * np.sqrt(WIRE_A * WIRE_K), rtol=1e-6)


def test_wire_saddle_is_a_wall_entering_at_one_end(device):
    # Far below the coherent barrier KV (1 - h)²: the great-circle start is coherent, so this needs the noise (D48 Correction).
    path, g = wire_barrier(device, 4e-9)
    S = g.dy * g.dz
    expected = wall_entry_barrier(WIRE_H, S)
    assert expected < WIRE_K * g.nx * g.dx * S * (1 - WIRE_H) ** 2 / 3
    assert abs(path.barrier / expected - 1) <= 0.01, (path.barrier, expected)
    mz = path.m[path.climbing][2, 0, 0]
    assert mz.max() > 0.9 and mz.min() < -0.5  # a wall, not a uniform state


def test_wire_barrier_converges_at_second_order(device):
    # The discrete wall's energy differs from the continuum by O(dx²) (D13).
    exact = wall_entry_barrier(WIRE_H, 16e-18)
    err = [abs(wire_barrier(device, dx)[0].barrier / exact - 1) for dx in (8e-9, 4e-9)]
    assert err[0] > 0 and 3 < err[0] / err[1] < 5, err


def test_wire_barrier_does_not_depend_on_the_seed(device):
    # The noise only decides the route's details; a different seed is the cheap check for a wrong saddle.
    a, b = (wire_barrier(device, 4e-9, seed=k)[0] for k in (1, 2))
    assert not np.array_equal(a.m, b.m)
    assert abs(a.barrier / b.barrier - 1) <= 1e-6


def test_the_same_seed_gives_the_same_path():
    g, p, up = wire("cpu", 8e-9)
    a, b = (gneb(p, g, up, -up, images=9, torque_tol=1e-4, seed=3) for _ in range(2))
    assert np.array_equal(a.m, b.m)


# ---- the path itself ----

def test_a_small_barrier_on_a_long_slope_is_found_by_truncating(device):
    # At h = 0.9 the slope is 360× the barrier, so the pass fits between two images of the first path (D48 Correction).
    h = 0.9
    g, p = macrospin(device, (0.0, 0.0, -h * HK))
    path = gneb(p, g, spin(0.0), spin(np.pi), images=7, torque_tol=1e-9)
    KV = K * g.dx**3
    assert 4 * h * KV / (KV * (1 - h) ** 2) > 100
    assert path.truncations > 0 and path.m[-1, 2, 0, 0, 0] > -1 + 1e-3  # the far end moved in
    assert path.energy.max() - path.energy.min() < 10.0 * path.barrier
    assert abs(path.barrier / (KV * (1 - h) ** 2) - 1) <= 1e-9
    assert np.array_equal(path.m[0], spin(0.0))



def test_path_properties_on_a_macrospin(device):
    g, p = macrospin(device, (0.0, 0.0, -0.3 * HK))
    a, b = spin(0.0), spin(np.pi)
    path = gneb(p, g, a, b, images=9, torque_tol=1e-8)
    w = np.ones((1, 1, 1))
    gaps = np.array([distance(path.m[i], path.m[i + 1], w) for i in range(8)])
    assert np.allclose(np.linalg.norm(path.m, axis=1), 1, rtol=1e-12, atol=0)
    assert np.array_equal(path.m[0], a) and np.array_equal(path.m[-1], b)
    assert path.torque < 1e-8 and path.iterations > 0
    assert np.isclose(path.s[-1], gaps.sum(), rtol=1e-12) and path.s[0] == 0
    climbing = path.climbing
    for even in (gaps[:climbing], gaps[climbing:]):  # the climbing image has no spring, so it splits the chain in two
        assert len(even) > 1 and even.max() / even.min() < 1.0001, gaps
    H = np.empty_like(a)
    p.effective_field(np.ascontiguousarray(path.m[climbing]), H)
    assert np.abs(np.cross(path.m[climbing], H, axis=0)).max() / Ms < 1e-8  # the saddle: no torque at all


def test_energy_comes_from_the_field_as_from_problem(device):
    g, p = macrospin(device, (1e4, 2e4, -3e4))
    path = gneb(p, g, spin(0.1), spin(3.0), images=5, torque_tol=1e-4)
    for i in (1, 2, 4):
        expected = sum(p.energies(np.ascontiguousarray(path.m[i])).values())
        assert np.isclose(path.energy[i], expected, rtol=1e-10, atol=0)


def test_barrier_does_not_depend_on_the_number_of_images(device):
    g, p = macrospin(device, (0.0, 0.0, -0.4 * HK))
    E = [gneb(p, g, spin(0.0), spin(np.pi), images=n, torque_tol=1e-9).barrier for n in (5, 10)]
    assert E[0] > 0 and abs(E[0] / E[1] - 1) <= 1e-9


def test_unreached_tolerance_raises(device):
    g, p = macrospin(device, (0.0, 0.0, -0.3 * HK))
    with pytest.raises(RuntimeError, match="torque"):
        gneb(p, g, spin(0.0), spin(np.pi), images=6, torque_tol=1e-12, max_iter=2)


def test_vacuum_cells_take_no_part(device):
    # Two magnetic cells uncoupled by a vacuum cell between them: the barrier is that of two macrospins turning together
    # (or in turn), and the vacuum cell keeps its start value.
    d = 5e-9
    g = mm.Grid(3, 1, 1, d, d, d)
    s = (1, 1, 3)
    Msv = np.array([Ms, 0.0, Ms]).reshape(s)
    k = np.zeros((3, *s))
    k[2] = 1
    p = mm.Problem(g, Msv, np.full(s, A), np.full(s, K), k, (0, 0, 0), device=device, demag=False)
    up = np.zeros((3, *s))
    up[2] = 1
    end = -up
    end[:, 0, 0, 1] = up[:, 0, 0, 1]
    path = gneb(p, g, up, end, images=9, torque_tol=1e-7)
    assert np.array_equal(path.m[:, :, 0, 0, 1], np.broadcast_to(up[:, 0, 0, 1], (9, 3)))
    assert K * d**3 * 0.99 <= path.barrier <= 2 * K * d**3 * 1.01  # one at a time, or both


def test_a_small_nucleus_does_not_kick_the_path_out_of_its_pass():
    # D49 Correction: an rms cap averages over every cell, so a nucleus of a few cells can turn as far as it likes and the
    # chain never settles. The 40 nm cube of D51 on 8³ cells, 3.94 T: this goes red in max_iter with the rms cap.
    n = 8
    d = 40e-9 / n
    g = mm.Grid(n, n, n, d, d, d)
    s = (n, n, n)
    k = np.zeros((3, *s))
    k[2] = 1
    tilt = np.radians(1)
    h = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    p = mm.Problem(g, np.full(s, 1.61 / mm.mu0), np.full(s, 7.7e-12), np.full(s, 4.3e6), k, tuple(-3.94 * h / mm.mu0))
    mn = mm.Minimizer(p)
    start = np.ascontiguousarray(np.broadcast_to(h[:, None, None, None], (3, *s)))
    mn.minimize(start, 1e-6, 100000)
    end = np.ascontiguousarray(-start)
    mn.minimize(end, 1e-6, 100000)
    path = gneb(p, g, start, end, images=10, torque_tol=1e-3, max_iter=1500)
    top = path.m[path.climbing]
    kT = mm.kB * 300
    assert path.truncations > 0 and 10 * kT < path.barrier < 50 * kT
    assert top[2].mean() > 0.9 and top[2].min() < 0.6  # a corner nucleus, not a coherent turn


# ---- the card ----

CARD = """
[grid]
n = [1, 1, 1]
cell = [5e-9, 5e-9, 5e-9]
demag = false

[material.main]
Ms = 8e5
A = 1e-11
K = 5e5

[initial]
m = [0, 0, 1]

[[stage]]
type = "gneb"
direction = [0, 0, 1]
mu0_H = {fields}
images = 5
temperature = {T}
torque_tol = 1e-8
"""


def test_gneb_stage_writes_its_barriers_and_paths(tmp_path, device):
    BK = mm.mu0 * HK
    fields = [float(-0.2 * BK), float(-0.5 * BK)]
    card = tmp_path / "card.toml"
    card.write_text(CARD.format(fields=fields, T=300.0))
    run(card, tmp_path / "run", device=device)

    V = (5e-9) ** 3
    expected = [K * V * (1 - h) ** 2 for h in (0.2, 0.5)]
    stage = json.loads((tmp_path / "run" / "provenance.json").read_text())["stages"][0]
    assert np.allclose(stage["summary"]["barrier_J"], expected, rtol=1e-6)
    assert np.allclose(stage["summary"]["barrier_kT"], np.array(expected) / (mm.kB * 300), rtol=1e-6)

    with open(tmp_path / "run" / "stage1_gneb.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert {"mu0_Hx", "mu0_Hy", "mu0_Hz", "image", "s", "E_total", "climbing"} <= set(rows[0])
    assert sum(int(r["climbing"]) for r in rows) == 2

    with h5py.File(tmp_path / "run" / "trajectory.h5") as f:
        assert f["stage1/m"].shape == (10, 3, 1, 1, 1)
        assert np.allclose(f["stage1/mu0_H"][:5, 2], fields[0]) and np.allclose(f["stage1/mu0_H"][5:, 2], fields[1])
        assert np.allclose(f["final/m"][2], 1)  # m is left at the last field's start state


def test_gneb_stage_finds_the_25_kT_crossing(tmp_path):
    # (1 - h)² KV = 25 kT at h*; with fields either side the summary interpolates linearly between them.
    BK = mm.mu0 * HK
    V = (5e-9) ** 3
    T = 100.0
    h_star = 1 - np.sqrt(25 * mm.kB * T / (K * V))
    assert 0 < h_star < 1
    h = [h_star - 0.02, h_star + 0.02]
    card = tmp_path / "card.toml"
    card.write_text(CARD.format(fields=[float(-x * BK) for x in h], T=T))
    run(card, tmp_path / "run")
    s = json.loads((tmp_path / "run" / "provenance.json").read_text())["stages"][0]["summary"]
    E = [(1 - x) ** 2 * K * V / (mm.kB * T) for x in h]
    linear = h[0] + (h[1] - h[0]) * (E[0] - 25) / (E[0] - E[1])
    assert np.isclose(s["Hc_T"], -linear * BK, rtol=1e-5)
    assert abs(s["Hc_T"] / (-h_star * BK) - 1) < 5e-3


def test_no_crossing_gives_none(tmp_path):
    card = tmp_path / "card.toml"
    card.write_text(CARD.format(fields=[float(-0.1 * mm.mu0 * HK)], T=300.0))
    run(card, tmp_path / "run")
    assert json.loads((tmp_path / "run" / "provenance.json").read_text())["stages"][0]["summary"]["Hc_T"] is None


@pytest.mark.parametrize("old, new, key", [
    ("mu0_H = [-0.1]", "mu0_H = []", "mu0_H"),
    ("mu0_H = [-0.1]", 'mu0_H = ["x"]', "mu0_H"),
    ("mu0_H = [-0.1]", "mu0_H = -0.1", "mu0_H"),
    ("images = 5", "images = 2", "images"),
    ("images = 5", "images = 5.5", "images"),
    ("temperature = 300.0", "temperature = 0.0", "temperature"),
    ("direction = [0, 0, 1]", "direction = [0, 0, 0]", "direction"),
    ("images = 5", "images = 5\nseed = -1", "seed"),
])
def test_gneb_card_is_validated(tmp_path, old, new, key):
    from micromag.card import CardError, load
    text = CARD.format(fields=[-0.1], T=300.0)
    assert old in text
    path = tmp_path / "card.toml"
    path.write_text(text)
    assert load(path).stages[0].images == 5
    path.write_text(text.replace(old, new))
    with pytest.raises(CardError, match=key):
        load(path)


# ---- the physics check ----

CUBE = """
[grid]
n = [30, 30, 30]
cell = [{d!r}, {d!r}, {d!r}]

[material.main]
Ms = {Ms!r}
A = 7.7e-12
K = 4.3e6

[initial]
m = {h}

[[stage]]
type = "gneb"
direction = {h}
mu0_H = [-3.7, -3.94, -5.2]
images = 10
temperature = 300.0
"""


@pytest.mark.slow
def test_cube_barrier_crosses_25_kT_at_the_published_field(tmp_path, device):
    # Fischbacher et al., APL 111, 072404 (2017): the 40 nm cube of D32, 1° off z, reaches 25 k_BT at 3.94 T, and its barrier
    # vanishes at the static switching field (ours: 5.275 T). A coherent path would stand near 11,000 k_BT at 3.94 T (D48).
    tilt = np.radians(1)
    h = [float(np.sin(tilt)), 0.0, float(np.cos(tilt))]
    card = tmp_path / "card.toml"
    card.write_text(CUBE.format(d=40e-9 / 30, Ms=1.61 / mm.mu0, h=h))
    run(card, tmp_path / "run", device=device)
    s = json.loads((tmp_path / "run" / "provenance.json").read_text())["stages"][0]["summary"]
    print(f"\ncube ({device}): barriers {np.round(s['barrier_kT'], 3).tolist()} k_BT at {s['mu0_H_T']} T, "
          f"25 k_BT at {s['Hc_T']:.3f} T (ref -3.94 T)")
    assert abs(s["Hc_T"] / -3.94 - 1) <= 0.03
    assert 0 < s["barrier_kT"][-1] < 1
