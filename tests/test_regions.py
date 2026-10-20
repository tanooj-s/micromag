"""Iteration-6c checks (D37): painting regions, Voronoi grains, the grain-boundary phase, and the Stoner–Wohlfarth ensemble."""

import json

import numpy as np
import pytest

import micromag as mm
from micromag.card import load
from micromag.regions import grain_axes, paint, voronoi

MATERIALS = """
[material.soft]
Ms = 8e5
A = 1.3e-11

[material.hard]
Ms = 1.2e6
A = 8e-12
K = 4e6
axis = [0, 0, 1]
"""

TAIL = """
[initial]
m = [0, 0, 1]

[[stage]]
type = "minimize"
"""


def painted(tmp_path, n, cell, body, materials=MATERIALS):
    path = tmp_path / "card.toml"
    path.write_text(f"[grid]\nn = {list(n)}\ncell = {[float(c) for c in cell]}\n" + materials + body + TAIL)
    return paint(load(path))


def centres(n, cell):
    """Cell centres (nz, ny, nx) each, x fastest (D9)."""
    return np.meshgrid(*[(np.arange(k) + 0.5) * d for k, d in zip(n[::-1], cell[::-1])], indexing="ij")[::-1]


def test_regions_paint_cell_centres_in_order(tmp_path):
    # D33 (from D30's test): a cell joins a shape if its centre is inside (cuboid lower bound in, upper out);
    # later paints win; the background is region 0 with the first material.
    p = painted(tmp_path, (4, 3, 2), (2e-9,) * 3, """
[[region]]
id = 1
material = "hard"
shape = "cuboid"
corners = [[2e-9, 0, 0], [8e-9, 6e-9, 4e-9]]
axis = [1, 0, 0]

[[region]]
id = 2
material = "vacuum"
shape = "cuboid"
corners = [[0, 0, 2e-9], [4e-9, 6e-9, 4e-9]]
""")
    x_in = np.array([False, True, True, True])             # cell centres at 1, 3, 5, 7 nm; box from 2 nm
    vacuum = np.zeros((2, 3, 4), bool)
    vacuum[1, :, :2] = True                                # z centre 3 nm, x centres 1 and 3 nm
    expect = np.where(vacuum, 2, np.where(x_in, 1, 0) * np.ones((2, 3, 1), int))
    assert p.regions.dtype == np.uint8 and np.array_equal(p.regions, expect)
    assert np.array_equal(p.Ms, np.choose(expect, [8e5, 1.2e6, 0.0]))
    assert np.array_equal(p.A, np.choose(expect, [1.3e-11, 8e-12, 0.0]))
    assert np.array_equal(p.K, np.choose(expect, [0.0, 4e6, 0.0]))
    assert np.array_equal(p.axis[0], (expect == 1).astype(float))   # the region's axis overrides the material's
    assert np.array_equal(p.axis[2], (expect != 1).astype(float))
    assert p.names == {0: "soft", 1: "hard", 2: "vacuum"}
    assert p.coupling is None


def test_inverse_paints_the_complement_and_ids_can_repeat(tmp_path):
    p = painted(tmp_path, (8, 8, 1), (1e-9,) * 3, """
[[region]]
id = 1
material = "hard"
shape = "cylinder"
center = [4e-9, 4e-9, 0.5e-9]
radius = 3e-9
height = 1e-9

[[region]]
id = 2
material = "vacuum"
shape = "cylinder"
center = [4e-9, 4e-9, 0.5e-9]
radius = 3e-9
height = 1e-9
inverse = true

[[region]]
id = 1
material = "hard"
shape = "cuboid"
corners = [[0, 0, 0], [1e-9, 1e-9, 1e-9]]
""")
    x, y, z = centres((8, 8, 1), (1e-9,) * 3)
    disc = (x - 4e-9) ** 2 + (y - 4e-9) ** 2 <= (3e-9) ** 2
    expect = np.where(disc, 1, 2)
    expect[0, 0, 0] = 1
    assert np.array_equal(p.regions, expect)
    assert 0 not in p.regions and p.Ms[0, 0, 0] == 1.2e6


@pytest.mark.parametrize("shape, exact", [
    ("ellipsoid", lambda R: 4 / 3 * np.pi * R * 0.8 * R * 0.6 * R),
    ("cylinder", lambda R: np.pi * R**2 * 1.5 * R),
])
def test_shape_volumes_converge_to_the_exact_volume(tmp_path, shape, exact):
    for k, ratio in enumerate((8, 16, 32)):
        R, dx = 1e-8, 1e-8 / ratio
        n = (2 * ratio + 2,) * 3
        c = [(ratio + 1) * dx] * 3
        keys = f"semi_axes = [{R!r}, {0.8 * R!r}, {0.6 * R!r}]" if shape == "ellipsoid" else f"radius = {R!r}\nheight = {1.5 * R!r}"
        sub = tmp_path / str(k)
        sub.mkdir()
        p = painted(sub, n, (dx,) * 3, f'[[region]]\nid = 1\nmaterial = "hard"\nshape = "{shape}"\ncenter = {c}\n{keys}\n')
        (r,) = p.report
        assert np.isclose(r["volume_m3"], exact(R), rtol=1e-12)
        assert r["cells_m3"] == np.count_nonzero(p.regions == 1) * dx**3
        err = abs(r["cells_m3"] / r["volume_m3"] - 1)
        assert 0 < err <= dx / R, (ratio, err)  # staircasing is visible, and shrinks with the cell


def test_report_flags_a_clipped_shape(tmp_path):
    p = painted(tmp_path, (4, 4, 4), (1e-9,) * 3, """
[[region]]
id = 1
material = "hard"
shape = "cuboid"
corners = [[2e-9, 0, 0], [8e-9, 4e-9, 4e-9]]

[[region]]
id = 2
material = "hard"
shape = "all"
""")
    clipped, full = p.report
    assert clipped["volume_m3"] == pytest.approx(96e-27) and clipped["cells_m3"] == pytest.approx(32e-27)
    assert full["volume_m3"] is None and full["cells_m3"] == pytest.approx(64e-27)


GRAINS = """
[[region]]
id = 1
material = "hard"
shape = "cuboid"
corners = [[2e-9, 2e-9, 0], [14e-9, 12e-9, 4e-9]]

[grains]
count = 9
first_id = 10
material = "hard"
seed = {seed}
within = 1
axes = "random"
"""


def test_grains_take_their_nearest_seed(tmp_path):
    p = painted(tmp_path, (16, 14, 4), (1e-9,) * 3, GRAINS.format(seed=7))
    x, y, z = centres((16, 14, 4), (1e-9,) * 3)
    box = (x >= 2e-9) & (x < 14e-9) & (y >= 2e-9) & (y < 12e-9)
    assert np.all(p.regions[~box] == 0)                       # nothing outside `within` changes
    pts = np.stack([x[box], y[box], z[box]], axis=1)
    d2 = ((pts[:, None, :] - p.seeds[None, :, :]) ** 2).sum(axis=2)  # brute force: every cell against every seed
    assert np.array_equal(p.regions[box], 10 + d2.argmin(axis=1))
    assert p.seeds.shape == (9, 3) and len(np.unique(p.regions[box])) >= 6   # guard: a real tessellation
    for g in range(9):  # each grain's cells share its axis, a unit vector
        cells = p.axis[:, p.regions == 10 + g]
        if cells.size:
            assert np.allclose(cells, cells[:, :1]) and np.isclose(np.linalg.norm(cells[:, 0]), 1)
    seed_cells = np.floor(p.seeds / 1e-9).astype(int)
    assert np.all(box[seed_cells[:, 2], seed_cells[:, 1], seed_cells[:, 0]])  # seeds are drawn inside `within`


def test_grains_are_reproducible_from_their_seed(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir(), (tmp_path / "c").mkdir()
    a, b = (painted(tmp_path / s, (16, 14, 4), (1e-9,) * 3, GRAINS.format(seed=7)) for s in "ab")
    c = painted(tmp_path / "c", (16, 14, 4), (1e-9,) * 3, GRAINS.format(seed=8))
    assert np.array_equal(a.regions, b.regions) and np.array_equal(a.axis, b.axis)
    assert not np.array_equal(a.regions, c.regions)


def test_random_axes_are_isotropic_and_cones_stay_in_their_cap():
    rng = np.random.default_rng(0)
    k = grain_axes(rng, 40_000, "random", np.array([0.0, 0, 1]))
    assert np.allclose(np.linalg.norm(k, axis=1), 1)
    for c in range(3):  # uniform on the sphere: <|k_c|> = 1/2 for each axis; the error is 0.29/200
        assert abs(np.abs(k[:, c]).mean() - 0.5) < 0.005
    mean = np.array([1.0, 1, 0]) / np.sqrt(2)
    k = grain_axes(rng, 40_000, {"mean": mean, "cone_deg": 20.0}, None)
    cos = k @ mean
    assert cos.min() >= np.cos(np.radians(20)) - 1e-12
    assert abs(cos.mean() - (1 + np.cos(np.radians(20))) / 2) < 1e-3  # uniform over the cap: cos θ uniform
    assert np.allclose(grain_axes(rng, 5, "material", np.array([0.0, 1, 0])), [0, 1, 0])


def test_boundary_distance_is_exact_for_a_mirrored_pair():
    # Two seeds mirrored across x = 8 nm: the boundary is that plane, so the distance is |x - 8 nm| (D34).
    x, y, z = centres((16, 4, 4), (1e-9,) * 3)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    seeds = np.array([[5e-9, 1.3e-9, 2.9e-9], [11e-9, 1.3e-9, 2.9e-9]])
    owner, dist = voronoi(pts, seeds)
    assert np.array_equal(owner, (pts[:, 0] > 8e-9).astype(int))
    assert np.allclose(dist, np.abs(pts[:, 0] - 8e-9), rtol=0, atol=1e-21)
    assert np.count_nonzero(dist < 4e-9 / 2) == 4 * 16  # thickness 4 nm: exactly 4 cell layers of 4 x 4


def test_boundary_phase_matches_a_sampled_distance_oracle(tmp_path):
    n, dx, t = (16, 16, 3), 1e-9, 3e-9
    p = painted(tmp_path, n, (dx,) * 3, f"""
[material.gbp]
Ms = 3e5
A = 5e-12

[grains]
count = 6
first_id = 1
material = "hard"
seed = 3
boundary = {{ thickness = {t!r}, material = "gbp", id = 200 }}
""")
    x, y, z = centres(n, (dx,) * 3)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    own = ((pts[:, None] - p.seeds[None]) ** 2).sum(2).argmin(1)
    # The oracle: distance to the nearest point of a 4x finer sampling (padded by t) that belongs to another grain.
    h = dx / 4
    axes = [np.arange(-t, k * dx + t, h) + h / 2 for k in n]
    fine = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1)
    fine_own = ((fine[:, None] - p.seeds[None]) ** 2).sum(2).argmin(1)
    oracle = np.array([np.sqrt(((fine[fine_own != o] - q) ** 2).sum(1).min()) for q, o in zip(pts, own)])
    sure = np.abs(oracle - t / 2) > 2 * h  # cells clearly inside or outside the layer
    gb = (p.regions.ravel() == 200)
    assert sure.mean() > 0.6 and 0.1 < gb.mean() < 0.9  # guard: a real layer, mostly decidable
    assert np.array_equal(gb[sure], oracle[sure] < t / 2)
    assert np.array_equal(p.regions.ravel()[~gb], 1 + own[~gb])
    assert np.all(p.Ms.ravel()[gb] == 3e5) and p.names[200] == "gbp"


def test_coupling_table_from_the_card(tmp_path):
    p = painted(tmp_path, (16, 14, 4), (1e-9,) * 3, GRAINS.format(seed=7) + """
[[coupling]]
regions = ["grains", "grains"]
scale = 0.25

[[coupling]]
regions = [1, 0]
A = 2e-12

[[coupling]]
regions = [10, 11]
scale = 0.0
""")
    c = p.coupling
    assert c.shape == (256, 256) and np.array_equal(c, c.T)
    grains = np.arange(10, 19)
    block = c[np.ix_(grains, grains)]
    assert np.all(block[~np.eye(9, dtype=bool)] == np.where(np.add.outer(grains, grains) == 21, 0.0, 0.25)[~np.eye(9, dtype=bool)])
    assert c[1, 0] == pytest.approx(2e-12 / (2 * 8e-12 * 1.3e-11 / (8e-12 + 1.3e-11)))  # A -> scale of the harmonic mean
    assert c[0, 10] == 1.0 and c[5, 6] == 1.0  # everything else is untouched


# ---- the headline physics check (D37): decoupled grains without demag are Stoner–Wohlfarth particles ----

def sw_branch(psi, fields):
    """m·h per particle along a descending branch of reduced fields h = H/H_K, starting at the remanent m ∥ k.

    1D gradient flow of e(φ) = ½ sin²(φ − ψ) − h cos φ, with φ the angle of m from the field direction."""
    phi = psi.copy()
    out = []
    for h in fields:
        for _ in range(100_000):
            g = 0.5 * np.sin(2 * (phi - psi)) + h * np.sin(phi)
            phi -= 0.3 * g
            if np.abs(g).max() < 1e-13:
                break
        out.append(np.cos(phi))
    return np.array(out)


def sw_coercivity(psi, w):
    """The first zero of Σ w cos φ on the descending branch, by a coarse scan then a fine one through the bracket."""
    coarse = -np.arange(0, 1.2, 1e-3)
    mh = sw_branch(psi, coarse) @ w
    k = np.argmax(mh < 0)
    fine = np.linspace(coarse[k - 1], coarse[k], 101)
    mh = sw_branch(psi, np.concatenate([coarse[:k], fine]))[k:] @ w
    j = np.argmax(mh < 0)
    return fine[j - 1] + (fine[j] - fine[j - 1]) * mh[j - 1] / (mh[j - 1] - mh[j]), mh, fine


@pytest.mark.slow
def test_decoupled_grains_are_a_stoner_wohlfarth_ensemble(tmp_path):
    Ms, K = 8e5, 5e5
    BK = 2 * K / Ms  # µ0 H_K in T
    path = tmp_path / "card.toml"
    path.write_text(f"""
[grid]
n = [40, 40, 40]
cell = [2e-9, 2e-9, 2e-9]
demag = false

[material.grain]
Ms = {Ms!r}
A = 1e-11
K = {K!r}

[grains]
count = 255
first_id = 1
material = "grain"
seed = 1
axes = "random"

[[coupling]]
regions = ["grains", "grains"]
scale = 0.0

[initial]
m = [0, 0, 1]

[[stage]]
type = "hysteresis"
direction = [0, 0, 1]
mu0_H_from = 0.0
mu0_H_to = {-1.2 * BK!r}
mu0_H_step = {0.025 * BK!r}
refine_to = {1e-4 * BK!r}
""")
    p = paint(load(path))
    ids = np.arange(1, 256)
    w = np.array([np.count_nonzero(p.regions == i) for i in ids], dtype=float)
    w /= w.sum()
    kz = np.array([abs(p.axis[2][p.regions == i][0]) if wi > 0 else 1.0 for i, wi in zip(ids, w)])
    psi = np.arccos(np.clip(kz, 0, 1))
    mr_exact = w @ kz
    hc_oracle, _, _ = sw_coercivity(psi, w)

    mm.run(path, tmp_path / "run")
    s = json.loads((tmp_path / "run" / "provenance.json").read_text())["stages"][0]["summary"]
    mr, hc = s["remanence"][2], s["Hc_T"] / BK

    sigma_mr = np.sqrt((w**2).sum() / 12)  # var |cos θ| = 1/3 - 1/4 for isotropic axes
    path = sw_branch(psi, [*np.linspace(0, hc_oracle + 1e-3, 600), hc_oracle, hc_oracle - 1e-3])  # follow the branch
    c = path[-2]
    slope = (path[-3] @ w - path[-1] @ w) / 2e-3
    sigma_hc = np.sqrt((w**2 * (c - c @ w) ** 2).sum()) / abs(slope)
    print(f"\nSW ensemble ({np.count_nonzero(w)} grains): Mr = {mr:.5f} (exact {mr_exact:.5f}, isotropic 0.5 ± {sigma_mr:.3f}); "
          f"Hc = {-hc:.5f} H_K (oracle {-hc_oracle:.5f}, isotropic 0.479 ± {sigma_hc:.3f}); "
          f"differences {mr - mr_exact:.1e}, {hc - hc_oracle:.1e}")
    assert abs(mr - mr_exact) <= 1e-4
    assert abs(hc - hc_oracle) <= 5e-3
    assert abs(mr_exact - 0.5) <= 3 * sigma_mr
    assert abs(-hc_oracle - 0.479) <= 3 * sigma_hc
