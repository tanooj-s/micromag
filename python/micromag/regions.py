"""Material regions on the MuMax3 model (D33–D35): a uint8 region map painted from a card, expanded into per-cell arrays."""

from dataclasses import dataclass

import numpy as np

from .card import VACUUM, CardError


@dataclass
class Painted:
    regions: np.ndarray        # (nz, ny, nx) uint8 ids
    Ms: np.ndarray             # (nz, ny, nx), A/m
    A: np.ndarray
    K: np.ndarray
    axis: np.ndarray           # (3, nz, ny, nx)
    names: dict                # id -> material name, for the ids in use
    coupling: np.ndarray | None  # (256, 256) pair scales, or None without [[coupling]] (D35)
    report: list               # per [[region]]: id, exact and painted volumes (D33)
    seeds: np.ndarray | None   # (count, 3) grain seeds in m (D34)


def _inside(r, x, y, z):
    """Cells whose centre is in region r's shape (D33); the complement if r.inverse."""
    p = r.params
    if r.shape == "all":
        inside = np.ones(x.shape, bool)
    elif r.shape == "cuboid":
        (x0, y0, z0), (x1, y1, z1) = p["corners"]
        inside = (x0 <= x) & (x < x1) & (y0 <= y) & (y < y1) & (z0 <= z) & (z < z1)  # lower bounds in, upper out (D30)
    elif r.shape == "cylinder":  # along z
        cx, cy, cz = p["center"]
        inside = ((x - cx) ** 2 + (y - cy) ** 2 <= p["radius"] ** 2) & (np.abs(z - cz) <= p["height"] / 2)
    else:  # ellipsoid
        (cx, cy, cz), (a, b, c) = p["center"], p["semi_axes"]
        inside = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 + ((z - cz) / c) ** 2 <= 1
    return ~inside if r.inverse else inside


def _volume(r):
    """The shape's exact volume (m³), or None for `all` and inverted shapes."""
    p = r.params
    if r.inverse or r.shape == "all":
        return None
    if r.shape == "cuboid":
        return float(np.prod(p["corners"][1] - p["corners"][0]))
    if r.shape == "cylinder":
        return float(np.pi * p["radius"] ** 2 * p["height"])
    return float(4 / 3 * np.pi * np.prod(p["semi_axes"]))


def voronoi(points, seeds, chunk=8192):
    """(owner, distance) per point: the index of its nearest seed, and its exact distance to that seed's Voronoi cell
    boundary, min over j of (|x - s_j|² - |x - s_i|²) / (2 |s_j - s_i|) (D34); inf with a single seed."""
    owner, dist = np.empty(len(points), int), np.empty(len(points))
    sep = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)  # |s_j - s_i|
    for a in range(0, len(points), chunk):
        p = points[a:a + chunk]
        d2 = ((p[:, None] - seeds[None]) ** 2).sum(axis=2)
        i = d2.argmin(axis=1)
        rows = np.arange(len(p))
        with np.errstate(divide="ignore", invalid="ignore"):
            plane = (d2 - d2[rows, i][:, None]) / (2 * sep[i])
        plane[rows, i] = np.inf
        owner[a:a + chunk], dist[a:a + chunk] = i, plane.min(axis=1)
    return owner, dist


def grain_axes(rng, count, spec, material_axis):
    """(count, 3) easy axes (D34): the material's, uniform on the sphere, or uniform over a cap {mean, cone_deg}."""
    if spec == "material":
        return np.tile(material_axis, (count, 1))
    if spec == "random":
        k = rng.normal(size=(count, 3))
        return k / np.linalg.norm(k, axis=1, keepdims=True)
    mean = spec["mean"]
    a = np.array([1.0, 0, 0]) if abs(mean[0]) < 0.9 else np.array([0.0, 1, 0])
    u = a - (a @ mean) * mean
    u /= np.linalg.norm(u)
    v = np.cross(mean, u)
    cos = rng.uniform(np.cos(np.radians(spec["cone_deg"])), 1, count)  # uniform in cos θ: uniform over the cap
    phi = rng.uniform(0, 2 * np.pi, count)
    sin = np.sqrt(1 - cos**2)
    return cos[:, None] * mean + sin[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)


def _seeds(rng, count, within, cell):
    """count points uniform over the grid box, kept only where they fall in a `within` cell (D34)."""
    if not within.any():
        raise CardError("[grains]: the 'within' region has no cells")
    size = np.array(within.shape[::-1]) * cell
    kept = np.empty((0, 3))
    for _ in range(1000):
        q = rng.uniform(0, size, (4 * count, 3))
        ix, iy, iz = np.minimum((q / cell).astype(int), np.array(within.shape[::-1]) - 1).T
        kept = np.concatenate([kept, q[within[iz, iy, ix]]])
        if len(kept) >= count:
            return kept[:count]
    raise CardError("[grains]: the 'within' region is too small a part of the grid to place seeds in")


def paint(card):
    """The region map and per-cell materials of a card: regions in order, then grains (D33, D34), and the pair table (D35)."""
    nx, ny, nz = card.n
    cell = np.array(card.cell)
    dV = float(np.prod(cell))
    z, y, x = np.meshgrid(*[(np.arange(k) + 0.5) * d for k, d in zip((nz, ny, nx), cell[::-1])], indexing="ij")
    known = {**card.materials, "vacuum": VACUUM}

    ids = np.zeros((nz, ny, nx), np.uint8)
    material = {0: next(iter(card.materials))}  # id -> name; region 0 is the background
    axis = np.zeros((256, 3))                   # id -> easy axis
    report = []
    for k, r in enumerate(card.regions, 1):
        inside = _inside(r, x, y, z)
        ids[inside] = r.id
        material[r.id] = r.material
        axis[r.id] = r.axis if r.axis is not None else known[r.material].axis
        report.append({"region": k, "id": r.id, "material": r.material, "shape": r.shape, "volume_m3": _volume(r),
                       "cells_m3": np.count_nonzero(inside) * dV})
    axis[0] = known[material[0]].axis

    seeds, g = None, card.grains
    if g is not None:
        rng = np.random.default_rng(g.seed)
        within = ids == g.within
        seeds = _seeds(rng, g.count, within, cell)
        owner, dist = voronoi(np.stack([x[within], y[within], z[within]], axis=1), seeds)
        grain = (g.first_id + owner).astype(np.uint8)
        for i in range(g.first_id, g.first_id + g.count):
            material[i] = g.material
        axis[g.first_id:g.first_id + g.count] = grain_axes(rng, g.count, g.axes, known[g.material].axis)
        if g.boundary is not None:
            b = g.boundary
            grain[dist < b.thickness / 2] = b.id
            material[b.id] = b.material
            axis[b.id] = known[b.material].axis
        ids[within] = grain

    table = lambda key: np.array([getattr(known[material[i]], key) if i in material else 0.0 for i in range(256)])
    Ms, A, K = (table(key)[ids] for key in ("Ms", "A", "K"))
    used = {int(i): material[int(i)] for i in np.unique(ids)}
    return Painted(ids, Ms, A, K, np.moveaxis(axis[ids], -1, 0).copy(), used, _coupling(card, material, known), report, seeds)


def _coupling(card, material, known):
    """The (256, 256) pair-scale table of the card's [[coupling]] entries, later ones winning, or None (D35)."""
    if not card.couplings:
        return None
    t = np.ones((256, 256))
    g = card.grains
    grains = list(range(g.first_id, g.first_id + g.count)) if g is not None else []
    for k, c in enumerate(card.couplings, 1):
        expand = lambda r: grains if r == "grains" else [r]
        for a in expand(c.regions[0]):
            for b in expand(c.regions[1]):
                if a == b:
                    continue
                if c.scale is not None:
                    s = c.scale
                else:
                    if a not in material or b not in material:
                        raise CardError(f"coupling {k}: 'A' needs both regions painted; {a if a not in material else b} is not")
                    Aa, Ab = known[material[a]].A, known[material[b]].A
                    if Aa * Ab == 0:
                        raise CardError(f"coupling {k}: regions {a} and {b} have no exchange to replace (A = 0 on one side)")
                    s = c.A / (2 * Aa * Ab / (Aa + Ab))
                t[a, b] = t[b, a] = s
    return t
