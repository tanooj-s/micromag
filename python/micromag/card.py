"""Input card (D27): a TOML file, validated into plain dataclasses. Never touches the solver."""

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CardError(ValueError):
    """Every problem found in a card, one per line."""


@dataclass
class LLGDynamics:
    alpha: float
    gamma: float
    mu0_H: np.ndarray  # T
    tol: float
    max_step: float
    duration: float | None
    sample_every: float | None
    until_torque: float | None
    check_every: float | None
    max_duration: float | None
    dump_every: float | None
    temperature: float  # K (D41)
    method: str         # rk45, heun or rk23 (D42)
    dt: float | None    # s, heun only
    seed: int | None    # required at T > 0
    type: str = "llg-dynamics"


@dataclass
class Minimize:
    mu0_H: np.ndarray  # T
    torque_tol: float
    max_iter: int
    type: str = "minimize"


@dataclass
class Hysteresis:
    direction: np.ndarray  # unit vector
    mu0_H_from: float      # T, along direction
    mu0_H_to: float
    mu0_H_step: float
    torque_tol: float
    max_iter: int
    dump: bool
    refine_to: float | None  # T: halve a step that crosses m·h = 0 or jumps, down to this (D31)
    refine_jump: float       # |Δ<m>| that counts as a jump
    type: str = "hysteresis"

    def fields_T(self):
        """(n, 3) applied fields µ0 H in T, from mu0_H_from to mu0_H_to inclusive."""
        n = round(abs(self.mu0_H_to - self.mu0_H_from) / self.mu0_H_step)
        h = self.mu0_H_from + math.copysign(self.mu0_H_step, self.mu0_H_to - self.mu0_H_from) * np.arange(n + 1)
        return h[:, None] * self.direction


@dataclass
class Material:
    Ms: float  # A/m; 0 is vacuum (D11)
    A: float   # J/m
    K: float   # J/m³
    axis: np.ndarray


@dataclass
class Region:
    id: int             # 1–255; 0 is the background (D33)
    material: str
    shape: str          # a key of SHAPES
    params: dict        # the shape's keys, m from the grid's lower corner
    inverse: bool       # paint the complement
    axis: np.ndarray | None  # overrides the material's


@dataclass
class Boundary:
    thickness: float  # m
    material: str
    id: int


@dataclass
class Grains:
    count: int
    first_id: int
    material: str
    seed: int
    within: int                # the region id the grains fill
    axes: str | dict           # "material", "random" or {"mean": unit vector, "cone_deg": θ} (D34)
    boundary: Boundary | None


@dataclass
class Coupling:
    regions: list          # [a, b]: ids or "grains"
    scale: float | None    # exactly one of scale and A (D35)
    A: float | None


@dataclass
class Card:
    path: Path
    text: bytes  # the card as read, for the byte-for-byte copy (D28)
    device: str
    n: tuple     # cells along x, y, z
    cell: tuple  # m
    demag: bool
    materials: dict        # name -> Material, in card order; the first fills region 0 (D33)
    m0: np.ndarray | None  # uniform start, normalised
    file: Path | None      # or a trajectory.h5 to start from ...
    state: str             # ... and which state in it
    stages: list
    regions: list          # ordered; later paints win (D33)
    grains: Grains | None  # applied after the regions (D34)
    couplings: list        # ordered; later entries win (D35)


# Value checkers: each returns (value, None) or (None, what was expected).
def _number(v, ok=lambda x: True, what="a number"):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not ok(v):  # bool is an int in Python, not here
        return None, what
    return float(v), None


def _positive(v):
    return _number(v, lambda x: x > 0, "a positive number")


def _nonnegative(v):
    return _number(v, lambda x: x >= 0, "a number ≥ 0")


def _count(v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None, "a whole number ≥ 1"
    return v, None


def _vector(v, ok=lambda a: True, what="3 numbers"):
    if not isinstance(v, list) or len(v) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v):
        return None, what
    a = np.array(v, dtype=float)
    return (a, None) if ok(a) else (None, what)


def _direction(v):
    a, err = _vector(v, lambda a: np.linalg.norm(a) > 0, "3 numbers, not all zero")
    return (a / np.linalg.norm(a), None) if err is None else (None, err)


def _bool(v):
    return (v, None) if isinstance(v, bool) else (None, "true or false")


def _string(v):
    return (v, None) if isinstance(v, str) else (None, "a string")


REQUIRED = object()


def _table(raw, spec, where, errors):
    """Checks raw (a dict) against spec {key: (checker, default)}; returns {key: value}, appending to errors."""
    if not isinstance(raw, dict):
        errors.append(f"{where}: must be a table")
        return {}
    for key in raw.keys() - spec.keys():
        errors.append(f"{where}: unknown key '{key}' (valid: {', '.join(spec)})")
    out = {}
    for key, (check, default) in spec.items():
        if key not in raw:
            if default is REQUIRED:
                errors.append(f"{where}: missing '{key}'")
            out[key] = None if default is REQUIRED else default
            continue
        value, expected = check(raw[key])
        if expected:
            errors.append(f"{where}: '{key}' must be {expected}, got {raw[key]!r}")
        out[key] = value
    return out


ZERO = np.zeros(3)
METHODS = ("rk45", "heun", "rk23")
STAGES = {
    "llg-dynamics": {
        "type": (_string, REQUIRED),
        "alpha": (_nonnegative, REQUIRED),
        "gamma": (_positive, 1.76086e11),  # rad/(T·s), D18
        "mu0_H": (_vector, ZERO),
        "tol": (_positive, 1e-6),
        "max_step": (_positive, math.inf),
        "duration": (_positive, None),
        "sample_every": (_positive, None),
        "until_torque": (_positive, None),
        "check_every": (_positive, None),
        "max_duration": (_positive, None),
        "dump_every": (_positive, None),
        "temperature": (_nonnegative, 0.0),
        "method": (lambda v: (v, None) if v in METHODS else (None, "rk45, heun or rk23"), None),  # None: by temperature
        "dt": (_positive, None),
        "seed": (lambda v: (v, None) if _seed(v)[1] is None and v < 2**32 else (None, "a whole number from 0 to 2³² − 1"), None),
    },
    "minimize": {
        "type": (_string, REQUIRED),
        "mu0_H": (_vector, ZERO),
        "torque_tol": (_positive, 1e-6),
        "max_iter": (_count, 100_000),
    },
    "hysteresis": {
        "type": (_string, REQUIRED),
        "direction": (_direction, REQUIRED),
        "mu0_H_from": (_number, REQUIRED),
        "mu0_H_to": (_number, REQUIRED),
        "mu0_H_step": (_positive, REQUIRED),
        "torque_tol": (_positive, 1e-6),
        "max_iter": (_count, 100_000),
        "dump": (_bool, False),
        "refine_to": (_positive, None),
        "refine_jump": (_positive, 0.1),
    },
}


def _stage(raw, where, errors):
    kind = raw.get("type") if isinstance(raw, dict) else None
    if kind not in STAGES:
        errors.append(f"{where}: 'type' must be one of {', '.join(STAGES)}, got {kind!r}")
        return None
    before = len(errors)
    v = _table(raw, STAGES[kind], f"{where} ({kind})", errors)
    if len(errors) > before:
        return None
    where = f"{where} ({kind})"
    if kind == "minimize":
        return Minimize(v["mu0_H"], v["torque_tol"], v["max_iter"])
    if kind == "hysteresis":
        span = abs(v["mu0_H_to"] - v["mu0_H_from"]) / v["mu0_H_step"]
        if abs(span - round(span)) > 1e-9 * max(1.0, span):
            errors.append(f"{where}: 'mu0_H_step' must divide mu0_H_to - mu0_H_from into whole steps")
            return None
        if v["refine_to"] is not None and v["refine_to"] > v["mu0_H_step"]:
            errors.append(f"{where}: 'refine_to' must be ≤ mu0_H_step")
            return None
        v.pop("type")
        return Hysteresis(**v)

    timed, torque = v["duration"] is not None, v["until_torque"] is not None
    if timed == torque:
        errors.append(f"{where}: give exactly one stopping rule: 'duration' (with optional 'sample_every') "
                      "or 'until_torque' (with 'check_every' and 'max_duration')")
        return None
    if timed and (v["check_every"] is not None or v["max_duration"] is not None):
        errors.append(f"{where}: 'check_every' and 'max_duration' go with 'until_torque', not 'duration'")
    if torque:
        if v["sample_every"] is not None:
            errors.append(f"{where}: 'sample_every' goes with 'duration'; with 'until_torque' rows come every 'check_every'")
        for key in ("check_every", "max_duration"):
            if v[key] is None:
                errors.append(f"{where}: 'until_torque' needs '{key}'")
        if v["tol"] > v["until_torque"] / 100:  # D20: each step re-excites a torque of a few × tol
            errors.append(f"{where}: 'tol' ({v['tol']:g}) must be ≤ until_torque / 100 ({v['until_torque'] / 100:g}), "
                          "or the torque never falls below until_torque (D20)")
    hot = v["temperature"] > 0
    if v["method"] is None:  # D43: existing cards keep rk45; heun is the default at T > 0
        v["method"] = "heun" if hot else "rk45"
    if hot and v["method"] == "rk45":
        errors.append(f"{where}: rk45 can't take noise; use method = \"heun\" or \"rk23\" at temperature > 0 (D42)")
    if v["method"] == "heun" and v["dt"] is None:
        errors.append(f"{where}: heun needs 'dt', its fixed step (s)")
    if v["method"] != "heun" and v["dt"] is not None:
        errors.append(f"{where}: 'dt' goes with heun; {v['method']} chooses its own steps from 'tol'")
    if hot and v["seed"] is None:
        errors.append(f"{where}: temperature > 0 needs 'seed' (D41)")
    if hot and torque:
        errors.append(f"{where}: 'until_torque' needs temperature = 0: the torque never settles under noise (D43)")
    if len(errors) > before:
        return None
    v.pop("type")
    return LLGDynamics(**v)


MATERIAL = {"Ms": (_nonnegative, REQUIRED), "A": (_nonnegative, REQUIRED), "K": (_number, 0.0), "axis": (_direction, np.array([0.0, 0, 1]))}
VACUUM = Material(0.0, 0.0, 0.0, np.array([0.0, 0, 1]))  # built in (D33)


def _id(v):
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 255:
        return None, "a whole number from 1 to 255 (0 is the background, D33)"
    return v, None


def _corners(v):
    if not (isinstance(v, list) and len(v) == 2):
        return None, "two corners [[x0, y0, z0], [x1, y1, z1]] (m)"
    corners = [_vector(c)[0] for c in v]
    if any(c is None for c in corners) or not np.all(corners[1] > corners[0]):
        return None, "two corners [[x0, y0, z0], [x1, y1, z1]] (m) with x1 > x0, y1 > y0, z1 > z0"
    return np.array(corners), None


# The keys of each shape (D33), in m from the grid's lower corner.
SHAPES = {
    "all": {},
    "cuboid": {"corners": (_corners, REQUIRED)},
    "cylinder": {"center": (_vector, REQUIRED), "radius": (_positive, REQUIRED), "height": (_positive, REQUIRED)},  # along z
    "ellipsoid": {"center": (_vector, REQUIRED), "semi_axes": (lambda v: _vector(v, lambda a: np.all(a > 0), "3 positive numbers"), REQUIRED)},
}


def _material_name(name, materials, where, errors):
    if name is not None and name not in materials:
        errors.append(f"{where}: material '{name}' is not defined (defined: {', '.join(materials)})")


def _region(raw, where, materials, errors):
    shape = raw.get("shape") if isinstance(raw, dict) else None
    if shape not in SHAPES:
        errors.append(f"{where}: 'shape' must be one of {', '.join(SHAPES)}, got {shape!r}")
        return None
    before = len(errors)
    v = _table(raw, {"id": (_id, REQUIRED), "material": (_string, REQUIRED), "shape": (_string, REQUIRED),
                     "inverse": (_bool, False), "axis": (_direction, None), **SHAPES[shape]}, f"{where} ({shape})", errors)
    _material_name(v["material"], materials, where, errors)
    if len(errors) > before:
        return None
    return Region(v["id"], v["material"], shape, {k: v[k] for k in SHAPES[shape]}, v["inverse"], v["axis"])


def _seed(v):
    return (v, None) if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else (None, "a whole number ≥ 0")


def _pair(v):
    ok = isinstance(v, list) and len(v) == 2 and all(
        e == "grains" or (isinstance(e, int) and not isinstance(e, bool) and 0 <= e <= 255) for e in v)
    return (v, None) if ok else (None, 'two region ids (0–255) or "grains"')


def _axes(v):
    if v in ("material", "random"):
        return v, None
    what = '"material", "random" or { mean = [x, y, z], cone_deg = θ } with 0 < θ ≤ 90'
    if not isinstance(v, dict) or set(v) != {"mean", "cone_deg"}:
        return None, what
    mean, err = _direction(v["mean"])
    cone, err2 = _number(v["cone_deg"], lambda x: 0 < x <= 90)
    return ({"mean": mean, "cone_deg": cone}, None) if not (err or err2) else (None, what)


def _grains(raw, cell, materials, regions, errors):
    before = len(errors)
    v = _table(raw, {"count": (_count, REQUIRED), "first_id": (_id, REQUIRED), "material": (_string, REQUIRED),
                     "seed": (_seed, REQUIRED),
                     "within": (lambda x: (x, None) if x == 0 else _id(x), 0), "axes": (_axes, "material"),
                     "boundary": (lambda x: (x, None) if isinstance(x, dict) else (None, "a table { thickness, material, id }"), None)},
               "[grains]", errors)
    _material_name(v["material"], materials, "[grains]", errors)
    if len(errors) > before:
        return None
    ids = set(range(v["first_id"], v["first_id"] + v["count"]))
    painted = {r.id for r in regions}
    if max(ids) > 255:
        errors.append(f"[grains]: ids first_id … first_id + count - 1 = {v['first_id']} … {max(ids)} must be ≤ 255 (D34)")
    if ids & painted:
        errors.append(f"[grains]: first_id … first_id + count - 1 overlaps [[region]] ids {sorted(ids & painted)}")
    if v["within"] != 0 and v["within"] not in painted:
        errors.append(f"[grains]: 'within' = {v['within']} is not a [[region]] id")
    boundary = None
    if v["boundary"] is not None:
        b = _table(v["boundary"], {"thickness": (_positive, REQUIRED), "material": (_string, REQUIRED), "id": (_id, REQUIRED)},
                   "[grains] boundary", errors)
        _material_name(b["material"], materials, "[grains] boundary", errors)
        if b["id"] is not None and (b["id"] in ids or b["id"] in painted):
            errors.append(f"[grains] boundary: id {b['id']} is already a grain or [[region]] id")
        if b["thickness"] is not None and b["thickness"] < 2 * max(cell):
            errors.append(f"[grains] boundary: thickness {b['thickness']:g} m is under 2 cells ({2 * max(cell):g} m) and cannot be "
                          "resolved; model a thinner boundary with [[coupling]] instead (D34, D35)")
        boundary = Boundary(b["thickness"], b["material"], b["id"])
    return Grains(v["count"], v["first_id"], v["material"], v["seed"], v["within"], v["axes"], boundary) if len(errors) == before else None


def _coupling(raw, where, has_grains, errors):
    before = len(errors)
    v = _table(raw, {"regions": (_pair, REQUIRED), "scale": (_nonnegative, None), "A": (_nonnegative, None)}, where, errors)
    if (v["scale"] is None) == (v["A"] is None):
        errors.append(f"{where}: give exactly one of 'scale' (× the harmonic mean) or 'A' (J/m) (D35)")
    r = v["regions"]
    if r is not None:
        if r[0] == r[1] and r[0] != "grains":
            errors.append(f"{where}: 'regions' must be two different regions, got {r}")
        if "grains" in r and not has_grains:
            errors.append(f"{where}: \"grains\" needs a [grains] table")
    return Coupling(r, v["scale"], v["A"]) if len(errors) == before else None


def _list(raw, key, errors):
    v = raw.get(key, [])
    if not isinstance(v, list):
        errors.append(f"card: '{key}' must be a list of [[{key}]] tables")
        return []
    return v


def load(path):
    """Reads and validates a card. Raises CardError listing every problem found."""
    path = Path(path)
    text = path.read_bytes()
    try:
        raw = tomllib.loads(text.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise CardError(f"{path}: not valid TOML: {e}") from None

    errors = []
    for key in raw.keys() - {"device", "grid", "material", "region", "grains", "coupling", "initial", "stage"}:
        errors.append(f"card: unknown key or section '{key}' "
                      "(valid: device, [grid], [material.<name>], [[region]], [grains], [[coupling]], [initial], [[stage]])")
    device = raw.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        errors.append(f"card: 'device' must be \"cpu\" or \"cuda\", got {device!r}")

    grid = _table(raw.get("grid", {}), {
        "n": (lambda v: _vector(v, lambda a: np.all(a >= 1), "3 whole numbers ≥ 1") if isinstance(v, list)
              and all(isinstance(x, int) and not isinstance(x, bool) for x in v) else (None, "3 whole numbers ≥ 1"), REQUIRED),
        "cell": (lambda v: _vector(v, lambda a: np.all(a > 0), "3 positive numbers"), REQUIRED),
        "demag": (_bool, True),
    }, "[grid]", errors)
    materials = {}
    tables = raw.get("material", {})
    if not tables or not all(isinstance(t, dict) for t in tables.values()):
        errors.append("card: materials are named tables: [material.<name>] with Ms, A, K, axis (D33)")
        tables = {}
    for name, t in tables.items():
        if name == "vacuum":
            errors.append("[material.vacuum]: 'vacuum' is built in (Ms = 0) and cannot be redefined")
            continue
        materials[name] = Material(**_table(t, MATERIAL, f"[material.{name}]", errors))
    known = {**materials, "vacuum": VACUUM}
    initial = _table(raw.get("initial", {}), {
        "m": (_direction, None), "file": (_string, None), "state": (_string, None),
    }, "[initial]", errors)
    if (initial["m"] is None) == (initial["file"] is None):
        errors.append("[initial]: give exactly one of 'm' (a uniform direction) or 'file' (a trajectory.h5)")
    if initial["state"] is not None and (initial["file"] is None or not re.fullmatch(r"final|stage[1-9][0-9]*", initial["state"])):
        errors.append("[initial]: 'state' goes with 'file' and must be \"final\" or \"stage<k>\"")

    regions = [_region(r, f"region {k}", known, errors) for k, r in enumerate(_list(raw, "region", errors), 1)]
    first = {}
    for k, r in enumerate(regions, 1):  # one id, one material and axis (D33)
        if r is None:
            continue
        material, axis, k0 = first.setdefault(r.id, (r.material, r.axis, k))
        same_axis = (axis is None and r.axis is None) or (axis is not None and r.axis is not None and np.array_equal(axis, r.axis))
        if material != r.material or not same_axis:
            errors.append(f"region {k}: id {r.id} was painted by region {k0} with another material or axis")
    regions = [r for r in regions if r is not None]
    grains = None
    if "grains" in raw:
        grains = _grains(raw["grains"], grid["cell"] if grid["cell"] is not None else [0.0], known, regions, errors)
    couplings = [_coupling(c, f"coupling {k}", "grains" in raw, errors) for k, c in enumerate(_list(raw, "coupling", errors), 1)]

    stages = raw.get("stage")
    if not isinstance(stages, list) or not stages:
        errors.append("card: no stages: add at least one [[stage]]")
        stages = []
    stages = [_stage(s, f"stage {k}", errors) for k, s in enumerate(stages, 1)]

    if errors:
        raise CardError(f"{path}:\n  " + "\n  ".join(errors))
    file = (path.parent / initial["file"]) if initial["file"] else None
    return Card(path, text, device, tuple(int(x) for x in grid["n"]), tuple(float(x) for x in grid["cell"]), grid["demag"], materials,
                initial["m"], file, initial["state"] or "final", stages, regions, grains, couplings)
