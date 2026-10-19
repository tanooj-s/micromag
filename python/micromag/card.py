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
class Region:
    box: np.ndarray  # [[x0, y0, z0], [x1, y1, z1]], m from the grid's lower corner
    values: dict     # material keys it overwrites (D30)


@dataclass
class Card:
    path: Path
    text: bytes  # the card as read, for the byte-for-byte copy (D28)
    device: str
    n: tuple     # cells along x, y, z
    cell: tuple  # m
    Ms: float
    A: float
    K: float
    axis: np.ndarray
    m0: np.ndarray | None  # uniform start, normalised
    file: Path | None      # or a trajectory.h5 to start from ...
    state: str             # ... and which state in it
    stages: list
    regions: list          # ordered; later boxes win (D30)


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
    v.pop("type")
    return LLGDynamics(**v)


def _box(v):
    if not (isinstance(v, list) and len(v) == 2):
        return None, "two corners [[x0, y0, z0], [x1, y1, z1]] (m)"
    corners = [_vector(c)[0] for c in v]
    if any(c is None for c in corners) or not np.all(corners[1] > corners[0]):
        return None, "two corners [[x0, y0, z0], [x1, y1, z1]] (m) with x1 > x0, y1 > y0, z1 > z0"
    return np.array(corners), None


def _region(raw, where, errors):
    before = len(errors)
    v = _table(raw, {"box": (_box, REQUIRED), "Ms": (_nonnegative, None), "A": (_nonnegative, None),
                     "K": (_number, None), "axis": (_direction, None)}, where, errors)
    values = {k: x for k, x in v.items() if k != "box" and x is not None}
    if len(errors) == before and not values:
        errors.append(f"{where}: sets nothing: give at least one of Ms, A, K, axis")
    return Region(v["box"], values) if len(errors) == before else None


def load(path):
    """Reads and validates a card. Raises CardError listing every problem found."""
    path = Path(path)
    text = path.read_bytes()
    try:
        raw = tomllib.loads(text.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise CardError(f"{path}: not valid TOML: {e}") from None

    errors = []
    for key in raw.keys() - {"device", "grid", "material", "initial", "region", "stage"}:
        errors.append(f"card: unknown key or section '{key}' (valid: device, [grid], [material], [[region]], [initial], [[stage]])")
    device = raw.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        errors.append(f"card: 'device' must be \"cpu\" or \"cuda\", got {device!r}")

    grid = _table(raw.get("grid", {}), {
        "n": (lambda v: _vector(v, lambda a: np.all(a >= 1), "3 whole numbers ≥ 1") if isinstance(v, list)
              and all(isinstance(x, int) and not isinstance(x, bool) for x in v) else (None, "3 whole numbers ≥ 1"), REQUIRED),
        "cell": (lambda v: _vector(v, lambda a: np.all(a > 0), "3 positive numbers"), REQUIRED),
    }, "[grid]", errors)
    material = _table(raw.get("material", {}), {
        "Ms": (_positive, REQUIRED), "A": (_nonnegative, REQUIRED), "K": (_number, 0.0), "axis": (_direction, np.array([0.0, 0, 1])),
    }, "[material]", errors)
    initial = _table(raw.get("initial", {}), {
        "m": (_direction, None), "file": (_string, None), "state": (_string, None),
    }, "[initial]", errors)
    if (initial["m"] is None) == (initial["file"] is None):
        errors.append("[initial]: give exactly one of 'm' (a uniform direction) or 'file' (a trajectory.h5)")
    if initial["state"] is not None and (initial["file"] is None or not re.fullmatch(r"final|stage[1-9][0-9]*", initial["state"])):
        errors.append("[initial]: 'state' goes with 'file' and must be \"final\" or \"stage<k>\"")

    regions = raw.get("region", [])
    if not isinstance(regions, list):
        errors.append("card: 'region' must be a list of [[region]] tables")
        regions = []
    regions = [_region(r, f"region {k}", errors) for k, r in enumerate(regions, 1)]

    stages = raw.get("stage")
    if not isinstance(stages, list) or not stages:
        errors.append("card: no stages: add at least one [[stage]]")
        stages = []
    stages = [_stage(s, f"stage {k}", errors) for k, s in enumerate(stages, 1)]

    if errors:
        raise CardError(f"{path}:\n  " + "\n  ".join(errors))
    file = (path.parent / initial["file"]) if initial["file"] else None
    return Card(path, text, device, tuple(int(x) for x in grid["n"]), tuple(float(x) for x in grid["cell"]), material["Ms"], material["A"],
                material["K"], material["axis"], initial["m"], file, initial["state"] or "final", stages, regions)
