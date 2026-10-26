"""Run one card into a results folder (D28): parameters in, results out.

    micromag run CARD -o DIR [--device cuda]

The folder holds the card, provenance.json, one macrostate table per stage (stage<k>_<type>.csv, like LAMMPS thermo
output) and the microstates in trajectory.h5 (like a LAMMPS dump). See D27 for the card and D28 for the folder.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

from . import LLG, Grid, Minimizer, Problem, mu0
from .card import CardError, Hysteresis, LLGDynamics, Minimize, load
from .regions import paint

ENERGIES = ("exchange", "anisotropy", "zeeman", "demag")
COLUMNS = {
    "llg-dynamics": ["t", "mu0_Hx", "mu0_Hy", "mu0_Hz", "mx", "my", "mz",
                     *(f"E_{e}" for e in ENERGIES), "E_total", "steps", "rejected"],
    "minimize": ["mu0_Hx", "mu0_Hy", "mu0_Hz", "mx", "my", "mz", *(f"E_{e}" for e in ENERGIES), "E_total", "iterations"],
}
COLUMNS["hysteresis"] = COLUMNS["minimize"]


def git_state(path):
    """{hash, dirty, diff} of the git checkout containing path, or None outside one. Untracked files count as dirty."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout
    try:
        return {"hash": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain").strip()),
                "diff": git("diff", "HEAD")}
    except (OSError, subprocess.CalledProcessError):
        return None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return out.stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _initial_state(card):
    """m to start from (3, nz, ny, nx), and its provenance entry. Raises CardError before anything is written."""
    nx, ny, nz = card.n
    if card.m0 is not None:
        m = np.empty((3, nz, ny, nx))  # a fresh writable array: a broadcast view of a 1-cell grid would be read-only
        m[:] = card.m0[:, None, None, None]
        return m, {"m": card.m0.tolist()}
    try:
        f = h5py.File(card.file, "r")
    except OSError as e:
        raise CardError(f"[initial]: cannot read {card.file}: {e}") from None
    with f:
        if tuple(int(x) for x in f.attrs["n"]) != card.n:
            raise CardError(f"[initial]: {card.file} has grid n = {list(f.attrs['n'])}, but the card has {list(card.n)}")
        if card.state not in f:
            raise CardError(f"[initial]: {card.file} has no state '{card.state}' (it has: {', '.join(f)})")
        d = f[card.state]["m"]
        m = np.array(d[()] if d.ndim == 4 else d[-1], dtype=np.float64)  # final/failed: one state; stage<k>: its last snapshot
    sha = hashlib.sha256(Path(card.file).read_bytes()).hexdigest()
    return m, {"file": str(card.file), "state": card.state, "sha256": sha}


class Macrostate:
    """One stage's table of ⟨m⟩, energies and solver effort, flushed row by row so `tail -f` works (D28)."""

    def __init__(self, path, columns):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)
        self.file.flush()

    def row(self, values):
        self.writer.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in values])  # exact floats
        self.file.flush()

    def close(self):
        self.file.close()


class Microstate:
    """trajectory.h5, the full m: per stage an extendable stack of m with its times or fields; float64, shuffle + gzip (D28)."""

    def __init__(self, path, card):
        self.f = h5py.File(path, "w")
        self.shape = (3, card.n[2], card.n[1], card.n[0])
        self.f.attrs.update(n=list(card.n), cell=list(card.cell), cell_units="m", t_units="s", mu0_H_units="T",
                            m_layout="m[snapshot, component, z, y, x], x fastest; unit vectors")

    def stage(self, k, kind):
        g = self.f.create_group(f"stage{k}")
        g.attrs["type"] = kind
        g.create_dataset("m", shape=(0, *self.shape), maxshape=(None, *self.shape), chunks=(1, *self.shape),
                         dtype="f8", compression="gzip", shuffle=True)
        if kind == "llg-dynamics":
            g.create_dataset("t", shape=(0,), maxshape=(None,), dtype="f8")
        else:
            g.create_dataset("mu0_H", shape=(0, 3), maxshape=(None, 3), dtype="f8")
        return g

    def append(self, g, m, label):
        name = "t" if "t" in g else "mu0_H"
        for ds, value in ((g["m"], m), (g[name], label)):
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = value
        self.f.flush()

    def regions(self, painted):
        """The region map (D33), with its id -> material names as JSON."""
        d = self.f.create_dataset("regions", data=painted.regions, compression="gzip", shuffle=True)
        d.attrs["materials"] = json.dumps({str(k): v for k, v in painted.names.items()})
        d.attrs["layout"] = "regions[z, y, x], x fastest; uint8 ids"

    def single(self, name, m):
        self.f.create_group(name).create_dataset("m", data=m, compression="gzip", shuffle=True)
        self.f.flush()

    def close(self):
        self.f.close()


def _schedule(end, rows_every, dump_every):
    """[(t, row, dump)] in time order: rows at 0, every rows_every and at end; dumps every dump_every from 0, and at end."""
    marks = [(0.0, True, False), (end, True, True)]
    for every, is_row in ((rows_every, True), (dump_every, False)):
        if every is not None:
            marks += [(j * every, is_row, not is_row) for j in range(int(end / every * (1 + 1e-12)) + 1)]
    marks.sort()
    tol = 1e-9 * min(x for x in (rows_every, dump_every, end) if x is not None)
    out = []
    for t, row, dump in marks:  # k·a and j·b can differ by rounding where they should coincide
        if out and t - out[-1][0] <= tol:
            out[-1] = (t, out[-1][1] or row, out[-1][2] or dump)
        else:
            out.append((t, row, dump))
    return [(min(t, end), row, dump) for t, row, dump in out]


def run(card, out, device=None):
    """Runs a card into the folder out, which must be new or empty. device overrides the card's (D28).

    Raises CardError for a bad card or initial state, FileExistsError for a non-empty out, and re-raises a stage's
    error after recording it, keeping everything written so far."""
    card = load(card)
    m, initial = _initial_state(card)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: refusing to overwrite a result (D28)")
    device = device or card.device

    nx, ny, nz = card.n
    p = paint(card)
    prob = Problem(Grid(nx, ny, nz, *card.cell), p.Ms, p.A, p.K, p.axis, (0, 0, 0), device=device,
                   regions=p.regions if p.coupling is not None else None, coupling=p.coupling, demag=card.demag)
    magnetic = prob.Ms > 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "card.toml").write_bytes(card.text)
    git = git_state(Path(__file__).parent)
    if git is None:
        print("micromag: warning: not running from a git checkout; provenance has no code version", file=sys.stderr)
    elif git["dirty"]:
        print("micromag: warning: uncommitted changes; recorded in git.diff", file=sys.stderr)
        (out / "git.diff").write_text(git["diff"])
    prov = {
        "card": card.path.name, "status": "running", "error": None, "started": _now(), "finished": None,
        "device": device, "gpu": _gpu_name() if device == "cuda" else None,
        "versions": {"micromag": version("micromag"), "numpy": np.__version__, "h5py": h5py.__version__,
                     "python": platform.python_version()},
        "git": None if git is None else {"hash": git["hash"], "dirty": git["dirty"]},
        "initial": initial, "regions": p.report, "stages": [],
    }
    for r in p.report:  # staircasing or clipping (D33)
        if r["volume_m3"] and abs(r["cells_m3"] / r["volume_m3"] - 1) > 0.05:
            print(f"micromag: warning: region {r['region']} (id {r['id']}, {r['shape']}): painted volume {r['cells_m3']:.4g} m³ "
                  f"differs from the shape's {r['volume_m3']:.4g} m³ by {100 * (r['cells_m3'] / r['volume_m3'] - 1):+.1f}%",
                  file=sys.stderr)

    def save():
        (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    def measure():
        E = prob.energies(m)
        e = [E[t] for t in ENERGIES]
        return [*m[:, magnetic].mean(axis=1), *e, sum(e)]

    save()
    micro = Microstate(out / "trajectory.h5", card)
    micro.regions(p)
    try:
        for k, stage in enumerate(card.stages, 1):
            start = time.perf_counter()
            macro = Macrostate(out / f"stage{k}_{stage.type}.csv", COLUMNS[stage.type])
            group = micro.stage(k, stage.type)
            try:
                summary, extra = _run_stage(stage, k, prob, m, magnetic, measure, macro, micro, group)
            except Exception as e:
                micro.single("failed", m)
                prov.update(status="failed", error=f"stage {k} ({stage.type}): {e}", finished=_now())
                save()
                print(f"stage {k}/{len(card.stages)} {stage.type}: FAILED: {e}", file=sys.stderr)
                raise
            finally:
                macro.close()
            wall = time.perf_counter() - start
            prov["stages"].append({"type": stage.type, "wall_s": wall, **extra})
            save()
            print(f"stage {k}/{len(card.stages)} {stage.type}: {summary} ({wall:.1f} s)", flush=True)
        micro.single("final", m)
    finally:
        micro.close()
    prov.update(status="completed", finished=_now())
    save()
    return out


def _run_stage(stage, k, prob, m, magnetic, measure, macro, micro, group):
    """Runs one stage on m in place, writing its rows and snapshots. Returns the progress-line summary."""
    if isinstance(stage, Hysteresis):
        return _hysteresis(stage, prob, m, magnetic, measure, macro, micro, group)

    prob.H_ext = tuple(stage.mu0_H / mu0)  # stage-local: every stage sets its own field, zero by default (D27)
    field = f"µ0H = {stage.mu0_H.tolist()} T"
    if isinstance(stage, Minimize):
        it = Minimizer(prob).minimize(m, stage.torque_tol, stage.max_iter)
        macro.row([*stage.mu0_H, *measure(), it])
        micro.append(group, m, stage.mu0_H)
        return f"{it} iterations at {field}", {}

    assert isinstance(stage, LLGDynamics)
    llg = LLG(prob, stage.alpha, gamma=stage.gamma, tol=stage.tol, max_step=stage.max_step, method=stage.method,
              temperature=stage.temperature, dt=stage.dt or 0.0, seed=stage.seed or 0, stage=k)
    timed = stage.duration is not None
    end = stage.duration if timed else stage.max_duration
    H = np.empty_like(m)
    for t, row, snap in _schedule(end, stage.sample_every if timed else stage.check_every, stage.dump_every):
        if t > 0:
            llg.run_until(m, t)
        done = False
        if row:
            macro.row([t, *stage.mu0_H, *measure(), llg.steps, llg.rejected])
            if not timed:
                prob.effective_field(m, H)
                torque = (np.linalg.norm(np.cross(m, H, axis=0), axis=0)[magnetic] / prob.Ms[magnetic]).max()
                done = torque < stage.until_torque
                if not done and t >= end:
                    raise RuntimeError(f"until_torque = {stage.until_torque:g} not reached by max_duration = {end:g} s "
                                       f"(max |m×H|/Ms = {torque:.2e})")
        if snap or done:
            micro.append(group, m, t)
        if done:
            break
    step = {"dt": stage.dt} if stage.method == "heun" else {"tol": stage.tol}
    return (f"{llg.t * 1e9:.3f} ns, {llg.steps} steps, {llg.rejected} rejected, at {field}",
            {"temperature": stage.temperature, "method": stage.method, **step, "seed": stage.seed, "steps": llg.steps,
             "rejected": llg.rejected, "max_bridge_depth": llg.max_bridge_depth})  # D43


def _hysteresis(stage, prob, m, magnetic, measure, macro, micro, group):
    """The ramp of fields_T(), minimizing at each field (D22). With refine_to, a step that crosses m·h = 0 or makes
    <m> jump by more than refine_jump is undone and halved, down to refine_to; the ramp then resumes (D31)."""
    h = stage.direction
    grid = stage.fields_T() @ h  # field values along h, from -> to
    mn = Minimizer(prob)
    rows, iterations = [], 0     # (field along h, <m>) of every accepted field

    def settle(b):
        nonlocal iterations
        prob.H_ext = tuple(b * h / mu0)
        iterations += mn.minimize(m, stage.torque_tol, stage.max_iter)
        return m[:, magnetic].mean(axis=1)

    def accept(b, mean):
        rows.append((b, mean))
        macro.row([*(b * h), *measure(), iterations])
        if stage.dump or b == grid[-1]:
            micro.append(group, m, b * h)

    accept(grid[0], settle(grid[0]))
    for target in grid[1:]:
        while rows[-1][0] != target:
            b0, mean0 = rows[-1]
            saved, trial = m.copy(), target
            while True:
                mean = settle(trial)
                event = np.sign(mean0 @ h) != np.sign(mean @ h) or np.linalg.norm(mean - mean0) > stage.refine_jump
                if stage.refine_to is None or not event or abs(trial - b0) <= stage.refine_to:
                    break
                m[...] = saved  # undo the step and try half of it
                trial = b0 + (trial - b0) / 2
            accept(trial, mean)

    b = np.array([r[0] for r in rows])
    mh = np.array([r[1] @ h for r in rows])
    jumps = np.linalg.norm(np.diff([r[1] for r in rows], axis=0), axis=1)
    i = next((k for k in range(len(mh) - 1) if np.sign(mh[k]) != np.sign(mh[k + 1])), None)
    zero = np.flatnonzero(np.abs(b) <= 1e-9 * stage.mu0_H_step)
    summary = {
        "Hc_T": None if i is None else float(b[i] + (b[i + 1] - b[i]) * mh[i] / (mh[i] - mh[i + 1])),
        "switch_T": None if len(jumps) == 0 else float((b[jumps.argmax()] + b[jumps.argmax() + 1]) / 2),
        "remanence": rows[zero[0]][1].tolist() if len(zero) else None,
    }
    fmt = lambda v: "none" if v is None else f"{v:.4g} T"
    return (f"{len(rows)} fields, µ0H {stage.mu0_H_from:g} → {stage.mu0_H_to:g} T along {np.round(h, 4).tolist()}, "
            f"{iterations} iterations; Hc {fmt(summary['Hc_T'])}, switching {fmt(summary['switch_T'])}"), {"summary": summary}


def main(argv=None):
    p = argparse.ArgumentParser(prog="micromag", description="Finite-difference micromagnetics.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run one input card into a results folder")
    r.add_argument("card", help="the input card (.toml)")
    r.add_argument("-o", "--out", required=True, help="results folder; must be new or empty")
    r.add_argument("--device", choices=["cpu", "cuda"], help="overrides the card's device")
    a = p.parse_args(argv)
    try:
        run(a.card, a.out, a.device)
    except (CardError, FileExistsError) as e:
        print(f"micromag: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:  # a failed stage (already recorded in the folder) or a solver error
        print(f"micromag: run failed: {e}", file=sys.stderr)
        return 1
    return 0

