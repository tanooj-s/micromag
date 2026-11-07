"""How the 40 nm cube's GNEB barrier depends on the cell size (D56): chains at four meshes, then Newton from every image.

    uv run --group docs --with scipy python benchmarks/barrier_cell_size.py

Runs the study into barrier_cell_size.csv and _points.csv if the first is absent, then draws barrier_cell_size.pdf. Delete
the CSV to run again. scipy is for this study only, so it is not a project dependency.
"""

import csv
import multiprocessing
import textwrap
import time
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import micromag as mm  # noqa: E402
from micromag.gneb import MAX_ROTATION, _rotate, distance, gneb  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV, POINTS, PDF = HERE / "barrier_cell_size.csv", HERE / "barrier_cell_size_points.csv", HERE / "barrier_cell_size.pdf"

L = 40e-9                # the cube of D32
MS, A, K = 1.61 / mm.mu0, 7.7e-12, 4.3e6
DELTA0 = np.sqrt(A / K)
AXIS = np.array([np.sin(np.radians(1)), 0.0, np.cos(np.radians(1))])
B = -3.94                # T, along AXIS
IMAGES = 10
TOL = 1e-4
MAX_CHAIN = 2000
MAX_NEWTON = 50
FORCING = 1e-2           # MINRES stops at this relative residual
KICK = 0.03              # rad on the largest cell: D54's 0.1 rad could overshoot
MESHES = (25, 30, 35, 40)
SEEDS = 3
KT = mm.kB * 300
COARSE = (15, 14.2492)   # D54: the highest pass at 15³, for the plot only

# The dataviz reference palette: one series in blue, the reference lines in the ink roles.
BLUE, INK, INK2, MUTED, GRID, AXIS_INK = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

CONDITIONS = ("The 40 nm Nd₂Fe₁₄B cube of D32: µ₀Ms = 1.61 T, K = 4.3 MJ/m³, A = 7.7 pJ/m, so δ₀ = 1.34 nm. Field −3.94 T, 1° off z, 10 images, torque 1e-4, limit 2000 iterations, 3 seeds, "
              "L-BFGS as in gneb(). Newton from all 8 inner images of each chain, curvatures by Lanczos, pushes of 0.03 rad. Reversed cells are those with m·m_start < 0 at the saddle, so the "
              "volume is a rough measure at about ±½ cell. The fit is E₀ + c a² through the four meshes, and E₀ is an extrapolation below the finest cell run. The 15³ point is from D54 and is "
              "not in the fit. The 25 k_BT line is the published criterion at this field (Fischbacher et al. 2017), from finite elements at 1.5 nm. Meshes 16–24 were not run, so where the "
              "staircase starts is unknown. Decision D56. Regenerate with benchmarks/barrier_cell_size.py.")


def cube(n):
    g = mm.Grid(n, n, n, L / n, L / n, L / n)
    s = (n, n, n)
    k = np.zeros((3, *s))
    k[2] = 1
    prob = mm.Problem(g, np.full(s, MS), np.full(s, A), np.full(s, K), k, tuple(B * AXIS / mm.mu0), device="cpu", demag=True)
    start = np.broadcast_to(AXIS.reshape(3, 1, 1, 1), (3, *s)).copy()
    mn = mm.Minimizer(prob)
    mn.minimize(start, TOL, 100_000)
    end = -start
    mn.minimize(end, TOL, 100_000)
    return g, prob, start, end


class Tracker:
    """Stands in for Problem inside gneb(): counts field evaluations and keeps the states of the last sweep over the images."""

    def __init__(self, prob):
        self.prob, self.calls, self.last = prob, 0, deque(maxlen=IMAGES - 2)

    def __getattr__(self, name):
        return getattr(self.prob, name)

    def effective_field(self, m, H):
        self.prob.effective_field(m, H)
        self.calls += 1
        self.last.append(m.copy())


def total_energy(grid, prob, m, H):
    return mm.energy(grid, m, prob.Ms, H + np.array(prob.H_ext).reshape(3, 1, 1, 1), 0.5)  # D48 Correction


def torque(m, H):
    return float((np.linalg.norm(np.cross(m, H, axis=0), axis=0) / MS).max())


class Curvature:
    """The Hessian of E on the product of spheres at m, in units of µ0 Ms V / Ms: A v = -P(L v) + (m·H) v, on a local (e1, e2)
    frame per cell so the operator has no null space from the normals. L v is the field of v without H_ext: one evaluation."""

    def __init__(self, prob, m):
        from scipy.sparse.linalg import LinearOperator

        self.prob, self.m, self.products = prob, m, 0
        self.H_ext = np.array(prob.H_ext).reshape(3, 1, 1, 1)
        self.H = np.empty_like(m)
        prob.effective_field(m, self.H)
        self.mH = (m * self.H).sum(axis=0)
        ref = np.zeros_like(m)
        along = np.abs(m[0]) > 0.9
        ref[0], ref[1] = ~along, along
        self.e1 = ref - (ref * m).sum(axis=0) * m
        self.e1 /= np.linalg.norm(self.e1, axis=0)
        self.e2 = np.cross(m, self.e1, axis=0)
        self.size = 2 * m[0].size
        self.op = LinearOperator((self.size, self.size), matvec=self.apply, dtype=float)

    def lift(self, x):
        a, b = x.reshape(2, *self.m.shape[1:])
        return a * self.e1 + b * self.e2

    def drop(self, v):
        return np.concatenate([(self.e1 * v).sum(axis=0).ravel(), (self.e2 * v).sum(axis=0).ravel()])

    def apply(self, x):
        self.products += 1
        v = np.ascontiguousarray(self.lift(np.asarray(x).ravel()))
        Lv = np.empty_like(v)
        self.prob.effective_field(v, Lv)
        return self.drop(self.mH * v - (Lv - self.H_ext)) / MS

    def force(self):
        return self.drop(self.H - self.mH * self.m) / MS


def newton(prob, grid, m):
    """Newton on one state: A d = F by MINRES, then a per-cell rotation capped as in gneb(). Returns state, status, E, steps."""
    from scipy.sparse.linalg import minres

    m, first = m.copy(), None
    for step in range(MAX_NEWTON + 1):
        c = Curvature(prob, m)
        t = torque(m, c.H)
        first = first or t
        E = total_energy(grid, prob, m, c.H)
        if t < TOL:
            return m, "converged", E, step
        if t > 100 * first or not np.isfinite(t):
            return m, "diverged", E, step
        if step == MAX_NEWTON:
            return m, "stalled", E, step
        x, _ = minres(c.op, c.force(), rtol=FORCING, maxiter=2000)
        d = c.lift(x)
        largest = float(np.linalg.norm(d, axis=0).max())
        if largest > MAX_ROTATION:
            d = d * (MAX_ROTATION / largest)
        m = _rotate(m[None], d[None])[0]


def modes(prob, m, seed, k=3):
    """The k lowest curvatures in T and their eigenvectors as tangent fields."""
    from scipy.sparse.linalg import eigsh

    c = Curvature(prob, m)
    vals, vecs = eigsh(c.op, k=k, which="SA", tol=1e-3, v0=np.random.default_rng(seed).normal(size=c.size), ncv=60, maxiter=200)
    order = np.argsort(vals)
    return vals[order] * MS * mm.mu0, [c.lift(vecs[:, i]) for i in order]


def sides(prob, grid, m, mode, start, end, w, E0):
    """Pushes a saddle both ways along its negative mode, relaxes, and names where each side lands."""
    d = mode * (KICK / float(np.linalg.norm(mode, axis=0).max()))
    out = []
    mn = mm.Minimizer(prob)
    H = np.empty_like(m)
    for sign in (1, -1):
        x = _rotate(m[None], sign * d[None])[0]
        mn.minimize(x, TOL, 100_000)
        a, b = distance(x, start, w), distance(x, end, w)
        prob.effective_field(x, H)
        out.append("start" if a < 1e-2 else "end" if b < 1e-2 else f"min {(total_energy(grid, prob, x, H) - E0) / KT:.3f} kT")
    return sorted(out, reverse=True)


def run_job(job):
    n, seed = job
    t0 = time.perf_counter()
    grid, prob, start, end = cube(n)
    w = np.full(start.shape[1:], 1 / start[0].size)
    H = np.empty_like(start)
    prob.effective_field(start, H)
    E0 = total_energy(grid, prob, start, H)
    tr = Tracker(prob)
    try:
        path = gneb(tr, grid, start, end, IMAGES, TOL, MAX_CHAIN, seed)
        status, iterations, resets, states, barrier = "converged", path.iterations, path.resets, list(path.m[1:-1]), path.barrier / KT
    except RuntimeError:
        status, iterations, resets, states, barrier = "failed", MAX_CHAIN, -1, list(tr.last), float("nan")
    chain_s, evals = time.perf_counter() - t0, tr.calls
    points = []
    for i, s in enumerate(states):
        m, kind, E, steps = newton(prob, grid, s)
        negative, lam1, landed = -1, float("nan"), ["-", "-"]
        if kind == "converged":
            lam, vecs = modes(prob, m, seed)
            negative, lam1 = int((lam < 0).sum()), lam[0]
            if negative == 1:
                landed = sides(prob, grid, m, vecs[0], start, end, w, E0)
        points.append(dict(n=n, seed=seed, image=i + 1, newton=kind, steps=steps, kT=f"{(E - E0) / KT:.4f}", negative=negative, lam1_T=f"{lam1:.4f}",
                           side_a=landed[0], side_b=landed[1], reversed_cells=int(((m * start).sum(axis=0) < 0).sum())))
    row = dict(n=n, cell_nm=f"{L / n * 1e9:.3f}", cell_delta0=f"{L / n / DELTA0:.3f}", seed=seed, chain=status, iterations=iterations, resets=resets,
               evals=evals, chain_s=round(chain_s), total_s=round(time.perf_counter() - t0), barrier_kT=f"{barrier:.4f}")
    return row, points


def check():
    """The operator against central differences of the force on a random state."""
    rng = np.random.default_rng(0)
    _, prob, _, _ = cube(6)
    m = rng.normal(size=(3, 6, 6, 6))
    m /= np.linalg.norm(m, axis=0)
    c = Curvature(prob, m)
    v, eps = c.lift(rng.normal(size=c.size)), 1e-5
    v /= np.linalg.norm(v, axis=0).max()

    def force(state):
        H = np.empty_like(state)
        prob.effective_field(state, H)
        return H - (H * state).sum(axis=0) * state

    diff = (force(_rotate(m[None], eps * v[None])[0]) - force(_rotate(m[None], -eps * v[None])[0])) / (2 * eps)
    exact = c.lift(c.apply(c.drop(v))) * MS
    error = np.abs(exact + diff - (diff * m).sum(axis=0) * m).max() / np.abs(exact).max()
    assert error < 1e-6, error
    print(f"operator check passed: {error:.1e}", flush=True)


def sweep():
    check()
    jobs = [(n, s) for n in reversed(MESHES) for s in range(SEEDS)]
    rows, points = [], []
    with multiprocessing.Pool(6) as pool:
        for i, (row, pts) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            rows.append(row)
            points += pts
            print(f"{i}/{len(jobs)} {row}", flush=True)
    rows.sort(key=lambda r: (r["n"], r["seed"]))
    for path, data in ((CSV, rows), (POINTS, points)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fit(a, E):
    c, E0 = np.polyfit(np.array(a) ** 2, E, 1)
    return E0, c


def draw():
    rows, points = read(CSV), read(POINTS)
    a = [L / n * 1e9 for n in MESHES]
    E = [float(next(r for r in rows if int(r["n"]) == n)["barrier_kT"]) for n in MESHES]
    E0, c = fit(a, E)
    E0_fine, _ = fit(a[1:], E[1:])
    rel = c * (DELTA0 * 1e9) ** 2 / E0  # fractional error at a = δ0
    cells = {n: next(int(p["reversed_cells"]) for p in points if int(p["n"]) == n and p["negative"] == "1") for n in MESHES}
    volume = {n: cells[n] * (L / n * 1e9) ** 3 for n in MESHES}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS_INK, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.955, "Energy barrier against cell size", fontsize=16, weight="bold")
    finding = (f"From 25³ to 40³ the landscape is one clean pass: all 12 chains converge, seeds agree to four decimals, and all 96 Newton starts land on that pass or the "
               f"start. The barrier still rises as the mesh is refined, from {E[0]:.2f} to {E[-1]:.2f} k_BT, and extrapolates to about {E0:.1f} k_BT. A cell of δ₀ reads about "
               f"{abs(rel) * 100:.1f}% low, and error scales as a². The nucleus has the same volume on every mesh, about {np.mean(list(volume.values())):.0f} nm³, which is less than one 15³ cell (19 nm³).")
    fig.text(0.08, 0.862, textwrap.fill(finding, 112), fontsize=9.3, color=INK2, linespacing=1.4)

    ax = fig.add_axes([0.1, 0.5, 0.82, 0.33])
    x2 = np.linspace(0, 7.3, 50)
    ax.plot(x2, E0 + c * x2, color=BLUE, linewidth=1.2, linestyle="--", label=f"fit: {E0:.2f} − {abs(c):.2f} a²  (a in nm)", zorder=2)
    ax.scatter(np.array(a) ** 2, E, s=46, color=BLUE, edgecolors="white", linewidths=1, zorder=3, label="25³–40³, 3 seeds each (seeds coincide)")
    ax.scatter([(L / COARSE[0] * 1e9) ** 2], [COARSE[1]], s=46, facecolors="white", edgecolors=BLUE, linewidths=1.2, zorder=3, label="15³, highest pass (D54), not in the fit")
    for n, x, y in zip(MESHES, a, E):
        ax.annotate(f"{n}³", (x**2, y), xytext=(6, 6), textcoords="offset points", color=INK2)
    ax.annotate("15³", ((L / COARSE[0] * 1e9) ** 2, COARSE[1]), xytext=(6, 6), textcoords="offset points", color=INK2)
    ax.axhline(25, color=MUTED, linewidth=0.9)
    ax.annotate("25 k_BT, the published criterion at 3.94 T", (7.3, 25), xytext=(0, 4), textcoords="offset points", ha="right", color=INK2, fontsize=7.5)
    ax.axvline((DELTA0 * 1e9) ** 2, color=MUTED, linewidth=0.9)
    ax.annotate("a = δ₀", ((DELTA0 * 1e9) ** 2, 13), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(12, 26.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("cell size squared, a² (nm²)")
    ax.set_ylabel("barrier at −3.94 T (k_BT)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9), labelcolor=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    tab = fig.add_axes([0.08, 0.2, 0.84, 0.235])
    tab.axis("off")
    head = ["mesh", "a (nm)", "a / δ₀", "converged", "iterations", "barrier", "vs E₀", "mode (T)", "reversed", "nm³"]
    body = []
    for n, x, y in zip(MESHES, a, E):
        r = [q for q in rows if int(q["n"]) == n]
        its = [int(q["iterations"]) for q in r]
        lam = next(float(p["lam1_T"]) for p in points if int(p["n"]) == n and p["negative"] == "1")
        body.append([f"{n}³", f"{x:.3f}", f"{x * 1e-9 / DELTA0:.3f}", f"{sum(q['chain'] == 'converged' for q in r)} of {SEEDS}", f"{min(its)}–{max(its)}", f"{y:.4f}",
                     f"{100 * (y / E0 - 1):+.1f}%", f"{lam:.3f}", cells[n], f"{volume[n]:.1f}"])
    x15 = L / COARSE[0] * 1e9
    body.append(["15³ (D54)", f"{x15:.3f}", f"{x15 * 1e-9 / DELTA0:.3f}", "0 of 3 at 1e-4", "—", f"{COARSE[1]:.4f}", f"{100 * (COARSE[1] / E0 - 1):+.1f}%", "−3.510", "0–1", "one cell is 19"])
    t = tab.table(cellText=body, colLabels=head, cellLoc="center", loc="upper center", bbox=[0, 0, 1, 1],
                  colWidths=[0.08, 0.08, 0.08, 0.12, 0.1, 0.11, 0.08, 0.12, 0.1, 0.11])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        cell.visible_edges = "horizontal"
        if r == 0:
            cell.set_text_props(color=INK2, weight="bold")

    both = sum(p["side_a"] == "start" and p["side_b"] == "end" for p in points)
    pushed = sum(p["negative"] == "1" for p in points)
    note = (f"Barrier in k_BT. Mode is the lowest curvature; reversed and nm³ describe the nucleus at the saddle. Fit through all four meshes: E₀ = {E0:.2f} k_BT; through the three finest: "
            f"{E0_fine:.2f} k_BT. Pushes reached start and end on {both} of {pushed} saddles, and both sides reached the end on the others.")
    fig.text(0.08, 0.15, textwrap.fill(note, 135), fontsize=7.8, color=INK2, linespacing=1.4)
    fig.text(0.08, 0.05, textwrap.fill(CONDITIONS, 150), fontsize=7, color=INK2, linespacing=1.45)
    fig.savefig(PDF)
    plt.close(fig)
    print(f"wrote {PDF}; E0 {E0:.3f} (three finest {E0_fine:.3f}), c {c:.3f}, error at delta0 {rel:.4f}, volumes {volume}, pushes {both}/{pushed}")


if __name__ == "__main__":
    if not CSV.exists():
        sweep()
    draw()
