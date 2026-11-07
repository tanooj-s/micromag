"""How ⟨m⟩(T) of a film depends on the cell size (D52): a sweep and its one-page summary (about 15 min on a laptop CPU).

    uv run --group docs python benchmarks/thermal_cell_size.py

Runs the sweep into thermal_cell_size.csv if that file is absent, then draws thermal_cell_size.pdf from it. Delete the
CSV to run again.
"""

import csv
import multiprocessing
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
from test_thermal import FILM_L, FILM_T, ND_A, ND_K, ND_MS, film_dt, film_mz, mm, spin_wave_deficit  # noqa: E402

CSV, PDF = HERE / "thermal_cell_size.csv", HERE / "thermal_cell_size.pdf"
CELLS_NM = (8, 4, 2, 1)
TEMPS = (100, 300, 600, 900)
CHECK_TEMPS = (300, 900)  # repeated at dt/2
MAIN = 2  # film_dt / 2: at film_dt itself, dt/2 still moved the deficit by many σ (D52 Result)
SEEDS = 8
H_K = 2 * ND_K / (mm.mu0 * ND_MS)

# The dataviz reference palette: T is a magnitude, so one sequential blue ramp, light to dark, and the ink roles.
RAMP = dict(zip(TEMPS, ["#9ec5f4", "#5598e7", "#256abf", "#104281"]))
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


FINDING = ("Finer meshes make the same film look hotter, and the result does not converge. At 300 K the\n"
           "thermal deficit grows from {lo:.5f} at 8 nm to {hi:.4f} at 1 nm. Each halving of the cell multiplies it by\n"
           "5.9, then 4.0, then 2.9, and the spin-wave sum heads for 2. The sum holds at low T: the measured\n"
           "deficit runs 2% above it at a 2% deficit, and 25% above it at a 20% deficit.")
CONDITIONS = ("Film 32 × 32 × 8 nm, periodic in x and y, free in z, no demag, H = 0, m starts along +z. Nd₂Fe₁₄B: µ₀Ms = 1.61 T, K = 4.30 MJ/m³,\n"
              "A = 7.7 pJ/m, easy axis z. Heun, α = 1, dt = 1/160 of the fastest spin-wave period. 8 seeds per point, each 0.25 ns · (a/nm)³ after a 20 ps\n"
              "burn-in; σ from the spread of the seed means. Halving dt again moves the deficit by at most 0.14% (2σ).\n"
              "Why the slope changes: well below √(A/K) = 1.3 nm, exchange ties neighbouring cells and the deficit goes as 1/a; far above it, each cell\n"
              "acts as its own macrospin and the deficit goes as 1/a³. Real Nd₂Fe₁₄B has T_c = 585 K, so 600 and 900 K probe the model, not the material.\n"
              "Decision D52. Regenerate with benchmarks/thermal_cell_size.py.")


def one_run(job):
    a_nm, T, div, seed = job
    a = a_nm * 1e-9
    duration = 0.25e-9 * a_nm**3  # every point gets the same cells × time, so similar statistics
    return job, film_mz(a, T, duration, seed + 1, div).mean()


def sweep():
    points = [(a, T, MAIN) for a in CELLS_NM for T in TEMPS] + [(a, T, 2 * MAIN) for a in CELLS_NM for T in CHECK_TEMPS]
    jobs = sorted(((*p, s) for p in points for s in range(SEEDS)), key=lambda j: -j[2] / j[0] ** 2)  # slowest first
    means = {}
    with multiprocessing.Pool(8) as pool:
        for i, (job, m) in enumerate(pool.imap_unordered(one_run, jobs), 1):
            means.setdefault(job[:3], []).append(m)
            print(f"{i}/{len(jobs)} {job}: {m:.5f}", flush=True)
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a_nm", "T_K", "dt_div", "dt_fs", "run_ns", "seeds", "m", "sigma", "harmonic_deficit"])
        for a, T, div in points:
            v = np.array(means[a, T, div])
            n, nz = round(FILM_L / (a * 1e-9)), round(FILM_T / (a * 1e-9))
            w.writerow([a, T, div, f"{film_dt(a * 1e-9) / div * 1e15:.2f}", 0.25 * a**3, SEEDS, f"{v.mean():.9f}",
                        f"{v.std(ddof=1) / np.sqrt(SEEDS):.9f}",
                        f"{spin_wave_deficit(n, n, nz, a * 1e-9, ND_MS, ND_A, H_K, T):.9f}"])


def load():
    with open(CSV, newline="") as f:
        return {(int(r["a_nm"]), int(r["T_K"]), int(r["dt_div"])): r for r in csv.DictReader(f)}


def draw(rows):
    m = lambda k: float(rows[k]["m"])
    sig = lambda k: float(rows[k]["sigma"])
    harm = lambda k: float(rows[k]["harmonic_deficit"])
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.955, "Thermal results against cell size", fontsize=16, weight="bold")
    fig.text(0.08, 0.875, FINDING.format(lo=1 - m((8, 300, MAIN)), hi=1 - m((1, 300, MAIN))), fontsize=9.5, color=INK2, linespacing=1.4)

    ax = fig.add_axes([0.1, 0.515, 0.74, 0.335])
    x = np.array(CELLS_NM, float)
    for T in TEMPS:
        keys = [(a, T, MAIN) for a in CELLS_NM]
        ax.plot(x, [harm(k) for k in keys], color=RAMP[T], linewidth=1.2, linestyle="--", zorder=2)
        ax.errorbar(x, [1 - m(k) for k in keys], yerr=[sig(k) for k in keys], fmt="o", color=RAMP[T], markersize=7, zorder=3,
                    markeredgecolor="white", markeredgewidth=1, label=f"{T} K")
        ax.annotate(f"{T} K", (x[-1], 1 - m(keys[-1])), xytext=(10, 0), textcoords="offset points", va="center", color=INK2)
    ax.plot([], [], color=MUTED, linewidth=1.2, linestyle="--", label="spin-wave sum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{a:g}" for a in x])
    ax.minorticks_off()
    ax.set_xlim(10, 0.72)  # smaller cells to the right
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("cell size a (nm), film of fixed size")
    ax.set_ylabel("thermal deficit 1 − ⟨m⟩")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    tab = fig.add_axes([0.08, 0.135, 0.84, 0.33])
    tab.axis("off")
    head = ["a (nm)", "cells", "T (K)", "⟨m⟩ ± σ", "1 − ⟨m⟩", "sum", "measured / sum", "⟨m⟩ at dt/2"]
    body = []
    for a in CELLS_NM:
        for T in TEMPS:
            k = (a, T, MAIN)
            half = (a, T, 2 * MAIN)
            body.append([a, round(FILM_L * 1e9 / a) ** 2 * round(FILM_T * 1e9 / a), T, f"{m(k):.7f} ± {sig(k):.7f}", f"{1 - m(k):.5f}",
                         f"{harm(k):.5f}", f"{(1 - m(k)) / harm(k):.3f}", f"{m(half):.7f} ± {sig(half):.7f}" if half in rows else "—"])
    t = tab.table(cellText=body, colLabels=head, cellLoc="center", loc="upper center", bbox=[0, 0, 1, 1],
                  colWidths=[0.06, 0.06, 0.06, 0.22, 0.09, 0.11, 0.12, 0.22])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    for (r, _), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.5)
        c.visible_edges = "horizontal"
        if r == 0:
            c.set_text_props(color=INK2, weight="bold")

    fig.text(0.08, 0.03, CONDITIONS, fontsize=7.5, color=INK2, linespacing=1.45)
    fig.savefig(PDF)
    plt.close(fig)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    if not CSV.exists():
        sweep()
    draw(load())
