"""The README's hero figure (D39), drawn from a fresh run of examples/grains.toml (about a minute on a laptop CPU).

    uv run --group docs python docs/make_figures.py

Writes docs/img/hero-light.png and docs/img/hero-dark.png. Left: a mid-plane slice of the microstructure, grains shaded
by their easy-axis tilt from z and the grain-boundary phase in gray. Right: the demagnetization curve with its H_c.
"""

import csv
import json
import tempfile
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import micromag as mm  # noqa: E402
from micromag.card import load  # noqa: E402
from micromag.regions import paint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "examples" / "grains.toml"
BOUNDARY = 100  # the card's boundary-phase id

# The dataviz reference palette: sequential blue (light: light -> dark; dark: dark -> light, so low tilt recedes into
# the surface), series slot 1 for the curve, muted neutral for the boundary phase, and the ink roles. GitHub surfaces.
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  line="#2a78d6", ramp=["#9ec5f4", "#5598e7", "#256abf", "#104281"]),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 line="#3987e5", ramp=["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"]),
}


def simulate(out):
    mm.run(CARD, out)
    with open(out / "stage1_hysteresis.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    h = load(CARD).stages[0].direction
    H = np.array([[float(r[k]) for k in ("mu0_Hx", "mu0_Hy", "mu0_Hz")] for r in rows]) @ h
    mh = np.array([[float(r[k]) for k in ("mx", "my", "mz")] for r in rows]) @ h
    hc = json.loads((out / "provenance.json").read_text())["stages"][0]["summary"]["Hc_T"]
    with h5py.File(out / "trajectory.h5") as f:
        regions = f["regions"][()]
    return H, mh, hc, regions


def draw(theme, H, mh, hc, regions, axis, cell, path):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": t["ink"], "axes.labelcolor": t["ink2"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.edgecolor": t["axis"]})
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.5]}, facecolor=t["surface"])
    for ax in (left, right):
        ax.set_facecolor(t["surface"])

    # Microstructure: the z mid-plane, x right and y up, in nm.
    z = regions.shape[0] // 2
    tilt = np.degrees(np.arccos(np.clip(np.abs(axis[2, z]), 0, 1)))
    shown = np.ma.masked_where(regions[z] == BOUNDARY, tilt)
    extent = [0, regions.shape[2] * cell[0] * 1e9, 0, regions.shape[1] * cell[1] * 1e9]
    left.imshow(np.zeros_like(tilt), cmap=LinearSegmentedColormap.from_list("gb", [t["muted"]] * 2), origin="lower", extent=extent)
    im = left.imshow(shown, cmap=LinearSegmentedColormap.from_list("tilt", t["ramp"]), vmin=0, vmax=15, origin="lower",
                     extent=extent, interpolation="nearest")
    left.set_title(f"Microstructure, z = {(z + 0.5) * cell[2] * 1e9:.0f} nm", color=t["ink"], fontsize=10, loc="left")
    left.set_xlabel("x (nm)")
    left.set_ylabel("y (nm)")
    bar = fig.colorbar(im, ax=left, fraction=0.05, pad=0.03)
    bar.set_label("easy-axis tilt from z (°)", color=t["ink2"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=t["muted"])
    left.legend(handles=[Patch(color=t["muted"], label="grain-boundary phase")], loc="upper left", bbox_to_anchor=(-0.02, -0.2),
                frameon=False, labelcolor=t["ink2"], fontsize=9, handlelength=1.2)

    # Demagnetization curve, field decreasing from 0.
    right.plot(H, mh, color=t["line"], linewidth=2, solid_capstyle="round")
    right.plot([hc], [0], "o", markersize=8, color=t["line"], markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
    right.annotate(f"$H_c$ = {abs(hc):.2f} T", (hc, 0), xytext=(10, 8), textcoords="offset points", color=t["ink"])
    right.axhline(0, color=t["axis"], linewidth=1, zorder=0)
    right.grid(True, color=t["grid"], linewidth=0.6)
    right.set_axisbelow(True)
    right.set_xlim(H.min(), H.max())
    right.set_ylim(-1.08, 1.08)
    right.set_title("Demagnetization curve, field 1° off z", color=t["ink"], fontsize=10, loc="left")
    right.set_xlabel("µ₀H along the field direction (T)")
    right.set_ylabel("m · ĥ")
    for side in ("top", "right"):
        right.spines[side].set_visible(False)
    for side in ("top", "right", "left", "bottom"):
        left.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    out_dir = ROOT / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        H, mh, hc, regions = simulate(Path(tmp) / "run")
    p = paint(load(CARD))  # the same seed gives the same grains and axes as the run
    assert np.array_equal(p.regions, regions)
    for theme in THEMES:
        draw(theme, H, mh, hc, regions, p.axis, load(CARD).cell, out_dir / f"hero-{theme}.png")
        print(f"wrote {out_dir / f'hero-{theme}.png'}")


if __name__ == "__main__":
    main()
