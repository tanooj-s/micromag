# Input cards

A simulation is one [TOML](https://toml.io) file: a few setup sections, then an ordered list of stages. Run it with

```console
$ uv run micromag run card.toml -o runs/name [--device cuda]
```

Units are SI throughout: A/m, J/m, J/m³, m and s. Applied fields are given as **µ₀H in tesla** (`mu0_H`), because the references quote T or mT. Damping is the dimensionless Gilbert α.

Validation is strict and happens before anything runs. An unknown key is an error that names the valid keys, and every problem in the card is reported together. Source: [`python/micromag/card.py`](../python/micromag/card.py). The design is in [DECISIONS.md](../DECISIONS.md): D27 for the card, D33–D36 for regions.

```toml
device = "cpu"              # optional: "cpu" (default) or "cuda"; `--device` on the command line overrides it

[grid]
n    = [128, 32, 1]         # cells along x, y, z
cell = [3.9e-9, 3.9e-9, 3e-9]
demag = true                # optional; false switches the demag field off (D36)

[material.permalloy]        # one or more named materials; the FIRST fills the grid
Ms = 8e5
A  = 1.3e-11

[initial]
m = [1, 1, 1]

[[stage]]
type = "minimize"
```

## `[grid]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `n` | 3 whole numbers ≥ 1 | required | cells along x, y, z |
| `cell` | 3 positive numbers | required | cell size (m) |
| `demag` | bool | `true` | include the demagnetizing field; `false` makes it zero in the field and the energies |

Coordinates elsewhere in the card are measured in metres from the grid's lower corner. Cell (i, j, k) has its centre at ((i + ½) dx, (j + ½) dy, (k + ½) dz).

## `[material.<name>]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `Ms` | number ≥ 0 | required | saturation magnetisation (A/m); 0 is vacuum |
| `A` | number ≥ 0 | required | exchange stiffness (J/m) |
| `K` | number | `0` | uniaxial anisotropy (J/m³); the energy is −K (m·k̂)² |
| `axis` | 3 numbers, not all 0 | `[0, 0, 1]` | easy axis k̂, normalised on load |

The first material listed fills the grid as **region 0**, the background. `vacuum` is built in (Ms = 0) and its name is reserved. Between two cells, the exchange uses the harmonic mean of their A values.

## `[[region]]`: painted shapes

Regions are painted in card order, and later paints win. A cell belongs to a shape if its **centre** is inside it.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | whole number 1–255 | required | the region id in the map (0 is the background) |
| `material` | string | required | a `[material.<name>]` or `"vacuum"` |
| `shape` | string | required | `all`, `cuboid`, `cylinder` or `ellipsoid` |
| `inverse` | bool | `false` | paint everything *outside* the shape |
| `axis` | 3 numbers | the material's | override the easy axis for this region |
| *shape keys* | see below | required | |

| Shape | Keys |
|---|---|
| `all` | none |
| `cuboid` | `corners = [[x0, y0, z0], [x1, y1, z1]]`: lower bounds inside, upper bounds outside |
| `cylinder` | `center`, `radius`, `height`; the axis is along z |
| `ellipsoid` | `center`, `semi_axes = [a, b, c]` |

An id may be painted more than once, but always with the same material and axis. The results folder records each region's exact volume next to its painted volume (a sum of whole cells), and a mismatch over 5% prints a warning. That catches shapes that are clipped by the grid or too coarsely resolved.

```toml
[material.shell]                 # first: the background
Ms = 3.8e5
A  = 6.12e-12

[material.core]
Ms = 1.28e6
A  = 8e-12
K  = 4.9e6

[[region]]
id       = 1
material = "core"
shape    = "cuboid"
corners  = [[3e-9, 3e-9, 3e-9], [22e-9, 22e-9, 22e-9]]

[[region]]                       # trim the particle to a sphere: everything outside is vacuum
id       = 2
material = "vacuum"
shape    = "ellipsoid"
center   = [12.5e-9, 12.5e-9, 12.5e-9]
semi_axes = [12.5e-9, 12.5e-9, 12.5e-9]
inverse  = true
```

## `[grains]`: Voronoi polycrystals

One optional table. It is applied after all `[[region]]` painting and fills one region with 3D Voronoi grains: each cell joins the grain of its nearest seed.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `count` | whole number ≥ 1 | required | number of grains (seeds) |
| `first_id` | whole number 1–255 | required | grains get ids `first_id` … `first_id + count − 1`, which must stay ≤ 255 and not clash with `[[region]]` ids |
| `material` | string | required | the grains' material |
| `seed` | whole number ≥ 0 | required | random seed: the same seed gives the same grains and axes |
| `within` | region id | `0` | the region to fill |
| `axes` | see below | `"material"` | easy axis per grain |
| `boundary` | table | none | a grain-boundary phase (below) |

Seeds are drawn uniformly over the grid box and kept if they land in `within`, so the grains don't depend on the cell size.

`axes` is one of:
- `"material"`: every grain takes the material's axis
- `"random"`: uniform on the sphere (isotropic)
- `{ mean = [x, y, z], cone_deg = θ }`: uniform over the cap within θ of the mean (0 < θ ≤ 90)

`boundary = { thickness = t, material = "...", id = n }` coats every grain face with a layer of its own material. A cell whose distance to its grain's boundary is below t/2 joins region n, so two neighbouring grains are separated by t in total. The distance is exact: it comes from the bisecting planes between seeds, not from a grid estimate. t must be at least 2 cells (the largest cell edge). For thinner boundaries, scale the exchange with `[[coupling]]` instead.

```toml
[grains]
count    = 27
first_id = 1
material = "grain"
seed     = 1
axes     = { mean = [0, 0, 1], cone_deg = 15 }
boundary = { thickness = 4e-9, material = "boundary", id = 100 }
```

## `[[coupling]]`: exchange between regions

Each entry rescales the exchange across the faces between two regions, as MuMax3's `ext_ScaleExchange` and `ext_InterExchange` do. Coupling inside a region is unchanged. Later entries win.

| Key | Type | Meaning |
|---|---|---|
| `regions` | `[a, b]` | two region ids (0–255) or `"grains"` (every grain id); a ≠ b |
| `scale` | number ≥ 0 | multiplies the harmonic-mean A across those faces |
| `A` | number ≥ 0 | sets the face exchange (J/m) directly |

Give exactly one of `scale` or `A`. `regions = ["grains", "grains"]` means every pair of distinct grains.

```toml
[[coupling]]             # decouple the grains from one another
regions = ["grains", "grains"]
scale   = 0.0

[[coupling]]             # a weak grain ↔ boundary-phase interface
regions = ["grains", 100]
A       = 1e-12
```

## `[initial]`

Give exactly one of:

| Key | Meaning |
|---|---|
| `m = [x, y, z]` | a uniform start, normalised on load |
| `file = "path/trajectory.h5"` | a state from an earlier run, relative to the card. Optional `state`: `"final"` (default) or `"stage<k>"`, the last snapshot of stage k |

A restart must have the same cell counts `n`; the cell size may differ. The file's SHA-256 is recorded in the provenance.

## `[[stage]]`

Stages run top to bottom. The magnetisation carries over from one stage to the next; the applied field does **not**: it defaults to zero in every stage.

### `type = "minimize"`

The nearest equilibrium at a fixed field (steepest descent with Barzilai–Borwein steps).

| Key | Default | Meaning |
|---|---|---|
| `mu0_H` | `[0, 0, 0]` | applied field (T) |
| `torque_tol` | `1e-6` | stop when max \|m × H\| / Ms is below this |
| `max_iter` | `100000` | give up after this many iterations (the stage fails) |

### `type = "hysteresis"`

A quasi-static field sweep, minimizing at each field.

| Key | Default | Meaning |
|---|---|---|
| `direction` | required | field direction, normalised |
| `mu0_H_from`, `mu0_H_to` | required | field range along `direction` (T), inclusive |
| `mu0_H_step` | required | step (T); it must divide the range into whole steps |
| `torque_tol`, `max_iter` | `1e-6`, `100000` | as for `minimize` |
| `dump` | `false` | save a snapshot at every field, not just the last |
| `refine_to` | none | halve any step that crosses m·ĥ = 0 or jumps, down to this step (T) |
| `refine_jump` | `0.1` | a change of \|Δ⟨m⟩\| above this counts as a jump |

The stage reports `Hc_T` (the first zero of m·ĥ, linearly interpolated), `switch_T` (the midpoint of the largest jump) and `remanence` (⟨m⟩ at zero field, if the sweep passes through it). They appear on the progress line and in `provenance.json`. A full loop is two stages.

### `type = "llg-dynamics"`

Time integration of the Landau–Lifshitz–Gilbert equation with an adaptive Dormand–Prince RK45.

| Key | Default | Meaning |
|---|---|---|
| `alpha` | required | Gilbert damping (dimensionless) |
| `gamma` | `1.76086e11` | gyromagnetic ratio (rad/(T·s)) |
| `mu0_H` | `[0, 0, 0]` | applied field (T) |
| `tol` | `1e-6` | per-step error tolerance |
| `max_step` | none | largest step (s) |
| `dump_every` | none | snapshot interval (s) |

Give exactly one stopping rule:
- `duration` (s), with an optional `sample_every` (s): rows at the start, at every interval and at the end.
- `until_torque`, `check_every` and `max_duration`, all required: stop when max \|m × H\| / Ms falls below `until_torque`, checked every `check_every`; the stage fails at `max_duration`. `tol` must be ≤ `until_torque / 100`, or the threshold is never reached.

## Examples

| Card | What it runs |
|---|---|
| [`examples/sp4.toml`](../examples/sp4.toml) | µMAG standard problem 4: relax the S-state, then switch in a field |
| [`examples/sp2.toml`](../examples/sp2.toml) | µMAG standard problem 2 at d/l_ex = 30 (GPU-sized) |
| [`examples/cube_core_shell.toml`](../examples/cube_core_shell.toml) | a 100 nm Nd₂Fe₁₄B cube with a 3 nm soft shell |
| [`examples/grains.toml`](../examples/grains.toml) | an 8-grain Nd₂Fe₁₄B polycrystal with a boundary phase (about 1 min on a CPU) |
