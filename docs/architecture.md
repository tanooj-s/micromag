# Architecture

micromag is a C++ core with a thin Python layer on top. Python reads the card, builds the per-cell arrays, drives the stages and writes the results. C++ does all the per-cell arithmetic, on the CPU or on a CUDA GPU. Every design choice below has an entry in [DECISIONS.md](../DECISIONS.md) (Dn).

## From card to results

```
card.toml
   │  card.load()                 validate everything up front; plain dataclasses        python/micromag/card.py
   ▼
Card
   │  regions.paint()             region map (uint8) → per-cell Ms, A, K, k̂             python/micromag/regions.py
   ▼                              + the 256×256 exchange pair table
Problem(grid, Ms, A, K, k, H_ext, device, regions, coupling, demag)                      src/problem.{hpp,cpp}
   │  owns a Backend: the CPU one (the oracle) or CUDA                                    src/backend.hpp, src/cuda_backend.cu
   │  field terms: exchange, anisotropy, Zeeman (eqs. 8, 9, 11), demag (eq. 21)           src/fields.cpp, src/demag.cpp
   ▼
drivers: LLG (eq. 25, Dormand–Prince RK45)  ·  Minimizer (eqs. 22–23, Barzilai–Borwein)   src/llg.cpp, src/minimizer.cpp
   │  run.py loops over the stages: llg-dynamics, minimize, hysteresis (with refinement)
   ▼
results folder: card.toml, provenance.json, stage<k>_<type>.csv, trajectory.h5            python/micromag/run.py
```

## Principles

- **Data layout (D9).** Structure-of-arrays float64 in numpy-owned C-order arrays: vectors are `(3, nz, ny, nx)` and scalars `(nz, ny, nx)`, with x fastest. The bindings use `noconvert`, so a wrong dtype or layout raises an error instead of silently writing into a copy.
- **Materials per cell (D9, D33).** The solver only sees per-cell Ms, A, K, k̂, plus an optional region map and pair table. Shapes, grains and materials are expanded in Python, so new geometry never touches C++.
- **Fields first (D10).** Each term writes its field. Energies are derived from the field as E = −c µ₀ dV Σ Ms m·H, so one code path per term is checked two independent ways.
- **One driver, two backends (D24).** `LLG` and `Minimizer` hold the control logic, and a `Backend` holds the materials and does the arithmetic: fields, linear combinations, reductions. The CPU backend is the original loop code and serves as the **oracle**. The CUDA backend runs the same arithmetic, one thread per entry, using cuFFT and CUB, and every kernel is tested against the CPU (D26).
- **No global state.** Every solver object is constructed and passed in, so runs are re-entrant. This is one of the MagTense weaknesses avoided on purpose.
- **Vacuum is Ms = 0 (D11).** Vacuum cells get H = 0 and are not exchange neighbours. Exchange between unlike cells uses the harmonic mean of A, optionally scaled per region pair (D35).
- **Demag (D14–D16).** The Newell tensor for near cells and a Gauss-averaged dipole for far ones, mirrored from one octant, then a zero-padded real-to-complex FFT convolution: pocketfft on the CPU, cuFFT on the GPU.

## File map

| Path | Role |
|---|---|
| `src/fields.{hpp,cpp}` | `Grid`; the exchange, anisotropy and Zeeman fields (the CPU oracle) |
| `src/demag.{hpp,cpp}` | the demag tensor and the FFT convolution `Demag` |
| `src/backend.hpp` | the `Backend` interface and the `Buffer` owning array |
| `src/problem.{hpp,cpp}` | the CPU backend; `Problem` sums H_eff and computes the energies |
| `src/cuda_backend.cu` | the CUDA backend (built with `MICROMAG_CUDA=ON`) |
| `src/llg.{hpp,cpp}`, `src/minimizer.{hpp,cpp}` | the drivers |
| `src/bindings.cpp` | the nanobind module `micromag._core` |
| `python/micromag/__init__.py` | the public Python API, plus `energy()` and the in-memory `hysteresis()` |
| `python/micromag/card.py` | the card schema and its validation |
| `python/micromag/regions.py` | painting, Voronoi grains, the boundary phase, the pair table |
| `python/micromag/run.py` | `micromag run`: stages, the results folder, provenance |
| `tests/` | pytest, the single test runner (see [verification.md](verification.md)) |
| `examples/` | input cards |
| `hpc/` | Slurm job scripts |
| `docs/` | these pages, and `make_figures.py` for the README figure |

## Python API

The card is the main interface, and everything it does is also available from Python:

```python
import numpy as np
import micromag as mm

g = mm.Grid(64, 64, 1, 2e-9, 2e-9, 2e-9)
s = (g.nz, g.ny, g.nx)
Ms, A, K = np.full(s, 8e5), np.full(s, 1.3e-11), np.zeros(s)
k = np.zeros((3, *s)); k[2] = 1
prob = mm.Problem(g, Ms, A, K, k, (0, 0, 0), device="cpu")   # or "cuda"

m = np.zeros((3, *s)); m[0] = 1
mm.Minimizer(prob).minimize(m, torque_tol=1e-6)               # m is updated in place
print(prob.energies(m))                                       # {"exchange": …, "demag": …} in J

llg = mm.LLG(prob, alpha=0.02)
llg.run_until(m, 1e-9)                                        # advance to t = 1 ns
```

`mm.run(card, out, device=None)` is the same as `micromag run`.

## Where things are written down

| File | Holds |
|---|---|
| [NOTES.md](../NOTES.md) | the roadmap, status, open work |
| [DECISIONS.md](../DECISIONS.md) | the decision log D1…: context, decision, alternatives, consequence, plus results and corrections |
| [BENCHMARKS.md](../BENCHMARKS.md) | published reference values and our comparisons |
| [CLAUDE.md](../CLAUDE.md) | the working rules and commands for development |
