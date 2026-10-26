# Verification

Every numerical routine is tested against an analytic result or an independent oracle. The CPU implementation is the oracle for every GPU kernel. Published reference values, with their sources and conditions, are in [BENCHMARKS.md](../BENCHMARKS.md). The decision behind each check is in [DECISIONS.md](../DECISIONS.md).

## Running the tests

```console
$ uv run pytest                 # fast suite, about 1.5 min; CUDA cases skip on a CPU-only build
$ uv run pytest -m slow -s      # physics benchmarks, about 15 min on a laptop CPU; -s prints the comparisons
$ uv run pytest -m ""           # everything
```

Physics tests take a `device` fixture, so each one runs on `cpu` and, on a CUDA build, again on `cuda`. CI runs the fast suite on Linux and macOS for every push. The GPU runs are manual (see [gpu.md](gpu.md)).

| File | What it checks |
|---|---|
| `test_toolchain.py` | the build, and a CUDA kernel against the CPU bit for bit |
| `test_fields.py` | exchange, anisotropy and Zeeman: uniform-state energies, H = −∇E by finite differences, the analytic Bloch wall at second order, pair-scaled exchange, the demag switch |
| `test_demag.py` | the Newell tensor against Aharoni's closed form, the far field against a point dipole, FFT against a direct O(N²) sum |
| `test_llg.py` | the macrospin against its analytic solution, fifth-order convergence, energy conservation, µMAG SP4 |
| `test_minimizer.py` | the Stoner–Wohlfarth astroid, µMAG SP3 |
| `test_card.py`, `test_run.py` | card validation, the results folder, restarts, failure handling, refined hysteresis |
| `test_coercivity.py` | µMAG SP2 and the Nd₂Fe₁₄B cubes |
| `test_regions.py` | shapes, Voronoi grains, the boundary phase, the coupling table, the Stoner–Wohlfarth ensemble |
| `test_thermal.py` | Philox against its known-answer vectors; the thermal field's variance and independence; Heun and RK23 orders at T = 0; uncoupled macrospins against the Boltzmann distribution and the Langevin curve, at two damping values; a single macrospin under heavy step rejection and the Brownian bridge; the escape rate against the Fokker–Planck operator (slow); CPU = CUDA |

## Validation against published results

| Benchmark | Ours | Reference | Difference |
|---|---|---|---|
| µMAG SP3: flower/vortex crossover (20³) | L = 8.4551 l_ex | 8.47 (consensus); MagTense 8.4556 | −0.18% |
| µMAG SP4 field 1: first ⟨mx⟩ = 0 | 138.6 ps | OOMMF 1.2: 138.4 ps | +0.14%; RMS ⟨m⟩ 0.0051 |
| µMAG SP2: H_c/Ms at d/l_ex = 1 | 0.05704 | exact limit 0.05707 | −0.05% |
| µMAG SP2: H_c/Ms at d/l_ex = 10 | 0.05446 | OOMMF 1.1: 0.05454 | −0.15% |
| µMAG SP2: H_c/Ms at d/l_ex = 20 | 0.04960 | OOMMF 1.1: 0.04991 | −0.6% |
| µMAG SP2: H_c/Ms at d/l_ex = 30 | 0.04533 | OOMMF 1.1: 0.04594; MagTense 0.04558 | −1.3% (known miss, open) |
| Stoner–Wohlfarth astroid, ψ = 15–75° | — | (cos^{2/3}ψ + sin^{2/3}ψ)^{−3/2} | ≤ 5·10⁻⁴ H_K |
| Pure 40 nm Nd₂Fe₁₄B cube: switching field | 5.275 T | Fischbacher et al. 2017: 5.29 T | −0.3% |
| 25 nm core–shell cube: H_c | 2.721 T | Fischbacher et al. 2018, Fig. 8a: 2.76 T | −0.04 T |
| 50 nm core–shell cube: H_c | 2.675 T | Fig. 8a: 2.71 T | −0.035 T |
| 255 decoupled random grains: M_r, H_c | 0.4705 M_s, 0.4627 H_K | the same axes, exactly: 0.4705, 0.4627; isotropic SW: 0.5 ± 0.02, 0.479 ± 0.014 | 3·10⁻¹⁴, 7·10⁻⁸ |

Every value the CPU computed is the same on the GPU (Tesla T4). The SP2 d/l_ex = 30 test is marked as an expected failure. A likely cause is the mesh: we use 0.5 l_ex cells where OOMMF used 0.60 l_ex. A cell-size study is open; see D32.

## Mutation checks

After each iteration, 3–4 realistic bugs are planted one at a time, and the tests that catch each one are recorded, for example a transposed index, a missing factor of 2 or a wrong default. The records are the **Mutation check** lines under each iteration's checks decision in DECISIONS.md. A bug no test catches is logged as a known gap.
