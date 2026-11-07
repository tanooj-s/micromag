# NOTES

Roadmap, status and open work: read at the start of each session. Decisions D1… are in `DECISIONS.md`, and published
comparisons in `benchmarks/BENCHMARKS.md` (D38, D53).

## Roadmap
Each iteration is done when its check passes and its decisions are logged in `DECISIONS.md`. The full check is in the decision named in each row.

| # | Iteration | Check | Status |
|---|-----------|-------|--------|
| 0 | Toolchain: CMake, bindings, pytest, one CUDA kernel | Kernel test passes on a laptop and on a cluster GPU node | laptop ✓ · cluster optional (D25) |
| 1 | Grid, per-cell materials, exchange, anisotropy, Zeeman (eqs. 8, 9, 11) | Uniform energies, H = −∇E, the analytic Bloch wall (D13) | ✓ CPU |
| 2 | Demag: Newell tensor, O(N²) oracle, padded FFT (eqs. 17–21) | Aharoni's prisms to 1e-10, the far field, FFT = direct to 1e-12 (D17) | ✓ CPU |
| 3 | LLG (eq. 25), adaptive Dormand–Prince RK45 | Analytic macrospin, 5th order, energy behaviour, µMAG SP4 (D20) | ✓ CPU |
| 4 | Minimizer (eqs. 22–23) and hysteresis driver | Stoner–Wohlfarth astroid, µMAG SP3 (D23) | ✓ CPU |
| 5 | GPU backend: CUDA, cuFFT, the backend interface | Every test with `device="cuda"`, CPU = GPU to D26's tolerances | ✓ CPU + GPU (Tesla T4, 2026-10-18) |
| 6a | Workflow: TOML card → results folder through `micromag run` and `micromag.run()` (D27, D28) | The SP4 card meets D20; validation, dumps, restarts, failures (D29) | ✓ |
| 6b | Science runs: hysteresis refinement and summary on a GPU (D30, D31) | SP2, the 40 nm cube, the Fig. 8a core–shell cube (D32) | ✓ CPU + GPU (T4, 2026-10-20), except SP2 at d/l_ex = 30: −1.3% against 1%, a known miss (xfail) |
| 6c | Material regions on the MuMax3 model (D33–D36) | Shapes, grains and coupling against oracles; the SW ensemble; the 25 nm cube unchanged (D37) | ✓ CPU + GPU (T4, 2026-10-20) |
| 6d | An original study using the 6b/6c tools | Set in its design | shelved, optional (D40) |
| 7 | Brown's thermal field; Heun and RK23 with a Brownian bridge; Philox noise (D41–D43) | Philox vectors, orders, Boltzmann and Langevin ensembles, the bridge, the escape time, CPU = CUDA (D44) | ✓ CPU (2026-10-26); GPU run pending |
| 8 | Periodic boundaries on any axes (D45, D46) | Spiral exchange, periodic FFT = direct, demag factors, grains, the perpendicular film, CPU = CUDA (D47) | ✓ CPU (2026-11-03); GPU run pending, CUDA not yet compiled |
| 9 | Energy barriers: geodesic NEB, L-BFGS, a `gneb` stage (D48–D50) | Macrospin, wire closed form, path properties, the cube's 25 k_BT crossing within 3% of 3.94 T, CPU = CUDA (D51) | ✓ CPU (2026-11-06); GPU run pending |
| 10 | Stretch, optional: multi-GPU | — | — |

## Resume here
State on 2026-11-06: iterations 0–9 are done on the CPU, and everything through 6c is verified on a Tesla T4.
- **Test counts:** `uv run pytest` gives 205 passed, 89 skipped, 26 deselected in 94 s; `-m slow` gives 11 passed, 15 skipped in 27 min (laptop, 2026-11-06). Last GPU run: 139 passed, 1 skipped on a Colab T4 (2026-10-20).
- **Thermal cell-size study done (D52):** `benchmarks/thermal_cell_size.pdf`. Demag on and a second material are not measured.
- **First thing next session (deferred by the user, 2026-11-03):** one GPU session covers iterations 7, 8 and 9. First the CUDA compile check with Docker (`docs/gpu.md`), since the iteration-8 CUDA changes (exchange wrap, unpadded periodic axes) have never been compiled. Then `uv run pytest -m "" -v -s` on a Colab T4 (setup in `docs/gpu.md`; a later `uv run` without `SKBUILD_CMAKE_DEFINE` silently rebuilds without CUDA), and the D44, D41, D47 and D51 GPU Results.
- **Open work, in rough order:**
  1. **SP2 d/l_ex = 30 cell-size study** (the user's call: a fundamental scaling study to keep, 2026-10-20). Run H_c/Ms at d/l_ex = 30 with nz = 3, 4, 5, 7 cells through the thickness on a GPU; nz = 5 is OOMMF 1.1's 0.60 l_ex, and nz = 6 (0.5 l_ex) gave 0.04533. If nz = 5 lands within 1% of 0.04594, the test should use OOMMF's cell; otherwise the references disagree at this size. A 6d candidate. The script is below.
  2. **GNEB convergence study** (deferred by the user for token and time cost, 2026-11-06; D49): why the 15³ cube fails, whether to adapt the L-BFGS history, and whether L-BFGS can be beaten here. Nothing has been run.
     - **Settled:**
       - Harness: a separate lab copy under `studies/gneb/`, checked to reproduce `gneb()` iteration for iteration. `gneb.py` stays untouched until a winner is chosen. Scripts committed, raw results in the git-ignored `runs/`.
       - Adaptive history is adopted only with at least 25% fewer median field evaluations than the best fixed history, paired by seed, with no lower success rate and the same barrier, on held-out cases (the wire and the core–shell cube).
       - The one alternative to test: an exact-Hessian saddle stage. Converge the chain loosely, then finish the climbing image alone by Newton–Krylov. `effective_field` is linear in m apart from H_ext, so a Hessian-vector product is exact and costs one field evaluation. It also checks for exactly one negative mode.
     - **Still open:** the compute budget (seeds, meshes, history values). Estimates: about 0.03 s per iteration at 15³ and 0.26 s at 30³ with 10 images, about 8 runs in parallel.
     - **Test these first:**
       - The record is weak: 3 seeds at 15³ with no tolerance logged, while 8³ and 12³ used 1e-3 and 1500 iterations. History 15 came from four noisy cases.
       - Stiffness is an unlikely cause: 30³ is about 4× stiffer and converges, so D49's preconditioner may not be the fix.
       - Likelier causes: an under-resolved wall (2.67 nm cells, twice δ0) giving a rugged path and a hopping climbing image; truncation repeatedly wiping the history; the stop test needing every image below tolerance when only the saddle sets the barrier; odd n (15 is the only odd mesh tried).
       - A Krylov solve on a converged path gives a lower bound on what any such method could achieve.
  3. **Whether `gneb` host work dominates on a GPU** (D48): measure it in the GPU session.
  4. **`micromag summarize runs/*`:** one table from each run's `provenance.json`. Suggested, not yet agreed.
  5. **Known test gaps:** a wrong embedded-error weight in the LLG integrator wastes steps but gives correct results (a step-count bound on the macrospin would catch it; offered in iteration 3, not added); the bridge's conditional mean (D44); L-BFGS history not projected onto the tangent planes (D51).
  6. **Optional:** the 6d design (D40); multi-GPU (iteration 10); the recorded runs of D32 (the full SP2 curve and the 100 nm core–shell cube, benchmark data only); the cluster build (below).
- **SP2 cell-size script** (run on Colab after the `docs/gpu.md` setup, `uv run python sp2_conv.py 3 4 5 7`):
  ```python
  import pathlib, sys, tempfile
  import numpy as np
  import micromag as mm
  sys.path.insert(0, "tests")
  from test_coercivity import SP2_A, SP2_BM, SP2_MS, card_text, run_card

  d = 30
  lex = np.sqrt(SP2_A / (0.5 * mm.mu0 * SP2_MS**2))
  for nz in map(int, sys.argv[1:]):          # cells through the 3 l_ex thickness
      cell = 0.1 * d * lex / nz
      text = card_text((50 * nz, 10 * nz, nz), [cell] * 3, f"Ms = {SP2_MS!r}\nA = {SP2_A!r}", [1, 1, 1],
                       0.1 * SP2_BM, -0.08 * SP2_BM, 0.005 * SP2_BM, 1e-5)
      s = run_card(pathlib.Path(tempfile.mkdtemp()), text, "cuda")
      print(f"nz = {nz}: cell = {cell / lex:.3f} l_ex, Hc/Ms = {abs(s['Hc_T']) / SP2_BM:.5f}, "
            f"remanence {np.round(s['remanence'], 4).tolist()}", flush=True)
  ```

## Cluster environment
- **GPU nodes:** 8× Tesla V100-SXM2-32GB per node (sm_70). The driver is 535.104 and supports CUDA ≤ 12.2.
- **Modules:**
  - `cmake/3.17.3`, which is too old. scikit-build-core fetches cmake ≥ 3.18 from PyPI.
  - `gcc/8.4.0/1` (default) and `gcc/9.3.0/1`.
  - `cuda/10.1` through `cuda/12.1`.
  - `pgi/19.10`, `matlab`, `git/2.25.1`.
- **No Python module.** A shared conda env exists, but it belongs to another project, so we don't use it.
- **Slurm partition and account:** TBD. These are only needed if `sbatch` rejects the job without them. Find them with `sinfo -s` and `sacctmgr show assoc user=$USER format=account,partition`.

### Optional open issue: cluster build (deferred 2026-10-17; optional since 2026-10-18, D25)
- **Symptom:** `uv sync` builds numpy 2.5.4 from source, and the build fails because the system `cc` is gcc 4.8.5 (RHEL 7).
- **Cause (verified 2026-10-18 against PyPI and the numpy 2.3.0 release notes):** RHEL 7 has glibc 2.17. numpy 2.3 moved from manylinux2014 to manylinux_2_28 wheels, which drops RHEL/CentOS 7. **numpy 2.2.6 is the last release with glibc 2.17 wheels for cp313.** With no matching wheel, uv falls back to a source build.
- **Also h5py (added in 6a, D28):** its current Linux wheels are manylinux_2_28 too, so it needs the same treatment: a cap at a version with manylinux2014 wheels, or a source build.
- **Fix:** cap numpy below 2.3 (`numpy<2.3`). `uv.lock` is universal, so the laptop also moves to 2.2.6, which keeps both machines identical (the D8 reasoning).
- **Won't work:** `export CC=gcc` after the gcc 8.4 load. numpy ≥ 2.3 needs gcc ≥ 9.3 to build, and `cuda/12.1` is only offered under gcc 8.4 (D4).
- **After that:** the CMake log should report `The CXX compiler identification is GNU 8.4.0`. If it shows 4.8.5, `export CXX=g++`. The cmake and ninja wheels from PyPI still support glibc 2.17.
- Already fixed: home-directory quota (uv cache moved to scratch) and the empty `CUDA_ARCHITECTURES`.

## MagTense weaknesses we want to avoid
- One 3,500-line Fortran module holds global `gb_problem` and `gb_solution` state, so solves are not re-entrant.
- The default demag is a dense O(N²) matrix; the FFT path was added on later.
- Damping is the dimensional LL `alpha0` rather than Gilbert's dimensionless α.
- Tests are example scripts with loose tolerances (SP4 allows 100% error), and there are no unit tests per field term.
- The Python and MATLAB front ends duplicate each other, and the build needs Intel ifx/MKL plus an `#if` flag matrix.
