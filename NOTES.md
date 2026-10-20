# NOTES

Roadmap, status and open work: read at the start of each session. Decisions D1… are in `DECISIONS.md`, and published
comparisons in `BENCHMARKS.md` (D38).

## Roadmap
Each iteration is done when its check passes and its decisions are logged in `DECISIONS.md`.

| # | Iteration | Check | Status |
|---|-----------|-------|--------|
| 0 | Toolchain: CMake + bindings + pytest + one CUDA kernel, run on a laptop and on a cluster GPU node through Slurm | Kernel test passes in both places | laptop ✓ · cluster optional (see below, D25) |
| 1 | Grid, per-cell materials, exchange / anisotropy / Zeeman (eqs. 8, 9, 11), CPU | Energies of uniform states; H matches the finite-difference gradient of E; analytic 1D Bloch wall (width π√(A/K)) is in equilibrium (torque → 0) with σ → 4√(AK), both errors falling ~4× per halving of dx (D13) | ✓ CPU (`tests/test_fields.py`) |
| 2 | Demag: Newell tensor, direct O(N²) oracle, zero-padded FFT (eqs. 17–21) | Uniform prisms (cube, 1:2:3 cell, 32×32×1 film) match Aharoni's closed form to 1e-10; far-field tensor → point dipole with error ~4× smaller per doubling of distance; FFT and direct sum agree to 1e-12 (D17) | ✓ CPU (`tests/test_demag.py`) |
| 3 | LLG (eq. 25) with a self-written adaptive Dormand–Prince RK45 | Macrospin matches the analytic solution; fixed-step error falls ~32× per halving of h (5th order); energy conserved at α = 0 and non-increasing at α > 0; µMAG SP4 field 1: first ⟨mx⟩ = 0 crossing within 3% of OOMMF and RMS ⟨m⟩ difference ≤ 0.05 over 0–1 ns (D20) | ✓ CPU (`tests/test_llg.py`; SP4 with `-m slow`) |
| 4 | Minimizer: projected steepest descent with Barzilai–Borwein steps (eqs. 22–23), plus a hysteresis driver | Stoner–Wohlfarth astroid: switching fields at ψ = 15–75° within 2·10⁻³ H_K of `(cos^{2/3}ψ + sin^{2/3}ψ)^{−3/2}`; µMAG SP3 (20³): flower/vortex crossover L within 0.5% of 8.47 l_ex (D23) | ✓ CPU (`tests/test_minimizer.py`; SP3 with `-m slow`) |
| 5 | GPU backend: CUDA kernels + cuFFT; the backend interface is introduced here | Every test passes with `device="cuda"` and the direct CPU-vs-GPU comparisons hold their D26 tolerances, run on at least one CUDA GPU of sm_70 or newer (which GPU is recorded). Nsight roofline numbers are optional (D25) | ✓ CPU + GPU (Tesla T4, CUDA 13.0, Colab, 2026-10-18; D26 Result) |
| 6a | Workflow: one simulation from a TOML input card into a results folder (per-stage CSV thermo tables, an HDF5 dump of magnetic states, provenance) through `micromag run` and `micromag.run()` (D27, D28). **Requirement (user, 2026-10-17):** a simple local workflow, parameters in → results out, from an input file, without hand-editing a Python script, with the same entry point on any machine | `examples/sp4.toml` run through `run()` meets D20's SP4 criteria; fast tests of validation, the results folder, restarts, dumps and failure handling (D29) | ✓ (`tests/test_card.py`, `tests/test_run.py`; SP4 card with `-m slow`) |
| 6b | Science runs: interim box material regions, field refinement near switching, a hysteresis summary (H_c, switching field, remanence), on a GPU; scaling study and Nsight optional (D25, D30, D31) | µMAG SP2 H_c and remanence vs d/l_ex against the exact limit and OOMMF 1.1; the pure 40 nm Nd₂Fe₁₄B cube against Fischbacher et al. 2017 (5.29 T); the Fig. 8a core–shell cube against the paper's finest-mesh curve (D32) | ✓ CPU + GPU (Colab T4, 2026-10-20), except SP2 at d/l_ex = 30: −1.3% against a 1% tolerance, accepted as a known miss (xfail) pending a cell-size study (D32) |
| 6c | Material regions on the MuMax3 model: a per-cell region map painted from named materials and shapes, 3D Voronoi grains with per-grain easy axes and an optional grain-boundary phase, exchange scaled per region pair, a demag switch (D33–D37) | Shapes' cell volumes → exact volumes under refinement; grain labels and the boundary layer against brute-force oracles (the two-seed slab exactly); exchange with a pair table bitwise unchanged at scale 1, against a hand-computed chain, still −∇E, CPU = CUDA; decoupled random-axis grains without demag: remanence exactly Σ n_g\|k̂_g·ĥ\| / Σ n_g and H_c within 5·10⁻³ H_K of a Stoner–Wohlfarth oracle for the same axes, both within 3σ of the isotropic 0.5 and 0.479 H_K; the migrated 25 nm core–shell cube still gives 2.721 T (D37) | ✓ CPU + GPU (`tests/test_regions.py`, `tests/test_fields.py`; ensemble with `-m slow`; Colab T4, 2026-10-20) |
| 6d | An original study using the 6b/6c tools: a question the paper only touches (e.g. shell thickness or anisotropy, field angle, a corner defect, grain-boundary coupling) | Set in the 6d design, after 6b is validated | todo |
| 7 | Stretch, pick one: string method (NEB) / multi-GPU / thermal field (Voronoi grains moved to 6c) | — | — |

## Resume here
State at the end of the 2026-10-20 session: everything is committed and pushed.
- **Done:** iterations 1–5 (5 verified on a Tesla T4), 6a (the card → `micromag run` → results folder workflow), 6b and 6c on CPU and GPU.
- **Docs (D39, 2026-10-20):** README.md, docs/ (cards, results, verification, architecture, gpu), GPL-3.0 LICENSE, CPU CI on GitHub Actions. 6d is shelved for later (user, 2026-10-20).
- **Test counts (2026-10-20):** `uv run pytest` gives 105 passed, 35 skipped (CUDA) and 20 deselected (slow) on the laptop; 139 passed and 1 skipped on a Colab T4. The GPU slow coercivity tests give 6 passed and SP2 d/l_ex = 30 xfail (20 min). `-m slow` takes about 15 minutes on the laptop CPU, mostly the cube checks; the user chose to keep them on both devices (D32).
- **User plan:** start running simulations and logging results by **2026-10-19**. The run-and-record workflow is the priority.
- **First thing next session:** ask which simulations the user wants to run first, and check the card covers them:
  - regions: named materials, shapes, Voronoi grains with a boundary phase, pair coupling (D33–D35); see `examples/grains.toml`
  - stages are `llg-dynamics`, `minimize` and `hysteresis`
  - there is no cross-run results table yet: each run is its own folder under the git-ignored `runs/`. A small `micromag summarize runs/*`, reading each `provenance.json` into one table, was suggested and not yet agreed.
- **Open work, in rough order:**
  1. **SP2 d/l_ex = 30 cell-size study** (the user's call: a fundamental scaling study to keep, 2026-10-20). Run H_c/Ms at d/l_ex = 30 with nz = 3, 4, 5, 7 cells through the thickness on a GPU; nz = 5 is OOMMF 1.1's 0.60 l_ex, and nz = 6 (0.5 l_ex) gave 0.04533. If nz = 5 lands within 1% of 0.04594, the test should use OOMMF's cell; otherwise the references disagree at this size. A 6d candidate. The script is the `sp2_conv.py` cell below.
  2. **The 6d design:** an original study using the 6b/6c tools.
  3. **Iteration 7** (stretch).
- **Optional recorded runs** (D32 Correction): the full SP2 curve and the 100 nm core–shell cube (`examples/sp2.toml`, `examples/cube_core_shell.toml`), for benchmark data only.
- **Run a card:** `uv run micromag run examples/sp4.toml -o runs/sp4` (add `--device cuda` on a GPU machine).
- **Optional:** the cluster is one GPU target among many (D25). Its build needs numpy and h5py capped at versions with glibc 2.17 wheels (see the optional open issue below).
- **CUDA compile check without a GPU** (rerun after CUDA changes), from the repo root, with Docker Desktop for Apple Silicon:
  ```
  docker run --rm -v "$PWD":/src -w /src -e UV_PROJECT_ENVIRONMENT=/tmp/venv -e SKBUILD_CMAKE_DEFINE=MICROMAG_CUDA=ON \
    nvidia/cuda:12.1.1-devel-ubuntu22.04 bash -c \
    'apt-get update -qq && apt-get install -y -qq curl >/dev/null && curl -LsSf https://astral.sh/uv/install.sh | sh && . $HOME/.local/bin/env && uv sync'
  ```
  The laptop's original Docker Desktop was an Intel 4.3.0 build that refuses to start on Apple Silicon; the user reinstalled the arm64 build (4.95.0) on 2026-10-18.
- **GPU run on Colab:** pick the T4 runtime, clone, install uv, set `PATH` to include `/root/.local/bin` and `/usr/local/cuda/bin`, `CUDACXX=/usr/local/cuda/bin/nvcc` and `SKBUILD_CMAKE_DEFINE=MICROMAG_CUDA=ON` in `os.environ` (a later `uv run` without that variable silently rebuilds without CUDA), then `uv sync` and `uv run pytest -m "" -v -s`.
- **SP2 cell-size script** (run on Colab after the setup above, `uv run python sp2_conv.py 3 4 5 7`):
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
- **Known test gap (iteration 3):** a wrong embedded-error weight in the LLG integrator makes the error estimate too large, which wastes steps but gives correct results, and no test catches it. A step-count bound on the macrospin run would close the gap. This was offered to the user and not added (iteration 3).

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

## Material regions: settled 2026-10-20 (D33–D37)
The open design (R1–R9, 2026-10-18) was first approached from first principles; the user then chose to copy MuMax3's model. How each question was answered:
- R1 label map: yes, `uint8`, as MuMax3 (D33). R2 shapes in the card; label files deferred (D33). R3 centre membership, with each shape's exact and painted volume in provenance (D33).
- R4 named materials, built-in `vacuum`, later paints win (D33). R5 lower corner, unchanged from D30.
- R6 interphase: a grain-boundary phase in the grain generator, by an exact distance rule (D34). R7 zero-thickness pair coupling: done, as MuMax3 (D35). R8 no distance transform needed (D34). R9 per-region `axis` overrides: yes (D33).

## MagTense weaknesses we want to avoid
- One 3,500-line Fortran module holds global `gb_problem` and `gb_solution` state, so solves are not re-entrant.
- The default demag is a dense O(N²) matrix; the FFT path was added on later.
- Damping is the dimensional LL `alpha0` rather than Gilbert's dimensionless α.
- Tests are example scripts with loose tolerances (SP4 allows 100% error), and there are no unit tests per field term.
- The Python and MATLAB front ends duplicate each other, and the build needs Intel ifx/MKL plus an `#if` flag matrix.
