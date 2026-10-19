# NOTES

## Roadmap
Each iteration is done when its check passes and its decisions are logged below.

| # | Iteration | Check | Status |
|---|-----------|-------|--------|
| 0 | Toolchain: CMake + bindings + pytest + one CUDA kernel, run on a laptop and on a cluster GPU node through Slurm | Kernel test passes in both places | laptop ✅ · cluster optional (see below, D25) |
| 1 | Grid, per-cell materials, exchange / anisotropy / Zeeman (eqs. 8, 9, 11), CPU | Energies of uniform states; H matches the finite-difference gradient of E; analytic 1D Bloch wall (width π√(A/K)) is in equilibrium (torque → 0) with σ → 4√(AK), both errors falling ~4× per halving of dx (D13) | ✅ CPU (`tests/test_fields.py`) |
| 2 | Demag: Newell tensor, direct O(N²) oracle, zero-padded FFT (eqs. 17–21) | Uniform prisms (cube, 1:2:3 cell, 32×32×1 film) match Aharoni's closed form to 1e-10; far-field tensor → point dipole with error ~4× smaller per doubling of distance; FFT and direct sum agree to 1e-12 (D17) | ✅ CPU (`tests/test_demag.py`) |
| 3 | LLG (eq. 25) with a self-written adaptive Dormand–Prince RK45 | Macrospin matches the analytic solution; fixed-step error falls ~32× per halving of h (5th order); energy conserved at α = 0 and non-increasing at α > 0; µMAG SP4 field 1: first ⟨mx⟩ = 0 crossing within 3% of OOMMF and RMS ⟨m⟩ difference ≤ 0.05 over 0–1 ns (D20) | ✅ CPU (`tests/test_llg.py`; SP4 with `-m slow`) |
| 4 | Minimizer: projected steepest descent with Barzilai–Borwein steps (eqs. 22–23), plus a hysteresis driver | Stoner–Wohlfarth astroid: switching fields at ψ = 15–75° within 2·10⁻³ H_K of `(cos^{2/3}ψ + sin^{2/3}ψ)^{−3/2}`; µMAG SP3 (20³): flower/vortex crossover L within 0.5% of 8.47 l_ex (D23) | ✅ CPU (`tests/test_minimizer.py`; SP3 with `-m slow`) |
| 5 | GPU backend: CUDA kernels + cuFFT; the backend interface is introduced here | Every test passes with `device="cuda"` and the direct CPU-vs-GPU comparisons hold their D26 tolerances, run on at least one CUDA GPU of sm_70 or newer (which GPU is recorded). Nsight roofline numbers are optional (D25) | ✅ CPU + GPU (Tesla T4, CUDA 13.0, Colab, 2026-10-18; D26 Result) |
| 6a | Workflow: one simulation from a TOML input card into a results folder (per-stage CSV thermo tables, an HDF5 dump of magnetic states, provenance) through `micromag run` and `micromag.run()` (D27, D28). **Requirement (user, 2026-10-17):** a simple local workflow, parameters in → results out, from an input file, without hand-editing a Python script, with the same entry point on any machine | `examples/sp4.toml` run through `run()` meets D20's SP4 criteria; fast tests of validation, the results folder, restarts, dumps and failure handling (D29) | ✅ (`tests/test_card.py`, `tests/test_run.py`; SP4 card with `-m slow`) |
| 6b | Science runs: interim box material regions, field refinement near switching, a hysteresis summary (H_c, switching field, remanence), on a GPU; scaling study and Nsight optional (D25, D30, D31) | µMAG SP2 H_c and remanence vs d/l_ex against the exact limit and OOMMF 1.1; the pure 40 nm Nd₂Fe₁₄B cube against Fischbacher et al. 2017 (5.29 T); the Fig. 8a core–shell cube against the paper's finest-mesh curve (D32) | CPU checks ✅ (`tests/test_coercivity.py`, `-m slow`); the CUDA-only sizes and the recorded runs need a GPU |
| 6c | An original study using the 6b tools: a question the paper only touches (e.g. shell thickness or anisotropy, field angle, a corner defect) | Set in the 6c design, after 6b is validated | todo |
| 7 | Stretch, pick one: Voronoi grains / string method (NEB) / multi-GPU / thermal field | — | — |

## Resume here
State on 2026-10-18: iterations 1–5 are done. Iteration 5 passes on a Tesla T4 (Colab, CUDA 13.0; D26 Result). `uv run pytest` on the laptop gives 43 passed, 30 skipped (the CUDA cases) and 4 deselected (slow).
- **2026-10-18:** iteration 6 was split into 6a and 6b; 6a is done (D27–D29). Run a card with `uv run micromag run examples/sp4.toml -o runs/sp4` (`runs/` is git-ignored).
- **User plan (2026-10-18):** start using the code to run simulations and log results within 24 hours, i.e. by 2026-10-19. Priority: make the run-and-record workflow (cards → `micromag run` → results folders) dependable for that.
- **Slow suite (user choice, 2026-10-18):** the cube checks stay on both devices (D32), even though they take about 11 minutes on the laptop CPU; `-m slow` is opt-in.
- **Next:** iteration 6b on a GPU: `uv run pytest -m slow -k coercivity -s` on Colab (SP2 at d/l_ex = 20 and 30, the 50 nm cube), then the recorded runs `examples/sp2.toml` (and the other SP2 sizes) and `examples/cube_core_shell.toml`. 6b's CPU checks pass (D32 Result). The large runs (SP2 d/l_ex = 20 and 30, the 50 nm cube, the full SP2 curve, the 100 nm cube) need a GPU (Colab recipe above). The cluster remains an optional GPU target (D25); its build needs the numpy cap (see the optional open issue below).
- **CUDA compile check without a GPU** (rerun after CUDA changes), from the repo root, with Docker Desktop for Apple Silicon:
  ```
  docker run --rm -v "$PWD":/src -w /src -e UV_PROJECT_ENVIRONMENT=/tmp/venv -e SKBUILD_CMAKE_DEFINE=MICROMAG_CUDA=ON \
    nvidia/cuda:12.1.1-devel-ubuntu22.04 bash -c \
    'apt-get update -qq && apt-get install -y -qq curl >/dev/null && curl -LsSf https://astral.sh/uv/install.sh | sh && . $HOME/.local/bin/env && uv sync'
  ```
  The laptop's original Docker Desktop was an Intel 4.3.0 build that refuses to start on Apple Silicon; the user reinstalled the arm64 build (4.95.0) on 2026-10-18.
- **GPU run on Colab:** pick the T4 runtime, clone, install uv, set `PATH` to include `/root/.local/bin` and `/usr/local/cuda/bin`, `CUDACXX=/usr/local/cuda/bin/nvcc` and `SKBUILD_CMAKE_DEFINE=MICROMAG_CUDA=ON` in `os.environ` (a later `uv run` without that variable silently rebuilds without CUDA), then `uv sync` and `uv run pytest -m "" -v -s`.
- **Known test gap:** a wrong embedded-error weight in the LLG integrator makes the error estimate too large, which wastes steps but gives correct results, and no test catches it. A step-count bound on the macrospin run would close the gap. This was offered to the user and not added (iteration 3).
- **Carried requirement:** the local "parameters in → results out" workflow belongs to iteration 6 (see its roadmap row).

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

## Open design: material regions (no decisions yet, 2026-10-18)
Discussed and saved for later. 6b uses interim boxes (D30).
- **What "robust" means here:**
  - any shape, with no solver change
  - no ambiguity: one material per cell, a fixed rule for overlaps, explicit vacuum
  - checkable: staircasing must be visible (a 3 nm shell on 1 nm cells comes out as 2 or 3 cells depending on alignment)
  - resolution-independent: shapes given in metres
  - room for physics that depends on regions (coupling between grains, a different axis per grain)
- **Precedents:**
  - **OOMMF:** "atlases" map geometry to named regions (box, image or scripted atlases); materials are given per region, and A per pair of regions.
  - **MuMax3:** primitive shapes combined into up to 256 numbered regions, Voronoi grains via `ext_makegrains`, inter-grain coupling via `ext_ScaleExchange` / `ext_InterExchange`.
  - **MagTense:** explicit tile lists.
  - **This paper (FEM):** Neper and Salome. The boundary phase is made "by moving the grain surfaces by a fixed distance along their surface normal" (§2.3.2).
- **Candidate design:** a per-cell **region-label map** is the only geometry the solver sees.
  - It is built either from shapes in the card, painted in order with later shapes winning (`all`, `box`, `ellipsoid`; union and difference come from the painting order), or loaded from an h5 label file (Voronoi grains, voxelised CAD, images).
  - Materials are named (`[material.<name>]`), with a built-in `vacuum`.
  - A cell belongs to a shape if its centre is inside.
  - A check compares each shape's exact volume with its volume in cells.
  - Coordinates are measured from the grid's lower corner.
  - The map is saved in `magnetic_states.h5`.
- **Interphase between grains: two physical models.**
  - **(1) A volume layer** with its own Ms, A and K: the paper's 3 nm coat per grain, giving 6 nm between grains and 3 nm at the outer surface. It would be a `shell` operation using a distance transform (e.g. `scipy.ndimage`). It needs no solver change and requires a thickness of at least 2 cells.
  - **(2) A zero-thickness coupling scaled per region pair** (MuMax3, OOMMF). This is an exchange-kernel change on both backends, worth it only for boundaries thinner than a cell.
- **Orientation:** anisotropy is K and the easy axis k̂ per cell (eq. 9), and a grain's orientation *is* its k̂. The solver already stores k̂ per cell (D9).
  - A is isotropic exchange (eq. 8). It affects orientation only through the coupling across boundaries (harmonic mean, D11).
  - Possible card support: per-region `axis` overrides, per-label axes in label files, and random orientation spreads in a grain generator (iteration 7).
  - Only uniaxial anisotropy is supported.
- **Open questions:**
  - R1: a label map, or materials painted directly into per-cell arrays?
  - R2: shapes in the card, label files, or both?
  - R3: centre membership plus the volume check (default 5%), or partial-volume weighting?
  - R4: named materials, a built-in vacuum, and painting order for overlaps?
  - R5: coordinates from the lower corner or from the centre?
  - R6: an interphase `shell` operation (a layer coating each grain)?
  - R7: defer the zero-thickness pair coupling?
  - R8: scipy, or our own distance transform?
  - R9: per-region `axis` overrides?

## MagTense weaknesses we want to avoid
- One 3,500-line Fortran module holds global `gb_problem` and `gb_solution` state, so solves are not re-entrant.
- The default demag is a dense O(N²) matrix; the FFT path was added on later.
- Damping is the dimensional LL `alpha0` rather than Gilbert's dimensionless α.
- Tests are example scripts with loose tolerances (SP4 allows 100% error), and there are no unit tests per field term.
- The Python and MATLAB front ends duplicate each other, and the build needs Intel ifx/MKL plus an `#if` flag matrix.

## Reference benchmarks
Published numbers we compare against. They are kept here so the source and the conditions travel with them.

### µMAG Standard Problem 3: single-domain limit of a cube
Source: https://www.ctcms.nist.gov/~rdm/results3.html (fetched 2026-10-17). L is the cube edge where the flower and vortex energies are equal, in units of l_ex = √(A/K_m), with K_m = µ₀Ms²/2 and K_u = 0.1 K_m along one cube axis. e is the reduced energy E/(K_m L³).

| Submission | Grid | L / l_ex | e at L |
|---|---|---|---|
| Rave, Hubert & Fabian (1998) | not stated | 8.47 | 0.3027 |
| Martins et al. (1998) | 10³ | 8.4073 | 0.2937 |
| Martins et al. | extrapolated from 10³–70³ | 8.4687 | 0.3026 |
| Hertel & Kronmüller (1998), symmetric flower (metastable) | not stated | 8.52 | 0.3049 |
| Hertel & Kronmüller, twisted flower | not stated | 8.57 | 0.3032 |
| **Bjørk, Poulsen & Insinga (2021), MagTense** | **20³** | **8.4556** | 0.3029 |
| Bjørk et al., MagTense | power-law fit, 5³–20³ | 8.477 ± 0.007 | — |

Rave et al.'s partial energies at L = 8.47, in units of K_m L³:

| State | Demag | Exchange | Anisotropy |
|---|---|---|---|
| Flower | 0.2794 | 0.0177 | 0.0056 |
| Vortex | 0.0783 | 0.1723 | 0.0521 |

**Ours (iteration 4, 20³):** L = 8.4551 (MagTense 20³: 8.4556). Flower 0.2797/0.0176/0.0056, total 0.3029; vortex 0.0783/0.1725/0.0520, total 0.3028 (demag/exchange/anisotropy, anisotropy in Rave's convention).

**Ours (iteration 5, GPU, Tesla T4):** identical to the CPU to the printed digits: L = 8.4551, same partial energies.

Coarse grids bias L low (10³ gives about 0.7% low). Hertel's twisted flower is a different, lower-energy state, so it is not comparable to a symmetric-flower computation.

### Stoner–Wohlfarth astroid
The analytic switching field of a single-domain particle with uniaxial anisotropy, with the field at angle ψ to the easy axis: h_sw = (cos^{2/3}ψ + sin^{2/3}ψ)^{−3/2}, in units of H_K = 2K/(µ₀Ms).

**Ours (iteration 4):** one cubic cell, easy axis z, Ms = 8e5 A/m, K = 5e5 J/m³. The field is swept from 0 to −1.2 H_K in steps of 10⁻³ H_K, and the switch is placed at the midpoint of the step with the largest |Δm|. Measured on 2026-10-17.

| ψ | h_sw analytic | h_sw ours | error | max minimizer iterations per field step |
|---|---|---|---|---|
| 15° | 0.6147 | 0.6145 | −2×10⁻⁴ | 163 |
| 30° | 0.5240 | 0.5245 | +5×10⁻⁴ | 151 |
| 45° | 0.5000 | 0.5005 | +5×10⁻⁴ | 139 |
| 60° | 0.5240 | 0.5245 | +5×10⁻⁴ | 128 |
| 75° | 0.6147 | 0.6145 | −2×10⁻⁴ | 112 |

Every error is within one field step, which is the resolution of the sweep. 0° and 90° are excluded, because there the reversed state is an exact (unstable) equilibrium (D23).

### µMAG Standard Problem 4, field 1
Source: OOMMF 1.2 (Donahue & Porter), 1 nm cells. The trimmed time series is `tests/data/sp4_field1_oommf.csv`.
- S-state ⟨m⟩ = (0.9666, 0.1260, 0.0000).
- First ⟨mx⟩ = 0 crossing at 138.4 ps (interpolated from the 1 ps samples).
- Ours (iteration 3, 128×32×1): S-state ⟨m⟩ = (0.9670, 0.1253, 0.0000); crossing at 138.6 ps; RMS ⟨m⟩ difference over 0–1 ns = 0.0051.
- Ours (iteration 5, GPU, Tesla T4): identical to the CPU, including the step count (1979 steps, 1 rejected) and the 2.80 ns S-state relaxation.

### µMAG Standard Problem 2: coercivity and remanence of a thin film vs size
Source: https://www.ctcms.nist.gov/~rdm/std2/spec2.html and `results.html` (fetched 2026-10-18).
- **Specification:**
  - geometry: a film with t/d = 0.1 and L/d = 5
  - energies: exchange and magnetostatic only (K = 0); l_ex = √(A/K_m) with K_m = µ₀Ms²/2
  - procedure: the field along [1,1,1]; report H_c/Ms, where m·[1,1,1] = 0, and the remanence ⟨m⟩ at H = 0, against d/l_ex
  - expected behaviour: the film switches as one rigid spin near d/l_ex ≈ 0.1, rotates non-uniformly around 2, and forms vortices above that
- **Exact small-particle limit:** H_c/Ms = 0.05707; switching field 0.05714.
- **OOMMF 1.1** (Donahue et al., cells of 0.03–0.6 l_ex), selected rows:

| d/l_ex | H_c/Ms | switching field / Ms | mx | my |
|---|---|---|---|---|
| 1 | 0.05704 | 0.05716 | 1.0000 | 0.0000 |
| 10 | 0.05454 | 0.05562 | 0.9989 | 0.0000 |
| 20 | 0.04991 | 0.04991 | 0.9701 | 0.0819 |
| 30 | 0.04594 | 0.04594 | 0.9627 | 0.0756 |

- **MagTense** (Bjørk et al., 100×20×3 cells at every size): H_c/Ms = 0.05705 at d/l_ex = 3.00, 0.04938 at 20.36, 0.04558 at 30.00. It agrees with OOMMF 1.1 to about 0.8%.
- **The older OOMMF 1.0 submission** is off in the small-particle limit because of its demag model (Donahue's note).
- **Ours (iteration 6b, CPU, cell = min(0.5 l_ex, t)):** H_c/Ms = 0.05704 at d/l_ex = 1 and 0.05446 at 10; remanence (1.0000, 0.0000) and (0.9989, 0.0000). The larger sizes are pending a GPU.

### Nd₂Fe₁₄B cubes: switching and coercive fields
- **Pure 40 nm cube** (Fischbacher et al., APL 111, 072404 (2017), arXiv:1708.01880):
  - setup: finite elements, 1.5 nm mesh, T = 300 K; µ₀Ms = 1.61 T, K = 4.30 MJ/m³, A = 7.7 pJ/m; field one degree off the easy axis, which is parallel to one edge
  - results: static switching field µ₀H₀ = 5.29 T, and 6.09 T without magnetostatics (= Stoner–Wohlfarth at 1°)
- **Core–shell cube, Fig. 8a of the review** (finite elements):
  - setup: core K = 4.9 MJ/m³, µ₀Ms = 1.61 T, A = 8 pJ/m; a 3 nm shell with K = 0, µ₀Ms = 0.477 T, A = 6.12 pJ/m. The field direction is not stated.
  - results, read off the figure at 600 dpi on the finest mesh (1.3 nm): µ₀H_c ≈ 2.76 T at 25 nm, 2.71 T at 50 nm, 2.58 T at 100 nm.
  - mesh dependence: the 4 nm mesh sits up to 0.13 T higher, and the curve falls by about 0.002 T per nm.
- **Ours (iteration 6b, CPU, field 1° off z):** the pure 40 nm cube (cells of 1.33 nm) switches at 5.275 T (−0.3%); the 25 nm core–shell cube (1 nm cells) has H_c = 2.721 T (Fig. 8a: 2.76 T).

## Decision log
<!-- Format, one entry per decision:
### D<n>: <title> (<date>)
- **Context:** what forced the decision.
- **Decision:** what we chose.
- **Alternatives:** what we rejected, and why.
- **Consequence:** what this makes easy, and what it makes hard.
-->

### D1: C++/CMake + CUDA core with Python bindings (2026-10-16)
- **Context:** we have 2.5 months on an NVIDIA/Slurm cluster, and the goal is to learn the architecture of scientific software.
- **Decision:** a C++ core built with CMake, CUDA and cuFFT for the GPU, and Python bindings for setting up problems and for tests.
- **Alternatives:**
  - Python + JAX: fastest to iterate, but teaches less about HPC builds.
  - Julia.
  - Fortran, which would mirror MagTense.
- **Consequence:** we get the full HPC toolchain experience, but iteration is slower. Build friction has to be paid off early, which is why iteration 0 runs on the cluster.

### D2: Uniform Cartesian finite-difference grid with FFT demag first (2026-10-16)
- **Context:** the paper offers both FD and FEM. MagTense supports uniform, tetrahedral and unstructured-prism grids.
- **Decision:** a uniform FD grid only. Demag is computed by FFT convolution from the start, not as a dense matrix.
- **Alternatives:** tetrahedral FEM, which needs a mesher and a hard demag solver (algebraic multigrid or boundary-element methods).
- **Consequence:** O(N log N) demag and simple data layouts. Curved grain boundaries come out staircased; per-cell materials (iteration 7) partly mitigate this.

### D3: Two .md files: CLAUDE.md + NOTES.md (2026-10-16)
- **Context:** we want the least documentation that keeps both the agent and the human oriented across sessions.
- **Decision:** CLAUDE.md for stable rules and commands; NOTES.md for the roadmap, status and decision log.
- **Alternatives:**
  - README: add it when there is an outside user.
  - One file per ADR: split once the log passes about 20 entries.
  - Separate physics spec: the paper serves as the spec.
- **Consequence:** one place to look for anything that changes. The risk is that NOTES.md grows; split it when it does.

### D4: Toolchain is CUDA 12.1 + gcc 8.4, pinned (2026-10-17)
- **Correction (2026-10-17):** Lmod only offers `cuda/12.1` under `gcc/8.4.0/1` or PGI, not under gcc 9.3, so we use gcc 8.4. It still handles C++17 and is supported by CUDA 12.1 and nanobind. The text below originally said gcc 9.3.
- **Context:**
  - The driver supports CUDA up to 12.2, which rules out 12.2+ toolkits and CUDA 13 (13 also drops Volta).
  - Two gcc versions are available: 8.4 and 9.3.
- **Decision:**
  - Use `cuda/12.1` and `gcc/9.3.0/1`.
  - `CMAKE_CUDA_ARCHITECTURES` defaults to 70, and you can override it with a CMake define.
  - The build and the job script load the same modules.
- **Alternatives:** CUDA 11.8, which would work but is older for no reason.
- **Consequence:** If the run modules differ from the build modules, you get libstdc++ or cudart mismatches, so `hpc/test.sbatch` loads the same modules the build used.

### D5: Python comes from uv; the build uses scikit-build-core + nanobind (2026-10-17)
- **Context:** The cluster has no Python module. Borrowing someone else's conda env would make our builds depend on their changes.
- **Decision:**
  - `uv` manages Python and the venv on both the laptop and the cluster.
  - scikit-build-core drives CMake.
  - nanobind provides the bindings.
  - `[tool.uv] cache-keys` make `uv run` rebuild the extension when C++ sources change.
- **Alternatives:**
  - Our own conda or micromamba env: heavier.
  - pybind11: more common, but slower to compile, with larger binaries and no first-class ndarray/DLPack type.
  - A plain CMake build with PYTHONPATH hacks: fragile.
- **Consequence:**
  - The project has one `pyproject.toml`, and `uv.lock` pins Python dependencies everywhere.
  - Compute nodes may lack internet, so we build on the login node and jobs run `uv run --no-sync`.

### D6: CUDA is an explicit opt-in build flag, not auto-detected (2026-10-17)
- **Context:** The laptop has no CUDA; the cluster does.
- **Decision:**
  - `option(MICROMAG_CUDA OFF)`, enabled with `SKBUILD_CMAKE_DEFINE=MICROMAG_CUDA=ON`.
  - The module exposes `has_cuda`.
  - The GPU job script fails hard if `has_cuda` is false.
- **Alternatives:** Auto-detecting nvcc. A missing module would then silently produce a CPU build, and the GPU tests would "pass" by being skipped.
- **Consequence:** The GPU tests use `skipif(not has_cuda)` locally, but they cannot be skipped by accident on the cluster.

### D7: Toolchain tests are bitwise CPU-vs-GPU comparisons (2026-10-17)
- **Context:** We needed to confirm the whole chain works: numpy → nanobind → host → device → kernel → back.
- **Decision:**
  - `scale(x, a, cuda)` multiplies in place.
  - The test uses an odd length (1,000,003) to exercise the kernel's bounds check.
  - It requires bitwise equality. A single IEEE multiply is correctly rounded on both CPU and GPU, so any difference is a bug.
- **Consequence:** For real kernels (reductions, FFTs) bitwise equality won't hold. Each test there must state its tolerance and the reason for it.

### D8: Pin Python 3.13 with `.python-version` (2026-10-17)
- **Context:** With nothing pinned, uv chose 3.14 on the laptop and 3.15 on the cluster. 3.15 is new, and dependency wheels for it may be missing.
- **Decision:** `uv python pin 3.13`. The laptop and the cluster now use the same interpreter.
- **Consequence:** Bump the version deliberately once numpy and the other dependencies ship wheels for newer Python.

### D9: Data model: SoA doubles in numpy-owned arrays, C++ `Grid` (2026-10-17)
- **Context:** iteration 1 needs a layout for m and the per-cell materials that will also suit the GPU (iteration 5) and cuFFT (iteration 2).
- **Decision:**
  - Vectors (m, k̂) are `(3, nz, ny, nx)` (separate component arrays, x fastest); scalars (Ms, A, K) are `(nz, ny, nx)`. Everything is float64.
  - numpy owns all arrays. C++ functions take non-owning nanobind views and write into a caller-supplied `out`.
  - `Grid{nx, ny, nz, dx, dy, dz}` is one C++ struct bound to Python and passed to every call.
  - The binding checks dtype, contiguity (via nanobind's typed `ndarray`, with `.noconvert()` on every array) and shape against `grid`. Without `noconvert`, nanobind silently copies a mismatched array, and a copied `out` would lose the result. It does not enforce |m| = 1 or |k̂| = 1: the fields are defined for any m, the gradient test perturbs m off the sphere, and normalising is the integrator's job.
- **Alternatives:**
  - Interleaved `(nz, ny, nx, 3)`: slightly simpler CPU loops, but strided GPU reads and a reshuffle before every FFT.
  - float32 or a precision template: V100 runs FP64 at half rate, and we want CPU/GPU agreement near 1e-12. No second precision exists yet.
  - A C++ `State` that owns `std::vector`s: needed once data stays resident on the GPU, speculative on the CPU.
  - A Python dataclass grid with loose scalars passed to C++: the grid would be defined twice and shapes would go unchecked.
- **Consequence:** each field function is a pure "views in → out" kernel that is easy to test. Ownership gets revisited in iteration 5.

### D10: Field-first API; energy derived from the field (2026-10-17)
- **Context:** LLG and the minimizer need H_eff, not energy. Energy is still needed for the physics checks.
- **Decision:**
  - Each term computes H (A/m) and **overwrites** `out`: `exchange_field(grid, m, Ms, A, out)`, `anisotropy_field(grid, m, Ms, K, k, out)`, `zeeman_field(grid, Ms, H_ext, out)`.
  - Energy is `E = −c·µ₀·dV·Σ Ms (m·H)`, with c = ½ for the quadratic terms (exchange, anisotropy) and c = 1 for Zeeman. It is a numpy one-liner for now.
  - H_ext is one uniform `(3,)` vector.
- **Alternatives:**
  - Separate energy and field implementations: twice the code, and a mismatch is a silent bug.
  - Accumulating into `out` (as MuMax3 does): saves temporaries in the integrator, but a forgotten zeroing gives a silently wrong field. Switch if iteration 3 profiling says so.
  - Energy in C++: a reduction kernel to port and tolerance-justify (D7) before anything needs it.
  - Per-cell H_ext: nothing on the roadmap uses one.
- **Consequence:** one code path per term, checked two independent ways: closed-form uniform energies, and H against a finite-difference gradient of E. The gradient test also proves the exchange operator is symmetric.

### D11: Exchange discretisation: vacuum cells, Neumann boundaries, harmonic-mean A (2026-10-17)
- **Context:** the paper's microstructures vary A sharply across grain boundaries, and real geometries are shapes cut out of a box.
- **Decision:**
  - Second-order 6-neighbour Laplacian.
  - `Ms = 0` means vacuum: H = 0 there, and vacuum cells are not exchange neighbours.
  - Free (Neumann) boundaries: a missing or vacuum neighbour contributes nothing.
  - The coupling across a face uses `A_face = 2·A₁A₂/(A₁+A₂)` (0 if either is 0).
- **Alternatives:**
  - Arithmetic mean of A: lets a strong grain couple straight through a non-magnetic boundary phase.
  - Periodic boundaries: no test needs them. The 1D wall uses a grid 1 cell wide in y and z, which is exact without demag.
  - No vacuum support: it would have to be retrofitted later, along with "fills the box" assumptions in the tests.
- **Consequence:** matches MuMax3's scheme. Interfaces are handled by flux continuity, and arbitrary shapes cost one `if`.

### D12: Uniaxial anisotropy with the paper's sign; µ₀ = 4π×10⁻⁷ (2026-10-17)
- **Context:** anisotropy energy can be written as `−K(m·k̂)²` or `K(1 − (m·k̂)²)`. µ₀ has been a measured constant since the 2019 SI redefinition.
- **Decision:** eq. 9 as written (uniform m ∥ k̂ has energy −K·V). Uniaxial only. `mu0 = 4π·1e-7`, defined once in C++ and exported.
- **Alternatives:** the `K(1 − …)` form (same fields, different constant); cubic anisotropy (not on the roadmap); CODATA µ₀ (differs by ~5e-10 relative and breaks exact comparison with MuMax3 and the µMAG references).
- **Consequence:** energies are directly comparable with the paper.

### D13: Iteration-1 check: analytic Bloch wall in equilibrium, verified by convergence order (2026-10-17)
- **Context:** the roadmap check was "Bloch wall width = π√(A/K)", but there is no minimizer until iteration 4 to relax a wall.
- **Decision:**
  - Place the exact profile `θ(x) = 2·atan(exp(x/δ₀))`, `δ₀ = √(A/K)` (width π·δ₀) on the grid.
  - Assert that the torque `m × H_eff` → 0 and that the wall energy → `σ = 4√(AK)` per unit area.
  - A factor-of-2 error in either field shifts the true width by √2, so the torque would no longer vanish: the torque check tests the width.
  - Pass/fail is the convergence ratio: both errors must fall ~4× (accepted range 3.5–4.5) as dx goes δ₀/4 → δ₀/8 → δ₀/16, with a loose absolute bound as a backstop.
  - Use NdFeB-like values (A = 7.7e-12 J/m, K = 4.3e6 J/m³, Ms = 1.28e6 A/m) on a domain 40·δ₀ long.
- **Alternatives:**
  - A small relaxer inside the test: a throwaway copy of iteration 4.
  - Deferring the width check to iteration 4: iteration 1 would then have no physics check.
  - A fixed tolerance alone: a small constant-error bug could hide under it, but it cannot fake second-order convergence.
- **Consequence:** the check verifies the discretisation order, not just a number. The roadmap check text was reworded to match.

### D14: Demag tensor: Newell near, Gauss-averaged dipole far, mirrored from one octant (2026-10-17)
- **Context:** the Newell formula (Newell, Williams & Dunlop 1993) sums 27 large terms that nearly cancel. Measured against mpmath at 50 digits, its error in double grows like ε·r⁶: 2e-10 at 10 cells, 8e-7 at 30, 7e-3 at 127. A plain point dipole is only accurate to O(1/r²) for non-cubic cells (4e-3 at 10 cells for 1:1.5:2).
- **Decision:**
  - Newell f/g formulas for offsets below 20 cells, measured as √(i² + j² + k²) in cell counts.
  - Beyond that, the dipole kernel averaged over 3-point Gauss–Legendre nodes per axis in both cells. The nodes are symmetric, so the 27×27 pairs collapse to 5 distinct differences per axis, i.e. 125 kernel evaluations per offset.
  - Target accuracy: 1e-6 relative to the largest entry at that offset. Measured at the cutoff, the worst cases are about 4e-8 (cubic), 3e-7 (1:1.5:2) and 7e-7 (1:1:0.25).
  - Compute one octant of offsets and mirror it. Diagonal components are even in each offset coordinate, and Nxy is odd in x and y. Nyy, Nzz, Nxz and Nyz come from the same f and g by permuting axes.
  - `demag_tensor(grid)` returns `(6, Pz, Py, Px)` (xx, yy, zz, xy, xz, yz) in FFT wraparound order, so a negative index is a negative offset. Index n of a padded axis is unused and holds 0.
- **Alternatives:**
  - Newell everywhere: about 2 digits left at 127 cells.
  - Point dipole far: cannot reach better than about 1e-3 for non-cubic cells at reasonable cutoffs.
  - More Gauss points (4–5 per axis) to keep 1e-8: slower setup, and flat cells still need more.
  - OOMMF's higher-order asymptotic expansion (Donahue): most accurate and fast, but a lot of formula. A stretch goal if a test ever needs better than 1e-6.
- **Consequence:** the tensor is good to about 1e-6 (as built, the worst measured case is 5.3e-7: 1:1:0.25 cells, 21 cells away along z), far below the O(dx²) discretisation error (about 1e-3). **Known limit:** cells flatter than about 1:4 are untested and may exceed 1e-6 at the cutoff.

### D15: FFT convolution: pocketfft, 2n zero padding, real-to-complex (2026-10-17)
- **Context:** eq. 21 is a convolution, because N depends only on the offset between cells. The CPU path is the oracle; cuFFT is the production path.
- **Decision:**
  - pocketfft (the C++ header-only version, BSD-3), fetched with CMake FetchContent at a pinned commit and run single-threaded.
  - Pad each axis to 2n; an axis with n = 1 stays at 1.
  - Real-to-complex transforms for M = Ms·m and N. The kernel spectrum is computed once.
- **Alternatives:**
  - FFTW: faster, but GPL, and it needs a system install on every machine.
  - numpy FFT: unusable from the C++ LLG integrator.
  - 2n−1 padding: can be a prime length, which is slow to transform.
  - complex-to-complex: about 2× the work and memory.
  - Multithreading: needs OpenMP and gives results that are not reproducible run to run, for an oracle.
- **Consequence:** identical builds on the laptop and the cluster. The real-to-complex layout maps one-to-one onto cuFFT D2Z/Z2D in iteration 5.

### D16: `Demag(grid)` owns its workspace (refines D9) (2026-10-17)
- **Context:** demag is the first term that does expensive work once (the tensor and its spectrum) and reuses it on every call.
- **Decision:** a C++ class `Demag(grid)` holds the kernel spectrum and the padded scratch buffers. `demag.field(m, Ms, out)` keeps the D10 conventions: it overwrites `out`, sets H = 0 in vacuum, rejects mismatched arrays, and uses energy prefactor c = ½. numpy still owns everything that crosses the API; objects own their internal workspace.
- **Alternatives:** a free function that rebuilds the kernel on every call, which defeats the FFT. A global cache, which is forbidden by CLAUDE.md.
- **Consequence:** the GPU version (iteration 5) can keep the spectrum resident on the device behind the same method.

### D17: Iteration-2 oracles and check (2026-10-17)
- **Context:** the FFT path and any direct sum share the tensor, so comparing them alone cannot catch a wrong tensor.
- **Decision:**
  - **Aharoni (1998):** the closed-form demag factors of a rectangular prism. Uniformly magnetise a prism of cells, run `Demag.field`, and require −⟨H_a⟩/Ms to match to 1e-10. Cases: an 8³ cube, a single 1:2:3 cell (the self-term, trace 1), and a 32×32×1 film. This replaces the roadmap's loose "Nzz → 1".
  - **Point-dipole limit:** with 1:1.5:2 cells (for cubic cells the error falls like 1/r⁴, not 1/r²), Newell's error relative to the point dipole falls about 4× from offset (10,7,3) to (20,14,6). This is the only check on the sign and magnitude of the off-diagonal components.
  - **mpmath:** the same f and g formulas at 50 digits. The tensor must match to 1e-6 at offsets up to 127 cells, for cubic, 1:1.5:2 and 1:1:0.25 cells.
  - **Direct sum in numpy (tests only):** must match the FFT path to 1e-12 (on the norm), with odd sizes, vacuum, non-cubic cells and nz = 1.
  - Demag joins the iteration 1 gradient, vacuum and rejection tests. The gradient test proves N_ij = N_jiᵀ, which checks the mirroring.
- **Alternatives:** the roadmap's original "trace = 1, Nxx = 1/3, Nzz → 1", which is weaker and does not test the off-diagonal components.
- **Consequence:** mpmath becomes a dev dependency. The roadmap check was reworded to match.

### D18: LLG integrator: C++ Dormand–Prince RK45 (2026-10-17)
- **Context:** iteration 3 needs time integration of eq. 25. Once the arrays live on the GPU (iteration 5), a Python-driven loop would copy m between host and device at every stage.
- **Decision:**
  - The LL form of LLG (eq. 25): `dm/dt = −γ′ m × H − α γ′ m × (m × H)`, with `γ′ = µ₀|γ|/(1+α²)`.
  - α (dimensionless Gilbert damping) and γ are scalars on the integrator. γ defaults to the paper's 1.76086e11 rad/(T·s); SP4 passes its own µ₀γ = 2.211e5 m/(A·s).
  - Dormand–Prince 5(4) in C++, with FSAL. The reused stage is discarded at the start of every `run_until` call, because Python may have changed m or H_ext in between.
  - Error control: the step error is the largest |m₅ − m₄| over all cells. `h_new = h·0.9·(tol/err)^(1/5)`, clamped to 0.2–5×. A step is rejected if err > tol.
  - m is renormalised after each accepted step. The error estimate is taken before renormalising.
  - `run_until(m, t)` shortens its last step to land exactly on t. m is numpy-owned and updated in place. `max_step` defaults to ∞.
  - Defaults: tol = 1e-6. The first step is 1% of a precession period at the current largest field. The step size carries over between calls.
  - The integrator exposes `t`, `steps` and `rejected`.
- **Alternatives:**
  - A numpy driver: easiest to read, but forces device↔host copies on every stage on the GPU.
  - Per-cell α: nothing on the roadmap varies it.
  - No renormalisation: |m| drift accumulates over long runs.
  - A geometric (norm-preserving) integrator: a different family of method.
  - RMS error norm: it dilutes a few fast-switching cells in a large grid.
  - Dense output: more coefficients to get right, for a gain we don't need.
  - Keeping the reused stage across calls: faster, but stale-state bugs show up only as subtly wrong physics.
- **Consequence:** a fresh field evaluation per output sample costs about 8% at SP4's 1 ps sampling. Switch to invalidating the stored stage only on change if profiling says so. The GPU port changes where the arrays live, not who drives the loop.

### D19: `Problem` assembles H_eff (2026-10-17)
- **Context:** the integrator, and the minimizer in iteration 4, need the sum of the four field terms.
- **Decision:**
  - `Problem(grid, Ms, A, K, k, H_ext)` copies the material arrays and owns a `Demag` and one scratch buffer.
  - `effective_field(m, out)` sums the terms. Each term keeps D10's overwrite semantics and is added from the scratch buffer.
  - `H_ext` is settable. All four terms are always on; set K = 0 or H_ext = 0 to switch one off.
  - `LLG` holds its `Problem` through nanobind `keep_alive`.
- **Alternatives:**
  - Storing pointers into numpy arrays: Python could free them while C++ still uses them.
  - Accumulate semantics for the terms: D10 says switch only on profiling evidence.
  - Flags to enable each term: unneeded, since zero parameters already disable a term.
- **Consequence:** changing a material means building a new `Problem`.

### D20: Iteration-3 checks (2026-10-17)
- **Context:** an adaptive integrator hides its own bugs. A wrong coefficient only lowers the order, and the controller compensates with smaller steps.
- **Decision:**
  - **Macrospin:** one cell, field along z, α = 0.1, θ₀ = 30°, 3 periods. The analytic solution is `φ = φ₀ + ωt`, `tan(θ/2) = tan(θ₀/2)·e^(−αωt)`, with `ω = µ₀|γ|H/(1+α²)`. Error ≤ 10·tol at tol = 1e-8. The demag field of a single cube is parallel to m, so it exerts no torque.
  - **Order:** tol = ∞ with max_step = h forces fixed steps. The error ratio per halving of h must be 26–38 (5th order gives 32).
  - **Energy:** on a random multi-cell problem, the relative drift is < 1e-6 at α = 0 (tol 1e-9); energy never increases between samples at α > 0; |m| = 1 within 1e-14.
  - **µMAG SP4:** a 128×32×1 grid (3.9×3.9×3 nm, below the 5.7 nm exchange length). The S-state comes from relaxing uniform (1,1,1)/√3 at H = 0 and α = 1 until max |m × H|/Ms < 1e-6, and its ⟨m⟩ must be within 0.01 of OOMMF's (0.9666, 0.1260, 0). For field 1 (−24.6, 4.3, 0 mT), α = 0.02, µ₀γ = 2.211e5: the first ⟨mx⟩ = 0 crossing must be within 3% of OOMMF's, and the RMS ⟨m⟩ difference over 0–1 ns ≤ 0.05.
  - **SP4 reference:** the OOMMF 1.2 time series (Donahue & Porter, 1 nm cells) from https://www.ctcms.nist.gov/~rdm/std4/Donahue/stdprob4a.odt, trimmed to t ≤ 1 ns and committed as `tests/data/sp4_field1_oommf.csv` with a provenance header.
  - SP4 is marked `slow` and excluded from the default `uv run pytest`. Run it with `-m slow`; the cluster job runs everything.
- **Alternatives:**
  - Field 2: the µMAG page reports that the solvers diverge on it.
  - Downloading the reference data at test time: compute nodes have no internet.
  - The µMAG field ramp from saturation: costlier, and it ends in the same S-state basin.
  - OOMMF's 1 nm grid: 187,500 cells, which waits for the GPU.
- **Consequence:** if SP4 fails, the first response is a grid-refinement study, not a looser tolerance.
- **Finding (2026-10-17):** the S-state relaxation must use tol = 1e-9. At tol = 1e-6, each step's error re-excites a torque of a few × tol, so the max torque plateaus at about 5e-6 and never reaches the 1e-6 threshold. Lowering tol costs almost nothing here: at α = 1 the step size is set by the stability of the stiff exchange term, not by accuracy (similar step counts at 1e-6 and 1e-9). General rule: a torque-based stopping threshold must sit well above tol.
- **Result:** the S-state ⟨m⟩ is (0.9670, 0.1253, 0.0000) after 2.8 ns. The first ⟨mx⟩ = 0 crossing is at 138.6 ps (OOMMF 138.4 ps, 0.15%). The RMS ⟨m⟩ difference over 0–1 ns is 0.0051. The whole run takes about 23 s on the laptop CPU.

### D21: Minimizer: projected steepest descent with Barzilai–Borwein steps (2026-10-17)
- **Context:** permanent-magnet fields change slowly compared with precession, so a hysteresis loop is a sequence of energy minima (paper §2.2.3). Iteration 4 needs a minimizer (eqs. 22–23).
- **Decision:**
  - A C++ `Minimizer(problem)`, which holds its `Problem` through nanobind `keep_alive`.
  - Direction `d = −m×(m×H)`, in A/m (eq. 22). Working in field units avoids eq. 23's µ₀MsV factor, which matters when Ms varies per cell.
  - Update `m ← normalise(m + τd)`. τ alternates BB1 (`s·s / s·y`) and BB2 (`s·y / y·y`), with s = Δm and y = Δd. No line search.
  - Initial step `τ₀ = 0.01/max|d|`, which turns the largest-torque cell by about 0.01 rad. If `s·y ≤ 0` (non-convex region, common near switching), fall back to the same τ₀ rule. τ resets on every `minimize` call.
  - `minimize(m, torque_tol=1e-6, max_iter=100_000)` stops when max|m×H|/Ms < torque_tol, updates m in place, and returns the iteration count. It raises if it reaches max_iter, so a half-converged state near a switching field is never returned silently.
- **Alternatives:**
  - Adding a line search: monotone, but costs extra field and demag evaluations per iteration.
  - Nonlinear conjugate gradient: more machinery for a similar result at our sizes.
  - Gill's threefold stopping criterion: three tolerances to tune.
  - A Python loop: same GPU objection as D18.
- **Consequence:** this is Exl et al. (2014)'s method and essentially MuMax3's `minimize`. The energy can rise on individual iterations; convergence is judged by torque only.

### D22: `Problem.energies` in C++; hysteresis driver in Python (refines D10) (2026-10-17)
- **Context:** the hysteresis driver and SP3 need energies from a `Problem`. `Problem` copies its materials (D19), so the numpy formula from D10 would need the arrays a second time.
- **Decision:**
  - `Problem.energies(m)` returns `{exchange, anisotropy, zeeman, demag}` in J, each `−c·µ₀·dV·Σ Ms(m·H)` with c = ½, ½, 1, ½.
  - `mm.hysteresis(problem, m, fields, minimizer)` (Python) minimizes at each field in `fields` (shape (n,3), A/m) and returns a flat dict of arrays: `H` (n,3), `m` (n,3) = ⟨m⟩ over magnetic cells (Ms > 0), the four energies (n,), and `iterations` (n,). The final state stays in `m`.
- **Alternatives:**
  - Keeping energy in numpy (D10): duplicates material state.
  - Returning only the total energy: SP3 asks for the partial energies, and Rave et al. publish theirs.
  - Averaging ⟨m⟩ over all cells: vacuum dilutes it.
  - Storing full-m snapshots per field: costs n×3N numbers before anything needs them.
- **Consequence:** `np.savez(path, **result)` saves a loop, which is the seed of iteration 6's results output. The GPU port needs a reduction kernel for the energies.

### D23: Iteration-4 checks (2026-10-17)
- **Context:** the minimizer needs one check with an exact answer (the macrospin) and one multi-cell check against published results.
- **Decision:**
  - **Stoner–Wohlfarth astroid:**
    - Setup: one cubic cell (its demag field is parallel to m), easy axis z, H_K = 2K/(µ₀Ms), field at ψ ∈ {15°, 30°, 45°, 60°, 75°} from the easy axis.
    - Sweep: from 0 to −1.2 H_K in steps of Δh = 10⁻³ H_K.
    - Measurement: the switching field is the step with the largest |Δm|. Away from the switch m moves about Δh per step, and about √Δh just before it; the jump itself is O(1).
    - Pass: within 2Δh of `(cos^{2/3}ψ + sin^{2/3}ψ)^{−3/2}`.
    - 0° and 90° are excluded, because there the reversed state is an exact (unstable) equilibrium that an exact minimizer never leaves.
  - **µMAG SP3** (marked `slow`):
    - Setup: a 20³ cube, K_u = 0.1 K_m along z.
    - Starting states: the flower starts uniform along z, and the vortex starts at `normalise(−y, x, L/5)`. Both starts are symmetric, with no noise, so the result is the symmetric flower that Rave, Martins and MagTense computed, not Hertel's twisted flower.
    - Distinctness: after minimizing, ⟨mz⟩ > 0.9 for the flower and < 0.5 for the vortex.
    - Search: bisect Δe(L) = e_flower − e_vortex over [8, 9] l_ex until the bracket is < 0.005 l_ex. Each new L starts from the previous minimized states; at a fixed grid, changing L only rescales the cells.
    - Pass: |L/8.47 − 1| ≤ 0.5%. MagTense's 20³ value (8.4556) and Rave et al.'s partial energies are printed for comparison (see Reference benchmarks).
- **Alternatives:**
  - µMAG SP2: needs grids up to about 30 l_ex, a job for the GPU in iteration 6.
  - The secant method: can diverge if Δe(L) is not smooth.
  - scipy's root finders: a new dependency for 10 lines.
  - Random-noise starts: a different state and a different reference value.
- **Consequence:** the roadmap check was reworded to match.
- **Correction (2026-10-17):** the vortex start is `normalise(0.2, −z, y)`, which circulates around x, **perpendicular** to the easy axis. The design said `normalise(−y, x, L/5)`, around z. That puts m in-plane against the anisotropy, and both the minimizer and damped LLG (α = 1, so not a basin-hopping artefact) relax it into the flower even at L = 9. With the corrected start at L = 8.47, our partial energies match Rave et al.: flower demag/exchange/anisotropy 0.2796/0.0176/0.0056 (Rave 0.2794/0.0177/0.0056); vortex 0.0780/0.1723/0.0519 (Rave 0.0783/0.1723/0.0521). Our anisotropy values are shifted +0.1 into Rave's K(1 − (m·k)²) convention (D12).
- **Result:** at 20³, the crossover is L = 8.4551 l_ex. That is 0.18% below the 8.47 consensus and 0.0005 l_ex from MagTense's own 20³ result (8.4556). Stoner–Wohlfarth switching fields agree with the astroid to within 5×10⁻⁴ H_K at every angle tested. The SP3 run takes about 8 s on the laptop CPU.

### D24: One driver, two backends; the device is chosen on `Problem` (refines D9, D16, D18, D19) (2026-10-18)
- **Context:** iteration 5 is the first second implementation, so the backend interface is introduced here. D18 and D21 put the LLG and minimizer loops in C++ so that m never crosses between host and device inside a run.
- **Decision:**
  - `LLG` and `Minimizer` keep one copy of their control logic: step accept/reject and step-size control (D18), and the Barzilai–Borwein steps (D21). They call a small backend interface for the per-cell work: effective field, energies, the LLG right-hand side, combining Runge–Kutta stages, the per-cell error max-norm, normalising, the minimizer's direction and torque, and the BB dot products.
  - `CpuBackend` wraps the existing loops unchanged. `CudaBackend` runs kernels on buffers it owns on the device.
  - `Problem(..., device="cpu" | "cuda")`. The materials are uploaded once. `LLG` and `Minimizer` follow their `Problem`'s device.
  - m stays a numpy array (D9). It is copied to the device at the start of each `run_until` or `minimize` call and back at the end, with no host copies in between. `effective_field` and `energies` copy in and out per call.
  - `device="cuda"` on a build without CUDA raises.
- **Alternatives:**
  - Separate `CudaLLG` and `CudaMinimizer` classes: no interface, but the subtle controller logic lives in two places and can drift.
  - GPU fields only, with the drivers on the host: per-stage transfers, which D18 rejected.
  - Separate Python classes (`mm.cuda.Problem`): every test and `hysteresis` would need two code paths.
  - Device arrays in Python (CuPy, DLPack): a new dependency before anything needs it.
- **Consequence:** the CPU backend is the oracle for the CUDA one by construction, and the existing physics tests run on the GPU unchanged apart from a `device` parameter. The interface is shaped by exactly the two drivers that use it.
- **Refinement (2026-10-18, implementation):** `Problem.field(term, m, out)` exposes one term, so the per-term tests (gradient, vacuum, Bloch wall, Aharoni, FFT vs direct sum) reach the GPU. The free functions and `Demag` stay CPU-only: they are the oracle that `CpuBackend` calls. `effective_field` and `energies` moved out of the backends into `Problem`, built from `field`, `lincomb` and `ms_dot`, so their prefactors exist once. The interface has 12 methods (`src/backend.hpp`).
- **Result:** on the CPU, the refactor reproduces the iteration 3–4 numbers exactly: SP4 crossing 138.6 ps, RMS 0.0051, 1979 steps with 1 rejected; SP3 L = 8.4551 l_ex. `uv run pytest`: 43 passed, 30 skipped (the CUDA cases), 4 deselected (slow). Mutation check on the new CPU path, each bug caught: `lincomb` dropping h on the base path (fifth order, energy, Stoner–Wohlfarth); `effective_field` omitting demag (energy conservation at α = 0 only; SP4 with `-m slow`); the minimizer stepping along the new direction instead of the previous one (Stoner–Wohlfarth); `field("anisotropy")` dispatching to exchange (the oracle comparison).

### D25: CUDA portability: any sm_70+ GPU, float64, cluster optional (refines D4, D6) (2026-10-18)
- **Context:** the user expects to run on machines and GPUs beyond the V100 cluster later, and chose not to block iteration 5 on the cluster build. The laptop has no NVIDIA GPU.
- **Decision:**
  - `CMAKE_CUDA_ARCHITECTURES` stays 70 by default. CMake's `70` embeds both V100 machine code and compute_70 PTX, which the driver JIT-compiles for any newer GPU (T4, A100, RTX, H100) on first load.
  - Precision stays float64 (D9).
  - The demag tensor is still built on the CPU with the D14 code and uploaded once. The convolution runs on the GPU in cuFFT (batched D2Z/Z2D).
  - Device-wide sums and maxima use CUB, which ships with the CUDA toolkit.
  - **Local check:** a CUDA build inside an `nvidia/cuda` Docker image. nvcc compiles without a GPU, so compile errors surface on the laptop.
  - **GPU check:** the tests run unchanged on any NVIDIA GPU of sm_70 or newer with CUDA 12. The cluster is one optional target, and so are its numpy fix and the Nsight roofline numbers.
- **Alternatives:**
  - `all-major` architectures: no first-load JIT delay, but several times the compile time.
  - `native`: fails on GPU-less build hosts (login nodes, Docker).
  - A float32 or mixed-precision path: no second precision is needed yet (D9).
  - Fixing the cluster first: the earlier recommendation, declined by the user.
- **Correction (2026-10-18, implementation):** the design said the kernel *spectrum* would be built on the CPU and uploaded. Instead the tensor is uploaded and transformed by cuFFT with the same plan layout as M. cuFFT plans leave out length-1 axes, so on a thin film the halved axis is y, not x, and a pocketfft spectrum would not match cuFFT's layout. A 1×1×1 grid has no axis to transform and multiplies by N(0) directly.
- **Correction (2026-10-18, measured on Colab):** CUDA 13 dropped Volta, so the default `70` is not portable: Colab's nvcc 13.0.88 rejects it (`nvcc fatal : Unsupported gpu architecture 'compute_70'`). The default is now chosen by toolkit version (user's choice): 70 for CUDA < 13, 75 (T4, the oldest CUDA 13 supports) for CUDA ≥ 13. The embedded PTX still JIT-compiles for newer GPUs. `-DCMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS` overrides it. Checked in Docker with CUDA 12.1: the build compiles for `compute_70,sm_70`. Rejected: keeping 70 with a manual override on every CUDA 13 machine; `all-major` (several times the compile, deprecation warnings).
- **Implementation note:** the CUDA sources build as their own static library (`micromag_cuda`, linking `CUDA::cudart_static` and `CUDA::cufft`), so nanobind's host compiler flags never reach nvcc.
- **Consequence:** the build is portable without per-machine flags. **Known limit:** consumer RTX cards run FP64 at 1/32 to 1/64 of their FP32 rate, compared with 1/2 on V100 and A100, so they will be slow. GPU tests cannot go red against stubs, and the mutation check cannot run, until a GPU machine is used. Iteration 5 stays unchecked until then.

### D26: Iteration-5 checks (2026-10-18)
- **Context:** D7 requires every CPU-vs-GPU comparison that is not bitwise to state its tolerance and the reason.
- **Decision:**
  - **Physics:** the existing tests take a `device` parameter (`cpu`, `cuda`), and the CUDA case is skipped without a CUDA build: macrospin, fifth order, energy, gradient, vacuum, Bloch wall, Aharoni, SP4, Stoner–Wohlfarth and SP3.
  - **Direct comparisons:**
    - Exchange and anisotropy fields agree to ≤ 1e-14·max|H|: nvcc fuses multiply-adds, and per-cell operation order can differ.
    - The Zeeman field is bitwise equal, because it is a copy.
    - cuFFT demag agrees with pocketfft to 1e-12 on the norm, the D17 bar for a different FFT.
    - Energies agree to 1e-12 relative, because the reduction order differs.
    - A fixed-step LLG run (tol = ∞, `max_step` = h, 5 steps on the random problem) agrees to 1e-12 in max |Δm|. Adaptive runs are not compared directly, because rounding differences change the accepted step sequence; the physics tests cover them.
  - Python-level tests of the new interface run on the CPU too: an unknown `device` raises, and `device="cuda"` raises on a build without CUDA.
- **Alternatives:** bitwise equality (impossible with fused multiply-adds and cuFFT), or comparing only the physics tests (a constant-factor kernel bug could hide inside a loose physics tolerance).
- **Consequence:** the roadmap check was reworded to match.
- **Result (2026-10-18):** on Colab, a Tesla T4 (sm_75) with nvcc 13.0.88, GCC 13.3 and Python 3.13.15, built with the toolkit-chosen architecture 75 (D25 correction): `pytest -m ""` gives 76 passed, 1 skipped (the no-CUDA-build check) in 141 s. Every physics test passes on both devices, and every direct comparison holds its tolerance. SP3 and SP4 print the same numbers on the GPU as on the CPU, including SP4's step count (1979 accepted, 1 rejected), so the rounding differences never changed an accept/reject decision. T4 runs FP64 at 1/32 of FP32, so its timings are not performance figures.
- **Mutation check (GPU, run on Colab):**
  - M1, demag multiply using N_xy in place of N_xz in the z row: caught by FFT vs direct sum, the demag gradient, the oracle comparison, energy conservation, the fixed-step comparison and the energies test. Aharoni does not catch it: a symmetric prism has zero off-diagonal body tensor (as D17 anticipated).
  - M2, exchange using dx in place of dy for the +y neighbour: caught by the exchange gradient test (the operator is no longer symmetric), the oracle comparison, energy conservation, the fixed-step comparison and the energies test.
  - M3, `lincomb` dropping its last term: caught by the macrospin test (the run used `-x`, so only the first failure is recorded). Without `-x` the run is very slow, because the broken step keeps the error estimate high and the controller keeps shrinking the step.
  - M4, the BB sums over the x components only: caught by Stoner–Wohlfarth at 15° (with `-x`). The design expected this one might survive.

### D27: Input card: TOML, setup then ordered stages (2026-10-18)
- **Context:** iteration 6a needs one simulation described by a file, with no Python edits (user requirement, 2026-10-17). The user asked for something like a LAMMPS input script, formatted as YAML unless something better fits.
- **Decision:**
  - **TOML**, read by the standard library's `tomllib` (Python ≥ 3.11; `requires-python` rises to 3.11).
  - **Setup sections:**
    - optional `device` (default "cpu")
    - `[grid]`: `n` (cells) and `cell` (m)
    - `[material]`: `Ms` (A/m) and `A` (J/m), plus optional `K` (J/m³, default 0) and `axis` (default [0, 0, 1])
    - `[initial]`: exactly one of `m = [...]` (uniform, normalised on load) or `file` pointing to a `magnetic_states.h5`. A file takes an optional `state`: `"final"` (the default), or `"stage<k>"` for the last snapshot of stage k. The path is relative to the card. The file's SHA-256 is recorded, and its grid must match the card's.
  - **Stages:** an ordered list of `[[stage]]` entries, run top to bottom. The card has no variables, loops or conditionals; sweeps belong to 6b.
  - **Units:** SI throughout. Applied fields are `mu0_H`, µ₀H_ext in tesla, because every reference quotes T or mT.
  - **What carries over:** the magnetisation always carries over between stages, as an MD configuration does. The applied field is **stage-local**: it defaults to zero and is never carried over. Every thermo row records the field it was computed at.
  - **Stage types** (they solve different problems; see the physics notes in the 6a design discussion):
    - **`llg-dynamics`**: eq. 25 in real time.
      - Keys: `alpha` (required), `gamma` (default 1.76086e11 rad/(T·s)), `mu0_H`, `tol` (default 1e-6), `max_step` (default none), optional `dump_every` (s).
      - Stopping rule, exactly one of: `duration` with an optional `sample_every` (rows at the start, every interval and the end); or `until_torque`, `check_every` and `max_duration`, all required.
      - Validation rejects `tol > until_torque/100`: each step's error re-excites a torque of a few × tol, so the threshold would never be reached (D20).
    - **`minimize`**: eqs. 22–23, the nearest equilibrium at a fixed field. Keys: `mu0_H`, `torque_tol` (1e-6), `max_iter` (100000).
    - **`hysteresis`**: a quasi-static sweep, minimizing at each field (D22).
      - Keys: `direction` (normalised), `mu0_H_from`, `mu0_H_to` and `mu0_H_step` (T, step > 0, from → to inclusive), `torque_tol`, `max_iter`, optional `dump = true`.
      - A full loop is two stages.
  - **Validation** is strict and runs before anything else: an unknown key is an error that names it and lists the valid keys, and missing required keys and wrong types or shapes are errors too. It's plain Python with dataclasses.
- **Alternatives:**
  - YAML: PyYAML reads `8e5` as the string `'8e5'`, while it reads `1.3e-11` as a float (checked with PyYAML 6.0.3). That's a silent trap in an SI input file, and YAML is also a new dependency.
  - A LAMMPS-style command language: a parser and error messages to write and test.
  - Carrying the field over between stages, as a LAMMPS `fix` persists: discussed and rejected by the user; the field is an external condition each stage states for itself.
  - Fields in A/m: −19,576 A/m instead of −24.6 mT invites mistakes.
  - Both `gamma` and `mu0_gamma`: two ways to say one thing. SP4's card states γ with a comment.
  - pydantic for validation: a dependency for something this small.
- **Consequence:** a card reads like a LAMMPS script, but each stage can be understood on its own. 6b adds material regions and sweeps on top.

### D28: Entry point and results folder: CSV thermo, HDF5 dump, provenance (2026-10-18)
- **Context:** a run must be reproducible from its folder alone and must never silently overwrite a result.
- **Decision:**
  - **Entry point:** `micromag run CARD -o DIR [--device cuda]`, a thin wrapper around `micromag.run(card, out, device=None)` (`[project.scripts]`). `--device` overrides the card, so one card runs on the laptop, Colab or a cluster.
  - **Output folder:** `-o` is required, and a non-empty `DIR` is refused.
  - **Folder contents:**
    - `card.toml`: copied byte for byte.
    - `provenance.json`: the git hash and dirty flag (`null` with a warning outside a git checkout), micromag, numpy, h5py and Python versions, the device and GPU name, start and end times, wall time per stage, and the status and error.
    - `git.diff`: when the tree is dirty. A dirty tree is recorded and warned about, never refused.
    - **Thermo** (like LAMMPS thermo output): `stage<k>_<type>.csv`, one file per stage because the columns differ by stage type. Rows are flushed as they're written, so `tail -f` works.
      - `llg-dynamics`: `t, mu0_Hx, mu0_Hy, mu0_Hz, mx, my, mz, E_exchange, E_anisotropy, E_zeeman, E_demag, E_total, steps, rejected`.
      - `minimize` and `hysteresis`: `mu0_Hx, mu0_Hy, mu0_Hz, mx, my, mz`, the same energies, and `iterations`.
      - ⟨m⟩ is averaged over magnetic cells. Time restarts at 0 in each stage. Energies are in J and always recorded; computing them costs about 25% extra at SP4's 1 ps sampling, so make them optional only if profiling says so.
    - **Dump** (like a LAMMPS dump file): `magnetic_states.h5`, written with h5py.
      - Data: float64, so restarts are exact, compressed losslessly with shuffle + gzip.
      - Group `stage<k>`: an extendable `m` stack (n, 3, nz, ny, nx), with `t` (`llg-dynamics`) or `mu0_H` (`minimize`, `hysteresis`) per snapshot. The default is one snapshot at the end of the stage; `dump_every` or `dump = true` adds more.
      - Groups `final` and, on failure, `failed`.
      - Attributes: grid `n`, `cell` (m), units, stage type.
  - **Progress:** one line printed when each stage ends, including the field.
  - **Failure:** a stage that fails (`max_duration`, `max_iter`, or a solver error) stops the run. Everything written so far is kept, the last state goes into `failed`, `provenance.json` records `status: failed` and the error, and the command exits non-zero.
- **Measurements behind the format (2026-10-18):** lossless compression of relaxed float64 states saves little. On the SP4 S-state, zlib gives 0.78× and shuffle + gzip 0.79×. On the SP3 vortex, they give 0.70× and 0.58×. Float32 + shuffle + gzip reaches 0.21–0.34×, but it is lossy and would make restarts inexact. HDF5 was chosen for organisation, not size: one self-describing file, appendable snapshot series, partial reads, and readers in MATLAB, ParaView (via XDMF) and Julia. A 100 nm cube at 1 nm cells is 24 MB per snapshot.
- **Alternatives:**
  - `.npy` per state: no dependency, but no metadata and no series.
  - `np.savez_compressed`: about 0.7×, but awkward outside Python.
  - OVF: the micromagnetics viewers' format, with no compression; add it when a viewer needs it.
  - float32 states: lossy.
  - A single `thermo.csv` with a stage column: blank cells, and rows to filter.
  - A default output folder with a timestamp: fills up with unnamed runs.
  - Refusing to run on a dirty tree: blocks everyday work.
- **Refinement (2026-10-18, implementation):**
  - A restart checks that the cell counts `n` match. The cell size may differ, so a state can be rescaled onto another cell size, as SP3's bisection does.
  - The hysteresis stage runs its own loop in `run.py` rather than calling `mm.hysteresis`, so each row is written and flushed as it happens. `mm.hysteresis` stays as the in-memory API.
  - Floats are written to the CSV with `repr`, so a value read back is exactly the value computed.
  - `git_state` counts untracked files as dirty, but `git diff HEAD` does not include them.
- **Correction (2026-10-18, found by a 6b test):** for a 1×1×1 grid, `np.ascontiguousarray(np.broadcast_to(...))` returns the read-only broadcast view itself, so the starting m (and the axis array) could not be written and the minimizer rejected it. Both are now built with `np.empty` and filled in.
- **Consequence:** h5py becomes a dependency. Its Linux wheels need glibc ≥ 2.28, so the optional cluster build needs a cap, like numpy. 6b's sweeps write one such folder per run.

### D29: Iteration-6a checks (2026-10-18)
- **Context:** the workflow layer must reproduce a solver result exactly, and its own behaviour (validation, restarts, dumps, failure) needs tests that run in milliseconds.
- **Decision:**
  - **Slow:** `examples/sp4.toml` run through `run()` into a temporary folder. Read `stage2_llg-dynamics.csv` and apply D20's criteria: S-state ⟨m⟩ within 0.01 of OOMMF, first ⟨mx⟩ = 0 crossing within 3%, RMS ≤ 0.05. It should print the direct test's numbers (138.6 ps, RMS 0.0051). The card's γ is 2.211e5/µ₀ = 1.759457895880903e11. The existing SP4 test stays, because it checks the solver without the card layer.
  - **Fast:**
    - `tests/test_card.py`: validation errors and parsing, including `8e5` as a number and the `tol` versus `until_torque` rule.
    - `tests/test_run.py`, on a tiny card:
      - the folder layout, the byte-identical card copy, the provenance keys and the diff on a dirty tree
      - refusal of a non-empty folder
      - restarts from `final` and from `stage<k>`, and rejection of a grid mismatch
      - snapshot counts and times or fields for `dump_every` and `dump`
      - one hysteresis row per field
      - the `--device` override through the CLI
      - failure handling: the `failed` state, the status, and the exit code
- **Consequence:** the roadmap row for 6a states this check.
- **Result (2026-10-18):**
  - `uv run pytest`: 67 passed, 30 skipped (CUDA), 5 deselected (slow). The 24 new workflow tests take 0.6 s.
  - `examples/sp4.toml` through `run()` reproduces the direct test exactly: S-state ⟨m⟩ = (0.9670, 0.1253, 0.0000) after 2.80 ns, crossing at 138.6 ps, RMS 0.0051, 1979 steps with 1 rejected, 25 s on the laptop CPU.
  - Through the command line, the results folder holds `card.toml`, `provenance.json`, `git.diff` (the tree was dirty), two thermo CSVs (10 kB and 194 kB) and a 281 kB `magnetic_states.h5`. A second run into the same folder exits with code 2 and writes nothing.
- **Mutation check:** each bug was planted alone, and each was caught.
  - The field carried over unless a stage set a non-zero one: caught by the Zeeman-energy check in the thermo test, which compares the energy against the recorded field.
  - `tol` checked against `until_torque` instead of `until_torque/100`: caught by the validation test.
  - The dump schedule starting at the first interval instead of t = 0: caught by the dump and thermo tests.
  - A restart reading a stage's first snapshot instead of its last: caught by the restart test.


### D30: Interim material regions: ordered boxes (2026-10-18)
- **Context:** 6b needs a core–shell cube. The general region design (arbitrary shapes, label maps, interphases, grain orientations) is open; see "Open design: material regions".
- **Decision:**
  - `[material]` fills the grid.
  - An ordered `[[region]]` list of boxes, `box = [[x0, y0, z0], [x1, y1, z1]]` in metres from the grid's lower corner, overwrites any of `Ms`, `A`, `K` and `axis`.
  - A cell belongs to a box if its centre is inside it, and later boxes win.
  - `Ms = 0` makes vacuum (D11).
  - Region values are written into the per-cell arrays the solver already takes (D9), so the solver doesn't change.
- **Alternatives:** the full label-map design (deferred by the user until it's thought through); named materials (part of the open design).
- **Consequence:** small, and enough for core–shell cubes. **Expected to be refactored** when the open region design is settled; cards using `[[region]]` may need updating then.

### D31: Hysteresis refinement and summary (2026-10-18)
- **Context:** coercivity needs the field resolved to about 10⁻⁴ Ms near the switch (SP2), which takes about 600 steps with a fixed fine step. For small films, H_c (where m·ĥ = 0) and the switching field (the jump) differ by only 0.1% (0.05707 vs 0.05714).
- **Decision:**
  - **Refinement:** `hysteresis` takes an optional `refine_to` (T) and `refine_jump` (default 0.1). A coarse step triggers refinement when m·ĥ changes sign (ĥ is the stage's direction) or when |Δ⟨m⟩| > `refine_jump`. The stage then restores the state from before the step and halves the interval, recursively, until the step is ≤ `refine_to`, and then carries on at the coarse step. Every field computed, coarse or refined, becomes a table row (and a snapshot if `dump = true`).
  - **Summary:** per hysteresis stage, in `provenance.json` under that stage, and on its progress line:
    - `Hc_T`: the first zero of m·ĥ, linearly interpolated between the two rows on either side of it
    - `switch_T`: the midpoint of the step with the largest |Δ⟨m⟩|
    - `remanence`: ⟨m⟩ at the row where the field is zero, or null if the ramp doesn't pass through zero
- **Alternatives:**
  - a fixed fine step: simple, but about 10× slower
  - a separate `summary.json`: one more file to keep in sync
  - refining on only one of the two triggers: misses either H_c or the switch when they differ
- **Consequence:** loops are about 45 field steps instead of about 600. The sequence of fields depends on the state (a refined interval is revisited), so a loop's fields are recorded in its table, not implied by the card.

### D32: Iteration-6b checks (2026-10-18)
- **Context:** the paper's Fig. 8a can only be read to about ±0.1 T, so the tight checks come from µMAG SP2 and a published single cube.
- **Decision:** all are slow tests parametrised by device.
  - **µMAG SP2:**
    - setup: permalloy-like Ms = 8e5 A/m and A = 1.3e-11 J/m (l_ex = 5.69 nm); cell = min(0.5 l_ex, t) with t = 0.1 d, and the thickness in whole cells
    - procedure: saturate along [1,1,1], then a hysteresis stage down through zero with `refine_to` about 1e-5 T
    - pass criteria:
      - d/l_ex = 1: H_c/Ms within 0.3% of the exact 0.05707
      - d/l_ex = 10: within 1% of OOMMF 1.1's 0.05454, remanence mx within 0.002
      - d/l_ex = 20 and 30 (CUDA only): within 1% of 0.04991 and 0.04594, remanence (mx, my) within (0.002, 0.005)
    - the 1% is about OOMMF 1.1 and MagTense's own 0.8% disagreement
  - **Pure 40 nm Nd₂Fe₁₄B cube:**
    - setup: the APL 2017 material; easy axis z; the field 1° off z ("one degree off the easy axis which is parallel to one edge"); cells of about 1.33 nm
    - pass: switching field within 3% of µ₀H₀ = 5.29 T. The 3% is about the spread across meshes for one cube in the paper's Fig. 8b.
  - **Fig. 8a core–shell cube:**
    - setup: the Fig. 8 materials; 1 nm cells, so the 3 nm shell is exactly 3 cells; the field 1° off the easy axis (an **assumption**: the group's stated convention for cubes, not stated for Fig. 8a); "grain size" = the outer edge, shell included (consistent with their 27-grain cube; the other reading moves H_c by about 0.01 T)
    - pass: H_c within ±0.15 T (their mesh spread of 0.13 T plus reading error) of 2.76 T at 25 nm (both devices) and 2.71 T at 50 nm (CUDA only)
  - **Recorded runs, not tests** (on a GPU, numbers in "Reference benchmarks"): the full SP2 curve (about 15 sizes) and the 100 nm core–shell cube (2.58 T).
  - **If a cube misses its tolerance:** the first response is a cell-size convergence study (1.5 → 1.0 → 0.75 nm at 25 nm), not a looser tolerance (as D20).
- **Alternatives:**
  - Fig. 8a alone: too loose to catch real bugs
  - the full SP2 curve as a test: tens of minutes on a GPU
  - a fixed cell count for SP2 (as MagTense did): cells of 1.5 l_ex at d/l_ex = 30
- **Consequence:** the roadmap check for 6b was reworded to match. The large cases depend on a GPU machine (D25).
- **Result (2026-10-18, laptop CPU):**

  | Check | Ours | Reference | Difference | Run time |
  |---|---|---|---|---|
  | SP2, d/l_ex = 1 (50×10×1) | H_c/Ms = 0.05704, remanence (1.0000, 0.0000) | 0.05707 exact; OOMMF 1.1 also gives 0.05704 | −0.05% | 2 min for both SP2 sizes |
  | SP2, d/l_ex = 10 (100×20×2) | H_c/Ms = 0.05446, remanence (0.9989, 0.0000) | OOMMF 1.1: 0.05454, (0.9989, 0.0000) | −0.15% | (included above) |
  | Pure 40 nm cube (30³ cells) | switching field 5.275 T | 5.29 T | −0.3% | 8.2 min (72 fields, 46,299 minimizer iterations) |
  | 25 nm core–shell cube (1 nm cells) | H_c = 2.721 T | Fig. 8a: 2.76 T | −0.04 T | 3.0 min (87 fields) |

  The cube results come from 72 and 87 refined fields instead of the 1,300–6,500 a fixed fine step would need.
- **Mutation check:** each bug was planted alone, and each was caught by the fast tests.
  - Regions painted with x and z swapped: caught by the region painting test.
  - Refinement triggered by sign changes only, not jumps: caught by Stoner–Wohlfarth at 60°, where H_c and the switch differ.
  - Halving without restoring the state from before the step: caught by both Stoner–Wohlfarth cases.
  - Remanence taken from the last row: caught by both.
- **Still to run on a GPU:** SP2 at d/l_ex = 20 and 30, the 50 nm cube, the full SP2 curve and the 100 nm cube.
