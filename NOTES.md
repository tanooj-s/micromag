# NOTES

## Roadmap
Each iteration is done when its check passes and its decisions are logged below.

| # | Iteration | Check | Status |
|---|-----------|-------|--------|
| 0 | Toolchain: CMake + bindings + pytest + one CUDA kernel, run on a laptop and on a cluster GPU node through Slurm | Kernel test passes in both places | laptop ✅ · cluster blocked (see below) |
| 1 | Grid, per-cell materials, exchange / anisotropy / Zeeman (eqs. 8, 9, 11), CPU | Energies of uniform states; 1D Bloch wall width = π√(A/K) | todo |
| 2 | Demag: Newell tensor, direct O(N²) oracle, zero-padded FFT (eqs. 17–21) | Cube: trace N = 1, Nxx = 1/3; thin film: Nzz → 1; FFT and direct agree to ~1e-12 | todo |
| 3 | LLG (eq. 25) with a self-written adaptive Dormand–Prince RK45 | Macrospin matches the analytic solution; µMAG SP4 within a few % | todo |
| 4 | Minimizer: projected steepest descent with Barzilai–Borwein steps (eqs. 22–23), plus a hysteresis driver | Stoner–Wohlfarth astroid; µMAG SP2 or SP3 | todo |
| 5 | GPU backend: CUDA kernels + cuFFT; the backend interface is introduced here | GPU matches CPU on every test; Nsight roofline numbers recorded | todo |
| 6 | Cluster science: Slurm job-array sweeps, output with provenance (git hash and parameters), scaling study | Coercivity vs cube size compared with paper Fig. 8a | todo |
| 7 | Stretch, pick one: Voronoi grains / string method (NEB) / multi-GPU / thermal field | — | — |

## Cluster environment
- **GPU nodes:** 8× Tesla V100-SXM2-32GB per node (sm_70). The driver is 535.104 and supports CUDA ≤ 12.2.
- **Modules:**
  - `cmake/3.17.3`, which is too old. scikit-build-core fetches cmake ≥ 3.18 from PyPI.
  - `gcc/8.4.0/1` (default) and `gcc/9.3.0/1`.
  - `cuda/10.1` through `cuda/12.1`.
  - `pgi/19.10`, `matlab`, `git/2.25.1`.
- **No Python module.** A shared conda env exists, but it belongs to another project, so we don't use it.
- **Slurm partition and account:** TBD. These are only needed if `sbatch` rejects the job without them. Find them with `sinfo -s` and `sacctmgr show assoc user=$USER format=account,partition`.

### Open issue: cluster build (deferred 2026-10-17)
- **Symptom:** `uv sync` builds numpy 2.5.4 from source, and the build fails because the system `cc` is gcc 4.8.5 (RHEL 7).
- **Likely cause:** RHEL 7 has glibc 2.17, while numpy 2.5 only ships wheels for glibc ≥ 2.28 (manylinux_2_28). So there is no wheel to install, and uv falls back to compiling from source.
- **Fixes to try:**
  - Pin numpy to the last version with manylinux2014 (glibc 2.17) wheels for cp313.
  - Or export `CC=gcc` after the gcc 8.4 module load, so the source build can at least compile.
- Already fixed: home-directory quota (uv cache moved to scratch) and the empty `CUDA_ARCHITECTURES`.

## MagTense weaknesses we want to avoid
- One 3,500-line Fortran module holds global `gb_problem` and `gb_solution` state, so solves are not re-entrant.
- The default demag is a dense O(N²) matrix; the FFT path was added on later.
- Damping is the dimensional LL `alpha0` rather than Gilbert's dimensionless α.
- Tests are example scripts with loose tolerances (SP4 allows 100% error), and there are no unit tests per field term.
- The Python and MATLAB front ends duplicate each other, and the build needs Intel ifx/MKL plus an `#if` flag matrix.

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
