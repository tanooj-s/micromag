# NOTES

## Roadmap
Each iteration is done when its check passes and its decisions are logged below.

| # | Iteration | Check | Status |
|---|-----------|-------|--------|
| 0 | Toolchain: CMake + bindings + pytest + one CUDA kernel, run on a laptop and on a cluster GPU node through Slurm | Kernel test passes in both places | laptop ✅ · cluster blocked (see below) |
| 1 | Grid, per-cell materials, exchange / anisotropy / Zeeman (eqs. 8, 9, 11), CPU | Energies of uniform states; H matches the finite-difference gradient of E; analytic 1D Bloch wall (width π√(A/K)) is in equilibrium (torque → 0) with σ → 4√(AK), both errors falling ~4× per halving of dx (D13) | ✅ CPU (`tests/test_fields.py`) |
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
