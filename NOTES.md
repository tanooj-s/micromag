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
| 5 | GPU backend: CUDA kernels + cuFFT; the backend interface is introduced here | Every test passes with `device="cuda"` and the direct CPU-vs-GPU comparisons hold their D26 tolerances, run on at least one CUDA GPU of sm_70 or newer (which GPU is recorded). Nsight roofline numbers are optional (D25) | **in progress**: design settled 2026-10-18 (D24–D26) |
| 6 | Cluster science: Slurm job-array sweeps, output with provenance (git hash and parameters), scaling study. **Requirement (user, 2026-10-17):** a simple local workflow, parameters in → results out, that runs one simulation on the laptop CPU from an input file or CLI and writes its outputs, without hand-editing a Python script. The cluster sweeps should use the same entry point. | Coercivity vs cube size compared with paper Fig. 8a; one local run from an input file reproduces SP4 | todo |
| 7 | Stretch, pick one: Voronoi grains / string method (NEB) / multi-GPU / thermal field | — | — |

## Resume here
State on 2026-10-17: iterations 1–4 are done on the CPU. Everything is committed and pushed to `origin main`. `uv run pytest` gives 37 passed, 1 skipped (CUDA) and 2 deselected (slow).
- **2026-10-18:** the user chose to write the CUDA code now and treat the cluster as one optional GPU target among many (D25). The iteration-5 design is D24–D26.
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

### D25: CUDA portability: any sm_70+ GPU, float64, cluster optional (refines D4, D6) (2026-10-18)
- **Context:** the user expects to run on machines and GPUs beyond the V100 cluster later, and chose not to block iteration 5 on the cluster build. The laptop has no NVIDIA GPU.
- **Decision:**
  - `CMAKE_CUDA_ARCHITECTURES` stays 70 by default. CMake's `70` embeds both V100 machine code and compute_70 PTX, which the driver JIT-compiles for any newer GPU (T4, A100, RTX, H100) on first load.
  - Precision stays float64 (D9).
  - The demag tensor and its spectrum are still built on the CPU with the D14/D15 code and uploaded once. Only the per-call convolution runs on the GPU, in cuFFT (batched D2Z/Z2D).
  - Device-wide sums and maxima use CUB, which ships with the CUDA toolkit.
  - **Local check:** a CUDA build inside an `nvidia/cuda` Docker image. nvcc compiles without a GPU, so compile errors surface on the laptop.
  - **GPU check:** the tests run unchanged on any NVIDIA GPU of sm_70 or newer with CUDA 12. The cluster is one optional target, and so are its numpy fix and the Nsight roofline numbers.
- **Alternatives:**
  - `all-major` architectures: no first-load JIT delay, but several times the compile time.
  - `native`: fails on GPU-less build hosts (login nodes, Docker).
  - A float32 or mixed-precision path: no second precision is needed yet (D9).
  - Fixing the cluster first: the earlier recommendation, declined by the user.
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
