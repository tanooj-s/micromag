# DECISIONS

The decision log (D3, D38). The roadmap and status are in `NOTES.md`, and published comparisons in `BENCHMARKS.md`.

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
- **Refined (2026-10-20):** by D38: the log and the benchmarks moved out of NOTES.md.

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
    - Pass: |L/8.47 − 1| ≤ 0.5%. MagTense's 20³ value (8.4556) and Rave et al.'s partial energies are printed for comparison (see BENCHMARKS.md).
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
    - `[initial]`: exactly one of `m = [...]` (uniform, normalised on load) or `file` pointing to a `trajectory.h5`. A file takes an optional `state`: `"final"` (the default), or `"stage<k>"` for the last snapshot of stage k. The path is relative to the card. The file's SHA-256 is recorded, and its grid must match the card's.
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
    - **Dump** (like a LAMMPS dump file): `trajectory.h5`, written with h5py.
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
- **Renamed (user, 2026-10-18):** the dump file is `trajectory.h5` (it was `magnetic_states.h5`). A restart card points `[initial] file` at a run's `trajectory.h5`.
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
  - Through the command line, the results folder holds `card.toml`, `provenance.json`, `git.diff` (the tree was dirty), two thermo CSVs (10 kB and 194 kB) and a 281 kB `trajectory.h5`. A second run into the same folder exits with code 2 and writes nothing.
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
- **Superseded (2026-10-20):** by D33. The box form is removed, and the examples and tests are migrated.

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
  - **Recorded runs, not tests** (on a GPU, numbers in BENCHMARKS.md): the full SP2 curve (about 15 sizes) and the 100 nm core–shell cube (2.58 T).
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
- **Result (2026-10-20, GPU: Colab T4 runtime, Python 3.13.15):**

  | Check | Ours | Reference | Difference | Run time |
  |---|---|---|---|---|
  | SP2, d/l_ex = 1 | 0.05704, remanence (1.0000, 0.0000) | 0.05707 exact | −0.05% | 76 s |
  | SP2, d/l_ex = 10 | 0.05446, (0.9989, 0.0001) | OOMMF 1.1: 0.05454 | −0.15% | 14 s |
  | SP2, d/l_ex = 20 (200×40×4) | 0.04960, (0.9701, 0.0820) | 0.04991, (0.9701, 0.0819) | −0.6% | 121 s |
  | SP2, d/l_ex = 30 (300×60×6) | 0.04533, (0.9626, 0.0756) | 0.04594, (0.9627, 0.0756) | **−1.3%: fails the 1%** | 586 s |
  | Pure 40 nm cube | 5.275 T | 5.29 T | −0.3% | 72 s |
  | 25 nm core–shell cube | 2.721 T | 2.76 T | −0.04 T | 27 s |
  | 50 nm core–shell cube | 2.675 T | 2.71 T | −0.035 T | 292 s |

  Every number the CPU also computed is the same on the GPU.
- **Correction (2026-10-20, user's call):** the recorded runs (the full SP2 curve, the 100 nm cube) are optional benchmark data, not part of 6b's check: the tests already compare against every published value they would.
- **Correction (2026-10-20, measured):** SP2 at d/l_ex = 30 misses its 1% by −1.3%, while the remanence matches to 10⁻⁴. The gap grows with size (−0.15%, −0.6%, −1.3% at 10, 20, 30). The user accepted the miss: the test stays, marked xfail (non-strict, so a fix shows as XPASS). Not established why. Our cells are 0.5 l_ex, OOMMF 1.1's 0.60 l_ex at this size, and MagTense's 100×20×3 tiles give 0.04558 (we are −0.55% from it). Next step: the cell-size study in "Resume here" (nz = 3, 4, 5, 7; nz = 5 is OOMMF's cell).
- **Still to run on a GPU:** SP2 at d/l_ex = 20 and 30, the 50 nm cube, the full SP2 curve and the 100 nm cube.


### D33: Material regions on the MuMax3 model (supersedes D30) (2026-10-20)
- **Context:** grain structures need arbitrary shapes, many grains with their own easy axes, and boundary phases. The user chose MuMax3's region model over a from-scratch design.
- **Decision:**
  - The geometry the solver sees is a per-cell **region map** of `uint8` ids (0–255), as in MuMax3. It is expanded in Python into the per-cell Ms, A, K, axis arrays (D9), and into the pair table of D35.
  - Materials are named: `[material.<name>]` with Ms (≥ 0), A, K, axis. `vacuum` is built in (Ms = 0, D11) and reserved. A card needs at least one material; the flat `[material]` form is gone.
  - Region 0 is the background: every cell starts in it, with the first material listed.
  - `[[region]]` entries have `id` (1–255), `material`, `shape`, the shape's keys, optional `inverse` (paint the complement) and optional `axis` (overrides the material's). They are painted in card order and later paints win, as MuMax3's `DefRegion`. An id may be painted more than once, but always with the same material and axis.
  - Shapes, in metres from the grid's lower corner: `all`; `cuboid` with `corners = [[x0, y0, z0], [x1, y1, z1]]` (lower bounds in, upper out, as D30); `cylinder` along z with `center`, `radius`, `height`; `ellipsoid` with `center` and `semi_axes`. A cell is inside if its centre is.
  - Vacuum is a region with material `vacuum`; there is no separate geometry mask (MuMax3's `SetGeom`).
  - The map is saved in `trajectory.h5` as `regions` with the id → material names as an attribute. Provenance lists each `[[region]]` with its exact volume (null for `all` and inverted shapes) and its painted volume in cells, so staircasing and clipping are visible; a mismatch over 5% is warned on stderr.
- **Alternatives:**
  - MuMax3's combinators (add, sub, intersect, xor, rotate, translate): painting order plus `inverse` covers the cards we have. Translation is meaningless when every shape has a `center`.
  - Label files (h5 maps from elsewhere): deferred until a card needs one.
  - Partial-volume weighting of edge cells (MuMax3's `EdgeSmooth`): changes the solver; staircasing is visible in provenance instead.
  - Keeping the flat `[material]` as a shorthand: two forms to document and test.
- **Consequence:** the solver is unchanged by shapes. Every card has to name its materials, so the examples and tests were migrated.

### D34: 3D Voronoi grains with a grain-boundary phase (2026-10-20)
- **Context:** the paper's microstructures are polycrystals: grains with random or textured easy axes, coated by a few nm of boundary phase. MuMax3's `ext_makegrains` is 2D (Voronoi in xy, extruded along z) and reuses region ids, because 256 regions are shared by all grains.
- **Decision:**
  - One optional `[grains]` table: `count` seeds, `first_id` (ids first_id … first_id + count − 1 must fit in 1–255 and not clash with `[[region]]` ids), `material`, `seed` (the RNG seed), optional `within` (a region id; default 0).
  - Grains are applied after all `[[region]]` painting. Seeds are drawn uniformly over the grid box and kept if they fall in a cell of `within`, until `count` are kept. Each cell of `within` joins the grain of its nearest seed (3D Voronoi, by its centre).
  - Each grain has its own id, so the pair table can couple grains individually. That departs from MuMax3, which reuses ids.
  - Easy axes per grain: `axes = "material"` (the default), `"random"` (uniform on the sphere) or `{mean = [...], cone_deg = θ}` (uniform over the spherical cap of half-angle θ about mean).
  - Optional `boundary = {thickness, material, id}`: a grain-boundary phase, the paper's coat. A cell of grain i whose distance to the grain's Voronoi boundary is below thickness/2 joins the boundary region. The distance is exact: for a point inside a convex cell, it is the smallest distance to the cell's bounding planes, min over j ≠ i of (|x − s_j|² − |x − s_i|²) / (2 |s_j − s_i|). The thickness must be at least 2 cells (the largest cell edge); the card error points to `[[coupling]]` for thinner boundaries.
  - Plain numpy, chunked over cells (cells × seeds work, about 3·10⁸ for 10⁶ cells and 255 grains).
- **Alternatives:**
  - MuMax3's `grainsize`: a seed count from a grain size is the same thing said indirectly, and `count` makes the id range explicit.
  - A distance transform (scipy.ndimage) for the boundary: approximate on the grid, a new dependency, and resolution-dependent; the exact plane formula is none of these.
  - A coat on the outer surface of the grained region too (the paper has 3 nm there): not between grains; painting covers it.
  - Seeds on cell centres: resolution-dependent grains.
- **Consequence:** at most 255 grains in one card. The boundary region is an ordinary region, so `[[coupling]]` can scale grain ↔ boundary exchange too.

### D35: Exchange scaled per region pair (2026-10-20)
- **Context:** a boundary phase thinner than a cell cannot be painted. MuMax3 handles it with `ext_ScaleExchange` and `ext_InterExchange`, a 256 × 256 table that replaces the coupling across faces between two regions.
- **Decision:**
  - `[[coupling]]` entries: `regions = [a, b]` (ids, or `"grains"` for every grain id) and exactly one of `scale` (≥ 0, multiplies D11's harmonic mean) or `A` (≥ 0, the face A, J/m). `A` is converted to a scale in Python, since each id has one material. Pairs with a ≠ b only; later entries win.
  - The kernel takes an optional region map and a 256 × 256 scale table: the face weight is `harmonic_mean(A_i, A_j) · scale[r_i, r_j]` when r_i ≠ r_j. Without a table, the kernel is exactly the old one. Both backends.
  - `Problem(..., regions=None, coupling=None)` and `exchange_field(..., regions=None, coupling=None)` take them.
- **Alternatives:**
  - Per-face A arrays (3n doubles): general but 3× the memory of A, and no card needs a face that is not a region boundary.
  - Scaling only the A of the cells next to the boundary: changes those cells' coupling within their own region too.
- **Consequence:** one multiply per face across region boundaries. The table is symmetric, so the operator stays symmetric (H is still −∇E).

### D36: A switch for demag (2026-10-20)
- **Context:** the Stoner–Wohlfarth ensemble check (D37) needs grains that interact only through exchange, which is set to zero there.
- **Decision:** `[grid] demag = true` (the default) and `Problem(..., demag=True)`. With `false`, the demag field is zero everywhere: `effective_field` skips it, and `field("demag")` and the demag energy return 0.
- **Alternatives:** a test-only flag: cards could not express the same check. Not building the demag workspace when it is off: setup time only, not worth a second constructor path.
- **Consequence:** the demag tensor is still computed when demag is off.

### D37: Iteration-6c checks (2026-10-20)
- **Painting:** an ellipsoid's and a z-cylinder's cell volume tends to the exact volume (|V_cells/V − 1| ≤ dx/R at R/dx = 8, 16, 32); overlapping regions, `inverse`, axis overrides, vacuum and the background as expected cell by cell; the D30 test rewritten in the new syntax.
- **Grains:** every cell of `within` carries its nearest seed's id (against a brute-force oracle), and nothing outside changes; the same seed gives the same grains; `"random"` axes have ⟨|k̂·ĥ|⟩ = 1/2 and cone axes stay inside the cap.
- **Boundary phase:** two seeds mirrored across a plane give a slab of exactly thickness/dx cell layers; with several seeds, the boundary cells match a brute-force distance oracle: the distance to the nearest point of a 4× finer sampling that belongs to another grain, within the fine spacing.
- **Coupling kernel:** with all scales 1 the field is bitwise identical to no table; a 3-cell chain of regions 1-2-2 with scale s matches the hand-computed 2 s A (m_j − m_i) / (µ0 Ms d²); with a random symmetric table, H = −∇E / (µ0 Ms dV) by finite differences; CPU = CUDA (D26).
- **Ensemble (slow):** 255 grains on 40³ cells, `"random"` axes, all grain pairs at scale 0, no demag, saturated along z then a hysteresis stage down to −1.2 H_K. Each grain is then an exact Stoner–Wohlfarth particle: with no demag and no coupling, a uniform grain stays uniform. Pass:
  - remanence ⟨m·ĥ⟩ within 10⁻⁴ of Σ n_g |k̂_g·ĥ| / Σ n_g (n_g: the grain's cells), exact for these axes
  - H_c within 5·10⁻³ H_K of a 1D Stoner–Wohlfarth oracle (gradient flow of e(φ) = ½ sin²(φ − ψ_g) − h cos φ per grain, cell-weighted) for the same axes
  - both within 3σ of the isotropic Stoner–Wohlfarth (1948) values, M_r/M_s = 0.5 and H_c = 0.479 H_K, with σ from the cell-weighted spread of the drawn axes
- **Regression:** the 6b core–shell cube test (25 nm, 2.721 T) runs unchanged in the new syntax.
- **Alternatives:** only the statistical 0.5/0.479 check: about 12% wide at 255 grains, too loose to catch real bugs; the per-axis oracle is exact.
- **Consequence:** the headline physics check is analytic and runs on the CPU in the slow suite.
- **Result (2026-10-20, laptop CPU):**
  - fast suite: 105 passed, 35 skipped (CUDA), 20 deselected; all region, grain, boundary and coupling checks pass
  - ensemble (255 grains on 40³ cells, 58 fields, 32 s): remanence 0.47049 against the exact 0.47049 for these axes (difference 3·10⁻¹⁴); H_c = 0.46273 H_K against the oracle's 0.46273 (difference 7·10⁻⁸). Both within 3σ of the isotropic values: 0.5 ± 0.020 and 0.479 ± 0.014
  - regression: the migrated 25 nm core–shell cube gives 2.721 T, unchanged; SP2, the pure cube, SP3 and SP4 are unchanged too (full slow suite 15 min)
  - `examples/grains.toml` (8 grains, 32% boundary phase by volume, 20³ cells): H_c = 3.616 T in 72 s
  - CUDA: compiles in Docker (CUDA 12.1); the GPU run is the next Result
- **Result (2026-10-20, GPU: Colab T4 runtime):** `uv run pytest -q -rs` gives 139 passed, 1 skipped (the check that a CPU-only build refuses CUDA). Every coupling, demag-switch and CPU = CUDA test passes on the GPU.
- **Mutation check:** each bug was planted alone; every one was caught by the fast tests.
  - Boundary threshold at the full thickness instead of half: caught by the sampled-distance oracle.
  - Plane distance without its factor 2: caught by the mirrored pair and the oracle.
  - Pair table filled one way only: caught by the coupling-table test.
  - Cone axes uniform in θ instead of cos θ: caught by the axes test.
  - Pair table row stride 255 instead of 256 (C++): caught by the 3-cell chain and the gradient test.
  - Scale applied inside regions instead of between them (C++): caught only by the 3-cell chain; the oracle shares the code, and the operator stays symmetric.
  - Demag kept in `effective_field`, or in the energies, when switched off: caught by the demag-switch test.

### D38: Split NOTES.md into NOTES, DECISIONS and BENCHMARKS (refines D3) (2026-10-20)
- **Context:** NOTES.md reached 750 lines, 585 of them the decision log, and it is read at the start of every session. D3 planned to split it once the log passed about 20 entries.
- **Decision:** three files at the repo root.
  - `NOTES.md`: the roadmap, "Resume here", cluster notes, settled-design summaries and MagTense weaknesses. Read every session.
  - `DECISIONS.md`: the decision log D1…, in order. Result, Correction and mutation lines stay under their decision.
  - `BENCHMARKS.md`: published reference values and our comparisons with them.
  - CLAUDE.md names the right file for each kind of entry.
- **Alternatives:** one file per decision (37 files): loses the chronological read, and every new or corrected entry touches more files. A `docs/` folder: one more level for three files.
- **Consequence:** the session-start read is about 100 lines. Decisions are found with `grep -n "### D32" DECISIONS.md`.

### D39: Documentation, license and CI (2026-10-20)
- **Context:** the repo is public on GitHub but had no README, no license and no CI. Its how-to knowledge lived only in the decision log. The user asked for developer-appropriate docs with a README modelled on astral-sh/uv.
- **Decision:**
  - **README.md** in uv's order: badges, a hero figure, highlights, installation, quick start, links to the docs, features, validation, contributing, FAQ, acknowledgements, license.
  - **docs/**: Markdown pages that GitHub renders. `cards.md` (every card key), `results.md` (the folder), `verification.md` (the suite and validation table), `architecture.md` (the flow, principles, file map, Python API), `gpu.md` (CUDA builds, Colab, Slurm, the Docker check). They are written from the code, and CLAUDE.md requires them to change with it.
  - **Hero figure:** `docs/make_figures.py` runs `examples/grains.toml` and draws a mid-plane microstructure slice (grains shaded by easy-axis tilt, the boundary phase gray) beside the demagnetization curve and its H_c, in light and dark versions for a `<picture>` element. The colours come from the dataviz reference palette on GitHub's surfaces. matplotlib is in a separate `docs` dependency group, so tests and CI don't install it.
  - **License: GPL-3.0-or-later** (user's choice, after weighing it against MIT, BSD-3-Clause and MIT/Apache-2.0). It is copyleft, as MagTense and mumax3 are. All dependencies (nanobind, pocketfft, numpy, h5py) are BSD and compatible with it. Declared in `pyproject.toml` with PEP 639 `license` / `license-files`.
  - **CI:** `.github/workflows/tests.yml` runs the fast suite (`uv sync`, `uv run pytest -q`) on ubuntu-latest and macos-latest for every push and pull request. The GPU tests stay manual (Colab).
- **Alternatives:**
  - A MkDocs site on GitHub Pages, as uv has: a build and deploy to maintain before there are outside users.
  - One long README: mixes the pitch with the reference.
  - A validation table as the hero: no dependency, but less immediate; the table is in the README body instead.
  - No CI: the badge would be decoration.
  - Per-file GPL headers: suggested by the GPL, not required; the LICENSE file and the package metadata suffice.
- **Consequence:** docs can drift from the code, so the CLAUDE.md rule ties them together. The hero PNGs have to be regenerated when the grains example or its result changes. GPL-3.0 means derived distributions must stay open.


### D40: The thermal field is the next iteration; 6d becomes optional (2026-10-23)
- **Context:** 6a–6c are done. The candidates for iteration 7 were the string method, a thermal field and multi-GPU. The user requires that any treatment of thermal effects be at the same continuum scale the dynamics run at.
- **Decision:** iteration 7 is a thermal field: a Gaussian white-noise field added to H_eff in eq. 25, with a variance set by α, T, Ms, the cell volume and the time step (Brown 1963), on the existing grid. 6d is no longer required, since the project doesn't aim to replicate every result in the paper. The string method and multi-GPU move to an optional iteration 8.
- **Alternatives:**
  - String method first: reuses the minimizer, has deterministic tests, and has the paper's Fig. 7 cube as a benchmark. It gives energy barriers, not dynamics at temperature.
  - Multi-GPU: pays off only beyond about 10⁷ cells per V100. Slurm arrays already run independent jobs in parallel.
  - 6d first: not needed.
- **Consequence (open questions for the iteration-7 design):**
  - A fixed-step stochastic integrator (e.g. Heun) alongside the adaptive RK45 (D18), which can't take noise.
  - Random numbers reproducible on CPU and CUDA, so the CPU stays the oracle (e.g. counter-based Philox).
  - Statistical tests (the Boltzmann distribution, Brown's macrospin escape time) with explicit σ-based tolerances.
  - The time step must resolve the fastest precession (about 40 fs for the 40 nm cube at 30³ cells), so direct runs reach microseconds, not the ~1 s behind the 25 k_BT criterion. The design picks observables within reach.
  - Results depend on cell size, and noise added to the 300 K parameters Ms(T), A(T), K(T) counts some fluctuations twice. The design states how this is handled.

### D41: The thermal field model (2026-10-26)
- **Context:** iteration 7 (D40) adds thermal fluctuations at the continuum scale of the dynamics. The noise strength, where it lives and the random numbers have to be fixed before the integrator.
- **Decision:**
  - Brown's (1963) thermal field: Gaussian white noise added to H_eff in eq. 25, independent per cell and per component, with σ² = 2 α k_B T / (µ0 γ0 Ms V Δt) per step, γ0 = µ0 |γ|, V the cell volume. No 1+α² factor: the noise enters H in the Gilbert form, and our LL form (D18) carries it into both terms (MuMax3's convention).
  - Zero noise in vacuum cells (Ms = 0). σ is per cell, since Ms is (D33).
  - T is one scalar per `llg-dynamics` stage, in K.
  - Random numbers: a self-written Philox4x32-10 (Salmon et al. 2011) in one header shared by the CPU and CUDA backends. Key: (seed, stage index). Counter: (draw number, cell). Gaussians by Box–Muller. The integers are bitwise equal on both devices; the Gaussians agree to about 1 ulp (CUDA's log and cos against libm).
- **Alternatives:**
  - cuRAND: the same speed (the same Philox), but a CPU-only build can't reproduce it, so the CPU stops being the oracle for the noise.
  - A different generator per device: CPU and GPU could then only be compared statistically.
  - Per-cell temperature: for hot spots or gradients, which nothing on the roadmap needs.
  - The 1+α² variant of the variance: the equilibrium temperature comes out wrong by that factor (2× at α = 1).
  - A cell-size renormalisation of the parameters (e.g. Grinstein–Koch): research-level, with no settled method.
- **Consequence:**
  - Speed: the noise is one elementwise kernel per draw, small next to the two demag FFT evaluations of a Heun step (to be measured, D44).
  - Results depend on cell size: smaller cells carry shorter-wavelength fluctuations. Recorded in NOTES.md as an open research question (user, 2026-10-25), to be addressed during the code's development.
  - The 300 K parameters Ms(T), A(T), K(T) already contain the mean effect of atomic-scale fluctuations, and the noise adds them again: some fluctuations count twice. This is a warning to users in `docs/thermal.md` and `docs/cards.md` (user, 2026-10-25), not corrected in the code.

### D42: Stochastic integrators: Heun and adaptive RK23 with a Brownian bridge (refines D18) (2026-10-26)
- **Context:** the adaptive RK45 (D18) can't take noise: its error estimate assumes a smooth right-hand side. The noise is multiplicative (it enters as m × H_th), so the scheme also decides the calculus. Brown's physics is the Stratonovich reading.
- **Decision:**
  - One `LLG` class with a `method`: `rk45` (D18, unchanged, T = 0 only), `heun` (the default at T > 0) and `rk23`. Every method also runs at T = 0. The methods share the field evaluation, the noise buffer and the driver loop. One new backend function, `thermal_field`, fills H_th per cell; it is added to H through `lincomb`.
  - `heun`: fixed step dt. One noise sample per step, held over the predictor and the corrector; m renormalised after the step. Converges to the Stratonovich solution (weak order 1; 2nd order at T = 0).
  - `rk23`: adaptive Bogacki–Shampine 3(2), the noise frozen over each step as the field √(2D) ΔW / h, D = α k_B T / (µ0 γ0 Ms V). The D18 error norm (largest per-cell error vector), `h_new = h·0.9·(tol/err)^(1/3)` clamped to 0.2–5×.
  - Rejected `rk23` steps keep their noise through a Brownian bridge (rejection sampling with memory, Rackauckas & Nie 2017). Retrying h → h′ < h splits ΔW exactly: ΔW′ = (h′/h) ΔW + √(h′(h − h′)/h) η, and the remainder (ΔW − ΔW′ over h − h′) goes on a stack. Later steps use the stack before any fresh draw: a step never spans more than the top entry, and one shorter than it splits that entry by the same bridge. The stack is a `std::vector<Buffer>` with no cap; its deepest level is exposed with `steps` and `rejected`. Philox's counter is the draw number, so a bridge draw is just another draw.
- **Alternatives:**
  - Euler–Maruyama with the Itô drift: dropped (user, 2026-10-26). The Itô correction ½ Σ_jk G_kj ∂_k G_ij for G(m)ξ = −γ′σ√Δt [m × ξ + α m × (m × ξ)] is −γ′² σ² Δt (1+α²) m: it points purely along m and only stops |m| from growing (checked numerically: the tangential part is ~10⁻¹¹ of the radial part at α = 0, 0.3 and 1). The renormalisation after each step supplies it to first order, so the explicit term is redundant and untestable, and projected Euler offers nothing Heun doesn't.
  - Reusing a rejected step's noise rescaled to the shorter step (Leliaert et al. 2017, MuMax3): simpler, but approximate; the user chose the exact bridge (2026-10-25).
  - A step that spans several stack entries (RSwM3): fewer steps, more bookkeeping.
  - An integrator class per method behind an interface: a layer nothing else plugs into.
  - Implicit midpoint: preserves |m|, but a nonlinear solve per step.
- **Consequence:**
  - `rk45` with T > 0 is refused.
  - A Heun step costs 2 field evaluations, an `rk23` step 3 (with FSAL), an `rk45` step 6.
  - On the GPU, a floating-point difference in `err` can flip an `rk23` accept/reject decision and so change the draw sequence: CPU = CUDA for `rk23` holds only for trajectories short enough to avoid a near-tie (D44).

### D43: Thermal card keys, outputs and docs (refines D27, D28) (2026-10-26)
- **Context:** cards must express thermal runs (D27), and the results folder must record how a run was made (D28).
- **Decision:**
  - New keys on `llg-dynamics`: `temperature` (K, default 0); `method` (`rk45` at T = 0, `heun` at T > 0, when not given); `dt` (required for `heun`); `seed` (required at T > 0). `tol` and `max_step` apply to `rk45` and `rk23`.
  - Card errors: `rk45` with T > 0; `heun` without `dt`; `dt` with an adaptive method; T > 0 without `seed`; `until_torque` with T > 0 (the torque never settles under noise).
  - dt is never chosen automatically. `docs/cards.md` gives a rule of thumb (a small fraction of the precession period at the largest field, and below the exchange stability limit).
  - Thermal settings exist only on `llg-dynamics`; `minimize` and `hysteresis` stay at T = 0.
  - Provenance records, per stage: temperature, method, dt or tol, seed, steps, rejected and the deepest bridge stack. No new thermo CSV column: T is constant in a stage, and ⟨m⟩ and the energies become noisy time series at T > 0.
  - Docs: a "Thermal runs" section in `docs/cards.md` (keys, the dt rule of thumb, the caveats in short), and a new `docs/thermal.md` with the physics in plain terms and the caveats in full (cell size, double counting, no per-cell T); `docs/verification.md` and `docs/architecture.md` updated; a README pointer.
- **Alternatives:** an automatic dt (hides a key numerical choice, and the largest field changes during reversal); an optional seed with a default; a temperature column in the CSV (repeats a constant).
- **Consequence:** every existing card runs unchanged (T = 0, `rk45`); SP4 must still give 138.6 ps.

### D44: Iteration-7 checks (2026-10-26)
- **Context:** a thermal code can be wrong without looking wrong: it runs and the needles wiggle, at the wrong temperature or the wrong speed. Random runs never land exactly on a formula, so the checks compare averages against their error bars.
- **Decision:**
  - **Statistics:** every statistical test runs with a fixed seed, so it is deterministic. Its tolerance is 4σ (a fresh seed fails about 1 in 16,000), with σ from block averages over correlated samples. dt is small enough that the integrator's bias is ≤ 1σ, checked by halving dt once.
  - **(a) Noise:** Philox against the Random123 known-answer vectors; the field has mean 0, variance σ² (D41) and no correlation between cells, components or draws; zero in vacuum.
  - **Order (T = 0):** against the D20 analytic macrospin with fixed steps, the error ratio per halving of h is 3.4–4.6 for `heun` (2nd order) and 6.8–9.2 for `rk23` (3rd).
  - **(b) Boltzmann ensemble:** A = 0 and demag off (D36) make every cell an independent uniaxial macrospin, so one grid is an ensemble. The cos θ histogram matches P ∝ exp(−KV sin²θ / k_BT) bin by bin, and at K = 0 in a field, ⟨m·ĥ⟩ matches the Langevin function coth ξ − 1/ξ, ξ = µ0 Ms V H / k_BT.
  - **(c) α-independence:** the (b) averages at α = 0.1 and α = 1 agree within 4σ combined.
  - **Bridge:** with a tiny tol forcing many rejections, Var(Σ ΔW) over [0, t] is exactly t per component; a fresh redraw on rejection fails this.
  - **(d) Escape time (slow):** the relaxation rate of ⟨m_z⟩ of the ensemble started at +z, at KV/k_BT ≈ 3–6, against the smallest nonzero eigenvalue of the Fokker–Planck operator in a Legendre basis (exact at any barrier height).
  - **CPU = CUDA:** the noise field to 1e-14 relative; a 100-step `heun` trajectory to 1e-10; `rk23` over a run short enough to avoid an accept/reject near-tie (D42).
  - (a)–(c), the order tests and the bridge run for `heun` and `rk23` in the default suite; (d) for both under `-m slow`. The existing suite runs unchanged.
- **Alternatives:** 3σ (a fresh seed fails about 0.3% of the time per check); Brown's high-barrier asymptote for (d) (wrong by ~1/σ at barriers we can reach); all statistical tests slow (keeps the default run shorter, but hides the main checks).
- **Consequence:** the default CPU run grows by about 1–2 minutes (estimate, to be measured).
