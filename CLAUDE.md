# micromag

This is a from-scratch finite-difference micromagnetic solver, modelled on the micromagnetic part of MagTense (github.com/cmt-dtu-energy/MagTense). It is first a teaching exercise in architecture for scientific software, and second a physics code.

- **Physics spec:** Fischbacher et al. 2018, J. Phys. D 51 193002, https://doi.org/10.1088/1361-6463/aab7d1 (open access). A local copy, kept out of git, is at `refs/Fischbacher_2018_J._Phys._D__Appl._Phys._51_193002.pdf`.
- **Roadmap, status and decision log:** `NOTES.md`. Read it at the start of each session.

## Working with the user
- This is a teaching project, and the user is the learner, so explain the *why* in plain language.
- **Before each design decision:** say what the decision is, give the realistic options, and recommend one. Let the user choose before you implement.
- **After each iteration:** give a short walkthrough of what was built and how the pieces connect.
- **While the user is running commands** (e.g. on the cluster): keep replies short, with just the fix and the next command.
- **Cluster environment problems:** don't block the local work on them. Log them as open issues in NOTES.md and keep going.

## Stack
- C++ core, built with CMake. CUDA and cuFFT for the GPU path.
- Python bindings and pytest as the single test runner.
- Target machine: a Slurm cluster with nodes of 8× V100-SXM2-32GB (sm_70). The driver supports CUDA ≤ 12.2. Job scripts live in `hpc/`.

## Commands
The build uses uv with scikit-build-core and nanobind. `uv run` rebuilds the C++ automatically when `src/`, `CMakeLists.txt` or `pyproject.toml` change.
- **Test (laptop, CPU):** `uv run pytest`. On this Mac, `/usr/local/bin/cmake` is an x86_64 binary, so export `ARCHFLAGS="-arch arm64"`, or the extension builds for the wrong architecture.
- **Cluster storage:** the home directory quota is small, so the uv cache and Python installs go on scratch. These lines are in `~/.bashrc` on the cluster:
  ```
  export UV_CACHE_DIR=/gpfs/u/scratch/TMGN/TMGNtjsh/.cache/uv
  export UV_PYTHON_INSTALL_DIR=/gpfs/u/scratch/TMGN/TMGNtjsh/.local/uv-python
  export UV_LINK_MODE=copy
  ```
- **Build (cluster, GPU):** do this on the login node, where there is internet access.
  ```
  module load gcc/8.4.0/1 cuda/12.1
  export SKBUILD_CMAKE_DEFINE="MICROMAG_CUDA=ON"
  uv sync
  ```
- **Test (cluster, GPU):** `sbatch hpc/test.sbatch`. The log goes to `micromag-test-<jobid>.out`.

## Conventions
- SI units throughout: A/m, J/m³, J/m, T.
- Damping is the **dimensionless Gilbert α**, never the dimensional LL `alpha0` that MagTense uses.
- Name the paper equation a function implements in a comment, e.g. `// eq. 21`.
- Material parameters (Ms, A, K, k̂) are stored per cell, not as global scalars.
- Solver state lives in objects that are passed in. No module-level or global state.

## Hard rules
- An iteration is done only when its physics check, listed in NOTES.md, passes, and you have shown the test output.
- The CPU implementation is the oracle. Every GPU kernel is tested against it.
- When you make a non-obvious decision, add an entry to the decision log in NOTES.md (context / decision / alternatives / consequence) in the same change.
- Do not add an abstraction until there is a second implementation that needs it.
- Every non-trivial numerical routine needs a test against an analytic result or an independent oracle.
