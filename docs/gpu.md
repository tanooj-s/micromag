# Running on a GPU

The CUDA backend runs the same float64 arithmetic as the CPU backend, with cuFFT for demag and CUB for reductions. It is tested against the CPU on every kernel. Any NVIDIA GPU of compute capability 7.0 (Volta) or newer works with CUDA 12; CUDA 13 needs 7.5 (Turing) or newer. Data-centre cards (V100, A100, H100) are much faster than consumer cards and the T4, whose FP64 runs at 1/32 of FP32.

## Build with CUDA

The CUDA build is opt-in (D6). You need the CUDA toolkit (12.x or 13.x) with `nvcc` on `PATH`, then:

```console
$ export SKBUILD_CMAKE_DEFINE="MICROMAG_CUDA=ON"
$ uv sync
$ uv run python -c "import micromag; print(micromag.has_cuda)"
True
```

Keep `SKBUILD_CMAKE_DEFINE` set in the shell you use afterwards. `uv run` rebuilds when it changes, so a later `uv run` without it silently rebuilds a CPU-only extension.

The target architecture defaults to `70` for CUDA < 13 and to `75` for CUDA ≥ 13, which dropped Volta. The embedded PTX runs on any newer GPU. Override it with `CUDAARCHS=80`, or with `SKBUILD_CMAKE_DEFINE="MICROMAG_CUDA=ON;CMAKE_CUDA_ARCHITECTURES=80"`.

Then choose the device per run:

```console
$ uv run micromag run examples/sp2.toml -o runs/sp2 --device cuda
$ uv run pytest                   # every device-parametrised test also runs on cuda
```

## Google Colab

Pick **Runtime → Change runtime type → T4 GPU**, then:

```python
!git clone https://github.com/tanooj-s/micromag.git
%cd /content/micromag
!curl -LsSf https://astral.sh/uv/install.sh | sh
```

```python
import os
os.environ["PATH"] = "/root/.local/bin:/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
os.environ["SKBUILD_CMAKE_DEFINE"] = "MICROMAG_CUDA=ON"
```

```python
!uv sync
!uv run python -c "import micromag; print('has_cuda =', micromag.has_cuda)"
!uv run pytest -q -rs                                   # fast suite on both devices
!uv run pytest -m slow -k "coercivity and cuda" -s -v   # the GPU-sized benchmarks, about 20 min on a T4
```

Rerun the environment cell after every runtime restart. Write long runs to Google Drive (`drive.mount("/content/drive")`) so a disconnect doesn't lose them.

## A Slurm cluster

The scripts in [`hpc/`](../hpc) target nodes with 8× V100. Build on the login node, which has internet access, with the same modules the job will load:

```console
$ module load gcc/8.4.0/1 cuda/12.1
$ export SKBUILD_CMAKE_DEFINE="MICROMAG_CUDA=ON"
$ uv sync
$ sbatch hpc/test.sbatch          # log: micromag-test-<jobid>.out
```

On old systems (glibc 2.17, e.g. RHEL 7), numpy and h5py have no current wheels and have to be capped. See the cluster notes in [NOTES.md](../NOTES.md).

## Checking that CUDA code compiles, without a GPU

From the repository root, with Docker:

```console
$ docker run --rm -v "$PWD":/src -w /src -e UV_PROJECT_ENVIRONMENT=/tmp/venv -e SKBUILD_CMAKE_DEFINE=MICROMAG_CUDA=ON \
    nvidia/cuda:12.1.1-devel-ubuntu22.04 bash -c \
    'apt-get update -qq && apt-get install -y -qq curl >/dev/null && curl -LsSf https://astral.sh/uv/install.sh | sh && . $HOME/.local/bin/env && uv sync'
```

This builds into `build/<linux tag>`, separate from the host build.
