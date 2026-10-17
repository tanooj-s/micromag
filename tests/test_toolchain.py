import numpy as np
import pytest

import micromag


def test_scale_cpu():
    x = np.arange(5.0)
    micromag.scale(x, 2.0)
    assert np.array_equal(x, 2.0 * np.arange(5.0))


@pytest.mark.skipif(not micromag.has_cuda, reason="CPU-only build")
def test_scale_cuda_matches_cpu():
    # odd length exercises the kernel's tail guard
    x = np.random.default_rng(0).random(1_000_003)
    y = x.copy()
    micromag.scale(x, 3.0)
    micromag.scale(y, 3.0, cuda=True)
    assert np.array_equal(x, y)  # one IEEE multiply: bitwise equal on CPU and GPU
