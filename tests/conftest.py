import pytest

import micromag as mm


# D26: every physics test runs on each device; the CUDA case is skipped on a build without CUDA.
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not mm.has_cuda, reason="CPU-only build"))])
def device(request):
    return request.param
