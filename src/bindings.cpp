#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

using Vec = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

#ifdef MICROMAG_CUDA
void scale_cuda(double* x, size_t n, double a);  // scale.cu
#endif

// Iteration-0 toolchain smoke test: x *= a in place, on the CPU or the GPU.
static void scale(Vec x, double a, bool cuda) {
    double* p = x.data();
    size_t n = x.shape(0);
    if (!cuda) {
        for (size_t i = 0; i < n; ++i) p[i] *= a;
        return;
    }
#ifdef MICROMAG_CUDA
    scale_cuda(p, n, a);
#else
    throw std::runtime_error("micromag was built without CUDA (set MICROMAG_CUDA=ON)");
#endif
}

NB_MODULE(_core, m) {
#ifdef MICROMAG_CUDA
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
    m.def("scale", &scale, "x"_a, "a"_a, "cuda"_a = false);
}
