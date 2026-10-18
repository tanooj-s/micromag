#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

#include "demag.hpp"
#include "fields.hpp"
#include "llg.hpp"
#include "problem.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Vec = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using In = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using Out = nb::ndarray<double, nb::c_contig, nb::device::cpu>;

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

// Throws ValueError unless a is (3, nz, ny, nx) for a vector field or (nz, ny, nx) for a scalar one.
template <class T>
static void check_shape(const T& a, const Grid& g, bool vector, const char* name) {
    size_t want[4] = {3, g.nz, g.ny, g.nx};
    size_t* w = vector ? want : want + 1;
    size_t nd = vector ? 4 : 3;
    bool ok = a.ndim() == nd;
    for (size_t i = 0; ok && i < nd; ++i) ok = a.shape(i) == w[i];
    if (!ok) throw std::invalid_argument(std::string(name) + " must have shape " + (vector ? "(3, nz, ny, nx)" : "(nz, ny, nx)"));
}

NB_MODULE(_core, m) {
#ifdef MICROMAG_CUDA
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
    m.attr("mu0") = mu0;
    m.def("scale", &scale, "x"_a, "a"_a, "cuda"_a = false);

    nb::class_<Grid>(m, "Grid")
        .def(nb::init<size_t, size_t, size_t, double, double, double>(), "nx"_a, "ny"_a, "nz"_a, "dx"_a, "dy"_a, "dz"_a)
        .def_ro("nx", &Grid::nx).def_ro("ny", &Grid::ny).def_ro("nz", &Grid::nz)
        .def_ro("dx", &Grid::dx).def_ro("dy", &Grid::dy).def_ro("dz", &Grid::dz);

    // noconvert: a dtype or layout mismatch must raise, not silently copy (a copied `out` would lose the result).
    m.def("exchange_field", [](const Grid& g, In mag, In Ms, In A, Out out) {
        check_shape(mag, g, true, "m"); check_shape(Ms, g, false, "Ms"); check_shape(A, g, false, "A"); check_shape(out, g, true, "out");
        exchange_field(g, mag.data(), Ms.data(), A.data(), out.data());
    }, "grid"_a, "m"_a.noconvert(), "Ms"_a.noconvert(), "A"_a.noconvert(), "out"_a.noconvert(), "Exchange field (eq. 8), A/m.");

    m.def("anisotropy_field", [](const Grid& g, In mag, In Ms, In K, In k, Out out) {
        check_shape(mag, g, true, "m"); check_shape(Ms, g, false, "Ms"); check_shape(K, g, false, "K"); check_shape(k, g, true, "k"); check_shape(out, g, true, "out");
        anisotropy_field(g, mag.data(), Ms.data(), K.data(), k.data(), out.data());
    }, "grid"_a, "m"_a.noconvert(), "Ms"_a.noconvert(), "K"_a.noconvert(), "k"_a.noconvert(), "out"_a.noconvert(), "Uniaxial anisotropy field (eq. 9), A/m.");

    m.def("zeeman_field", [](const Grid& g, In Ms, std::array<double, 3> H_ext, Out out) {
        check_shape(Ms, g, false, "Ms"); check_shape(out, g, true, "out");
        zeeman_field(g, Ms.data(), H_ext.data(), out.data());
    }, "grid"_a, "Ms"_a.noconvert(), "H_ext"_a, "out"_a.noconvert(), "Zeeman field (eq. 11), A/m: H_ext in magnetic cells.");

    m.def("demag_tensor", [](const Grid& g) {
        auto* N = new std::vector<double>(demag_tensor(g));
        nb::capsule owner(N, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
        return nb::ndarray<nb::numpy, double>(N->data(), {6, padded(g.nz), padded(g.ny), padded(g.nx)}, owner);
    }, "grid"_a, "Demag tensor (6, Pz, Py, Px): xx, yy, zz, xy, xz, yz in FFT wraparound order (negative index = negative offset).");

    nb::class_<Demag>(m, "Demag")
        .def(nb::init<const Grid&>(), "grid"_a)
        .def("field", [](Demag& d, In mag, In Ms, Out out) {
            const Grid& g = d.grid();
            check_shape(mag, g, true, "m"); check_shape(Ms, g, false, "Ms"); check_shape(out, g, true, "out");
            d.field(mag.data(), Ms.data(), out.data());
        }, "m"_a.noconvert(), "Ms"_a.noconvert(), "out"_a.noconvert(), "Demagnetizing field (eq. 21), A/m.");

    nb::class_<Problem>(m, "Problem")
        .def("__init__", [](Problem* p, const Grid& g, In Ms, In A, In K, In k, std::array<double, 3> H_ext) {
            check_shape(Ms, g, false, "Ms"); check_shape(A, g, false, "A"); check_shape(K, g, false, "K"); check_shape(k, g, true, "k");
            new (p) Problem(g, Ms.data(), A.data(), K.data(), k.data(), H_ext);  // copies the materials (D19)
        }, "grid"_a, "Ms"_a.noconvert(), "A"_a.noconvert(), "K"_a.noconvert(), "k"_a.noconvert(), "H_ext"_a)
        .def_rw("H_ext", &Problem::H_ext, "External field (A/m).")
        .def("effective_field", [](Problem& p, In mag, Out out) {
            check_shape(mag, p.grid(), true, "m"); check_shape(out, p.grid(), true, "out");
            p.effective_field(mag.data(), out.data());
        }, "m"_a.noconvert(), "out"_a.noconvert(), "Sum of exchange, anisotropy, Zeeman and demag fields, A/m.");

    nb::class_<LLG>(m, "LLG")
        .def(nb::init<Problem&, double, double, double, double>(), "problem"_a, "alpha"_a, "gamma"_a = 1.76086e11,
             "tol"_a = 1e-6, "max_step"_a = std::numeric_limits<double>::infinity(), nb::keep_alive<1, 2>())
        .def("run_until", [](LLG& l, Out mag, double t) {
            check_shape(mag, l.grid(), true, "m");
            l.run_until(mag.data(), t);
        }, "m"_a.noconvert(), "t"_a, "Advance m in place from llg.t to t (s).")
        .def_ro("t", &LLG::t).def_ro("steps", &LLG::steps).def_ro("rejected", &LLG::rejected);
}
