#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "demag.hpp"
#include "fields.hpp"
#include "llg.hpp"
#include "minimizer.hpp"
#include "problem.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Vec = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using In = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using Out = nb::ndarray<double, nb::c_contig, nb::device::cpu>;
using Labels = std::optional<nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu>>;
using Table = std::optional<nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>>;

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

// D35: the region map and pair table come together; returns their pointers, or nulls.
static std::pair<const uint8_t*, const double*> check_regions(const Labels& regions, const Table& coupling, const Grid& g) {
    if (regions.has_value() != coupling.has_value())
        throw std::invalid_argument("give both regions and coupling, or neither (D35)");
    if (!regions) return {nullptr, nullptr};
    check_shape(*regions, g, false, "regions");
    if (coupling->shape(0) != n_regions || coupling->shape(1) != n_regions)
        throw std::invalid_argument("coupling must have shape (256, 256)");
    return {regions->data(), coupling->data()};
}

static Device parse_device(const std::string& name) {
    if (name == "cpu") return Device::cpu;
    if (name == "cuda") return Device::cuda;
    throw std::invalid_argument("device must be \"cpu\" or \"cuda\", not \"" + name + "\"");
}

static Term parse_term(const std::string& name) {
    if (name == "exchange") return Term::exchange;
    if (name == "anisotropy") return Term::anisotropy;
    if (name == "zeeman") return Term::zeeman;
    if (name == "demag") return Term::demag;
    throw std::invalid_argument("term must be exchange, anisotropy, zeeman or demag, not \"" + name + "\"");
}

// Runs f(m, H) on copies of mag and out in the problem's memory space, then copies H back into out (D24).
template <class F>
static void on_device(Problem& p, In mag, Out out, F f) {
    const size_t N = 3 * p.grid().n();
    Buffer m(p.backend(), N), H(p.backend(), N);
    p.backend().to_device(m.data(), mag.data(), N);
    f(m.data(), H.data());
    p.backend().to_host(out.data(), H.data(), N);
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
    m.def("exchange_field", [](const Grid& g, In mag, In Ms, In A, Out out, Labels regions, Table coupling) {
        check_shape(mag, g, true, "m"); check_shape(Ms, g, false, "Ms"); check_shape(A, g, false, "A"); check_shape(out, g, true, "out");
        const auto [r, c] = check_regions(regions, coupling, g);
        exchange_field(g, mag.data(), Ms.data(), A.data(), out.data(), r, c);
    }, "grid"_a, "m"_a.noconvert(), "Ms"_a.noconvert(), "A"_a.noconvert(), "out"_a.noconvert(),
       "regions"_a.noconvert() = nb::none(), "coupling"_a.noconvert() = nb::none(),
       "Exchange field (eq. 8), A/m. regions (uint8) and coupling (256 x 256) scale it between unlike regions (D35).");

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
        .def("__init__", [](Problem* p, const Grid& g, In Ms, In A, In K, In k, std::array<double, 3> H_ext, const std::string& device,
                            Labels regions, Table coupling, bool demag) {
            check_shape(Ms, g, false, "Ms"); check_shape(A, g, false, "A"); check_shape(K, g, false, "K"); check_shape(k, g, true, "k");
            const auto [r, c] = check_regions(regions, coupling, g);
            new (p) Problem(g, Ms.data(), A.data(), K.data(), k.data(), H_ext, parse_device(device), r, c, demag);  // copies all (D19)
        }, "grid"_a, "Ms"_a.noconvert(), "A"_a.noconvert(), "K"_a.noconvert(), "k"_a.noconvert(), "H_ext"_a, "device"_a = "cpu",
           "regions"_a.noconvert() = nb::none(), "coupling"_a.noconvert() = nb::none(), "demag"_a = true,
           "device: \"cpu\" or \"cuda\". LLG and Minimizer run on their Problem's device (D24). regions/coupling: D35; demag: D36.")
        .def_rw("H_ext", &Problem::H_ext, "External field (A/m).")
        .def("field", [](Problem& p, const std::string& term, In mag, Out out) {
            const Term t = parse_term(term);
            check_shape(mag, p.grid(), true, "m"); check_shape(out, p.grid(), true, "out");
            on_device(p, mag, out, [&](const double* m, double* H) { p.field(t, m, H); });
        }, "term"_a, "m"_a.noconvert(), "out"_a.noconvert(), "One field term (exchange, anisotropy, zeeman or demag), A/m.")
        .def("effective_field", [](Problem& p, In mag, Out out) {
            check_shape(mag, p.grid(), true, "m"); check_shape(out, p.grid(), true, "out");
            on_device(p, mag, out, [&](const double* m, double* H) { p.effective_field(m, H); });
        }, "m"_a.noconvert(), "out"_a.noconvert(), "Sum of exchange, anisotropy, Zeeman and demag fields, A/m.")
        .def("energies", [](Problem& p, In mag) {
            check_shape(mag, p.grid(), true, "m");
            const size_t N = 3 * p.grid().n();
            Buffer m(p.backend(), N);
            p.backend().to_device(m.data(), mag.data(), N);
            const auto E = p.energies(m.data());
            nb::dict d;
            d["exchange"] = E[0]; d["anisotropy"] = E[1]; d["zeeman"] = E[2]; d["demag"] = E[3];
            return d;
        }, "m"_a.noconvert(), "Partial energies (J): exchange, anisotropy, zeeman, demag.")
        .def_prop_ro("Ms", [](const Problem& p) {
            const Grid& g = p.grid();
            auto* Ms = new std::vector<double>(p.Ms());
            nb::capsule owner(Ms, [](void* q) noexcept { delete static_cast<std::vector<double>*>(q); });
            return nb::ndarray<nb::numpy, double>(Ms->data(), {g.nz, g.ny, g.nx}, owner);
        }, nb::rv_policy::move, "Copy of the per-cell saturation magnetisation (A/m).");  // the capsule owns it, not the Problem

    nb::class_<Minimizer>(m, "Minimizer")
        .def(nb::init<Problem&>(), "problem"_a, nb::keep_alive<1, 2>())
        .def("minimize", [](Minimizer& mn, Out mag, double torque_tol, size_t max_iter) {
            check_shape(mag, mn.grid(), true, "m");
            return mn.minimize(mag.data(), torque_tol, max_iter);
        }, "m"_a.noconvert(), "torque_tol"_a = 1e-6, "max_iter"_a = 100000,
           "Minimize the energy in place until max |m x H| / Ms < torque_tol; returns the iteration count.");

    nb::class_<LLG>(m, "LLG")
        .def(nb::init<Problem&, double, double, double, double>(), "problem"_a, "alpha"_a, "gamma"_a = 1.76086e11,
             "tol"_a = 1e-6, "max_step"_a = std::numeric_limits<double>::infinity(), nb::keep_alive<1, 2>())
        .def("run_until", [](LLG& l, Out mag, double t) {
            check_shape(mag, l.grid(), true, "m");
            l.run_until(mag.data(), t);
        }, "m"_a.noconvert(), "t"_a, "Advance m in place from llg.t to t (s).")
        .def_ro("t", &LLG::t).def_ro("steps", &LLG::steps).def_ro("rejected", &LLG::rejected);
}
