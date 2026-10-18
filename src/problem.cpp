#include "problem.hpp"

Problem::Problem(const Grid& g, const double* Ms, const double* A, const double* K, const double* k, std::array<double, 3> H_ext)
    : H_ext(H_ext), g_(g), Ms_(Ms, Ms + g.n()), A_(A, A + g.n()), K_(K, K + g.n()), k_(k, k + 3 * g.n()), tmp_(3 * g.n()), demag_(g) {}

// Each term overwrites (D10), so the rest are added through one scratch buffer.
void Problem::effective_field(const double* m, double* H) {
    exchange_field(g_, m, Ms_.data(), A_.data(), H);
    auto add = [&] { for (size_t i = 0; i < tmp_.size(); ++i) H[i] += tmp_[i]; };
    anisotropy_field(g_, m, Ms_.data(), K_.data(), k_.data(), tmp_.data());
    add();
    zeeman_field(g_, Ms_.data(), H_ext.data(), tmp_.data());
    add();
    demag_.field(m, Ms_.data(), tmp_.data());
    add();
}

std::array<double, 4> Problem::energies(const double* m) {
    const size_t n = g_.n();
    const double dV = g_.dx * g_.dy * g_.dz;
    auto E = [&](double c) {  // -c µ0 dV Σ Ms (m·H) over the field now in tmp_
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += Ms_[i] * (m[i] * tmp_[i] + m[n + i] * tmp_[n + i] + m[2 * n + i] * tmp_[2 * n + i]);
        return -c * mu0 * dV * s;
    };
    std::array<double, 4> out;
    exchange_field(g_, m, Ms_.data(), A_.data(), tmp_.data());
    out[0] = E(0.5);
    anisotropy_field(g_, m, Ms_.data(), K_.data(), k_.data(), tmp_.data());
    out[1] = E(0.5);
    zeeman_field(g_, Ms_.data(), H_ext.data(), tmp_.data());
    out[2] = E(1.0);
    demag_.field(m, Ms_.data(), tmp_.data());
    out[3] = E(0.5);
    return out;
}
