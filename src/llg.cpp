#include "llg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// Dormand–Prince 5(4) tableau. Row 7 of A is the 5th-order solution, so its stage is the next step's first (FSAL).
constexpr double A[7][6] = {
    {},
    {1.0 / 5},
    {3.0 / 40, 9.0 / 40},
    {44.0 / 45, -56.0 / 15, 32.0 / 9},
    {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
    {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
    {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84},
};
// 5th-order minus embedded 4th-order weights: the local error estimate.
constexpr double E[7] = {71.0 / 57600, 0, -71.0 / 16695, 71.0 / 1920, -17253.0 / 339200, 22.0 / 525, -1.0 / 40};

}  // namespace

LLG::LLG(Problem& p, double alpha, double gamma, double tol, double max_step)
    : p_(p), alpha_(alpha), gamma_(gamma), tol_(tol), max_step_(max_step), m_(p.backend(), 3 * p.grid().n()),
      y_(p.backend(), 3 * p.grid().n()), H_(p.backend(), 3 * p.grid().n()), e_(p.backend(), 3 * p.grid().n()) {
    for (int s = 0; s < 7; ++s) k_.emplace_back(p.backend(), 3 * p.grid().n());
}

// eq. 25: dm/dt = -γ' m × H - α γ' m × (m × H), with γ' = µ0 |γ| / (1 + α²).
void LLG::dmdt(const double* m, double* dm) {
    p_.effective_field(m, H_.data());
    p_.backend().llg_rhs(m, H_.data(), alpha_, mu0 * gamma_ / (1 + alpha_ * alpha_), dm);
}

void LLG::run_until(double* m_host, double t_end) {
    Backend& b = p_.backend();
    const size_t N = 3 * p_.grid().n();
    double* m = m_.data();
    b.to_device(m, m_host, N);
    const double* k[7];
    for (int s = 0; s < 7; ++s) k[s] = k_[s].data();
    // ponytail: fresh first stage on every call (one extra field evaluation per output sample, ~8% for SP4);
    // keep the FSAL stage across calls, invalidated on any change to m or H_ext, if profiling says it matters (D18)
    dmdt(m, k_[0].data());
    if (h_ == 0) {  // first step: 1% of a precession period at the largest field
        const double Hmax = b.max_norm(H_.data());
        h_ = Hmax > 0 ? 0.01 * 2 * pi * (1 + alpha_ * alpha_) / (mu0 * gamma_ * Hmax) : 1e-12;
        h_ = std::min(h_, max_step_);
    }

    while (t < t_end) {
        const bool clipped = h_ >= t_end - t;
        const double h = clipped ? t_end - t : h_;
        if (t + h == t) throw std::runtime_error("LLG: step size underflow");

        for (int s = 1; s < 7; ++s) {
            b.lincomb(y_.data(), m, h, A[s], k, s);
            dmdt(y_.data(), k_[s].data());
        }

        b.lincomb(e_.data(), nullptr, h, E, k, 7);
        const double err = b.max_norm(e_.data());  // largest per-cell error vector (D18)

        const bool accept = err <= tol_;
        if (accept) {
            b.normalise(m, y_.data());  // renormalise after the error estimate (D18)
            std::swap(k_[0], k_[6]);    // FSAL: the last stage is the next step's first
            std::swap(k[0], k[6]);
            t = clipped ? t_end : t + h;
            ++steps;
        } else {
            ++rejected;
        }
        // A clipped step that succeeded says nothing about how large the next step may be.
        if (!(accept && clipped)) {
            const double factor = std::clamp(0.9 * std::pow(tol_ / err, 0.2), 0.2, 5.0);  // err = 0 gives inf -> 5
            h_ = std::min(h * factor, max_step_);
        }
    }
    b.to_host(m_host, m, N);
}
