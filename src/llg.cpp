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
    : p_(p), alpha_(alpha), gamma_(gamma), tol_(tol), max_step_(max_step) {
    const size_t N = 3 * p.grid().n();
    for (auto& k : k_) k.resize(N);
    y_.resize(N);
    H_.resize(N);
}

// eq. 25: dm/dt = -γ' m × H - α γ' m × (m × H), with γ' = µ0 |γ| / (1 + α²).
void LLG::dmdt(const double* m, double* dm) {
    p_.effective_field(m, H_.data());
    const size_t n = p_.grid().n();
    const double g = mu0 * gamma_ / (1 + alpha_ * alpha_);
    for (size_t i = 0; i < n; ++i) {
        const double mx = m[i], my = m[n + i], mz = m[2 * n + i];
        const double Hx = H_[i], Hy = H_[n + i], Hz = H_[2 * n + i];
        const double px = my * Hz - mz * Hy, py = mz * Hx - mx * Hz, pz = mx * Hy - my * Hx;  // m × H
        const double dx = my * pz - mz * py, dy = mz * px - mx * pz, dz = mx * py - my * px;  // m × (m × H)
        dm[i] = -g * (px + alpha_ * dx);
        dm[n + i] = -g * (py + alpha_ * dy);
        dm[2 * n + i] = -g * (pz + alpha_ * dz);
    }
}

void LLG::run_until(double* m, double t_end) {
    const size_t n = p_.grid().n(), N = 3 * n;
    // ponytail: fresh first stage on every call (one extra field evaluation per output sample, ~8% for SP4);
    // keep the FSAL stage across calls, invalidated on any change to m or H_ext, if profiling says it matters (D18)
    dmdt(m, k_[0].data());
    if (h_ == 0) {  // first step: 1% of a precession period at the largest field
        double H2 = 0;
        for (size_t i = 0; i < n; ++i) H2 = std::max(H2, H_[i] * H_[i] + H_[n + i] * H_[n + i] + H_[2 * n + i] * H_[2 * n + i]);
        h_ = H2 > 0 ? 0.01 * 2 * pi * (1 + alpha_ * alpha_) / (mu0 * gamma_ * std::sqrt(H2)) : 1e-12;
        h_ = std::min(h_, max_step_);
    }

    while (t < t_end) {
        const bool clipped = h_ >= t_end - t;
        const double h = clipped ? t_end - t : h_;
        if (t + h == t) throw std::runtime_error("LLG: step size underflow");

        for (int s = 1; s < 7; ++s) {
            for (size_t i = 0; i < N; ++i) {
                double acc = 0;
                for (int j = 0; j < s; ++j) acc += A[s][j] * k_[j][i];
                y_[i] = m[i] + h * acc;
            }
            dmdt(y_.data(), k_[s].data());
        }

        double err = 0;  // largest per-cell error vector (D18)
        for (size_t i = 0; i < n; ++i) {
            double e2 = 0;
            for (size_t c = 0; c < 3; ++c) {
                double e = 0;
                for (int j = 0; j < 7; ++j) e += E[j] * k_[j][c * n + i];
                e2 += e * e;
            }
            err = std::max(err, h * std::sqrt(e2));
        }

        const bool accept = err <= tol_;
        if (accept) {
            for (size_t i = 0; i < n; ++i) {  // renormalise after the error estimate (D18)
                const double inv = 1 / std::sqrt(y_[i] * y_[i] + y_[n + i] * y_[n + i] + y_[2 * n + i] * y_[2 * n + i]);
                for (size_t c = 0; c < 3; ++c) m[c * n + i] = y_[c * n + i] * inv;
            }
            std::swap(k_[0], k_[6]);  // FSAL: the last stage is the next step's first
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
}
