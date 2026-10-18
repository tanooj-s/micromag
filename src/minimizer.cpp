#include "minimizer.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

Minimizer::Minimizer(Problem& p) : p_(p) {
    const size_t N = 3 * p.grid().n();
    H_.resize(N);
    d_.resize(N);
    d_prev_.resize(N);
    m_prev_.resize(N);
}

size_t Minimizer::minimize(double* m, double torque_tol, size_t max_iter) {
    const size_t n = p_.grid().n(), N = 3 * n;
    const std::vector<double>& Ms = p_.Ms();
    double max_d = 0, torque = 0;

    // eq. 22 in field units: d = -m × (m × H), the descent direction. |d| = |m × H| for unit m.
    auto direction = [&] {
        p_.effective_field(m, H_.data());
        max_d = torque = 0;
        for (size_t i = 0; i < n; ++i) {
            const double mx = m[i], my = m[n + i], mz = m[2 * n + i];
            const double px = my * H_[2 * n + i] - mz * H_[n + i], py = mz * H_[i] - mx * H_[2 * n + i], pz = mx * H_[n + i] - my * H_[i];
            d_[i] = -(my * pz - mz * py);
            d_[n + i] = -(mz * px - mx * pz);
            d_[2 * n + i] = -(mx * py - my * px);
            const double t = std::sqrt(px * px + py * py + pz * pz);
            max_d = std::max(max_d, t);
            if (Ms[i] != 0) torque = std::max(torque, t / Ms[i]);
        }
    };

    direction();
    double tau = max_d > 0 ? 0.01 / max_d : 0;  // first step turns the largest-torque cell by ~0.01 rad (D21)
    size_t iter = 0;
    while (torque >= torque_tol) {
        if (iter == max_iter)
            throw std::runtime_error("Minimizer: no convergence after " + std::to_string(max_iter) + " iterations (max torque/Ms " +
                                     std::to_string(torque) + ")");
        m_prev_.assign(m, m + N);
        d_prev_ = d_;
        for (size_t i = 0; i < n; ++i) {
            double v[3], norm2 = 0;
            for (size_t c = 0; c < 3; ++c) norm2 += (v[c] = m[c * n + i] + tau * d_[c * n + i]) * v[c];
            const double inv = 1 / std::sqrt(norm2);
            for (size_t c = 0; c < 3; ++c) m[c * n + i] = v[c] * inv;
        }
        direction();
        ++iter;

        // Barzilai–Borwein with s = Δm and y = Δ(gradient) = -Δd, alternating BB1 and BB2.
        double ss = 0, sy = 0, yy = 0;
        for (size_t i = 0; i < N; ++i) {
            const double s = m[i] - m_prev_[i], y = d_prev_[i] - d_[i];
            ss += s * s;
            sy += s * y;
            yy += y * y;
        }
        if (sy > 0)
            tau = iter % 2 ? ss / sy : sy / yy;
        else  // non-convex region (common near switching): restart from the small rotation rule
            tau = max_d > 0 ? 0.01 / max_d : 0;
    }
    return iter;
}
