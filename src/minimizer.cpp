#include "minimizer.hpp"

#include <stdexcept>
#include <string>

Minimizer::Minimizer(Problem& p)
    : p_(p), m_(p.backend(), 3 * p.grid().n()), H_(p.backend(), 3 * p.grid().n()), d_(p.backend(), 3 * p.grid().n()),
      d_prev_(p.backend(), 3 * p.grid().n()), m_prev_(p.backend(), 3 * p.grid().n()) {}

size_t Minimizer::minimize(double* m_host, double torque_tol, size_t max_iter) {
    Backend& b = p_.backend();
    const size_t N = 3 * p_.grid().n();
    double* m = m_.data();
    b.to_device(m, m_host, N);
    double max_d = 0, torque = 0;

    // eq. 22 in field units: d = -m × (m × H), the descent direction. |d| = |m × H| for unit m.
    auto direction = [&] {
        p_.effective_field(m, H_.data());
        const auto r = b.descent(m, H_.data(), d_.data());
        max_d = r[0];
        torque = r[1];
    };

    direction();
    double tau = max_d > 0 ? 0.01 / max_d : 0;  // first step turns the largest-torque cell by ~0.01 rad (D21)
    size_t iter = 0;
    const double one = 1;
    while (torque >= torque_tol) {
        if (iter == max_iter) {
            b.to_host(m_host, m, N);
            throw std::runtime_error("Minimizer: no convergence after " + std::to_string(max_iter) + " iterations (max torque/Ms " +
                                     std::to_string(torque) + ")");
        }
        b.copy(m_prev_.data(), m, N);
        std::swap(d_prev_, d_);  // direction() overwrites all of d_
        const double* d_prev = d_prev_.data();
        b.lincomb(m, m, tau, &one, &d_prev, 1);
        b.normalise(m, m);
        direction();
        ++iter;

        // Barzilai–Borwein with s = Δm and y = Δ(gradient) = -Δd, alternating BB1 and BB2.
        const auto [ss, sy, yy] = b.bb_dots(m, m_prev_.data(), d_.data(), d_prev_.data());
        if (sy > 0)
            tau = iter % 2 ? ss / sy : sy / yy;
        else  // non-convex region (common near switching): restart from the small rotation rule
            tau = max_d > 0 ? 0.01 / max_d : 0;
    }
    b.to_host(m_host, m, N);
    return iter;
}
