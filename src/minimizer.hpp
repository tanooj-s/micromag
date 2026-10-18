#pragma once

#include <vector>

#include "problem.hpp"

// Projected steepest descent with alternating Barzilai–Borwein steps (eqs. 22–23, D21).
class Minimizer {
public:
    explicit Minimizer(Problem& p);
    // Updates host m in place until max |m × H| / Ms < torque_tol; returns the iteration count, throws at max_iter.
    // The iterations stay on the device (D24).
    size_t minimize(double* m, double torque_tol, size_t max_iter);
    const Grid& grid() const { return p_.grid(); }

private:
    Problem& p_;
    Buffer m_, H_, d_, d_prev_, m_prev_;
};
