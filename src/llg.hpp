#pragma once

#include <vector>

#include "problem.hpp"

// Landau–Lifshitz–Gilbert (eq. 25) integrated with adaptive Dormand–Prince RK45 (D18).
class LLG {
public:
    LLG(Problem& p, double alpha, double gamma, double tol, double max_step);
    void run_until(double* m, double t_end);  // advances m in place from t to t_end
    const Grid& grid() const { return p_.grid(); }

    double t = 0;
    size_t steps = 0, rejected = 0;

private:
    void dmdt(const double* m, double* dm);  // eq. 25

    Problem& p_;
    double alpha_, gamma_, tol_, max_step_, h_ = 0;
    std::vector<double> k_[7], y_, H_;
};
