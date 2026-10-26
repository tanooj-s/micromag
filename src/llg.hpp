#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "problem.hpp"

constexpr double kB = 1.380649e-23;  // J/K, exact (SI 2019)

// rk45: adaptive Dormand–Prince (D18), T = 0 only. heun: fixed step dt. rk23: adaptive Bogacki–Shampine with a
// Brownian bridge on rejected steps (D42).
enum class Method { rk45, heun, rk23 };

// Landau–Lifshitz–Gilbert (eq. 25), with Brown's thermal field at T > 0 (D41).
class LLG {
public:
    LLG(Problem& p, double alpha, double gamma, double tol, double max_step, Method method = Method::rk45, double T = 0,
        double dt = 0, uint32_t seed = 0, uint32_t stage = 0);
    void run_until(double* m, double t_end);  // advances host m in place from t to t_end; the run stays on the device (D24)
    void thermal_field(uint64_t draw, double* H) const;  // host H: that draw's thermal field for a step of dt (D41)
    void wiener(double* W) const;                  // host W: Σ ΔW over the accepted steps, per cell and component
    const Grid& grid() const { return p_.grid(); }

    double t = 0;
    size_t steps = 0, rejected = 0, max_bridge_depth = 0;

private:
    void dmdt(const double* m, double* dm);  // eq. 25
    // eq. 25 with the thermal field of the increment dW over a step h; the deterministic field is left in Hd.
    void dmdt(const double* m, double* dm, double* Hd, const double* dW, double h);
    void rhs(const double* m, double* dm, const double* Hd, const double* dW, double h);  // the same, given Hd
    void take_increment(double& h);
    void run_rk45(double* m, double t_end);
    void run_heun(double* m, double t_end);
    void run_rk23(double* m, double t_end);
    void first_step(const double* H);
    void fresh_increment(double* dW, double h);  // dW = √h η from the next draw

    Problem& p_;
    Method method_;
    double alpha_, gamma_, tol_, max_step_, dt_, c_, h_ = 0;  // c_: 2 α kB T / (µ0 γ0 V), so σ² = c_ / (Ms h)
    bool thermal_;
    uint32_t seed_, stage_;
    uint64_t draw_ = 0;
    std::vector<Buffer> k_;  // the stages
    Buffer m_, y_, H_, e_, Hn_, Hd_, dW_, W_, Hs_;  // Hs_: the last stage's deterministic field (rk23)
    std::vector<std::pair<Buffer, double>> stack_;  // the bridge: future increments and their lengths, top at the back
};
