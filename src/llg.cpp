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

// Bogacki–Shampine 3(2). Row 4 of A is the 3rd-order solution, so its field is the next step's first (FSAL).
constexpr double BS_A[4][3] = {{}, {1.0 / 2}, {0, 3.0 / 4}, {2.0 / 9, 1.0 / 3, 4.0 / 9}};
// 3rd-order minus embedded 2nd-order weights: the local error estimate.
constexpr double BS_E[4] = {-5.0 / 72, 1.0 / 12, 1.0 / 9, -1.0 / 8};

}  // namespace

LLG::LLG(Problem& p, double alpha, double gamma, double tol, double max_step, Method method, double T, double dt,
         uint32_t seed, uint32_t stage)
    : p_(p), method_(method), alpha_(alpha), gamma_(gamma), tol_(tol), max_step_(max_step), dt_(dt),
      c_(2 * alpha * kB * T / (mu0 * mu0 * gamma * p.grid().dx * p.grid().dy * p.grid().dz)), thermal_(T > 0), seed_(seed),
      stage_(stage), m_(p.backend(), 3 * p.grid().n()), y_(p.backend(), 3 * p.grid().n()), H_(p.backend(), 3 * p.grid().n()),
      e_(p.backend(), 3 * p.grid().n()), Hn_(p.backend(), 3 * p.grid().n()), Hd_(p.backend(), 3 * p.grid().n()),
      dW_(p.backend(), 3 * p.grid().n()), W_(p.backend(), 3 * p.grid().n()), Hs_(p.backend(), 3 * p.grid().n()) {
    if (!(T >= 0)) throw std::invalid_argument("LLG: temperature must be >= 0");
    if (thermal_ && method == Method::rk45) throw std::invalid_argument("LLG: rk45 can't take noise; use heun or rk23 at T > 0 (D42)");
    if (method == Method::heun && !(dt > 0)) throw std::invalid_argument("LLG: heun needs dt > 0 (D43)");
    for (int s = 0; s < 7; ++s) k_.emplace_back(p.backend(), 3 * p.grid().n());
    p.backend().lincomb(W_.data(), nullptr, 0, nullptr, nullptr, 0);  // W = 0
}

// eq. 25: dm/dt = -γ' m × H - α γ' m × (m × H), with γ' = µ0 |γ| / (1 + α²).
void LLG::dmdt(const double* m, double* dm) {
    p_.effective_field(m, H_.data());
    p_.backend().llg_rhs(m, H_.data(), alpha_, mu0 * gamma_ / (1 + alpha_ * alpha_), dm);
}

// D41: the thermal field over a step h is √(c / Ms) ΔW / h, with ΔW ~ N(0, h) per component.
void LLG::dmdt(const double* m, double* dm, double* Hd, const double* dW, double h) {
    p_.effective_field(m, Hd);
    rhs(m, dm, Hd, dW, h);
}

void LLG::rhs(const double* m, double* dm, const double* Hd, const double* dW, double h) {
    const double* H = Hd;
    if (thermal_) {
        p_.backend().add_thermal(Hn_.data(), Hd, dW, c_ / (h * h));
        H = Hn_.data();
    }
    p_.backend().llg_rhs(m, H, alpha_, mu0 * gamma_ / (1 + alpha_ * alpha_), dm);
}

// dW_ = ΔW over the next h (D42): the top of the bridge stack if there is one, shortening h to it, or split from it by
// the Brownian bridge if h is shorter; a fresh draw when the stack is empty.
void LLG::take_increment(double& h) {
    if (stack_.empty()) return fresh_increment(dW_.data(), h);
    Backend& b = p_.backend();
    auto& [B, hs] = stack_.back();
    // Within rounding of the whole entry, use all of it: a sliver left on the stack would later be a step too short to
    // advance t (t + h == t).
    if (hs - h <= std::max(1e-9 * hs, 1e-12 * t)) {
        h = hs;
        std::swap(dW_, B);
        stack_.pop_back();
        return;
    }
    // Given ΔW = B over hs, the part over h < hs is N((h / hs) B, h (hs − h) / hs); the rest stays on the stack.
    b.normals(seed_, stage_, draw_++, dW_.data());
    const double c[2] = {h / hs, std::sqrt(h * (hs - h) / hs)}, one = 1;
    const double* k[2] = {B.data(), dW_.data()};
    b.lincomb(dW_.data(), nullptr, 1, c, k, 2);
    b.lincomb(B.data(), B.data(), -1, &one, &k[1], 1);
    hs -= h;
}

void LLG::fresh_increment(double* dW, double h) {
    Backend& b = p_.backend();
    b.normals(seed_, stage_, draw_++, dW);
    const double one = 1;
    b.lincomb(dW, nullptr, std::sqrt(h), &one, &dW, 1);
}

void LLG::thermal_field(uint64_t draw, double* H_host) const {
    Backend& b = p_.backend();
    Buffer H(b, 3 * p_.grid().n());
    b.normals(seed_, stage_, draw, H.data());
    b.add_thermal(H.data(), nullptr, H.data(), dt_ > 0 ? c_ / dt_ : 0.0);
    b.to_host(H_host, H.data(), 3 * p_.grid().n());
}

void LLG::wiener(double* W) const { p_.backend().to_host(W, W_.data(), 3 * p_.grid().n()); }

void LLG::first_step(const double* H) {  // 1% of a precession period at the largest field
    if (h_ != 0) return;
    const double Hmax = p_.backend().max_norm(H);
    h_ = Hmax > 0 ? 0.01 * 2 * pi * (1 + alpha_ * alpha_) / (mu0 * gamma_ * Hmax) : 1e-12;
    h_ = std::min(h_, max_step_);
}

void LLG::run_until(double* m_host, double t_end) {
    Backend& b = p_.backend();
    const size_t N = 3 * p_.grid().n();
    b.to_device(m_.data(), m_host, N);
    switch (method_) {
        case Method::rk45: run_rk45(m_.data(), t_end); break;
        case Method::heun: run_heun(m_.data(), t_end); break;
        case Method::rk23: run_rk23(m_.data(), t_end); break;
    }
    b.to_host(m_host, m_.data(), N);
}

void LLG::run_rk45(double* m, double t_end) {
    Backend& b = p_.backend();
    const double* k[7];
    for (int s = 0; s < 7; ++s) k[s] = k_[s].data();
    // ponytail: fresh first stage on every call (one extra field evaluation per output sample, ~8% for SP4);
    // keep the FSAL stage across calls, invalidated on any change to m or H_ext, if profiling says it matters (D18)
    dmdt(m, k_[0].data());
    first_step(H_.data());

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
}

// Heun: one increment per step, held over the predictor and the corrector, so the scheme is Stratonovich (D42).
void LLG::run_heun(double* m, double t_end) {
    Backend& b = p_.backend();
    double *k1 = k_[0].data(), *k2 = k_[1].data(), *y = y_.data(), *dW = dW_.data();
    const double* k[2] = {k1, k2};
    const double one = 1, half[2] = {0.5, 0.5};
    while (t < t_end) {
        const bool clipped = t_end - t <= dt_ * (1 + 1e-9);  // within rounding of dt: land, no sliver step after
        const double h = clipped ? t_end - t : dt_;
        if (thermal_) fresh_increment(dW, h);
        dmdt(m, k1, Hd_.data(), dW, h);
        b.lincomb(y, m, h, &one, k, 1);
        dmdt(y, k2, Hd_.data(), dW, h);
        b.lincomb(y, m, h, half, k, 2);
        b.normalise(m, y);
        if (thermal_) b.lincomb(W_.data(), W_.data(), 1, &one, &dW, 1);  // the path's running total (D44)
        t = clipped ? t_end : t + h;
        ++steps;
    }
}

void LLG::run_rk23(double* m, double t_end) {
    Backend& b = p_.backend();
    const double* k[4];
    for (int s = 0; s < 4; ++s) k[s] = k_[s].data();
    const double one = 1;
    const double* dW = dW_.data();
    p_.effective_field(m, H_.data());  // H_: the deterministic field of the first stage; later, the last accepted stage's
    first_step(H_.data());

    while (t < t_end) {
        const bool clipped = h_ >= t_end - t;
        const double h0 = clipped ? t_end - t : h_;
        double h = h0;
        if (thermal_) take_increment(h);  // may shorten h
        dW = dW_.data();                  // take_increment may have swapped the buffer
        const bool lands = clipped && h0 - h <= std::max(1e-9 * h0, 1e-12 * t);  // likewise for a sliver before t_end
        if (t + h == t) throw std::runtime_error("LLG: step size underflow");

        rhs(m, k_[0].data(), H_.data(), dW, h);  // FSAL: the field is reused, the noise is this step's
        for (int s = 1; s < 4; ++s) {
            b.lincomb(y_.data(), m, h, BS_A[s], k, s);
            dmdt(y_.data(), k_[s].data(), s == 3 ? Hs_.data() : Hd_.data(), dW, h);
        }
        b.lincomb(e_.data(), nullptr, h, BS_E, k, 4);
        const double err = b.max_norm(e_.data());  // largest per-cell error vector (D18)

        const bool accept = err <= tol_;
        if (accept) {
            b.normalise(m, y_.data());
            std::swap(H_, Hs_);
            if (thermal_) b.lincomb(W_.data(), W_.data(), 1, &one, &dW, 1);  // the path's running total (D44)
            t = lands ? t_end : t + h;
            ++steps;
        } else {
            ++rejected;
            if (thermal_) {  // the increment is drawn: put it back, so the retry splits it by the bridge
                // ponytail: one allocation per rejection; keep spare buffers if it shows in a GPU profile
                stack_.emplace_back(Buffer(b, 3 * p_.grid().n()), h);
                std::swap(stack_.back().first, dW_);
                max_bridge_depth = std::max(max_bridge_depth, stack_.size());
            }
        }
        // A step cut short (to land on t_end or to the bridge) that succeeded says nothing about the next step size.
        if (!(accept && (clipped || h < h0))) {
            const double factor = std::clamp(0.9 * std::pow(tol_ / err, 1.0 / 3), 0.2, 5.0);  // err = 0 gives inf -> 5
            h_ = std::min(h * factor, max_step_);
        }
    }
}
