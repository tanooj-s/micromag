#include "problem.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "demag.hpp"
#include "philox.hpp"

namespace {

// The CPU oracle: the iteration 1–4 loops, unchanged.
class CpuBackend : public Backend {
public:
    CpuBackend(const Grid& g, const double* Ms, const double* A, const double* K, const double* k, const uint8_t* region,
               const double* pair)
        : g_(g), Ms_(Ms, Ms + g.n()), A_(A, A + g.n()), K_(K, K + g.n()), k_(k, k + 3 * g.n()), demag_(g) {
        if (region) {
            region_.assign(region, region + g.n());
            pair_.assign(pair, pair + n_regions * n_regions);
        }
    }

    double* alloc(size_t len) override { return new double[len]; }
    void release(double* p) override { delete[] p; }
    void to_device(double* dst, const double* host, size_t len) override { std::copy(host, host + len, dst); }
    void to_host(double* host, const double* src, size_t len) override { std::copy(src, src + len, host); }
    void copy(double* dst, const double* src, size_t len) override { std::copy(src, src + len, dst); }

    void field(Term t, const double* m, const std::array<double, 3>& H_ext, double* H) override {
        switch (t) {
            case Term::exchange:
                return exchange_field(g_, m, Ms_.data(), A_.data(), H, region_.empty() ? nullptr : region_.data(),
                                      region_.empty() ? nullptr : pair_.data());
            case Term::anisotropy: return anisotropy_field(g_, m, Ms_.data(), K_.data(), k_.data(), H);
            case Term::zeeman: return zeeman_field(g_, Ms_.data(), H_ext.data(), H);
            case Term::demag: return demag_.field(m, Ms_.data(), H);
        }
    }

    double ms_dot(const double* m, const double* H) override {
        const size_t n = g_.n();
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += Ms_[i] * (m[i] * H[i] + m[n + i] * H[n + i] + m[2 * n + i] * H[2 * n + i]);
        return s;
    }

    void lincomb(double* out, const double* base, double h, const double* c, const double* const* k, int count) override {
        const size_t N = 3 * g_.n();
        for (size_t i = 0; i < N; ++i) {
            double acc = 0;
            for (int j = 0; j < count; ++j) acc += c[j] * k[j][i];
            out[i] = base ? base[i] + h * acc : h * acc;
        }
    }

    double max_norm(const double* v) override {
        const size_t n = g_.n();
        double v2 = 0;
        for (size_t i = 0; i < n; ++i) v2 = std::max(v2, v[i] * v[i] + v[n + i] * v[n + i] + v[2 * n + i] * v[2 * n + i]);
        return std::sqrt(v2);
    }

    void normalise(double* out, const double* v) override {
        const size_t n = g_.n();
        for (size_t i = 0; i < n; ++i) {
            const double inv = 1 / std::sqrt(v[i] * v[i] + v[n + i] * v[n + i] + v[2 * n + i] * v[2 * n + i]);
            for (size_t c = 0; c < 3; ++c) out[c * n + i] = v[c * n + i] * inv;
        }
    }

    void llg_rhs(const double* m, const double* H, double alpha, double g, double* dm) override {
        const size_t n = g_.n();
        for (size_t i = 0; i < n; ++i) {
            const double mx = m[i], my = m[n + i], mz = m[2 * n + i];
            const double Hx = H[i], Hy = H[n + i], Hz = H[2 * n + i];
            const double px = my * Hz - mz * Hy, py = mz * Hx - mx * Hz, pz = mx * Hy - my * Hx;  // m × H
            const double dx = my * pz - mz * py, dy = mz * px - mx * pz, dz = mx * py - my * px;  // m × (m × H)
            dm[i] = -g * (px + alpha * dx);
            dm[n + i] = -g * (py + alpha * dy);
            dm[2 * n + i] = -g * (pz + alpha * dz);
        }
    }

    std::array<double, 2> descent(const double* m, const double* H, double* d) override {
        const size_t n = g_.n();
        double max_d = 0, torque = 0;
        for (size_t i = 0; i < n; ++i) {
            const double mx = m[i], my = m[n + i], mz = m[2 * n + i];
            const double px = my * H[2 * n + i] - mz * H[n + i], py = mz * H[i] - mx * H[2 * n + i], pz = mx * H[n + i] - my * H[i];
            d[i] = -(my * pz - mz * py);
            d[n + i] = -(mz * px - mx * pz);
            d[2 * n + i] = -(mx * py - my * px);
            const double t = std::sqrt(px * px + py * py + pz * pz);
            max_d = std::max(max_d, t);
            if (Ms_[i] != 0) torque = std::max(torque, t / Ms_[i]);
        }
        return {max_d, torque};
    }

    std::array<double, 3> bb_dots(const double* m, const double* m_prev, const double* d, const double* d_prev) override {
        const size_t N = 3 * g_.n();
        double ss = 0, sy = 0, yy = 0;
        for (size_t i = 0; i < N; ++i) {
            const double s = m[i] - m_prev[i], y = d_prev[i] - d[i];
            ss += s * s;
            sy += s * y;
            yy += y * y;
        }
        return {ss, sy, yy};
    }

    void normals(uint32_t seed, uint32_t stage, uint64_t draw, double* out) override {
        const size_t n = g_.n();
        for (size_t i = 0; i < n; ++i) {
            double z[3];
            normal3(seed, stage, draw, uint32_t(i), z);
            for (size_t c = 0; c < 3; ++c) out[c * n + i] = z[c];
        }
    }

    void add_thermal(double* out, const double* H, const double* dW, double c) override {
        const size_t n = g_.n();
        for (size_t i = 0; i < n; ++i) {
            const double s = Ms_[i] != 0 ? std::sqrt(c / Ms_[i]) : 0.0;
            for (size_t j = 0; j < 3; ++j) out[j * n + i] = (H ? H[j * n + i] : 0.0) + s * dW[j * n + i];
        }
    }

private:
    Grid g_;
    std::vector<double> Ms_, A_, K_, k_, pair_;
    std::vector<uint8_t> region_;  // empty: no pair table (D35)
    Demag demag_;
};

std::unique_ptr<Backend> make_backend(Device d, const Grid& g, const double* Ms, const double* A, const double* K, const double* k,
                                      const uint8_t* region, const double* pair) {
    if (d == Device::cpu) return make_cpu_backend(g, Ms, A, K, k, region, pair);
#ifdef MICROMAG_CUDA
    return make_cuda_backend(g, Ms, A, K, k, region, pair);
#else
    throw std::runtime_error("micromag was built without CUDA (set MICROMAG_CUDA=ON)");
#endif
}

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const Grid& g, const double* Ms, const double* A, const double* K, const double* k,
                                          const uint8_t* region, const double* pair) {
    return std::make_unique<CpuBackend>(g, Ms, A, K, k, region, pair);
}

Problem::Problem(const Grid& g, const double* Ms, const double* A, const double* K, const double* k, std::array<double, 3> H_ext,
                 Device device, const uint8_t* region, const double* pair, bool demag)
    : H_ext(H_ext), g_(g), demag_(demag), Ms_(Ms, Ms + g.n()), b_(make_backend(device, g, Ms, A, K, k, region, pair)),
      tmp_(*b_, 3 * g.n()) {
    if (!region != !pair) throw std::invalid_argument("give both the region map and the pair table, or neither (D35)");
}

// D36: with demag off, its field is zero (lincomb with no terms writes h · 0).
void Problem::field(Term t, const double* m, double* H) {
    if (t == Term::demag && !demag_) return b_->lincomb(H, nullptr, 0, nullptr, nullptr, 0);
    b_->field(t, m, H_ext, H);
}

// Each term overwrites (D10), so the rest are added through one scratch buffer.
void Problem::effective_field(const double* m, double* H) {
    b_->field(Term::exchange, m, H_ext, H);
    const double one = 1;
    const double* tmp = tmp_.data();
    for (Term t : {Term::anisotropy, Term::zeeman, Term::demag}) {
        if (t == Term::demag && !demag_) continue;
        b_->field(t, m, H_ext, tmp_.data());
        b_->lincomb(H, H, 1, &one, &tmp, 1);
    }
}

std::array<double, 4> Problem::energies(const double* m) {
    const double dV = g_.dx * g_.dy * g_.dz;
    const Term terms[4] = {Term::exchange, Term::anisotropy, Term::zeeman, Term::demag};
    const double c[4] = {0.5, 0.5, 1.0, 0.5};
    std::array<double, 4> out;
    for (int i = 0; i < 4; ++i) {
        if (terms[i] == Term::demag && !demag_) {
            out[i] = 0;
            continue;
        }
        b_->field(terms[i], m, H_ext, tmp_.data());
        out[i] = -c[i] * mu0 * dV * b_->ms_dot(m, tmp_.data());
    }
    return out;
}
