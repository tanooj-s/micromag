#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "fields.hpp"

enum class Device { cpu, cuda };
enum class Term { exchange, anisotropy, zeeman, demag };

// The per-cell and whole-array work that Problem, LLG and Minimizer need (D24). The drivers keep the control logic;
// a backend owns the materials and does the arithmetic. Every pointer lives in the backend's memory space, and
// vector fields are (3, n) as in D9.
class Backend {
public:
    virtual ~Backend() = default;

    virtual double* alloc(size_t len) = 0;  // uninitialised
    virtual void release(double* p) = 0;
    virtual void to_device(double* dst, const double* host, size_t len) = 0;
    virtual void to_host(double* host, const double* src, size_t len) = 0;
    virtual void copy(double* dst, const double* src, size_t len) = 0;

    // One term's field, overwriting H (D10); H = 0 in vacuum.
    virtual void field(Term t, const double* m, const std::array<double, 3>& H_ext, double* H) = 0;
    // Σ Ms (m·H) over all cells: the energy sum of D22.
    virtual double ms_dot(const double* m, const double* H) = 0;
    // out = base + h Σ_j c[j] k[j] over all 3n entries; base == nullptr means 0. out may alias base. count ≤ 7.
    virtual void lincomb(double* out, const double* base, double h, const double* c, const double* const* k, int count) = 0;
    // max over cells of |v| (the 3-vector length).
    virtual double max_norm(const double* v) = 0;
    // out = v / |v| per cell. out may alias v.
    virtual void normalise(double* out, const double* v) = 0;
    // eq. 25: dm = -g m × H - α g m × (m × H), with g = µ0 |γ| / (1 + α²).
    virtual void llg_rhs(const double* m, const double* H, double alpha, double g, double* dm) = 0;
    // eq. 22: d = -m × (m × H). Returns {max |m × H|, max over magnetic cells of |m × H| / Ms}.
    virtual std::array<double, 2> descent(const double* m, const double* H, double* d) = 0;
    // Barzilai–Borwein sums with s = m - m_prev and y = d_prev - d: {s·s, s·y, y·y}.
    virtual std::array<double, 3> bb_dots(const double* m, const double* m_prev, const double* d, const double* d_prev) = 0;
    // D41: unit normals (3, n) for one draw: Philox key (seed, stage), counter (cell, block, draw).
    virtual void normals(uint32_t seed, uint32_t stage, uint64_t draw, double* out) = 0;
    // D41: out = H + sqrt(c / Ms) dW per cell, and out = H in vacuum. H == nullptr means 0. out may alias H.
    virtual void add_thermal(double* out, const double* H, const double* dW, double c) = 0;
};

// Owning array in a backend's memory space.
class Buffer {
public:
    Buffer(Backend& b, size_t len) : b_(&b), p_(b.alloc(len)) {}
    Buffer(Buffer&& o) noexcept : b_(o.b_), p_(std::exchange(o.p_, nullptr)) {}
    Buffer& operator=(Buffer&& o) noexcept {
        std::swap(b_, o.b_);
        std::swap(p_, o.p_);
        return *this;
    }
    ~Buffer() {
        if (p_) b_->release(p_);
    }
    double* data() const { return p_; }

private:
    Backend* b_;
    double* p_;
};

// Both copy the materials, and the region map and pair table when given (D35; both or neither).
std::unique_ptr<Backend> make_cpu_backend(const Grid& g, const double* Ms, const double* A, const double* K, const double* k,
                                          const uint8_t* region, const double* pair);
#ifdef MICROMAG_CUDA
std::unique_ptr<Backend> make_cuda_backend(const Grid& g, const double* Ms, const double* A, const double* K, const double* k,
                                          const uint8_t* region, const double* pair);
#endif
