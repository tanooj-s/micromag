#pragma once

#include <complex>
#include <vector>

#include "fields.hpp"

// Padded FFT length for an axis of n cells (D15): 2n, but an axis with one cell needs no padding.
inline size_t padded(size_t n) { return n == 1 ? 1 : 2 * n; }

// Demag tensor (6, Pz, Py, Px), components xx, yy, zz, xy, xz, yz, in FFT wraparound order (D14).
std::vector<double> demag_tensor(const Grid& g);

// eq. 21: H_i = -Σ_j N_ij Ms_j m_j by zero-padded FFT convolution. Owns the kernel spectrum and scratch (D16).
class Demag {
public:
    explicit Demag(const Grid& g);
    void field(const double* m, const double* Ms, double* H);  // overwrites H; H = 0 in vacuum
    const Grid& grid() const { return g_; }

private:
    Grid g_;
    std::vector<std::complex<double>> kernel_;  // (6, Pz, Py, Px/2+1)
    std::vector<double> real_;                  // (3, Pz, Py, Px) scratch
    std::vector<std::complex<double>> spec_;    // (3, Pz, Py, Px/2+1) scratch
};
