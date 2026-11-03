#pragma once

#include <complex>
#include <vector>

#include "fields.hpp"

// FFT length for an axis of n cells with P periodic copies per side: 2n zero padding on a free axis (D15), but an axis
// with one cell needs none, and a periodic axis must wrap, so it is not padded (D45).
inline size_t padded(size_t n, size_t P) { return P > 0 || n == 1 ? n : 2 * n; }

// Demag tensor (6, Pz, Py, Px), components xx, yy, zz, xy, xz, yz, in FFT wraparound order (D14).
// On a periodic axis, entry i sums the copies in a window centred on each cell (D45).
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
