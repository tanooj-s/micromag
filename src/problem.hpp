#pragma once

#include <array>
#include <memory>
#include <vector>

#include "backend.hpp"

// Sums the four field terms into H_eff (D19) on the device the backend lives on (D24).
// m and H here are in the backend's memory space; the bindings copy to and from numpy.
class Problem {
public:
    Problem(const Grid& g, const double* Ms, const double* A, const double* K, const double* k, std::array<double, 3> H_ext,
            Device device);
    void field(Term t, const double* m, double* H);    // one term, overwrites H
    void effective_field(const double* m, double* H);  // overwrites H: exchange + anisotropy + Zeeman + demag
    // Partial energies (J): exchange, anisotropy, Zeeman, demag, each -c µ0 dV Σ Ms (m·H) with c = ½, ½, 1, ½ (D22).
    std::array<double, 4> energies(const double* m);
    Backend& backend() { return *b_; }
    const std::vector<double>& Ms() const { return Ms_; }  // host copy
    const Grid& grid() const { return g_; }

    std::array<double, 3> H_ext;  // A/m

private:
    Grid g_;
    std::vector<double> Ms_;
    std::unique_ptr<Backend> b_;
    Buffer tmp_;  // declared after b_, so it is released before the backend goes
};
