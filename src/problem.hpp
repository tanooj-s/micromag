#pragma once

#include <array>
#include <vector>

#include "demag.hpp"

// Owns copies of the materials and sums the four field terms into H_eff (D19).
class Problem {
public:
    Problem(const Grid& g, const double* Ms, const double* A, const double* K, const double* k, std::array<double, 3> H_ext);
    void effective_field(const double* m, double* H);  // overwrites H: exchange + anisotropy + Zeeman + demag
    // Partial energies (J): exchange, anisotropy, Zeeman, demag, each -c µ0 dV Σ Ms (m·H) with c = ½, ½, 1, ½ (D22).
    std::array<double, 4> energies(const double* m);
    const std::vector<double>& Ms() const { return Ms_; }
    const Grid& grid() const { return g_; }

    std::array<double, 3> H_ext;  // A/m

private:
    Grid g_;
    std::vector<double> Ms_, A_, K_, k_, tmp_;
    Demag demag_;
};
