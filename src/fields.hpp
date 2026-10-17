#pragma once

#include <cstddef>
#include <stdexcept>

constexpr double pi = 3.141592653589793;
constexpr double mu0 = 4e-7 * pi;  // D12: classical value, matches MuMax3 and µMAG references

// Uniform grid. Arrays are C-order with x fastest: vectors (3, nz, ny, nx), scalars (nz, ny, nx) (D9).
struct Grid {
    size_t nx, ny, nz;
    double dx, dy, dz;

    Grid(size_t nx, size_t ny, size_t nz, double dx, double dy, double dz)
        : nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz) {
        if (nx == 0 || ny == 0 || nz == 0) throw std::invalid_argument("grid needs at least one cell per axis");
        if (!(dx > 0 && dy > 0 && dz > 0)) throw std::invalid_argument("cell sizes must be positive");
    }
    size_t n() const { return nx * ny * nz; }
};

// Each writes H (A/m) into H, overwriting it. Cells with Ms == 0 are vacuum and get H = 0.
void exchange_field(const Grid& g, const double* m, const double* Ms, const double* A, double* H);
void anisotropy_field(const Grid& g, const double* m, const double* Ms, const double* K, const double* k, double* H);
void zeeman_field(const Grid& g, const double* Ms, const double H_ext[3], double* H);
