#include "demag.hpp"

#include <array>
#include <cmath>

#include "pocketfft_hdronly.h"

namespace {

// Newell, Williams & Dunlop (1993). Each term is skipped when its prefactor is 0, which also avoids 0/0.
double newell_f(double x, double y, double z) {
    const double x2 = x * x, y2 = y * y, z2 = z * z, R = std::sqrt(x2 + y2 + z2);
    double r = (2 * x2 - y2 - z2) * R / 6;
    if (y != 0 && z2 != x2) r += y / 2 * (z2 - x2) * std::asinh(y / std::sqrt(x2 + z2));
    if (z != 0 && y2 != x2) r += z / 2 * (y2 - x2) * std::asinh(z / std::sqrt(x2 + y2));
    if (x != 0 && y != 0 && z != 0) r -= x * y * z * std::atan(y * z / (x * R));
    return r;
}

double newell_g(double x, double y, double z) {
    const double x2 = x * x, y2 = y * y, z2 = z * z, R = std::sqrt(x2 + y2 + z2);
    double r = -x * y * R / 3;
    if (x != 0 && y != 0 && z != 0) r += x * y * z * std::asinh(z / std::sqrt(x2 + y2));
    if (y != 0 && 3 * z2 != y2) r += y / 6 * (3 * z2 - y2) * std::asinh(x / std::sqrt(y2 + z2));
    if (x != 0 && 3 * z2 != x2) r += x / 6 * (3 * z2 - x2) * std::asinh(y / std::sqrt(x2 + z2));
    if (z != 0) r -= z * z2 / 6 * std::atan(x * y / (z * R));
    if (z != 0 && y != 0) r -= z * y2 / 2 * std::atan(x * z / (y * R));
    if (z != 0 && x != 0) r -= z * x2 / 2 * std::atan(y * z / (x * R));
    return r;
}

// Second difference of F in each direction: the cell-averaged field of one box on another.
template <class F>
double newell(F f, double X, double Y, double Z, double dx, double dy, double dz) {
    const double c[3] = {-1, 2, -1};
    double s = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) s += c[i] * c[j] * c[k] * f(X + (i - 1) * dx, Y + (j - 1) * dy, Z + (k - 1) * dz);
    return s / (4 * pi * dx * dy * dz);
}

using Six = std::array<double, 6>;  // xx, yy, zz, xy, xz, yz

Six near_tensor(double X, double Y, double Z, double dx, double dy, double dz) {
    return {newell(newell_f, X, Y, Z, dx, dy, dz), newell(newell_f, Y, X, Z, dy, dx, dz), newell(newell_f, Z, Y, X, dz, dy, dx),
            newell(newell_g, X, Y, Z, dx, dy, dz), newell(newell_g, X, Z, Y, dx, dz, dy), newell(newell_g, Y, Z, X, dy, dz, dx)};
}

// Point dipole -(3 r̂r̂ - I) V/(4π r³) averaged over 3-point Gauss-Legendre nodes in both cells (D14).
// The node differences per axis collapse to 0, ±s, ±2s with weights 114, 80, 25 (/324).
Six far_tensor(double X, double Y, double Z, double dx, double dy, double dz) {
    const double s = std::sqrt(0.6) / 2, w[5] = {25, 80, 114, 80, 25};
    const double V = dx * dy * dz;
    Six N{};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k) {
                const double r[3] = {X + (i - 2) * s * dx, Y + (j - 2) * s * dy, Z + (k - 2) * s * dz};
                const double R2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                const double pre = -w[i] * w[j] * w[k] / (324.0 * 324 * 324) * V / (4 * pi * R2 * std::sqrt(R2));
                N[0] += pre * (3 * r[0] * r[0] / R2 - 1);
                N[1] += pre * (3 * r[1] * r[1] / R2 - 1);
                N[2] += pre * (3 * r[2] * r[2] / R2 - 1);
                N[3] += pre * 3 * r[0] * r[1] / R2;
                N[4] += pre * 3 * r[0] * r[2] / R2;
                N[5] += pre * 3 * r[1] * r[2] / R2;
            }
    return N;
}

pocketfft::shape_t pshape(const Grid& g) { return {padded(g.nz, g.pz), padded(g.ny, g.py), padded(g.nx, g.px)}; }

// One axis of the tensor (D45). Entry c of a free axis is the offset c (c < n), mirrored to 2n - c. Entry c of a periodic
// axis (c ≤ n/2) sums every offset d ≡ c (mod n) with |d| ≤ W = Pn + n/2, the two ends at weight ½ when n is even, and is
// mirrored to n - c. Either way the mirror of c holds the negated offsets; `none` marks an entry that is its own mirror.
struct Axis {
    static constexpr size_t none = ~size_t(0);
    std::vector<std::vector<std::pair<long, double>>> offsets;  // per computed entry c: (offset in cells, weight)
    std::vector<size_t> mirror;

    Axis(size_t n, size_t P) {
        const long N = long(n), W = long(P) * N + N / 2;
        for (size_t c = 0; c < (P > 0 ? n / 2 + 1 : n); ++c) {
            if (P == 0) {
                offsets.push_back({{long(c), 1.0}});
                mirror.push_back(c == 0 ? none : 2 * n - c);
                continue;
            }
            std::vector<std::pair<long, double>> o;
            for (long d = long(c) - (long(c) + W) / N * N; d <= W; d += N)  // the smallest d ≡ c with d ≥ -W, upwards
                o.push_back({d, N % 2 == 0 && std::abs(d) == W ? 0.5 : 1.0});
            offsets.push_back(o);
            mirror.push_back(c == 0 || 2 * c == n ? none : n - c);
        }
    }
};

}  // namespace

std::vector<double> demag_tensor(const Grid& g) {
    const size_t Px = padded(g.nx, g.px), Py = padded(g.ny, g.py), Pz = padded(g.nz, g.pz), P = Px * Py * Pz;
    const Axis ax(g.nx, g.px), ay(g.ny, g.py), az(g.nz, g.pz);
    std::vector<double> N(6 * P, 0.0);
    // Compute one octant of entries and mirror it: diagonals are even in each coordinate,
    // Nxy is odd in x and y, Nxz in x and z, Nyz in y and z (D14).
    for (size_t k = 0; k < az.mirror.size(); ++k)
        for (size_t j = 0; j < ay.mirror.size(); ++j)
            for (size_t i = 0; i < ax.mirror.size(); ++i) {
                Six t{};
                bool first = true;  // assign the first term, so a single offset (a free entry) is bitwise the D14 value
                for (const auto& [dk, wk] : az.offsets[k])
                    for (const auto& [dj, wj] : ay.offsets[j])
                        for (const auto& [di, wi] : ax.offsets[i]) {
                            // Evaluate at |offset| and apply the parity, as the octant mirroring does: Newell's
                            // cancellations make a direct evaluation at -X round differently (D14).
                            const double X = std::abs(di) * g.dx, Y = std::abs(dj) * g.dy, Z = std::abs(dk) * g.dz;
                            const double w = wi * wj * wk, sx = di < 0 ? -1 : 1, sy = dj < 0 ? -1 : 1, sz = dk < 0 ? -1 : 1;
                            const Six s = di * di + dj * dj + dk * dk < 400 ? near_tensor(X, Y, Z, g.dx, g.dy, g.dz)
                                                                            : far_tensor(X, Y, Z, g.dx, g.dy, g.dz);
                            const double sign[6] = {1, 1, 1, sx * sy, sx * sz, sy * sz};
                            for (int c = 0; c < 6; ++c) t[c] = first ? w * sign[c] * s[c] : t[c] + w * sign[c] * s[c];
                            first = false;
                        }
                for (int sk : {1, -1}) {
                    if (sk < 0 && az.mirror[k] == Axis::none) continue;
                    for (int sj : {1, -1}) {
                        if (sj < 0 && ay.mirror[j] == Axis::none) continue;
                        for (int si : {1, -1}) {
                            if (si < 0 && ax.mirror[i] == Axis::none) continue;
                            const size_t idx = ((sk > 0 ? k : az.mirror[k]) * Py + (sj > 0 ? j : ay.mirror[j])) * Px
                                               + (si > 0 ? i : ax.mirror[i]);
                            const double sign[6] = {1, 1, 1, double(si * sj), double(si * sk), double(sj * sk)};
                            for (int c = 0; c < 6; ++c) N[c * P + idx] = sign[c] * t[c];
                        }
                    }
                }
            }
    return N;
}

Demag::Demag(const Grid& g) : g_(g) {
    const auto shape = pshape(g);
    const size_t P = shape[0] * shape[1] * shape[2], Pc = shape[0] * shape[1] * (shape[2] / 2 + 1);
    real_.resize(3 * P);
    spec_.resize(3 * Pc);
    kernel_.resize(6 * Pc);
    const std::vector<double> N = demag_tensor(g);
    const pocketfft::stride_t rs = {ptrdiff_t(shape[1] * shape[2] * sizeof(double)), ptrdiff_t(shape[2] * sizeof(double)), sizeof(double)};
    const pocketfft::stride_t cs = {ptrdiff_t(shape[1] * (shape[2] / 2 + 1) * sizeof(std::complex<double>)),
                                    ptrdiff_t((shape[2] / 2 + 1) * sizeof(std::complex<double>)), sizeof(std::complex<double>)};
    for (size_t c = 0; c < 6; ++c) pocketfft::r2c(shape, rs, cs, {0, 1, 2}, pocketfft::FORWARD, &N[c * P], &kernel_[c * Pc], 1.0);
}

// eq. 21, in Fourier space: H(k) = -N(k) M(k), with M = Ms m zero-padded.
void Demag::field(const double* m, const double* Ms, double* H) {
    const auto shape = pshape(g_);
    const size_t Px = shape[2], Py = shape[1], P = shape[0] * Py * Px, Pc = shape[0] * Py * (Px / 2 + 1), n = g_.n();
    const pocketfft::stride_t rs = {ptrdiff_t(Py * Px * sizeof(double)), ptrdiff_t(Px * sizeof(double)), sizeof(double)};
    const pocketfft::stride_t cs = {ptrdiff_t(Py * (Px / 2 + 1) * sizeof(std::complex<double>)),
                                    ptrdiff_t((Px / 2 + 1) * sizeof(std::complex<double>)), sizeof(std::complex<double>)};
    auto padded_index = [&](size_t i) {  // cell index (z, y, x) -> padded index
        const size_t x = i % g_.nx, y = i / g_.nx % g_.ny, z = i / (g_.nx * g_.ny);
        return (z * Py + y) * Px + x;
    };

    std::fill(real_.begin(), real_.end(), 0.0);
    for (size_t c = 0; c < 3; ++c)
        for (size_t i = 0; i < n; ++i) real_[c * P + padded_index(i)] = Ms[i] * m[c * n + i];
    for (size_t c = 0; c < 3; ++c) pocketfft::r2c(shape, rs, cs, {0, 1, 2}, pocketfft::FORWARD, &real_[c * P], &spec_[c * Pc], 1.0);

    for (size_t q = 0; q < Pc; ++q) {
        const std::complex<double> Mx = spec_[q], My = spec_[Pc + q], Mz = spec_[2 * Pc + q];
        const std::complex<double>* K = &kernel_[q];  // K[c * Pc] is component c
        spec_[q] = -(K[0] * Mx + K[3 * Pc] * My + K[4 * Pc] * Mz);
        spec_[Pc + q] = -(K[3 * Pc] * Mx + K[Pc] * My + K[5 * Pc] * Mz);
        spec_[2 * Pc + q] = -(K[4 * Pc] * Mx + K[5 * Pc] * My + K[2 * Pc] * Mz);
    }

    for (size_t c = 0; c < 3; ++c)
        pocketfft::c2r(shape, cs, rs, {0, 1, 2}, pocketfft::BACKWARD, &spec_[c * Pc], &real_[c * P], 1.0 / double(P));
    for (size_t c = 0; c < 3; ++c)
        for (size_t i = 0; i < n; ++i) H[c * n + i] = Ms[i] != 0 ? real_[c * P + padded_index(i)] : 0.0;
}
