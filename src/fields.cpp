#include "fields.hpp"

// D11: face coupling between unlike cells; 0 if either side has A = 0.
static double harmonic_mean(double a, double b) { return a + b > 0 ? 2 * a * b / (a + b) : 0.0; }

// eq. 8: H = 2/(µ0 Ms) ∇·(A ∇m), discretised as the 6-neighbour sum of A_face (m_j - m_i) / d².
// Missing (boundary) or vacuum neighbours contribute nothing: free (Neumann) surfaces (D11).
void exchange_field(const Grid& g, const double* m, const double* Ms, const double* A, double* H) {
    const size_t n = g.n(), sy = g.nx, sz = g.nx * g.ny;
    for (size_t z = 0; z < g.nz; ++z)
        for (size_t y = 0; y < g.ny; ++y)
            for (size_t x = 0; x < g.nx; ++x) {
                const size_t i = z * sz + y * sy + x;
                double h[3] = {0, 0, 0};
                auto add = [&](size_t j, double d) {
                    if (Ms[j] == 0) return;
                    const double w = harmonic_mean(A[i], A[j]) / (d * d);
                    for (size_t c = 0; c < 3; ++c) h[c] += w * (m[c * n + j] - m[c * n + i]);
                };
                if (Ms[i] != 0) {
                    if (x > 0) add(i - 1, g.dx);
                    if (x + 1 < g.nx) add(i + 1, g.dx);
                    if (y > 0) add(i - sy, g.dy);
                    if (y + 1 < g.ny) add(i + sy, g.dy);
                    if (z > 0) add(i - sz, g.dz);
                    if (z + 1 < g.nz) add(i + sz, g.dz);
                }
                const double pre = Ms[i] != 0 ? 2 / (mu0 * Ms[i]) : 0.0;
                for (size_t c = 0; c < 3; ++c) H[c * n + i] = pre * h[c];
            }
}

// eq. 9: E = -K (m·k)², so H = 2K/(µ0 Ms) (m·k) k.
void anisotropy_field(const Grid& g, const double* m, const double* Ms, const double* K, const double* k, double* H) {
    const size_t n = g.n();
    for (size_t i = 0; i < n; ++i) {
        const double mk = m[i] * k[i] + m[n + i] * k[n + i] + m[2 * n + i] * k[2 * n + i];
        const double pre = Ms[i] != 0 ? 2 * K[i] / (mu0 * Ms[i]) * mk : 0.0;
        for (size_t c = 0; c < 3; ++c) H[c * n + i] = pre * k[c * n + i];
    }
}

// eq. 11: H = H_ext in magnetic cells.
void zeeman_field(const Grid& g, const double* Ms, const double H_ext[3], double* H) {
    const size_t n = g.n();
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < 3; ++c) H[c * n + i] = Ms[i] != 0 ? H_ext[c] : 0.0;
}
