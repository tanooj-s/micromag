// The CUDA backend (D24, D25): the CPU loops of fields.cpp, demag.cpp and problem.cpp as one-thread-per-entry kernels,
// demag by cuFFT, and device-wide sums and maxima by CUB. The CPU backend is the oracle (D26).
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "demag.hpp"

namespace {

void check(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err));
}
void check(cufftResult err) {
    if (err != CUFFT_SUCCESS) throw std::runtime_error("cuFFT error " + std::to_string(int(err)));
}

constexpr unsigned block = 256;
unsigned blocks(size_t len) { return unsigned((len + block - 1) / block); }
void check_launch() { check(cudaGetLastError()); }

template <class T>
class DevArray {
public:
    DevArray() = default;
    explicit DevArray(size_t len) { check(cudaMalloc(&p_, len * sizeof(T))); }
    DevArray(DevArray&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    DevArray& operator=(DevArray&& o) noexcept {
        std::swap(p_, o.p_);
        return *this;
    }
    ~DevArray() { cudaFree(p_); }
    T* get() const { return p_; }

private:
    T* p_ = nullptr;
};

__device__ size_t thread_index() { return blockIdx.x * size_t(blockDim.x) + threadIdx.x; }

// ---- field terms: same arithmetic as fields.cpp ----

__device__ double harmonic_mean(double a, double b) { return a + b > 0 ? 2 * a * b / (a + b) : 0.0; }

// eq. 8, D11.
__global__ void exchange_kernel(Grid g, const double* m, const double* Ms, const double* A, double* H) {
    const size_t sy = g.nx, sz = g.nx * g.ny, n = sz * g.nz, i = thread_index();
    if (i >= n) return;
    const size_t x = i % g.nx, y = i / g.nx % g.ny, z = i / sz;
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

// eq. 9.
__global__ void anisotropy_kernel(size_t n, const double* m, const double* Ms, const double* K, const double* k, double* H) {
    const size_t i = thread_index();
    if (i >= n) return;
    const double mk = m[i] * k[i] + m[n + i] * k[n + i] + m[2 * n + i] * k[2 * n + i];
    const double pre = Ms[i] != 0 ? 2 * K[i] / (mu0 * Ms[i]) * mk : 0.0;
    for (size_t c = 0; c < 3; ++c) H[c * n + i] = pre * k[c * n + i];
}

struct Vec3 {
    double v[3];
};

// eq. 11.
__global__ void zeeman_kernel(size_t n, const double* Ms, Vec3 H_ext, double* H) {
    const size_t i = thread_index();
    if (i >= n) return;
    for (size_t c = 0; c < 3; ++c) H[c * n + i] = Ms[i] != 0 ? H_ext.v[c] : 0.0;
}

// ---- demag: eq. 21 by cuFFT, as Demag::field in demag.cpp ----

struct Padded {
    size_t Px, Py, P;
};

__device__ size_t padded_index(const Grid& g, Padded p, size_t i) {  // cell index (z, y, x) -> padded index
    const size_t x = i % g.nx, y = i / g.nx % g.ny, z = i / (g.nx * g.ny);
    return (z * p.Py + y) * p.Px + x;
}

__global__ void scatter_kernel(Grid g, Padded p, const double* m, const double* Ms, double* real) {
    const size_t n = g.nx * g.ny * g.nz, j = thread_index();
    if (j >= 3 * n) return;
    const size_t c = j / n, i = j % n;
    real[c * p.P + padded_index(g, p, i)] = Ms[i] * m[j];
}

__global__ void gather_kernel(Grid g, Padded p, const double* real, const double* Ms, double* H) {
    const size_t n = g.nx * g.ny * g.nz, j = thread_index();
    if (j >= 3 * n) return;
    const size_t c = j / n, i = j % n;
    H[j] = Ms[i] != 0 ? real[c * p.P + padded_index(g, p, i)] / double(p.P) : 0.0;  // cuFFT does not normalise
}

__device__ cuDoubleComplex neg_sum(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    const cuDoubleComplex s = cuCadd(cuCadd(a, b), c);
    return make_cuDoubleComplex(-cuCreal(s), -cuCimag(s));
}

// H(k) = -N(k) M(k); K[c * Pc] is component c in the order xx, yy, zz, xy, xz, yz.
__global__ void multiply_kernel(size_t Pc, const cuDoubleComplex* K, cuDoubleComplex* spec) {
    const size_t q = thread_index();
    if (q >= Pc) return;
    const cuDoubleComplex Mx = spec[q], My = spec[Pc + q], Mz = spec[2 * Pc + q];
    const cuDoubleComplex xx = K[q], yy = K[Pc + q], zz = K[2 * Pc + q], xy = K[3 * Pc + q], xz = K[4 * Pc + q], yz = K[5 * Pc + q];
    spec[q] = neg_sum(cuCmul(xx, Mx), cuCmul(xy, My), cuCmul(xz, Mz));
    spec[Pc + q] = neg_sum(cuCmul(xy, Mx), cuCmul(yy, My), cuCmul(yz, Mz));
    spec[2 * Pc + q] = neg_sum(cuCmul(xz, Mx), cuCmul(yz, My), cuCmul(zz, Mz));
}

struct Tensor6 {
    double N[6];
};

// A 1×1×1 grid has nothing to transform: H = -N(0) Ms m.
__global__ void single_cell_kernel(Tensor6 t, const double* m, const double* Ms, double* H) {
    const double* N = t.N;
    const double Mx = Ms[0] * m[0], My = Ms[0] * m[1], Mz = Ms[0] * m[2];
    const bool vac = Ms[0] == 0;
    H[0] = vac ? 0.0 : -(N[0] * Mx + N[3] * My + N[4] * Mz);
    H[1] = vac ? 0.0 : -(N[3] * Mx + N[1] * My + N[5] * Mz);
    H[2] = vac ? 0.0 : -(N[4] * Mx + N[5] * My + N[2] * Mz);
}

class CudaDemag {
public:
    // The tensor is built on the CPU (D14) and transformed on the device with the same cuFFT layout as M (D25).
    // Axes of length 1 are left out of the transform, so the halved axis is the last one longer than 1.
    explicit CudaDemag(const Grid& g)
        : g_(g), pad_{padded(g.nx), padded(g.ny), padded(g.nx) * padded(g.ny) * padded(g.nz)}, N_(demag_tensor(g)) {
        for (size_t a : {padded(g.nz), padded(g.ny), padded(g.nx)})
            if (a > 1) dims_.push_back(int(a));
        if (dims_.empty()) return;
        Pc_ = pad_.P / dims_.back() * (dims_.back() / 2 + 1);
        const int rank = int(dims_.size());
        real_ = DevArray<double>(3 * pad_.P);
        spec_ = DevArray<cuDoubleComplex>(3 * Pc_);
        kernel_ = DevArray<cuDoubleComplex>(6 * Pc_);

        DevArray<double> N(6 * pad_.P);
        check(cudaMemcpy(N.get(), N_.data(), 6 * pad_.P * sizeof(double), cudaMemcpyHostToDevice));
        cufftHandle plan;
        check(cufftPlanMany(&plan, rank, dims_.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 6));
        const cufftResult r = cufftExecD2Z(plan, N.get(), kernel_.get());
        cufftDestroy(plan);
        check(r);
        check(cufftPlanMany(&fwd_, rank, dims_.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 3));
        check(cufftPlanMany(&inv_, rank, dims_.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_Z2D, 3));
    }
    CudaDemag(const CudaDemag&) = delete;
    CudaDemag& operator=(const CudaDemag&) = delete;
    ~CudaDemag() {
        if (fwd_) cufftDestroy(fwd_);
        if (inv_) cufftDestroy(inv_);
    }

    void field(const double* m, const double* Ms, double* H) {
        const size_t n = g_.n();
        if (dims_.empty()) {
            Tensor6 t;
            std::copy(N_.begin(), N_.end(), t.N);
            single_cell_kernel<<<1, 1>>>(t, m, Ms, H);
            return check_launch();
        }
        check(cudaMemset(real_.get(), 0, 3 * pad_.P * sizeof(double)));
        scatter_kernel<<<blocks(3 * n), block>>>(g_, pad_, m, Ms, real_.get());
        check_launch();
        check(cufftExecD2Z(fwd_, real_.get(), spec_.get()));
        multiply_kernel<<<blocks(Pc_), block>>>(Pc_, kernel_.get(), spec_.get());
        check_launch();
        check(cufftExecZ2D(inv_, spec_.get(), real_.get()));
        gather_kernel<<<blocks(3 * n), block>>>(g_, pad_, real_.get(), Ms, H);
        check_launch();
    }

private:
    Grid g_;
    Padded pad_;
    std::vector<double> N_;  // host tensor; only the 1×1×1 case reads it after construction
    std::vector<int> dims_;
    size_t Pc_ = 1;
    cufftHandle fwd_ = 0, inv_ = 0;
    DevArray<double> real_;
    DevArray<cuDoubleComplex> spec_, kernel_;
};

// ---- vector arithmetic for the drivers: same arithmetic as CpuBackend in problem.cpp ----

__global__ void ms_dot_kernel(size_t n, const double* Ms, const double* m, const double* H, double* w) {
    const size_t i = thread_index();
    if (i < n) w[i] = Ms[i] * (m[i] * H[i] + m[n + i] * H[n + i] + m[2 * n + i] * H[2 * n + i]);
}

struct Combo {
    double c[7];
    const double* k[7];
    int count;
};

__global__ void lincomb_kernel(size_t N, double* out, const double* base, double h, Combo cb) {
    const size_t i = thread_index();
    if (i >= N) return;
    double acc = 0;
    for (int j = 0; j < cb.count; ++j) acc += cb.c[j] * cb.k[j][i];
    out[i] = base ? base[i] + h * acc : h * acc;
}

__global__ void norm2_kernel(size_t n, const double* v, double* w) {
    const size_t i = thread_index();
    if (i < n) w[i] = v[i] * v[i] + v[n + i] * v[n + i] + v[2 * n + i] * v[2 * n + i];
}

__global__ void normalise_kernel(size_t n, double* out, const double* v) {
    const size_t i = thread_index();
    if (i >= n) return;
    const double x = v[i], y = v[n + i], z = v[2 * n + i];
    const double inv = 1 / sqrt(x * x + y * y + z * z);
    out[i] = x * inv;
    out[n + i] = y * inv;
    out[2 * n + i] = z * inv;
}

// eq. 25.
__global__ void llg_rhs_kernel(size_t n, const double* m, const double* H, double alpha, double g, double* dm) {
    const size_t i = thread_index();
    if (i >= n) return;
    const double mx = m[i], my = m[n + i], mz = m[2 * n + i];
    const double Hx = H[i], Hy = H[n + i], Hz = H[2 * n + i];
    const double px = my * Hz - mz * Hy, py = mz * Hx - mx * Hz, pz = mx * Hy - my * Hx;  // m × H
    const double dx = my * pz - mz * py, dy = mz * px - mx * pz, dz = mx * py - my * px;  // m × (m × H)
    dm[i] = -g * (px + alpha * dx);
    dm[n + i] = -g * (py + alpha * dy);
    dm[2 * n + i] = -g * (pz + alpha * dz);
}

// eq. 22; w1 = |m × H|, w2 = |m × H| / Ms in magnetic cells, 0 in vacuum.
__global__ void descent_kernel(size_t n, const double* m, const double* H, const double* Ms, double* d, double* w1, double* w2) {
    const size_t i = thread_index();
    if (i >= n) return;
    const double mx = m[i], my = m[n + i], mz = m[2 * n + i];
    const double px = my * H[2 * n + i] - mz * H[n + i], py = mz * H[i] - mx * H[2 * n + i], pz = mx * H[n + i] - my * H[i];
    d[i] = -(my * pz - mz * py);
    d[n + i] = -(mz * px - mx * pz);
    d[2 * n + i] = -(mx * py - my * px);
    const double t = sqrt(px * px + py * py + pz * pz);
    w1[i] = t;
    w2[i] = Ms[i] != 0 ? t / Ms[i] : 0.0;
}

__global__ void bb_kernel(size_t N, const double* m, const double* m_prev, const double* d, const double* d_prev, double* ss,
                          double* sy, double* yy) {
    const size_t i = thread_index();
    if (i >= N) return;
    const double s = m[i] - m_prev[i], y = d_prev[i] - d[i];
    ss[i] = s * s;
    sy[i] = s * y;
    yy[i] = y * y;
}

class CudaBackend : public Backend {
public:
    CudaBackend(const Grid& g, const double* Ms, const double* A, const double* K, const double* k)
        : g_(g), n_(g.n()), Ms_(n_), A_(n_), K_(n_), k_(3 * n_), w1_(3 * n_), w2_(3 * n_), w3_(3 * n_), out_(1), demag_(g) {
        to_device(Ms_.get(), Ms, n_);
        to_device(A_.get(), A, n_);
        to_device(K_.get(), K, n_);
        to_device(k_.get(), k, 3 * n_);
    }

    double* alloc(size_t len) override {
        double* p = nullptr;
        check(cudaMalloc(&p, len * sizeof(double)));
        return p;
    }
    void release(double* p) override { cudaFree(p); }
    void to_device(double* dst, const double* host, size_t len) override {
        check(cudaMemcpy(dst, host, len * sizeof(double), cudaMemcpyHostToDevice));
    }
    void to_host(double* host, const double* src, size_t len) override {
        check(cudaMemcpy(host, src, len * sizeof(double), cudaMemcpyDeviceToHost));
    }
    void copy(double* dst, const double* src, size_t len) override {
        check(cudaMemcpy(dst, src, len * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    void field(Term t, const double* m, const std::array<double, 3>& H_ext, double* H) override {
        switch (t) {
            case Term::exchange: exchange_kernel<<<blocks(n_), block>>>(g_, m, Ms_.get(), A_.get(), H); break;
            case Term::anisotropy: anisotropy_kernel<<<blocks(n_), block>>>(n_, m, Ms_.get(), K_.get(), k_.get(), H); break;
            case Term::zeeman: zeeman_kernel<<<blocks(n_), block>>>(n_, Ms_.get(), Vec3{{H_ext[0], H_ext[1], H_ext[2]}}, H); break;
            case Term::demag: return demag_.field(m, Ms_.get(), H);
        }
        check_launch();
    }

    double ms_dot(const double* m, const double* H) override {
        ms_dot_kernel<<<blocks(n_), block>>>(n_, Ms_.get(), m, H, w1_.get());
        check_launch();
        return sum(w1_.get(), n_);
    }

    void lincomb(double* out, const double* base, double h, const double* c, const double* const* k, int count) override {
        if (count > 7) throw std::invalid_argument("lincomb: at most 7 terms");
        Combo cb{};
        for (int j = 0; j < count; ++j) cb.c[j] = c[j], cb.k[j] = k[j];
        cb.count = count;
        lincomb_kernel<<<blocks(3 * n_), block>>>(3 * n_, out, base, h, cb);
        check_launch();
    }

    double max_norm(const double* v) override {
        norm2_kernel<<<blocks(n_), block>>>(n_, v, w1_.get());
        check_launch();
        return std::sqrt(max(w1_.get(), n_));
    }

    void normalise(double* out, const double* v) override {
        normalise_kernel<<<blocks(n_), block>>>(n_, out, v);
        check_launch();
    }

    void llg_rhs(const double* m, const double* H, double alpha, double g, double* dm) override {
        llg_rhs_kernel<<<blocks(n_), block>>>(n_, m, H, alpha, g, dm);
        check_launch();
    }

    std::array<double, 2> descent(const double* m, const double* H, double* d) override {
        descent_kernel<<<blocks(n_), block>>>(n_, m, H, Ms_.get(), d, w1_.get(), w2_.get());
        check_launch();
        return {max(w1_.get(), n_), max(w2_.get(), n_)};
    }

    std::array<double, 3> bb_dots(const double* m, const double* m_prev, const double* d, const double* d_prev) override {
        bb_kernel<<<blocks(3 * n_), block>>>(3 * n_, m, m_prev, d, d_prev, w1_.get(), w2_.get(), w3_.get());
        check_launch();
        return {sum(w1_.get(), 3 * n_), sum(w2_.get(), 3 * n_), sum(w3_.get(), 3 * n_)};
    }

private:
    // Device-wide reductions by CUB (D25). Their summation order differs from the CPU loops (D26 tolerances).
    double sum(const double* in, size_t len) {
        size_t bytes = 0;
        check(cub::DeviceReduce::Sum(nullptr, bytes, in, out_.get(), int(len)));
        check(cub::DeviceReduce::Sum(temp(bytes), bytes, in, out_.get(), int(len)));
        return result();
    }
    double max(const double* in, size_t len) {
        size_t bytes = 0;
        check(cub::DeviceReduce::Max(nullptr, bytes, in, out_.get(), int(len)));
        check(cub::DeviceReduce::Max(temp(bytes), bytes, in, out_.get(), int(len)));
        return result();
    }
    void* temp(size_t bytes) {  // grows once, then reused
        if (bytes > temp_bytes_) {
            temp_ = DevArray<char>(bytes);
            temp_bytes_ = bytes;
        }
        return temp_.get();
    }
    double result() {
        double r;
        check(cudaMemcpy(&r, out_.get(), sizeof(double), cudaMemcpyDeviceToHost));
        return r;
    }

    Grid g_;
    size_t n_;
    DevArray<double> Ms_, A_, K_, k_, w1_, w2_, w3_, out_;  // w*: per-entry scratch for the reductions
    DevArray<char> temp_;
    size_t temp_bytes_ = 0;
    CudaDemag demag_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(const Grid& g, const double* Ms, const double* A, const double* K, const double* k) {
    return std::make_unique<CudaBackend>(g, Ms, A, K, k);
}
