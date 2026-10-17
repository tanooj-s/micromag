#include <cuda_runtime.h>

#include <stdexcept>

static void check(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

__global__ void scale_kernel(double* x, size_t n, double a) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) x[i] *= a;
}

// ponytail: alloc + copy per call, fine for a smoke test; real kernels keep state resident on the device
void scale_cuda(double* x, size_t n, double a) {
    if (n == 0) return;
    double* d = nullptr;
    check(cudaMalloc(&d, n * sizeof(double)));
    check(cudaMemcpy(d, x, n * sizeof(double), cudaMemcpyHostToDevice));
    unsigned block = 256;
    scale_kernel<<<unsigned((n + block - 1) / block), block>>>(d, n, a);
    check(cudaGetLastError());
    check(cudaMemcpy(x, d, n * sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaFree(d));
}
