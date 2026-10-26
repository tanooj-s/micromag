#pragma once

// Counter-based random numbers shared by the CPU and CUDA backends (D41), so the CPU stays the oracle for the noise.

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define MM_HD __host__ __device__
#else
#define MM_HD
#endif

struct Philox4 {
    uint32_t v[4];
};

// Philox4x32-10 (Salmon et al. 2011), as Random123's philox4x32: 4 counter words, 2 key words.
MM_HD inline Philox4 philox4x32(Philox4 c, uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) k0 += 0x9E3779B9, k1 += 0xBB67AE85;  // the Weyl key schedule
        const uint64_t p0 = uint64_t(0xD2511F53) * c.v[0], p1 = uint64_t(0xCD9E8D57) * c.v[2];
        c = {{uint32_t(p1 >> 32) ^ c.v[1] ^ k0, uint32_t(p1), uint32_t(p0 >> 32) ^ c.v[3] ^ k1, uint32_t(p0)}};
    }
    return c;
}

// A uniform in (0, 1) from 53 bits of two words; never 0, so its log is finite.
MM_HD inline double uniform53(uint32_t hi, uint32_t lo) { return ((((uint64_t(hi) << 32) | lo) >> 11) + 0.5) * 0x1p-53; }

// Three standard normals for one cell: two Philox blocks (counter (cell, block, draw)), Box–Muller on 53-bit uniforms.
MM_HD inline void normal3(uint32_t seed, uint32_t stage, uint64_t draw, uint32_t cell, double out[3]) {
    const double two_pi = 6.283185307179586;
    for (uint32_t block = 0; block < 2; ++block) {
        const Philox4 r = philox4x32({{cell, block, uint32_t(draw), uint32_t(draw >> 32)}}, seed, stage);
        const double rho = sqrt(-2 * log(uniform53(r.v[0], r.v[1]))), phi = two_pi * uniform53(r.v[2], r.v[3]);
        out[2 * block] = rho * cos(phi);
        if (block == 0) out[1] = rho * sin(phi);  // block 1 needs only one of its pair
    }
}
