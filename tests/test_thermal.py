"""Iteration 7: Brown's thermal field (D41), the stochastic integrators (D42) and their checks (D44).

Statistical checks run with fixed seeds, so they are deterministic; each tolerance is 4σ of the estimate. The ensembles
are uncoupled macrospins (A = 0, no demag, D36): every cell is an independent sample.
"""
from functools import lru_cache

import numpy as np
import pytest

import micromag as mm
from test_fields import H_EXT, random_problem
from test_llg import GAMMA, macrospin_error, period

G0 = mm.mu0 * GAMMA  # γ0 = µ0|γ|, m/(A·s)
T = 300.0
KT = mm.kB * T
CELL = 4e-9
V = CELL**3
MS = 8e5
MV = mm.mu0 * MS * V
METHODS = ["heun", "rk23"]

# Random123 tests/kat_vectors, "philox4x32 10": counter, key -> output.
KAT = [
    ([0, 0, 0, 0], [0, 0], [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
    ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2, [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]),
    ([0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344], [0xA4093822, 0x299F31D0], [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]),
]


def test_philox_matches_known_answers():
    for counter, key, want in KAT:
        assert mm.philox(counter, key) == want


# ---- (a) the noise itself ----

NOISE_ALPHA, NOISE_DT = 0.5, 1e-13


def noise_llg(device, seed=7, stage=1):
    """32×32×8 cells in two materials (Ms and 2 Ms) with a vacuum layer."""
    g = mm.Grid(32, 32, 8, CELL, CELL, CELL)
    s = (8, 32, 32)
    Ms = np.full(s, MS)
    Ms[:, :, :16] = 2 * MS
    Ms[0] = 0
    zero, k = np.zeros(s), np.zeros((3, *s))
    prob = mm.Problem(g, Ms, zero, zero, k, (0, 0, 0), device=device, demag=False)
    return Ms, mm.LLG(prob, NOISE_ALPHA, method="heun", temperature=T, dt=NOISE_DT, seed=seed, stage=stage)


def draws(llg, count, start=0):
    H = np.empty((count, 3, 8, 32, 32))
    for d in range(count):
        llg.thermal_field(start + d, H[d])
    return H


def test_thermal_field_has_brown_variance_and_no_correlations(device):
    Ms, llg = noise_llg(device)
    H = draws(llg, 8)
    assert np.all(H[:, :, Ms == 0] == 0)  # no noise in vacuum (D41)
    sigma2 = 2 * NOISE_ALPHA * KT / (mm.mu0 * G0 * Ms * V * NOISE_DT)  # D41: Brown's variance, γ0 = µ0|γ|
    for ms in (MS, 2 * MS):  # the per-cell Ms enters as 1/Ms, so each material is checked on its own
        z = H[:, :, Ms == ms] / np.sqrt(sigma2[Ms == ms][0])
        N = z.size
        assert abs(z.mean()) < 4 / np.sqrt(N)
        assert abs((z**2).mean() - 1) < 4 * np.sqrt(2 / N)
        assert abs((z**4).mean() - 3) < 4 * np.sqrt(96 / N)  # Gaussian: a uniform or clipped draw fails this
        for name, (a, b) in {"components": (z[:, 0], z[:, 1]), "cells": (z[..., :-1], z[..., 1:]),
                             "draws": (z[:-1], z[1:])}.items():
            assert abs((a * b).mean()) < 4 / np.sqrt(a.size), name


def test_streams_are_reproducible_and_independent(device):
    _, llg = noise_llg(device)
    H = draws(llg, 2)
    assert np.abs(H).max() > 0
    assert np.array_equal(draws(noise_llg(device)[1], 2), H)  # same (seed, stage, draw): the same field
    for other in (noise_llg(device, seed=8)[1], noise_llg(device, stage=2)[1]):
        G = draws(other, 2)
        r = (H * G).sum() / np.sqrt((H * H).sum() * (G * G).sum())
        assert abs(r) < 4 / np.sqrt(np.count_nonzero(H)), r


# ---- order at T = 0 against the D20 analytic macrospin ----

@pytest.mark.parametrize("method, lo, hi", [("heun", 3.4, 4.6), ("rk23", 6.8, 9.2)])
def test_fixed_steps_converge_at_their_order(method, lo, hi, device):
    # rk23 with tol = ∞ never rejects and max_step pins every step to h, as the D20 RK45 check.
    Tp = period(0.1)
    args = (lambda n: dict(method="heun", dt=Tp / n)) if method == "heun" else (lambda n: dict(method="rk23", tol=np.inf, max_step=Tp / n))
    errs = np.array([macrospin_error(0.1, Tp, 1, device, **args(n)) for n in (16, 32, 64)])
    ratios = errs[:-1] / errs[1:]
    assert np.all((lo < ratios) & (ratios < hi)), (errs, ratios)


def test_noise_needs_a_stochastic_method():
    prob = mm.Problem(mm.Grid(1, 1, 1, CELL, CELL, CELL), np.full((1, 1, 1), MS), np.zeros((1, 1, 1)), np.zeros((1, 1, 1)),
                      np.zeros((3, 1, 1, 1)), (0, 0, 0))
    with pytest.raises(ValueError, match="rk45"):
        mm.LLG(prob, 0.5, temperature=T)
    with pytest.raises(ValueError, match="dt"):
        mm.LLG(prob, 0.5, method="heun", temperature=T)


# ---- (b), (c): the Boltzmann ensemble ----

def tau_N(alpha):
    """Néel–Brown time (s): ⟨m_z⟩ of a free macrospin decays as exp(−t/τ_N)."""
    return (1 + alpha**2) * MV / (2 * alpha * G0 * KT)


def ensemble(n, device, alpha, barrier=0.0, xi=0.0, seed=1, **llg):
    """n³ uncoupled macrospins: KV/kT = barrier along z, and µ0 Ms V H / kT = xi with H along z."""
    g = mm.Grid(n, n, n, CELL, CELL, CELL)
    one = np.ones((n, n, n))
    k = np.zeros((3, n, n, n))
    k[2] = 1
    prob = mm.Problem(g, MS * one, 0 * one, barrier * KT / V * one, k, (0, 0, xi * KT / MV), device=device, demag=False)
    return mm.LLG(prob, alpha, temperature=T, seed=seed, **llg)


def isotropic(n, seed=0):
    m = np.random.default_rng(seed).normal(size=(3, n, n, n))
    return m / np.linalg.norm(m, axis=0)


def method_args(method, alpha):
    # dt and tol keep the integrator's bias below 1σ of these checks (D44); see the D44 Result.
    return dict(method="heun", dt=tau_N(alpha) / 200) if method == "heun" else dict(method="rk23", tol=1e-3)


@lru_cache(maxsize=None)
def equilibrium(method, device, alpha, barrier, xi, n=16, burn=5, samples=100, every=0.5):
    """m_z of every cell at `samples` times, every `every` τ_N after `burn` τ_N, from an isotropic start: (samples, n³)."""
    llg = ensemble(n, device, alpha, barrier, xi, **method_args(method, alpha))
    m = isotropic(n)
    tau = tau_N(alpha)
    llg.run_until(m, burn * tau)
    out = np.empty((samples, n**3))
    for i in range(samples):
        llg.run_until(m, (burn + (i + 1) * every) * tau)
        out[i] = m[2].ravel()
    assert np.abs(np.linalg.norm(m, axis=0) - 1).max() < 1e-12
    return out


def cell_mean(values):
    """Mean over cells of each cell's time average, and its 1σ error: cells are independent, samples in time are not."""
    per_cell = values.mean(axis=0)
    return per_cell.mean(), per_cell.std(ddof=1) / np.sqrt(per_cell.size)


EDGES = np.linspace(-1, 1, 21)


def boltzmann_bins(barrier):
    """P(cos θ) ∝ exp(KV cos²θ / kT): each bin's probability, by the trapezoid rule on 1000 points per bin."""
    x = np.linspace(-1, 1, 20 * 1000 + 1)
    p = np.exp(barrier * x**2)
    cum = np.concatenate([[0], np.cumsum((p[:-1] + p[1:]) / 2 * np.diff(x))])
    edges = cum[::1000]
    return np.diff(edges) / edges[-1]


@pytest.mark.parametrize("method", METHODS)
def test_ensemble_reaches_the_boltzmann_distribution(method, device):
    mz = equilibrium(method, device, 1.0, 2.0, 0.0)
    want = boltzmann_bins(2.0)
    for b, p in enumerate(want):
        inside = (EDGES[b] <= mz) & (mz < EDGES[b + 1])
        got, sigma = cell_mean(inside.astype(float))
        assert abs(got - p) < 4 * sigma, (b, got, p, sigma)


@pytest.mark.parametrize("method", METHODS)
def test_free_spins_follow_the_langevin_curve(method, device):
    xi = 2.0
    got, sigma = cell_mean(equilibrium(method, device, 1.0, 0.0, xi))
    want = 1 / np.tanh(xi) - 1 / xi
    assert abs(got - want) < 4 * sigma, (got, want, sigma)


@pytest.mark.parametrize("method", METHODS)
def test_equilibrium_does_not_depend_on_damping(method, device):
    # Thick or thin honey: the same spread, reached at a different speed. Catches α missing from Brown's variance.
    (a, sa), (b, sb) = (cell_mean(equilibrium(method, device, alpha, 2.0, 0.0) ** 2) for alpha in (0.1, 1.0))
    assert min(abs(a - 1 / 3) / sa, abs(b - 1 / 3) / sb) > 8  # both left the isotropic start (⟨cos²θ⟩ = 1/3)
    assert abs(a - b) < 4 * np.hypot(sa, sb), (a, b, sa, sb)


# ---- the Brownian bridge (D42) ----

def test_single_macrospin_with_rejections_reaches_boltzmann(device):
    # The ensemble tests hide what rk23 does on a rejection: there the worst of 4096 cells triggers it. A lone
    # macrospin rejects about 40% of its steps; drawing fresh noise on rejection gives ⟨cos²θ⟩ ≈ 0.38 (D44 Result).
    llg = ensemble(1, device, 1.0, barrier=2.0, method="rk23", tol=1e-4)
    m = isotropic(1)
    tau, samples = tau_N(1.0), 100_000
    x2 = np.empty(samples)
    for i in range(samples):
        llg.run_until(m, (i + 1) * tau)
        x2[i] = m[2, 0, 0, 0] ** 2
    blocks = x2.reshape(100, -1).mean(axis=1)  # 1000 τ_N each: much longer than the correlation time
    got, sigma = blocks.mean(), blocks.std(ddof=1) / np.sqrt(blocks.size)
    x = np.linspace(-1, 1, 200_001)
    p = np.exp(2.0 * x**2)
    want = np.trapezoid(x**2 * p, x) / np.trapezoid(p, x)
    assert llg.rejected > llg.steps / 4
    assert abs(got - want) < 4 * sigma, (got, want, sigma)


def test_bridge_keeps_the_wiener_increments_exact(device):
    # One macrospin with a tight tol rejects many steps, each split by the bridge. The path's increments over fixed
    # intervals must still be independent N(0, Δ). Redrawing on rejection keeps only the small kicks: the variance
    # falls to about 0.6 (D44 Result). On a large grid the rejections follow the worst of many cells, which hides it.
    llg = ensemble(1, device, 1.0, barrier=2.0, method="rk23", tol=1e-4)
    m = isotropic(1)
    d, blocks = tau_N(1.0) / 200, 10_000  # about one step per interval, so most intervals end inside a split entry
    W = np.zeros((blocks + 1, 3))
    w = np.empty_like(m)
    for i in range(blocks):
        llg.run_until(m, (i + 1) * d)
        llg.wiener(w)
        W[i + 1] = w.ravel()
    assert llg.rejected > 1000 and llg.max_bridge_depth >= 2, (llg.steps, llg.rejected, llg.max_bridge_depth)
    z = np.diff(W, axis=0).ravel() / np.sqrt(d)
    N = z.size
    assert abs(z.mean()) < 4 / np.sqrt(N)
    assert abs((z**2).mean() - 1) < 4 * np.sqrt(2 / N), (z**2).mean()


# ---- (d) the escape rate against the Fokker–Planck operator ----

def fokker_planck_rate(barrier, order=80):
    """λ1 τ_N: the smallest nonzero decay rate of the uniaxial Fokker–Planck operator (Brown 1963), in units of 1/τ_N.

    2 τ_N ∂W/∂t = ∂x[(1 − x²)(∂x W − 2 σ x W)], x = cos θ, σ = KV/kT, solved by Galerkin on Legendre polynomials.
    """
    L = np.polynomial.legendre.Legendre
    x, w = np.polynomial.legendre.leggauss(order + 20)
    P = np.array([L.basis(k)(x) for k in range(order)])
    dP = np.array([L.basis(k).deriv()(x) for k in range(order)])
    k = np.arange(order)
    G = np.diag(-k * (k + 1) * 2 / (2 * k + 1)) + 2 * barrier * (dP * w) @ (x * (1 - x**2) * P).T
    rates = np.sort(-np.linalg.eigvals((2 * k + 1)[:, None] / 2 * G).real)
    assert abs(rates[0]) < 1e-8  # the equilibrium
    return rates[1] / 2


def test_fokker_planck_oracle():
    assert abs(fokker_planck_rate(0.0) - 1) < 1e-12  # free diffusion: ⟨m_z⟩ decays as exp(−t/τ_N)
    s = 20.0  # Brown's high-barrier asymptote, λ1 τ_N ≈ (2/√π) σ^{3/2} e^{−σ}, good to O(1/σ)
    assert abs(fokker_planck_rate(s) / (2 / np.sqrt(np.pi) * s**1.5 * np.exp(-s)) - 1) < 2 / s


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_escape_rate_matches_fokker_planck(method, device):
    barrier, alpha, n, groups = 3.0, 1.0, 32, 8
    rate = fokker_planck_rate(barrier)
    tau = tau_N(alpha)
    llg = ensemble(n, device, alpha, barrier, **method_args(method, alpha))
    m = np.zeros((3, n, n, n))
    m[2] = 1
    times = np.arange(1, 2 / rate, 0.25)  # in τ_N; from 1 τ_N the faster modes have died away, and ⟨m_z⟩ stays ≥ 0.1
    mz = np.empty((times.size, groups))
    for i, t in enumerate(times):
        llg.run_until(m, t * tau)
        mz[i] = m[2].reshape(groups, -1).mean(axis=1)
    assert mz.min() > 0.05
    # Each group's rate from log ⟨m_z⟩, weighted by ⟨m_z⟩: the error of a log is σ / ⟨m_z⟩.
    fits = np.array([-np.polyfit(times, np.log(g), 1, w=g)[0] for g in mz.T])
    got, sigma = fits.mean(), fits.std(ddof=1) / np.sqrt(groups)
    print(f"\n{method}: escape rate {got:.4f} ± {sigma:.4f} /τ_N, Fokker–Planck {rate:.4f} /τ_N (KV/kT = {barrier})")
    assert abs(got - rate) < 4 * sigma


# ---- CPU = CUDA (D26) ----

@pytest.mark.skipif(not mm.has_cuda, reason="CPU-only build")
def test_noise_and_thermal_trajectories_match_cpu():
    a, b = (draws(noise_llg(device)[1], 2) for device in ("cpu", "cuda"))
    assert np.abs(a).max() > 0
    assert np.abs(b - a).max() <= 1e-14 * np.abs(a).max()

    # Fixed steps on both devices, so only rounding differs (D26): heun, and rk23 with every step pinned to h.
    p = random_problem()
    H = np.empty_like(p["m"])
    mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT).effective_field(p["m"], H)
    h = 2 * np.pi / (mm.mu0 * GAMMA * np.abs(H).max()) / 1000
    for args in (dict(method="heun", dt=h), dict(method="rk23", tol=np.inf, max_step=h)):
        out = []
        for device in ("cpu", "cuda"):
            prob = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device)
            m = p["m"].copy()
            mm.LLG(prob, 0.5, temperature=T, seed=3, **args).run_until(m, 100 * h)
            out.append(m)
        assert np.abs(out[0] - p["m"]).max() > 1e-3, args
        assert np.abs(out[1] - out[0]).max() <= 1e-10, args
