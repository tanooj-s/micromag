import numpy as np
import pytest

import micromag as mm

H_EXT = np.array([1e5, -2e5, 3e5])
C = {"exchange": 0.5, "anisotropy": 0.5, "zeeman": 1.0, "demag": 0.5}  # energy prefactor per term (D10)


def random_problem(seed=0):
    """Small grid with unequal cell sizes, mixed materials and some vacuum cells."""
    rng = np.random.default_rng(seed)
    g = mm.Grid(5, 4, 3, 2e-9, 3e-9, 4e-9)
    s = (g.nz, g.ny, g.nx)
    m = rng.normal(size=(3, *s))
    m /= np.linalg.norm(m, axis=0)
    k = rng.normal(size=(3, *s))
    k /= np.linalg.norm(k, axis=0)
    Ms = rng.uniform(0.5e6, 1.5e6, s)
    Ms[rng.random(s) < 0.2] = 0.0
    assert (Ms == 0).any()
    A = rng.uniform(5e-12, 15e-12, s)
    K = rng.uniform(1e6, 5e6, s)
    return dict(grid=g, m=m, Ms=Ms, A=A, K=K, k=k, demag=mm.Demag(g))


def field(term, p, m, device="cpu"):
    """One term's field: on "cpu" through the free functions (the oracle), otherwise through Problem.field (D24)."""
    out = np.empty_like(m)
    if device != "cpu":
        if device not in p:  # one Problem per device, reused across calls
            p[device] = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device)
        p[device].field(term, m, out)
    elif term == "exchange":
        mm.exchange_field(p["grid"], m, p["Ms"], p["A"], out)
    elif term == "anisotropy":
        mm.anisotropy_field(p["grid"], m, p["Ms"], p["K"], p["k"], out)
    elif term == "demag":
        p["demag"].field(m, p["Ms"], out)
    else:
        mm.zeeman_field(p["grid"], p["Ms"], H_EXT, out)
    return out


def energy(term, p, m, device="cpu"):
    return mm.energy(p["grid"], m, p["Ms"], field(term, p, m, device), C[term])


def test_uniform_state_energies():
    p = random_problem()
    g, Ms = p["grid"], p["Ms"]
    dV = g.dx * g.dy * g.dz
    u = np.empty_like(p["m"])
    u[:] = np.array([0.6, 0.0, 0.8])[:, None, None, None]
    # atol=0 everywhere: energies here are ~1e-20 J, far below numpy's default atol
    assert energy("exchange", p, u) == 0.0  # m_j - m_i is exactly 0, even with mixed A
    assert np.isclose(energy("anisotropy", p, p["k"].copy()), -np.sum(p["K"][Ms > 0]) * dV, rtol=1e-12, atol=0)
    assert np.isclose(energy("zeeman", p, u), -mm.mu0 * dV * np.sum(Ms) * (u[:, 0, 0, 0] @ H_EXT), rtol=1e-12, atol=0)


@pytest.mark.parametrize("term", C)
def test_field_is_minus_energy_gradient(term, device):
    # H = -1/(µ0 Ms dV) ∂E/∂m. E is quadratic (exchange, anisotropy) or linear (Zeeman) in m,
    # so central differences are exact up to rounding. Passing also proves the exchange operator is symmetric.
    p = random_problem()
    g, m = p["grid"], p["m"]
    h = 1e-3
    grad = np.empty_like(m)
    for i in np.ndindex(m.shape):
        mp, mn = m.copy(), m.copy()
        mp[i] += h
        mn[i] -= h
        grad[i] = (energy(term, p, mp, device) - energy(term, p, mn, device)) / (2 * h)
    expected = -mm.mu0 * p["Ms"] * g.dx * g.dy * g.dz * field(term, p, m, device)
    assert np.linalg.norm(expected) > 0  # an all-zero field would otherwise pass trivially
    assert np.linalg.norm(grad - expected) <= 1e-7 * np.linalg.norm(expected)


@pytest.mark.parametrize("term", C)
def test_vacuum_cells_have_zero_field(term, device):
    p = random_problem()
    H = field(term, p, p["m"], device)
    assert np.all(H[:, p["Ms"] == 0] == 0)


A_W, K_W, MS_W = 7.7e-12, 4.3e6, 1.28e6  # NdFeB-like
DELTA = np.sqrt(A_W / K_W)  # wall width is π·δ0


def bloch_errors(cells_per_delta, device):
    """Place the exact Bloch wall on a 40·δ0 line; return (max torque / H_K, relative wall-energy error)."""
    dx = DELTA / cells_per_delta
    nx = 40 * cells_per_delta
    g = mm.Grid(nx, 1, 1, dx, dx, dx)
    x = (np.arange(nx) + 0.5 - nx / 2) * dx
    theta = 2 * np.arctan(np.exp(x / DELTA))
    m = np.zeros((3, 1, 1, nx))
    m[1, 0, 0], m[2, 0, 0] = np.sin(theta), np.cos(theta)  # rotates in the y-z plane, from +z to -z
    Ms, A, K = (np.full((1, 1, nx), v) for v in (MS_W, A_W, K_W))
    k = np.zeros_like(m)
    k[2] = 1.0
    p = dict(grid=g, Ms=Ms, A=A, K=K, k=k)
    H_ex, H_an = field("exchange", p, m, device), field("anisotropy", p, m, device)

    H_K = 2 * K_W / (mm.mu0 * MS_W)
    torque = np.linalg.norm(np.cross(m, H_ex + H_an, axis=0), axis=0).max() / H_K
    area = dx * dx
    E = mm.energy(g, m, Ms, H_ex, 0.5) + mm.energy(g, m, Ms, H_an, 0.5)
    sigma = (E + K_W * nx * dx * area) / area  # relative to the uniform state, energy -K·V
    return torque, abs(sigma / (4 * np.sqrt(A_W * K_W)) - 1)


def test_bloch_wall_converges_at_second_order(device):
    # D13: torque -> 0 checks the width π√(A/K); σ -> 4√(AK) checks the energy.
    errs = np.array([bloch_errors(n, device) for n in (4, 8, 16)])  # rows: resolution; cols: torque, energy
    ratios = errs[:-1] / errs[1:]
    assert np.all((3.5 < ratios) & (ratios < 4.5)), ratios
    assert np.all(errs[-1] < 2e-3), errs


def test_binding_rejects_mismatched_arrays():
    p = random_problem()
    g = p["grid"]
    out = np.empty_like(p["m"])
    with pytest.raises(ValueError):  # axes in (x, y, z) order instead of (z, y, x)
        mm.exchange_field(g, np.empty((3, g.nx, g.ny, g.nz)), p["Ms"], p["A"], out)
    with pytest.raises(ValueError):
        p["demag"].field(np.empty((3, g.nx, g.ny, g.nz)), p["Ms"], out)
    with pytest.raises(TypeError):  # implicit conversion would write H into a hidden copy
        mm.exchange_field(g, p["m"], p["Ms"], p["A"], out.astype(np.float32))


@pytest.mark.parametrize("term", C)
def test_problem_field_matches_cpu_oracle(term, device):
    # D26: exchange and anisotropy within 1e-14·max|H| (fused multiply-adds, operation order); Zeeman bitwise (a copy);
    # demag within 1e-12 on the norm (a different FFT). On "cpu", Problem.field calls the oracle itself: bitwise.
    p = random_problem()
    expected = field(term, p, p["m"])
    assert np.abs(expected).max() > 0  # an all-zero field would otherwise pass trivially
    prob = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device)
    H = np.full_like(p["m"], np.nan)
    prob.field(term, p["m"], H)
    if device == "cpu" or term == "zeeman":
        assert np.array_equal(H, expected)
    elif term == "demag":
        assert np.linalg.norm(H - expected) <= 1e-12 * np.linalg.norm(expected)
    else:
        assert np.abs(H - expected).max() <= 1e-14 * np.abs(expected).max()


def test_problem_rejects_unknown_device_and_term():
    p = random_problem()
    with pytest.raises(ValueError):
        mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device="gpu")
    prob = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT)
    with pytest.raises(ValueError):
        prob.field("magnetostatic", p["m"], np.empty_like(p["m"]))


@pytest.mark.skipif(mm.has_cuda, reason="CUDA build")
def test_cuda_device_needs_a_cuda_build():
    p = random_problem()
    with pytest.raises(RuntimeError, match="without CUDA"):
        mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device="cuda")


# ---- D35: exchange scaled per region pair; D36: the demag switch ----

def coupled_problem(seed=1):
    """random_problem plus a region map of ids 0–3 and a random symmetric pair table."""
    p = random_problem(seed)
    rng = np.random.default_rng(seed)
    p["regions"] = rng.integers(0, 4, p["Ms"].shape).astype(np.uint8)
    table = np.ones((256, 256))
    table[:4, :4] = rng.uniform(0, 2, (4, 4))
    p["coupling"] = (table + table.T) / 2
    return p


def coupled_field(p, m, device="cpu", coupling=None):
    out = np.empty_like(m)
    coupling = p["coupling"] if coupling is None else coupling
    if device == "cpu":
        mm.exchange_field(p["grid"], m, p["Ms"], p["A"], out, regions=p["regions"], coupling=coupling)
    else:
        prob = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device, regions=p["regions"], coupling=coupling)
        prob.field("exchange", m, out)
    return out


def test_unit_pair_table_is_bitwise_the_plain_kernel(device):
    p = coupled_problem()
    plain = field("exchange", p, p["m"])
    assert np.array_equal(coupled_field(p, p["m"], device, coupling=np.ones((256, 256))), field("exchange", p, p["m"], device))
    assert np.array_equal(coupled_field(p, p["m"], "cpu", coupling=np.ones((256, 256))), plain)
    assert not np.allclose(coupled_field(p, p["m"], device), plain)  # guard: the random table does change the field


def test_pair_scale_on_a_three_cell_chain(device):
    # Regions 1-2-2 along x: only the 1|2 face is scaled. H_i = 2/(µ0 Ms) Σ_faces w A (m_j - m_i) / dx² (eq. 8, D11, D35).
    g = mm.Grid(3, 1, 1, 2e-9, 3e-9, 4e-9)
    Ms, A, s = 8e5, 1e-11, 0.3
    one = np.ones((1, 1, 3))
    m = np.ascontiguousarray(np.array([[1.0, 0, 0], [0, 1, 0], [0, 0.6, 0.8]]).T.reshape(3, 1, 1, 3))
    regions = np.array([1, 2, 2], np.uint8).reshape(1, 1, 3)
    table = np.ones((256, 256))
    table[1, 2] = table[2, 1] = s
    p = dict(grid=g, Ms=Ms * one, A=A * one, K=0 * one, k=np.zeros((3, 1, 1, 3)), regions=regions, coupling=table)
    pre = 2 * A / (mm.mu0 * Ms * g.dx**2)
    m0, m1, m2 = m[:, 0, 0, 0], m[:, 0, 0, 1], m[:, 0, 0, 2]
    expected = pre * np.stack([s * (m1 - m0), s * (m0 - m1) + (m2 - m1), m1 - m2], axis=1).reshape(3, 1, 1, 3)
    assert np.allclose(coupled_field(p, m, device), expected, rtol=1e-14, atol=0)


def test_coupled_field_is_minus_energy_gradient(device):
    p = coupled_problem()
    g, m = p["grid"], p["m"]
    E = lambda m: mm.energy(g, m, p["Ms"], coupled_field(p, m, device), 0.5)
    h = 1e-3
    grad = np.empty_like(m)
    for i in np.ndindex(m.shape):
        mp, mn = m.copy(), m.copy()
        mp[i] += h
        mn[i] -= h
        grad[i] = (E(mp) - E(mn)) / (2 * h)
    expected = -mm.mu0 * p["Ms"] * g.dx * g.dy * g.dz * coupled_field(p, m, device)
    assert not np.allclose(coupled_field(p, m, device), field("exchange", p, m))  # guard: the table is in use
    assert np.linalg.norm(grad - expected) <= 1e-7 * np.linalg.norm(expected)


def test_coupled_problem_matches_cpu_oracle(device):
    # D26 tolerance; on "cpu" this is Problem against the free function.
    p = coupled_problem()
    expected = coupled_field(p, p["m"])
    assert not np.allclose(expected, field("exchange", p, p["m"]))  # guard: the table is in use
    H = np.empty_like(expected)
    mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device, regions=p["regions"],
               coupling=p["coupling"]).field("exchange", p["m"], H)
    assert np.abs(H - expected).max() <= 1e-14 * np.abs(expected).max()


def test_binding_checks_regions_and_coupling():
    p = coupled_problem()
    out = np.empty_like(p["m"])
    with pytest.raises(ValueError):  # one without the other
        mm.exchange_field(p["grid"], p["m"], p["Ms"], p["A"], out, regions=p["regions"])
    with pytest.raises(ValueError):
        mm.exchange_field(p["grid"], p["m"], p["Ms"], p["A"], out, regions=p["regions"], coupling=np.ones((4, 4)))
    with pytest.raises(TypeError):  # ids are uint8, as MuMax3's
        mm.exchange_field(p["grid"], p["m"], p["Ms"], p["A"], out, regions=p["regions"].astype(np.int64), coupling=p["coupling"])


def test_demag_switch(device):
    p = random_problem()
    m = p["m"]
    on = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device)
    off = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device, demag=False)
    H = np.full_like(m, np.nan)
    off.field("demag", m, H)
    assert np.all(H == 0)
    assert off.energies(m)["demag"] == 0 and on.energies(m)["demag"] != 0
    total, rest = np.empty_like(m), np.zeros_like(m)
    off.effective_field(m, total)
    for term in ("exchange", "anisotropy", "zeeman"):
        on.field(term, m, H)
        rest += H
    assert np.allclose(total, rest, rtol=0, atol=1e-12 * np.abs(rest).max())
    on.effective_field(m, H)
    assert not np.allclose(H, rest)  # guard: demag is on by default
