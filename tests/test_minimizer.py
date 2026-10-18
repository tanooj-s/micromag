import numpy as np
import pytest

import micromag as mm
from test_fields import C, H_EXT, energy, random_problem


def test_problem_energies_match_numpy(device):
    # Problem.energies (C++) against the numpy formula of D10 on CPU fields, term by term.
    # D26: 1e-12 relative also covers the GPU, whose reductions sum in a different order.
    p = random_problem()
    prob = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT, device=device)
    E = prob.energies(p["m"])
    for term in C:
        assert np.isclose(E[term], energy(term, p, p["m"]), rtol=1e-12, atol=0), term


@pytest.mark.parametrize("psi_deg", [15, 30, 45, 60, 75])
def test_stoner_wohlfarth_switching_field(psi_deg, device):
    # One cubic cell: its demag field is parallel to m, so this is the Stoner–Wohlfarth particle (D23).
    Ms, K = 8e5, 5e5
    HK = 2 * K / (mm.mu0 * Ms)
    g = mm.Grid(1, 1, 1, 2e-9, 2e-9, 2e-9)
    one = np.ones((1, 1, 1))
    k = np.zeros((3, 1, 1, 1))
    k[2] = 1.0
    prob = mm.Problem(g, Ms * one, 1e-11 * one, K * one, k, (0, 0, 0), device=device)

    psi, dh = np.radians(psi_deg), 1e-3
    h = -np.arange(0, 1.2 + dh / 2, dh)  # field in units of H_K, swept from 0 to -1.2
    fields = h[:, None] * HK * np.array([np.sin(psi), 0, np.cos(psi)])
    m = k.copy()
    res = mm.hysteresis(prob, m, fields, mm.Minimizer(prob))

    # Away from the switch m moves ~dh per step, just before it ~√dh; the switch itself is an O(1) jump.
    i = np.argmax(np.linalg.norm(np.diff(res["m"], axis=0), axis=1))
    found = -(h[i] + h[i + 1]) / 2
    expected = (np.cos(psi) ** (2 / 3) + np.sin(psi) ** (2 / 3)) ** -1.5
    assert abs(found - expected) <= 2 * dh, (found, expected)
    assert res["m"][-1] @ [0, 0, 1] < 0  # it really did reverse


@pytest.mark.slow
def test_sp3_single_domain_limit(device):
    N, Ms, A = 20, 8e5, 1.3e-11
    Km = mm.mu0 * Ms**2 / 2
    lex = np.sqrt(A / Km)
    s = (N, N, N)
    one = np.ones(s)
    k = np.zeros((3, *s))
    k[2] = 1.0

    # Symmetric starts, no noise: the symmetric flower that Rave, Martins and MagTense computed (D23).
    c = (np.arange(N) + 0.5) / N - 0.5
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
    flower = k.copy()
    # The vortex circulates around x, perpendicular to the easy axis, so half its volume still lies along ±z.
    # (Circulating around z puts m in-plane, against the anisotropy, and it relaxes into the flower.)
    vortex = np.stack([np.full(s, 0.2), -Z, Y])  # soft core along x
    vortex /= np.linalg.norm(vortex, axis=0)

    def reduced_energies(L):
        """Minimize both states at edge L (in l_ex), in place; return their partial energies / (K_m L³)."""
        d = L * lex / N
        prob = mm.Problem(mm.Grid(N, N, N, d, d, d), Ms * one, A * one, 0.1 * Km * one, k, (0, 0, 0), device=device)
        mn = mm.Minimizer(prob)
        out = []
        for m in (flower, vortex):  # warm start: at a fixed grid, changing L only rescales the cells
            mn.minimize(m)
            out.append({t: E / (Km * (L * lex) ** 3) for t, E in prob.energies(m).items()})
        assert flower[2].mean() > 0.9 and vortex[2].mean() < 0.5, "a state collapsed into the other"
        return out

    def delta(L):
        e_f, e_v = reduced_energies(L)
        return sum(e_f.values()) - sum(e_v.values())

    lo, hi = 8.0, 9.0
    assert delta(lo) < 0 < delta(hi)  # the flower wins below the limit, the vortex above it
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if delta(mid) < 0 else (lo, mid)
    L = (lo + hi) / 2

    e_f, e_v = reduced_energies(L)
    print(f"\nSP3 at 20³: L = {L:.4f} l_ex (consensus 8.47, MagTense 20³ 8.4556)")
    # Rave uses K(1 - (m·k)²); our eq. 9 form -K(m·k)² is lower by K per volume, i.e. 0.1 in these units (D12).
    for name, e, rave in [("flower", e_f, (0.2794, 0.0177, 0.0056)), ("vortex", e_v, (0.0783, 0.1723, 0.0521))]:
        print(f"  {name}: demag {e['demag']:.4f} exchange {e['exchange']:.4f} anisotropy {e['anisotropy'] + 0.1:.4f} "
              f"total {sum(e.values()) + 0.1:.4f}  (Rave: {rave[0]}, {rave[1]}, {rave[2]}, total 0.3027)")
    assert abs(L / 8.47 - 1) <= 0.005
