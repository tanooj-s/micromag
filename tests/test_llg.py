from pathlib import Path

import numpy as np
import pytest

import micromag as mm
from test_fields import C, H_EXT, energy, random_problem

GAMMA = 1.76086e11  # rad/(T·s), the paper's value and the LLG default


def macrospin(Hz):
    """One cubic cell in a field along z. Its demag field is -Ms m/3, parallel to m, so it exerts no torque."""
    g = mm.Grid(1, 1, 1, 2e-9, 2e-9, 2e-9)
    one = np.ones((1, 1, 1))
    return mm.Problem(g, 8e5 * one, 1e-11 * one, 0 * one, np.zeros((3, 1, 1, 1)), (0, 0, Hz))


def macrospin_error(alpha, t_end, samples, **llg_args):
    """Largest |m - m_exact| at the sample times, starting from θ0 = 30° in 1e5 A/m along z."""
    H, th0 = 1e5, np.radians(30)
    w = mm.mu0 * GAMMA * H / (1 + alpha**2)
    llg = mm.LLG(macrospin(H), alpha, **llg_args)
    m = np.array([np.sin(th0), 0, np.cos(th0)]).reshape(3, 1, 1, 1)
    err = 0.0
    for t in np.linspace(0, t_end, samples + 1)[1:]:
        llg.run_until(m, t)
        th = 2 * np.arctan(np.tan(th0 / 2) * np.exp(-alpha * w * t))  # damping spirals m towards +z
        exact = [np.sin(th) * np.cos(w * t), np.sin(th) * np.sin(w * t), np.cos(th)]  # precession: φ = ωt
        err = max(err, np.linalg.norm(m[:, 0, 0, 0] - exact))
    return err


def period(alpha, H=1e5):
    return 2 * np.pi * (1 + alpha**2) / (mm.mu0 * GAMMA * H)


def test_macrospin_matches_analytic():
    err = macrospin_error(0.1, 3 * period(0.1), 30, tol=1e-8)
    assert err <= 10 * 1e-8, err


def test_fixed_steps_converge_at_fifth_order():
    # tol = ∞ never rejects and always grows h, so max_step pins every step to h (D20).
    # An adaptive run would hide a wrong Dormand–Prince coefficient by taking smaller steps.
    T = period(0.1)
    errs = np.array([macrospin_error(0.1, T, 1, tol=np.inf, max_step=T / n) for n in (16, 32, 64)])
    ratios = errs[:-1] / errs[1:]
    assert np.all((26 < ratios) & (ratios < 38)), (errs, ratios)


def total_energy(p, m):
    return sum(energy(term, p, m) for term in C)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_energy_is_conserved_without_damping_and_falls_with_it(alpha):
    p = random_problem()
    prob = mm.Problem(p["grid"], p["Ms"], p["A"], p["K"], p["k"], H_EXT)
    H = np.empty_like(p["m"])
    prob.effective_field(p["m"], H)
    T = 10 * 2 * np.pi / (mm.mu0 * GAMMA * np.abs(H).max())  # ~10 periods of the fastest precession
    m = p["m"].copy()
    llg = mm.LLG(prob, alpha, tol=1e-9)
    E = [total_energy(p, m)]
    for t in np.linspace(0, T, 51)[1:]:
        llg.run_until(m, t)
        E.append(total_energy(p, m))
    E = np.array(E)
    assert np.abs(m - p["m"]).max() > 0.1  # a frozen m would conserve energy trivially
    if alpha == 0:
        assert np.abs(E - E[0]).max() <= 1e-6 * abs(E[0])
    else:
        assert np.all(np.diff(E) <= 1e-12 * abs(E[0])), np.diff(E).max()
        assert E[-1] < E[0]
    assert np.abs(np.linalg.norm(m, axis=0) - 1).max() <= 1e-14


SP4_REF = Path(__file__).parent / "data" / "sp4_field1_oommf.csv"


def first_zero_crossing(t, mx):
    i = np.argmax(mx < 0)
    return t[i - 1] + (t[i] - t[i - 1]) * mx[i - 1] / (mx[i - 1] - mx[i])


@pytest.mark.slow
def test_sp4_field1_matches_oommf():
    ref = np.loadtxt(SP4_REF, delimiter=",", skiprows=4)
    t_ref, m_ref = ref[:, 0], ref[:, 1:]
    g = mm.Grid(128, 32, 1, 500e-9 / 128, 125e-9 / 32, 3e-9)
    s = (g.nz, g.ny, g.nx)
    one = np.ones(s)
    prob = mm.Problem(g, 8e5 * one, 1.3e-11 * one, 0 * one, np.zeros((3, *s)), (0, 0, 0))
    gamma = 2.211e5 / mm.mu0  # SP4 specifies µ0γ = 2.211e5 m/(A·s)

    # S-state: relax uniform (1,1,1)/√3 at zero field with α = 1 until the largest torque is below 1e-6 Ms.
    # tol must sit well below that threshold: each step's error re-excites a torque of a few × tol (D20).
    m = np.full((3, *s), 1 / np.sqrt(3))
    relax, H = mm.LLG(prob, 1.0, gamma=gamma, tol=1e-9), np.empty_like(m)
    while True:
        relax.run_until(m, relax.t + 50e-12)
        prob.effective_field(m, H)
        torque = np.linalg.norm(np.cross(m, H, axis=0), axis=0).max() / 8e5
        if torque < 1e-6:
            break
        assert relax.t < 20e-9, f"S-state not relaxed after 20 ns (torque {torque:.1e})"
    m0 = m.mean(axis=(1, 2, 3))
    print(f"\nS-state after {relax.t * 1e9:.2f} ns: <m> = {m0.round(4)}, OOMMF {m_ref[0].round(4)}")
    assert np.abs(m0 - m_ref[0]).max() <= 0.01

    prob.H_ext = np.array([-24.6e-3, 4.3e-3, 0.0]) / mm.mu0
    llg = mm.LLG(prob, 0.02, gamma=gamma)
    traj = [m0]
    for t in t_ref[1:]:
        llg.run_until(m, t)
        traj.append(m.mean(axis=(1, 2, 3)))
    traj = np.array(traj)

    t_cross, t_cross_ref = first_zero_crossing(t_ref, traj[:, 0]), first_zero_crossing(t_ref, m_ref[:, 0])
    rms = np.sqrt(np.mean(np.sum((traj - m_ref) ** 2, axis=1)))  # RMS over time of |<m> - <m>_ref|
    print(f"<mx> = 0 at {t_cross * 1e12:.1f} ps (OOMMF {t_cross_ref * 1e12:.1f} ps); RMS {rms:.4f}; "
          f"{llg.steps} steps, {llg.rejected} rejected")
    assert abs(t_cross / t_cross_ref - 1) <= 0.03
    assert rms <= 0.05
