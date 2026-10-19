import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

import micromag as mm
from micromag.card import CardError
from micromag.card import load
from micromag.run import git_state, main, materials, run
from test_llg import SP4_REF, first_zero_crossing

HEAD = """
[grid]
n = [4, 3, 2]
cell = [2e-9, 2e-9, 2e-9]

[material]
Ms = 8e5
A = 1.3e-11
K = 5e5
"""

CARD = HEAD + """
[initial]
m = [1, 0, 1]

[[stage]]
type = "llg-dynamics"
alpha = 0.5
mu0_H = [0, 0, 0.1]
duration = 20e-12
sample_every = 5e-12
dump_every = 10e-12

[[stage]]          # no mu0_H: this stage runs at zero field (D27)
type = "minimize"

[[stage]]
type = "hysteresis"
direction = [0, 0, 1]
mu0_H_from = 0.0
mu0_H_to = -0.2
mu0_H_step = 0.1
dump = true
"""

ENERGIES = ["E_exchange", "E_anisotropy", "E_zeeman", "E_demag", "E_total"]
FIELD = ["mu0_Hx", "mu0_Hy", "mu0_Hz", "mx", "my", "mz"]
LLG_COLUMNS = ["t", *FIELD, *ENERGIES, "steps", "rejected"]
MIN_COLUMNS = [*FIELD, *ENERGIES, "iterations"]
N_CELLS, DV, MS = 24, 8e-27, 8e5


def macrostate(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        return header, np.array([[float(v) for v in row] for row in r])


@pytest.fixture
def done(tmp_path):
    card = tmp_path / "card.toml"
    card.write_text(CARD)
    run(card, tmp_path / "run")
    return card, tmp_path / "run"


def test_results_folder_layout_and_provenance(done):
    card, out = done
    prov = json.loads((out / "provenance.json").read_text())
    dirty = prov["git"] is not None and prov["git"]["dirty"]
    expected = ["card.toml", "provenance.json", "trajectory.h5",
                "stage1_llg-dynamics.csv", "stage2_minimize.csv", "stage3_hysteresis.csv"] + (["git.diff"] if dirty else [])
    assert sorted(p.name for p in out.iterdir()) == sorted(expected)
    assert (out / "card.toml").read_bytes() == card.read_bytes()
    assert prov["status"] == "completed" and prov["device"] == "cpu" and prov["error"] is None
    assert set(prov["versions"]) == {"micromag", "numpy", "h5py", "python"}
    assert prov["git"] is None or len(prov["git"]["hash"]) == 40
    assert [s["type"] for s in prov["stages"]] == ["llg-dynamics", "minimize", "hysteresis"]
    assert all(s["wall_s"] >= 0 for s in prov["stages"]) and prov["started"] <= prov["finished"]


def test_macrostate_rows_record_the_field_actually_applied(done):
    _, out = done
    header, llg = macrostate(out / "stage1_llg-dynamics.csv")
    assert header == LLG_COLUMNS
    assert np.allclose(llg[:, 0], [0, 5e-12, 10e-12, 15e-12, 20e-12], rtol=0, atol=1e-24)
    assert np.all(llg[:, 3] == 0.1) and np.all(llg[1:, -2] > 0)  # field column; steps taken
    header, mn = macrostate(out / "stage2_minimize.csv")
    assert header == MIN_COLUMNS and len(mn) == 1 and np.all(mn[:, :3] == 0)
    header, hy = macrostate(out / "stage3_hysteresis.csv")
    assert header == MIN_COLUMNS and np.allclose(hy[:, 2], [0, -0.1, -0.2], rtol=0, atol=1e-15)

    for table, cols in [(llg, LLG_COLUMNS), (mn, MIN_COLUMNS), (hy, MIN_COLUMNS)]:
        c = {name: table[:, cols.index(name)] for name in cols}
        # Zeeman energy from the solver's own H_ext: -µ0 dV Ms N <m>·H, with µ0 H the recorded field.
        # Holds only if the field the solver used is the one the row reports (stage-local, D27).
        mH = c["mx"] * c["mu0_Hx"] + c["my"] * c["mu0_Hy"] + c["mz"] * c["mu0_Hz"]
        assert np.allclose(c["E_zeeman"], -DV * MS * N_CELLS * mH, rtol=1e-12, atol=1e-35)
        assert np.allclose(c["E_total"], sum(c[e] for e in ENERGIES[:-1]), rtol=1e-12, atol=0)
    assert np.abs(llg[-1, 4:7] - llg[0, 4:7]).max() > 0.01  # m really moved


def test_trajectory_dump(done):
    _, out = done
    _, hy = macrostate(out / "stage3_hysteresis.csv")
    with h5py.File(out / "trajectory.h5") as f:
        assert list(f.attrs["n"]) == [4, 3, 2] and np.allclose(f.attrs["cell"], 2e-9)
        s1, s2, s3 = f["stage1"], f["stage2"], f["stage3"]
        assert s1.attrs["type"] == "llg-dynamics" and s1["m"].shape == (3, 3, 2, 3, 4)  # t = 0, 10, 20 ps
        assert np.allclose(s1["t"][:], [0, 10e-12, 20e-12], rtol=0, atol=1e-24)
        assert s1["m"].compression == "gzip" and s1["m"].shuffle and s1["m"].dtype == np.float64
        assert s2["m"].shape == (1, 3, 2, 3, 4) and np.all(s2["mu0_H"][:] == 0)
        assert s3["m"].shape == (3, 3, 2, 3, 4) and np.allclose(s3["mu0_H"][:, 2], [0, -0.1, -0.2], rtol=0, atol=1e-15)
        final = f["final"]["m"][:]
        assert np.array_equal(final, s3["m"][-1])
        assert np.allclose(final.reshape(3, -1).mean(axis=1), hy[-1, 3:6], rtol=0, atol=1e-15)


def restart(tmp_path, initial):
    card = tmp_path / f"restart{len(list(tmp_path.iterdir()))}.toml"
    card.write_text(HEAD + initial + '\n[[stage]]\ntype = "minimize"\ntorque_tol = 1e9\n')  # returns at once
    out = tmp_path / card.stem
    run(card, out)
    with h5py.File(out / "trajectory.h5") as f:
        return f["final"]["m"][:], json.loads((out / "provenance.json").read_text())


def test_restart_from_a_saved_state(done, tmp_path):
    _, first = done
    with h5py.File(first / "trajectory.h5") as f:
        final, stage1 = f["final"]["m"][:], f["stage1"]["m"][-1]
    m, prov = restart(tmp_path, '[initial]\nfile = "run/trajectory.h5"\n')  # relative to the card
    assert np.array_equal(m, final)  # float64 in, float64 out: exact (D28)
    assert prov["initial"]["sha256"] == hashlib.sha256((first / "trajectory.h5").read_bytes()).hexdigest()
    m, _ = restart(tmp_path, '[initial]\nfile = "run/trajectory.h5"\nstate = "stage1"\n')
    assert np.array_equal(m, stage1)


def test_restart_rejects_a_grid_mismatch_before_writing(done, tmp_path):
    card = tmp_path / "other.toml"
    card.write_text(HEAD.replace("[4, 3, 2]", "[4, 3, 1]") + '[initial]\nfile = "run/trajectory.h5"\n'
                    + '[[stage]]\ntype = "minimize"\n')
    with pytest.raises(CardError, match="grid"):
        run(card, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_refuses_a_non_empty_output_folder(tmp_path):
    card = tmp_path / "card.toml"
    card.write_text(CARD)
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep.txt").write_text("earlier result")
    with pytest.raises(FileExistsError):
        run(card, out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_git_state_records_hash_dirty_flag_and_diff(tmp_path):
    assert git_state(tmp_path) is None  # not a checkout
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (tmp_path / "a.txt").write_text("one\n")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    clean = git_state(tmp_path)
    assert len(clean["hash"]) == 40 and clean["dirty"] is False and clean["diff"] == ""
    (tmp_path / "a.txt").write_text("two\n")
    dirty = git_state(tmp_path)
    assert dirty["hash"] == clean["hash"] and dirty["dirty"] is True and "+two" in dirty["diff"]


def test_cli_device_override_and_exit_codes(tmp_path, capsys):
    card = tmp_path / "card.toml"
    card.write_text('device = "cuda"\n' + HEAD + "[initial]\nm = [0, 0, 1]\n" + '[[stage]]\ntype = "minimize"\n')
    assert main(["run", str(card), "-o", str(tmp_path / "a"), "--device", "cpu"]) == 0
    assert json.loads((tmp_path / "a" / "provenance.json").read_text())["device"] == "cpu"
    bad = tmp_path / "bad.toml"
    bad.write_text(HEAD + "[initial]\nm = [0, 0, 1]\n" + '[[stage]]\ntype = "minimize"\ntorque_toll = 1e-6\n')
    assert main(["run", str(bad), "-o", str(tmp_path / "b")]) == 2
    assert "torque_toll" in capsys.readouterr().err


def test_a_failed_stage_keeps_its_outputs(tmp_path):
    card = tmp_path / "card.toml"
    card.write_text(HEAD + "[initial]\nm = [0, 0, 1]\n" + """
[[stage]]
type = "llg-dynamics"
alpha = 0.01
mu0_H = [0.5, 0, 0]      # strong transverse field: the torque stays large
until_torque = 1e-3
check_every = 1e-12
max_duration = 3e-12
""")
    with pytest.raises(RuntimeError, match="max_duration"):
        run(card, tmp_path / "run")
    out = tmp_path / "run"
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["status"] == "failed" and "max_duration" in prov["error"]
    _, rows = macrostate(out / "stage1_llg-dynamics.csv")
    assert np.allclose(rows[:, 0], [0, 1e-12, 2e-12, 3e-12], rtol=0, atol=1e-24)
    with h5py.File(out / "trajectory.h5") as f:
        assert f["failed"]["m"].shape == (3, 2, 3, 4) and "final" not in f
        assert np.allclose(f["failed"]["m"][:].reshape(3, -1).mean(axis=1), rows[-1, 4:7], rtol=0, atol=1e-15)
    assert main(["run", str(card), "-o", str(tmp_path / "again")]) == 1


@pytest.mark.slow
def test_sp4_card_matches_oommf(tmp_path):
    # D29: the same criteria as the direct SP4 test (D20), through the card and the results folder.
    out = tmp_path / "sp4"
    run(Path(__file__).parent.parent / "examples" / "sp4.toml", out)
    ref = np.loadtxt(SP4_REF, delimiter=",", skiprows=4)
    t_ref, m_ref = ref[:, 0], ref[:, 1:]
    _, relax = macrostate(out / "stage1_llg-dynamics.csv")
    _, switch = macrostate(out / "stage2_llg-dynamics.csv")
    m0 = relax[-1, 4:7]
    assert np.allclose(switch[:, 0], t_ref, rtol=0, atol=1e-21) and np.array_equal(switch[0, 4:7], m0)
    traj = switch[:, 4:7]
    t_cross, t_cross_ref = first_zero_crossing(t_ref, traj[:, 0]), first_zero_crossing(t_ref, m_ref[:, 0])
    rms = np.sqrt(np.mean(np.sum((traj - m_ref) ** 2, axis=1)))
    print(f"\nSP4 from card: S-state <m> = {m0.round(4)} after {relax[-1, 0] * 1e9:.2f} ns; "
          f"<mx> = 0 at {t_cross * 1e12:.1f} ps (OOMMF {t_cross_ref * 1e12:.1f}); RMS {rms:.4f}")
    assert np.abs(m0 - m_ref[0]).max() <= 0.01
    assert abs(t_cross / t_cross_ref - 1) <= 0.03
    assert rms <= 0.05


def test_regions_paint_cell_centres_in_order(tmp_path):
    # D30: a cell takes a box's values if its centre is inside (lower bound included); later boxes win.
    path = tmp_path / "card.toml"
    path.write_text(HEAD.replace("K = 5e5", "K = 0") + """
[initial]
m = [0, 0, 1]

[[region]]
box = [[2e-9, 0, 0], [8e-9, 6e-9, 4e-9]]
K = 1e5
axis = [1, 0, 0]

[[region]]
box = [[0, 0, 2e-9], [4e-9, 6e-9, 4e-9]]
Ms = 0

[[stage]]
type = "minimize"
""")
    Ms, A, K, axis = materials(load(path))
    x_in = np.array([False, True, True, True])            # cell centres at 1, 3, 5, 7 nm; box from 2 nm
    assert np.array_equal(K, np.where(x_in, 1e5, 0.0) * np.ones((2, 3, 1)))
    assert np.array_equal(axis[0], np.where(x_in, 1.0, 0.0) * np.ones((2, 3, 1)))
    assert np.array_equal(axis[2], np.where(x_in, 0.0, 1.0) * np.ones((2, 3, 1)))
    vacuum = np.zeros((2, 3, 4), bool)
    vacuum[1, :, :2] = True                                # z centre 3 nm, x centres 1 and 3 nm
    assert np.array_equal(Ms, np.where(vacuum, 0.0, MS)) and np.all(A == 1.3e-11)


@pytest.mark.parametrize("psi, hc, hsw", [(30, 0.5240, 0.5240), (60, 0.4330, 0.5240)])
def test_refined_hysteresis_reports_coercive_and_switching_fields(tmp_path, capsys, psi, hc, hsw):
    # One cubic cell is a Stoner–Wohlfarth particle (D23). For psi <= 45° the coercive field (m·h = 0) is the
    # switching field; above 45° m passes through m ⟂ h first, at h_c = sin(2 psi)/2, and switches later (D31).
    BK = 2 * 5e5 / MS  # µ0 H_K = 2K/Ms, in T
    s, c = float(np.sin(np.radians(psi))), float(np.cos(np.radians(psi)))
    card = tmp_path / "card.toml"
    card.write_text(f"""
[grid]
n = [1, 1, 1]
cell = [2e-9, 2e-9, 2e-9]

[material]
Ms = 8e5
A = 1e-11
K = 5e5

[initial]
m = [0, 0, 1]

[[stage]]
type = "hysteresis"
direction = [{s!r}, 0.0, {c!r}]
mu0_H_from = 0.0
mu0_H_to = -0.75
mu0_H_step = 0.05
refine_to = 1e-4
""")
    run(card, tmp_path / "run")
    summary = json.loads((tmp_path / "run" / "provenance.json").read_text())["stages"][0]["summary"]
    assert abs(summary["switch_T"] + hsw * BK) <= 2e-4, summary
    assert abs(summary["Hc_T"] + hc * BK) <= 2e-4, summary
    assert np.allclose(summary["remanence"], [0, 0, 1], rtol=0, atol=1e-6)
    _, rows = macrostate(tmp_path / "run" / "stage1_hysteresis.csv")
    assert len(rows) < 60  # 16 coarse fields plus two bisections, not 7500 fine steps
    assert "Hc" in capsys.readouterr().out
