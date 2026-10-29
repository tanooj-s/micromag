import csv
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pytest

import micromag as mm
from micromag.card import CardError
from micromag.card import load
from micromag.regions import paint
from micromag.run import git_state, main, run
from test_llg import SP4_REF, first_zero_crossing

HEAD = """
[grid]
n = [4, 3, 2]
cell = [2e-9, 2e-9, 2e-9]

[material.permalloy]
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
    expected = ["card.toml", "provenance.json", "trajectory.h5", "trajectory.xdmf",
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


def _eval_xdmf(out):
    """Parses out/trajectory.xdmf and resolves its HyperSlab + JOIN($0, $1, $2) references against trajectory.h5."""
    root = ET.parse(out / "trajectory.xdmf").getroot()
    assert root.tag == "Xdmf"
    coll = root.find("./Domain/Grid")
    assert coll is not None and coll.attrib["GridType"] == "Collection" and coll.attrib["CollectionType"] == "Temporal"
    frames = []
    with h5py.File(out / "trajectory.h5") as f:
        for grid in coll.findall("Grid"):
            t = float(grid.find("Time").attrib["Value"])
            topo = [int(x) for x in grid.find("Topology").attrib["Dimensions"].split()]
            assert grid.find("Topology").attrib["TopologyType"] == "3DCoRectMesh"
            geom = grid.find("Geometry")
            assert geom.attrib["GeometryType"] == "ORIGIN_DXDYDZ"
            origin, spacing = [[float(x) for x in d.text.split()] for d in geom.findall("DataItem")]

            m_attr = grid.find("./Attribute[@Name='m']")
            assert m_attr.attrib["AttributeType"] == "Vector" and m_attr.attrib["Center"] == "Cell"
            join = m_attr.find("DataItem")
            assert join.attrib["ItemType"] == "Function" and join.attrib["Function"] == "JOIN($0, $1, $2)"
            join_dims = [int(x) for x in join.attrib["Dimensions"].split()]
            slabs = []
            for slab in join.findall("DataItem"):
                assert slab.attrib["ItemType"] == "HyperSlab"
                slab_dims = [int(x) for x in slab.attrib["Dimensions"].split()]
                spec_item, src_item = slab.findall("DataItem")
                ndim = len(src_item.attrib["Dimensions"].split())
                assert [int(x) for x in spec_item.attrib["Dimensions"].split()] == [3, ndim]
                start, stride, count = np.fromstring(spec_item.text, sep=" ", dtype=int).reshape(3, ndim)
                h5_file, h5_path = src_item.text.strip().split(":")
                assert h5_file == "trajectory.h5" and src_item.attrib["Format"] == "HDF"
                ds = f[h5_path]
                assert [int(x) for x in src_item.attrib["Dimensions"].split()] == list(ds.shape)
                idx = tuple(slice(s, s + k * d, d) for s, d, k in zip(start, stride, count))
                arr = np.squeeze(ds[idx], axis=tuple(i for i, k in enumerate(count) if k == 1 and i < ndim - 3))
                assert list(arr.shape) == slab_dims
                slabs.append(arr)
            m = np.stack(slabs, axis=-1)
            assert list(m.shape) == join_dims

            reg_attr = grid.find("./Attribute[@Name='regions']")
            assert reg_attr.attrib["AttributeType"] == "Scalar" and reg_attr.attrib["Center"] == "Cell"
            reg_item = reg_attr.find("DataItem")
            h5_file, h5_path = reg_item.text.strip().split(":")
            assert h5_file == "trajectory.h5" and reg_item.attrib["NumberType"] == "UInt"
            reg = f[h5_path][()]
            assert [int(x) for x in reg_item.attrib["Dimensions"].split()] == list(reg.shape)
            frames.append((grid.attrib["Name"], t, topo, origin, spacing, m, reg))
    return frames


def test_xdmf_sidecar_slices_match_trajectory_h5(done):
    _, out = done
    frames = _eval_xdmf(out)
    with h5py.File(out / "trajectory.h5") as f:
        expected = [(f"stage{k}_{i}", s) for k in (1, 2, 3) for i, s in enumerate(f[f"stage{k}"]["m"][:])]
        regions = f["regions"][()]
    assert len(frames) == 7 and [t for _, t, *_ in frames] == list(range(7))
    for (name, _, topo, origin, spacing, m, reg), (exp_name, exp_m) in zip(frames, expected):
        assert name == exp_name and topo == [3, 4, 5]
        assert origin == [0.0, 0.0, 0.0] and np.allclose(spacing, [2e-9, 2e-9, 2e-9], rtol=0, atol=1e-24)
        assert np.linalg.norm(exp_m) > 0 and np.array_equal(m, np.moveaxis(exp_m, 0, -1))
        assert np.array_equal(reg, regions)



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
    card.write_text(HEAD.replace("[2e-9, 2e-9, 2e-9]", "[1e-9, 2e-9, 3e-9]") + "[initial]\nm = [0, 0, 1]\n" + """
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
        failed_m = f["failed"]["m"][:]
    frames = _eval_xdmf(out)
    assert len(frames) == 1 and frames[0][0] == "failed" and frames[0][1] == 0.0
    assert np.allclose(frames[0][4], [3e-9, 2e-9, 1e-9], rtol=0, atol=1e-24)  # spacing is dz, dy, dx
    assert np.array_equal(frames[0][5], np.moveaxis(failed_m, 0, -1))
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


def test_region_map_is_saved_and_volumes_reported(tmp_path, capsys):
    # D33: the map goes to trajectory.h5 with its id -> material names; provenance has each region's volumes.
    path = tmp_path / "card.toml"
    path.write_text(HEAD + """
[material.hard]
Ms = 1.2e6
A = 8e-12
K = 4e6

[[region]]
id = 7
material = "hard"
shape = "cuboid"
corners = [[2e-9, 0, 0], [12e-9, 6e-9, 4e-9]]   # sticks out of the 8 nm grid: half of it is clipped

[initial]
m = [0, 0, 1]

[[stage]]
type = "minimize"
""")
    run(path, tmp_path / "run")
    with h5py.File(tmp_path / "run" / "trajectory.h5") as f:
        regions = f["regions"][()]
        names = json.loads(f["regions"].attrs["materials"])
    assert regions.dtype == np.uint8 and np.array_equal(regions, paint(load(path)).regions)
    assert set(np.unique(regions)) == {0, 7} and names == {"0": "permalloy", "7": "hard"}
    prov = json.loads((tmp_path / "run" / "provenance.json").read_text())
    (r,) = prov["regions"]
    assert r["id"] == 7 and r["volume_m3"] == pytest.approx(240e-27) and r["cells_m3"] == pytest.approx(144e-27)
    assert "region 1" in capsys.readouterr().err  # a 40% mismatch is warned (D33)


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

[material.permalloy]
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


THERMAL_CARD = HEAD + """
[initial]
m = [0, 0, 1]

[[stage]]
type = "llg-dynamics"
alpha = 0.5
temperature = 300
seed = 11
dt = 1e-14
duration = 2e-12
sample_every = 1e-12

[[stage]]
type = "llg-dynamics"
alpha = 0.5
method = "rk23"
temperature = 300
seed = 11
duration = 2e-12
"""


def test_thermal_stages_record_their_settings_and_reproduce(tmp_path):
    finals = []
    for name in ("a", "b"):
        card = tmp_path / f"{name}.toml"
        card.write_text(THERMAL_CARD)
        run(card, tmp_path / name)
        with h5py.File(tmp_path / name / "trajectory.h5") as f:
            finals.append(f["final"]["m"][:])
    assert np.abs(finals[0] - np.array([0, 0, 1.0])[:, None, None, None]).max() > 1e-3  # the noise moved m
    assert np.array_equal(finals[0], finals[1])  # same card, same seed: the same run (D41)
    s1, s2 = json.loads((tmp_path / "a" / "provenance.json").read_text())["stages"]
    assert {k: s1[k] for k in ("temperature", "method", "dt", "seed")} == dict(temperature=300.0, method="heun", dt=1e-14, seed=11)
    assert s1["steps"] == 200 and s1["rejected"] == 0 and s1["max_bridge_depth"] == 0
    assert s2["method"] == "rk23" and s2["tol"] == 1e-6 and s2["steps"] > 0 and "max_bridge_depth" in s2
