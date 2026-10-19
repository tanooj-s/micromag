import math

import numpy as np
import pytest

from micromag.card import CardError, load

GRID = """
[grid]
n = [4, 3, 2]
cell = [2e-9, 2e-9, 2e-9]

[material]
Ms = 8e5
A = 1.3e-11

[initial]
m = [1, 1, 1]
"""


def card(tmp_path, stages, head=GRID):
    path = tmp_path / "card.toml"
    path.write_text(head + stages)
    return load(path)


def errors(tmp_path, stages, head=GRID):
    with pytest.raises(CardError) as e:
        card(tmp_path, stages, head)
    return str(e.value).split("\n", 1)[1]  # drop the path line: test directory names would match anything


LLG = """
[[stage]]
type = "llg-dynamics"
alpha = 0.5
duration = 1e-11
"""


def test_minimal_card_gets_its_defaults(tmp_path):
    c = card(tmp_path, LLG)
    assert c.device == "cpu"
    assert c.n == (4, 3, 2) and c.cell == (2e-9, 2e-9, 2e-9)
    assert c.Ms == 8e5 and isinstance(c.Ms, float)  # `8e5` is a number in TOML, unlike in PyYAML (D27)
    assert c.K == 0.0 and np.array_equal(c.axis, [0, 0, 1])
    assert np.allclose(c.m0, np.ones(3) / math.sqrt(3))  # normalised on load
    (s,) = c.stages
    assert s.type == "llg-dynamics" and s.alpha == 0.5 and s.duration == 1e-11
    assert s.gamma == 1.76086e11 and s.tol == 1e-6 and s.max_step == math.inf
    assert np.array_equal(s.mu0_H, [0, 0, 0])
    assert s.sample_every is None and s.until_torque is None and s.dump_every is None


def test_unknown_key_is_named_with_the_valid_ones(tmp_path):
    msg = errors(tmp_path, LLG.replace("alpha", "alhpa"))
    assert "alhpa" in msg and "alpha" in msg and "stage 1" in msg


def test_all_errors_are_reported_together(tmp_path):
    msg = errors(tmp_path, LLG.replace("alpha = 0.5", "").replace("duration", "durration"), GRID.replace("Ms", "Mss"))
    for key in ("Mss", "durration", "alpha"):
        assert key in msg, (key, msg)


@pytest.mark.parametrize("bad, key", [
    ("n = [4, 3]", "n"),
    ("n = [4, 3, 0]", "n"),
    ("n = [4.0, 3, 2]", "n"),
    ("cell = [2e-9, -2e-9, 2e-9]", "cell"),
])
def test_grid_types_and_shapes_are_checked(tmp_path, bad, key):
    head = GRID.replace("n = [4, 3, 2]" if key == "n" else "cell = [2e-9, 2e-9, 2e-9]", bad)
    assert key in errors(tmp_path, LLG, head)


@pytest.mark.parametrize("value", ['"fast"', "true", "-0.1"])
def test_alpha_must_be_a_nonnegative_number(tmp_path, value):
    assert "alpha" in errors(tmp_path, LLG.replace("0.5", value))  # a bool is not a number, though Python says so


def test_initial_state_needs_exactly_one_source(tmp_path):
    both = GRID.replace('m = [1, 1, 1]', 'm = [1, 1, 1]\nfile = "run/trajectory.h5"')
    neither = GRID.replace('m = [1, 1, 1]', '')
    assert "initial" in errors(tmp_path, LLG, both)
    assert "initial" in errors(tmp_path, LLG, neither)


def test_unknown_stage_type_and_no_stages_are_errors(tmp_path):
    assert "llg-dynamics" in errors(tmp_path, LLG.replace('"llg-dynamics"', '"llg"'))  # names the valid types
    assert "stage" in errors(tmp_path, "")


def test_llg_needs_exactly_one_stopping_rule(tmp_path):
    both = LLG + "until_torque = 1e-3\ncheck_every = 1e-12\nmax_duration = 1e-10\n"
    neither = LLG.replace("duration = 1e-11", "")
    assert "duration" in errors(tmp_path, both)
    assert "duration" in errors(tmp_path, neither)
    assert "check_every" in errors(tmp_path, LLG.replace("duration = 1e-11", "until_torque = 1e-3"))


def test_tol_must_sit_well_below_until_torque(tmp_path):
    # D20: each step's error re-excites a torque of a few × tol, so a threshold near tol is never reached.
    torque = LLG.replace("duration = 1e-11", "until_torque = 1e-6\ncheck_every = 1e-12\nmax_duration = 1e-10")
    assert "tol" in errors(tmp_path, torque)  # default tol = 1e-6
    assert card(tmp_path, torque + "tol = 1e-9\n").stages[0].until_torque == 1e-6


def test_hysteresis_ramp_is_inclusive_and_normalised(tmp_path):
    ramp = """
[[stage]]
type = "hysteresis"
direction = [0, 3, -4]
mu0_H_from = 0.1
mu0_H_to = -0.4
mu0_H_step = 0.1
"""
    (s,) = card(tmp_path, ramp).stages
    assert np.allclose(s.fields_T(), np.outer([0.1, 0.0, -0.1, -0.2, -0.3, -0.4], [0, 0.6, -0.8]), rtol=0, atol=1e-15)
    assert s.dump is False and s.torque_tol == 1e-6 and s.max_iter == 100_000
    assert "mu0_H_step" in errors(tmp_path, ramp.replace("mu0_H_step = 0.1", "mu0_H_step = 0.3"))  # 0.5 / 0.3 is not whole


REGIONS = GRID + """
[[region]]
box = [[2e-9, 0, 0], [8e-9, 6e-9, 4e-9]]
K = 1e5
axis = [1, 0, 0]

[[region]]
box = [[0, 0, 2e-9], [4e-9, 6e-9, 4e-9]]
Ms = 0
"""


def test_regions_parse_in_order(tmp_path):
    c = card(tmp_path, LLG, REGIONS)
    r1, r2 = c.regions
    assert np.allclose(r1.box, [[2e-9, 0, 0], [8e-9, 6e-9, 4e-9]], rtol=0, atol=0)
    assert set(r1.values) == {"K", "axis"} and r1.values["K"] == 1e5 and np.array_equal(r1.values["axis"], [1, 0, 0])
    assert r2.values == {"Ms": 0.0}  # Ms = 0 is vacuum, allowed in a region (D30)
    assert card(tmp_path, LLG).regions == []


@pytest.mark.parametrize("box, what", [
    ("[[0, 0, 0], [1e-9, 1e-9]]", "box"),
    ("[[2e-9, 0, 0], [1e-9, 1e-9, 1e-9]]", "box"),  # upper corner below lower corner
])
def test_region_box_is_checked(tmp_path, box, what):
    assert what in errors(tmp_path, LLG, GRID + f"[[region]]\nbox = {box}\nK = 1.0\n")


def test_region_errors_name_the_problem(tmp_path):
    assert "Kk" in errors(tmp_path, LLG, GRID + "[[region]]\nbox = [[0, 0, 0], [1e-9, 1e-9, 1e-9]]\nKk = 1.0\n")
    assert "region 1" in errors(tmp_path, LLG, GRID + "[[region]]\nbox = [[0, 0, 0], [1e-9, 1e-9, 1e-9]]\n")  # sets nothing
    assert "Ms" in errors(tmp_path, LLG, GRID + "[[region]]\nbox = [[0, 0, 0], [1e-9, 1e-9, 1e-9]]\nMs = -1.0\n")


def test_refinement_keys(tmp_path):
    ramp = """
[[stage]]
type = "hysteresis"
direction = [0, 0, 1]
mu0_H_from = 0.0
mu0_H_to = -1.0
mu0_H_step = 0.1
"""
    (s,) = card(tmp_path, ramp).stages
    assert s.refine_to is None and s.refine_jump == 0.1
    (s,) = card(tmp_path, ramp + "refine_to = 1e-4\nrefine_jump = 0.05\n").stages
    assert s.refine_to == 1e-4 and s.refine_jump == 0.05
    assert "refine_to" in errors(tmp_path, ramp + "refine_to = 0.2\n")  # coarser than the step itself
