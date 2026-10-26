import math

import numpy as np
import pytest

from micromag.card import CardError, load

GRID = """
[grid]
n = [4, 3, 2]
cell = [2e-9, 2e-9, 2e-9]

[material.permalloy]
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
    (mat,) = c.materials.values()
    assert list(c.materials) == ["permalloy"] and c.demag is True
    assert mat.Ms == 8e5 and isinstance(mat.Ms, float)  # `8e5` is a number in TOML, unlike in PyYAML (D27)
    assert mat.K == 0.0 and np.array_equal(mat.axis, [0, 0, 1])
    assert c.regions == [] and c.grains is None and c.couplings == []
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


HARD = """
[material.hard]
Ms = 1.2e6
A = 8e-12
K = 4e6
axis = [1, 0, 0]
"""
BALL = """
[[region]]
id = 1
material = "hard"
shape = "ellipsoid"
center = [4e-9, 3e-9, 2e-9]
semi_axes = [3e-9, 2e-9, 1e-9]
"""


def test_regions_grains_and_couplings_parse(tmp_path):
    c = card(tmp_path, LLG, GRID + HARD + BALL + """
[[region]]
id = 2
material = "vacuum"
shape = "cuboid"
corners = [[0, 0, 0], [2e-9, 2e-9, 2e-9]]
inverse = true
axis = [0, 1, 1]

[grains]
count = 4
first_id = 10
material = "hard"
seed = 5
within = 1
axes = { mean = [0, 0, 2], cone_deg = 15 }
boundary = { thickness = 4e-9, material = "permalloy", id = 3 }

[[coupling]]
regions = ["grains", 3]
A = 1e-12
""")
    assert list(c.materials) == ["permalloy", "hard"]  # card order: the first is the background (D33)
    r1, r2 = c.regions
    assert (r1.id, r1.material, r1.shape, r1.inverse, r1.axis) == (1, "hard", "ellipsoid", False, None)
    assert np.array_equal(r1.params["semi_axes"], [3e-9, 2e-9, 1e-9])
    assert r2.inverse and np.allclose(r2.axis, [0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    g = c.grains
    assert (g.count, g.first_id, g.material, g.seed, g.within) == (4, 10, "hard", 5, 1)
    assert np.array_equal(g.axes["mean"], [0, 0, 1]) and g.axes["cone_deg"] == 15
    assert (g.boundary.thickness, g.boundary.material, g.boundary.id) == (4e-9, "permalloy", 3)
    (cp,) = c.couplings
    assert (cp.regions, cp.scale, cp.A) == (["grains", 3], None, 1e-12)
    assert card(tmp_path, LLG, GRID.replace("cell", "demag = false\ncell")).demag is False


@pytest.mark.parametrize("body, what", [
    (BALL.replace('"hard"', '"hrad"'), "hrad"),                                  # unknown material, named
    (BALL.replace("id = 1", "id = 0"), "id"),                                     # 0 is the background
    (BALL.replace("id = 1", "id = 256"), "id"),
    (BALL.replace('"ellipsoid"', '"sphere"'), "ellipsoid"),                       # names the valid shapes
    (BALL.replace("semi_axes", "radius"), "semi_axes"),                           # a key of the wrong shape
    (BALL + BALL.replace('"hard"', '"permalloy"'), "region 2"),                   # one id, two materials
    (BALL.replace('"hard"', '"vacuum"') + "\n[material.vacuum]\nMs = 1.0\nA = 0.0\n", "vacuum"),  # reserved
    ("\n[grains]\ncount = 250\nfirst_id = 10\nmaterial = \"hard\"\nseed = 1\n", "255"),
    (BALL + "\n[grains]\ncount = 3\nfirst_id = 1\nmaterial = \"hard\"\nseed = 1\n", "first_id"),  # clashes with region 1
    ("\n[grains]\ncount = 3\nfirst_id = 5\nmaterial = \"hard\"\nseed = 1\nwithin = 9\n", "within"),  # never painted
    ("\n[grains]\ncount = 3\nfirst_id = 5\nmaterial = \"hard\"\nseed = 1\nboundary = { thickness = 3e-9, material = \"hard\", id = 2 }\n",
     "coupling"),                                                                 # < 2 cells of 2 nm: use [[coupling]]
    ("\n[[coupling]]\nregions = [0, 1]\n", "scale"),                            # neither scale nor A
    (BALL + "\n[[coupling]]\nregions = [0, 1]\nscale = 1.0\nA = 1e-12\n", "scale"),  # both
    (BALL + "\n[[coupling]]\nregions = [1, 1]\nscale = 0.5\n", "regions"),    # a pair is two different regions
    (BALL + "\n[[coupling]]\nregions = [0, \"grains\"]\nscale = 0.5\n", "grains"),  # no [grains] table
])
def test_region_errors_name_the_problem(tmp_path, body, what):
    assert what in errors(tmp_path, LLG, GRID + HARD + body)


def test_flat_material_table_points_to_named_materials(tmp_path):
    assert "[material.<name>]" in errors(tmp_path, LLG, GRID.replace("[material.permalloy]", "[material]"))
    assert "demag" in errors(tmp_path, LLG, GRID.replace("cell", "demag = 1\ncell"))


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


THERMAL = LLG + "temperature = 300\nseed = 4\ndt = 1e-14\n"


def test_thermal_keys_and_their_defaults(tmp_path):
    (s,) = card(tmp_path, LLG).stages
    assert s.temperature == 0.0 and s.method == "rk45" and s.dt is None and s.seed is None  # existing cards unchanged (D43)
    (s,) = card(tmp_path, THERMAL).stages
    assert s.temperature == 300.0 and s.method == "heun" and s.dt == 1e-14 and s.seed == 4  # heun by default at T > 0
    (s,) = card(tmp_path, LLG + 'temperature = 300\nseed = 4\nmethod = "rk23"\n').stages
    assert s.method == "rk23" and s.dt is None


@pytest.mark.parametrize("extra, what", [
    ('temperature = 300\nseed = 4\nmethod = "rk45"\n', "rk45 can't take noise"),
    ("temperature = 300\nseed = 4\n", "heun needs 'dt'"),
    ("temperature = 300\ndt = 1e-14\n", "temperature > 0 needs 'seed'"),
    ('method = "rk23"\ndt = 1e-14\n', "'dt' goes with heun"),
    ('method = "euler"\n', "'method' must be"),
    ("temperature = -1\n", "'temperature' must be"),
    ("temperature = 300\nseed = 4294967296\ndt = 1e-14\n", "'seed' must be"),
])
def test_thermal_errors_name_the_problem(tmp_path, extra, what):
    assert what in errors(tmp_path, LLG + extra)


def test_until_torque_is_refused_at_temperature(tmp_path):
    stage = """
[[stage]]
type = "llg-dynamics"
alpha = 0.5
method = "rk23"
temperature = 300
seed = 1
until_torque = 1e-3
check_every = 1e-12
max_duration = 1e-9
"""
    assert "'until_torque' needs temperature = 0" in errors(tmp_path, stage)
