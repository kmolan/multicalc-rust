import math

import pytest

import multicalc_py

def test_so2_act():
    point = multicalc_py.SO2.exp(math.pi / 2.0).act([1.0, 0.0])
    assert point[0] == pytest.approx(0.0, abs=1e-12)
    assert point[1] == pytest.approx(1.0, abs=1e-12)

def test_se2_identity():
    assert multicalc_py.SE2.identity().act([1.0, 2.0]) == pytest.approx([1.0, 2.0])

def test_quaternion_identity():
    point = multicalc_py.Quaternion.identity().transform_point([1.0, 0.0, 0.0])
    assert point == pytest.approx([1.0, 0.0, 0.0])

def test_spatial_inertia_and_error():
    body = multicalc_py.SpatialInertia.from_diagonal_inertia(
        2.0, [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]
    )
    assert body.mass() == 2.0
    with pytest.raises(multicalc_py.SpatialError):
        multicalc_py.SpatialInertia.from_diagonal_inertia(
            0.0, [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]
        )

def test_free_joint_identity():
    state = multicalc_py.FreeJointState.identity()
    assert state.velocity().as_array() == pytest.approx([0.0] * 6)
    