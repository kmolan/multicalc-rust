import pytest

import multicalc_py

def test_rigid_body_falls():
    inertia = multicalc_py.SpatialInertia.from_diagonal_inertia(
        2.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    )
    body = multicalc_py.RigidBody(inertia, [0.0, 0.0, -9.81])
    linear, angular = body.accelerations(
        multicalc_py.SO3.identity(),
        [0.0, 0.0, 0.0],
        multicalc_py.Wrench.zeros(),
    )
    assert linear[2] == pytest.approx(-9.81)
    assert angular[0] == pytest.approx(0.0)
