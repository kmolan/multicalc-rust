import pytest

import multicalc_py

def test_lqr_cart():
    controller = multicalc_py.Lqr2x1(
        [[1.0, 0.1], [0.0, 1.0]],
        [[0.005], [0.1]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0]],
    )
    controller.certify_stability()
    state = [1.0, 0.0]
    for _ in range(400):
        command = controller.control(state)
        state = [
            1.0 * state[0] + 0.1 * state[1] + 0.005 * command[0],
            0.0 * state[0] + 1.0 * state[1] + 0.1 * command[0],
        ]
    assert (state[0] ** 2 + state[1] ** 2) ** 0.5 < 1e-6

def test_geometric_attitude_zero_torque():
    inertia = [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.04]]
    controller = multicalc_py.GeometricAttitudeController(6.0, 1.2, inertia)
    still = [0.0, 0.0, 0.0]
    level = multicalc_py.SO3.identity()
    torque = controller.torque(level, still, level, still, still)
    assert torque[0] == pytest.approx(0.0, abs=1e-12)
    assert torque[1] == pytest.approx(0.0, abs=1e-12)
    assert torque[2] == pytest.approx(0.0, abs=1e-12)

def test_pure_pursuit_straight():
    curvature = multicalc_py.pure_pursuit_curvature(
        multicalc_py.SE2.identity(), [2.0, 0.0], 2.0
    )
    assert curvature == pytest.approx(0.0, abs=1e-12)

def test_thrust_hover_and_freefall():
    hover = multicalc_py.thrust_command_from_acceleration([0.0, 0.0, 0.0], 0.0, 9.81)
    assert hover.thrust_acceleration() == pytest.approx(9.81, abs=1e-12)
    with pytest.raises(multicalc_py.ControlError):
        multicalc_py.thrust_command_from_acceleration([0.0, 0.0, -9.81], 0.0, 9.81)

def test_follow_the_gap():
    follower = multicalc_py.FollowTheGap5.try_new(2.0, 4.0, 0.5, 0.5, 0.4)
    clear = follower.compute([4.0, 4.0, 4.0, 4.0, 4.0], 0.0)
    assert not clear.is_blocked()
    blocked = follower.compute([0.2, 0.2, 0.2, 0.2, 0.2], 0.0)
    assert blocked.is_blocked()
    