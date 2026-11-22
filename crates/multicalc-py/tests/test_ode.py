import math

import pytest

import multicalc_py

def growth(_time: float, state: list[float]) -> list[float]:
    return [state[0], 0.0]

def decay(_time: float, state: list[float]) -> list[float]:
    return [-state[0], 0.0]

def test_rk4_step_exponential():
    next_state = multicalc_py.rk4_step(growth, 0.0, [1.0, 0.0], 0.1)
    assert next_state[0] == pytest.approx(math.exp(0.1), rel=1e-5)

def test_rk45_solve_decay():
    final_state = multicalc_py.rk45_solve(decay, 0.0, [1.0, 0.0], 1.0)
    assert final_state[0] == pytest.approx(math.exp(-1.0), rel=1e-5)

def test_exponential_map_quarter_turn():
    timestep = math.pi / 8.0
    orientation = multicalc_py.SO3.identity()
    rate = [0.0, 0.0, 2.0]
    orientation = multicalc_py.exponential_map_attitude_step(orientation, rate, timestep)
    orientation = multicalc_py.exponential_map_attitude_step(orientation, rate, timestep)
    point = orientation.act([1.0, 0.0, 0.0])
    assert point[0] == pytest.approx(0.0, abs=1e-12)
    assert point[1] == pytest.approx(1.0, abs=1e-12)
    assert point[2] == pytest.approx(0.0, abs=1e-12)
    