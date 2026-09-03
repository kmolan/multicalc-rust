import pytest

import multicalc_py

def test_zoh_double_integrator():
    discrete_state, discrete_input = multicalc_py.zoh(
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0], [1.0]],
        0.1,
    )
    assert discrete_state[0][1] == pytest.approx(0.1, abs=1e-9)
    assert discrete_input[0][0] == pytest.approx(0.005, abs=1e-9)

def test_q_discrete_white_noise():
    covariance = multicalc_py.q_discrete_white_noise(0.1, 2.0)
    assert covariance[1][1] == pytest.approx(0.02, abs=1e-15)
