import pytest

import multicalc_py

def test_kalman_constant_velocity():
    filt = multicalc_py.KalmanFilter2x1(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0]],
        [[0.01, 0.0], [0.0, 0.01]],
        [[0.1]],
    )
    filt.predict()
    filt.update([1.0])
    assert filt.state()[0] > 0.0

def test_madgwick_finds_level():
    tilt = multicalc_py.SO3.exp([0.1, -0.05, 0.0])
    filt = multicalc_py.MadgwickFilter(tilt)
    still = [0.0, 0.0, 0.0]
    gravity = [0.0, 0.0, 9.81]
    north = [1.0, 0.0, 0.0]
    for _ in range(2000):
        filt.step(still, gravity, north, 0.005)
    residual = filt.orientation().act([0.0, 0.0, 1.0])
    assert residual[2] == pytest.approx(1.0, abs=1e-3)
    