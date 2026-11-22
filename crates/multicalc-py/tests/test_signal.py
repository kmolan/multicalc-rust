import pytest

import multicalc_py

def test_one_pole_passthrough():
    filt = multicalc_py.OnePoleLowPass(1.0)
    assert filt.filter(3.0) == 3.0
    assert filt.filter(-2.0) == -2.0

def test_deadband_plain():
    band = multicalc_py.Deadband.plain(0.1)
    assert abs(band.apply(0.05)) < 1e-12
    assert band.apply(0.5) == pytest.approx(0.5)

def test_moving_average_four():
    filt = multicalc_py.MovingAverage4()
    assert filt.filter(1.0) == 1.0
    assert filt.filter(5.0) == 2.0