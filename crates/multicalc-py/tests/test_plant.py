import pytest

import multicalc_py

def test_quadrotor_hover_mix():
    mixer = multicalc_py.MultirotorMixer4.quadrotor_x(0.2, 0.01, 0.0, 10.0)
    thrusts, saturated = mixer.rotor_thrusts(4.0, [0.0, 0.0, 0.0])
    assert len(thrusts) == 4
    assert saturated is False

def test_rotor_lag_steps_toward_command():
    rotors = multicalc_py.RotorLag4(0.02, 0.001)
    out = rotors.stepped([2.0, 2.0, 2.0, 2.0])
    assert 0.0 < out[0] < 2.0
