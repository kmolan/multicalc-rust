import pytest

import multicalc_py

def test_deifferential_drive_straight():
    chassis = multicalc_py.DifferentialDrive(0.05, 0.2)
    linear, angular = chassis.forward(2.0, 2.0)
    assert linear == pytest.approx(0.1)
    assert angular == pytest.approx(0.0)

def test_planar_two_link_stretched():
    tree = multicalc_py.KinematicTree2.planar_two_link(1.0)
    assert len(tree) == 2
    assert repr(tree) == "KinematicTree2(joints=2)"
    tip = tree.forward_kinematics([0.0, 0.0])
    assert tip[0] == pytest.approx(1.0)
    assert tip[1] == pytest.approx(0.0)
    assert tip[2] == pytest.approx(0.0)
