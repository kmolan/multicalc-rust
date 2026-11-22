import pytest

import multicalc_py

def test_derivative_of_cube():
    slope = multicalc_py.derivative(lambda argument: argument**3, 2.0)
    assert slope == pytest.approx(12.0, rel=1e-6)

def test_second_derivative_of_cube():
    bend = multicalc_py.second_derivative(lambda argument: argument**3, 2.0)
    assert bend == pytest.approx(12, rel=1e-4)

def test_partial_of_product():
    slope = multicalc_py.partial(lambda first, second: first * first * second, 0, [3.0, 4.0])
    assert slope == pytest.approx(24.0, rel=1e-6)

def test_integral_of_line():
    area = multicalc_py.integral(lambda argument: 2.0 * argument, 0.0, 2.0)
    assert area == pytest.approx(4.0, rel=1e-6)
