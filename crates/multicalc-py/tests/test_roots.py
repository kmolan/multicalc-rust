import math

import pytest

import multicalc_py

def residual(argument: float) -> float:
    return argument * argument - 2.0

def test_bisection_sqrt_two():
    root = multicalc_py.bisection(residual, 0.0, 2.0)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-9)

def test_brent_sqrt_two():
    root = multicalc_py.brent(residual, 0.0, 2.0)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-9)

def test_newton_sqrt_two():
    root = multicalc_py.newton(residual, 2.0)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-8)
