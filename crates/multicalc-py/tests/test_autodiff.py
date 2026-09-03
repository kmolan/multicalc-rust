import pytest

import multicalc_py

def test_dual_square():
    x = multicalc_py.Dual.variable(3.0)
    y = x * x
    assert y.value == pytest.approx(9.0)
    assert y.deriv == pytest.approx(6.0)

def test_hyperdual_square():
    x = multicalc_py.HyperDual.variable(3.0)
    y = x * x
    assert y.real == pytest.approx(9.0)
    assert y.eps1eps2 == pytest.approx(2.0)

def test_jet7_square():
    x = multicalc_py.Jet7.variable(3.0)
    y = x * x
    assert y.value() == pytest.approx(9.0)
    assert y.derivative(order=1) == pytest.approx(6.0)
