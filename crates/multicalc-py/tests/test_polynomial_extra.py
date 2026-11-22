import pytest

import multicalc_py

def test_piecewise_handover():
    curve = multicalc_py.PiecewisePolynomial2([[0.0, 1.0], [1.0, 2.0]], [2.0, 1.0])
    assert curve.evaluate(2.0) == pytest.approx(1.0, abs=1e-12)
    assert curve.evaluate(9.0) == pytest.approx(3.0, abs=1e-12)

def test_multivariate_point():
    polynomial = multicalc_py.MultivariatePolynomial2(
        [(3.0, [2, 1]), (2.0, [1, 1]), (-1.0, [0, 0])]
    )
    assert polynomial.evaluate([2.0, 3.0]) == pytest.approx(47.0, abs=1e-12)

def test_linear_root():
    roots = multicalc_py.Polynomial2([-6.0, 2.0]).real_roots()
    assert roots[0] == pytest.approx(3.0, abs=1e-12)
