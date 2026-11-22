import pytest

import multicalc_py

def rosenbrock(x, y):
    return [10.0 * (y - x * x), 1.0 - x]

def test_gauss_newton_rosenbrock():
    solution, objective, _evals = multicalc_py.GaussNewton2x2().minimize(
        rosenbrock, [-1.2, 1.0]
    )
    assert solution[0] == pytest.approx(1.0, abs=1e-4)
    assert solution[1] == pytest.approx(1.0, abs=1e-4)
    assert objective == pytest.approx(0.0, abs=1e-8)

def test_levenberg_marquardt_rosenbrock():
    solution, objective, _evals = multicalc_py.LevenbergMarquardt2x2().minimize(
        rosenbrock, [-1.2, 1.0]
    )
    assert solution[0] == pytest.approx(1.0, abs=1e-4)
    assert solution[1] == pytest.approx(1.0, abs=1e-4)
    assert objective == pytest.approx(0.0, abs=1e-8)
