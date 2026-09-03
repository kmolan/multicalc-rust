import pytest
import multicalc_py

def test_derivative_reraises_callback_exception():
    def bad(x):
        raise ValueError("domain error")

    with pytest.raises(ValueError, match="domain error"):
        multicalc_py.derivative(bad, 1.0)

def test_bisection_reraises_callback_exception():
    def bad(x):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        multicalc_py.bisection(bad, 0.0, 1.0)

def test_rk4_step_reraises_callback_exception():
    def bad(t, state):
        raise ValueError("ode failed")

    with pytest.raises(ValueError, match="ode failed"):
        multicalc_py.rk4_step(bad, 0.0, [1.0, 0.0], 0.1)
