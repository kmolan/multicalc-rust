import math
import struct

import pytest

import multicalc_py

def hex_f64(value: str) -> float:
    return struct.unpack("d", struct.pack("Q", int(value, 16)))[0]

def close(got: float, want: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(got - want) <= abs_tol + rel_tol * max(abs(got), abs(want))

def test_version():
    from importlib.metadata import version
    assert multicalc_py.version() == version("multicalc_py")

def test_vector_dot():
    a = multicalc_py.Vector4([1.0, 2.0, 3.0, 4.0])
    b = multicalc_py.Vector4([4.0, 3.0, 2.0, 1.0])
    assert a.dot(b) == pytest.approx(20.0)

def test_svd_golden():
    # Bit-exact input from tools/qa/fixtures/linalg/svd_3x3.json
    rows = [
        "0x3fe2763f5d1ce592",
        "0xbfd8d914ad03f36c",
        "0x3fb25b646e3094b0",
        "0x3fe622dfc7a9f89e",
        "0x3fd9e223541c2004",
        "0xbfe62d633c4f7204",
        "0xbfef15212fa242ca",
        "0x3f86822382d5fe80",
        "0xbfe18282767f4e98",
    ]
    matrix = [
        [hex_f64(rows[0]), hex_f64(rows[1]), hex_f64(rows[2])],
        [hex_f64(rows[3]), hex_f64(rows[4]), hex_f64(rows[5])],
        [hex_f64(rows[6]), hex_f64(rows[7]), hex_f64(rows[8])],
    ]
    expected = [
        hex_f64("0x3ff5400d80d4d692"),
        hex_f64("0x3feefaabc5b78ecf"),
        hex_f64("0x3fd907ffa3d1e959"),
    ]
    got = multicalc_py.Matrix3(matrix).svd()
    assert len(got) == 3
    for value, want in zip(got, expected):
        assert close(value, want, 1e-11, 1e-10)

def test_lu_singular():
    with pytest.raises(multicalc_py.LinalgError, match="Singular"):
        multicalc_py.Matrix3.zeros().lu_decompose()

def test_cholesky_not_positive_definite():
    matrix = multicalc_py.Matrix2([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(multicalc_py.LinalgError, match="NotPositiveDefinite"):
        matrix.cholesky()

def test_pid_step():
    dt = 0.01
    setpoint = 1.0
    pid = multicalc_py.Pid(2.0, 1.0, 0.0, dt)
    first = pid.update(setpoint, 0.0)
    assert first == pytest.approx(2.01, abs=1e-12)

    measurement = dt * first
    for _ in range(100):
        previous = measurement
        output = pid.update(setpoint, measurement)
        measurement += dt * output
        assert measurement > previous
    assert measurement == pytest.approx(setpoint, abs=0.01)

def test_polynomial_evaluate():
    coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    poly = multicalc_py.Polynomial8(coeffs)
    value, slope, bend = poly.evaluate_with_derivatives(0.5)
    assert value == pytest.approx(3.921875, abs=1e-12)
    assert slope == pytest.approx(14.5625, abs=1e-12)
    assert bend == pytest.approx(71.625, abs=1e-12)

def test_polynomial_quadratic_roots():
    poly = multicalc_py.Polynomial3([2.0, -3.0, 1.0])
    roots = poly.real_roots()
    assert len(roots) == 2
    assert roots[0] == pytest.approx(1.0, abs=1e-12)
    assert roots[1] == pytest.approx(2.0, abs=1e-12)

def test_so3_rotation():
    half_pi = math.pi / 2.0
    rot = multicalc_py.SO3.exp([0.0, 0.0, half_pi])
    point = rot.act([1.0, 0.0, 0.0])
    assert point[0] == pytest.approx(0.0, abs=1e-12)
    assert point[1] == pytest.approx(1.0, abs=1e-12)
    assert point[2] == pytest.approx(0.0, abs=1e-12)
