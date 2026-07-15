"""Analytic calculus goldens from mpmath.

Derivatives, partials, Jacobians, Hessians, vector-field operators, and Taylor
approximations are evaluated in closed form at high precision, so multicalc's
forward-mode autodiff is compared against the exact analytic value. Each
function-valued case also stores `f_at_probe` (the function at its point) so a
Rust/Python formula divergence fails locally rather than as a loose mismatch.
"""

import mpmath

import problems
import schema

mpmath.mp.dps = 50


def _as_args(f):
    # mpmath.diff calls a multivariate function with positional coordinates; the
    # registry functions take one vector, so adapt.
    return lambda *args: f(args)


def _partial(f, point, orders):
    # Mixed partial of a vector->scalar f; orders is the per-variable count.
    return mpmath.diff(_as_args(f), tuple(point), tuple(orders))


def _unit(n, idx):
    orders = [0] * n
    orders[idx] = 1
    return orders


def _mpf_point(point):
    return [mpmath.mpf(x) for x in point]


def _differentiation(out, meta):
    key = "cube"
    x0 = 2.0
    f = problems.scalar1(key)
    probe = float(f(mpmath.mpf(x0)))
    for order in (1, 2, 3):
        value = mpmath.diff(f, x0, order)
        inputs = {
            "op": schema.string("derivative"),
            "func": schema.string(key),
            "point": schema.scalar(x0),
            "order": schema.integer(order),
        }
        expected = {
            "derivative": schema.scalar(float(value)),
            "f_at_probe": schema.scalar(probe),
        }
        schema.write_fixture(
            out, "calculus", f"cube_diff_o{order}", meta,
            {"f64/host": schema.tol(1e-12, 1e-12)}, inputs, expected,
        )


def _partials(out, meta):
    key = "g_transcendental"
    point = [1.0, 2.0, 3.0]
    f = problems.scalarn(key)
    probe = float(f(_mpf_point(point)))
    for suffix, axes in (("x", [0]), ("xy", [0, 1]), ("xxy", [0, 0, 1])):
        orders = [0, 0, 0]
        for a in axes:
            orders[a] += 1
        value = _partial(f, point, orders)
        inputs = {
            "op": schema.string("partial"),
            "func": schema.string(key),
            "point": schema.vector(point),
            "axes": schema.string(",".join(str(a) for a in axes)),
        }
        expected = {
            "partial": schema.scalar(float(value)),
            "f_at_probe": schema.scalar(probe),
        }
        schema.write_fixture(
            out, "calculus", f"g_transcendental_partial_{suffix}", meta,
            {"f64/host": schema.tol(1e-10, 1e-10)}, inputs, expected,
        )


def _jacobians(out, meta):
    for case, key, point, funcs, nvars in (
        ("jacobian_23", "jac_23", [1.0, 2.0, 3.0], 2, 3),
        ("jacobian_66", "jac_66", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6, 6),
    ):
        f = problems.vectorfn(key)
        probe = [float(x) for x in f(_mpf_point(point))]
        jac = []
        for i in range(funcs):
            row = []
            for j in range(nvars):
                comp = lambda *a, i=i: f(a)[i]
                row.append(float(mpmath.diff(comp, tuple(point), _unit(nvars, j))))
            jac.append(row)
        inputs = {
            "op": schema.string("jacobian"),
            "func": schema.string(key),
            "point": schema.vector(point),
        }
        expected = {
            "jacobian": schema.matrix(jac),
            "f_at_probe": schema.vector(probe),
        }
        schema.write_fixture(
            out, "calculus", case, meta,
            {"f64/host": schema.tol(1e-10, 1e-10)}, inputs, expected,
        )


def _hessian(out, meta):
    key = "hessian_target"
    point = [1.0, 2.0, 3.0]
    n = len(point)
    f = problems.scalarn(key)
    probe = float(f(_mpf_point(point)))
    hess = []
    for i in range(n):
        row = []
        for j in range(n):
            orders = [0] * n
            orders[i] += 1
            orders[j] += 1
            row.append(float(_partial(f, point, orders)))
        hess.append(row)
    inputs = {
        "op": schema.string("hessian"),
        "func": schema.string(key),
        "point": schema.vector(point),
    }
    expected = {"hessian": schema.matrix(hess), "f_at_probe": schema.scalar(probe)}
    schema.write_fixture(
        out, "calculus", "hessian_3x3", meta,
        {"f64/host": schema.tol(1e-10, 1e-10)}, inputs, expected,
    )


def _vector_field(out, meta):
    key = "vfield_3d"
    point = [1.0, 2.0, 3.0]
    f = problems.vectorfn(key)
    probe = [float(x) for x in f(_mpf_point(point))]

    def d(idx, var):
        comp = lambda *a, idx=idx: f(a)[idx]
        return mpmath.diff(comp, tuple(point), _unit(3, var))

    curl = [
        float(d(2, 1) - d(1, 2)),
        float(d(0, 2) - d(2, 0)),
        float(d(1, 0) - d(0, 1)),
    ]
    divergence = float(d(0, 0) + d(1, 1) + d(2, 2))
    inputs = {
        "op": schema.string("curl_div"),
        "func": schema.string(key),
        "point": schema.vector(point),
    }
    expected = {
        "curl": schema.vector(curl),
        "divergence": schema.scalar(divergence),
        "f_at_probe": schema.vector(probe),
    }
    schema.write_fixture(
        out, "calculus", "vfield_curl_div", meta,
        {"f64/host": schema.tol(1e-12, 1e-12)}, inputs, expected,
    )

    # Line and flux integrals of the field [y, -x] along the unit circle
    # (cos t, sin t) for t in [0, 2*pi]. The Rust suite drives the same field and
    # transforms through multicalc's trapezoidal line/flux integrators.
    two_pi = 2 * mpmath.pi
    px, qx = (lambda x, y: y), (lambda x, y: -x)
    xt, yt = mpmath.cos, mpmath.sin
    xp = lambda t: -mpmath.sin(t)
    yp = lambda t: mpmath.cos(t)
    line = mpmath.quad(lambda t: px(xt(t), yt(t)) * xp(t) + qx(xt(t), yt(t)) * yp(t), [0, two_pi])
    flux = mpmath.quad(lambda t: px(xt(t), yt(t)) * xp(t) - qx(xt(t), yt(t)) * yp(t), [0, two_pi])
    limits = [0.0, float(two_pi)]
    # multicalc integrates the line/flux with a default 120-point secant-trapezoidal
    # rule, so its result trails the exact analytic value by O(1/N^2) ~ 3e-3.
    tol_int = {"f64/host": schema.tol(5e-3, 5e-3)}
    schema.write_fixture(
        out, "calculus", "vfield_line_circle", meta, tol_int,
        {"op": schema.string("line_integral"), "limits": schema.vector(limits)},
        {"line_integral": schema.scalar(float(line))},
    )
    schema.write_fixture(
        out, "calculus", "vfield_flux_circle", meta, tol_int,
        {"op": schema.string("flux_integral"), "limits": schema.vector(limits)},
        {"flux_integral": schema.scalar(float(flux))},
    )


def _approximation(out, meta):
    key = "approx_target"
    p = [1.0, 2.0, 3.0]
    q = [1.1, 2.1, 2.9]
    n = len(p)
    f = problems.scalarn(key)
    fp = f(_mpf_point(p))
    grad = [_partial(f, p, _unit(n, i)) for i in range(n)]
    hess = [
        [
            _partial(f, p, [(1 if k == i else 0) + (1 if k == j else 0) for k in range(n)])
            for j in range(n)
        ]
        for i in range(n)
    ]
    dq = [mpmath.mpf(q[i]) - mpmath.mpf(p[i]) for i in range(n)]
    linear = fp + sum(grad[i] * dq[i] for i in range(n))
    quadratic = linear + mpmath.mpf("0.5") * sum(
        dq[i] * hess[i][j] * dq[j] for i in range(n) for j in range(n)
    )
    inputs = {
        "op": schema.string("approx"),
        "func": schema.string(key),
        "p": schema.vector(p),
        "q": schema.vector(q),
    }
    expected = {
        "linear_predict": schema.scalar(float(linear)),
        "quadratic_predict": schema.scalar(float(quadratic)),
        "f_at_probe": schema.scalar(float(fp)),
    }
    schema.write_fixture(
        out, "calculus", "approx_taylor", meta,
        {"f64/host": schema.tol(1e-12, 1e-12)}, inputs, expected,
    )


def run(out, seed):
    meta = schema.metadata(
        "calculus", seed,
        "closed-form analytic; mpmath high-precision derivatives",
        libraries=("mpmath",),
    )
    _differentiation(out, meta)
    _partials(out, meta)
    _jacobians(out, meta)
    _hessian(out, meta)
    _vector_field(out, meta)
    _approximation(out, meta)
