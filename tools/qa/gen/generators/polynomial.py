"""Polynomial goldens: reference values from numpy.

Every fixture carries a `kind` input saying what was computed, so one directory
covers the whole module — evaluation, calculus, products, composition,
interpolation, fitting, exact real roots, and polynomials in several variables.

`numpy.polynomial.polynomial` stores coefficients lowest power first, which is
how multicalc stores them, so those routines are used directly. `numpy.roots` is
the exception: it takes coefficients highest power first, so the generator
reverses on the way in and the roots come back needing a sort.

The multivariate route is deliberately unlike ours. Each sparse term list is
expanded into a dense coefficient grid and evaluated with `polyval2d`/`polyval3d`,
which is nested repeated multiply-and-add over the grid rather than a power per
term. Matching values therefore mean two different layouts agree, not that the
same code ran twice.
"""

import numpy as np
import numpy.polynomial.polynomial as npoly

import schema

# Evaluation, coefficients, areas and multivariate values.
EXACT = (1e-12, 1e-12)
# numpy finds roots as the eigenvalues of a companion matrix, which is only good
# to about this; the closed forms multicalc uses are better, so the tolerance is
# set by the oracle rather than by us.
ROOTS = (1e-9, 1e-9)
# A least-squares fit inherits the conditioning of its design matrix.
FIT = (1e-8, 1e-8)

F32 = (1e-4, 1e-4)


def _write(out, meta, case, kind, inputs, expected, tol, *, equation, operations, f32=True):
    tolerances = {"f64": schema.tol(*tol)}
    if f32:
        tolerances["f32"] = schema.tol(*F32)
    schema.write_fixture(
        out,
        "polynomial",
        case,
        meta,
        tolerances,
        {"kind": schema.string(kind), **inputs},
        expected,
        equation=equation,
        operations=operations,
    )


def _dense_grid(terms, variable_count):
    """A sparse term list as the dense coefficient grid numpy's polyval2d/3d want."""
    shape = [max(term[1][v] for term in terms) + 1 for v in range(variable_count)]
    grid = np.zeros(shape)
    for coefficient, exponents in terms:
        grid[tuple(exponents)] += coefficient
    return grid


def _terms_value(terms, variable_count):
    """Inputs describing a sparse polynomial: one row per term, coefficient then powers."""
    rows = [[float(coefficient)] + [float(e) for e in exponents] for coefficient, exponents in terms]
    return {
        "terms": schema.matrix(rows),
        "variable_count": schema.integer(variable_count),
    }


def _sorted_terms(grid):
    """A dense grid back to a sorted term list, so a comparison cannot depend on order."""
    rows = []
    for index in np.ndindex(grid.shape):
        value = float(grid[index])
        if value != 0.0:
            rows.append([value] + [float(i) for i in index])
    # Sorted by powers so both sides agree on order without the Rust side sorting.
    rows.sort(key=lambda row: row[1:])
    return rows


def run(out, seed):
    meta = schema.metadata(
        "polynomial",
        seed,
        "hand-chosen coefficients and factored polynomials with known roots",
        libraries=("numpy",),
        reference="NumPy {numpy}",
    )

    # ---- evaluation ---------------------------------------------------------

    degree7 = [0.2, -1.1, 0.4, 2.0, -0.3, 0.05, 0.9, -0.15]
    points = [-1.3, -0.25, 0.0, 0.37, 1.8]
    orders = []
    for point in points:
        row = []
        working = np.array(degree7, dtype=float)
        for _ in range(4):
            row.append(float(npoly.polyval(point, working)))
            working = npoly.polyder(working)
        orders.append(row)
    _write(
        out, meta, "horner_degree7", "evaluation",
        {"coefficients": schema.vector(degree7), "points": schema.vector(points)},
        {"orders": schema.matrix(orders)},
        EXACT,
        equation="p(x) = 0.2 − 1.1x + 0.4x² + 2x³ − 0.3x⁴ + 0.05x⁵ + 0.9x⁶ − 0.15x⁷",
        operations=["value and first three derivatives at five points"],
    )

    # ---- calculus -----------------------------------------------------------

    degree5 = [2.0, -3.5, 0.75, 4.0, -1.25, 0.5]
    _write(
        out, meta, "derivative_degree5", "coefficients",
        {"coefficients": schema.vector(degree5)},
        {"result": schema.vector(list(npoly.polyder(degree5)) + [0.0])},
        EXACT,
        equation="p(x) = 2 − 3.5x + 0.75x² + 4x³ − 1.25x⁴ + 0.5x⁵",
        operations=["derivative coefficients"],
    )

    lower, upper = -0.6, 2.3
    antiderivative = npoly.polyint(degree5)
    area = float(npoly.polyval(upper, antiderivative) - npoly.polyval(lower, antiderivative))
    _write(
        out, meta, "definite_integral_degree5", "scalar",
        {"coefficients": schema.vector(degree5), "bounds": schema.vector([lower, upper])},
        {"result": schema.scalar(area)},
        EXACT,
        equation="∫ (2 − 3.5x + 0.75x² + 4x³ − 1.25x⁴ + 0.5x⁵) dx over [−0.6, 2.3]",
        operations=["area under the curve"],
    )

    # ---- products and composition -------------------------------------------

    left = [1.0, 2.0, -1.0, 0.5]
    right = [3.0, -1.0, 4.0, 0.25, -0.75]
    _write(
        out, meta, "product_degree3_by_degree4", "coefficients",
        {"coefficients": schema.vector(left), "other": schema.vector(right)},
        {"result": schema.vector(list(npoly.polymul(left, right)))},
        EXACT,
        equation="(1 + 2x − x² + 0.5x³)(3 − x + 4x² + 0.25x³ − 0.75x⁴)",
        operations=["product coefficients"],
    )

    outer = [2.0, -1.0, 0.5, 0.25]
    inner = [1.0, 3.0, -0.5]
    composed = npoly.Polynomial(outer)(npoly.Polynomial(inner)).coef
    _write(
        out, meta, "composition_degree3_in_degree2", "coefficients",
        {"coefficients": schema.vector(outer), "other": schema.vector(inner)},
        {"result": schema.vector([float(c) for c in composed])},
        EXACT,
        equation="p(q(x)) with p = 2 − x + 0.5x² + 0.25x³ and q = 1 + 3x − 0.5x²",
        operations=["composition coefficients"],
    )

    # ---- exact real roots ---------------------------------------------------

    def real_roots(ascending):
        # numpy.roots wants the highest power first, the opposite of how the rest
        # of this file (and multicalc) stores coefficients.
        found = np.roots(list(ascending)[::-1])
        real = sorted(float(r.real) for r in found if abs(r.imag) < 1e-9)
        return real

    root_cases = [
        ("roots_quadratic", [2.0, -3.0, 1.0], "(x − 1)(x − 2)"),
        ("roots_quadratic_dominant_linear", [1.0, 1e8, 1.0], "x² + 1e8·x + 1"),
        ("roots_cubic_three_real", [-6.0, 11.0, -6.0, 1.0], "(x − 1)(x − 2)(x − 3)"),
        ("roots_cubic_one_real", [1.0, 1.0, 0.0, 1.0], "x³ + x + 1"),
        ("roots_quartic_four_real", [0.0, -8.0, 14.0, -7.0, 1.0], "x(x − 1)(x − 2)(x − 4)"),
        ("roots_quartic_two_real", [-2.0, 1.0, -1.0, 1.0, 1.0], "(x² + 1)(x − 1)(x + 2)"),
    ]
    for case, coefficients, equation in root_cases:
        found = real_roots(coefficients)
        _write(
            out, meta, case, "roots",
            {"coefficients": schema.vector(coefficients)},
            {"roots": schema.vector(found)},
            ROOTS,
            equation=equation,
            operations=["real roots, ascending"],
        )

    degree6 = npoly.polyfromroots([-3.0, -1.0, 0.5, 2.0, 4.0, 7.0])
    _write(
        out, meta, "roots_degree6_sturm", "roots_any_degree",
        {"coefficients": schema.vector([float(c) for c in degree6])},
        {
            "roots": schema.vector(real_roots(degree6)),
            "count": schema.integer(6),
        },
        ROOTS,
        equation="(x + 3)(x + 1)(x − 0.5)(x − 2)(x − 4)(x − 7)",
        operations=["six real roots past where a formula exists", "how many roots a range holds"],
    )

    # ---- building from data -------------------------------------------------

    nodes = [-2.0, -0.5, 0.75, 1.5, 3.0]
    source = [2.0, -1.0, 0.5, -0.25, 0.125]
    values = [float(npoly.polyval(node, source)) for node in nodes]
    _write(
        out, meta, "interpolation_five_points", "coefficients",
        {"nodes": schema.vector(nodes), "values": schema.vector(values)},
        {"result": schema.vector(source)},
        EXACT,
        equation="the one quartic through five points off 2 − x + 0.5x² − 0.25x³ + 0.125x⁴",
        operations=["interpolating coefficients"],
    )

    rng = np.random.default_rng(seed)
    cubic = [1.5, -2.0, 0.75, 0.25]
    fit_nodes = [-3.0 + 0.4 * index for index in range(20)]
    fit_values = [
        float(npoly.polyval(node, cubic)) + float(rng.normal(0.0, 0.05)) for node in fit_nodes
    ]
    fitted = npoly.polyfit(fit_nodes, fit_values, 3)
    _write(
        out, meta, "least_squares_cubic_fit", "coefficients",
        {"nodes": schema.vector(fit_nodes), "values": schema.vector(fit_values)},
        {"result": schema.vector([float(c) for c in fitted])},
        FIT,
        equation="the closest cubic to twenty samples off 1.5 − 2x + 0.75x² + 0.25x³ with noise",
        operations=["least-squares coefficients"],
    )

    # ---- several variables --------------------------------------------------

    bivariate = [(3.0, (2, 1)), (2.0, (1, 1)), (-1.0, (0, 0))]
    bivariate_grid = _dense_grid(bivariate, 2)
    bivariate_points = [
        [1.5, -2.0], [0.0, 0.0], [-0.75, 3.25], [2.0, 1.0], [-1.5, -1.5], [0.4, 0.9],
    ]
    _write(
        out, meta, "bivariate_evaluation", "multivariate_evaluation",
        {**_terms_value(bivariate, 2), "points": schema.matrix(bivariate_points)},
        {
            "values": schema.vector(
                [float(npoly.polyval2d(x, y, bivariate_grid)) for x, y in bivariate_points]
            )
        },
        EXACT,
        equation="3x²y + 2xy − 1",
        operations=["values at six points, from a dense grid"],
    )

    partials = []
    for x, y in bivariate_points:
        in_x = npoly.polyder(bivariate_grid, axis=0)
        in_y = npoly.polyder(bivariate_grid, axis=1)
        partials.append(
            [float(npoly.polyval2d(x, y, in_x)), float(npoly.polyval2d(x, y, in_y))]
        )
    _write(
        out, meta, "bivariate_partial_derivatives", "multivariate_partials",
        {**_terms_value(bivariate, 2), "points": schema.matrix(bivariate_points)},
        {"partials": schema.matrix(partials)},
        EXACT,
        equation="3x²y + 2xy − 1",
        operations=["both partial derivatives at six points"],
    )

    trivariate = [
        (1.5, (2, 1, 0)), (-0.5, (0, 2, 1)), (2.0, (1, 1, 1)),
        (0.25, (3, 0, 0)), (-1.0, (0, 0, 2)), (0.75, (1, 0, 1)),
    ]
    trivariate_grid = _dense_grid(trivariate, 3)
    trivariate_points = [
        [1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [-0.5, 1.5, 0.25],
        [2.0, -1.0, 1.0], [0.3, 0.7, -0.4], [-1.2, -0.8, 2.0],
    ]
    _write(
        out, meta, "trivariate_evaluation", "multivariate_evaluation",
        {**_terms_value(trivariate, 3), "points": schema.matrix(trivariate_points)},
        {
            "values": schema.vector(
                [
                    float(npoly.polyval3d(x, y, z, trivariate_grid))
                    for x, y, z in trivariate_points
                ]
            )
        },
        EXACT,
        equation="1.5x²y − 0.5y²z + 2xyz + 0.25x³ − z² + 0.75xz",
        operations=["values at six points in three variables"],
    )

    product_left = [(2.0, (1, 0)), (-1.0, (0, 1)), (3.0, (0, 0))]
    product_right = [(1.0, (1, 1)), (0.5, (2, 0)), (-2.0, (0, 0))]
    # A dense grid multiply is exactly a two-dimensional convolution.
    product_grid = np.zeros((
        _dense_grid(product_left, 2).shape[0] + _dense_grid(product_right, 2).shape[0] - 1,
        _dense_grid(product_left, 2).shape[1] + _dense_grid(product_right, 2).shape[1] - 1,
    ))
    left_grid = _dense_grid(product_left, 2)
    right_grid = _dense_grid(product_right, 2)
    for i in np.ndindex(left_grid.shape):
        for j in np.ndindex(right_grid.shape):
            product_grid[i[0] + j[0], i[1] + j[1]] += left_grid[i] * right_grid[j]
    _write(
        out, meta, "bivariate_product", "multivariate_product",
        {
            **_terms_value(product_left, 2),
            "other_terms": schema.matrix(
                [[float(c)] + [float(e) for e in ex] for c, ex in product_right]
            ),
        },
        {"result_terms": schema.matrix(_sorted_terms(product_grid))},
        EXACT,
        equation="(2x − y + 3)(xy + 0.5x² − 2)",
        operations=["the collected product's terms"],
    )

    fixed_y = 1.7
    # Fixing y collapses the grid along that axis, leaving coefficients in x alone.
    collapsed = [
        float(sum(bivariate_grid[power, k] * fixed_y**k for k in range(bivariate_grid.shape[1])))
        for power in range(bivariate_grid.shape[0])
    ]
    _write(
        out, meta, "bivariate_substitution", "multivariate_substitution",
        {**_terms_value(bivariate, 2), "value": schema.scalar(fixed_y), "variable": schema.integer(1)},
        {"result": schema.vector(collapsed)},
        EXACT,
        equation="3x²y + 2xy − 1 with y fixed at 1.7",
        operations=["the one-variable coefficients left behind"],
    )
