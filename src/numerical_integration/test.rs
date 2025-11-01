use crate::numerical_integration::mode::*;

use crate::numerical_integration::gaussian_integration;
use crate::numerical_integration::integrator::*;
use crate::numerical_integration::iterative_integration;
use crate::utils::error_codes::*;

#[test]
fn test_booles_integration_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 {
        return 2.0 * args;
    };

    let integration_limit = [0.0, 2.0];

    let integrator =
        iterative_integration::SingleVariableSolver::from_parameters(100, IterativeMethod::Booles);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 1e-14);
}

#[test]
fn test_booles_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 {
        return 2.0 * args[0] + args[1] * args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator =
        iterative_integration::MultiVariableSolver::from_parameters(100, IterativeMethod::Booles);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 1e-25);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = integrator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = integrator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_booles_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 {
        return 6.0 * args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator =
        iterative_integration::SingleVariableSolver::from_parameters(20, IterativeMethod::Booles);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_gauss_legendre_quadrature_integration_1() {
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = |args: f64| -> f64 {
        return 4.0 * args * args * args - 3.0 * args * args;
    };

    let integration_limit = [0.0, 2.0];

    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        4,
        GaussianQuadratureMethod::GaussLegendre,
    );

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 8.0) < 1e-14);
}

#[test]
fn test_gauss_legendre_quadrature_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 {
        return 2.0 * args[0] + args[1] * args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator = gaussian_integration::MultiVariableSolver::from_parameters(
        2,
        GaussianQuadratureMethod::GaussLegendre,
    );

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 1e-14);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = integrator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 1e-14);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = integrator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 1e-14);
}

#[test]
fn test_gauss_legendre_quadrature_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 {
        return 6.0 * args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        2,
        GaussianQuadratureMethod::GaussLegendre,
    );

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 1e-14);
}

/// ∫_{-∞}^{∞} e^{-x²} dx = √π ≈ 1.77245
#[test]
fn test_gauss_hermite_1() {
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        1,
        GaussianQuadratureMethod::GaussHermite,
    );

    let func = |_x: f64| 1.0; // f(x) = 1
    let bounds = [[-10.0, 10.0]];
    let result = integrator.get(1, &func, &bounds).unwrap();
    let expected = core::f64::consts::PI.sqrt();

    assert!(
        (result - expected).abs() < 1e-10,
        " got {}, expected {}",
        result,
        expected
    );
}

/// ∫_{-∞}^{∞} x² e^{-x²} dx = √π / 2 ≈ 0.88623
#[test]
fn test_gauss_hermite_2() {
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        2, // order for the quadrature
        GaussianQuadratureMethod::GaussHermite,
    );

    let func = |x: f64| x * x;
    let bounds = [[-10.0, 10.0]];
    let result = integrator.get(1, &func, &bounds).unwrap();
    let expected = core::f64::consts::PI.sqrt() / 2.0;

    assert!(
        (result - expected).abs() < 1e-5,
        " got {}, expected {}",
        result,
        expected
    );
}

/// ∫∫ e^{-(x² + y²)} dx dy = π
#[test]
fn test_gauss_hermite_3() {
    let solver = gaussian_integration::MultiVariableSolver::from_parameters(
        10, //order of the quadrature
        GaussianQuadratureMethod::GaussHermite,
    );

    let func = |_xy: &[f64; 2]| 1.0; // f(x, y) = 1

    let idx_to_integrate = [0, 1];
    let limits = [[-10.0, 10.0], [-10.0, 10.0]];
    let point = [0.0, 0.0];

    let result = solver
        .get(2, idx_to_integrate, &func, &limits, &point)
        .unwrap();
    let expected = core::f64::consts::PI;

    assert!(
        (result - expected).abs() < 1e-5,
        " got {}, expected {}",
        result,
        expected
    );
}

/// ∫₀^∞ e^{-x} dx = 1
#[test]
fn test_gauss_laguerre_1() {
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        1, // order for the quadrature
        GaussianQuadratureMethod::GaussLaguerre,
    );

    let func = |_x: f64| 1.0; // f(x) = 1
    let bounds = [[0.0, 10.0]];
    let result = integrator.get(1, &func, &bounds).unwrap();
    let expected = 1.0;

    assert!(
        (result - expected).abs() < 1e-5,
        " got {}, expected {}",
        result,
        expected
    );
}

/// ∫₀^∞ x e^{-x} dx = 1
#[test]
fn test_gauss_laguerre_2() {
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        1, // order for the quadrature
        GaussianQuadratureMethod::GaussLaguerre,
    );

    let func = |x: f64| x;
    let bounds = [[0.0, 10.0]];
    let result = integrator.get(1, &func, &bounds).unwrap();
    let expected = 1.0;

    assert!(
        (result - expected).abs() < 1e-5,
        " got {}, expected {}",
        result,
        expected
    );
}

/// ∫₀^∞ x² e^{-x} dx = 2
#[test]
fn test_gauss_laguerre_3() {
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        2, // order for the quadrature
        GaussianQuadratureMethod::GaussLaguerre,
    );

    let func = |x: f64| x * x;
    let bounds = [[0.0, 10.0]];
    let result = integrator.get(1, &func, &bounds).unwrap();
    let expected = 2.0;

    assert!(
        (result - expected).abs() < 1e-5,
        " got {}, expected {}",
        result,
        expected
    );
}

/// ∫₀^∞∫₀^∞ e^{-(x + y)} dx dy = 1
#[test]
fn test_gauss_laguerre_4() {
    let solver = gaussian_integration::MultiVariableSolver::from_parameters(
        10, // order for the quadrature
        GaussianQuadratureMethod::GaussLaguerre,
    );

    let func = |_xy: &[f64; 2]| 1.0; // f(x, y) = 1

    let idx_to_integrate = [0, 1];
    let limits = [[0.0, 10.0], [0.0, 10.0]];
    let point = [0.0, 0.0];

    let result = solver
        .get(2, idx_to_integrate, &func, &limits, &point)
        .unwrap();
    let expected = 1.0;

    assert!(
        (result - expected).abs() < 1e-5,
        " got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simpsons_integration_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 {
        return 2.0 * args;
    };

    let integration_limit = [0.0, 2.0];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(
        200,
        IterativeMethod::Simpsons,
    );

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 0.05);
}

#[test]
fn test_simpsons_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 {
        return 2.0 * args[0] + args[1] * args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator =
        iterative_integration::MultiVariableSolver::from_parameters(200, IterativeMethod::Simpsons);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 0.05);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = integrator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 0.05);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = integrator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 0.05);
}

#[test]
fn test_simpsons_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 {
        return 6.0 * args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(
        200,
        IterativeMethod::Simpsons,
    );

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_simpsons_integration_4() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 {
        return 2.0 * args[0] + args[1] * args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 1.0, 1.0];

    let integrator =
        iterative_integration::MultiVariableSolver::from_parameters(200, IterativeMethod::Simpsons);

    //double partial integration for first x then y, expect a value of ~1.50
    let val = integrator
        .get_double_partial(&func, [0, 1], &integration_limits, &point)
        .unwrap();
    assert!(f64::abs(val - 1.50) < 0.05);
}

#[test]
fn test_trapezoidal_integration_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 {
        return 2.0 * args;
    };

    let integration_limit = [0.0, 2.0];

    let iterator = iterative_integration::SingleVariableSolver::from_parameters(
        100,
        IterativeMethod::Trapezoidal,
    );
    let val = iterator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 {
        return 2.0 * args[0] + args[1] * args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let iterator = iterative_integration::MultiVariableSolver::from_parameters(
        100,
        IterativeMethod::Trapezoidal,
    );

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = iterator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = iterator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = iterator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 {
        return 6.0 * args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(
        100,
        IterativeMethod::Trapezoidal,
    );

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.0001);
}

#[test]
fn test_trapezoidal_integration_4() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 {
        return 2.0 * args[0] + args[1] * args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 2.0]];
    let point = [1.0, 2.0, 3.0];

    let integrator = iterative_integration::MultiVariableSolver::from_parameters(
        10,
        IterativeMethod::Trapezoidal,
    );

    //double partial integration for first x then y, expect a value of ~2.50
    let val = integrator
        .get_double_partial(&func, [0, 1], &integration_limits, &point)
        .unwrap();
    assert!(f64::abs(val - 8.0) < 0.00001);
}

// === ∫₀^∞ e^{-x} dx = 1 ===
#[test]
fn test_infinite_limits_single_variable_1() {
    let methods = [
        IterativeMethod::Trapezoidal,
        IterativeMethod::Simpsons,
        IterativeMethod::Booles,
    ];

    let func = |x: f64| -> f64 {
        return (-x).exp();
    };

    let integration_limits = [0.0, f64::INFINITY];

    for method in methods {
        let iterator = iterative_integration::SingleVariableSolver::from_parameters(1000, method);
        let result = iterator.get_single(&func, &integration_limits).unwrap();

        let expected = 1.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "{:?}: got {}, expected {} for exp(-(x+y))",
            method,
            result,
            expected
        );
    }
}

// === ∫₋∞^∞ e^{-x²} dx = √π ≈ 1.77245385091 ===
#[test]
fn test_infinite_limits_single_variable_2() {
    let methods = [
        IterativeMethod::Trapezoidal,
        IterativeMethod::Simpsons,
        IterativeMethod::Booles,
    ];

    let func = |x: f64| -> f64 {
        return (-x * x).exp();
    };

    let integration_limits = [f64::NEG_INFINITY, f64::INFINITY];

    for method in methods {
        let iterator = iterative_integration::SingleVariableSolver::from_parameters(1000, method);
        let result = iterator.get_single(&func, &integration_limits).unwrap();

        let expected = core::f64::consts::PI.sqrt();
        assert!(
            (result - expected).abs() < 1e-5,
            "{:?}: got {}, expected {} for exp(-(x+y))",
            method,
            result,
            expected
        );
    }
}

// === ∫₀^∞ (1 / (1 + x²)) dx = π/2 ===
#[test]
fn test_infinite_limits_single_variable_3() {
    let methods = [
        IterativeMethod::Trapezoidal,
        IterativeMethod::Simpsons,
        IterativeMethod::Booles,
    ];

    let func = |x: f64| -> f64 {
        return 1.0 / (1.0 + x * x);
    };

    let integration_limits = [0.0, f64::INFINITY];

    for method in methods {
        let iterator = iterative_integration::SingleVariableSolver::from_parameters(1000, method);
        let result = iterator.get_single(&func, &integration_limits).unwrap();

        let expected = core::f64::consts::FRAC_PI_2;
        assert!(
            (result - expected).abs() < 1e-3,
            "{:?}: got {}, expected {} for exp(-(x+y))",
            method,
            result,
            expected
        );
    }
}

// === ∫₀^∞ ∫₀^∞ e^{-(x + y)} dx dy = 1 ===
// (Product of two independent exp(-x) integrals, each = 1)
#[test]
fn test_double_infinite_integrals_multivariable_1() {
    let methods = [
        IterativeMethod::Trapezoidal,
        IterativeMethod::Simpsons,
        IterativeMethod::Booles,
    ];

    let func = |args: &[f64; 2]| (-args[0] - args[1]).exp();
    let integration_limits = [[0.0, f64::INFINITY], [0.0, f64::INFINITY]];
    let point = [0.0, 0.0];
    let idx_to_integrate = [0, 1];

    for method in methods {
        let iterator = iterative_integration::MultiVariableSolver::from_parameters(1000, method);
        let result = iterator
            .get(2, idx_to_integrate, &func, &integration_limits, &point)
            .unwrap();

        let expected = 1.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "{:?}: got {}, expected {} for exp(-(x+y))",
            method,
            result,
            expected
        );
    }
}

// === ∫₋∞^∞ ∫₋∞^∞ e^{-(x² + y²)} dx dy = π ===
// (Product of two Gaussian integrals, each = √π)
#[test]
fn test_double_infinite_integrals_multivariable_2() {
    let methods = [
        IterativeMethod::Trapezoidal,
        IterativeMethod::Simpsons,
        IterativeMethod::Booles,
    ];

    let func = |args: &[f64; 2]| (-(args[0] * args[0] + args[1] * args[1])).exp();
    let integration_limits = [
        [f64::NEG_INFINITY, f64::INFINITY],
        [f64::NEG_INFINITY, f64::INFINITY],
    ];
    let point = [0.0, 0.0];
    let idx_to_integrate = [0, 1];

    for method in methods {
        let iterator = iterative_integration::MultiVariableSolver::from_parameters(1000, method);
        let result = iterator
            .get(2, idx_to_integrate, &func, &integration_limits, &point)
            .unwrap();

        let expected = core::f64::consts::PI;
        assert!(
            (result - expected).abs() < 1e-5,
            "{:?}: got {}, expected {} for exp(-(x²+y²))",
            method,
            result,
            expected
        );
    }
}

// === ∫₀^∞ ∫₀^∞ (1 / ((1 + x²) * (1 + y²))) dx dy = (π/2)² ===
#[test]
fn test_double_infinite_integrals_multivariable_3() {
    let methods = [
        IterativeMethod::Trapezoidal,
        IterativeMethod::Simpsons,
        IterativeMethod::Booles,
    ];

    let func = |args: &[f64; 2]| 1.0 / ((1.0 + args[0] * args[0]) * (1.0 + args[1] * args[1]));
    let integration_limits = [[0.0, f64::INFINITY], [0.0, f64::INFINITY]];
    let point = [0.0, 0.0];
    let idx_to_integrate = [0, 1];

    for method in methods {
        let iterator = iterative_integration::MultiVariableSolver::from_parameters(1000, method);
        let result = iterator
            .get(2, idx_to_integrate, &func, &integration_limits, &point)
            .unwrap();

        let expected = (core::f64::consts::FRAC_PI_2).powi(2);
        assert!(
            (result - expected).abs() < 1e-3,
            "{:?}: got {}, expected {} for rational function",
            method,
            result,
            expected
        );
    }
}

#[test]
fn test_error_checking_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 {
        return 2.0 * args;
    };

    let integration_limit = [10.0, 1.0];

    let integrator = iterative_integration::SingleVariableSolver::default();

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == INTEGRATION_LIMITS_ILL_DEFINED);
}

#[test]
fn test_error_checking_2() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 {
        return 2.0 * args;
    };

    let integration_limit = [0.0, 1.0];

    let integrator =
        iterative_integration::SingleVariableSolver::from_parameters(0, IterativeMethod::Booles);

    //expect failure because number of steps is 0
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
}

//TODO: add more tests

#[test]
fn test_error_checking_3() {
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = |args: f64| -> f64 {
        return 4.0 * args * args * args - 3.0 * args * args;
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n < 1
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        0,
        GaussianQuadratureMethod::GaussLegendre,
    );
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
}

#[test]
fn test_error_checking_4() {
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = |args: f64| -> f64 {
        return 4.0 * args * args * args - 3.0 * args * args;
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n > 30
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(
        31,
        GaussianQuadratureMethod::GaussLegendre,
    );
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
}
