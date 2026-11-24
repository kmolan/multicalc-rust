//! The math checks of the arm app: linear algebra, autodiff, PID and the error path.

use core::hint::black_box;

use multicalc::control::Pid;
use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Vector};
use multicalc::numerical_derivative::{AutoDiffSingle, DerivatorSingleVariable};
use multicalc::scalar_fn;

use super::{CheckOutcome, close};
use crate::cycles;
use crate::fixtures;

/// The known answer of the dot product.
const VECTOR_DOT_EXPECTED: f64 = 20.0;

/// Identity: `[1,2,3,4]·[4,3,2,1] = 20`, the atomics-free library path.
pub fn vector_dot() -> CheckOutcome {
    let left = black_box(Vector::new([1.0_f64, 2.0, 3.0, 4.0]));
    let right = black_box(Vector::new([4.0_f64, 3.0, 2.0, 1.0]));

    let start = cycles::read();
    let product = black_box(left.dot(right));
    let elapsed = cycles::read().wrapping_sub(start);

    CheckOutcome {
        name: "vector_dot",
        passed: close(product, VECTOR_DOT_EXPECTED, 1e-12, 0.0),
        cycles: elapsed,
        value: product,
    }
}

/// Golden: singular values of the `linalg/svd_3x3` fixture, descending.
pub fn svd_singular_values() -> CheckOutcome {
    let matrix: Matrix3D = black_box(Matrix::new(fixtures::SVD_3X3_INPUT));

    let start = cycles::read();
    let decomposition = black_box(matrix.svd());
    let elapsed = cycles::read().wrapping_sub(start);

    let Ok(decomposition) = decomposition else {
        return CheckOutcome {
            name: "svd_singular_values",
            passed: false,
            cycles: elapsed,
            value: f64::NAN,
        };
    };

    let singular_values = decomposition.singular_values();
    let mut passed = true;
    for index in 0..3 {
        passed &= close(
            singular_values[index],
            fixtures::SVD_3X3_SINGULAR_VALUES[index],
            fixtures::SVD_3X3_ABS,
            fixtures::SVD_3X3_REL,
        );
    }

    CheckOutcome {
        name: "svd_singular_values",
        passed,
        cycles: elapsed,
        value: singular_values[0],
    }
}

/// Identity: forward-mode autodiff of `x³` at `x = 2` is 12.
pub fn autodiff_derivative() -> CheckOutcome {
    let cubic = scalar_fn!(|x| x * x * x);
    let derivator = AutoDiffSingle::default();
    let point = black_box(2.0_f64);

    let start = cycles::read();
    let derivative = black_box(derivator.differentiate(1, &cubic, point));
    let elapsed = cycles::read().wrapping_sub(start);

    let (passed, value) = match derivative {
        Ok(slope) => (close(slope, 12.0, 1e-12, 0.0), slope),
        Err(_) => (false, f64::NAN),
    };

    CheckOutcome {
        name: "autodiff_derivative",
        passed,
        cycles: elapsed,
        value,
    }
}

/// Identity: a PID controller's first output is `kp·e + ki·e·dt`, and the integrator
/// it drives climbs to within 1% of the setpoint. Cycles cover all 101 updates.
pub fn pid_step() -> CheckOutcome {
    let dt = 0.01_f64;
    let setpoint = 1.0_f64;
    let Ok(mut controller) = Pid::new(2.0, 1.0, 0.0, dt) else {
        return CheckOutcome {
            name: "pid_step",
            passed: false,
            cycles: 0,
            value: f64::NAN,
        };
    };

    let start = cycles::read();
    let first = black_box(controller.update(setpoint, black_box(0.0)));
    let mut measurement = dt * first;
    let mut climbs = true;
    for _ in 0..100 {
        let previous = measurement;
        let output = controller.update(setpoint, measurement);
        measurement += dt * output;
        climbs &= measurement > previous;
    }
    let measurement = black_box(measurement);
    let climbs = black_box(climbs);
    let elapsed = cycles::read().wrapping_sub(start);

    let passed =
        close(first, 2.01, 1e-12, 0.0) && climbs && close(measurement, setpoint, 0.01, 0.0);

    CheckOutcome {
        name: "pid_step",
        passed,
        cycles: elapsed,
        value: measurement,
    }
}

/// Negative path: a singular `lu_decompose()` and an indefinite `cholesky()` return a
/// typed `Err`, not a panic.
pub fn error_path_returns_err() -> CheckOutcome {
    let singular = black_box(Matrix3D::<f64>::zeros());
    let indefinite = black_box(Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]));

    let start = cycles::read();
    let singular_result = black_box(singular.lu_decompose());
    let indefinite_result = black_box(indefinite.cholesky());
    let elapsed = cycles::read().wrapping_sub(start);

    let passed = matches!(singular_result, Err(LinalgError::Singular))
        && matches!(indefinite_result, Err(LinalgError::NotPositiveDefinite));

    CheckOutcome {
        name: "error_path_returns_err",
        passed,
        cycles: elapsed,
        value: 0.0,
    }
}
