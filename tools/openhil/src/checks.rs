//! The multicalc calls run on target, each timed and compared against a known answer.
//! Golden checks use multicalc's QA fixtures; identity checks use a mathematical identity.
//!
//! A wrong answer returns a failed [`CheckOutcome`] rather than panicking, so `main`
//! prints a `FAIL` line instead of the process exiting without a verdict.
//!
//! Every result passes through `core::hint::black_box` inside its timed section. The work
//! is pure, so an unpinned result lets the optimizer move it past the second reading.
//!
//! Every `cycles` figure includes one system-call round trip, which
//! `cycle_counter_overhead` measures on its own. [`CHECKS`] holds each check's time budget.

use core::hint::black_box;

use multicalc::control::Pid;
use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Vector};
use multicalc::numerical_derivative::{AutoDiffSingle, DerivatorSingleVariable};
use multicalc::scalar_fn;

use crate::cycles;
use crate::fixtures;

/// Whether the answer agreed, the cycles it took, and the headline scalar.
pub struct CheckOutcome {
    pub name: &'static str,
    pub passed: bool,
    pub cycles: u32,
    pub value: f64,
}

/// Every check with its time budget in microseconds, in run order. The two counter checks
/// run first: the other figures are only worth reading once they pass.
///
/// A counter check's budget bounds its warm figure. Every later budget bounds the warm
/// figure less the round trip `cycle_counter_overhead` measured in the same run.
///
/// Each is 1.1 × the Nucleo-F411RE's figure, rounded up. `README.md` has the figures and
/// what measures them again.
pub const CHECKS: [(fn() -> CheckOutcome, u32); 7] = [
    (cycle_counter_sanity, 348),
    (cycle_counter_overhead, 18),
    (vector_dot, 17),
    (svd_singular_values, 1_672),
    (autodiff_derivative, 10),
    (pid_step, 4_113),
    (error_path_returns_err, 35),
];

/// `cycle_counter_overhead`'s place in [`CHECKS`].
pub const ROUND_TRIP_CHECK_INDEX: usize = 1;

/// Sanity: the counter counts CPU cycles. `SPINS` iterations of `subs` plus a taken
/// branch cost between 1 and 8 cycles each, the system-call round trip included; a counter
/// that is stopped or absent misses that band entirely. Value is cycles per iteration.
///
/// The loop starts a 16-byte flash line. The kernel runs the flash at three wait states with
/// no instruction cache, so a loop whose prefetch runs into the next line pays two flash
/// reads an iteration.
fn cycle_counter_sanity() -> CheckOutcome {
    const SPINS: u32 = 10_000;
    let mut remaining = SPINS;

    let start = cycles::read();
    unsafe {
        core::arch::asm!(
            ".p2align 4",
            "2:",
            "subs {remaining}, {remaining}, #1",
            "bne 2b",
            remaining = inout(reg) remaining,
            options(nomem, nostack),
        );
    }
    let elapsed = cycles::read().wrapping_sub(start);

    CheckOutcome {
        name: "cycle_counter_sanity",
        passed: remaining == 0 && (SPINS..=SPINS * 8).contains(&elapsed),
        cycles: elapsed,
        value: f64::from(elapsed) / f64::from(SPINS),
    }
}

/// Calibration: an empty timed section, so the figure is one system-call round trip, the
/// overhead inside every other check's cycles. Passes when the counter moved.
fn cycle_counter_overhead() -> CheckOutcome {
    let start = cycles::read();
    let elapsed = cycles::read().wrapping_sub(start);

    CheckOutcome {
        name: "cycle_counter_overhead",
        passed: elapsed != 0,
        cycles: elapsed,
        value: f64::from(elapsed),
    }
}

/// `|got - want| <= abs + rel * max(|got|, |want|)`.
fn close(got: f64, want: f64, abs: f64, rel: f64) -> bool {
    (got - want).abs() <= abs + rel * got.abs().max(want.abs())
}

/// The dot product's known answer. The `fail` feature moves it, so the board reports a
/// red on demand.
#[cfg(not(feature = "fail"))]
const VECTOR_DOT_EXPECTED: f64 = 20.0;
#[cfg(feature = "fail")]
const VECTOR_DOT_EXPECTED: f64 = 21.0;

/// What the `slow` feature adds inside `vector_dot`'s timed section: 10 000 adds the
/// optimizer cannot fold, far past the check's budget.
#[cfg(feature = "slow")]
fn stall() {
    let mut accumulator = 0_u32;
    for iteration in 0..10_000_u32 {
        accumulator = black_box(accumulator.wrapping_add(iteration));
    }
}

/// Identity: `[1,2,3,4]·[4,3,2,1] = 20`, the atomics-free library path.
fn vector_dot() -> CheckOutcome {
    let left = black_box(Vector::new([1.0_f64, 2.0, 3.0, 4.0]));
    let right = black_box(Vector::new([4.0_f64, 3.0, 2.0, 1.0]));

    let start = cycles::read();
    #[cfg(feature = "slow")]
    stall();
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
fn svd_singular_values() -> CheckOutcome {
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
fn autodiff_derivative() -> CheckOutcome {
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
fn pid_step() -> CheckOutcome {
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
fn error_path_returns_err() -> CheckOutcome {
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
