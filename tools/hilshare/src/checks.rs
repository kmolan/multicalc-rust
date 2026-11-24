//! The multicalc calls run on target, each timed and compared against a known answer.
//! Golden checks use multicalc's QA fixtures; identity checks use a mathematical identity.
//!
//! A wrong answer returns a failed [`CheckOutcome`] rather than panicking, so `main`
//! prints a `FAIL` line instead of the process exiting without a verdict.
//!
//! Every result passes through `core::hint::black_box` inside its timed section. The work
//! is pure, so an unpinned result lets the optimizer move it past the second reading.
//!
//! Each `cycles` reading of a [`CheckOutcome`] includes one system-call round trip, which
//! `cycle_counter_overhead` measures on its own. [`CHECKS`] holds the checks of the app and their
//! time budgets. Each app starts with the two counter checks.

#[cfg(feature = "arm")]
use core::hint::black_box;

#[cfg(feature = "arm")]
use multicalc::control::Pid;
#[cfg(feature = "arm")]
use multicalc::error::LinalgError;
#[cfg(feature = "arm")]
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Vector};
#[cfg(feature = "arm")]
use multicalc::numerical_derivative::{AutoDiffSingle, DerivatorSingleVariable};
#[cfg(feature = "arm")]
use multicalc::scalar_fn;

use crate::Budget;
use crate::cycles;
#[cfg(feature = "arm")]
use crate::fixtures;

#[cfg(feature = "arm")]
mod arm;
#[cfg(feature = "drone")]
mod drone;
#[cfg(feature = "mobile_and_policy")]
mod mobile_and_policy;

/// Whether the answer agreed, the cycles it took, and the headline scalar.
pub struct CheckOutcome {
    pub name: &'static str,
    pub passed: bool,
    pub cycles: u32,
    pub value: f64,
}

/// One check and its time budgets.
pub type Check = (fn() -> CheckOutcome, Budget);

/// Each check with its time budgets in microseconds, in run order. The two counter checks run
/// first: the other figures are only worth reading when they pass.
///
/// The budget of a counter check limits its warm figure. Each later budget limits the warm
/// figure less the round trip that `cycle_counter_overhead` measured in the same run.
///
/// Each budget is 1.1 × the highest figure of its board over three runs, rounded up, and 1.5 × on
/// the Raspberry Pi Pico 2. `benchmarks/silicon.md` has the figures, and `README.md` how to
/// measure them again.
#[cfg(feature = "arm")]
pub const CHECKS: &[Check] = &[
    (
        cycle_counter_sanity,
        Budget {
            nucleo_f411re: 347,
            raspberry_pi_pico_2: 318,
            xiao_esp32c3: 284,
        },
    ),
    (
        cycle_counter_overhead,
        Budget {
            nucleo_f411re: 17,
            raspberry_pi_pico_2: 20,
            xiao_esp32c3: 9,
        },
    ),
    (
        vector_dot,
        Budget {
            nucleo_f411re: 19,
            raspberry_pi_pico_2: 25,
            xiao_esp32c3: 8,
        },
    ),
    (
        svd_singular_values,
        Budget {
            nucleo_f411re: 1256,
            raspberry_pi_pico_2: 1192,
            xiao_esp32c3: 788,
        },
    ),
    (
        autodiff_derivative,
        Budget {
            nucleo_f411re: 2,
            raspberry_pi_pico_2: 16,
            xiao_esp32c3: 4,
        },
    ),
    (
        pid_step,
        Budget {
            nucleo_f411re: 3329,
            raspberry_pi_pico_2: 2530,
            xiao_esp32c3: 1773,
        },
    ),
    (
        error_path_returns_err,
        Budget {
            nucleo_f411re: 29,
            raspberry_pi_pico_2: 75,
            xiao_esp32c3: 15,
        },
    ),
    (
        arm::franka_inverse_dynamics,
        Budget {
            nucleo_f411re: 13403,
            raspberry_pi_pico_2: 15402,
            xiao_esp32c3: 8167,
        },
    ),
    (
        arm::franka_mass_matrix,
        Budget {
            nucleo_f411re: 12629,
            raspberry_pi_pico_2: 12758,
            xiao_esp32c3: 7429,
        },
    ),
    (
        arm::franka_forward_dynamics,
        Budget {
            nucleo_f411re: 17595,
            raspberry_pi_pico_2: 19098,
            xiao_esp32c3: 11456,
        },
    ),
    (
        arm::arm_control_tick_f32,
        Budget {
            nucleo_f411re: 772,
            raspberry_pi_pico_2: 1395,
            xiao_esp32c3: 7604,
        },
    ),
    // Last: a stack fault here leaves the other lines printed.
    (
        arm::franka_inverse_dynamics_derivative,
        Budget {
            nucleo_f411re: 33864,
            raspberry_pi_pico_2: 34015,
            xiao_esp32c3: 20988,
        },
    ),
];

/// The checks of the drone app, with the budgets as [`CHECKS`] of the arm app describes.
#[cfg(feature = "drone")]
pub const CHECKS: &[Check] = &[
    (
        cycle_counter_sanity,
        Budget {
            nucleo_f411re: 347,
            raspberry_pi_pico_2: 320,
            xiao_esp32c3: 284,
        },
    ),
    (
        cycle_counter_overhead,
        Budget {
            nucleo_f411re: 17,
            raspberry_pi_pico_2: 18,
            xiao_esp32c3: 9,
        },
    ),
    (
        drone::drone_control_tick_f32,
        Budget {
            nucleo_f411re: 2048,
            raspberry_pi_pico_2: 3372,
            xiao_esp32c3: 18187,
        },
    ),
    (
        drone::attitude_and_mixer_f32,
        Budget {
            nucleo_f411re: 18,
            raspberry_pi_pico_2: 118,
            xiao_esp32c3: 298,
        },
    ),
];

/// The checks of the mobile-and-policy app, with the budgets as [`CHECKS`] of the arm app
/// describes.
#[cfg(feature = "mobile_and_policy")]
pub const CHECKS: &[Check] = &[
    (
        cycle_counter_sanity,
        Budget {
            nucleo_f411re: 347,
            raspberry_pi_pico_2: 319,
            xiao_esp32c3: 284,
        },
    ),
    (
        cycle_counter_overhead,
        Budget {
            nucleo_f411re: 17,
            raspberry_pi_pico_2: 19,
            xiao_esp32c3: 9,
        },
    ),
    (
        mobile_and_policy::mobile_robot_tick_f32,
        Budget {
            nucleo_f411re: 985,
            raspberry_pi_pico_2: 1148,
            xiao_esp32c3: 2126,
        },
    ),
    (
        mobile_and_policy::mlp_policy_forward,
        Budget {
            nucleo_f411re: 18105,
            raspberry_pi_pico_2: 22962,
            xiao_esp32c3: 11705,
        },
    ),
    (
        mobile_and_policy::mlp_policy_forward_f32,
        Budget {
            nucleo_f411re: 585,
            raspberry_pi_pico_2: 2525,
            xiao_esp32c3: 6725,
        },
    ),
];

/// `cycle_counter_overhead`'s place in [`CHECKS`].
pub const ROUND_TRIP_CHECK_INDEX: usize = 1;

/// Sanity: the counter counts CPU cycles. `SPINS` iterations of a decrement and a taken
/// branch cost from 1 to 8 cycles each, with the system-call round trip; a counter that is
/// stopped or absent misses that band. The value is the cycles of one iteration.
///
/// The loop starts a 16-byte line, so the prefetch of one iteration stays in one line on each
/// board. The Nucleo-F411RE kernel turns on the flash prefetch and the ART caches, and the Pico 2
/// and the XIAO run from flash through a cache, so the warm runs read the loop from a cache.
fn cycle_counter_sanity() -> CheckOutcome {
    const SPINS: u32 = 10_000;
    let mut remaining = SPINS;

    let start = cycles::read();
    #[cfg(target_arch = "arm")]
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
    #[cfg(target_arch = "riscv32")]
    unsafe {
        core::arch::asm!(
            ".p2align 4",
            "2:",
            "addi {remaining}, {remaining}, -1",
            "bnez {remaining}, 2b",
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
pub(crate) fn close(got: f64, want: f64, abs: f64, rel: f64) -> bool {
    (got - want).abs() <= abs + rel * got.abs().max(want.abs())
}

/// A failed outcome: the work returned an error or a setup step gave nothing.
pub(crate) fn failed(name: &'static str, cycles: u32) -> CheckOutcome {
    CheckOutcome {
        name,
        passed: false,
        cycles,
        value: f64::NAN,
    }
}

/// The known answer of the dot product.
#[cfg(feature = "arm")]
const VECTOR_DOT_EXPECTED: f64 = 20.0;

/// Identity: `[1,2,3,4]·[4,3,2,1] = 20`, the atomics-free library path.
#[cfg(feature = "arm")]
fn vector_dot() -> CheckOutcome {
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
#[cfg(feature = "arm")]
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
#[cfg(feature = "arm")]
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
#[cfg(feature = "arm")]
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
#[cfg(feature = "arm")]
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
