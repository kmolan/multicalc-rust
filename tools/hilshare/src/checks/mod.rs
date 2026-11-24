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

use crate::Budget;
use crate::cycles;

#[cfg(feature = "arm")]
mod arm;
#[cfg(feature = "drone")]
mod drone;
#[cfg(feature = "arm")]
mod math;

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
/// Each budget is 1.1 × the figure of its board, rounded up, and 1.5 × on the Raspberry Pi
/// Pico 2. `README.md` has the figures and how to measure them again.
#[cfg(feature = "arm")]
pub const CHECKS: &[Check] = &[
    (
        cycle_counter_sanity,
        Budget {
            nucleo_f411re: 347,
            raspberry_pi_pico_2: 323,
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
        math::vector_dot,
        Budget {
            nucleo_f411re: 13,
            raspberry_pi_pico_2: 33,
            xiao_esp32c3: 8,
        },
    ),
    (
        math::svd_singular_values,
        Budget {
            nucleo_f411re: 1192,
            raspberry_pi_pico_2: 1068,
            xiao_esp32c3: 774,
        },
    ),
    (
        math::autodiff_derivative,
        Budget {
            nucleo_f411re: 7,
            raspberry_pi_pico_2: 20,
            xiao_esp32c3: 4,
        },
    ),
    (
        math::pid_step,
        Budget {
            nucleo_f411re: 3088,
            raspberry_pi_pico_2: 2403,
            xiao_esp32c3: 1774,
        },
    ),
    (
        math::error_path_returns_err,
        Budget {
            nucleo_f411re: 27,
            raspberry_pi_pico_2: 48,
            xiao_esp32c3: 15,
        },
    ),
    (arm::franka_inverse_dynamics, UNMEASURED),
    (arm::franka_mass_matrix, UNMEASURED),
    (arm::franka_forward_dynamics, UNMEASURED),
    (arm::arm_control_tick_f32, UNMEASURED),
    // Last: a stack fault here leaves the other lines printed.
    (arm::franka_inverse_dynamics_derivative, UNMEASURED),
];

/// The checks of the drone app, with the budgets as [`CHECKS`] of the arm app describes.
#[cfg(feature = "drone")]
pub const CHECKS: &[Check] = &[
    (cycle_counter_sanity, UNMEASURED),
    (cycle_counter_overhead, UNMEASURED),
    (drone::drone_control_tick_f32, UNMEASURED),
    (drone::attitude_and_mixer_f32, UNMEASURED),
];

/// The checks of the mobile-and-policy app, with the budgets as [`CHECKS`] of the arm app
/// describes.
#[cfg(feature = "mobile_and_policy")]
pub const CHECKS: &[Check] = &[
    (cycle_counter_sanity, UNMEASURED),
    (cycle_counter_overhead, UNMEASURED),
];

/// No limit: the budget of a check that has no figure on its board yet.
#[allow(dead_code)]
const UNMEASURED: Budget = Budget {
    nucleo_f411re: u32::MAX,
    raspberry_pi_pico_2: u32::MAX,
    xiao_esp32c3: u32::MAX,
};

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
#[cfg_attr(not(feature = "arm"), allow(dead_code))]
pub(crate) fn close(got: f64, want: f64, abs: f64, rel: f64) -> bool {
    (got - want).abs() <= abs + rel * got.abs().max(want.abs())
}

/// A failed outcome: the work returned an error or a setup step gave nothing.
#[allow(dead_code)]
pub(crate) fn failed(name: &'static str, cycles: u32) -> CheckOutcome {
    CheckOutcome {
        name,
        passed: false,
        cycles,
        value: f64::NAN,
    }
}
