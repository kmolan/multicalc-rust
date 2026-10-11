//! multicalc on a Nucleo-F411RE, as a Tock process under the OpenHIL kernel.
//!
//! Runs each check on the chip and reports whether the answer matched the known-good value
//! and how long it took. A check passes when every run agreed and the warm figure is within
//! its time budget. Returning from `main` is the exit the kernel reports.
//!
//! `checks.rs` is the only module that touches multicalc's API.
//!
//! A panic exits the process with no `OPENHIL_RESULT`, which OpenHIL reports as a run that
//! started and did not finish.

#![no_main]
#![no_std]

use core::fmt::Write;

use libtock::console::Console;
use libtock::runtime::{set_main, stack_size};

mod checks;
mod cycles;
/// multicalc's QA goldens, as `multicalc-qa` generates them for `embedded-smoke`.
#[path = "../../embedded-smoke/src/fixtures.rs"]
mod fixtures;

set_main! {main}
stack_size! {0x2000}

/// The OpenHIL kernel's SYSCLK on the Nucleo-F411RE. Cycle counts divide by this for wall
/// time.
const CPU_CLOCK_HZ: u32 = 100_000_000;
const CYCLES_PER_MICROSECOND: u32 = CPU_CLOCK_HZ / 1_000_000;

/// Runs per check. The first pays for cold cache and flash fetches; the reported figure is
/// the best of the rest, the steady-state cost in a loop.
const REPEATS: usize = 8;

fn main() {
    let mut console = Console::writer();

    let _ = writeln!(console, "OPENHIL_START_EXECUTION");
    let _ = writeln!(console, "OPENHIL_BOARD=nucleo-f411re");
    let _ = writeln!(console, "OPENHIL_TARGET=thumbv7em-none-eabi");
    let _ = writeln!(console, "OPENHIL_CPU_HZ={CPU_CLOCK_HZ}");
    let _ = writeln!(console, "OPENHIL_FIRMWARE={}", env!("CARGO_PKG_VERSION"));

    cycles::start();

    let mut passed = 0_usize;
    let mut round_trip_cycles = 0_u32;
    for (index, (check, budget_microseconds)) in checks::CHECKS.into_iter().enumerate() {
        let first = check();
        let mut warm = u32::MAX;
        let mut agreed = first.passed;
        for _ in 1..REPEATS {
            let repeat = check();
            agreed &= repeat.passed;
            warm = warm.min(repeat.cycles);
        }

        // The counter checks are judged on the raw figure. Every later one has the round
        // trip measured in this run subtracted first.
        let overhead = if index > checks::ROUND_TRIP_CHECK_INDEX {
            round_trip_cycles
        } else {
            0
        };
        if index == checks::ROUND_TRIP_CHECK_INDEX {
            round_trip_cycles = warm;
        }
        let net = warm.saturating_sub(overhead);

        // The budget gates the warm figure only: a single run can straddle a timeslice.
        let within_budget = net <= budget_microseconds * CYCLES_PER_MICROSECOND;
        passed += usize::from(agreed && within_budget);
        let _ = writeln!(
            console,
            "OPENHIL_CHECK name={} status={} us={} budget_us={} overhead_us={} cycles={} cold_cycles={} value={:e}",
            first.name,
            if agreed && within_budget {
                "PASS"
            } else {
                "FAIL"
            },
            microseconds(net),
            budget_microseconds,
            microseconds(overhead),
            warm,
            first.cycles,
            first.value
        );
    }

    let total = checks::CHECKS.len();
    let _ = writeln!(console, "OPENHIL_CHECKS_PASSED={passed}/{total}");
    let _ = writeln!(
        console,
        "OPENHIL_RESULT={}",
        if passed == total { "PASS" } else { "FAIL" }
    );
}

/// Cycles as microseconds at the kernel's CPU clock.
fn microseconds(cycles: u32) -> f64 {
    f64::from(cycles) / f64::from(CYCLES_PER_MICROSECOND)
}
