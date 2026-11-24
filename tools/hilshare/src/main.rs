//! multicalc as a Tock process on the three HILShare boards.
//!
//! Runs each check on the chip, and reports if the answer agreed with the known value and the
//! time of the check. A check passes when each run agreed and the warm figure is not above the
//! budget of the board. The return from `main` is the exit that the kernel reports.
//!
//! `checks` is the only module that uses the API of multicalc. A Cargo feature selects the app,
//! and each app has its own `CHECKS`.
//!
//! A panic ends the process with no `HILSHARE_RESULT`, and HILShare reports a run that started
//! and did not finish.

#![no_main]
#![no_std]

#[cfg(not(any(feature = "arm", feature = "drone", feature = "mobile_and_policy")))]
compile_error!("select one app: --features arm, drone or mobile_and_policy");
#[cfg(any(
    all(feature = "arm", feature = "drone"),
    all(feature = "arm", feature = "mobile_and_policy"),
    all(feature = "drone", feature = "mobile_and_policy"),
))]
compile_error!("select one app at a time");

use core::fmt::Write;

use libtock::console::Console;
use libtock::runtime::{set_main, stack_size};

mod checks;
mod cycles;
/// The QA goldens of multicalc, as `multicalc-qa` generates them for `embedded-smoke`.
#[path = "../../embedded-smoke/src/fixtures.rs"]
mod fixtures;

set_main! {main}
// Must equal `--stack` in build-tab.sh.
stack_size! {0x2000}

/// The board of this build. build-tab.sh sets `HILSHARE_BOARD` for each build, and build.rs
/// refuses another name.
pub const BOARD: &str = env!("HILSHARE_BOARD");

/// The app of this build, from its Cargo feature.
#[cfg(feature = "arm")]
const APP: &str = "arm";
#[cfg(feature = "drone")]
const APP: &str = "drone";
#[cfg(feature = "mobile_and_policy")]
const APP: &str = "mobile_and_policy";

/// A time limit in microseconds for each board.
#[derive(Clone, Copy)]
pub struct Budget {
    pub nucleo_f411re: u32,
    pub raspberry_pi_pico_2: u32,
    pub xiao_esp32c3: u32,
}

impl Budget {
    /// The limit of `BOARD`.
    #[must_use]
    pub fn microseconds(self) -> u32 {
        match BOARD {
            "nucleo-f411re" => self.nucleo_f411re,
            "raspberry-pi-pico-2" => self.raspberry_pi_pico_2,
            "xiao-esp32c3" => self.xiao_esp32c3,
            _ => unreachable!("build.rs accepts no other board"),
        }
    }
}

/// Runs of each check. The first run pays for the cold cache and the flash fetches. The reported
/// figure is the best of the other runs, the steady-state cost in a loop.
const REPEATS: usize = 8;

fn main() {
    let mut console = Console::writer();

    let _ = writeln!(console, "HILSHARE_START_EXECUTION");
    let _ = writeln!(
        console,
        "HILSHARE_FIRMWARE={}+{APP}",
        env!("CARGO_PKG_VERSION")
    );

    let Some(clock_hz) = cycles::clock_hz() else {
        let _ = writeln!(console, "HILSHARE_CHECK name=clock_rate status=FAIL");
        let _ = writeln!(console, "HILSHARE_RESULT=FAIL");
        return;
    };
    #[cfg(target_abi = "eabihf")]
    reset_floating_point_status();
    cycles::start();

    let mut passed = 0_usize;
    let mut round_trip_cycles = 0_u32;
    for (index, &(check, budget)) in checks::CHECKS.iter().enumerate() {
        let first = check();
        let mut warm = u32::MAX;
        let mut agreed = first.passed;
        for _ in 1..REPEATS {
            let repeat = check();
            agreed &= repeat.passed;
            warm = warm.min(repeat.cycles);
        }

        // The counter checks keep the raw figure. Each later check has the round trip of this
        // run subtracted first.
        let overhead = if index > checks::ROUND_TRIP_CHECK_INDEX {
            round_trip_cycles
        } else {
            0
        };
        if index == checks::ROUND_TRIP_CHECK_INDEX {
            round_trip_cycles = warm;
        }
        let net = warm.saturating_sub(overhead);
        let cold_net = first.cycles.saturating_sub(overhead);

        // The budget judges the warm figure only: one run can cross a timeslice.
        let budget_microseconds = budget.microseconds();
        let within_budget =
            u64::from(net) * 1_000_000 <= u64::from(budget_microseconds) * u64::from(clock_hz);
        let check_passed = agreed && within_budget;
        passed += usize::from(check_passed);
        let time = hundredths_of_a_microsecond(net, clock_hz);
        let overhead_time = hundredths_of_a_microsecond(overhead, clock_hz);
        let _ = writeln!(
            console,
            "HILSHARE_CHECK name={} status={} us={}.{:02} budget_us={} overhead_us={}.{:02} cycles={} cold_cycles={} value={:e}",
            first.name,
            if check_passed { "PASS" } else { "FAIL" },
            time / 100,
            time % 100,
            budget_microseconds,
            overhead_time / 100,
            overhead_time % 100,
            net,
            cold_net,
            first.value
        );
    }

    let total = checks::CHECKS.len();
    let _ = writeln!(console, "HILSHARE_CHECKS_PASSED={passed}/{total}");
    let _ = writeln!(
        console,
        "HILSHARE_RESULT={}",
        if passed == total { "PASS" } else { "FAIL" }
    );
}

/// `FPSCR = 0`: round to nearest, no flush-to-zero, no default NaN. The kernel saves no FPU
/// state and does not write FPSCR.
#[cfg(target_abi = "eabihf")]
fn reset_floating_point_status() {
    // SAFETY: one `vmsr` to FPSCR. The kernel gives this process the FPU.
    unsafe {
        core::arch::asm!("vmsr fpscr, {status}", status = in(reg) 0_u32, options(nomem, nostack));
    }
}

/// `cycles × 100 000 000 / clock_hz`.
fn hundredths_of_a_microsecond(cycles: u32, clock_hz: u32) -> u64 {
    u64::from(cycles) * 100_000_000 / u64::from(clock_hz)
}
