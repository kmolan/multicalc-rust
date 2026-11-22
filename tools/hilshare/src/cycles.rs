//! The CPU clock of the kernel, and its cycle counter behind Tock's `CycleCount` driver: `CYCCNT`
//! of the DWT on the Nucleo-F411RE and the Raspberry Pi Pico 2, `mpccr` on the XIAO ESP32-C3.
//!
//! A reading is a system call, so a timed section includes one round trip through the kernel and
//! the work that the kernel did in that time. The counter is 32 bits: it wraps after 42.9 s at
//! 100 MHz, 28.6 s at 150 MHz and 26.8 s at 160 MHz, and `wrapping_sub` holds across one wrap.

use libtock::platform::Syscalls;
use libtock::runtime::TockSyscalls;

/// `capsules_extra::cycle_count::DRIVER_NUM`, and its commands.
const CYCLE_COUNT_DRIVER: u32 = 0x90008;
const CYCLE_COUNT_START: u32 = 1;
const CYCLE_COUNT_READ: u32 = 2;
const CYCLE_COUNT_RESET: u32 = 3;
/// The clock driver of the HILShare kernel. Command 1 returns the CPU clock in hertz.
const CLOCK_RATE_DRIVER: u32 = 0xA0000;
const CLOCK_RATE_HERTZ: u32 = 1;

/// The CPU clock in hertz, `None` if the kernel has no clock driver.
#[must_use]
pub fn clock_hz() -> Option<u32> {
    TockSyscalls::command(CLOCK_RATE_DRIVER, CLOCK_RATE_HERTZ, 0, 0).get_success_u32()
}

/// Starts the counter from zero. Call it one time, before [`read`]. A kernel with no counter
/// gives zero for each reading.
pub fn start() {
    let _ = TockSyscalls::command(CYCLE_COUNT_DRIVER, CYCLE_COUNT_RESET, 0, 0);
    let _ = TockSyscalls::command(CYCLE_COUNT_DRIVER, CYCLE_COUNT_START, 0, 0);
}

/// The cycle count, zero if the kernel has no counter. Take differences with `wrapping_sub`.
#[inline(always)]
#[must_use]
pub fn read() -> u32 {
    TockSyscalls::command(CYCLE_COUNT_DRIVER, CYCLE_COUNT_READ, 0, 0)
        .get_success_u64()
        .map_or(0, |count| count as u32)
}
