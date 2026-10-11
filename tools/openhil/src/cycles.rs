//! The kernel's cycle counter: the Cortex-M4 DWT's CYCCNT behind Tock's `CycleCount`
//! driver.
//!
//! A reading is a system call, so a timed section includes one round trip through the
//! kernel and whatever the kernel ran in between. CYCCNT is 32 bits: a reading wraps every
//! `2^32 / 100 MHz` ≈ 42.9 s, and `wrapping_sub` holds across one wrap.

use libtock::platform::Syscalls;
use libtock::runtime::TockSyscalls;

/// `capsules_extra::cycle_count::DRIVER_NUM`, and its commands.
const CYCLE_COUNT_DRIVER: u32 = 0x90008;
const CYCLE_COUNT_START: u32 = 1;
const CYCLE_COUNT_READ: u32 = 2;
const CYCLE_COUNT_RESET: u32 = 3;

/// Start the counter from zero. Call once, before [`read`]. A kernel without the driver
/// leaves every reading at zero.
pub fn start() {
    let _ = TockSyscalls::command(CYCLE_COUNT_DRIVER, CYCLE_COUNT_RESET, 0, 0);
    let _ = TockSyscalls::command(CYCLE_COUNT_DRIVER, CYCLE_COUNT_START, 0, 0);
}

/// Current cycle count, zero when the kernel has no counter. Take differences with
/// `wrapping_sub`.
#[inline(always)]
#[must_use]
pub fn read() -> u32 {
    TockSyscalls::command(CYCLE_COUNT_DRIVER, CYCLE_COUNT_READ, 0, 0)
        .get_success_u64()
        .map_or(0, |count| count as u32)
}
