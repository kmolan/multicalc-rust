//! Runs the math checks on a Cortex-M target under QEMU.
//! A panic exits QEMU with a failure code. A clean finish exits with success.

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_semihosting as _;

mod checks;

// Byte written across free stack so we can find how deep the stack went.
const PAINT: u8 = 0xAA;
// Bytes of free stack to watch. Must be smaller than the free stack on the
// target (64 KB RAM) and larger than the deepest check. 4 KB fits both. If the
// printed stack ever equals WINDOW it has saturated — raise WINDOW and confirm
// the target still has room.
const WINDOW: usize = 4096;
// Bytes just below the current stack pointer we never touch (our own frame).
const GUARD: usize = 64;

#[entry]
fn main() -> ! {
    let top = sp().saturating_sub(GUARD);
    let bottom = top.saturating_sub(WINDOW);
    paint(bottom, top);

    checks::lm_fit();
    checks::autodiff_derivative();
    checks::portable_path();
    checks::svd_kabsch();

    let used = stack_used(bottom, top);
    // The size and stack gate reads this exact line from the run output.
    let _ = hprintln!("STACK_HWM_BYTES={}", used);

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

/// Address of a local, close to the current stack pointer.
fn sp() -> usize {
    let x = 0u8;
    core::ptr::addr_of!(x) as usize
}

/// Fill [bottom, top) with PAINT.
fn paint(bottom: usize, top: usize) {
    let mut a = bottom;
    while a < top {
        unsafe { core::ptr::write_volatile(a as *mut u8, PAINT) };
        a += 1;
    }
}

/// Peak stack use: distance from the deepest changed byte up to `top`.
fn stack_used(bottom: usize, top: usize) -> usize {
    let mut a = bottom;
    while a < top {
        let b = unsafe { core::ptr::read_volatile(a as *const u8) };
        if b != PAINT {
            break; // first changed byte from the bottom = deepest the stack reached
        }
        a += 1;
    }
    top - a
}
