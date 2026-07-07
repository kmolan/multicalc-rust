//! A tiny test program for very small chips.
//!
//! This crate is not part of the library. It is a stand-alone check. It runs
//! real math from `multicalc` on three kinds of small chips inside a fake chip
//! (an emulator). Each answer is compared to a value we already know. If every
//! answer is right, the run ends cleanly and reports success. If any answer is
//! wrong, or the program crashes, the run stops and reports a failure.
//!
//! Why we need it: the library is built to run on tiny chips that have no
//! operating system and very little memory. The normal tests run on a big
//! computer and cannot prove that. This program does, on real chip layouts.
//! It also reports how much memory the math used, so we can catch it growing
//! too large.
//!
//! How the memory measurement works: before the checks run, the program fills a
//! block of free memory with one known byte. After the checks, it finds how far
//! into that block the byte was overwritten. That distance is the most memory
//! the math used at once. Three values control this:
//! - `PAINT`: the known byte written into free memory. Anywhere this byte is
//!   still present, the math never reached.
//! - `WINDOW`: how many bytes of free memory to fill and watch. It must be
//!   larger than the math ever needs, but small enough to fit the chip.
//! - `GUARD`: a small gap just below the current spot that we leave untouched,
//!   so filling never clobbers the memory the program is using right now.

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_semihosting as _;

mod checks;
mod fixtures;

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
    checks::svd_golden();
    checks::error_path_returns_err();

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
