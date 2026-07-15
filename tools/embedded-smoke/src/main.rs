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

#[cfg(target_arch = "arm")]
use {
    cortex_m_rt::{ExceptionFrame, entry, exception},
    cortex_m_semihosting::{debug, hprintln},
    panic_semihosting as _,
};

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
use {
    riscv_rt::entry,
    riscv_semihosting::{debug, hprintln},
};

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("{}", info);
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}

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

    // Canary set: runs on every target, including the thumbv6m M0. Covers the
    // portable (no-atomics) path, one golden, and the no-panic negative path.
    checks::portable_path();
    let svd_sv = checks::svd_golden();
    checks::error_path_returns_err();

    // Full set: thumbv7em only (default features). Each check returns its headline
    // scalar, captured now and emitted below (after the stack is measured).
    #[cfg(feature = "full-smoke")]
    let full = {
        checks::lm_fit();
        checks::autodiff_derivative();
        checks::lie_group_identity();
        checks::ode_identity();
        (
            checks::quadrature_identity(),
            checks::jacobian_identity(),
            checks::vector_field_identity(),
            checks::root_finding_golden(),
        )
    };

    let used = stack_used(bottom, top);
    // A saturated window (used == WINDOW) means the scan clipped: the number is a floor,
    // not the peak. Fail rather than gate on a false-low value. Raise WINDOW if this trips.
    assert!(used < WINDOW, "stack window saturated at {WINDOW} bytes; raise WINDOW");
    // The size and stack gate reads this exact line from the run output.
    let _ = hprintln!("STACK_HWM_BYTES={}", used);

    // Headline scalars for the cross-ABI divergence guard (ci/check_cross_abi.sh):
    // soft-float (eabi) and hardware-FPU (eabihf) must agree here. Printed as f64
    // in `{:e}` (shortest round-trip decimal).
    let _ = hprintln!("SMOKE_VAL_svd_s0={:e}", svd_sv[0]);
    let _ = hprintln!("SMOKE_VAL_svd_s1={:e}", svd_sv[1]);
    let _ = hprintln!("SMOKE_VAL_svd_s2={:e}", svd_sv[2]);

    // Full-set headlines. Both thumbv7em ABIs build the full set, so the key set
    // matches across them; the thumbv6m canary emits neither and is not compared.
    #[cfg(feature = "full-smoke")]
    {
        let (quad, jac00, div3d, wien_root) = full;
        let _ = hprintln!("SMOKE_VAL_quad={:e}", quad);
        let _ = hprintln!("SMOKE_VAL_jac00={:e}", jac00);
        let _ = hprintln!("SMOKE_VAL_div3d={:e}", div3d);
        let _ = hprintln!("SMOKE_VAL_wien_root={:e}", wien_root);
    }

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

/// A fault must fail the run loudly, not spin until CI times out.
#[cfg(target_arch = "arm")]
#[exception]
unsafe fn HardFault(ef: &ExceptionFrame) -> ! {
    let _ = hprintln!("HARDFAULT: {:#?}", ef);
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}

/// Any other unhandled exception (bus/usage fault, illegal-instruction trap on the M0
/// machine) exits non-zero the same way instead of hanging.
#[cfg(target_arch = "arm")]
#[exception]
unsafe fn DefaultHandler(irqn: i16) {
    let _ = hprintln!("EXCEPTION: irqn={}", irqn);
    debug::exit(debug::EXIT_FAILURE);
}

/// Same contract as the ARM handlers: a trap exits QEMU non-zero, not into riscv-rt's
/// default abort loop.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[unsafe(export_name = "ExceptionHandler")]
fn exception_handler(frame: &riscv_rt::TrapFrame) -> ! {
    let _ = hprintln!("TRAP: {:#x?}", frame);
    debug::exit(debug::EXIT_FAILURE);
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
