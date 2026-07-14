//! Select the linker memory map for the target chip and put it where cortex-m-rt's
//! `link.x` (which does `INCLUDE memory.x`) will find it.
//!
//!   thumbv6m (Cortex-M0) -> memory/cortex-m0.x    (nRF51/micro:bit: 256K flash @ 0x0, 16K RAM)
//!   everything else      -> memory/cortex-m4.x    (netduinoplus2:    256K flash @ 0x08000000, 64K RAM)
//!   riscv32*             -> memory/riscv32-virt.x (QEMU virt, -bios none)
//!
//! The chosen file is copied to OUT_DIR/memory.x and OUT_DIR is added to the linker
//! search path. There is deliberately NO memory.x at the repo root: a CWD memory.x
//! wins `INCLUDE` resolution over any -L path and would shadow this per-target choice.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // TARGET is always set by cargo for build scripts; default to the M4 map otherwise.
    let target = env::var("TARGET").unwrap_or_default();
    let map = match target.as_str() {
        t if t.starts_with("riscv32") => "memory/riscv32-virt.x",
        t if t.starts_with("thumbv6m") => "memory/cortex-m0.x",
        _other => "memory/cortex-m4.x",
    };

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    fs::copy(map, out.join("memory.x"))
        .unwrap_or_else(|e| panic!("copying {map} to OUT_DIR/memory.x failed: {e}"));

    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory/cortex-m4.x");
    println!("cargo:rerun-if-changed=memory/cortex-m0.x");
    println!("cargo:rerun-if-changed=memory/riscv32-virt.x");
    println!("cargo:rerun-if-changed=build.rs");
}
