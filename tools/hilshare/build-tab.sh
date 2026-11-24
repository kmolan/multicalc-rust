#!/bin/sh
# Builds each app one time for each board, and packages each app as
# target/tab/multicalc-hilshare-<app>.tab: cortex-m4 for the Nucleo-F411RE,
# cortex-m4.0x10040080.0x20008000 for the Raspberry Pi Pico 2 and rv32imc.0x42100080.0x3FCA0000
# for the XIAO ESP32-C3. The ELF of each build is target/tab/multicalc-hilshare-<app>-<board>.elf,
# and elf2tab writes its TBF beside it. The apps are the Cargo features arm, drone and
# mobile_and_policy. elf2tab 0.13.0 is necessary.
# `--kernel-minor 3`: the Tock release-2.3.0 kernel reports version 2.3, and its loader skips an
# app that requires a later version.
# The Arm builds are hard-float. rustc warns that the libtock-rs syscall `asm!` clobbers
# D16–D31: FPv4-SP-D16 and FPv5-SP have no D16–D31, so the clobber has no effect.
set -eu
cd "$(dirname "$0")"
# RUSTFLAGS replaces each `target.*.rustflags` of the cargo configuration, also the embedded-smoke
# flags of the repository root. libtock-rs apps are statically relocated.
export RUSTFLAGS="-C relocation-model=static -C link-arg=-icf=all"
mkdir -p target/tab
for app in arm drone mobile_and_policy; do
    # The addresses of [env] in .cargo/config.toml.
    HILSHARE_BOARD=nucleo-f411re cargo build --release --features "$app"
    cp target/thumbv7em-none-eabihf/release/multicalc-hilshare "target/tab/multicalc-hilshare-$app-nucleo-f411re.elf"
    # The fixed addresses of the Pico 2 kernel: app flash from 0x10040000 (with the TBF header)
    # and app RAM from 0x20008000.
    HILSHARE_BOARD=raspberry-pi-pico-2 \
        LIBTOCK_LINKER_FLASH=0x10040000 LIBTOCK_LINKER_FLASH_LENGTH=511K \
        LIBTOCK_LINKER_RAM=0x20008000 LIBTOCK_LINKER_RAM_LENGTH=484K \
        cargo build --release --features "$app"
    cp target/thumbv7em-none-eabihf/release/multicalc-hilshare "target/tab/multicalc-hilshare-$app-raspberry-pi-pico-2.elf"
    # The fixed addresses of the XIAO kernel: app flash from 0x42100000 (with the TBF header) and
    # app RAM from 0x3FCA0000.
    HILSHARE_BOARD=xiao-esp32c3 \
        LIBTOCK_LINKER_FLASH=0x42100000 LIBTOCK_LINKER_FLASH_LENGTH=511K \
        LIBTOCK_LINKER_RAM=0x3FCA0000 LIBTOCK_LINKER_RAM_LENGTH=248K \
        cargo build --release --target riscv32imc-unknown-none-elf --features "$app"
    cp target/riscv32imc-unknown-none-elf/release/multicalc-hilshare "target/tab/multicalc-hilshare-$app-xiao-esp32c3.elf"
    elf2tab --kernel-major 2 --kernel-minor 3 \
        --package-name "multicalc-hilshare-$app" \
        --stack 16384 --minimum-footer-size 256 \
        --output-file "target/tab/multicalc-hilshare-$app.tab" \
        "target/tab/multicalc-hilshare-$app-nucleo-f411re.elf,cortex-m4" \
        "target/tab/multicalc-hilshare-$app-raspberry-pi-pico-2.elf,cortex-m4.0x10040080.0x20008000" \
        "target/tab/multicalc-hilshare-$app-xiao-esp32c3.elf,rv32imc.0x42100080.0x3FCA0000"
done
