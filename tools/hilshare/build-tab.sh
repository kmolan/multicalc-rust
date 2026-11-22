#!/bin/sh
# Builds the app one time for each board, and packages the three TBFs as
# target/tab/multicalc-hilshare.tab: cortex-m4 for the Nucleo-F411RE,
# cortex-m4.0x10040080.0x20008000 for the Raspberry Pi Pico 2 and rv32imc.0x42100080.0x3FCA0000
# for the XIAO ESP32-C3. The ELF of each board is target/tab/multicalc-hilshare-<board>.elf.
# elf2tab 0.13.0 is necessary.
# `--kernel-minor 3`: the Tock release-2.3.0 kernel reports version 2.3, and its loader skips an
# app that requires a later version.
set -eu
cd "$(dirname "$0")"
# RUSTFLAGS replaces each `target.*.rustflags` of the cargo configuration, also the embedded-smoke
# flags of the repository root. libtock-rs apps are statically relocated.
export RUSTFLAGS="-C relocation-model=static -C link-arg=-icf=all"
mkdir -p target/tab
# The addresses of [env] in .cargo/config.toml.
HILSHARE_BOARD=nucleo-f411re cargo build --release
cp target/thumbv7em-none-eabi/release/multicalc-hilshare target/tab/multicalc-hilshare-nucleo-f411re.elf
# The fixed addresses of the Pico 2 kernel: app flash from 0x10040000 (with the TBF header) and
# app RAM from 0x20008000.
HILSHARE_BOARD=raspberry-pi-pico-2 \
    LIBTOCK_LINKER_FLASH=0x10040000 LIBTOCK_LINKER_FLASH_LENGTH=511K \
    LIBTOCK_LINKER_RAM=0x20008000 LIBTOCK_LINKER_RAM_LENGTH=484K \
    cargo build --release
cp target/thumbv7em-none-eabi/release/multicalc-hilshare target/tab/multicalc-hilshare-raspberry-pi-pico-2.elf
# The fixed addresses of the XIAO kernel: app flash from 0x42100000 (with the TBF header) and app
# RAM from 0x3FCA0000.
HILSHARE_BOARD=xiao-esp32c3 \
    LIBTOCK_LINKER_FLASH=0x42100000 LIBTOCK_LINKER_FLASH_LENGTH=511K \
    LIBTOCK_LINKER_RAM=0x3FCA0000 LIBTOCK_LINKER_RAM_LENGTH=248K \
    cargo build --release --target riscv32imc-unknown-none-elf
cp target/riscv32imc-unknown-none-elf/release/multicalc-hilshare target/tab/multicalc-hilshare-xiao-esp32c3.elf
elf2tab --kernel-major 2 --kernel-minor 3 \
    --package-name multicalc-hilshare \
    --stack 8192 --minimum-footer-size 256 \
    --output-file target/tab/multicalc-hilshare.tab \
    "target/tab/multicalc-hilshare-nucleo-f411re.elf,cortex-m4" \
    "target/tab/multicalc-hilshare-raspberry-pi-pico-2.elf,cortex-m4.0x10040080.0x20008000" \
    "target/tab/multicalc-hilshare-xiao-esp32c3.elf,rv32imc.0x42100080.0x3FCA0000"
