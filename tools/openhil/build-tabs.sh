#!/bin/sh
# Builds the app once per behaviour and packages each as a TAB:
# target/tab/multicalc-openhil-{pass,fail,slow}.tab, with the TBF each holds beside it.
# Needs elf2tab 0.13.0.
set -eu
cd "$(dirname "$0")"
# RUSTFLAGS replaces every `target.*.rustflags` in cargo's configuration, the repository
# root's for embedded-smoke included. libtock-rs apps are statically relocated.
export RUSTFLAGS="-C relocation-model=static -C link-arg=-icf=all"
mkdir -p target/tab
for behaviour in pass fail slow; do
    if [ "$behaviour" = pass ]; then
        cargo build --release
    else
        cargo build --release --features "$behaviour"
    fi
    elf="target/tab/multicalc-openhil-$behaviour.elf"
    cp target/thumbv7em-none-eabi/release/multicalc-openhil "$elf"
    elf2tab --kernel-major 2 --kernel-minor 1 \
        --package-name multicalc-openhil \
        --stack 8192 --minimum-footer-size 256 \
        --output-file "target/tab/multicalc-openhil-$behaviour.tab" \
        "$elf,cortex-m4"
done
