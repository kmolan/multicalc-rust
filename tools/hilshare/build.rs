const BOARDS: [&str; 3] = ["nucleo-f411re", "raspberry-pi-pico-2", "xiao-esp32c3"];

fn main() {
    println!("cargo:rerun-if-env-changed=HILSHARE_BOARD");
    let board = std::env::var("HILSHARE_BOARD").expect(
        "HILSHARE_BOARD is not set: build with ./build-tab.sh, which sets it for each board",
    );
    assert!(
        BOARDS.contains(&board.as_str()),
        "HILSHARE_BOARD={board} is not one of {BOARDS:?}"
    );
    libtock_build_scripts::auto_layout();
}
