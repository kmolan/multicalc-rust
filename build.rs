use std::process::Command;

fn main() {
    autogenerate_gaussian_tables();
    run_cargo_fmt();
}

fn autogenerate_gaussian_tables() {
    let script_path = "scripts/build_gaussian_integration_tables.py";

    let status = Command::new("python3")
        .arg(script_path)
        .status()
        .expect("Failed to run Python script");

    if !status.success() {
        panic!("Python script failed with status: {:?}", status);
    }
}

fn run_cargo_fmt() {
    let status = Command::new("cargo")
        .args(&["fmt"])
        .status()
        .expect("Failed to run cargo fmt");

    if !status.success() {
        panic!("cargo fmt failed with status: {:?}", status);
    }
}
