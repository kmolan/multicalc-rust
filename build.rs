use std::process::Command;

fn main() {
    let script_path = "scripts/build_gaussian_integration_tables.py";

    let status = Command::new("python3")
        .arg(script_path)
        .status()
        .expect("Failed to run Python script");

    if !status.success() {
        panic!("Python script failed with status: {:?}", status);
    }
    
}
