//! Renders `benchmarks/silicon.md` from the logs of HILShare Board runs: the best and the worst
//! warm time of each check on each board.
//!
//! Run `cargo run -p multicalc-qa --bin gen_silicon_table -- --commit <sha> --date <YYYY-MM-DD>
//! <run directory>...`. Each run directory holds the artifacts of one run, from
//! `gh run download <run-id> --pattern 'hilshare-multicalc-hilshare-*'`: one artifact for each
//! app, with one `<TAB name>-<board>` directory for each board. The boards are not in CI, so CI
//! does not regenerate this table.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use multicalc_qa::docs::replace_marked_region;

const BEGIN: &str = "<!-- BEGIN generated: silicon -->";
const END: &str = "<!-- END generated -->";

const USAGE: &str =
    "usage: gen_silicon_table --commit <sha> --date <YYYY-MM-DD> <run directory>...";

/// Each board: its artifact suffix, its name, its core.
const BOARDS: [(&str, &str, &str); 3] = [
    ("nucleo-f411re", "Nucleo-F411RE", "Cortex-M4F"),
    ("raspberry-pi-pico-2", "Raspberry Pi Pico 2", "Cortex-M33"),
    ("xiao-esp32c3", "XIAO ESP32-C3", "RV32IMC"),
];

/// The two counter checks: calibration, not results, so the table leaves them out.
const COUNTER_CHECKS: [&str; 2] = ["cycle_counter_sanity", "cycle_counter_overhead"];

/// Each check of the table, in run order: app, check, what it runs, scalar.
const CHECKS: &[(&str, &str, &str, &str)] = &[
    ("arm", "vector_dot", "Dot product, 4 elements", "`f64`"),
    ("arm", "svd_singular_values", "SVD, 3×3", "`f64`"),
    (
        "arm",
        "autodiff_derivative",
        "Forward-mode derivative of `x³`",
        "`f64`",
    ),
    (
        "arm",
        "pid_step",
        "101 PID updates with a first-order plant",
        "`f64`",
    ),
    (
        "arm",
        "error_path_returns_err",
        "Singular LU and indefinite Cholesky return `Err`",
        "`f64`",
    ),
    (
        "arm",
        "franka_inverse_dynamics",
        "Franka Panda RNEA, 11 slots: `(q, q̇, q̈) → τ`",
        "`f64`",
    ),
    (
        "arm",
        "franka_mass_matrix",
        "Franka Panda CRBA: `q → H(q)`",
        "`f64`",
    ),
    (
        "arm",
        "franka_forward_dynamics",
        "Franka Panda ABA: `(q, q̇, τ) → q̈`",
        "`f64`",
    ),
    (
        "arm",
        "arm_control_tick_f32",
        "Franka Panda tick: computed torque and momentum observer",
        "`f32`",
    ),
    (
        "arm",
        "franka_inverse_dynamics_derivative",
        "Exact `∂τ/∂q̈ · eᵢ` by one RNEA pass",
        "`Dual<f64>`",
    ),
    (
        "drone",
        "drone_control_tick_f32",
        "Skydio X2 tick: ESKF predict and update, LQR, thrust, attitude, mixer",
        "`f32`",
    ),
    (
        "drone",
        "attitude_and_mixer_f32",
        "Skydio X2 attitude controller and mixer",
        "`f32`",
    ),
    (
        "drone",
        "eskf_trajectory",
        "15-state ESKF: 200 IMU steps and 30 updates",
        "`f64`",
    ),
    (
        "mobile_and_policy",
        "mobile_robot_tick_f32",
        "Mobile robot tick: 5-state EKF and pure pursuit",
        "`f32`",
    ),
    (
        "mobile_and_policy",
        "mlp_policy_forward",
        "MLP policy, 22→64→64→4",
        "`f64`",
    ),
    (
        "mobile_and_policy",
        "mlp_policy_forward_f32",
        "MLP policy, 22→64→64→4",
        "`f32`",
    ),
];

/// The `key=value` fields of a marker line, split at spaces.
#[must_use]
fn fields(line: &str) -> HashMap<&str, &str> {
    line.split(' ')
        .filter_map(|field| field.split_once('='))
        .collect()
}

/// Microseconds as `"12.34 µs"`, `"1.23 ms"` or `"1.23 s"`.
#[must_use]
fn format_time(microseconds: f64) -> String {
    if microseconds < 1_000.0 {
        format!("{microseconds:.2} µs")
    } else if microseconds < 1_000_000.0 {
        format!("{:.2} ms", microseconds / 1_000.0)
    } else {
        format!("{:.2} s", microseconds / 1_000_000.0)
    }
}

/// The figures of one check on one board: the best warm time, the worst warm time and the runs
/// that reported it, all in microseconds.
#[derive(Default)]
struct Figures {
    best: f64,
    worst: f64,
    runs: usize,
}

/// What the logs of all runs give.
#[derive(Default)]
struct Measurements {
    /// `(app, check, board)` to its figures.
    figures: HashMap<(String, String, &'static str), Figures>,
    /// Board to its kernel version and its clock in hertz.
    kernels: HashMap<&'static str, (String, u64)>,
}

impl Measurements {
    /// Reads every log of one run directory.
    fn read_run(&mut self, run: &Path) {
        let entries =
            std::fs::read_dir(run).unwrap_or_else(|err| unreachable!("read {run:?}: {err}"));
        let mut artifacts: Vec<PathBuf> = entries
            .map(|entry| {
                entry
                    .unwrap_or_else(|err| unreachable!("read {run:?}: {err}"))
                    .path()
            })
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("hilshare-multicalc-hilshare-"))
            })
            .collect();
        artifacts.sort();
        assert!(!artifacts.is_empty(), "{run:?} holds no HILShare artifact");
        let mut seen = HashMap::new();
        for artifact in artifacts {
            for log in logs(&artifact) {
                // The action names the directory of each board's log `<TAB name>-<board>`.
                let board = log
                    .ancestors()
                    .filter_map(|directory| directory.file_name()?.to_str())
                    .find_map(|name| {
                        BOARDS
                            .iter()
                            .find(|(suffix, _, _)| name.ends_with(suffix))
                            .map(|(suffix, _, _)| *suffix)
                    })
                    .unwrap_or_else(|| unreachable!("{log:?}: no board in the path"));
                let text = String::from_utf8_lossy(
                    &std::fs::read(&log).unwrap_or_else(|err| unreachable!("read {log:?}: {err}")),
                )
                .into_owned();
                self.read_log(&text, board, &log, &mut seen);
            }
        }
        for (app, check, _, _) in CHECKS {
            for (board, _, _) in BOARDS {
                assert!(
                    seen.contains_key(&(app.to_string(), check.to_string(), board)),
                    "{run:?}: no line for {app} {check} on {board}"
                );
            }
        }
    }

    /// Reads the marker lines of one log. `seen` collects the checks of this run.
    fn read_log(
        &mut self,
        text: &str,
        board: &'static str,
        log: &Path,
        seen: &mut HashMap<(String, String, &'static str), ()>,
    ) {
        let mut app = None;
        for raw in text.lines() {
            let Some(start) = raw.find("HILSHARE_") else {
                continue;
            };
            let line = raw[start..].trim_end();
            if let Some(firmware) = line.strip_prefix("HILSHARE_FIRMWARE=") {
                let (_, name) = firmware
                    .split_once('+')
                    .unwrap_or_else(|| unreachable!("{log:?}: firmware {firmware:?} names no app"));
                app = Some(name.to_owned());
            } else if line.starts_with("HILSHARE_KERNEL_BOOT ") {
                let boot = fields(line);
                let version = boot["version"].to_owned();
                let clock = boot["clock_hz"]
                    .parse::<u64>()
                    .unwrap_or_else(|err| unreachable!("{log:?}: clock_hz: {err}"));
                let previous = self
                    .kernels
                    .entry(board)
                    .or_insert_with(|| (version.clone(), clock));
                assert!(
                    *previous == (version.clone(), clock),
                    "{board}: kernel {version} at {clock} Hz, and {} at {} Hz in another run",
                    previous.0,
                    previous.1
                );
            } else if line.starts_with("HILSHARE_CHECK ") {
                let check = fields(line);
                let name = check["name"];
                if COUNTER_CHECKS.contains(&name) {
                    continue;
                }
                let app = app.clone().unwrap_or_else(|| {
                    unreachable!("{log:?}: a check line before the firmware line")
                });
                assert!(
                    CHECKS
                        .iter()
                        .any(|(row_app, row_name, _, _)| *row_app == app && *row_name == name),
                    "{log:?}: {app} {name} has no row in CHECKS"
                );
                assert_eq!(
                    check["status"], "PASS",
                    "{log:?}: {app} {name} on {board} did not pass"
                );
                let parse = |key: &str| {
                    check[key]
                        .parse::<f64>()
                        .unwrap_or_else(|err| unreachable!("{log:?}: {app} {name} {key}: {err}"))
                };
                let (best, worst) = (parse("us"), parse("worst_us"));
                let key = (app.clone(), name.to_owned(), board);
                let figures = self.figures.entry(key.clone()).or_insert(Figures {
                    best,
                    worst,
                    runs: 0,
                });
                figures.best = figures.best.min(best);
                figures.worst = figures.worst.max(worst);
                if seen.insert(key, ()).is_none() {
                    figures.runs += 1;
                }
            }
        }
    }

    /// The table between the markers.
    #[must_use]
    fn render(&self, commit: &str, date: &str, run_count: usize) -> String {
        let mut body = format!(
            "Commit `{commit}`, {date}, {run_count} runs. Rust 1.92, `opt-level = 3`. `f32` runs on the FPU\n\
             of the Nucleo-F411RE and the Raspberry Pi Pico 2; `f64` runs in software on each board, and the\n\
             XIAO ESP32-C3 has no FPU. Each cell is the best and the worst time of seven repeats.\n\n\
             | Board | Core | Clock |\n|---|---|---:|\n"
        );
        for (suffix, name, core) in BOARDS {
            let (_, clock) = self
                .kernels
                .get(suffix)
                .unwrap_or_else(|| unreachable!("{suffix}: no kernel boot line"));
            let _ = writeln!(body, "| {name} | {core} | {} MHz |", clock / 1_000_000);
        }
        body.push_str("\n| Check | What it runs | Scalar |");
        for (_, name, _) in BOARDS {
            let _ = write!(body, " {name} |");
        }
        body.push_str("\n|---|---|---|---:|---:|---:|\n");
        for (app, check, description, scalar) in CHECKS {
            let _ = write!(body, "| `{check}` | {description} | {scalar} |");
            for (suffix, _, _) in BOARDS {
                let figures = &self.figures[&(app.to_string(), check.to_string(), suffix)];
                assert_eq!(
                    figures.runs, run_count,
                    "{app} {check} on {suffix}: {} of {run_count} runs",
                    figures.runs
                );
                let _ = write!(
                    body,
                    " {} / {} |",
                    format_time(figures.best),
                    format_time(figures.worst)
                );
            }
            body.push('\n');
        }
        body
    }
}

/// Every log file of one artifact, at any depth: each file that is not JUnit XML.
#[must_use]
fn logs(artifact: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![artifact.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .unwrap_or_else(|err| unreachable!("read {directory:?}: {err}"));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|err| unreachable!("read {directory:?}: {err}"))
                .path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) != Some("xml") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mut options = BTreeMap::new();
    let mut runs = Vec::new();
    while let Some(argument) = arguments.next() {
        if let Some(key) = argument.strip_prefix("--") {
            let value = arguments.next().unwrap_or_else(|| unreachable!("{USAGE}"));
            options.insert(key.to_owned(), value);
        } else {
            runs.push(PathBuf::from(argument));
        }
    }
    let commit = options
        .get("commit")
        .unwrap_or_else(|| unreachable!("{USAGE}"));
    let date = options
        .get("date")
        .unwrap_or_else(|| unreachable!("{USAGE}"));
    assert!(!runs.is_empty(), "{USAGE}");

    let mut measurements = Measurements::default();
    for run in &runs {
        measurements.read_run(run);
    }
    let body = measurements.render(commit, date, runs.len());

    let doc = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/silicon.md");
    replace_marked_region(&doc, BEGIN, END, body.trim_end());
    println!("updated {}", doc.display());
}

#[cfg(test)]
mod tests {
    use super::{fields, format_time};

    #[test]
    fn check_line_fields() {
        let line = "HILSHARE_CHECK name=vector_dot status=PASS us=11.32 worst_us=12.01 budget_us=13 \
                    overhead_us=15.26 cycles=1132 worst_cycles=1201 cold_cycles=2400 value=2e1";
        let check = fields(line);
        assert_eq!(check["name"], "vector_dot");
        assert_eq!(check["us"], "11.32");
        assert_eq!(check["worst_us"], "12.01");
    }

    #[test]
    fn time_format() {
        assert_eq!(format_time(11.32), "11.32 µs");
        assert_eq!(format_time(1234.5), "1.23 ms");
        assert_eq!(format_time(3_412_000.0), "3.41 s");
    }
}
