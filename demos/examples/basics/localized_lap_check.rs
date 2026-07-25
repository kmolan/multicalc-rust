//! Headless acceptance check for the localized-lap showcase: boot not knowing where it is, localize
//! on the map with a particle filter, then fuse wheel odometry, an IMU, and GPS while Follow-the-Gap
//! laps the course — asserting zero contacts, centimetre-level fused accuracy, and a fast per-tick cost.
//!
//! The startup localizer runs before the timed loop and is not part of the per-tick cost; the lidar
//! reports every tick, which is a property of this simulation, not a hardware claim.
//!
//! Run with: `cargo run --release -p multicalc-demos --example localized_lap_check`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc_demos::sim::localization_obstacle_avoidance_2d::{LapWorld, Phase};

const DEFAULT_TICKS: u64 = 600_000; // ten minutes of simulated driving, after localization

fn check(label: &str, condition: bool) {
    assert!(condition, "{label}: failed");
    println!("  {label:<44} ok");
}

fn report(label: &str, value: f64) {
    println!("  {label:<44} = {value:>12.4}");
}

/// The value at the given fraction of a sorted list, e.g. `0.5` for the median.
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example localized_lap_check"
        );
    }

    // Optional `--ticks <N>` overrides the driving budget.
    let driving_ticks = std::env::args()
        .skip_while(|argument| argument != "--ticks")
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TICKS);

    let mut world = LapWorld::new(20260722).expect("the pinned configuration is valid");

    // Localization runs before the timed loop and is not counted against the driving budget.
    while world.phase() == Phase::Localizing {
        world.step();
    }

    let mut tick_costs = Vec::with_capacity(driving_ticks as usize);
    for _ in 0..driving_ticks {
        let record = world.step();
        tick_costs.push(record.math_microseconds);
    }

    tick_costs.sort_by(|a, b| a.total_cmp(b));
    let median_microseconds = quantile(&tick_costs, 0.5);
    let p99_microseconds = quantile(&tick_costs, 0.99);

    let metrics = world.metrics();

    println!("\nLocalized lap — {driving_ticks} driving ticks (seed 20260722):\n");
    report("localization ticks", metrics.localization_ticks as f64);
    report("laps", f64::from(metrics.laps));
    report("contacts", metrics.contacts as f64);
    report("minimum clearance (m)", metrics.minimum_clearance);
    report("distance travelled (m)", metrics.distance_travelled);
    report("fused position RMS (m)", metrics.fused_position_rms_error());
    report(
        "dead-reckoned position RMS (m)",
        metrics.dead_reckoned_position_rms_error(),
    );
    report("heading RMS (rad)", metrics.heading_rms_error());
    report(
        "GPS NIS in bounds (fraction)",
        metrics.gps_nis_in_bounds_fraction(),
    );
    report("GPS updates", metrics.gps_updates as f64);
    report("median tick math (us)", median_microseconds);
    report("p99 tick math (us)", p99_microseconds);

    println!("\nChecks:");
    check("localized within 3 s", metrics.localization_ticks <= 3000);
    check("at least five laps", metrics.laps >= 5);
    check("zero contacts", metrics.contacts == 0);
    check("clearance over 6 cm", metrics.minimum_clearance > 0.06);
    check(
        "fused position RMS under 5 cm",
        metrics.fused_position_rms_error() < 0.05,
    );
    check(
        "fusion beats dead reckoning",
        metrics.dead_reckoned_position_rms_error() > 3.0 * metrics.fused_position_rms_error(),
    );
    check(
        "heading RMS under 3 degrees",
        metrics.heading_rms_error() < 0.052,
    );
    check(
        "GPS NIS in chi-square bounds over 90 %",
        metrics.gps_nis_in_bounds_fraction() > 0.90,
    );
    check("median tick math under 200 us", median_microseconds < 200.0);

    println!("\nAll checks passed.");
}
