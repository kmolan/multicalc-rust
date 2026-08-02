//! Minimum-snap trajectories: planning one off the loop, then evaluating it inside one.
//!
//! Run with: `cargo run -p multicalc-demos --example minimum_snap_trajectory`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Instant;

use multicalc::linear_algebra::Vector;
use multicalc::motion::{MinimumSnapPlanner, PiecewisePolynomial, durations_from_average_speed};

/// Six waypoints in three dimensions, so five segments.
const WAYPOINTS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [2.0, 1.0, 1.0],
    [4.0, -1.0, 1.5],
    [6.0, 1.5, 2.0],
    [8.0, 0.5, 1.0],
    [10.0, 0.0, 0.0],
];
const AVERAGE_SPEED: f64 = 2.0;

type Trajectory = PiecewisePolynomial<8, 8, 3, f64>;

fn main() {
    let (trajectory, durations) = planning();
    checking(&trajectory, &durations);
    evaluating(&trajectory);
}

/// Time the segments, then solve for the smoothest path through them.
#[must_use]
fn planning() -> (Trajectory, Vec<f64>) {
    println!("== Planning ==");

    let waypoints: Vec<Vector<3, f64>> = WAYPOINTS.iter().map(|p| Vector::new(*p)).collect();
    let mut durations = vec![0.0; waypoints.len() - 1];
    durations_from_average_speed(&waypoints, AVERAGE_SPEED, &mut durations).unwrap();

    // Five segments need fifteen chosen values, so the workspace is sized past that.
    let planner = MinimumSnapPlanner::<8, 21, 3, f64>::new();
    let started = Instant::now();
    let trajectory = planner.plan(&waypoints, &durations).unwrap();
    let elapsed = started.elapsed();

    println!("  waypoints                 {}", waypoints.len());
    println!("  segments                  {}", trajectory.piece_count());
    println!(
        "  durations from {AVERAGE_SPEED} m/s      {:?}",
        durations
            .iter()
            .map(|d| (d * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    println!(
        "  total duration            {:.3} s",
        trajectory.total_span()
    );
    println!(
        "  planning took             {:.1} µs  (once, off the loop)",
        elapsed.as_secs_f64() * 1e6
    );
    println!();

    (trajectory, durations)
}

/// The path has to go where it was told, start and finish still, and not jerk at the joins.
fn checking(trajectory: &Trajectory, durations: &[f64]) {
    println!("== Checking ==");

    let mut boundaries = vec![0.0];
    for duration in durations {
        boundaries.push(boundaries.last().unwrap() + duration);
    }

    let mut worst_waypoint = 0.0_f64;
    for (time, waypoint) in boundaries.iter().zip(WAYPOINTS.iter()) {
        let found = trajectory.evaluate(*time).unwrap();
        for axis in 0..3 {
            worst_waypoint = worst_waypoint.max((found[axis] - waypoint[axis]).abs());
        }
    }
    println!("  largest miss at a waypoint            {worst_waypoint:.2e} m");
    assert!(worst_waypoint < 1e-9);

    let mut worst_rest = 0.0_f64;
    for time in [0.0, trajectory.total_span()] {
        let orders = trajectory.evaluate_with_derivatives::<3>(time).unwrap();
        for motion in orders.iter().skip(1) {
            for value in motion.into_array() {
                worst_rest = worst_rest.max(value.abs());
            }
        }
    }
    println!("  largest speed or turn at either end   {worst_rest:.2e}");
    assert!(worst_rest < 1e-9);

    // Sampling either side of each join: a break in the curve would show as a jump here.
    let offset = 1e-6;
    let mut worst_jump = 0.0_f64;
    for joint in &boundaries[1..boundaries.len() - 1] {
        let before = trajectory
            .evaluate_with_derivatives::<3>(joint - offset)
            .unwrap();
        let after = trajectory
            .evaluate_with_derivatives::<3>(joint + offset)
            .unwrap();
        for order in 0..3 {
            for axis in 0..3 {
                worst_jump = worst_jump.max((before[order][axis] - after[order][axis]).abs());
            }
        }
    }
    println!("  largest jump across a join            {worst_jump:.2e}");
    println!("  (that is the 1e-6 sampling either side, not a break in the curve)");
    assert!(worst_jump < 1e-3);
    println!();
}

/// What the control loop actually runs.
fn evaluating(trajectory: &Trajectory) {
    println!("== Evaluating ==");

    let total = trajectory.total_span();
    let calls = 200;
    let started = Instant::now();
    let mut checksum = 0.0;
    for step in 0..calls {
        let time = total * step as f64 / calls as f64;
        let orders = trajectory.evaluate_with_derivatives::<3>(time).unwrap();
        checksum += orders[0][0] + orders[1][1] + orders[2][2];
    }
    let elapsed = started.elapsed();

    println!(
        "  {calls} evaluations took            {:.1} µs",
        elapsed.as_secs_f64() * 1e6
    );
    println!(
        "  per call                        {:.0} ns  (bounded, no allocation)",
        elapsed.as_secs_f64() * 1e9 / calls as f64
    );
    println!("  checksum                        {checksum:.6}");
    println!("  (both figures are for whichever profile this was built in; add --release");
    println!("   for numbers worth quoting)");
    println!();
    println!("  Planning grows with the number of waypoints and factorizes a matrix.");
    println!("  Evaluating is fixed work per call, which is why one runs off the loop");
    println!("  and the other runs inside it.");
    assert!(checksum.is_finite());
}
