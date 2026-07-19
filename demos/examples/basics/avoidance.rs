//! Follow-the-Gap reactive avoidance driven by a simulated 2D lidar: a closed-form ray-cast scan
//! through a corridor with a pillar, the gap-follower's twist integrated by RK4, and the blocked
//! stop on a walled-in scan.
//!
//! Run with: `cargo run -p multicalc-demos --example avoidance`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::f64::consts::PI;

use multicalc::control::FollowTheGap;
use multicalc::kinematics::Unicycle;
use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk4;
use multicalc_demos::sim::{Lidar2d, Map};
use rand::SeedableRng;
use rand_pcg::Pcg32;

const BEAMS: usize = 61;
const SEED: u64 = 20260719;

fn check(label: &str, condition: bool) {
    assert!(condition, "{label}: failed");
    println!("  {label:<34} ok");
}

fn report(label: &str, value: f64) {
    println!("  {label:<34} = {value:>10.4}");
}

/// What one run of the corridor produced.
struct RunSummary {
    pose: Vector<3, f64>,
    minimum_clearance: f64,
    travelled: f64,
    blocked_ticks: usize,
}

/// Drives the corridor for 30 s of simulated time and reports what happened.
fn run(
    map: &Map,
    lidar: &Lidar2d<BEAMS>,
    follower: &FollowTheGap<BEAMS, f64>,
    seed: u64,
) -> RunSummary {
    let mut rng = Pcg32::seed_from_u64(seed);
    let mut pose = Vector::new([0.5, 0.0, 0.0]);
    let dt = 0.02;
    let mut minimum_clearance = f64::INFINITY;
    let mut travelled = 0.0;
    let mut blocked_ticks = 0;

    for tick in 0..1500 {
        let scan = lidar.simulate(map, [pose[0], pose[1], pose[2]], &mut rng);
        let plan = follower.plan(&scan, 0.0).unwrap();
        if plan.is_blocked() {
            blocked_ticks += 1;
        }
        // Skip a short warm-up so the very first scans do not dominate the minimum.
        if tick >= 10 {
            minimum_clearance = minimum_clearance.min(plan.minimum_clearance());
        }
        travelled += plan.body_twist().linear() * dt;

        let plant = Unicycle::new(plan.body_twist());
        pose = Rk4::step(&plant.field(), 0.0, &pose, dt);
    }

    RunSummary {
        pose,
        minimum_clearance,
        travelled,
        blocked_ticks,
    }
}

fn main() {
    // A 2 m-wide corridor 12 m long, with a pillar offset toward the left wall.
    let map = Map::new()
        .with_segment([0.0, -1.0], [12.0, -1.0]) // right wall
        .with_segment([0.0, 1.0], [12.0, 1.0]) // left wall
        .with_segment([12.0, -1.0], [12.0, 1.0]) // end cap
        .with_circle([5.0, 0.35], 0.35); // pillar

    // The same field of view on both, which is the point of sharing the bearing formula.
    let lidar = Lidar2d::<BEAMS>::new(2.0 * PI / 3.0, 4.0, 0.03, 0.01);
    // The default clear distance is the sensor's maximum range, which suits open ground. Inside a
    // 2 m corridor the frontal arc never sees past about 2 m — a beam 30° off-axis meets the side
    // wall at 1.0 / sin(30°) — so the robot would crawl at half speed the whole way. Scale against
    // distances the corridor can actually produce instead.
    let follower = FollowTheGap::<BEAMS, f64>::try_new(2.0 * PI / 3.0, 4.0, 0.50, 0.60, 0.40)
        .unwrap()
        .with_speed_scaling(0.30, 1.50)
        .unwrap();

    // (1) Drive the corridor.
    let summary = run(&map, &lidar, &follower, SEED);
    println!("Corridor run: 1500 ticks at dt = 0.02 s, seed {SEED}");
    report("x [m]", summary.pose[0]);
    report("y [m]", summary.pose[1]);
    report("heading [rad]", summary.pose[2]);
    report("minimum clearance [m]", summary.minimum_clearance);
    report("distance travelled [m]", summary.travelled);

    // (2) The bars. The pillar spans y in [0.00, 0.70], leaving 0.30 m to the left wall and 1.00 m
    // to the right. A 0.50 m chassis does not fit through the left gap, so the width gate rejects
    // it and the robot must commit to the right. Holding the aim a half-width off each edge keeps
    // it around 0.25 m clear, so a 0.15 m bar has headroom without sitting on the design point.
    println!("\nChecks");
    check("never came within 15 cm", summary.minimum_clearance > 0.15);
    check("kept moving", summary.travelled > 4.0);
    check("got past the pillar", summary.pose[0] > 4.0);
    check("never fully blocked", summary.blocked_ticks == 0);

    // (3) The blocked case, checked directly and independently of the loop.
    let walled_in = follower.plan(&[0.2_f64; BEAMS], 0.0).unwrap();
    check(
        "blocked scan stops",
        walled_in.is_blocked() && walled_in.body_twist().linear() == 0.0,
    );

    // (4) The same seed reproduces the run exactly.
    let repeat = run(&map, &lidar, &follower, SEED);
    check(
        "a fixed seed reproduces the run",
        repeat.pose[0] == summary.pose[0]
            && repeat.pose[1] == summary.pose[1]
            && repeat.pose[2] == summary.pose[2],
    );

    println!("\nAll checks passed.");
}
