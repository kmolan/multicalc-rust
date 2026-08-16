//! Point-to-point motion profiles: the fastest move a set of limits allows, and several axes
//! made to finish together.
//!
//! Run with: `cargo run -p multicalc-demos --example motion_profiles`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Instant;

use multicalc::motion::{
    MotionProfile, MotionProfilePlanner, ProfileLimits, ProfileStrategy, SynchronizedProfile,
};

const DISTANCE: f64 = 10.0;
const SPEED_LIMIT: f64 = 2.0;
const ACCELERATION_LIMIT: f64 = 1.0;
const JERK_LIMIT: f64 = 2.0;

/// Three joints with their own limits: velocity, acceleration, jerk.
const JOINT_LIMITS: [(f64, f64, f64); 3] = [(1.0, 2.0, 10.0), (0.4, 1.0, 5.0), (2.0, 2.0, 10.0)];
const JOINT_DISTANCES: [f64; 3] = [1.0, -0.5, 2.0];

fn main() {
    let eased = shapes();
    ceilings(&eased);
    synchronizing();
    evaluating(&eased);
}

#[must_use]
fn plan(speed: f64, acceleration: f64, jerk: Option<f64>, distance: f64) -> MotionProfile<f64> {
    let limits = ProfileLimits::try_new(speed, acceleration, jerk).unwrap();
    MotionProfilePlanner::new(limits).plan(distance).unwrap()
}

/// The same move planned three ways, to show what bounding the jerk costs.
#[must_use]
fn shapes() -> MotionProfile<f64> {
    println!("== Shapes ==");

    let trapezoid = MotionProfilePlanner::new(
        ProfileLimits::try_new(SPEED_LIMIT, ACCELERATION_LIMIT, Some(JERK_LIMIT)).unwrap(),
    )
    .with_strategy(ProfileStrategy::Trapezoidal)
    .plan(DISTANCE)
    .unwrap();
    let eased = plan(SPEED_LIMIT, ACCELERATION_LIMIT, Some(JERK_LIMIT), DISTANCE);
    let barely_eased = plan(SPEED_LIMIT, ACCELERATION_LIMIT, Some(100.0), DISTANCE);

    println!("  {DISTANCE} m at up to {SPEED_LIMIT} m/s and {ACCELERATION_LIMIT} m/s²");
    println!("  shape                 duration   phase durations");
    for (name, profile) in [
        ("trapezoidal", &trapezoid),
        ("jerk 2 m/s³", &eased),
        ("jerk 100 m/s³", &barely_eased),
    ] {
        let phases: Vec<String> = profile
            .phase_durations()
            .iter()
            .map(|duration| format!("{duration:.3}"))
            .collect();
        println!(
            "  {name:<14}     {:>7.4} s   [{}]",
            profile.duration(),
            phases.join(", ")
        );
    }

    // Easing the acceleration in costs time, and how much is the jerk limit's choice.
    assert!(trapezoid.duration() < eased.duration());
    assert!(trapezoid.duration() < barely_eased.duration());
    assert!(barely_eased.duration() - trapezoid.duration() < 0.1);
    println!();
    println!("  Easing the acceleration in and out costs time. Raising the jerk limit");
    println!("  buys it back, closing on the trapezoid it never quite reaches.");
    println!();

    eased
}

/// Walk the profile and check nothing it reports breaks a limit.
fn ceilings(profile: &MotionProfile<f64>) {
    println!("== Limits ==");

    let samples = 2000;
    let mut peak_speed: f64 = 0.0;
    let mut peak_acceleration: f64 = 0.0;
    let mut peak_jerk: f64 = 0.0;
    for step in 0..=samples {
        let time = profile.duration() * f64::from(step) / f64::from(samples);
        let state = profile.state_at(time).unwrap();
        peak_speed = peak_speed.max(state.velocity.abs());
        peak_acceleration = peak_acceleration.max(state.acceleration.abs());
        peak_jerk = peak_jerk.max(state.jerk.abs());
    }
    let landed = profile.state_at(profile.duration()).unwrap().position;

    println!("  over {samples} samples          largest seen   limit");
    println!("  speed                        {peak_speed:>8.5}   {SPEED_LIMIT}");
    println!("  acceleration                 {peak_acceleration:>8.5}   {ACCELERATION_LIMIT}");
    println!("  jerk                         {peak_jerk:>8.5}   {JERK_LIMIT}");
    println!("  arrives at                   {landed:>8.5}   {DISTANCE}");

    assert!(peak_speed <= SPEED_LIMIT + 1e-9);
    assert!(peak_acceleration <= ACCELERATION_LIMIT + 1e-9);
    assert!(peak_jerk <= JERK_LIMIT + 1e-9);
    assert!((landed - DISTANCE).abs() < 1e-9);
    println!();
}

/// Three joints with different limits, made to finish at one instant.
fn synchronizing() {
    println!("== Synchronizing ==");

    let planned: [MotionProfile<f64>; 3] = core::array::from_fn(|index| {
        let (speed, acceleration, jerk) = JOINT_LIMITS[index];
        plan(speed, acceleration, Some(jerk), JOINT_DISTANCES[index])
    });
    let together = SynchronizedProfile::from_profiles(planned);

    println!("  joint   distance   on its own   together");
    for (index, distance) in JOINT_DISTANCES.iter().enumerate() {
        println!(
            "  {index}       {distance:>6.2} m    {:>7.4} s    {:>7.4} s",
            planned[index].duration(),
            together.axis(index).unwrap().duration()
        );
    }

    let end = together.state_at(together.duration()).unwrap();
    for (index, distance) in JOINT_DISTANCES.iter().enumerate() {
        assert!((end.position[index] - distance).abs() < 1e-9);
    }

    // The same three over half again as long, and the same three asked for a tenth of it.
    let stretched =
        SynchronizedProfile::try_from_profiles_over(planned, together.duration() * 1.5).unwrap();
    let hurried =
        SynchronizedProfile::try_from_profiles_over(planned, together.duration() * 0.1).unwrap();

    println!();
    println!("  asked for            got");
    println!(
        "  as fast as possible  {:>7.4} s   set by the slowest joint",
        together.duration()
    );
    println!(
        "  {:>6.4} s (1.5x)      {:>7.4} s   honoured",
        together.duration() * 1.5,
        stretched.duration()
    );
    println!(
        "  {:>6.4} s (0.1x)      {:>7.4} s   too fast, so the fastest feasible one instead",
        together.duration() * 0.1,
        hurried.duration()
    );

    let end = stretched.state_at(stretched.duration()).unwrap();
    for (index, distance) in JOINT_DISTANCES.iter().enumerate() {
        assert!((end.position[index] - distance).abs() < 1e-9);
    }
    assert!((hurried.duration() - together.duration()).abs() < 1e-12);
    println!();
}

/// What the control loop actually runs, once a tick.
fn evaluating(profile: &MotionProfile<f64>) {
    println!("== Evaluating ==");

    let calls = 200;
    let started = Instant::now();
    let mut checksum = 0.0;
    for step in 0..calls {
        let time = profile.duration() * f64::from(step) / f64::from(calls);
        let state = profile.state_at(time).unwrap();
        checksum += state.position + state.velocity + state.acceleration;
    }
    let elapsed = started.elapsed();

    println!(
        "  {calls} evaluations took            {:.1} µs",
        elapsed.as_secs_f64() * 1e6
    );
    println!(
        "  per call                        {:.0} ns  (bounded, no allocation)",
        elapsed.as_secs_f64() * 1e9 / f64::from(calls)
    );
    println!("  checksum                        {checksum:.6}");
    println!("  (both figures are for whichever profile this was built in; add --release");
    println!("   for numbers worth quoting)");
    println!();
    println!("  Unlike minimum snap, planning is bounded here too: a fixed handful of");
    println!("  square and cube roots with nothing to factorize. Both ends are safe on a chip.");
    assert!(checksum.is_finite());
}
