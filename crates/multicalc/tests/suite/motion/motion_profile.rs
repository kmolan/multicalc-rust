//! Motion profile tests: phase durations against hand-worked cases, limits held at every sampled
//! instant, continuity across the phase joins, and the degenerate moves.

use multicalc::error::MotionError;
use multicalc::motion::{
    MotionProfile, MotionProfilePlanner, PROFILE_PHASE_COUNT, ProfileLimits, ProfileState,
    ProfileStrategy, SynchronizedProfile,
};
use multicalc::scalar::{Dual, Numeric};

const SPEED_LIMIT: f64 = 2.0;
const ACCELERATION_LIMIT: f64 = 1.0;
const JERK_LIMIT: f64 = 2.0;

fn trapezoidal_limits() -> ProfileLimits<f64> {
    ProfileLimits::try_new(SPEED_LIMIT, ACCELERATION_LIMIT, None).unwrap()
}

fn jerk_limited_limits() -> ProfileLimits<f64> {
    ProfileLimits::try_new(SPEED_LIMIT, ACCELERATION_LIMIT, Some(JERK_LIMIT)).unwrap()
}

fn plan(limits: ProfileLimits<f64>, distance: f64) -> MotionProfile<f64> {
    MotionProfilePlanner::new(limits).plan(distance).unwrap()
}

/// Both shapes against a short, a medium and a long move.
fn every_case() -> [(ProfileLimits<f64>, f64); 6] {
    let mut cases = [(trapezoidal_limits(), 0.0); 6];
    for (slot, (limits, distance)) in cases.iter_mut().zip(
        [trapezoidal_limits(), jerk_limited_limits()]
            .into_iter()
            .flat_map(|limits| [0.25, 1.5, 10.0].map(|distance| (limits, distance))),
    ) {
        *slot = (limits, distance);
    }
    cases
}

/// `count` times evenly spaced across the whole profile, ends included.
fn sample_times(profile: &MotionProfile<f64>, count: usize) -> impl Iterator<Item = f64> + use<'_> {
    (0..=count).map(move |step| profile.duration() * step as f64 / count as f64)
}

fn assert_close(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!(
        (got - want).abs() < tolerance,
        "{what}: got {got}, want {want}"
    );
}

// ---- limits ------------------------------------------------------------------

#[test]
fn limits_reject_a_ceiling_that_is_not_positive() {
    let rejected = [
        (0.0, 1.0, None),
        (-1.0, 1.0, None),
        (f64::NAN, 1.0, None),
        (1.0, 0.0, None),
        (1.0, 1.0, Some(0.0)),
        (1.0, 1.0, Some(f64::INFINITY)),
    ];

    for (speed, acceleration, jerk) in rejected {
        assert_eq!(
            ProfileLimits::try_new(speed, acceleration, jerk).err(),
            Some(MotionError::LimitNotPositive),
            "({speed}, {acceleration}, {jerk:?})"
        );
    }
}

// ---- the shapes --------------------------------------------------------------

#[test]
fn trapezoid_with_a_cruise_has_the_expected_durations() {
    let profile = plan(trapezoidal_limits(), 10.0);

    assert_close(profile.duration(), 7.0, 1e-12, "duration");
    assert_close(profile.peak_speed(), 2.0, 1e-12, "peak speed");
    for (got, want) in profile
        .phase_durations()
        .iter()
        .zip([0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0].iter())
    {
        assert_close(*got, *want, 1e-12, "phase duration");
    }
}

#[test]
fn trapezoid_too_short_to_reach_the_speed_ceiling() {
    let profile = plan(trapezoidal_limits(), 1.0);

    assert_close(profile.duration(), 2.0, 1e-12, "duration");
    assert_close(profile.peak_speed(), 1.0, 1e-12, "peak speed");
    assert_close(profile.phase_durations()[3], 0.0, 1e-12, "cruise");
}

#[test]
fn jerk_limited_with_every_phase_present() {
    let profile = plan(jerk_limited_limits(), 10.0);

    assert_close(profile.duration(), 7.5, 1e-12, "duration");
    for (got, want) in profile
        .phase_durations()
        .iter()
        .zip([0.5, 1.5, 0.5, 2.5, 0.5, 1.5, 0.5].iter())
    {
        assert_close(*got, *want, 1e-12, "phase duration");
    }
}

#[test]
fn jerk_limited_reaches_the_acceleration_ceiling_but_never_cruises() {
    let profile = plan(jerk_limited_limits(), 1.5);

    assert_close(profile.duration(), 3.0, 1e-12, "duration");
    assert_close(profile.peak_speed(), 1.0, 1e-12, "peak speed");
    assert_close(profile.phase_durations()[3], 0.0, 1e-12, "cruise");
}

#[test]
fn jerk_limited_too_short_to_reach_the_acceleration_ceiling() {
    let profile = plan(jerk_limited_limits(), 0.25);

    // No constant-acceleration phase and no cruise: acceleration ramps up and straight back down.
    assert_close(profile.phase_durations()[1], 0.0, 1e-12, "hold");
    assert_close(profile.phase_durations()[3], 0.0, 1e-12, "cruise");

    let peak_acceleration = sample_times(&profile, 200).fold(0.0_f64, |peak, time| {
        peak.max(profile.state_at(time).unwrap().acceleration.abs())
    });
    assert!(
        peak_acceleration < ACCELERATION_LIMIT,
        "peak acceleration {peak_acceleration}"
    );
}

// ---- what every profile has to do --------------------------------------------

#[test]
fn a_profile_covers_exactly_the_distance_it_was_given() {
    for (limits, distance) in every_case() {
        let profile = plan(limits, distance);

        let end = profile.state_at(profile.duration()).unwrap();
        assert_close(end.position, distance, 1e-12, "end position");
        assert_close(profile.distance(), distance, 1e-12, "distance");
        assert_close(
            profile.state_at(0.0).unwrap().position,
            0.0,
            1e-12,
            "start position",
        );
    }
}

#[test]
fn a_profile_starts_and_finishes_at_rest() {
    for (limits, distance) in every_case() {
        let profile = plan(limits, distance);

        for time in [0.0, profile.duration()] {
            let state = profile.state_at(time).unwrap();
            assert!(state.velocity.abs() < 1e-12, "velocity at {time}");
        }
        // The move is over at the end, so nothing is commanded there either.
        let end = profile.state_at(profile.duration()).unwrap();
        assert!(end.acceleration.abs() < 1e-12, "acceleration at the end");
    }
}

#[test]
fn a_trapezoid_commands_full_acceleration_from_the_first_instant() {
    // Acceleration steps at t = 0 and `state_at` is right-continuous, so it reads the commanded
    // value rather than the zero it steps from. Bounding the jerk is what removes the step.
    let stepped = plan(trapezoidal_limits(), 10.0).state_at(0.0).unwrap();
    assert_close(stepped.acceleration, ACCELERATION_LIMIT, 1e-12, "trapezoid");

    let eased = plan(jerk_limited_limits(), 10.0).state_at(0.0).unwrap();
    assert_close(eased.acceleration, 0.0, 1e-12, "jerk-limited");

    // And it follows the direction of travel.
    let backward = plan(trapezoidal_limits(), -10.0).state_at(0.0).unwrap();
    assert_close(
        backward.acceleration,
        -ACCELERATION_LIMIT,
        1e-12,
        "backward",
    );
}

#[test]
fn a_profile_never_breaks_its_ceilings() {
    for (limits, distance) in every_case() {
        let profile = plan(limits, distance);

        for time in sample_times(&profile, 1000) {
            let state = profile.state_at(time).unwrap();
            assert!(
                state.velocity.abs() <= limits.speed() + 1e-9,
                "velocity {} at {time}",
                state.velocity
            );
            assert!(
                state.acceleration.abs() <= limits.acceleration() + 1e-9,
                "acceleration {} at {time}",
                state.acceleration
            );
            if let Some(jerk_limit) = limits.jerk() {
                assert!(
                    state.jerk.abs() <= jerk_limit + 1e-9,
                    "jerk {} at {time}",
                    state.jerk
                );
            }
        }
    }
}

#[test]
fn position_and_velocity_do_not_jump_across_a_phase_join() {
    let profile = plan(jerk_limited_limits(), 10.0);
    let offset = 1e-7;

    let mut boundary = 0.0;
    for phase in profile.phase_durations() {
        boundary += phase;
        let before = profile.state_at(boundary - offset).unwrap();
        let after = profile.state_at(boundary + offset).unwrap();

        // The residual is the sampling `offset` either side, not a break in the curve: it shrinks
        // with the offset, where a genuine discontinuity would not.
        assert_close(after.position, before.position, 1e-5, "position at a join");
        assert_close(after.velocity, before.velocity, 1e-5, "velocity at a join");
        assert_close(
            after.acceleration,
            before.acceleration,
            1e-5,
            "acceleration at a join",
        );
    }
}

#[test]
fn a_negative_distance_mirrors_a_positive_one() {
    let forward = plan(jerk_limited_limits(), 10.0);
    let backward = plan(jerk_limited_limits(), -10.0);

    assert_close(backward.duration(), forward.duration(), 1e-12, "duration");
    for time in sample_times(&forward, 50) {
        let one = forward.state_at(time).unwrap();
        let other = backward.state_at(time).unwrap();
        assert_close(other.position, -one.position, 1e-12, "position");
        assert_close(other.velocity, -one.velocity, 1e-12, "velocity");
    }
}

#[test]
fn a_move_of_no_distance_takes_no_time() {
    for strategy in [ProfileStrategy::Trapezoidal, ProfileStrategy::JerkLimited] {
        let profile = MotionProfilePlanner::new(jerk_limited_limits())
            .with_strategy(strategy)
            .plan(0.0)
            .unwrap();

        assert_eq!(profile.duration(), 0.0, "{strategy:?}");
        assert_eq!(profile.distance(), 0.0, "{strategy:?}");
        assert_eq!(profile.state_at(0.0).unwrap(), ProfileState::default());
    }
}

// ---- picking the shape -------------------------------------------------------

#[test]
fn a_jerk_limited_plan_needs_a_jerk_ceiling() {
    let planner =
        MotionProfilePlanner::new(trapezoidal_limits()).with_strategy(ProfileStrategy::JerkLimited);

    assert_eq!(
        planner.plan(10.0).err(),
        Some(MotionError::JerkLimitRequired)
    );
}

#[test]
fn automatic_picks_by_the_ceilings_it_was_given() {
    let pairs = [
        (trapezoidal_limits(), ProfileStrategy::Trapezoidal),
        (jerk_limited_limits(), ProfileStrategy::JerkLimited),
    ];

    for (limits, explicit) in pairs {
        let automatic = plan(limits, 10.0);
        let named = MotionProfilePlanner::new(limits)
            .with_strategy(explicit)
            .plan(10.0)
            .unwrap();

        for (got, want) in automatic
            .phase_durations()
            .iter()
            .zip(named.phase_durations().iter())
        {
            assert_close(*got, *want, 1e-12, "phase duration");
        }
    }
}

#[test]
fn a_high_jerk_ceiling_approaches_the_trapezoid() {
    let trapezoid = plan(trapezoidal_limits(), 10.0).duration();

    let mut previous_gap = f64::INFINITY;
    for jerk in [1e3, 1e5, 1e7] {
        let limits = ProfileLimits::try_new(SPEED_LIMIT, ACCELERATION_LIMIT, Some(jerk)).unwrap();
        let gap = plan(limits, 10.0).duration() - trapezoid;

        assert!(gap > 0.0, "easing in is free at a jerk ceiling of {jerk}");
        if previous_gap.is_finite() {
            let shrinkage = previous_gap / gap;
            assert!(
                (shrinkage - 100.0).abs() < 1.0,
                "gap shrank by {shrinkage} at {jerk}, not about 100"
            );
        }
        previous_gap = gap;
    }
    assert!(
        previous_gap < 1e-6,
        "gap at a jerk ceiling of 1e7 is {previous_gap}"
    );
}

// ---- evaluating and stretching -----------------------------------------------

#[test]
fn a_bad_time_is_refused_and_times_outside_the_profile_clamp() {
    let profile = plan(jerk_limited_limits(), 10.0);

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(profile.state_at(bad).err(), Some(MotionError::NonFinite));
    }
    assert_eq!(
        profile.state_at(-1.0).unwrap(),
        profile.state_at(0.0).unwrap()
    );
    assert_eq!(
        profile.state_at(profile.duration() + 1.0).unwrap(),
        profile.state_at(profile.duration()).unwrap()
    );
}

#[test]
fn stretching_keeps_the_path_and_slows_it_down() {
    let original = plan(jerk_limited_limits(), 10.0);
    let stretched = original.stretched_to(15.0);
    let scale = 15.0 / original.duration();

    assert_close(stretched.duration(), 15.0, 1e-12, "duration");
    assert_close(stretched.distance(), original.distance(), 1e-12, "distance");

    for step in 0..=100 {
        let fraction = f64::from(step) / 100.0;
        let slow = stretched.state_at(fraction * 15.0).unwrap();
        let fast = original.state_at(fraction * original.duration()).unwrap();

        assert_close(slow.position, fast.position, 1e-12, "position");
        assert_close(slow.velocity, fast.velocity / scale, 1e-12, "velocity");
    }
}

#[test]
fn stretching_to_something_shorter_changes_nothing() {
    let profile = plan(jerk_limited_limits(), 10.0);

    for duration in [3.0, 0.0, -1.0, f64::NAN] {
        assert_eq!(profile.stretched_to(duration), profile, "{duration}");
    }
}

// ---- autodiff ----------------------------------------------------------------

/// Where a jerk-limited profile for `distance` has got to at `time`.
fn position_at<T: Numeric>(distance: T, time: T) -> T {
    let limits = ProfileLimits::try_new(
        T::from_f64(SPEED_LIMIT),
        T::from_f64(ACCELERATION_LIMIT),
        Some(T::from_f64(JERK_LIMIT)),
    )
    .unwrap();
    MotionProfilePlanner::new(limits)
        .plan(distance)
        .unwrap()
        .state_at(time)
        .unwrap()
        .position
}

#[test]
fn a_profile_carries_a_dual_derivative() {
    // Mid-cruise, so both profiles either side of the step take the same branch.
    let time = 3.75;
    let distance = 10.0;

    let autodiff = position_at(Dual::variable(distance), Dual::constant(time)).deriv;
    assert!(autodiff.is_finite(), "dual part {autodiff}");

    let step = 1e-6;
    let finite_difference =
        (position_at(distance + step, time) - position_at(distance - step, time)) / (2.0 * step);
    assert!(
        (autodiff - finite_difference).abs() < 1e-5 * finite_difference.abs().max(1.0),
        "autodiff {autodiff}, finite difference {finite_difference}"
    );
}

#[test]
fn a_profile_has_seven_phases() {
    assert_eq!(
        plan(jerk_limited_limits(), 10.0).phase_durations().len(),
        PROFILE_PHASE_COUNT
    );
}

// ---- synchronizing -----------------------------------------------------------

/// Three joints with their own limits: velocity, acceleration, jerk.
const AXIS_LIMITS: [(f64, f64, f64); 3] = [(1.0, 2.0, 10.0), (0.5, 1.0, 5.0), (2.0, 2.0, 10.0)];

fn axis_profiles(displacements: [f64; 3]) -> [MotionProfile<f64>; 3] {
    let mut axes = [plan(jerk_limited_limits(), 0.0); 3];
    for (axis, (ceilings, displacement)) in axes
        .iter_mut()
        .zip(AXIS_LIMITS.iter().zip(displacements.iter()))
    {
        let (speed, acceleration, jerk) = *ceilings;
        let limits = ProfileLimits::try_new(speed, acceleration, Some(jerk)).unwrap();
        *axis = plan(limits, *displacement);
    }
    axes
}

fn slowest_of(axes: &[MotionProfile<f64>; 3]) -> f64 {
    axes.iter()
        .fold(0.0_f64, |longest, axis| longest.max(axis.duration()))
}

#[test]
fn every_axis_finishes_together() {
    let displacements = [1.0, -0.5, 2.0];
    let axes = axis_profiles(displacements);
    let slowest = slowest_of(&axes);
    let synchronized = SynchronizedProfile::from_profiles(axes);

    assert_close(synchronized.duration(), slowest, 1e-12, "duration");

    let end = synchronized.state_at(synchronized.duration()).unwrap();
    for (index, displacement) in displacements.iter().enumerate() {
        assert_close(end.position[index], *displacement, 1e-12, "end position");
        assert!(
            end.velocity[index].abs() < 1e-12,
            "velocity on axis {index}"
        );
    }
}

#[test]
fn a_still_axis_stays_still() {
    let moving = SynchronizedProfile::from_profiles(axis_profiles([1.0, -0.5, 2.0]));
    let with_still = SynchronizedProfile::from_profiles(axis_profiles([1.0, -0.5, 0.0]));

    // Axis 2 was the fastest of the three, so dropping it changes the pace.
    let still_axes = axis_profiles([1.0, -0.5, 0.0]);
    assert_close(
        with_still.duration(),
        slowest_of(&still_axes),
        1e-12,
        "duration",
    );
    assert!(with_still.duration() <= moving.duration() + 1e-12);

    for step in 0..=100 {
        let time = with_still.duration() * f64::from(step) / 100.0;
        assert_eq!(with_still.state_at(time).unwrap().position[2], 0.0);
    }
}

#[test]
fn synchronizing_never_breaks_a_ceiling() {
    let synchronized = SynchronizedProfile::from_profiles(axis_profiles([1.0, -0.5, 2.0]));

    for step in 0..=1000 {
        let time = synchronized.duration() * f64::from(step) / 1000.0;
        let state = synchronized.state_at(time).unwrap();

        for (index, (speed, acceleration, jerk)) in AXIS_LIMITS.iter().enumerate() {
            assert!(
                state.velocity[index].abs() <= speed + 1e-9,
                "velocity {} on axis {index} at {time}",
                state.velocity[index]
            );
            assert!(
                state.acceleration[index].abs() <= acceleration + 1e-9,
                "acceleration {} on axis {index} at {time}",
                state.acceleration[index]
            );
            assert!(
                state.jerk[index].abs() <= jerk + 1e-9,
                "jerk {} on axis {index} at {time}",
                state.jerk[index]
            );
        }
    }
}

#[test]
fn a_longer_duration_is_honoured() {
    let displacements = [1.0, -0.5, 2.0];
    let axes = axis_profiles(displacements);
    let asked = slowest_of(&axes) * 1.5;

    let synchronized = SynchronizedProfile::try_from_profiles_over(axes, asked).unwrap();

    assert_close(synchronized.duration(), asked, 1e-12, "duration");
    let end = synchronized.state_at(synchronized.duration()).unwrap();
    for (index, displacement) in displacements.iter().enumerate() {
        assert_close(end.position[index], *displacement, 1e-12, "end position");
    }
}

#[test]
fn a_duration_that_is_too_fast_falls_back_to_the_fastest_feasible_one() {
    let axes = axis_profiles([1.0, -0.5, 2.0]);
    let slowest = slowest_of(&axes);

    let synchronized = SynchronizedProfile::try_from_profiles_over(axes, slowest * 0.1).unwrap();

    assert_close(synchronized.duration(), slowest, 1e-12, "duration");
}

#[test]
fn a_duration_that_is_not_positive_is_refused() {
    let axes = axis_profiles([1.0, -0.5, 2.0]);

    for duration in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            SynchronizedProfile::try_from_profiles_over(axes, duration).err(),
            Some(MotionError::DurationNotPositive),
            "{duration}"
        );
    }
}

#[test]
fn a_bad_time_is_refused() {
    let synchronized = SynchronizedProfile::from_profiles(axis_profiles([1.0, -0.5, 2.0]));

    assert_eq!(
        synchronized.state_at(f64::NAN).err(),
        Some(MotionError::NonFinite)
    );
}

#[test]
fn axis_hands_back_the_profile_it_was_built_from() {
    let axes = axis_profiles([1.0, -0.5, 2.0]);
    let slowest = slowest_of(&axes);
    let synchronized = SynchronizedProfile::from_profiles(axes);

    assert!(synchronized.axis(0).is_some());
    assert!(synchronized.axis(2).is_some());
    assert!(synchronized.axis(3).is_none());

    // Every axis handed back runs to the common finish time.
    for index in 0..3 {
        let axis = synchronized.axis(index).unwrap();
        assert_close(axis.duration(), slowest, 1e-12, "axis duration");
    }
}
