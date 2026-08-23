#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the minimum-snap planner against a numpy golden solved a different way.
//!
//! multicalc works out which values are already known, substitutes them out, and
//! solves only for the rest. The generator keeps every coefficient as an unknown
//! and solves the whole constrained system with its multipliers. The two share
//! only the definition of the problem, so matching coefficients mean two
//! independent sets of equations agree.
//!
//! The motion profiles are checked the same way. multicalc derives the phase
//! lengths by case analysis; the generator hands them to a constrained minimizer
//! and asks for the shortest total time that still covers the distance without
//! breaking a limit, then produces the states by integrating the jerk schedule
//! rather than evaluating a cubic. So the durations here are an algebraic answer
//! against a searched one, and the states a closed form against an integration.

use multicalc::linear_algebra::Vector;
use multicalc::motion::{
    BoundaryDerivatives, MinimumSnapPlanner, MotionProfile, MotionProfilePlanner, ProfileLimits,
    SynchronizedProfile,
};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// Sized for the largest fixture: seven segments needs eighteen free values.
type Planner = MinimumSnapPlanner<8, 21, 3, f64>;

#[must_use]
fn boundary(fixture: &Fixture, prefix: &str) -> BoundaryDerivatives<3, f64> {
    BoundaryDerivatives {
        velocity: to_vector::<3>(&fixture.inputs[&format!("{prefix}_velocity")]),
        acceleration: to_vector::<3>(&fixture.inputs[&format!("{prefix}_acceleration")]),
        jerk: to_vector::<3>(&fixture.inputs[&format!("{prefix}_jerk")]),
    }
}

#[test]
fn minimum_snap_matches_an_independent_solve() {
    let fixtures: Vec<_> = load_dir("motion")
        .into_iter()
        .filter(|fixture| fixture.case.starts_with("minimum_snap"))
        .collect();
    assert!(
        fixtures.len() >= 4,
        "expected the whole minimum-snap family"
    );

    for fixture in &fixtures {
        let segments = fixture.inputs["segment_count"].as_int() as usize;
        let durations = fixture.inputs["durations"].as_vector();
        assert_eq!(
            durations.len(),
            segments,
            "{}: duration count",
            fixture.case
        );

        let (rows, columns, points) = fixture.inputs["waypoints"].as_matrix();
        assert_eq!(
            (rows, columns),
            (segments + 1, 3),
            "{}: waypoints",
            fixture.case
        );
        let waypoints: Vec<Vector<3, f64>> = (0..rows)
            .map(|row| Vector::from_fn(|axis| points[row * columns + axis]))
            .collect();

        let trajectory = Planner::new()
            .with_start(boundary(fixture, "start"))
            .with_end(boundary(fixture, "end"))
            .plan(&waypoints, &durations)
            .unwrap();

        // Every segment's every axis, against row `segment · 3 + axis`.
        let (coefficient_rows, coefficient_columns, expected) =
            fixture.expected["coefficients"].as_matrix();
        assert_eq!(
            (coefficient_rows, coefficient_columns),
            (segments * 3, 8),
            "{}: coefficient shape",
            fixture.case
        );
        for segment in 0..segments {
            for axis in 0..3 {
                let found = trajectory.piece_polynomial(segment, axis).unwrap();
                let base = (segment * 3 + axis) * coefficient_columns;
                for power in 0..coefficient_columns {
                    let got = found.coefficient(power).unwrap();
                    let want = expected[base + power];
                    assert!(
                        close(got, want, fixture.tolerances.f64),
                        "{}: segment {segment} axis {axis} coefficient {power}: got {got}, want {want}",
                        fixture.case
                    );
                }
            }
        }

        // And the states those coefficients produce, at the times the fixture names.
        let state_tolerance = fixture.expected["state_tolerance"].as_vector();
        let state_tolerance = Tol {
            abs: state_tolerance[0],
            rel: state_tolerance[1],
        };
        let times = fixture.inputs["sample_times"].as_vector();
        let (state_rows, state_columns, states) = fixture.expected["sampled_states"].as_matrix();
        assert_eq!(
            (state_rows, state_columns),
            (times.len() * 3, 3),
            "{}: sampled state shape",
            fixture.case
        );

        for (sample, time) in times.iter().enumerate() {
            let orders = trajectory.evaluate_with_derivatives::<3>(*time).unwrap();
            for (order, found) in orders.iter().enumerate() {
                let base = (sample * 3 + order) * state_columns;
                for axis in 0..3 {
                    let want = states[base + axis];
                    assert!(
                        close(found[axis], want, state_tolerance),
                        "{}: at {time} order {order} axis {axis}: got {}, want {want}",
                        fixture.case,
                        found[axis]
                    );
                }
            }
        }
    }
}

/// The tolerance a fixture asks its states to be compared at.
#[must_use]
fn state_tolerance(fixture: &Fixture) -> Tol {
    let values = fixture.expected["state_tolerance"].as_vector();
    Tol {
        abs: values[0],
        rel: values[1],
    }
}

/// The limits a single-axis fixture was planned against.
#[must_use]
fn limits_of(fixture: &Fixture) -> ProfileLimits<f64> {
    ProfileLimits::try_new(
        fixture.inputs["speed_limit"].as_scalar(),
        fixture.inputs["acceleration_limit"].as_scalar(),
        (fixture.inputs["has_jerk_limit"].as_int() == 1)
            .then(|| fixture.inputs["jerk_limit"].as_scalar()),
    )
    .unwrap()
}

/// Position, velocity and acceleration against one row of a fixture's states.
fn check_state(
    profile: &MotionProfile<f64>,
    time: f64,
    wanted: &[f64],
    tolerance: Tol,
    case: &str,
    what: &str,
) {
    let state = profile.state_at(time).unwrap();
    for (name, (got, want)) in ["position", "velocity", "acceleration"].iter().zip(
        [state.position, state.velocity, state.acceleration]
            .iter()
            .zip(wanted.iter()),
    ) {
        assert!(
            close(*got, *want, tolerance),
            "{case}: {what} {name} at {time}: got {got}, want {want}"
        );
    }
}

#[test]
fn motion_profiles_match_a_constrained_minimum_time_solve() {
    let fixtures: Vec<_> = load_dir("motion")
        .into_iter()
        .filter(|fixture| {
            fixture.case.starts_with("profile_") && !fixture.case.contains("synchronized")
        })
        .collect();
    assert!(
        fixtures.len() >= 5,
        "expected the whole single-axis profile family"
    );

    for fixture in &fixtures {
        let profile = MotionProfilePlanner::new(limits_of(fixture))
            .plan(fixture.inputs["distance"].as_scalar())
            .unwrap();

        // The seven phase lengths, against what the minimizer settled on.
        let wanted = fixture.expected["phase_durations"].as_vector();
        assert_eq!(wanted.len(), 7, "{}: phase count", fixture.case);
        for (index, (got, want)) in profile
            .phase_durations()
            .iter()
            .zip(wanted.iter())
            .enumerate()
        {
            assert!(
                close(*got, *want, fixture.tolerances.f64),
                "{}: phase {index}: got {got}, want {want}",
                fixture.case
            );
        }

        let total = fixture.expected["total_duration"].as_scalar();
        assert!(
            close(profile.duration(), total, fixture.tolerances.f64),
            "{}: total duration: got {}, want {total}",
            fixture.case,
            profile.duration()
        );

        // Then the states, against the numerically integrated ones.
        let tolerance = state_tolerance(fixture);
        let times = fixture.inputs["sample_times"].as_vector();
        let (rows, columns, states) = fixture.expected["sampled_states"].as_matrix();
        assert_eq!(
            (rows, columns),
            (times.len(), 3),
            "{}: state shape",
            fixture.case
        );

        for (sample, time) in times.iter().enumerate() {
            let base = sample * columns;
            check_state(
                &profile,
                *time,
                &states[base..base + columns],
                tolerance,
                &fixture.case,
                "state",
            );
        }
    }
}

#[test]
fn synchronized_profiles_match_a_constrained_minimum_time_solve() {
    let fixtures: Vec<_> = load_dir("motion")
        .into_iter()
        .filter(|fixture| fixture.case.contains("synchronized"))
        .collect();
    assert!(!fixtures.is_empty(), "expected a synchronized fixture");

    for fixture in &fixtures {
        let displacements = fixture.inputs["displacements"].as_vector();
        let speed_limits = fixture.inputs["speed_limits"].as_vector();
        let acceleration_limits = fixture.inputs["acceleration_limits"].as_vector();
        let jerk_limits = fixture.inputs["jerk_limits"].as_vector();
        let axes = fixture.inputs["axis_count"].as_int() as usize;
        assert_eq!(
            axes, 3,
            "{}: the fixture is written for three axes",
            fixture.case
        );

        // One planner per axis, so each carries its own limits.
        let planned: [MotionProfile<f64>; 3] = core::array::from_fn(|index| {
            let limits = ProfileLimits::try_new(
                speed_limits[index],
                acceleration_limits[index],
                Some(jerk_limits[index]),
            )
            .unwrap();
            MotionProfilePlanner::new(limits)
                .plan(displacements[index])
                .unwrap()
        });
        let synchronized = SynchronizedProfile::<3, f64>::from_profiles(planned);

        let total = fixture.expected["total_duration"].as_scalar();
        assert!(
            close(synchronized.duration(), total, fixture.tolerances.f64),
            "{}: total duration: got {}, want {total}",
            fixture.case,
            synchronized.duration()
        );

        // One row per time and axis, in that order.
        let tolerance = state_tolerance(fixture);
        let times = fixture.inputs["sample_times"].as_vector();
        let (rows, columns, states) = fixture.expected["sampled_states"].as_matrix();
        assert_eq!(
            (rows, columns),
            (times.len() * axes, 3),
            "{}: state shape",
            fixture.case
        );

        for (sample, time) in times.iter().enumerate() {
            for index in 0..axes {
                let base = (sample * axes + index) * columns;
                check_state(
                    synchronized.axis(index).unwrap(),
                    *time,
                    &states[base..base + columns],
                    tolerance,
                    &fixture.case,
                    &format!("axis {index}"),
                );
            }
        }
    }
}
