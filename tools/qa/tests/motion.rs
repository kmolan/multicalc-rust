#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the minimum-snap planner against a numpy golden solved a different way.
//!
//! multicalc works out which values are already known, substitutes them out, and
//! solves only for the rest. The generator keeps every coefficient as an unknown
//! and solves the whole constrained system with its multipliers. The two share
//! only the definition of the problem, so matching coefficients mean two
//! independent sets of equations agree.

use multicalc::linear_algebra::Vector;
use multicalc::motion::{BoundaryDerivatives, MinimumSnapPlanner};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// Sized for the largest fixture: seven segments needs eighteen free values.
type Planner = MinimumSnapPlanner<8, 21, 3, f64>;

fn boundary(fx: &Fixture, prefix: &str) -> BoundaryDerivatives<3, f64> {
    BoundaryDerivatives {
        velocity: to_vector::<3>(&fx.inputs[&format!("{prefix}_velocity")]),
        acceleration: to_vector::<3>(&fx.inputs[&format!("{prefix}_acceleration")]),
        jerk: to_vector::<3>(&fx.inputs[&format!("{prefix}_jerk")]),
    }
}

#[test]
fn minimum_snap_matches_an_independent_solve() {
    let fixtures = load_dir("motion");
    assert!(fixtures.len() >= 4, "expected the whole motion family");

    for fx in &fixtures {
        let segments = fx.inputs["segment_count"].as_int() as usize;
        let durations = fx.inputs["durations"].as_vector();
        assert_eq!(durations.len(), segments, "{}: duration count", fx.case);

        let (rows, columns, points) = fx.inputs["waypoints"].as_matrix();
        assert_eq!((rows, columns), (segments + 1, 3), "{}: waypoints", fx.case);
        let waypoints: Vec<Vector<3, f64>> = (0..rows)
            .map(|row| Vector::from_fn(|axis| points[row * columns + axis]))
            .collect();

        let trajectory = Planner::new()
            .with_start(boundary(fx, "start"))
            .with_end(boundary(fx, "end"))
            .plan(&waypoints, &durations)
            .unwrap();

        // Every segment's every axis, against row `segment · 3 + axis`.
        let (coefficient_rows, coefficient_columns, expected) =
            fx.expected["coefficients"].as_matrix();
        assert_eq!(
            (coefficient_rows, coefficient_columns),
            (segments * 3, 8),
            "{}: coefficient shape",
            fx.case
        );
        for segment in 0..segments {
            for axis in 0..3 {
                let found = trajectory.piece_polynomial(segment, axis).unwrap();
                let base = (segment * 3 + axis) * coefficient_columns;
                for power in 0..coefficient_columns {
                    let got = found.coefficient(power).unwrap();
                    let want = expected[base + power];
                    assert!(
                        close(got, want, fx.tolerances.f64),
                        "{}: segment {segment} axis {axis} coefficient {power}: got {got}, want {want}",
                        fx.case
                    );
                }
            }
        }

        // And the states those coefficients produce, at the times the fixture names.
        let state_tolerance = fx.expected["state_tolerance"].as_vector();
        let state_tolerance = Tol {
            abs: state_tolerance[0],
            rel: state_tolerance[1],
        };
        let times = fx.inputs["sample_times"].as_vector();
        let (state_rows, state_columns, states) = fx.expected["sampled_states"].as_matrix();
        assert_eq!(
            (state_rows, state_columns),
            (times.len() * 3, 3),
            "{}: sampled state shape",
            fx.case
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
                        fx.case,
                        found[axis]
                    );
                }
            }
        }
    }
}
