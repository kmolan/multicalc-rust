//! Minimum-snap planner tests: hitting every waypoint, honouring the boundary motion, staying
//! smooth across the joins, and being genuinely the lowest-cost answer rather than merely a
//! continuous one.

use multicalc::error::MotionError;
use multicalc::linear_algebra::Vector;
use multicalc::motion::{
    BoundaryDerivatives, MinimumSnapPlanner, PiecewisePolynomial, durations_from_average_speed,
};
use multicalc::polynomial::Polynomial;

/// Three segments in three dimensions.
type Trajectory = PiecewisePolynomial<4, 8, 3, f64>;

const WAYPOINTS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 0.5],
    [3.0, 1.0, 1.5],
    [4.0, 3.0, 1.0],
];
const DURATIONS: [f64; 3] = [1.0, 1.5, 1.2];
/// Where each waypoint falls along the trajectory.
const BOUNDARIES: [f64; 4] = [0.0, 1.0, 2.5, 3.7];

fn waypoints() -> [Vector<3, f64>; 4] {
    WAYPOINTS.map(Vector::new)
}

fn plan_from_rest() -> Trajectory {
    MinimumSnapPlanner::<4, 9, 3, f64>::new()
        .plan(&waypoints(), &DURATIONS)
        .unwrap()
}

/// The total snap the trajectory spends, by sampling the fourth derivative across the whole path.
fn snap_cost(trajectory: &Trajectory) -> f64 {
    let samples = 1000;
    let total = trajectory.total_span();
    let step = total / samples as f64;

    let mut cost = 0.0;
    for index in 0..samples {
        let time = (index as f64 + 0.5) * step;
        let orders = trajectory.evaluate_with_derivatives::<5>(time).unwrap();
        cost += orders[4].norm_squared() * step;
    }
    cost
}

/// The position and first three derivatives at each waypoint, read off a planned trajectory.
fn waypoint_blocks(trajectory: &Trajectory) -> [[Vector<3, f64>; 4]; 4] {
    BOUNDARIES.map(|time| trajectory.evaluate_with_derivatives::<4>(time).unwrap())
}

/// Rebuilds a trajectory from waypoint blocks, so a block can be altered and the cost compared.
fn rebuild(blocks: &[[Vector<3, f64>; 4]; 4]) -> Trajectory {
    let mut pieces = [[Polynomial::<8>::zeros(); 3]; 3];
    for (segment, piece) in pieces.iter_mut().enumerate() {
        for (axis, slot) in piece.iter_mut().enumerate() {
            let start: [f64; 4] = core::array::from_fn(|order| blocks[segment][order][axis]);
            let end: [f64; 4] = core::array::from_fn(|order| blocks[segment + 1][order][axis]);
            *slot = Polynomial::<8>::from_endpoint_derivatives(&start, &end, DURATIONS[segment])
                .unwrap();
        }
    }
    PiecewisePolynomial::try_from_pieces(&pieces, &DURATIONS).unwrap()
}

// ---- what the trajectory has to do -------------------------------------------

#[test]
fn passes_through_every_waypoint() {
    let trajectory = plan_from_rest();

    for (time, waypoint) in BOUNDARIES.iter().zip(WAYPOINTS.iter()) {
        let found = trajectory.evaluate(*time).unwrap();
        for axis in 0..3 {
            assert!((found[axis] - waypoint[axis]).abs() < 1e-9);
        }
    }
}

#[test]
fn starts_and_ends_at_rest() {
    let trajectory = plan_from_rest();

    for time in [0.0, trajectory.total_span()] {
        let orders = trajectory.evaluate_with_derivatives::<4>(time).unwrap();
        // Everything above position: velocity, acceleration and jerk.
        for motion in orders.iter().skip(1) {
            for value in motion.into_array() {
                assert!(value.abs() < 1e-9);
            }
        }
    }
}

#[test]
fn honours_non_zero_boundary_conditions() {
    let start = BoundaryDerivatives {
        velocity: Vector::new([1.0, 0.0, 0.0]),
        ..Default::default()
    };
    let trajectory = MinimumSnapPlanner::<4, 9, 3, f64>::new()
        .with_start(start)
        .plan(&waypoints(), &DURATIONS)
        .unwrap();

    let orders = trajectory.evaluate_with_derivatives::<4>(0.0).unwrap();
    assert!((orders[1][0] - 1.0).abs() < 1e-9);
    assert!(orders[1][1].abs() < 1e-9 && orders[1][2].abs() < 1e-9);
    // The path still arrives everywhere it should.
    let last = trajectory.evaluate(trajectory.total_span()).unwrap();
    for axis in 0..3 {
        assert!((last[axis] - WAYPOINTS[3][axis]).abs() < 1e-9);
    }
}

#[test]
fn is_smooth_across_joints() {
    let trajectory = plan_from_rest();
    let offset = 1e-6;

    // The two interior waypoints, where one segment hands over to the next.
    for joint in [BOUNDARIES[1], BOUNDARIES[2]] {
        let before = trajectory
            .evaluate_with_derivatives::<5>(joint - offset)
            .unwrap();
        let after = trajectory
            .evaluate_with_derivatives::<5>(joint + offset)
            .unwrap();

        for order in 0..4 {
            for axis in 0..3 {
                let gap = (before[order][axis] - after[order][axis]).abs();
                // Sampling either side of the joint moves each order by roughly the next one up
                // times the distance covered, so that is the whole of what the gap should be. A
                // genuine break in the curve would be nothing like this small, and unlike this it
                // would not shrink as the offset does.
                let from_sampling = before[order + 1][axis]
                    .abs()
                    .max(after[order + 1][axis].abs())
                    * 2.0
                    * offset;
                assert!(
                    gap <= 4.0 * from_sampling + 1e-9,
                    "joint {joint} order {order} axis {axis}: gap {gap:e} against {from_sampling:e}"
                );
            }
        }
    }
}

#[test]
fn cost_is_lowest_at_the_solution() {
    let trajectory = plan_from_rest();
    let blocks = waypoint_blocks(&trajectory);

    // Rebuilding from the planned values reproduces the planned trajectory, so the comparison below
    // is against the same thing measured the same way.
    let solved_cost = snap_cost(&rebuild(&blocks));
    assert!((solved_cost - snap_cost(&trajectory)).abs() < 1e-6 * solved_cost.max(1.0));

    // Moving an interior velocity either way has to cost more. Nothing else about the trajectory
    // changes: it still passes through every waypoint and is still continuous everywhere.
    for waypoint in 1..3 {
        for axis in 0..3 {
            for nudge in [-0.05, 0.05] {
                let mut altered = blocks;
                altered[waypoint][1][axis] += nudge;
                assert!(
                    snap_cost(&rebuild(&altered)) > solved_cost,
                    "moving waypoint {waypoint} axis {axis} by {nudge} did not cost more"
                );
            }
        }
    }
}

#[test]
fn single_segment_plans_without_a_solve() {
    // Two waypoints leave nothing for the solve to choose, so it is skipped entirely.
    let waypoints = [Vector::new([0.0, 0.0, 0.0]), Vector::new([2.0, 1.0, -1.0])];
    let trajectory = MinimumSnapPlanner::<4, 9, 3, f64>::new()
        .plan(&waypoints, &[2.0])
        .unwrap();

    assert_eq!(trajectory.piece_count(), 1);
    let finish = trajectory.evaluate(2.0).unwrap();
    for (axis, expected) in [2.0, 1.0, -1.0].iter().enumerate() {
        assert!((finish[axis] - expected).abs() < 1e-12);
    }
    // Still at rest at both ends.
    for time in [0.0, 2.0] {
        let orders = trajectory.evaluate_with_derivatives::<4>(time).unwrap();
        for motion in orders.iter().skip(1) {
            for value in motion.into_array() {
                assert!(value.abs() < 1e-12);
            }
        }
    }
}

// ---- what it refuses ---------------------------------------------------------

#[test]
fn rejects_a_short_workspace() {
    // Five segments need fifteen free values, and this planner holds three.
    let many = [
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([1.0, 0.0, 0.0]),
        Vector::new([2.0, 1.0, 0.0]),
        Vector::new([3.0, 1.0, 1.0]),
        Vector::new([4.0, 2.0, 1.0]),
        Vector::new([5.0, 2.0, 2.0]),
    ];
    assert_eq!(
        MinimumSnapPlanner::<8, 3, 3, f64>::new()
            .plan(&many, &[1.0; 5])
            .err(),
        Some(MotionError::WorkspaceTooSmall)
    );
}

#[test]
fn rejects_bad_input() {
    let planner = MinimumSnapPlanner::<4, 9, 3, f64>::new();
    let points = waypoints();

    assert_eq!(
        planner.plan(&points[..1], &[]).err(),
        Some(MotionError::PathTooShort)
    );
    assert_eq!(
        planner.plan(&points, &[1.0, 1.0]).err(),
        Some(MotionError::SegmentCountMismatch)
    );
    assert_eq!(
        planner.plan(&points, &[1.0, 0.0, 1.0]).err(),
        Some(MotionError::DurationNotPositive)
    );

    let mut broken = points;
    broken[2] = Vector::new([f64::NAN, 0.0, 0.0]);
    assert_eq!(
        planner.plan(&broken, &DURATIONS).err(),
        Some(MotionError::NonFinite)
    );

    // Three segments do not fit a planner built for two.
    assert_eq!(
        MinimumSnapPlanner::<2, 9, 3, f64>::new()
            .plan(&points, &DURATIONS)
            .err(),
        Some(MotionError::CapacityExceeded)
    );
}

// ---- the duration helper -----------------------------------------------------

#[test]
fn durations_from_average_speed_covers_the_path() {
    let points = waypoints();
    let mut durations = [0.0_f64; 3];
    let speed = 2.5;
    durations_from_average_speed(&points, speed, &mut durations).unwrap();

    let mut distance = 0.0;
    for pair in points.windows(2) {
        distance += (pair[1] - pair[0]).norm();
    }
    assert!((durations.iter().sum::<f64>() - distance / speed).abs() < 1e-12);

    // A pair of waypoints in the same place covers no distance, so no duration suits it.
    let repeated = [
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([1.0, 0.0, 0.0]),
    ];
    let mut two = [0.0_f64; 2];
    assert_eq!(
        durations_from_average_speed(&repeated, speed, &mut two).err(),
        Some(MotionError::DurationNotPositive)
    );
}

// ---- at single precision -----------------------------------------------------

#[test]
fn runs_in_f32() {
    let points: [Vector<3, f32>; 4] =
        WAYPOINTS.map(|point| Vector::new(point.map(|value| value as f32)));
    let durations: [f32; 3] = DURATIONS.map(|duration| duration as f32);

    let trajectory = MinimumSnapPlanner::<4, 9, 3, f32>::new()
        .plan(&points, &durations)
        .unwrap();

    for (time, waypoint) in BOUNDARIES.iter().zip(WAYPOINTS.iter()) {
        let found = trajectory.evaluate(*time as f32).unwrap();
        for axis in 0..3 {
            assert!((found[axis] - waypoint[axis] as f32).abs() < 1e-3);
        }
    }
}
