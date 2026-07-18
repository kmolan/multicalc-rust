//! PolylinePath tests: arc length, projection, lookahead advance and end-of-path modes, degenerate
//! waypoints, and the error paths, at f32 and f64.

use multicalc::error::MotionError;
use multicalc::linear_algebra::Vector;
use multicalc::motion::{EndOfPath, PolylinePath};
use multicalc::scalar::Numeric;

// ---- arc length -------------------------------------------------------------

fn assert_l_shaped_arc_length<T: Numeric>(tolerance: T) {
    let path: PolylinePath<3, 2, T> = PolylinePath::try_from_points(&[
        Vector::new([T::ZERO, T::ZERO]),
        Vector::new([T::from_f64(3.0), T::ZERO]),
        Vector::new([T::from_f64(3.0), T::from_f64(4.0)]),
    ])
    .unwrap();
    assert!((path.total_arc_length() - T::from_f64(7.0)).abs() < tolerance);
}

#[test]
fn total_arc_length_sums_segments_f64() {
    assert_l_shaped_arc_length(1e-12_f64);
}

#[test]
fn total_arc_length_sums_segments_f32() {
    assert_l_shaped_arc_length(1e-4_f32);
}

// ---- projection -------------------------------------------------------------

#[test]
fn point_on_path_projects_to_itself() {
    let path: PolylinePath<3, 2, f64> = PolylinePath::try_from_points(&[
        Vector::new([0.0, 0.0]),
        Vector::new([4.0, 0.0]),
        Vector::new([4.0, 4.0]),
    ])
    .unwrap();
    // (4, 2) sits on the second segment, arc length 4 + 2 = 6.
    let projection = path.closest_point(Vector::new([4.0, 2.0])).unwrap();
    assert!(projection.distance() < 1e-12);
    assert!((projection.arc_length() - 6.0).abs() < 1e-12);
    assert_eq!(projection.segment_index(), 1);
    let [x, y] = projection.point().into_array();
    assert!((x - 4.0).abs() < 1e-12 && (y - 2.0).abs() < 1e-12);
}

#[test]
fn single_point_path_projects_to_that_point() {
    let mut path = PolylinePath::<3, 2, f64>::new();
    path.push(Vector::new([2.0, 2.0])).unwrap();
    let projection = path.closest_point(Vector::new([5.0, 6.0])).unwrap();
    assert_eq!(projection.segment_index(), 0);
    assert!((projection.distance() - 5.0).abs() < 1e-12);
    let [x, y] = path.lookahead_point(0.0, 10.0).unwrap().into_array();
    assert!((x - 2.0).abs() < 1e-12 && (y - 2.0).abs() < 1e-12);
}

// ---- lookahead --------------------------------------------------------------

#[test]
fn lookahead_advances_monotonically() {
    let path: PolylinePath<3, 2, f64> = PolylinePath::try_from_points(&[
        Vector::new([0.0, 0.0]),
        Vector::new([5.0, 0.0]),
        Vector::new([10.0, 0.0]),
    ])
    .unwrap();
    let mut previous_x = -1.0;
    let mut from = 0.0;
    while from <= 8.0 {
        let [x, _] = path.lookahead_point(from, 1.0).unwrap().into_array();
        assert!(x >= previous_x - 1e-12);
        previous_x = x;
        from += 0.5;
    }
}

#[test]
fn lookahead_inside_first_segment_is_exact_distance() {
    let path: PolylinePath<2, 2, f64> =
        PolylinePath::try_from_points(&[Vector::new([0.0, 0.0]), Vector::new([10.0, 0.0])]).unwrap();
    let [x, y] = path.lookahead_point(0.0, 3.5).unwrap().into_array();
    assert!((x - 3.5).abs() < 1e-12 && y.abs() < 1e-12);
}

#[test]
fn stop_clamps_past_the_end() {
    let path: PolylinePath<2, 2, f64> =
        PolylinePath::try_from_points(&[Vector::new([0.0, 0.0]), Vector::new([4.0, 0.0])]).unwrap();
    let [x, y] = path.lookahead_point(0.0, 100.0).unwrap().into_array();
    assert!((x - 4.0).abs() < 1e-12 && y.abs() < 1e-12);
}

#[test]
fn loop_wraps_past_the_end() {
    let path: PolylinePath<2, 2, f64> =
        PolylinePath::try_from_points(&[Vector::new([0.0, 0.0]), Vector::new([4.0, 0.0])])
            .unwrap()
            .with_end_of_path(EndOfPath::Loop);
    // Total length 4; a target of 5 wraps to 1.
    let [x, y] = path.lookahead_point(0.0, 5.0).unwrap().into_array();
    assert!((x - 1.0).abs() < 1e-12 && y.abs() < 1e-12);
}

// ---- degenerate waypoints ---------------------------------------------------

#[test]
fn duplicate_waypoints_do_not_divide_by_zero() {
    let path: PolylinePath<4, 2, f64> = PolylinePath::try_from_points(&[
        Vector::new([0.0, 0.0]),
        Vector::new([0.0, 0.0]),
        Vector::new([2.0, 0.0]),
        Vector::new([2.0, 0.0]),
    ])
    .unwrap();
    assert!((path.total_arc_length() - 2.0).abs() < 1e-12);

    let projection = path.closest_point(Vector::new([1.0, 1.0])).unwrap();
    assert!(projection.distance().is_finite());
    assert!((projection.arc_length() - 1.0).abs() < 1e-12);

    let [x, y] = path.lookahead_point(0.0, 1.0).unwrap().into_array();
    assert!(x.is_finite() && y.is_finite());
}

// ---- error paths ------------------------------------------------------------

#[test]
fn capacity_exceeded_from_slice_and_push() {
    let too_many = [Vector::new([0.0_f64, 0.0]); 3];
    assert_eq!(
        PolylinePath::<2, 2, f64>::try_from_points(&too_many).err(),
        Some(MotionError::CapacityExceeded)
    );

    let mut path = PolylinePath::<1, 2, f64>::new();
    path.push(Vector::new([0.0, 0.0])).unwrap();
    assert_eq!(
        path.push(Vector::new([1.0, 1.0])).err(),
        Some(MotionError::CapacityExceeded)
    );
}

#[test]
fn queries_on_empty_path_are_too_short() {
    let empty = PolylinePath::<3, 2, f64>::new();
    assert_eq!(
        empty.closest_point(Vector::new([0.0, 0.0])).err(),
        Some(MotionError::PathTooShort)
    );
    assert_eq!(
        empty.lookahead_point(0.0, 1.0).err(),
        Some(MotionError::PathTooShort)
    );
}

#[test]
fn non_finite_waypoint_is_rejected() {
    assert_eq!(
        PolylinePath::<3, 2, f64>::try_from_points(&[Vector::new([f64::NAN, 0.0])]).err(),
        Some(MotionError::NonFinite)
    );
}
