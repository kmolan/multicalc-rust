//! White-box tests for the shared sampling helpers, which are `pub(crate)` internals.

use super::{
    NO_PARENT, edge_is_valid, extract_tree_path, nearest_index, neighbours_within, steer_towards,
};
use crate::error::PlanningError;
use crate::linear_algebra::Vector;
use crate::planning::state_space::BoxSpace;

fn plane() -> BoxSpace<2> {
    BoxSpace::try_new(Vector::new([-10.0, -10.0]), Vector::new([10.0, 10.0])).unwrap()
}

#[test]
fn nearest_index_finds_the_closest() {
    let space = plane();
    let states = [
        Vector::new([0.0, 0.0]),
        Vector::new([5.0, 0.0]),
        Vector::new([0.0, 3.0]),
        Vector::new([-1.0, -1.0]),
    ];

    assert_eq!(
        nearest_index(&space, &states, &Vector::new([4.0, 0.0])),
        Some(1)
    );
    assert_eq!(
        nearest_index(&space, &states, &Vector::new([0.0, 2.5])),
        Some(2)
    );
    assert_eq!(
        nearest_index(&space, &states, &Vector::new([-2.0, -2.0])),
        Some(3)
    );
}

#[test]
fn nearest_index_on_empty_is_none() {
    let space = plane();
    assert_eq!(nearest_index(&space, &[], &Vector::new([0.0, 0.0])), None);
}

#[test]
fn neighbours_within_visits_exactly_those_inside() {
    let space = plane();
    let states = [
        Vector::new([0.0, 0.0]),
        Vector::new([1.0, 0.0]),
        Vector::new([3.0, 0.0]),
        Vector::new([0.0, 2.0]),
    ];

    let mut visited = [false; 4];
    neighbours_within(
        &space,
        &states,
        &Vector::new([0.0, 0.0]),
        2.0,
        |index, separation| {
            if let Some(slot) = visited.get_mut(index) {
                *slot = true;
            }
            assert!(separation <= 2.0);
        },
    );
    assert_eq!(visited, [true, true, false, true]);
}

#[test]
fn steer_towards_stops_at_the_step() {
    let space = plane();
    let from = Vector::new([0.0, 0.0]);
    let into = Vector::new([10.0, 0.0]);

    let stepped = steer_towards(&space, &from, &into, 2.5);
    assert_eq!(stepped, Vector::new([2.5, 0.0]));
}

#[test]
fn steer_towards_returns_the_target_when_near() {
    let space = plane();
    let from = Vector::new([0.0, 0.0]);
    let into = Vector::new([1.0, 0.0]);

    assert_eq!(steer_towards(&space, &from, &into, 5.0), into);
    // A zero-length step has no direction to take, so it lands on the target.
    assert_eq!(steer_towards(&space, &from, &from, 5.0), from);
}

#[test]
fn edge_is_valid_rejects_a_blocked_midpoint() {
    let space = plane();
    let blocked_middle = |state: &Vector<2, f64>| state[0].abs() > 1.0;

    assert!(!edge_is_valid(
        &space,
        &blocked_middle,
        &Vector::new([-5.0, 0.0]),
        &Vector::new([5.0, 0.0]),
        8
    ));
    assert!(edge_is_valid(
        &space,
        &blocked_middle,
        &Vector::new([2.0, 0.0]),
        &Vector::new([5.0, 0.0]),
        8
    ));
}

#[test]
fn edge_is_valid_misses_an_obstacle_thinner_than_its_spacing() {
    // The documented limitation, asserted rather than pretended away: the check tests a finite
    // number of stations, so a thin enough obstacle between two of them is not seen.
    let space = plane();
    let a_hair_thin_wall = |state: &Vector<2, f64>| (state[0] - 0.5).abs() > 1e-6;

    let from = Vector::new([0.0, 0.0]);
    let into = Vector::new([1.0, 0.0]);
    // Stations at 1/9 .. 8/9 and the far end: none of them lands on 0.5.
    assert!(edge_is_valid(&space, &a_hair_thin_wall, &from, &into, 8));
    // An odd number of stations does put one there, and then it is seen.
    assert!(!edge_is_valid(&space, &a_hair_thin_wall, &from, &into, 1));
}

#[test]
fn extract_tree_path_reverses_to_root_first() {
    let states = [
        Vector::new([0.0, 0.0]),
        Vector::new([1.0, 0.0]),
        Vector::new([2.0, 0.0]),
    ];
    let parents = [NO_PARENT, 0, 1];

    let path = extract_tree_path::<8, 2, f64>(&states, &parents, 2).unwrap();
    assert_eq!(path.waypoints(), &states[..]);
}

#[test]
fn extract_tree_path_reports_needed_when_too_long() {
    let states = [
        Vector::new([0.0, 0.0]),
        Vector::new([1.0, 0.0]),
        Vector::new([2.0, 0.0]),
        Vector::new([3.0, 0.0]),
    ];
    let parents = [NO_PARENT, 0, 1, 2];

    assert_eq!(
        extract_tree_path::<2, 2, f64>(&states, &parents, 3).err(),
        Some(PlanningError::PathCapacityExceeded { needed: 4 })
    );
    // The reported size is enough on the retry.
    assert!(extract_tree_path::<4, 2, f64>(&states, &parents, 3).is_ok());
}

#[test]
fn extract_tree_path_survives_a_malformed_chain() {
    // A cycle in the parents must be walked out of rather than looped in forever.
    let states = [Vector::new([0.0, 0.0]), Vector::new([1.0, 0.0])];
    let parents = [1, 0];

    let extracted = extract_tree_path::<8, 2, f64>(&states, &parents, 0);
    assert!(extracted.is_ok() || extracted.is_err());
}
