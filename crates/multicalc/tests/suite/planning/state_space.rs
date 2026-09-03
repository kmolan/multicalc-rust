//! The box state space and the validity trait's closure impl.

use multicalc::error::PlanningError;
use multicalc::planning::{BoxSpace, StateSpace, StateValidity};
use multicalc::{Pcg32, Vector};

fn unit_box() -> BoxSpace<3> {
    BoxSpace::try_new(Vector::new([-1.0, 0.0, 2.0]), Vector::new([1.0, 4.0, 3.0])).unwrap()
}

#[test]
fn try_new_rejects_reversed_bounds() {
    assert_eq!(
        BoxSpace::<2>::try_new(Vector::new([1.0, 0.0]), Vector::new([0.0, 1.0])).err(),
        Some(PlanningError::BoundsReversed)
    );
}

#[test]
fn try_new_rejects_non_finite_bounds() {
    assert_eq!(
        BoxSpace::<2>::try_new(Vector::new([f64::NAN, 0.0]), Vector::new([1.0, 1.0])).err(),
        Some(PlanningError::NonFinite)
    );
    assert_eq!(
        BoxSpace::<2>::try_new(Vector::new([0.0, 0.0]), Vector::new([1.0, f64::INFINITY])).err(),
        Some(PlanningError::NonFinite)
    );
}

#[test]
fn samples_land_inside_the_box_f64() {
    let space = unit_box();
    let mut source = Pcg32::new(20260830);

    for _ in 0..10_000 {
        let drawn = space.sample(&mut source);
        assert!(space.contains(&drawn), "{drawn:?}");
    }
}

#[test]
fn samples_cover_the_box_f64() {
    let space = unit_box();
    let mut source = Pcg32::new(20260830);

    let mut lowest = [f64::INFINITY; 3];
    let mut highest = [f64::NEG_INFINITY; 3];
    for _ in 0..10_000 {
        let drawn = space.sample(&mut source);
        for axis in 0..3 {
            lowest[axis] = lowest[axis].min(drawn[axis]);
            highest[axis] = highest[axis].max(drawn[axis]);
        }
    }

    for axis in 0..3 {
        let low = space.lower()[axis];
        let high = space.upper()[axis];
        let covered = (highest[axis] - lowest[axis]) / (high - low);
        assert!(covered >= 0.95, "axis {axis} covered {covered}");
    }
}

#[test]
fn interpolate_at_zero_and_one_returns_the_endpoints_f64() {
    let space = unit_box();
    let from = Vector::new([-1.0, 1.0, 2.0]);
    let into = Vector::new([1.0, 3.0, 3.0]);

    assert_eq!(space.interpolate(&from, &into, 0.0), from);
    assert_eq!(space.interpolate(&from, &into, 1.0), into);
    assert_eq!(
        space.interpolate(&from, &into, 0.5),
        Vector::new([0.0, 2.0, 2.5])
    );
}

#[test]
fn distance_is_symmetric_and_zero_on_itself_f64() {
    let space = unit_box();
    let from = Vector::new([-1.0, 1.0, 2.0]);
    let into = Vector::new([1.0, 3.0, 3.0]);

    assert_eq!(space.distance(&from, &from), 0.0);
    assert_eq!(space.distance(&from, &into), space.distance(&into, &from));
    assert!((space.distance(&from, &into) - 3.0_f64).abs() < 1e-12);
}

#[test]
fn contains_rejects_outside_and_non_finite_f64() {
    let space = unit_box();

    assert!(space.contains(&Vector::new([0.0, 2.0, 2.5])));
    assert!(!space.contains(&Vector::new([2.0, 2.0, 2.5])));
    assert!(!space.contains(&Vector::new([0.0, 2.0, 9.0])));
    assert!(!space.contains(&Vector::new([f64::NAN, 2.0, 2.5])));
}

#[test]
fn a_closure_satisfies_state_validity_f64() {
    // The blanket impl is what keeps the everyday case a one-liner.
    let outside_a_disc = |state: &Vector<2, f64>| state[0].hypot(state[1]) > 1.0;

    assert!(outside_a_disc.is_state_valid(&Vector::new([2.0, 0.0])));
    assert!(!outside_a_disc.is_state_valid(&Vector::new([0.5, 0.5])));
}
