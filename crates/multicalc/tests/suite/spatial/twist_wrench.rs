//! Typed spatial-quantity tests: ordering, converter round trips, vector-space ops, and AD transparency.

use multicalc::linear_algebra::Vector;
use multicalc::scalar::Dual;
use multicalc::spatial::{Twist, Wrench};

#[test]
fn twist_ordering_is_linear_first() {
    let linear = Vector::new([1.0, 2.0, 3.0]);
    let angular = Vector::new([4.0, 5.0, 6.0]);
    let twist = Twist::new(linear, angular);
    assert_eq!(twist.linear(), linear);
    assert_eq!(twist.angular(), angular);
    // to_vector lays the parts out as [v; ω].
    assert_eq!(
        twist.to_vector(),
        Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
    assert_eq!(twist.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn wrench_ordering_is_force_first() {
    let force = Vector::new([1.0, 2.0, 3.0]);
    let torque = Vector::new([4.0, 5.0, 6.0]);
    let wrench = Wrench::new(force, torque);
    assert_eq!(wrench.force(), force);
    assert_eq!(wrench.torque(), torque);
    assert_eq!(
        wrench.to_vector(),
        Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
    assert_eq!(wrench.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn twist_converter_roundtrips() {
    let components = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let twist = Twist::from_array(components);
    assert_eq!(twist.as_array(), components);
    // vector <-> twist round trip, both explicit and via From/Into.
    let as_vector = twist.to_vector();
    assert_eq!(Twist::from_vector(as_vector).as_array(), components);
    let via_into: Twist<f64> = as_vector.into();
    assert_eq!(via_into, twist);
    assert_eq!(Vector::from(twist), as_vector);
    // named-part construction agrees with the flat form.
    assert_eq!(Twist::new(twist.linear(), twist.angular()), twist);
}

#[test]
fn wrench_converter_roundtrips() {
    let components = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let wrench = Wrench::from_array(components);
    assert_eq!(wrench.as_array(), components);
    let as_vector = wrench.to_vector();
    assert_eq!(Wrench::from_vector(as_vector).as_array(), components);
    let via_into: Wrench<f64> = as_vector.into();
    assert_eq!(via_into, wrench);
    assert_eq!(Vector::from(wrench), as_vector);
    assert_eq!(Wrench::new(wrench.force(), wrench.torque()), wrench);
}

#[test]
fn ops_are_componentwise_and_match_vector() {
    let first_twist = Twist::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let second_twist = Twist::from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    // Each op agrees with doing the same on the underlying Vector6D.
    assert_eq!(
        (first_twist + second_twist).to_vector(),
        first_twist.to_vector() + second_twist.to_vector()
    );
    assert_eq!(
        (first_twist - second_twist).to_vector(),
        first_twist.to_vector() - second_twist.to_vector()
    );
    assert_eq!((-first_twist).to_vector(), -first_twist.to_vector());
    assert_eq!(
        first_twist.scale(2.5).to_vector(),
        first_twist.to_vector().scale(2.5)
    );
    assert_eq!(Twist::<f64>::zeros().as_array(), [0.0; 6]);

    let first_wrench = Wrench::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let second_wrench = Wrench::from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    assert_eq!(
        (first_wrench + second_wrench).to_vector(),
        first_wrench.to_vector() + second_wrench.to_vector()
    );
    assert_eq!(
        (first_wrench - second_wrench).to_vector(),
        first_wrench.to_vector() - second_wrench.to_vector()
    );
    assert_eq!((-first_wrench).to_vector(), -first_wrench.to_vector());
    assert_eq!(
        first_wrench.scale(2.5).to_vector(),
        first_wrench.to_vector().scale(2.5)
    );
}

#[test]
fn ad_transparent_under_dual() {
    // A Twist of Duals carries derivatives through construction, ops, and conversion untouched.
    let twist = Twist::new(
        Vector::new([
            Dual::variable(2.0_f64),
            Dual::constant(0.0),
            Dual::constant(0.0),
        ]),
        Vector::new([
            Dual::constant(0.0),
            Dual::constant(0.0),
            Dual::constant(0.0),
        ]),
    );
    let scaled = twist.scale(Dual::constant(3.0));
    let outputs = scaled.to_vector();
    // d/dx (3·x) = 3 in the seeded component; the rest stay finite with zero derivative.
    assert!((outputs[0].value - 6.0).abs() < 1e-12);
    assert!((outputs[0].deriv - 3.0).abs() < 1e-12);
    for index in 1..6 {
        let output = outputs[index];
        assert!(output.value.is_finite() && output.deriv.is_finite());
        assert!(output.deriv.abs() < 1e-12);
    }
}

#[test]
fn wrench_default_is_zero() {
    let wrench = Wrench::<f64>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let zero = Wrench::<f64>::default();

    assert_eq!(zero, Wrench::zeros());
    assert_eq!(wrench + zero, wrench);
    assert_eq!(zero + wrench, wrench);
}

#[test]
fn twist_default_is_zero() {
    let twist = Twist::<f64>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let zero = Twist::<f64>::default();

    assert_eq!(zero, Twist::zeros());
    assert_eq!(twist + zero, twist);
    assert_eq!(zero + twist, twist);
}
