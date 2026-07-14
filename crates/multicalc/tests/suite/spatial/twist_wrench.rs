//! Typed spatial-quantity tests: ordering, converter round trips, vector-space ops, and AD transparency.

use multicalc::linear_algebra::Vector;
use multicalc::scalar::Dual;
use multicalc::spatial::{Twist, Wrench};

#[test]
fn twist_ordering_is_linear_first() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    let w = Vector::new([4.0, 5.0, 6.0]);
    let t = Twist::new(v, w);
    assert_eq!(t.linear(), v);
    assert_eq!(t.angular(), w);
    // to_vector lays the parts out as [v; ω].
    assert_eq!(t.to_vector(), Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_eq!(t.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn wrench_ordering_is_force_first() {
    let f = Vector::new([1.0, 2.0, 3.0]);
    let tau = Vector::new([4.0, 5.0, 6.0]);
    let w = Wrench::new(f, tau);
    assert_eq!(w.force(), f);
    assert_eq!(w.torque(), tau);
    assert_eq!(w.to_vector(), Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_eq!(w.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn twist_converter_roundtrips() {
    let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let t = Twist::from_array(a);
    assert_eq!(t.as_array(), a);
    // vector <-> twist round trip, both explicit and via From/Into.
    let vec = t.to_vector();
    assert_eq!(Twist::from_vector(vec).as_array(), a);
    let via_into: Twist<f64> = vec.into();
    assert_eq!(via_into, t);
    assert_eq!(Vector::from(t), vec);
    // named-part construction agrees with the flat form.
    assert_eq!(Twist::new(t.linear(), t.angular()), t);
}

#[test]
fn wrench_converter_roundtrips() {
    let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let w = Wrench::from_array(a);
    assert_eq!(w.as_array(), a);
    let vec = w.to_vector();
    assert_eq!(Wrench::from_vector(vec).as_array(), a);
    let via_into: Wrench<f64> = vec.into();
    assert_eq!(via_into, w);
    assert_eq!(Vector::from(w), vec);
    assert_eq!(Wrench::new(w.force(), w.torque()), w);
}

#[test]
fn ops_are_componentwise_and_match_vector() {
    let a = Twist::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Twist::from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    // Each op agrees with doing the same on the underlying Vector<6>.
    assert_eq!((a + b).to_vector(), a.to_vector() + b.to_vector());
    assert_eq!((a - b).to_vector(), a.to_vector() - b.to_vector());
    assert_eq!((-a).to_vector(), -a.to_vector());
    assert_eq!(a.scale(2.5).to_vector(), a.to_vector().scale(2.5));
    assert_eq!(Twist::<f64>::zeros().as_array(), [0.0; 6]);

    let c = Wrench::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let d = Wrench::from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    assert_eq!((c + d).to_vector(), c.to_vector() + d.to_vector());
    assert_eq!((c - d).to_vector(), c.to_vector() - d.to_vector());
    assert_eq!((-c).to_vector(), -c.to_vector());
    assert_eq!(c.scale(2.5).to_vector(), c.to_vector().scale(2.5));
}

#[test]
fn ad_transparent_under_dual() {
    // A Twist of Duals carries derivatives through construction, ops, and conversion untouched.
    let t = Twist::new(
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
    let scaled = t.scale(Dual::constant(3.0));
    let out = scaled.to_vector();
    // d/dx (3·x) = 3 in the seeded component; the rest stay finite with zero derivative.
    assert!((out[0].value - 6.0).abs() < 1e-12);
    assert!((out[0].deriv - 3.0).abs() < 1e-12);
    for i in 1..6 {
        assert!(out[i].value.is_finite() && out[i].deriv.is_finite());
        assert!(out[i].deriv.abs() < 1e-12);
    }
}
