//! SO(3) and SE(3) Lie groups: compose, act on a point, exp/log round trips, geodesic
//! interpolation, and a one-`Dual` autodiff derivative through the whole composition.
//!
//! Run with: `cargo run --example lie_groups`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::f64::consts::FRAC_PI_2;

use multicalc::linear_algebra::Vector;
use multicalc::scalar::Dual;
use multicalc::spatial::{SE3, SO3};

fn report(label: &str, value: f64, exact: f64) {
    println!(
        "  {label:<20} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    // (1) SO(3): a 90° rotation about z maps x -> y.
    let rz = SO3::<f64>::exp(Vector::new([0.0, 0.0, FRAC_PI_2]));
    let p = rz.act(Vector::new([1.0, 0.0, 0.0]));
    println!("SO(3): 90 deg about z applied to (1, 0, 0)");
    report("x", p[0], 0.0);
    report("y", p[1], 1.0);
    report("z", p[2], 0.0);

    // (2) exp/log round trip recovers the rotation vector.
    let phi = Vector::new([0.3, -0.6, 0.2]);
    let back = SO3::exp(phi).log();
    println!("\nSO(3): log(exp(phi)) recovers phi");
    for i in 0..3 {
        report(&format!("phi[{i}]"), back[i], phi[i]);
    }

    // (3) SE(3): rotate then translate a point.
    let g = SE3::from_parts(rz, Vector::new([1.0, 2.0, 3.0]));
    let q = g.act(Vector::new([1.0, 0.0, 0.0]));
    println!("\nSE(3): rotate then translate (1, 0, 0)");
    report("x", q[0], 1.0);
    report("y", q[1], 3.0);
    report("z", q[2], 3.0);

    // (4) Geodesic interpolation: midpoint of identity and a 90° rotation is 45°.
    let mid = SO3::<f64>::identity().interpolate(rz, 0.5);
    println!("\nSO(3): slerp midpoint angle about z");
    report("angle (rad)", mid.log()[2], FRAC_PI_2 / 2.0);

    // (5) Autodiff: d/dtheta of exp(theta*z).act(x) at theta = 0 is (0, 1, 0) — one Dual pushed
    //     through exp and the rotation, no hand-derived Jacobian.
    let theta = Dual::variable(0.0);
    let rot = SO3::exp(Vector::new([
        Dual::constant(0.0),
        Dual::constant(0.0),
        theta,
    ]));
    let out = rot.act(Vector::new([
        Dual::constant(1.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]));
    println!("\nAutodiff: d/dtheta exp(theta*z).act(x) at theta = 0");
    report("d x / d theta", out[0].deriv, 0.0);
    report("d y / d theta", out[1].deriv, 1.0);
    report("d z / d theta", out[2].deriv, 0.0);

    // (6) Left Jacobian: J_l(phi) times its inverse is the identity.
    let jjinv = SO3::left_jacobian(phi) * SO3::left_jacobian_inverse(phi);
    println!("\nSO(3): left Jacobian times its inverse is the identity");
    report("(J_l * J_l^-1)[0,0]", jjinv[(0, 0)], 1.0);
}
