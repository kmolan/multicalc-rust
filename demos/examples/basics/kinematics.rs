//! Differential-drive kinematics: wheel/body maps and their round trip, exact SE(2) odometry
//! against the closed-form arc, a figure eight driven through the encoder path, and a one-`Dual`
//! autodiff derivative through an odometry step.
//!
//! Run with: `cargo run -p multicalc-demos --example kinematics`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::f64::consts::PI;

use multicalc::kinematics::{BodyTwist, DifferentialDrive, WheelVelocities, integrate};
use multicalc::scalar::Dual;
use multicalc::spatial::SE2;

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<20} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    // A 36 mm wheel radius and a 235 mm track width.
    let dd = DifferentialDrive::new(0.036_f64, 0.235).unwrap();

    // (1) Wheel velocities to a body twist. Equal drives straight; opposite spins in place.
    let straight = dd.forward(WheelVelocities::new(10.0, 10.0));
    println!("Both wheels at 10 rad/s");
    report("v [m/s]", straight.linear(), 0.36);
    report("omega [rad/s]", straight.angular(), 0.0);

    let spin = dd.forward(WheelVelocities::new(-10.0, 10.0));
    println!("\nWheels at -10 and +10 rad/s");
    report("v [m/s]", spin.linear(), 0.0);
    report("omega [rad/s]", spin.angular(), 0.72 / 0.235);

    // (2) The maps are a bijection, so the round trip is an identity.
    let wheels = WheelVelocities::new(7.5, -2.25);
    let back = dd.inverse(dd.forward(wheels));
    println!("\nRound trip: wheels -> body twist -> wheels");
    report("left [rad/s]", back.left(), 7.5);
    report("right [rad/s]", back.right(), -2.25);

    // (3) Odometry along the exact constant-twist arc, against the closed form for radius v/omega.
    let (v, w, t) = (0.4_f64, 0.9, 1.3);
    let pose = integrate(SE2::identity(), BodyTwist::new(v, w).integrate_over(t));
    let (theta, radius) = (w * t, v / w);
    println!("\nArc of a constant twist (v = 0.4, omega = 0.9) held for 1.3 s");
    report("x [m]", pose.translation()[0], radius * theta.sin());
    report("y [m]", pose.translation()[1], radius * (1.0 - theta.cos()));
    report("heading [rad]", pose.rotation().log(), theta);

    // (4) The encoder path, end to end: two full circles of opposite curvature must return to the
    // start. The sign change is the point — it exercises both turning directions.
    let n = 2000;
    let dt = (2.0 * PI / w) / f64::from(n);
    let mut figure_eight = SE2::identity();
    for sign in [1.0, -1.0] {
        let vel = dd.inverse(BodyTwist::new(0.36, w * sign));
        let (left_m, right_m) = (
            vel.left() * dd.wheel_radius() * dt,
            vel.right() * dd.wheel_radius() * dt,
        );
        for _ in 0..n {
            let rotations = dd.wheel_rotations_from_travel(left_m, right_m);
            figure_eight = dd.odometry_step(figure_eight, rotations);
        }
    }
    println!("\nFigure eight: two opposed circles through the encoder path");
    report("x [m]", figure_eight.translation()[0], 0.0);
    report("y [m]", figure_eight.translation()[1], 0.0);
    report("heading [rad]", figure_eight.rotation().log(), 0.0);

    // (5) One Dual through an odometry step: d(x)/d(arc length), exact, no hand-derived formula.
    let step = integrate(
        SE2::<Dual<f64>>::identity(),
        BodyTwist::new(Dual::variable(v), Dual::constant(w)).integrate_over(Dual::constant(t)),
    );
    println!("\nAutodiff through an odometry step");
    // x = (v/omega)*sin(omega*t), so dx/dv = sin(omega*t)/omega.
    report("dx/dv [s]", step.translation()[0].deriv, theta.sin() / w);
    report(
        "dy/dv [s]",
        step.translation()[1].deriv,
        (1.0 - theta.cos()) / w,
    );

    println!("\nAll checks passed.");
}
