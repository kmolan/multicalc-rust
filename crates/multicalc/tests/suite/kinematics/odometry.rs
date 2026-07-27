//! SE(2) odometry tests: agreement with an independently integrated unicycle ODE and with the
//! closed-form arc, step-size invariance, degenerate motions, and the retract convention.

use std::f64::consts::PI;

use multicalc::kinematics::{
    BodyArc, BodyTwist, DifferentialDrive, Unicycle, WheelRotations, integrate,
};
use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk45;
use multicalc::spatial::{SE2, SO2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-14;

// ---- helpers ----------------------------------------------------------------

fn random_pose(rng: &mut StdRng) -> SE2<f64> {
    SE2::from_parts(
        SO2::exp(rng.gen_range(-2.5..2.5)),
        Vector::new([rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0)]),
    )
}

// ---- the reference test -----------------------------------------------------

/// The exact-arc integration versus Dormand–Prince quadrature on the unicycle field — a genuinely
/// different numerical path. Our `Rk45` is validated against `scipy.integrate.solve_ivp` goldens by
/// the `qa` suite, so this chains through to an external reference.
#[test]
fn arc_matches_rk45() {
    let final_time = 2.0;
    for (linear_speed, angular_speed) in [(0.4, 0.0), (0.4, 0.9), (0.4, -0.9), (0.0, 0.9)] {
        let rate = BodyTwist::new(linear_speed, angular_speed);
        let solved = Rk45::<f64>::default()
            .with_rtol(1e-12)
            .with_atol(1e-14)
            .solve(
                &Unicycle::new(rate).field(),
                0.0,
                &Vector::new([0.0, 0.0, 0.0]),
                final_time,
            )
            .unwrap();

        let arc = integrate(SE2::identity(), rate.integrate_over(final_time));
        let arc_translation = arc.translation();
        let solved_state = solved;
        assert!(
            (arc_translation[0] - solved_state[0]).abs() < 1e-9,
            "(v={linear_speed}, w={angular_speed}) x: arc {} vs rk45 {}",
            arc_translation[0],
            solved_state[0]
        );
        assert!(
            (arc_translation[1] - solved_state[1]).abs() < 1e-9,
            "(v={linear_speed}, w={angular_speed}) y: arc {} vs rk45 {}",
            arc_translation[1],
            solved_state[1]
        );
        assert!(
            (arc.rotation().log() - solved_state[2]).abs() < 1e-12,
            "(v={linear_speed}, w={angular_speed}) heading: arc {} vs rk45 {}",
            arc.rotation().log(),
            solved_state[2]
        );
    }
}

/// The exact-arc claim, stated as a property no Euler step can satisfy: the result does not depend
/// on how finely the constant twist is subdivided.
#[test]
fn one_big_step_equals_many_small_steps() {
    let rate = BodyTwist::new(0.4, 0.9);
    let total = 2.0;
    let step_count = 1000;

    let one_step = integrate(SE2::identity(), rate.integrate_over(total));

    let small_increment = rate.integrate_over(total / f64::from(step_count));
    let mut many_steps = SE2::identity();
    for _ in 0..step_count {
        many_steps = integrate(many_steps, small_increment);
    }

    let one_step_translation = one_step.translation();
    let many_step_translation = many_steps.translation();
    assert!(
        (one_step_translation[0] - many_step_translation[0]).abs() < TOL,
        "x: {} vs {}",
        one_step_translation[0],
        many_step_translation[0]
    );
    assert!(
        (one_step_translation[1] - many_step_translation[1]).abs() < TOL,
        "y: {} vs {}",
        one_step_translation[1],
        many_step_translation[1]
    );
    assert!((one_step.rotation().log() - many_steps.rotation().log()).abs() < TOL);
}

#[test]
fn arc_matches_closed_form() {
    let linear_speed = 0.4_f64;
    let angular_speed = 0.9;
    let time = 1.3;
    let heading_change = angular_speed * time;
    let radius = linear_speed / angular_speed;

    let arc = integrate(
        SE2::identity(),
        BodyTwist::new(linear_speed, angular_speed).integrate_over(time),
    );
    let translation = arc.translation();
    assert!((translation[0] - radius * heading_change.sin()).abs() < TOL);
    assert!((translation[1] - radius * (1.0 - heading_change.cos())).abs() < TOL);
    assert!((arc.rotation().log() - heading_change).abs() < TOL);
}

// ---- degenerate motions -----------------------------------------------------

#[test]
fn zero_angular_is_straight_line() {
    let pose = integrate(SE2::identity(), BodyArc::new(0.5_f64, 0.0));
    let translation = pose.translation();
    assert!(translation[0].is_finite() && translation[1].is_finite());
    assert_eq!(translation[0], 0.5);
    assert_eq!(translation[1], 0.0);
    assert_eq!(pose.rotation().log(), 0.0);
}

#[test]
fn zero_linear_is_pure_rotation() {
    let mut rng = StdRng::seed_from_u64(0x0d0_1111);
    let start = random_pose(&mut rng);
    let pose = integrate(start, BodyArc::new(0.0, 0.7));

    let start_translation = start.translation();
    let end_translation = pose.translation();
    assert_eq!(start_translation[0], end_translation[0]);
    assert_eq!(start_translation[1], end_translation[1]);
}

// ---- conventions ------------------------------------------------------------

#[test]
fn is_right_perturbation() {
    let mut rng = StdRng::seed_from_u64(0x0d0_2222);
    for _ in 0..100 {
        let pose = random_pose(&mut rng);
        let increment = BodyArc::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        let got = integrate(pose, increment);
        let want = pose * SE2::exp(Vector::new([increment.linear(), 0.0, increment.angular()]));
        assert_eq!(got, want);
    }
}

#[test]
fn identity_start_equals_exp() {
    let increment = BodyArc::new(0.3, -0.4);
    let got = integrate(SE2::identity(), increment);
    let want = SE2::exp(Vector::new([0.3, 0.0, -0.4]));
    assert_eq!(got, want);
}

#[test]
fn odometry_step_matches_integrate() {
    let drive = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    let mut rng = StdRng::seed_from_u64(0x0d0_3333);
    for _ in 0..100 {
        let pose = random_pose(&mut rng);
        let increment = WheelRotations::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        assert_eq!(
            drive.odometry_step(pose, increment),
            integrate(pose, drive.forward_arc(increment))
        );
    }
}

// ---- end to end -------------------------------------------------------------

/// Two full circles of opposite curvature, driven through the whole wheel-to-pose chain. The sign
/// change is the point: it exercises both curvature directions and must return to the start.
#[test]
fn figure_eight_closes() {
    let drive = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    let linear_speed = 0.36;
    let angular_speed = 0.9;
    let step_count = 2000;
    let timestep = (2.0 * PI / angular_speed) / f64::from(step_count);

    let mut pose = SE2::identity();
    for sign in [1.0, -1.0] {
        let wheel_rates = drive.inverse(BodyTwist::new(linear_speed, angular_speed * sign));
        let increment = WheelRotations::new(
            wheel_rates.left() * timestep,
            wheel_rates.right() * timestep,
        );
        for _ in 0..step_count {
            pose = drive.odometry_step(pose, increment);
        }
    }

    let translation = pose.translation();
    assert!(
        translation[0].abs() < 1e-9,
        "x did not close: {}",
        translation[0]
    );
    assert!(
        translation[1].abs() < 1e-9,
        "y did not close: {}",
        translation[1]
    );
    assert!(
        pose.rotation().log().abs() < 1e-9,
        "heading did not close: {}",
        pose.rotation().log()
    );
}
