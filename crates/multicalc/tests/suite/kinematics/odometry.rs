//! SE(2) odometry tests: agreement with an independently integrated unicycle ODE and with the
//! closed-form arc, step-size invariance, degenerate motions, and the retract convention.

use std::f64::consts::PI;

use multicalc::kinematics::{ChassisDelta, ChassisRate, DiffDrive, Unicycle, WheelDeltas, integrate};
use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk45;
use multicalc::spatial::{SE2, SO2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-14;

// ---- helpers ----------------------------------------------------------------

fn rand_pose(rng: &mut StdRng) -> SE2<f64> {
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
    let tf = 2.0;
    for (v, w) in [(0.4, 0.0), (0.4, 0.9), (0.4, -0.9), (0.0, 0.9)] {
        let rate = ChassisRate::new(v, w);
        let solved = Rk45::<f64>::default()
            .with_rtol(1e-12)
            .with_atol(1e-14)
            .solve(
                &Unicycle::new(rate).field(),
                0.0,
                &Vector::new([0.0, 0.0, 0.0]),
                tf,
            )
            .unwrap();

        let arc = integrate(SE2::identity(), rate.integrate_over(tf));
        let t = arc.translation();
        assert!(
            (t[0] - solved[0]).abs() < 1e-9,
            "(v={v}, w={w}) x: arc {} vs rk45 {}",
            t[0],
            solved[0]
        );
        assert!(
            (t[1] - solved[1]).abs() < 1e-9,
            "(v={v}, w={w}) y: arc {} vs rk45 {}",
            t[1],
            solved[1]
        );
        assert!(
            (arc.rotation().log() - solved[2]).abs() < 1e-12,
            "(v={v}, w={w}) heading: arc {} vs rk45 {}",
            arc.rotation().log(),
            solved[2]
        );
    }
}

/// The exact-arc claim, stated as a property no Euler step can satisfy: the result does not depend
/// on how finely the constant twist is subdivided.
#[test]
fn one_big_step_equals_many_small_steps() {
    let rate = ChassisRate::new(0.4, 0.9);
    let total = 2.0;
    let n = 1000;

    let one = integrate(SE2::identity(), rate.integrate_over(total));

    let small = rate.integrate_over(total / f64::from(n));
    let mut many = SE2::identity();
    for _ in 0..n {
        many = integrate(many, small);
    }

    let (a, b) = (one.translation(), many.translation());
    assert!((a[0] - b[0]).abs() < TOL, "x: {} vs {}", a[0], b[0]);
    assert!((a[1] - b[1]).abs() < TOL, "y: {} vs {}", a[1], b[1]);
    assert!((one.rotation().log() - many.rotation().log()).abs() < TOL);
}

#[test]
fn arc_matches_closed_form() {
    let (v, w, t) = (0.4_f64, 0.9, 1.3);
    let theta = w * t;
    let radius = v / w;

    let arc = integrate(SE2::identity(), ChassisRate::new(v, w).integrate_over(t));
    let p = arc.translation();
    assert!((p[0] - radius * theta.sin()).abs() < TOL);
    assert!((p[1] - radius * (1.0 - theta.cos())).abs() < TOL);
    assert!((arc.rotation().log() - theta).abs() < TOL);
}

// ---- degenerate motions -----------------------------------------------------

#[test]
fn zero_angular_is_straight_line() {
    let pose = integrate(SE2::identity(), ChassisDelta::new(0.5_f64, 0.0));
    let t = pose.translation();
    assert!(t[0].is_finite() && t[1].is_finite());
    assert_eq!(t[0], 0.5);
    assert_eq!(t[1], 0.0);
    assert_eq!(pose.rotation().log(), 0.0);
}

#[test]
fn zero_linear_is_pure_rotation() {
    let mut rng = StdRng::seed_from_u64(0x0d0_1111);
    let start = rand_pose(&mut rng);
    let pose = integrate(start, ChassisDelta::new(0.0, 0.7));

    let (a, b) = (start.translation(), pose.translation());
    assert_eq!(a[0], b[0]);
    assert_eq!(a[1], b[1]);
}

// ---- conventions ------------------------------------------------------------

#[test]
fn is_right_perturbation() {
    let mut rng = StdRng::seed_from_u64(0x0d0_2222);
    for _ in 0..100 {
        let p = rand_pose(&mut rng);
        let d = ChassisDelta::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        let got = integrate(p, d);
        let want = p * SE2::exp(Vector::new([d.linear(), 0.0, d.angular()]));
        assert_eq!(got, want);
    }
}

#[test]
fn identity_start_equals_exp() {
    let d = ChassisDelta::new(0.3, -0.4);
    let got = integrate(SE2::identity(), d);
    let want = SE2::exp(Vector::new([0.3, 0.0, -0.4]));
    assert_eq!(got, want);
}

#[test]
fn odometry_step_matches_integrate() {
    let dd = DiffDrive::new(0.036_f64, 0.235).unwrap();
    let mut rng = StdRng::seed_from_u64(0x0d0_3333);
    for _ in 0..100 {
        let p = rand_pose(&mut rng);
        let d = WheelDeltas::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        assert_eq!(dd.odometry_step(p, d), integrate(p, dd.forward_delta(d)));
    }
}

// ---- end to end -------------------------------------------------------------

/// Two full circles of opposite curvature, driven through the whole wheel-to-pose chain. The sign
/// change is the point: it exercises both curvature directions and must return to the start.
#[test]
fn figure_eight_closes() {
    let dd = DiffDrive::new(0.036_f64, 0.235).unwrap();
    let (v, w) = (0.36, 0.9);
    let n = 2000;
    let dt = (2.0 * PI / w) / f64::from(n);

    let mut pose = SE2::identity();
    for sign in [1.0, -1.0] {
        let rates = dd.inverse(ChassisRate::new(v, w * sign));
        let d = WheelDeltas::new(rates.left() * dt, rates.right() * dt);
        for _ in 0..n {
            pose = dd.odometry_step(pose, d);
        }
    }

    let t = pose.translation();
    assert!(t[0].abs() < 1e-9, "x did not close: {}", t[0]);
    assert!(t[1].abs() < 1e-9, "y did not close: {}", t[1]);
    assert!(
        pose.rotation().log().abs() < 1e-9,
        "heading did not close: {}",
        pose.rotation().log()
    );
}
