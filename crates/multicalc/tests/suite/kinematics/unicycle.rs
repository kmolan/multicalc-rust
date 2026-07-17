//! Unicycle field tests: convergence of RK4 on the field to the closed-form arc, and autodiff
//! against finite differences including at exactly zero curvature.

use multicalc::kinematics::{ChassisDelta, ChassisRate, OdometryStep, Unicycle, integrate};
use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk4;
use multicalc::scalar::{Dual, Numeric, VectorFn};
use multicalc::spatial::SE2;

// ---- helpers ----------------------------------------------------------------

/// The three outputs of one odometry increment from the identity: `[x, y, θ]`.
fn arc_outputs<T: Numeric>(ds: T, dtheta: T) -> [T; 3] {
    let pose = integrate(SE2::identity(), ChassisDelta::new(ds, dtheta));
    let t = pose.translation();
    [t[0], t[1], pose.rotation().log()]
}

fn rk4_to(rate: ChassisRate<f64>, dt: f64, tf: f64) -> Vector<3, f64> {
    let f = Unicycle::new(rate).field();
    let n = (tf / dt).round() as usize;
    let mut y = Vector::new([0.0, 0.0, 0.0]);
    let mut t = 0.0;
    for _ in 0..n {
        y = Rk4::step(&f, t, &y, dt);
        t += dt;
    }
    y
}

// ---- convergence ------------------------------------------------------------

#[test]
fn rk4_of_field_converges_to_integrate() {
    let rate = ChassisRate::new(0.4, 0.9);
    let tf = 1.0;

    let truth = integrate(SE2::identity(), rate.integrate_over(tf)).translation();
    let err = |dt: f64| {
        let y = rk4_to(rate, dt, tf);
        ((y[0] - truth[0]).powi(2) + (y[1] - truth[1]).powi(2)).sqrt()
    };

    let coarse = err(0.1);
    let fine = err(0.05);
    let ratio = coarse / fine;
    assert!(
        ratio >= 8.0,
        "expected fourth-order convergence, got ratio {ratio} ({coarse} -> {fine})"
    );
}

// ---- autodiff ---------------------------------------------------------------

#[test]
fn dual_matches_finite_differences() {
    let p0 = [0.4_f64, 0.3];
    let h = 1e-6;

    for k in 0..2 {
        let ds = if k == 0 {
            Dual::variable(p0[0])
        } else {
            Dual::constant(p0[0])
        };
        let dtheta = if k == 1 {
            Dual::variable(p0[1])
        } else {
            Dual::constant(p0[1])
        };
        let out = arc_outputs(ds, dtheta);

        let mut plus = p0;
        let mut minus = p0;
        plus[k] += h;
        minus[k] -= h;
        let fp = arc_outputs(plus[0], plus[1]);
        let fm = arc_outputs(minus[0], minus[1]);

        for i in 0..3 {
            let fd = (fp[i] - fm[i]) / (2.0 * h);
            assert!(
                (out[i].deriv - fd).abs() < 1e-6,
                "k={k} i={i}: dual {} vs fd {fd}",
                out[i].deriv
            );
        }
    }
}

/// Exactly zero curvature sits inside `SE2::exp`'s Taylor branch — the case a hand-rolled `1/ω` arc
/// produces NaN for, in both the value and the derivative.
#[test]
fn dual_finite_at_exactly_zero_angular() {
    let ds = 0.4_f64;
    // A wider step than the usual 1e-6: `(1−cosθ)/θ` and `sinθ/θ` cancel catastrophically as θ→0,
    // and that error grows as h shrinks. At 1e-6 the finite difference is the inaccurate side (~1e-5
    // off); 1e-4 balances it against truncation and lands near 1e-9. θ = ±1e-4 is still outside the
    // Taylor branch, so the comparison remains meaningful.
    let h = 1e-4;

    let out = arc_outputs(Dual::constant(ds), Dual::variable(0.0));
    for (i, o) in out.iter().enumerate() {
        assert!(
            o.value.is_finite() && o.deriv.is_finite(),
            "i={i}: value {} deriv {} must be finite",
            o.value,
            o.deriv
        );
    }

    // The finite differences straddle the branch, sampling just outside it.
    let fp = arc_outputs(ds, h);
    let fm = arc_outputs(ds, -h);
    for i in 0..3 {
        let fd = (fp[i] - fm[i]) / (2.0 * h);
        assert!(
            (out[i].deriv - fd).abs() < 1e-6,
            "i={i}: dual {} vs fd {fd}",
            out[i].deriv
        );
    }
}

#[test]
fn odometry_step_jacobian_autodiff_vs_fd() {
    // Headings stay well away from ±π, where `SO2::log` wraps and the Jacobian is meaningless.
    for p0 in [[0.3_f64, -0.2, 0.4, 0.1, 0.05], [0.3, -0.2, 0.4, 0.1, 0.0]] {
        // Wide enough to stay clear of the θ→0 cancellation in the `V(θ)` block; see
        // `dual_finite_at_exactly_zero_angular`.
        let h = 1e-4;
        for k in 0..5 {
            let mut input = [Dual::constant(0.0); 5];
            for (j, slot) in input.iter_mut().enumerate() {
                *slot = if j == k {
                    Dual::variable(p0[j])
                } else {
                    Dual::constant(p0[j])
                };
            }
            let out = OdometryStep.eval(&input);

            let mut plus = p0;
            let mut minus = p0;
            plus[k] += h;
            minus[k] -= h;
            let fp = OdometryStep.eval(&plus);
            let fm = OdometryStep.eval(&minus);

            for i in 0..3 {
                let fd = (fp[i] - fm[i]) / (2.0 * h);
                assert!(
                    (out[i].deriv - fd).abs() < 1e-6,
                    "p0={p0:?} k={k} i={i}: dual {} vs fd {fd}",
                    out[i].deriv
                );
            }
        }
    }
}
