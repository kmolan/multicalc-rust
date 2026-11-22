//! Unicycle field tests: convergence of RK4 on the field to the closed-form arc, and autodiff
//! against finite differences including at exactly zero curvature.

use multicalc::kinematics::{BodyArc, BodyTwist, OdometryStep, Unicycle, integrate};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::ode::Rk4;
use multicalc::scalar::{Dual, Numeric, VectorFn};
use multicalc::spatial::SE2;

// ---- helpers ----------------------------------------------------------------

/// The three outputs of one odometry increment from the identity: `[x, y, θ]`.
fn arc_outputs<T: Numeric>(arc_length: T, heading_change: T) -> [T; 3] {
    let pose = integrate(SE2::identity(), BodyArc::new(arc_length, heading_change));
    let translation = pose.translation();
    [translation[0], translation[1], pose.rotation().log()]
}

fn rk4_to(rate: BodyTwist<f64>, timestep: f64, final_time: f64) -> Vector3D {
    let field = Unicycle::new(rate).field();
    let step_count = (final_time / timestep).round() as usize;
    let mut state = Vector::new([0.0, 0.0, 0.0]);
    let mut time = 0.0;
    for _ in 0..step_count {
        state = Rk4::step(&field, time, &state, timestep);
        time += timestep;
    }
    state
}

// ---- convergence ------------------------------------------------------------

#[test]
fn rk4_of_field_converges_to_integrate() {
    let rate = BodyTwist::new(0.4, 0.9);
    let final_time = 1.0;

    let truth = integrate(SE2::identity(), rate.integrate_over(final_time)).translation();
    let translation_error = |timestep: f64| {
        let state = rk4_to(rate, timestep, final_time);
        ((state[0] - truth[0]).powi(2) + (state[1] - truth[1]).powi(2)).sqrt()
    };

    let coarse = translation_error(0.1);
    let fine = translation_error(0.05);
    let ratio = coarse / fine;
    assert!(
        ratio >= 8.0,
        "expected fourth-order convergence, got ratio {ratio} ({coarse} -> {fine})"
    );
}

// ---- autodiff ---------------------------------------------------------------

#[test]
fn dual_matches_finite_differences() {
    let start = [0.4_f64, 0.3];
    let step = 1e-6;

    for variable_index in 0..2 {
        let arc_length = if variable_index == 0 {
            Dual::variable(start[0])
        } else {
            Dual::constant(start[0])
        };
        let heading_change = if variable_index == 1 {
            Dual::variable(start[1])
        } else {
            Dual::constant(start[1])
        };
        let outputs = arc_outputs(arc_length, heading_change);

        let mut plus = start;
        let mut minus = start;
        plus[variable_index] += step;
        minus[variable_index] -= step;
        let outputs_at_plus = arc_outputs(plus[0], plus[1]);
        let outputs_at_minus = arc_outputs(minus[0], minus[1]);

        for output_index in 0..3 {
            let finite_difference =
                (outputs_at_plus[output_index] - outputs_at_minus[output_index]) / (2.0 * step);
            assert!(
                (outputs[output_index].deriv - finite_difference).abs() < 1e-6,
                "variable {variable_index} output {output_index}: dual {} vs finite difference {finite_difference}",
                outputs[output_index].deriv
            );
        }
    }
}

/// Exactly zero curvature sits inside `SE2::exp`'s Taylor branch — the case a hand-rolled `1/ω` arc
/// produces NaN for, in both the value and the derivative.
#[test]
fn dual_finite_at_exactly_zero_angular() {
    let arc_length = 0.4_f64;
    // A wider step than the usual 1e-6: `(1−cosθ)/θ` and `sinθ/θ` cancel catastrophically as θ→0,
    // and that error grows as h shrinks. At 1e-6 the finite difference is the inaccurate side (~1e-5
    // off); 1e-4 balances it against truncation and lands near 1e-9. θ = ±1e-4 is still outside the
    // Taylor branch, so the comparison remains meaningful.
    let step = 1e-4;

    let outputs = arc_outputs(Dual::constant(arc_length), Dual::variable(0.0));
    for (output_index, output) in outputs.iter().enumerate() {
        assert!(
            output.value.is_finite() && output.deriv.is_finite(),
            "output {output_index}: value {} deriv {} must be finite",
            output.value,
            output.deriv
        );
    }

    // The finite differences straddle the branch, sampling just outside it.
    let outputs_at_plus = arc_outputs(arc_length, step);
    let outputs_at_minus = arc_outputs(arc_length, -step);
    for output_index in 0..3 {
        let finite_difference =
            (outputs_at_plus[output_index] - outputs_at_minus[output_index]) / (2.0 * step);
        assert!(
            (outputs[output_index].deriv - finite_difference).abs() < 1e-6,
            "output {output_index}: dual {} vs finite difference {finite_difference}",
            outputs[output_index].deriv
        );
    }
}

#[test]
fn odometry_step_jacobian_autodiff_vs_fd() {
    // Headings stay well away from ±π, where `SO2::log` wraps and the Jacobian is meaningless.
    for start in [[0.3_f64, -0.2, 0.4, 0.1, 0.05], [0.3, -0.2, 0.4, 0.1, 0.0]] {
        // Wide enough to stay clear of the θ→0 cancellation in the `V(θ)` block; see
        // `dual_finite_at_exactly_zero_angular`.
        let step = 1e-4;
        for variable_index in 0..5 {
            let mut input = [Dual::constant(0.0); 5];
            for (index, slot) in input.iter_mut().enumerate() {
                *slot = if index == variable_index {
                    Dual::variable(start[index])
                } else {
                    Dual::constant(start[index])
                };
            }
            let outputs = OdometryStep.eval(&input);

            let mut plus = start;
            let mut minus = start;
            plus[variable_index] += step;
            minus[variable_index] -= step;
            let outputs_at_plus = OdometryStep.eval(&plus);
            let outputs_at_minus = OdometryStep.eval(&minus);

            for output_index in 0..3 {
                let finite_difference =
                    (outputs_at_plus[output_index] - outputs_at_minus[output_index]) / (2.0 * step);
                assert!(
                    (outputs[output_index].deriv - finite_difference).abs() < 1e-6,
                    "start={start:?} variable {variable_index} output {output_index}: dual {} vs finite difference {finite_difference}",
                    outputs[output_index].deriv
                );
            }
        }
    }
}
