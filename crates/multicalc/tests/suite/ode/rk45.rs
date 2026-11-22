use multicalc::error::IntegrateError;
use multicalc::linear_algebra::{Vector, Vector2D};
use multicalc::ode::{Rk45, Step};
use proptest::prelude::*;

// No AD-through-RK45 test: adaptive step control is not cleanly differentiable, since
// the step sequence depends on the primal error norm. The autodiff guarantee is carried
// by the RK4 test `ad_through_rk4_matches_fd`.

#[test]
fn solve_matches_closed_form() {
    // y' = -y over [0, 2]: y(2) = e^{-2}.
    let decay = |_time: f64, state: &Vector<1, f64>| -*state;
    let final_state = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&decay, 0.0, &Vector::new([1.0]), 2.0)
        .unwrap();
    assert!((final_state[0] - (-2.0_f64).exp()).abs() < 1e-7);

    // Harmonic oscillator over one period returns to [1, 0].
    let harmonic = |_time: f64, state: &Vector2D| Vector::new([state[1], -state[0]]);
    let final_time = core::f64::consts::TAU;
    let final_state = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&harmonic, 0.0, &Vector::new([1.0, 0.0]), final_time)
        .unwrap();
    assert!((final_state[0] - 1.0).abs() < 1e-7 && final_state[1].abs() < 1e-7);

    // Two-body unit circular orbit returns to its start after one period.
    let two_body = |_time: f64, state: &Vector<4, f64>| {
        let radius = (state[0] * state[0] + state[1] * state[1]).sqrt();
        let radius_cubed = radius * radius * radius;
        Vector::new([
            state[2],
            state[3],
            -state[0] / radius_cubed,
            -state[1] / radius_cubed,
        ])
    };
    let initial_state = Vector::new([1.0, 0.0, 0.0, 1.0]);
    let final_state = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&two_body, 0.0, &initial_state, final_time)
        .unwrap();
    for (got, want) in final_state.as_array().iter().zip(initial_state.as_array()) {
        assert!((got - want).abs() < 1e-7);
    }
}

#[test]
fn dense_output_endpoints_exact() {
    // Capture the first accepted step and check its cubic-Hermite interpolation.
    let decay = |_time: f64, state: &Vector<1, f64>| -*state;
    let initial_state = Vector::new([1.0]);
    let mut first: Option<Step<1, f64>> = None;
    let _ = Rk45::default()
        .with_rtol(1e-8)
        .with_atol(1e-10)
        .for_each_step(&decay, 0.0, &initial_state, 1.0, |step| {
            if first.is_none() {
                first = Some(*step);
            }
        })
        .unwrap();
    let step = first.unwrap();

    // The endpoints are reproduced exactly.
    assert_eq!(
        step.interpolate(step.time_start).as_array(),
        step.state_start.as_array()
    );
    assert_eq!(
        step.interpolate(step.time_end).as_array(),
        step.state_end.as_array()
    );

    // An interior sample matches a separate solve to that time.
    let midpoint_time = 0.5 * (step.time_start + step.time_end);
    let interpolated = step.interpolate(midpoint_time);
    let solved = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&decay, 0.0, &initial_state, midpoint_time)
        .unwrap();
    assert!((interpolated[0] - solved[0]).abs() < 1e-8);
}

#[test]
fn min_step_floor_errors() {
    // A floor larger than any feasible step forces StepSizeTooSmall.
    let decay = |_time: f64, state: &Vector<1, f64>| -*state;
    let result = Rk45::default()
        .with_min_step(1e9)
        .solve(&decay, 0.0, &Vector::new([1.0]), 1.0);
    assert_eq!(result.unwrap_err(), IntegrateError::StepSizeTooSmall);
}

#[test]
fn max_steps_budget_errors() {
    // One step cannot cross a long span, so the budget is exhausted.
    let decay = |_time: f64, state: &Vector<1, f64>| -*state;
    let result = Rk45::default()
        .with_max_steps(1)
        .solve(&decay, 0.0, &Vector::new([1.0]), 100.0);
    assert!(matches!(
        result.unwrap_err(),
        IntegrateError::DidNotConverge { .. }
    ));
}

#[test]
fn zero_span_errors() {
    let decay = |_time: f64, state: &Vector<1, f64>| -*state;
    let result = Rk45::default().solve(&decay, 1.0, &Vector::new([1.0]), 1.0);
    assert_eq!(result.unwrap_err(), IntegrateError::LimitsIllDefined);
}

#[test]
fn zero_dimensional_state_remains_empty() {
    let empty = Vector::<0, f64>::zeros();
    let zero = |_time: f64, _state: &Vector<0, f64>| Vector::zeros();

    let result = Rk45::default().solve(&zero, 0.0, &empty, 1.0);

    assert_eq!(result.unwrap(), empty);
}

#[test]
fn non_finite_rhs_errors() {
    let non_finite = |_time: f64, _state: &Vector<1, f64>| Vector::new([f64::NAN]);
    let result = Rk45::default().solve(&non_finite, 0.0, &Vector::new([1.0]), 1.0);
    assert_eq!(result.unwrap_err(), IntegrateError::NonFinite);
}

#[test]
fn grid_length_mismatch_errors() {
    let decay = |_time: f64, state: &Vector<1, f64>| -*state;
    let times = [0.5, 1.0];
    let mut outputs = [Vector::<1, f64>::zeros(); 1];
    let result =
        Rk45::default().solve_on_grid(&decay, 0.0, &Vector::new([1.0]), &times, &mut outputs);
    assert_eq!(result.unwrap_err(), IntegrateError::LimitsIllDefined);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn exponential_matches_closed_form(
        lambda in -3.0f64..-0.1,
        a in 0.5f64..2.0,
        time_final in 0.5f64..3.0,
    ) {
        // y' = lambda*y, y(0) = a  ->  y(tf) = a * e^{lambda*tf}.
        let decay = |_time: f64, state: &Vector<1, f64>| state.scale(lambda);
        let final_state = Rk45::default()
            .with_rtol(1e-9)
            .with_atol(1e-12)
            .solve(&decay, 0.0, &Vector::new([a]), time_final)
            .unwrap();
        let exact = a * (lambda * time_final).exp();
        prop_assert!((final_state[0] - exact).abs() < 1e-6 * (1.0 + exact.abs()));
    }
}
