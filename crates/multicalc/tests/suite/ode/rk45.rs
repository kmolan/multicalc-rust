use multicalc::error::IntegrateError;
use multicalc::linear_algebra::Vector;
use multicalc::ode::{Rk45, Step};
use proptest::prelude::*;

// No AD-through-RK45 test: adaptive step control is not cleanly differentiable, since
// the step sequence depends on the primal error norm. The autodiff guarantee is carried
// by the RK4 test `ad_through_rk4_matches_fd`.

#[test]
fn solve_matches_closed_form() {
    // y' = -y over [0, 2]: y(2) = e^{-2}.
    let decay = |_t: f64, y: &Vector<1, f64>| -*y;
    let yf = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&decay, 0.0, &Vector::new([1.0]), 2.0)
        .unwrap();
    assert!((yf[0] - (-2.0_f64).exp()).abs() < 1e-7);

    // Harmonic oscillator over one period returns to [1, 0].
    let harmonic = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
    let tf = core::f64::consts::TAU;
    let yf = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&harmonic, 0.0, &Vector::new([1.0, 0.0]), tf)
        .unwrap();
    assert!((yf[0] - 1.0).abs() < 1e-7 && yf[1].abs() < 1e-7);

    // Two-body unit circular orbit returns to its start after one period.
    let two_body = |_t: f64, y: &Vector<4, f64>| {
        let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let r3 = r * r * r;
        Vector::new([y[2], y[3], -y[0] / r3, -y[1] / r3])
    };
    let y0 = Vector::new([1.0, 0.0, 0.0, 1.0]);
    let yf = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&two_body, 0.0, &y0, tf)
        .unwrap();
    for (got, want) in yf.as_array().iter().zip(y0.as_array()) {
        assert!((got - want).abs() < 1e-7);
    }
}

#[test]
fn dense_output_endpoints_exact() {
    // Capture the first accepted step and check its cubic-Hermite interpolation.
    let decay = |_t: f64, y: &Vector<1, f64>| -*y;
    let y0 = Vector::new([1.0]);
    let mut first: Option<Step<1, f64>> = None;
    let _ = Rk45::default()
        .with_rtol(1e-8)
        .with_atol(1e-10)
        .for_each_step(&decay, 0.0, &y0, 1.0, |step| {
            if first.is_none() {
                first = Some(*step);
            }
        })
        .unwrap();
    let step = first.unwrap();

    // The endpoints are reproduced exactly.
    assert_eq!(step.interpolate(step.t0).as_array(), step.y0.as_array());
    assert_eq!(step.interpolate(step.t1).as_array(), step.y1.as_array());

    // An interior sample matches a separate solve to that time.
    let tm = 0.5 * (step.t0 + step.t1);
    let ym = step.interpolate(tm);
    let solved = Rk45::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve(&decay, 0.0, &y0, tm)
        .unwrap();
    assert!((ym[0] - solved[0]).abs() < 1e-8);
}

#[test]
fn min_step_floor_errors() {
    // A floor larger than any feasible step forces StepSizeTooSmall.
    let decay = |_t: f64, y: &Vector<1, f64>| -*y;
    let res = Rk45::default()
        .with_min_step(1e9)
        .solve(&decay, 0.0, &Vector::new([1.0]), 1.0);
    assert_eq!(res.unwrap_err(), IntegrateError::StepSizeTooSmall);
}

#[test]
fn max_steps_budget_errors() {
    // One step cannot cross a long span, so the budget is exhausted.
    let decay = |_t: f64, y: &Vector<1, f64>| -*y;
    let res = Rk45::default()
        .with_max_steps(1)
        .solve(&decay, 0.0, &Vector::new([1.0]), 100.0);
    assert!(matches!(
        res.unwrap_err(),
        IntegrateError::DidNotConverge { .. }
    ));
}

#[test]
fn zero_span_errors() {
    let decay = |_t: f64, y: &Vector<1, f64>| -*y;
    let res = Rk45::default().solve(&decay, 1.0, &Vector::new([1.0]), 1.0);
    assert_eq!(res.unwrap_err(), IntegrateError::LimitsIllDefined);
}

#[test]
fn non_finite_rhs_errors() {
    let f = |_t: f64, _y: &Vector<1, f64>| Vector::new([f64::NAN]);
    let res = Rk45::default().solve(&f, 0.0, &Vector::new([1.0]), 1.0);
    assert_eq!(res.unwrap_err(), IntegrateError::NonFinite);
}

#[test]
fn grid_length_mismatch_errors() {
    let decay = |_t: f64, y: &Vector<1, f64>| -*y;
    let times = [0.5, 1.0];
    let mut out = [Vector::<1, f64>::zeros(); 1];
    let res = Rk45::default().solve_on_grid(&decay, 0.0, &Vector::new([1.0]), &times, &mut out);
    assert_eq!(res.unwrap_err(), IntegrateError::LimitsIllDefined);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn exponential_matches_closed_form(
        lambda in -3.0f64..-0.1,
        a in 0.5f64..2.0,
        tf in 0.5f64..3.0,
    ) {
        // y' = lambda*y, y(0) = a  ->  y(tf) = a * e^{lambda*tf}.
        let f = |_t: f64, y: &Vector<1, f64>| y.scale(lambda);
        let yf = Rk45::default()
            .with_rtol(1e-9)
            .with_atol(1e-12)
            .solve(&f, 0.0, &Vector::new([a]), tf)
            .unwrap();
        let exact = a * (lambda * tf).exp();
        prop_assert!((yf[0] - exact).abs() < 1e-6 * (1.0 + exact.abs()));
    }
}
