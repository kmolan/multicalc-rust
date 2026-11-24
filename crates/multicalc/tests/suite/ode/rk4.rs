use multicalc::Dual;
use multicalc::error::IntegrateError;
use multicalc::linear_algebra::{Vector, Vector2D};
use multicalc::ode::Rk4;

// y' = -y, generic over the scalar so the same RHS runs at f64 and through Dual.
fn decay<T: multicalc::Numeric>(_time: T, state: &Vector<1, T>) -> Vector<1, T> {
    -*state
}

#[test]
fn exp_decay_matches_closed_form() {
    // y(1) = e^{-1} from 1000 fixed steps.
    let steps = 1000;
    let initial_state = Vector::new([1.0]);
    let timestep = 1.0 / steps as f64;
    let final_state = Rk4::integrate(
        &decay::<f64>,
        0.0,
        &initial_state,
        timestep,
        steps,
        |_, _| {},
    );
    assert!((final_state[0] - (-1.0_f64).exp()).abs() < 1e-9);
}

#[test]
fn harmonic_matches_closed_form() {
    // y'' = -y as [y2, -y1]; over one period the state returns to [cos, -sin].
    let harmonic = |_time: f64, state: &Vector2D| Vector::new([state[1], -state[0]]);
    let final_time = core::f64::consts::TAU;
    let steps = 2000;
    let initial_state = Vector::new([1.0, 0.0]);
    let timestep = final_time / steps as f64;
    let final_state = Rk4::integrate(&harmonic, 0.0, &initial_state, timestep, steps, |_, _| {});
    assert!((final_state[0] - final_time.cos()).abs() < 1e-7);
    assert!((final_state[1] - (-final_time.sin())).abs() < 1e-7);
}

#[test]
fn fourth_order_convergence() {
    // Halving the step should cut the global endpoint error by ~2^4 = 16.
    let exact = (-1.0_f64).exp();
    let endpoint_error = |steps: usize| {
        let initial_state = Vector::new([1.0]);
        let timestep = 1.0 / steps as f64;
        let final_state = Rk4::integrate(
            &decay::<f64>,
            0.0,
            &initial_state,
            timestep,
            steps,
            |_, _| {},
        );
        (final_state[0] - exact).abs()
    };
    let ratio = endpoint_error(50) / endpoint_error(100);
    assert!((12.0..=20.0).contains(&ratio), "convergence ratio {ratio}");
}

#[test]
fn ad_through_rk4_matches_fd() {
    // Differentiate the final state w.r.t. the initial condition. For y' = -y the
    // exact sensitivity is d y_f / d a = e^{-t_f}; check it against a central FD.
    let initial_value = 1.3_f64;
    let final_time = 0.7;
    let steps = 100;

    let initial_state = Vector::new([Dual::variable(initial_value)]);
    let timestep = Dual::constant(final_time / steps as f64);
    let final_state = Rk4::integrate(
        &decay::<Dual<f64>>,
        Dual::constant(0.0),
        &initial_state,
        timestep,
        steps,
        |_, _| {},
    );
    let autodiff = final_state[0].deriv;
    assert!((autodiff - (-final_time).exp()).abs() < 1e-6);

    let final_state_from = |initial_value: f64| {
        let initial_state = Vector::new([initial_value]);
        let timestep = final_time / steps as f64;
        Rk4::integrate(
            &decay::<f64>,
            0.0,
            &initial_state,
            timestep,
            steps,
            |_, _| {},
        )[0]
    };
    let step = 1e-6;
    let finite_difference = (final_state_from(initial_value + step)
        - final_state_from(initial_value - step))
        / (2.0 * step);
    assert!((autodiff - finite_difference).abs() < 1e-6);
}

#[test]
fn f32_energy_round_trip() {
    // Harmonic oscillator at f32 over one period conserves y0^2 + y1^2 = 1.
    let harmonic = |_time: f32, state: &Vector2D<f32>| Vector::new([state[1], -state[0]]);
    let final_time = core::f32::consts::TAU;
    let steps = 2000;
    let initial_state = Vector::new([1.0, 0.0]);
    let timestep = final_time / steps as f32;
    let final_state = Rk4::integrate(
        &harmonic,
        0.0_f32,
        &initial_state,
        timestep,
        steps,
        |_, _| {},
    );
    let energy = final_state[0] * final_state[0] + final_state[1] * final_state[1];
    assert!((energy - 1.0).abs() < 1e-3);
}

fn two_state_decay(_time: f64, state: &Vector<2, f64>) -> Vector<2, f64> {
    -*state
}

fn square(_time: f64, state: &Vector<1, f64>) -> Vector<1, f64> {
    Vector::new([state[0] * state[0]])
}

#[test]
fn try_step_matches_step() {
    let start = Vector::new([1.0, 2.0]);
    assert_eq!(
        Rk4::try_step(&two_state_decay, 0.0, &start, 0.1),
        Ok(Rk4::step(&two_state_decay, 0.0, &start, 0.1))
    );
}

#[test]
fn try_step_rejects_a_non_finite_state() {
    let start = Vector::new([f64::NAN, 0.0]);
    assert_eq!(
        Rk4::try_step(&two_state_decay, 0.0, &start, 0.1),
        Err(IntegrateError::NonFinite)
    );
}

#[test]
fn try_integrate_matches_integrate() {
    let start = Vector::new([1.0, 2.0]);
    let unchecked = Rk4::integrate(&two_state_decay, 0.0, &start, 0.01, 100, |_, _| {});
    let checked = Rk4::try_integrate(&two_state_decay, 0.0, &start, 0.01, 100, |_, _| {});
    assert_eq!(checked, Ok(unchecked));
}

#[test]
fn try_integrate_stops_at_a_blow_up() {
    // x' = x², x(0) = 1 blows up at t = 1. Step 103 is the first non-finite state.
    let mut observed = 0;
    let outcome = Rk4::try_integrate(&square, 0.0, &Vector::new([1.0]), 0.01, 200, |_, state| {
        assert!(state.is_finite());
        observed += 1;
    });
    assert_eq!(outcome, Err(IntegrateError::NonFinite));
    assert_eq!(observed, 103);
}

#[test]
fn try_integrate_rejects_a_non_finite_start() {
    let mut observed = 0;
    let outcome = Rk4::try_integrate(
        &square,
        0.0,
        &Vector::new([f64::INFINITY]),
        0.01,
        10,
        |_, _| {
            observed += 1;
        },
    );
    assert_eq!(outcome, Err(IntegrateError::NonFinite));
    assert_eq!(observed, 0);
}
