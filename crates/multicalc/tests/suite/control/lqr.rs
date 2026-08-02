//! LQR tests: the closed-form gain on a scalar system, the loop settling, the tracking call, the
//! stability certificate on a good and a bad loop, and constructor rejection.

use multicalc::control::Lqr;
use multicalc::error::{ControlError, LinalgError};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::Numeric;

/// A cart carrying its speed forward, pushed by one input, at a 0.1 s timestep.
fn double_integrator<T: Numeric>() -> (
    Matrix<2, 2, T>,
    Matrix<2, 1, T>,
    Matrix<2, 2, T>,
    Matrix<1, 1, T>,
) {
    let dt = T::from_f64(0.1);
    let state_transition = Matrix::new([[T::ONE, dt], [T::ZERO, T::ONE]]);
    let input_model = Matrix::new([[T::from_f64(0.005)], [dt]]);
    (
        state_transition,
        input_model,
        Matrix::identity(),
        Matrix::identity(),
    )
}

#[test]
fn scalar_gain_matches_the_closed_form() {
    let one = Matrix::<1, 1>::new([[1.0]]);
    let controller = Lqr::new(one, one, one, one).unwrap();
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    assert!((controller.cost_to_go()[(0, 0)] - golden_ratio).abs() < 1e-10);
    assert!((controller.gain()[(0, 0)] - golden_ratio / (1.0 + golden_ratio)).abs() < 1e-10);
}

#[test]
fn closed_loop_brings_the_state_home() {
    let (state_transition, input_model, state_cost, input_cost) = double_integrator::<f64>();
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    let mut state = Vector::new([1.0, 0.0]);
    for _ in 0..400 {
        state = state_transition * state + input_model * controller.control(state);
    }
    assert!(state.norm() < 1e-6, "state {state:?}");
}

#[test]
fn tracking_holds_the_reference() {
    let (state_transition, input_model, state_cost, input_cost) = double_integrator::<f64>();
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    let reference = Vector::new([2.0, 0.0]);
    let feedforward = Vector::new([0.0]);
    let mut state = Vector::new([0.0, 0.0]);
    for _ in 0..600 {
        let input = controller.control_tracking(state, reference, feedforward);
        state = state_transition * state + input_model * input;
    }
    assert!((state - reference).norm() < 1e-6, "state {state:?}");
}

#[test]
fn tracking_with_zero_error_returns_the_feedforward() {
    let (state_transition, input_model, state_cost, input_cost) = double_integrator::<f64>();
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    let reference = Vector::new([2.0, 0.0]);
    let held = controller.control_tracking(reference, reference, Vector::new([0.7]));
    assert!((held[0] - 0.7).abs() < 1e-15);
}

#[test]
fn certificate_is_positive_definite() {
    let (state_transition, input_model, state_cost, input_cost) = double_integrator::<f64>();
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    let certificate = controller.certify_stability().unwrap();
    assert!(certificate.cholesky().is_ok());

    let closed = controller.closed_loop();
    let residual = closed.transpose() * certificate * closed - certificate + state_cost;
    assert!(residual.frobenius_norm() < 1e-10);
}

#[test]
fn rejects_a_runaway_direction_the_input_cannot_reach() {
    // The second state runs away on its own and the input cannot touch it.
    let state_transition = Matrix::<2, 2>::new([[0.5, 0.0], [0.0, 1.5]]);
    let input_model = Matrix::<2, 1>::new([[1.0], [0.0]]);
    let state_cost = Matrix::<2, 2>::identity();
    let input_cost = Matrix::<1, 1>::new([[1.0]]);
    assert!(matches!(
        Lqr::new(state_transition, input_model, state_cost, input_cost),
        Err(ControlError::Linalg(LinalgError::DidNotConverge { .. }))
    ));
}

#[test]
fn rejects_a_non_positive_input_cost() {
    let (state_transition, input_model, state_cost, _) = double_integrator::<f64>();
    let input_cost = Matrix::<1, 1>::new([[0.0]]);
    assert_eq!(
        Lqr::new(state_transition, input_model, state_cost, input_cost).err(),
        Some(ControlError::Linalg(LinalgError::NotPositiveDefinite))
    );
}

#[test]
fn rejects_a_lopsided_state_cost() {
    let (state_transition, input_model, _, input_cost) = double_integrator::<f64>();
    let state_cost = Matrix::<2, 2>::new([[1.0, 0.5], [-0.5, 1.0]]);
    assert_eq!(
        Lqr::new(state_transition, input_model, state_cost, input_cost).err(),
        Some(ControlError::Linalg(LinalgError::NotSymmetric))
    );
}

#[test]
fn rejects_non_finite() {
    let (state_transition, _, state_cost, input_cost) = double_integrator::<f64>();
    let input_model = Matrix::<2, 1>::new([[f64::NAN], [0.1]]);
    assert_eq!(
        Lqr::new(state_transition, input_model, state_cost, input_cost).err(),
        Some(ControlError::Linalg(LinalgError::NonFinite))
    );
}

#[test]
fn works_at_f32() {
    let (state_transition, input_model, state_cost, input_cost) = double_integrator::<f32>();
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    let mut state = Vector::new([1.0_f32, 0.0]);
    for _ in 0..400 {
        state = state_transition * state + input_model * controller.control(state);
    }
    assert!(state.norm() < 1e-2_f32, "state {state:?}");
    assert!(controller.certify_stability().is_ok());
}

#[test]
fn four_state_system_settles() {
    use multicalc::zoh;
    // A cart of 1.0 kg carrying a pole of 0.1 kg whose balance point is 0.5 m up, under gravity of
    // 9.81 m/s². The state is cart position, cart speed, pole tilt, and how fast the tilt changes.
    let continuous_state = Matrix::<4, 4>::new([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.981, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 21.582, 0.0],
    ]);
    let continuous_input = Matrix::<4, 1>::new([[0.0], [1.0], [0.0], [-2.0]]);
    let (state_transition, input_model) =
        zoh::<4, 1, 5, f64>(continuous_state, continuous_input, 0.02).unwrap();

    let state_cost = Matrix::from_diagonal([10.0, 1.0, 10.0, 1.0]);
    let input_cost = Matrix::<1, 1>::new([[0.1]]);
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    assert!(controller.certify_stability().is_ok());

    // A small starting tilt, which the loop has to catch before the pole falls over.
    let mut state = Vector::new([0.0, 0.0, 0.05, 0.0]);
    for _ in 0..2000 {
        state = state_transition * state + input_model * controller.control(state);
    }
    assert!(state.norm() < 1e-4, "state {state:?}");
}
