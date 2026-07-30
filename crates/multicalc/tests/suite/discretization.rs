#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Discretization + matrix-exponential invariants and closed-form checks.

use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Matrix4D};
use multicalc::scalar::Dual;
use proptest::prelude::*;

fn matrix3(entries: [[f64; 3]; 3]) -> Matrix3D {
    Matrix::new(entries)
}

#[test]
fn expm_zero_is_identity() {
    let exponential = Matrix4D::<f64>::zeros().expm().unwrap();
    for row in 0..4 {
        for column in 0..4 {
            let expected = if row == column { 1.0 } else { 0.0 };
            assert!((exponential[(row, column)] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn expm_diagonal_is_elementwise_exp() {
    let diagonal = matrix3([[0.5, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]]);
    let exponential = diagonal.expm().unwrap();
    for (index, entry) in [0.5_f64, -1.0, 2.0].into_iter().enumerate() {
        assert!((exponential[(index, index)] - entry.exp()).abs() < 1e-10);
    }
}

#[test]
fn expm_derivative_finite_and_correct() {
    // d/dx expm(x·M)|_{x=0} = M. One Dual through expm; compare to central FD.
    let generator = matrix3([[0.1, 0.4, -0.2], [0.0, -0.3, 0.5], [0.2, 0.1, 0.05]]);
    let autodiff =
        Matrix3D::<Dual<f64>>::from_fn(|row, column| Dual::new(0.0, generator[(row, column)]))
            .expm()
            .unwrap();
    for row in 0..3 {
        for column in 0..3 {
            let cell = autodiff[(row, column)];
            assert!(cell.deriv.is_finite());
            assert!((cell.deriv - generator[(row, column)]).abs() < 1e-9);
        }
    }
}

#[test]
fn zoh_double_integrator() {
    let state_matrix = Matrix2D::new([[0.0, 1.0], [0.0, 0.0]]);
    let input_matrix = Matrix::<2, 1>::new([[0.0], [1.0]]);
    let timestep = 0.1;
    let (discrete_state, discrete_input) =
        zoh::<2, 1, 3, f64>(state_matrix, input_matrix, timestep).unwrap();
    let expected_state = [[1.0, timestep], [0.0, 1.0]];
    for (row, entries) in expected_state.iter().enumerate() {
        for (column, &expected) in entries.iter().enumerate() {
            assert!((discrete_state[(row, column)] - expected).abs() < 1e-9);
        }
    }
    assert!((discrete_input[(0, 0)] - timestep * timestep / 2.0).abs() < 1e-9);
    assert!((discrete_input[(1, 0)] - timestep).abs() < 1e-9);
}

#[test]
fn van_loan_qd_symmetric_and_f_matches_expm() {
    let state_matrix = matrix3([[-0.5, 0.2, 0.0], [0.1, -0.3, 0.4], [0.0, 0.2, -0.6]]);
    let continuous_noise = matrix3([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.1]]);
    let timestep = 0.05;
    let (discrete_state, discrete_noise) =
        van_loan::<3, 6, f64>(state_matrix, continuous_noise, timestep).unwrap();
    let expected_from_expm = state_matrix.scale(timestep).expm().unwrap();
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (discrete_state[(row, column)] - expected_from_expm[(row, column)]).abs() < 1e-9
            );
            assert!((discrete_noise[(row, column)] - discrete_noise[(column, row)]).abs() < 1e-10);
        }
    }
}

#[test]
fn qdwn_matches_closed_form() {
    let timestep = 0.1;
    let variance = 2.0;
    let noise = q_discrete_white_noise::<2, f64>(timestep, variance);
    assert!((noise[(0, 0)] - variance * timestep.powi(4) / 4.0).abs() < 1e-15);
    assert!((noise[(0, 1)] - variance * timestep.powi(3) / 2.0).abs() < 1e-15);
    assert!((noise[(1, 0)] - variance * timestep.powi(3) / 2.0).abs() < 1e-15);
    assert!((noise[(1, 1)] - variance * timestep * timestep).abs() < 1e-15);
}

proptest! {
    #[test]
    fn expm_times_neg_expm_is_identity(
        v in prop::collection::vec(-0.6f64..0.6, 9)
    ) {
        let matrix = Matrix3D::from_fn(|row, column| v[row * 3 + column]);
        let product = matrix.expm().unwrap() * matrix.scale(-1.0).expm().unwrap();
        for row in 0..3 {
            for column in 0..3 {
                let expected = if row == column { 1.0 } else { 0.0 };
                prop_assert!((product[(row, column)] - expected).abs() < 1e-9);
            }
        }
    }
}
