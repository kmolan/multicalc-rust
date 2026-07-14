#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Discretization + matrix-exponential invariants and closed-form checks.

use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use multicalc::linear_algebra::Matrix;
use multicalc::scalar::Dual;
use proptest::prelude::*;

fn mat3(a: [[f64; 3]; 3]) -> Matrix<3, 3> {
    Matrix::new(a)
}

#[test]
fn expm_zero_is_identity() {
    let e = Matrix::<4, 4>::zeros().expm().unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((e[(i, j)] - want).abs() < 1e-12);
        }
    }
}

#[test]
fn expm_diagonal_is_elementwise_exp() {
    let d = mat3([[0.5, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]]);
    let e = d.expm().unwrap();
    for (i, x) in [0.5_f64, -1.0, 2.0].into_iter().enumerate() {
        assert!((e[(i, i)] - x.exp()).abs() < 1e-10);
    }
}

#[test]
fn expm_derivative_finite_and_correct() {
    // d/dx expm(x·M)|_{x=0} = M. One Dual through expm; compare to central FD.
    let m = mat3([[0.1, 0.4, -0.2], [0.0, -0.3, 0.5], [0.2, 0.1, 0.05]]);
    let ad = Matrix::<3, 3, Dual<f64>>::from_fn(|i, j| Dual::new(0.0, m[(i, j)]))
        .expm()
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(ad[(i, j)].deriv.is_finite());
            assert!((ad[(i, j)].deriv - m[(i, j)]).abs() < 1e-9);
        }
    }
}

#[test]
fn zoh_double_integrator() {
    let a = Matrix::<2, 2>::new([[0.0, 1.0], [0.0, 0.0]]);
    let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
    let dt = 0.1;
    let (f, g) = zoh::<2, 1, 3, f64>(a, b, dt).unwrap();
    let want_f = [[1.0, dt], [0.0, 1.0]];
    for i in 0..2 {
        for j in 0..2 {
            assert!((f[(i, j)] - want_f[i][j]).abs() < 1e-9);
        }
    }
    assert!((g[(0, 0)] - dt * dt / 2.0).abs() < 1e-9);
    assert!((g[(1, 0)] - dt).abs() < 1e-9);
}

#[test]
fn van_loan_qd_symmetric_and_f_matches_expm() {
    let a = mat3([[-0.5, 0.2, 0.0], [0.1, -0.3, 0.4], [0.0, 0.2, -0.6]]);
    let qc = mat3([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.1]]);
    let dt = 0.05;
    let (f, qd) = van_loan::<3, 6, f64>(a, qc, dt).unwrap();
    let f_ref = a.scale(dt).expm().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((f[(i, j)] - f_ref[(i, j)]).abs() < 1e-9);
            assert!((qd[(i, j)] - qd[(j, i)]).abs() < 1e-10);
        }
    }
}

#[test]
fn qdwn_matches_closed_form() {
    let (dt, var) = (0.1, 2.0);
    let q = q_discrete_white_noise::<2, f64>(dt, var);
    assert!((q[(0, 0)] - var * dt.powi(4) / 4.0).abs() < 1e-15);
    assert!((q[(0, 1)] - var * dt.powi(3) / 2.0).abs() < 1e-15);
    assert!((q[(1, 0)] - var * dt.powi(3) / 2.0).abs() < 1e-15);
    assert!((q[(1, 1)] - var * dt * dt).abs() < 1e-15);
}

proptest! {
    #[test]
    fn expm_times_neg_expm_is_identity(
        v in prop::collection::vec(-0.6f64..0.6, 9)
    ) {
        let a = Matrix::<3, 3>::from_fn(|i, j| v[i * 3 + j]);
        let prod = a.expm().unwrap() * a.scale(-1.0).expm().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                prop_assert!((prod[(i, j)] - want).abs() < 1e-9);
            }
        }
    }
}
