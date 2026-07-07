//! Tiny on-target math checks. Each asserts a known answer to a tolerance.
//!
//! Golden checks assert against values taken from the host oracle fixtures (see
//! `fixtures.rs`), so the target and the host share one source of truth. Identity
//! checks assert a mathematical identity that needs no fixture. Every assertion is
//! a hard failure: a wrong answer panics, which the runner turns into a non-zero
//! QEMU exit.

use multicalc::LevenbergMarquardt;
use multicalc::linear_algebra::Matrix;
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::scalar_fn;
use multicalc::utils::error_codes::CalcError;

use crate::fixtures;

/// Combined absolute/relative closeness, matching `close` in
/// tools/oracle/src/load.rs: `|got - want| <= abs + rel * max(|got|, |want|)`.
fn close(got: f64, want: f64, abs: f64, rel: f64) -> bool {
    (got - want).abs() <= abs + rel * got.abs().max(want.abs())
}

/// Golden: the Rosenbrock least-squares minimizer must match the host oracle
/// golden (optimization/rosenbrock). Residuals `[10 (y - x^2), 1 - x]` are zero
/// at the minimum `(1, 1)`. Part of the full set only (thumbv7em).
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lm_fit() {
    struct Rosenbrock;
    impl VectorFn<2, 2> for Rosenbrock {
        fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; 2] {
            let (x, y) = (p[0], p[1]);
            [S::from_f64(10.0) * (y - x * x), S::from_f64(1.0) - x]
        }
    }
    let solver = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(100);
    let report = solver
        .minimize(&Rosenbrock, &fixtures::ROSENBROCK_X0)
        .expect("fit converges");
    for i in 0..2 {
        assert!(close(
            report.solution[i],
            fixtures::ROSENBROCK_SOLUTION[i],
            fixtures::ROSENBROCK_ABS,
            fixtures::ROSENBROCK_REL,
        ));
    }
}

/// Identity: differentiate x^3 at x = 2 by autodiff. Exact derivative is 12.
/// Part of the full set only (thumbv7em).
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn autodiff_derivative() {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value = d.get(1, &f, 2.0_f64).expect("derivative");
    assert!((value - 12.0).abs() < 1e-12);
}

/// Identity: a sum over the portable (no-atomics) path, for the Cortex-M0 target.
/// Σ 1..=4 = 10.
pub fn portable_path() {
    let v = [1.0_f64, 2.0, 3.0, 4.0];
    let sum: f64 = v.iter().copied().fold(0.0, |a, b| a + b);
    assert!((sum - 10.0).abs() < 1e-12);
}

/// No-panic negative path: a fallible decomposition returns a typed `Err` on bad
/// input instead of crashing. A singular (all-zero) matrix has no LU
/// factorization; an indefinite matrix has no Cholesky factorization.
pub fn error_path_returns_err() {
    assert!(matches!(
        Matrix::<3, 3>::zeros().lu(),
        Err(CalcError::SingularMatrix)
    ));
    let indefinite = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    assert!(matches!(
        indefinite.cholesky(),
        Err(CalcError::NotPositiveDefinite)
    ));
}

/// Golden: singular values of a fixture matrix must match the host oracle golden
/// (linalg/svd_3x3). Returns the values so the caller can emit them for the
/// cross-ABI divergence guard.
pub fn svd_golden() -> [f64; 3] {
    let a: Matrix<3, 3> = Matrix::new(fixtures::SVD_3X3_INPUT);
    let sv = a.svd().expect("svd").singular_values();
    for i in 0..3 {
        assert!(close(
            sv[i],
            fixtures::SVD_3X3_SINGULAR_VALUES[i],
            fixtures::SVD_3X3_ABS,
            fixtures::SVD_3X3_REL,
        ));
    }
    [sv[0], sv[1], sv[2]]
}
