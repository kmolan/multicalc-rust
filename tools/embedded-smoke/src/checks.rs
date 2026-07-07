//! Tiny on-target math checks. Each asserts a known answer to a tolerance.

use multicalc::LevenbergMarquardt;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::scalar_fn;

/// Fit y = a * e^(b t) and check the recovered parameters.
pub fn lm_fit() {
    struct Fit;
    impl VectorFn<2, 8> for Fit {
        fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; 8] {
            let (a, b) = (p[0], p[1]);
            core::array::from_fn(|i| {
                let t = i as f64;
                a * (b * S::from_f64(t)).exp() - S::from_f64(100.0 * (-0.5 * t).exp())
            })
        }
    }
    let solver = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(50);
    let report = solver.minimize(&Fit, &[80.0, -0.3]).expect("fit converges");
    assert!((report.solution[0] - 100.0).abs() < 1e-6);
    assert!((report.solution[1] + 0.5).abs() < 1e-6);
}

/// Differentiate x^3 at x = 2 by autodiff. Exact derivative is 12.
pub fn autodiff_derivative() {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value = d.get(1, &f, 2.0_f64).expect("derivative");
    assert!((value - 12.0).abs() < 1e-12);
}

/// A check that only uses the portable path (no atomics), for the Cortex-M0 target.
pub fn portable_path() {
    let v = [1.0_f64, 2.0, 3.0, 4.0];
    let sum: f64 = v.iter().copied().fold(0.0, |a, b| a + b);
    assert!((sum - 10.0).abs() < 1e-12);
}

/// SVD: recover a known rotation from a 3×3 cross-covariance matrix (Kabsch method).
pub fn svd_kabsch() {
    let rot = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    let pts = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, -1.0, 0.5],
        [-1.0, 0.5, 2.0],
    ];
    let mut h = Matrix::<3, 3>::zeros();
    for p in pts {
        let pv = Vector::new(p);
        let q = rot * pv;
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += q[i] * pv[j];
            }
        }
    }
    let f = h.svd().expect("svd");
    let (u, v) = (f.u(), f.v());
    let mut rhat = u * v.transpose();
    if rhat.determinant() < 0.0 {
        let mut uf = u;
        for i in 0..3 {
            uf[(i, 2)] = -uf[(i, 2)];
        }
        rhat = uf * v.transpose();
    }
    // Recovered rotation must match the original to 1e-10.
    for r in 0..3 {
        for c in 0..3 {
            assert!((rhat[(r, c)] - rot[(r, c)]).abs() < 1e-10);
        }
    }
    // RᵀR must equal the identity.
    let rtr = rhat.transpose() * rhat;
    let eye = Matrix::<3, 3>::identity();
    for r in 0..3 {
        for c in 0..3 {
            assert!((rtr[(r, c)] - eye[(r, c)]).abs() < 1e-10);
        }
    }
}
