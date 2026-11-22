//! Closeness helpers in two contracts: an abs+rel `Tol` bound for scalars and
//! vectors, and an absolute per-entry bound for the linear-algebra structural
//! checkers. `Numeric` is in scope so `.abs()`/`.max()` resolve to `libm` and
//! compile on bare metal.

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::Numeric;

/// Absolute and relative thresholds for one comparison.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tol {
    pub abs: f64,
    pub rel: f64,
}

/// True when `got` is within `tol` of `want`, using a combined absolute and
/// relative bound: `|got - want| <= abs + rel * max(|got|, |want|)`.
#[must_use]
pub fn close(got: f64, want: f64, tol: Tol) -> bool {
    (got - want).abs() <= tol.abs + tol.rel * got.abs().max(want.abs())
}

/// Asserts a scalar matches the expected value within `tol`.
pub fn assert_scalar_close(got: f64, want: f64, tol: Tol) {
    assert!(close(got, want, tol), "got {got}, want {want}, tol {tol:?}");
}

/// Asserts every component of a vector matches within `tol`.
pub fn assert_vector_close<const N: usize>(got: &Vector<N>, want: &Vector<N>, tol: Tol) {
    for i in 0..N {
        assert!(
            close(got[i], want[i], tol),
            "[{i}]: got {}, want {}, tol {tol:?}",
            got[i],
            want[i]
        );
    }
}

/// Asserts two matrices agree entrywise within an absolute `tol`.
pub fn assert_matrix_close<const R: usize, const C: usize, T: Numeric>(
    actual: Matrix<R, C, T>,
    expected: Matrix<R, C, T>,
    tol: T,
) {
    for row in 0..R {
        for col in 0..C {
            assert!((actual[(row, col)] - expected[(row, col)]).abs() < tol);
        }
    }
}

/// Asserts every entry of `matrix` is within `tol` of the identity matrix.
pub fn assert_identity<const N: usize, T: Numeric>(matrix: Matrix<N, N, T>, tol: T) {
    assert_matrix_close(matrix, Matrix::identity(), tol);
}

/// Factorizes `a`, checks the factors are triangular, and that they reconstruct `P·A`.
pub fn lu_reconstructs<const N: usize, T: Numeric>(a: Matrix<N, N, T>, tol: T) {
    let f = a.lu_decompose().unwrap();
    let lower = f.lower();
    let upper = f.upper();
    let perm = f.permutation();

    // lower is unit lower-triangular; upper is upper-triangular.
    for row in 0..N {
        assert_eq!(lower[(row, row)], T::ONE);
        for col in (row + 1)..N {
            assert_eq!(lower[(row, col)], T::ZERO);
        }
        for col in 0..row {
            assert_eq!(upper[(row, col)], T::ZERO);
        }
    }

    let permuted = Matrix::<N, N, T>::from_fn(|index, col| a[(perm[index], col)]);
    assert_matrix_close(lower * upper, permuted, tol);
}

/// Checks the Cholesky factor is lower-triangular with a positive diagonal and reconstructs `A`.
pub fn cholesky_reconstructs<const N: usize, T: Numeric>(a: Matrix<N, N, T>, tol: T) {
    let lower = a.cholesky().unwrap().lower();
    for row in 0..N {
        assert!(lower[(row, row)] > T::ZERO);
        for col in (row + 1)..N {
            assert_eq!(lower[(row, col)], T::ZERO);
        }
    }
    assert_matrix_close(lower * lower.transpose(), a, tol);
}

/// Checks the singular values are ordered and that `left·diag(sigma)·rightᵀ` reconstructs `A`.
pub fn svd_reconstructs<const M: usize, const N: usize, T: Numeric>(a: Matrix<M, N, T>, tol: T) {
    let f = a.svd().unwrap();
    let (left, sigma, right) = (f.left(), f.singular_values(), f.right());

    for k in 0..N {
        assert!(sigma[k] >= T::ZERO);
        if k + 1 < N {
            assert!(sigma[k] >= sigma[k + 1]);
        }
    }

    assert_identity(left.transpose() * left, tol);
    assert_identity(right.transpose() * right, tol);

    let recon = Matrix::<M, N, T>::from_fn(|row, col| {
        let mut acc = T::ZERO;
        for k in 0..N {
            acc += left[(row, k)] * sigma[k] * right[(col, k)];
        }
        acc
    });
    assert_matrix_close(recon, a, tol);
}

/// Verifies the four Moore–Penrose conditions for the pseudo-inverse of `a`.
pub fn svd_moore_penrose<const M: usize, const N: usize, T: Numeric>(a: Matrix<M, N, T>, tol: T) {
    let pinv = a.pseudo_inverse().unwrap();
    assert_matrix_close(a * pinv * a, a, tol);
    assert_matrix_close(pinv * a * pinv, pinv, tol);
    let aap = a * pinv;
    assert_matrix_close(aap, aap.transpose(), tol);
    let apa = pinv * a;
    assert_matrix_close(apa, apa.transpose(), tol);
}

/// Largest absolute entry of `a`, used to scale reconstruction/tolerance checks to the
/// magnitude of the input matrix.
#[must_use]
pub fn max_abs<const R: usize, const C: usize, T: Numeric>(a: Matrix<R, C, T>) -> T {
    let mut max = T::ZERO;
    for row in 0..R {
        for col in 0..C {
            max = max.max(a[(row, col)].abs());
        }
    }
    max
}

#[must_use]
fn f32_scaled_tol(scale: f32, dim: usize) -> f32 {
    512.0 * f32::EPSILON * dim as f32 * scale.max(1.0)
}

/// Verifies the four Moore-Penrose conditions for an f32 pseudo-inverse with
/// tolerances scaled by matrix magnitude and dimension.
pub fn svd_moore_penrose_f32<const M: usize, const N: usize>(a: Matrix<M, N, f32>) {
    let pinv = a.pseudo_inverse().unwrap();

    let aap_a = a * pinv * a;
    assert_matrix_close(aap_a, a, f32_scaled_tol(max_abs(a), M.max(N)));

    let apa_ap = pinv * a * pinv;
    assert_matrix_close(apa_ap, pinv, f32_scaled_tol(max_abs(pinv), M.max(N)));

    let aap = a * pinv;
    assert_matrix_close(aap, aap.transpose(), f32_scaled_tol(max_abs(aap), M));

    let apa = pinv * a;
    assert_matrix_close(apa, apa.transpose(), f32_scaled_tol(max_abs(apa), N));
}
