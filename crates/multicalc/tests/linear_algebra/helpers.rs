//! Shared helpers for the linear algebra test suite.

use multicalc::linear_algebra::Matrix;
use multicalc::scalar::Numeric;

/// Asserts two matrices agree entrywise within `tol`.
pub(crate) fn assert_close<const R: usize, const C: usize, T: Numeric>(
    actual: Matrix<R, C, T>,
    expected: Matrix<R, C, T>,
    tol: T,
) {
    for r in 0..R {
        for c in 0..C {
            assert!((actual[(r, c)] - expected[(r, c)]).abs() < tol);
        }
    }
}

/// Asserts every entry of `m` is within `tol` of the identity matrix.
pub(crate) fn assert_identity<const N: usize, T: Numeric>(m: Matrix<N, N, T>, tol: T) {
    assert_close(m, Matrix::identity(), tol);
}

/// Factorizes `a`, checks the factors are triangular, and that they reconstruct `P·A`.
pub(crate) fn lu_reconstructs<const N: usize, T: Numeric>(a: Matrix<N, N, T>, tol: T) {
    let f = a.lu().unwrap();
    let l = f.l();
    let u = f.u();
    let perm = f.permutation();

    // L is unit lower-triangular; U is upper-triangular.
    for r in 0..N {
        assert_eq!(l[(r, r)], T::ONE);
        for c in (r + 1)..N {
            assert_eq!(l[(r, c)], T::ZERO);
        }
        for c in 0..r {
            assert_eq!(u[(r, c)], T::ZERO);
        }
    }

    let pa = Matrix::<N, N, T>::from_fn(|i, c| a[(perm[i], c)]);
    assert_close(l * u, pa, tol);
}

/// Checks the Cholesky factor is lower-triangular with a positive diagonal and reconstructs `A`.
pub(crate) fn cholesky_reconstructs<const N: usize, T: Numeric>(a: Matrix<N, N, T>, tol: T) {
    let l = a.cholesky().unwrap().l();
    for r in 0..N {
        assert!(l[(r, r)] > T::ZERO);
        for c in (r + 1)..N {
            assert_eq!(l[(r, c)], T::ZERO);
        }
    }
    assert_close(l * l.transpose(), a, tol);
}

/// Checks the singular values are ordered and that `U·diag(σ)·Vᵀ` reconstructs `A`.
pub(crate) fn svd_reconstructs<const M: usize, const N: usize, T: Numeric>(
    a: Matrix<M, N, T>,
    tol: T,
) {
    let f = a.svd().unwrap();
    let (u, s, v) = (f.u(), f.singular_values(), f.v());

    for k in 0..N {
        assert!(s[k] >= T::ZERO);
        if k + 1 < N {
            assert!(s[k] >= s[k + 1]);
        }
    }

    assert_identity(u.transpose() * u, tol);
    assert_identity(v.transpose() * v, tol);

    let recon = Matrix::<M, N, T>::from_fn(|r, c| {
        let mut acc = T::ZERO;
        for k in 0..N {
            acc += u[(r, k)] * s[k] * v[(c, k)];
        }
        acc
    });
    assert_close(recon, a, tol);
}

/// Verifies the four Moore–Penrose conditions for the pseudo-inverse of `a`.
pub(crate) fn svd_moore_penrose<const M: usize, const N: usize, T: Numeric>(
    a: Matrix<M, N, T>,
    tol: T,
) {
    let ap = a.pseudo_inverse().unwrap();
    assert_close(a * ap * a, a, tol);
    assert_close(ap * a * ap, ap, tol);
    let aap = a * ap;
    assert_close(aap, aap.transpose(), tol);
    let apa = ap * a;
    assert_close(apa, apa.transpose(), tol);
}
