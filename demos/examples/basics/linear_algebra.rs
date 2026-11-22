//! Stress test of the fixed-size linear solves — LU and Cholesky factorizations, the symmetric
//! eigendecomposition, and the direct 4x4 inverse — reporting per-call latency and approximation
//! error (reconstruction, solve residual, inverse identity error, and how far the eigenvector
//! directions are from being at right angles) on well- and ill-conditioned inputs.
//!
//! Latency is illustrative in a debug build; run with `--release` for representative numbers:
//! `cargo run -p multicalc-demos --release --example linear_algebra`

use std::hint::black_box;
use std::time::Instant;

use multicalc::{CalcError, Matrix, Matrix2D, Matrix4D, Vector};

/// Mean wall-clock time per call, in nanoseconds, over `iters` runs.
#[must_use]
fn time<T>(iters: u32, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut last = black_box(f()); // warm up and keep the last result live
    let start = Instant::now();
    for _ in 0..iters {
        last = black_box(f());
    }
    (last, start.elapsed().as_nanos() as f64 / iters as f64)
}

/// Largest entrywise absolute difference between two matrices.
#[must_use]
fn max_abs<const R: usize, const C: usize>(a: Matrix<R, C>, b: Matrix<R, C>) -> f64 {
    let mut worst = 0.0f64;
    for row in 0..R {
        for col in 0..C {
            worst = worst.max((a[(row, col)] - b[(row, col)]).abs());
        }
    }
    worst
}

/// The N×N Hilbert matrix — symmetric positive-definite but notoriously ill-conditioned.
fn hilbert<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| 1.0 / ((i + j + 1) as f64))
}

/// Diagonally dominant and mildly non-symmetric — well-conditioned and invertible.
fn general<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| {
        if i == j {
            (N + 2) as f64
        } else {
            1.0 / (1.0 + i as f64 + 2.0 * j as f64)
        }
    })
}

/// Symmetric positive-definite: diagonally dominant with a unit off-diagonal.
fn spd<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| if i == j { (N + 1) as f64 } else { 1.0 })
}

/// Symmetric with well-separated eigenvalues: diagonally dominant with a decaying off-diagonal.
fn symmetric<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| {
        if i == j {
            (N + 2) as f64
        } else {
            1.0 / (1.0 + (i + j) as f64)
        }
    })
}

fn lu_report<const N: usize>(a: Matrix<N, N>, label: &str) -> Result<(), CalcError> {
    let x_true = Vector::<N>::from_fn(|i| 1.0 + i as f64);
    let b = a * x_true;

    // The factorization is fallible, so the timed closure hands back the `Result` and `?` takes
    // it from there: a singular matrix stops the demo with a typed error instead of a panic.
    let (factorization, nanos) = time(50_000, || black_box(a).lu_decompose());
    let f = factorization?;

    // Reconstruction: row i of P·A is row perm[i] of A, and P·A == L·U.
    let perm = f.permutation();
    let permuted = Matrix::<N, N>::from_fn(|i, col| a[(perm[i], col)]);
    let recon = max_abs(permuted, f.lower() * f.upper());

    let residual = (a * f.solve(b) - b).norm();
    println!("  {label:<14} {nanos:>8.1} ns   PA-LU {recon:.1e}   residual {residual:.1e}");
    Ok(())
}

fn cholesky_report<const N: usize>(a: Matrix<N, N>, label: &str) -> Result<(), CalcError> {
    let x_true = Vector::<N>::from_fn(|i| 1.0 + i as f64);
    let b = a * x_true;

    let (factorization, nanos) = time(50_000, || black_box(a).cholesky());
    let f = factorization?;

    let lower_tri = f.lower();
    let recon = max_abs(a, lower_tri * lower_tri.transpose());

    let x = f.solve(b);
    let residual = (a * x - b).norm();
    // Agreement with the general LU solve on the same system.
    let lu_x = a.lu_decompose()?.solve(b);
    let vs_lu = (0..N).map(|i| (x[i] - lu_x[i]).abs()).fold(0.0, f64::max);

    println!(
        "  {label:<14} {nanos:>8.1} ns   A-LLt {recon:.1e}   residual {residual:.1e}   vs LU {vs_lu:.1e}"
    );
    Ok(())
}

fn symmetric_eigen_report<const N: usize>(a: Matrix<N, N>, label: &str) -> Result<(), CalcError> {
    let (decomposition, nanos) = time(20_000, || black_box(a).symmetric_eigendecomposition());
    let f = decomposition?;

    let values = f.eigenvalues();
    let vectors = f.eigenvectors();

    // Reconstruction: V·diag(λ)·Vᵀ == A.
    let recon = max_abs(
        a,
        Matrix::<N, N>::from_fn(|row, col| {
            (0..N)
                .map(|k| vectors[(row, k)] * values[k] * vectors[(col, k)])
                .sum()
        }),
    );
    // The directions should be at right angles and of unit length, so VᵀV is the identity.
    let right_angles = max_abs(vectors.transpose() * vectors, Matrix::identity());
    let condition = f.condition_number();

    println!(
        "  {label:<14} {nanos:>8.1} ns   A-VLVt {recon:.1e}   VtV-I {right_angles:.1e}   cond {condition:.1e}"
    );
    Ok(())
}

fn inverse4_report(a: Matrix4D, label: &str) -> Result<(), CalcError> {
    let (inverse, nanos) = time(100_000, || black_box(a).inverse());
    let identity_err = max_abs(a * inverse?, Matrix4D::identity());
    println!("  {label:<14} {nanos:>8.1} ns   identity err {identity_err:.1e}");
    Ok(())
}

// Every fallible call below propagates with `?`. The error types are per-module — here
// `LinalgError` — and each converts into the `CalcError` umbrella on the way out, so one return
// type covers a program that mixes modules.
fn main() -> Result<(), CalcError> {
    // Sanity: the LU solve residual on a well-conditioned system is tiny.
    {
        let a = general::<4>();
        let x_true = Vector::<4>::from_fn(|i| 1.0 + i as f64);
        let b = a * x_true;
        let x = a.lu_decompose()?.solve(b);
        assert!((a * x - b).norm() < 1e-9, "LU solve residual too large");
    }

    println!("LU (any invertible matrix) - decompose + solve:");
    lu_report(general::<4>(), "general 4x4")?;
    lu_report(general::<8>(), "general 8x8")?;
    lu_report(hilbert::<6>(), "Hilbert 6x6")?;

    println!("\nCholesky (symmetric positive-definite) - decompose + solve:");
    cholesky_report(spd::<4>(), "SPD 4x4")?;
    cholesky_report(spd::<8>(), "SPD 8x8")?;
    cholesky_report(hilbert::<6>(), "Hilbert 6x6")?;

    // Error path: the guard rejects a non-positive-definite matrix before taking a root. This one
    // is expected to fail, so it is matched rather than propagated.
    let indefinite = Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]);
    match indefinite.cholesky() {
        Ok(_) => println!("  {:<14} unexpectedly accepted", "indefinite 2x2"),
        Err(error) => println!("  {:<14} rejected: {error}", "indefinite 2x2"),
    }

    println!("\nSymmetric eigendecomposition - eigenvalues + directions:");
    symmetric_eigen_report(symmetric::<4>(), "symmetric 4x4")?;
    symmetric_eigen_report(symmetric::<8>(), "symmetric 8x8")?;
    symmetric_eigen_report(hilbert::<6>(), "Hilbert 6x6")?;

    // Conditioning: raising every eigenvalue to a floor and rebuilding is what turns a covariance
    // that has drifted below zero back into one a filter can keep using.
    let drifted = Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]);
    let before = drifted.symmetric_eigendecomposition()?;
    let repaired = before.clamped(0.5);
    let after = repaired.symmetric_eigendecomposition()?.eigenvalues();
    let before_values = before.eigenvalues();
    println!(
        "  {:<14} eigenvalues {:.2}, {:.2} -> {:.2}, {:.2}",
        "clamped 2x2", before_values[0], before_values[1], after[0], after[1]
    );

    // Error path: a matrix that does not read the same across the diagonal is rejected rather than
    // quietly decomposed as if it did. Expected to fail, so it is matched rather than propagated.
    let lopsided = Matrix2D::new([[1.0, 2.0], [-2.0, 1.0]]);
    match lopsided.symmetric_eigendecomposition() {
        Ok(_) => println!("  {:<14} unexpectedly accepted", "lopsided 2x2"),
        Err(error) => println!("  {:<14} rejected: {error}", "lopsided 2x2"),
    }

    println!("\nDirect 4x4 inverse:");
    inverse4_report(general::<4>(), "general 4x4")?;
    inverse4_report(hilbert::<4>(), "Hilbert 4x4")?;

    Ok(())
}
