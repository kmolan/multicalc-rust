//! Stress test of the fixed-size linear solves — LU and Cholesky factorizations and the direct
//! 4x4 inverse — reporting per-call latency and approximation error (reconstruction, solve
//! residual, and inverse identity error) on well- and ill-conditioned inputs.
//!
//! Latency is illustrative in a debug build; run with `--release` for representative numbers:
//! `cargo run -p multicalc-demos --release --example linear_algebra`

use std::hint::black_box;
use std::time::Instant;

use multicalc::{CalcError, Matrix, Vector};

/// Mean wall-clock time per call, in nanoseconds, over `iters` runs.
fn time<T>(iters: u32, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut last = black_box(f()); // warm up and keep the last result live
    let start = Instant::now();
    for _ in 0..iters {
        last = black_box(f());
    }
    (last, start.elapsed().as_nanos() as f64 / iters as f64)
}

/// Largest entrywise absolute difference between two matrices.
fn max_abs<const R: usize, const C: usize>(a: Matrix<R, C>, b: Matrix<R, C>) -> f64 {
    let mut worst = 0.0f64;
    for r in 0..R {
        for c in 0..C {
            worst = worst.max((a[(r, c)] - b[(r, c)]).abs());
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

fn lu_report<const N: usize>(a: Matrix<N, N>, label: &str) -> Result<(), CalcError> {
    let x_true = Vector::<N>::from_fn(|i| 1.0 + i as f64);
    let b = a * x_true;

    // The factorization is fallible, so the timed closure hands back the `Result` and `?` takes
    // it from there: a singular matrix stops the demo with a typed error instead of a panic.
    let (factorization, ns) = time(50_000, || black_box(a).lu());
    let f = factorization?;

    // Reconstruction: row i of P·A is row perm[i] of A, and P·A == L·U.
    let perm = f.permutation();
    let pa = Matrix::<N, N>::from_fn(|i, c| a[(perm[i], c)]);
    let recon = max_abs(pa, f.l() * f.u());

    let residual = (a * f.solve(b) - b).norm();
    println!("  {label:<14} {ns:>8.1} ns   PA-LU {recon:.1e}   residual {residual:.1e}");
    Ok(())
}

fn cholesky_report<const N: usize>(a: Matrix<N, N>, label: &str) -> Result<(), CalcError> {
    let x_true = Vector::<N>::from_fn(|i| 1.0 + i as f64);
    let b = a * x_true;

    let (factorization, ns) = time(50_000, || black_box(a).cholesky());
    let f = factorization?;

    let l = f.l();
    let recon = max_abs(a, l * l.transpose());

    let x = f.solve(b);
    let residual = (a * x - b).norm();
    // Agreement with the general LU solve on the same system.
    let lu_x = a.lu()?.solve(b);
    let vs_lu = (0..N).map(|i| (x[i] - lu_x[i]).abs()).fold(0.0, f64::max);

    println!(
        "  {label:<14} {ns:>8.1} ns   A-LLt {recon:.1e}   residual {residual:.1e}   vs LU {vs_lu:.1e}"
    );
    Ok(())
}

fn inverse4_report(a: Matrix<4, 4>, label: &str) -> Result<(), CalcError> {
    let (inverse, ns) = time(100_000, || black_box(a).inverse());
    let identity_err = max_abs(a * inverse?, Matrix::<4, 4>::identity());
    println!("  {label:<14} {ns:>8.1} ns   identity err {identity_err:.1e}");
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
        let x = a.lu()?.solve(b);
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
    let indefinite = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    match indefinite.cholesky() {
        Ok(_) => println!("  {:<14} unexpectedly accepted", "indefinite 2x2"),
        Err(error) => println!("  {:<14} rejected: {error}", "indefinite 2x2"),
    }

    println!("\nDirect 4x4 inverse:");
    inverse4_report(general::<4>(), "general 4x4")?;
    inverse4_report(hilbert::<4>(), "Hilbert 4x4")?;

    Ok(())
}
