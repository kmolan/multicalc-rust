//! Stress test of the singular value decomposition and Moore–Penrose pseudo-inverse on
//! robotics-shaped problems, reporting per-call latency and approximation error: Kabsch rotation
//! recovery, a redundant-arm pseudo-inverse, a near-singular Jacobian, and an overdetermined fit.
//!
//! Latency is illustrative in a debug build; run with `--release` for representative numbers:
//! `cargo run -p multicalc-demos --release --example svd`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::hint::black_box;
use std::time::Instant;

use multicalc::linear_algebra::{Matrix, Vector};

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
            worst =
                worst.max((a.get(r, c).copied().unwrap() - b.get(r, c).copied().unwrap()).abs());
        }
    }
    worst
}

/// Largest absolute entry of a matrix.
fn max_entry<const R: usize, const C: usize>(a: Matrix<R, C>) -> f64 {
    let mut worst = 0.0f64;
    for r in 0..R {
        for c in 0..C {
            worst = worst.max(a.get(r, c).copied().unwrap().abs());
        }
    }
    worst
}

/// Kabsch/Wahba: recover a known rotation from paired point clouds via the 3x3 cross-covariance.
fn kabsch() {
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
    // Cross-covariance H = Σ (R·p) pᵀ.
    let mut h = Matrix::<3, 3>::zeros();
    for p in pts {
        let pv = Vector::new(p);
        let q = rot * pv;
        for i in 0..3 {
            for j in 0..3 {
                if let Some(slot) = h.get_mut(i, j) {
                    *slot += q.as_array()[i] * pv.as_array()[j];
                }
            }
        }
    }

    let (f, ns) = time(100_000, || black_box(h).svd().unwrap());
    let (u, v) = (f.u(), f.v());
    let mut rhat = u * v.transpose();
    // Reflection fix: force a proper rotation (determinant +1).
    if rhat.determinant() < 0.0 {
        let mut uf = u;
        for i in 0..3 {
            let u = uf.get(i, 2).copied().unwrap();
            if let Some(slot) = uf.get_mut(i, 2) {
                *slot = -u;
            }
        }
        rhat = uf * v.transpose();
    }
    let rot_err = max_abs(rhat, rot);
    let ortho_err = max_abs(rhat.transpose() * rhat, Matrix::<3, 3>::identity());
    assert!(
        rot_err < 1e-9 && ortho_err < 1e-9,
        "SVD should recover the rotation"
    );
    let label = "Kabsch 3x3";
    println!("  {label:<20} {ns:>8.1} ns   R-error {rot_err:.1e}   orthogonality {ortho_err:.1e}");
}

/// Redundant 7-DoF arm: resolve joint rates with the wide (6x7) Jacobian pseudo-inverse.
fn redundant_arm() {
    let j = Matrix::<6, 7>::from_fn(|r, c| {
        if c < 6 {
            if r == c {
                2.0
            } else {
                0.3 / (1.0 + (r + c) as f64)
            }
        } else {
            0.5 * (r as f64 + 1.0)
        }
    });
    let (jp, ns) = time(50_000, || black_box(j).pseudo_inverse().unwrap());
    let mp_err = max_abs(j * jp * j, j);
    let jjp = j * jp;
    let sym_err = max_abs(jjp, jjp.transpose());
    let label = "Redundant arm 6x7";
    println!("  {label:<20} {ns:>8.1} ns   JJ⁺J-J {mp_err:.1e}   symmetry {sym_err:.1e}");
}

/// Near kinematic singularity: two joint axes nearly align, so one singular value is tiny.
fn near_singular() {
    let mut j = Matrix::<6, 6>::from_fn(|i, jj| {
        if i == jj {
            1.0 + 0.1 * jj as f64
        } else {
            0.2 / (1.0 + (i + jj) as f64)
        }
    });
    for r in 0..6 {
        let j4 = j.get(r, 4).copied().unwrap();
        if let Some(slot) = j.get_mut(r, 5) {
            *slot = j4 + 1e-8 * (r as f64 + 1.0);
        }
    }
    let (f, ns) = time(100_000, || black_box(j).svd().unwrap());
    let cond = f.condition_number();
    let tol = 1e-4 * f.singular_values().as_array()[0];
    let rank = f.rank(tol);
    let pinv_max = max_entry(f.pseudo_inverse_tol(tol));
    let label = "Near-singular 6x6";
    println!(
        "  {label:<20} {ns:>8.1} ns   cond {cond:.1e}   rank {rank}   |pinv|max {pinv_max:.1e}"
    );
}

/// Overdetermined plane fit: solve 30 noisy samples and cross-check the normal equations.
fn overdetermined() {
    let sample = |i: usize| ((i as f64 * 0.37).sin(), (i as f64 * 0.53).cos());
    let design = Matrix::<30, 3>::from_fn(|i, c| {
        let (x, y) = sample(i);
        match c {
            0 => x,
            1 => y,
            _ => 1.0,
        }
    });
    let rhs = Vector::<30>::from_fn(|i| {
        let (x, y) = sample(i);
        0.5 * x - 1.2 * y + 2.0 + 1e-3 * (i as f64 * 1.7).sin()
    });
    let (x_svd, ns) = time(50_000, || {
        black_box(design).svd().unwrap().solve(black_box(rhs))
    });
    let x_ne = (design.transpose() * design)
        .solve(design.transpose() * rhs)
        .unwrap();
    let mut vs_ne = 0.0f64;
    for i in 0..3 {
        vs_ne = vs_ne.max((x_svd.as_array()[i] - x_ne.as_array()[i]).abs());
    }
    let residual = (design * x_svd - rhs).norm();
    let label = "Overdetermined 30x3";
    println!("  {label:<20} {ns:>8.1} ns   vs normal-eq {vs_ne:.1e}   residual {residual:.1e}");
}

fn main() {
    println!("Kabsch rotation recovery (H = Σ q pᵀ, R̂ = U Vᵀ):");
    kabsch();

    println!("\nRedundant-arm pseudo-inverse (6x7 Jacobian, wide path):");
    redundant_arm();

    println!("\nNear-singular Jacobian (kinematic singularity, truncated pseudo-inverse):");
    near_singular();

    println!("\nOverdetermined least squares (plane fit, 30x3):");
    overdetermined();
}
