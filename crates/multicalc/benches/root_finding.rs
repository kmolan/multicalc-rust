use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::finite_difference::FiniteDifferenceSingle;
use multicalc::root_finding::{Bisection, Newton, NewtonSystem};
use multicalc::scalar::{Numeric, ScalarFn, VectorFn, c};
use multicalc::scalar_fn;
use multicalc::scalar_fn_vec;

// Kepler's equation E - e*sin(E) - M, relating the mean anomaly M to the
// eccentric anomaly E of an orbit with eccentricity e.
struct Kepler {
    e: f64,
    m: f64,
}

impl ScalarFn for Kepler {
    fn eval<S: Numeric>(&self, big_e: S) -> S {
        big_e - S::from_f64(self.e) * big_e.sin() - S::from_f64(self.m)
    }
}

// Colebrook-White equation for the Darcy friction factor f of turbulent
// pipe flow: 1/√f + 2*log10(rel_roughness/3.7 + 2.51/(Re*√f)) = 0.
struct Colebrook {
    reynolds: f64,
    rel_roughness: f64,
}

impl ScalarFn for Colebrook {
    fn eval<S: Numeric>(&self, f: S) -> S {
        let re = S::from_f64(self.reynolds);
        let eps = S::from_f64(self.rel_roughness);
        let root_f = f.sqrt();
        let inner = eps / S::from_f64(3.7) + S::from_f64(2.51) / (re * root_f);
        let log10 = inner.ln() / S::from_f64(10.0).ln();
        S::ONE / root_f + S::TWO * log10
    }
}

// Two-link planar arm forward kinematics; the root recovers the joint angles
// that place the tip at the target (px, py).
struct TwoLinkArm {
    l1: f64,
    l2: f64,
    px: f64,
    py: f64,
}

impl VectorFn<2, 2> for TwoLinkArm {
    fn eval<S: Numeric>(&self, v: &[S; 2]) -> [S; 2] {
        let l1 = S::from_f64(self.l1);
        let l2 = S::from_f64(self.l2);
        let px = S::from_f64(self.px);
        let py = S::from_f64(self.py);
        [
            l1 * v[0].cos() + l2 * (v[0] + v[1]).cos() - px,
            l1 * v[0].sin() + l2 * (v[0] + v[1]).sin() - py,
        ]
    }
}

fn bisection(crit: &mut Criterion) {
    // Wien's displacement law on [1, 10].
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    crit.bench_function("bisection/wien", |b| {
        b.iter(|| {
            Bisection::default()
                .solve(black_box(&wien), black_box(1.0_f64), black_box(10.0))
                .unwrap()
        })
    });

    // Kepler's equation, e = 0.8, bracketed on [0, π].
    let e = 0.8_f64;
    let m = 1.0 - e * 1.0_f64.sin();
    let kepler = Kepler { e, m };
    crit.bench_function("bisection/kepler", |b| {
        b.iter(|| {
            Bisection::default()
                .solve(
                    black_box(&kepler),
                    black_box(0.0_f64),
                    black_box(std::f64::consts::PI),
                )
                .unwrap()
        })
    });
}

fn newton(crit: &mut Criterion) {
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    crit.bench_function("newton/wien", |b| {
        b.iter(|| {
            Newton::<AutoDiffSingle>::default()
                .solve(black_box(&wien), black_box(5.0_f64))
                .unwrap()
        })
    });

    let e = 0.8_f64;
    let m = 1.0 - e * 1.0_f64.sin();
    let kepler = Kepler { e, m };
    crit.bench_function("newton/kepler", |b| {
        b.iter(|| {
            Newton::<AutoDiffSingle>::default()
                .solve(black_box(&kepler), black_box(m))
                .unwrap()
        })
    });

    let colebrook = Colebrook {
        reynolds: 1.0e5,
        rel_roughness: 1.0e-4,
    };
    crit.bench_function("newton/colebrook", |b| {
        b.iter(|| {
            Newton::<AutoDiffSingle>::default()
                .solve(black_box(&colebrook), black_box(0.02_f64))
                .unwrap()
        })
    });

    // Finite-difference derivative in place of exact autodiff, on the Wien root.
    crit.bench_function("newton_fd/wien", |b| {
        b.iter(|| {
            Newton::from_derivator(FiniteDifferenceSingle::<f64>::default())
                .solve(black_box(&wien), black_box(5.0_f64))
                .unwrap()
        })
    });
}

fn damped_newton(crit: &mut Criterion) {
    // f(x) = x / sqrt(1 + x²), root at 0. Plain Newton diverges from x0 = 2; the
    // backtracking line search halves the step until |f| decreases.
    let sigmoid = scalar_fn!(|x| x / (c(1.0) + x * x).sqrt());
    crit.bench_function("damped_newton/sigmoid", |b| {
        b.iter(|| {
            Newton::<AutoDiffSingle>::default()
                .with_backtracking(true)
                .solve(black_box(&sigmoid), black_box(2.0_f64))
                .unwrap()
        })
    });
}

fn newton_system(crit: &mut Criterion) {
    // Two-link IK (N = 2): recover the joint angles from a tip target.
    let (l1, l2) = (1.0_f64, 1.0_f64);
    let (t1, t2) = (0.5_f64, 0.8_f64);
    let px = l1 * t1.cos() + l2 * (t1 + t2).cos();
    let py = l1 * t1.sin() + l2 * (t1 + t2).sin();
    let arm = TwoLinkArm { l1, l2, px, py };
    crit.bench_function("newton_system/two_link_ik", |b| {
        b.iter(|| {
            NewtonSystem::<AutoDiffMulti>::default()
                .solve(black_box(&arm), black_box(&[0.4_f64, 0.9]))
                .unwrap()
        })
    });

    // Circle ∩ hyperbola (N = 2): x² + y² = 4 and xy = 1.
    let circle =
        scalar_fn_vec!(
            |v: &[f64; 2]| [c(-4.0) + v[0] * v[0] + v[1] * v[1], c(-1.0) + v[0] * v[1],]
        );
    crit.bench_function("newton_system/circle_intersection", |b| {
        b.iter(|| {
            NewtonSystem::<AutoDiffMulti>::default()
                .solve(black_box(&circle), black_box(&[1.5_f64, 0.8]))
                .unwrap()
        })
    });

    // Chemical equilibrium mass balance (N = 3): Jacobian + LU scaling.
    let equilibrium = scalar_fn_vec!(|v: &[f64; 3]| [
        c(-1.0) + v[0] + v[1] + v[2],
        v[1] - c(1.25) * v[0] * v[0],
        v[2] - c(5.0) * v[0] * v[1],
    ]);
    crit.bench_function("newton_system/equilibrium_3x3", |b| {
        b.iter(|| {
            NewtonSystem::<AutoDiffMulti>::default()
                .solve(black_box(&equilibrium), black_box(&[0.5_f64, 0.25, 0.25]))
                .unwrap()
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bisection, newton, damped_newton, newton_system
}
criterion_main!(benches);
