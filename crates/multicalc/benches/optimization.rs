#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::optimization::{GaussNewton, LevenbergMarquardt};
use multicalc::scalar::{Numeric, VectorFn, c};
use multicalc::scalar_fn_vec;

// Sum of three damped sinusoids sampled at 60 points, [A, lambda, omega, phi] per component.
// Holds the sample times and targets so the residual is model - target.
struct DampedSinusoids {
    t: [f64; 60],
    y: [f64; 60],
}

impl VectorFn<12, 60> for DampedSinusoids {
    fn eval<S: Numeric>(&self, p: &[S; 12]) -> [S; 60] {
        core::array::from_fn(|i| {
            let t = S::from_f64(self.t[i]);
            let mut model = S::ZERO;
            for k in 0..3 {
                let a = p[4 * k];
                let lambda = p[4 * k + 1];
                let omega = p[4 * k + 2];
                let phi = p[4 * k + 3];
                model += a * (-(lambda * t)).exp() * (omega * t + phi).sin();
            }
            model - S::from_f64(self.y[i])
        })
    }
}

// The Moré-Garbow-Hillstrom trigonometric function in N variables; global minimum zero.
struct Trigonometric<const N: usize>;

impl<const N: usize> VectorFn<N, N> for Trigonometric<N> {
    fn eval<S: Numeric>(&self, x: &[S; N]) -> [S; N] {
        let n = S::from_f64(N as f64);
        let mut cos_sum = S::ZERO;
        for &xj in x {
            cos_sum += xj.cos();
        }
        core::array::from_fn(|i| {
            n - cos_sum + S::from_f64((i + 1) as f64) * (S::ONE - x[i].cos()) - x[i].sin()
        })
    }
}

// Geometric circle fit [cx, cy, r] minimizing sqrt((x-cx)^2 + (y-cy)^2) - r over measured points.
struct CircleFit {
    px: [f64; 40],
    py: [f64; 40],
}

impl VectorFn<3, 40> for CircleFit {
    fn eval<S: Numeric>(&self, p: &[S; 3]) -> [S; 40] {
        let cx = p[0];
        let cy = p[1];
        let r = p[2];
        core::array::from_fn(|i| {
            let dx = S::from_f64(self.px[i]) - cx;
            let dy = S::from_f64(self.py[i]) - cy;
            (dx * dx + dy * dy).sqrt() - r
        })
    }
}

fn levenberg_marquardt(crit: &mut Criterion) {
    let rosenbrock = scalar_fn_vec!(|v: &[f64; 2]| [c(10.0) * (v[1] - v[0] * v[0]), c(1.0) - v[0]]);
    crit.bench_function("lm/rosenbrock", |b| {
        b.iter(|| {
            LevenbergMarquardt::<AutoDiffMulti>::default()
                .minimize(black_box(&rosenbrock), black_box(&[-1.2, 1.0]))
                .unwrap()
        })
    });

    // a*e^(b*t) through (0,100), (1,50), (2,25).
    let decay = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-100.0) + v[0],
        c(-50.0) + v[0] * v[1].exp(),
        c(-25.0) + v[0] * (c(2.0) * v[1]).exp(),
    ]);
    crit.bench_function("lm/exponential_decay", |b| {
        b.iter(|| {
            LevenbergMarquardt::<AutoDiffMulti>::default()
                .minimize(black_box(&decay), black_box(&[80.0, -0.3]))
                .unwrap()
        })
    });

    // 12-parameter fit of three damped sinusoids to 60 noiseless samples.
    let truth = [1.0, 0.5, 2.0, 0.3, 0.7, 0.2, 5.0, 1.1, 1.3, 0.8, 8.5, -0.5];
    let t: [f64; 60] = core::array::from_fn(|i| i as f64 * 6.0 / 59.0);
    let mut sinusoids = DampedSinusoids { t, y: [0.0; 60] };
    sinusoids.y = sinusoids.eval(&truth);
    let start = [
        1.15, 0.55, 2.05, 0.2, 0.6, 0.18, 5.08, 1.2, 1.45, 0.72, 8.42, -0.65,
    ];
    crit.bench_function("lm/damped_sinusoids_12p", |b| {
        b.iter(|| {
            LevenbergMarquardt::<AutoDiffMulti>::default()
                .minimize(black_box(&sinusoids), black_box(&start))
                .unwrap()
        })
    });

    // 6-variable MGH trigonometric function to its global minimum.
    crit.bench_function("lm/trigonometric_6v", |b| {
        b.iter(|| {
            LevenbergMarquardt::<AutoDiffMulti>::default()
                .minimize(black_box(&Trigonometric::<6>), black_box(&[1.0 / 6.0; 6]))
                .unwrap()
        })
    });
}

fn gauss_newton(crit: &mut Criterion) {
    // A linear residual: Gauss-Newton reaches the exact least-squares solution in one step.
    let linear = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[1],
        c(-3.0) + v[0] + v[1],
        c(-5.0) + c(2.0) * v[0] + v[1],
    ]);
    crit.bench_function("gn/linear_least_squares", |b| {
        b.iter(|| {
            GaussNewton::<AutoDiffMulti>::default()
                .minimize(black_box(&linear), black_box(&[0.0, 0.0]))
                .unwrap()
        })
    });

    let rosenbrock = scalar_fn_vec!(|v: &[f64; 2]| [c(10.0) * (v[1] - v[0] * v[0]), c(1.0) - v[0]]);
    crit.bench_function("gn/rosenbrock", |b| {
        b.iter(|| {
            GaussNewton::<AutoDiffMulti>::default()
                .minimize(black_box(&rosenbrock), black_box(&[0.9, 0.9]))
                .unwrap()
        })
    });

    // 3-parameter geometric circle fit over 40 points on a circle of center (2, -1), radius 3.
    let angle = |i: usize| std::f64::consts::TAU * i as f64 / 40.0;
    let px = core::array::from_fn(|i| 2.0 + 3.0 * angle(i).cos());
    let py = core::array::from_fn(|i| -1.0 + 3.0 * angle(i).sin());
    let circle = CircleFit { px, py };
    crit.bench_function("gn/circle_fit_3p", |b| {
        b.iter(|| {
            GaussNewton::<AutoDiffMulti>::default()
                .minimize(black_box(&circle), black_box(&[2.4, -0.6, 3.5]))
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
    targets = levenberg_marquardt, gauss_newton
}
criterion_main!(benches);
