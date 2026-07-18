//! Runs criterion benchmarks and writes the results to benchmarks/latency.md.
//! Run with `cargo bench -p multicalc-qa`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::hint::black_box;
use std::path::PathBuf;

use criterion::Criterion;

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;
use multicalc::ode::{Rk4, Rk45};
use multicalc::root_finding::NewtonSystem;
use multicalc::scalar::{Numeric, VectorFn, c};
use multicalc::{LevenbergMarquardt, scalar_fn, scalar_fn_vec};

/// Diagonally dominant, mildly non-symmetric — well-conditioned and invertible.
fn general<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| {
        if i == j {
            (N + 2) as f64
        } else {
            1.0 / (1.0 + i as f64 + 2.0 * j as f64)
        }
    })
}

/// Residual struct for the Levenberg–Marquardt bench: y = a·e^(b·t).
struct SensorFit<const M: usize> {
    t: [f64; M],
    y: [f64; M],
}

impl<const M: usize> VectorFn<2, M> for SensorFit<M> {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; M] {
        let (a, b) = (p[0], p[1]);
        core::array::from_fn(|i| a * (b * S::from_f64(self.t[i])).exp() - S::from_f64(self.y[i]))
    }
}

fn bench_derivative(c: &mut Criterion) {
    // f(x) = x^2 sin(x); third derivative by autodiff at x = 1.0.
    let f = scalar_fn!(|x| x * x * x.sin());
    let d = AutoDiffSingle::default();
    c.bench_function("derivative", |b| {
        b.iter(|| d.get(black_box(3), &f, black_box(1.0)).unwrap())
    });
}

fn bench_jacobian_small(c: &mut Criterion) {
    // (x*y*z, x^2 + y^2): a 2x3 Jacobian.
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let j: Jacobian = Jacobian::default();
    let p = [1.0, 2.0, 3.0];
    c.bench_function("jacobian_small", |b| {
        b.iter(|| j.get(&f, black_box(&p)).unwrap())
    });
}

fn bench_jacobian_large(crit: &mut Criterion) {
    // A coupled 6-in / 6-out map: a 6x6 Jacobian
    let f = scalar_fn_vec!(|v: &[f64; 6]| [
        v[0] * v[1] + v[2].sin(),
        v[1] * v[2] + v[3].cos(),
        v[2] * v[3] + c(0.1) * v[4].exp(),
        v[3] * v[4] + v[0] * v[5],
        v[4] * v[5] + v[1].sin(),
        v[5] * v[0] + v[2] * v[3],
    ]);
    let j: Jacobian = Jacobian::default();
    let p = [0.5, 1.0, 1.5, 0.3, 0.8, 1.2];
    crit.bench_function("jacobian_large", |b| {
        b.iter(|| j.get(&f, black_box(&p)).unwrap())
    });
}

fn bench_gauss_quad(c: &mut Criterion) {
    // Non-polynomial integrand on [0, 1], Gauss-Legendre order 16.
    let quad = GaussianSingle::from_parameters(16, GaussianQuadratureMethod::GaussLegendre);
    c.bench_function("gauss_quad", |b| {
        b.iter(|| {
            quad.get_single(
                &|x: f64| (x.sin() - x.sqrt()) * (-x).exp(),
                black_box(&[0.0, 1.0]),
            )
            .unwrap()
        })
    });
}

fn bench_lu_solve(c: &mut Criterion) {
    // Well-conditioned 10x10; decompose + solve A x = b.
    let a = general::<10>();
    let x_true = Vector::<10>::from_fn(|i| 1.0 + i as f64);
    let b_rhs = a * x_true;
    c.bench_function("lu_solve", |b| {
        b.iter(|| black_box(a).lu().unwrap().solve(black_box(b_rhs)))
    });
}

fn bench_svd_solve(c: &mut Criterion) {
    // Overdetermined 30x3 least-squares.
    let sample = |i: usize| ((i as f64 * 0.37).sin(), (i as f64 * 0.53).cos());
    let design = Matrix::<30, 3>::from_fn(|i, col| {
        let (x, y) = sample(i);
        match col {
            0 => x,
            1 => y,
            _ => 1.0,
        }
    });
    let rhs = Vector::<30>::from_fn(|i| {
        let (x, y) = sample(i);
        0.5 * x - 1.2 * y + 2.0 + 1e-3 * (i as f64 * 1.7).sin()
    });
    c.bench_function("svd_solve", |b| {
        b.iter(|| black_box(design).svd().unwrap().solve(black_box(rhs)))
    });
}

fn bench_expm(c: &mut Criterion) {
    // Matrix exponential of a fixed 6x6.
    let m = general::<6>();
    c.bench_function("expm", |b| b.iter(|| black_box(m).expm().unwrap()));
}

fn bench_rk45_solve(c: &mut Criterion) {
    // Harmonic oscillator y'' = -y, adaptive solve to t = 2*pi.
    let f = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
    let y0 = Vector::new([1.0, 0.0]);
    let solver = Rk45::default().with_rtol(1e-9).with_atol(1e-12);
    c.bench_function("rk45_solve", |b| {
        b.iter(|| {
            solver
                .solve(&f, 0.0, black_box(&y0), core::f64::consts::TAU)
                .unwrap()
        })
    });
}

fn bench_rk4_integrate(c: &mut Criterion) {
    // Same harmonic oscillator, fixed-step RK4 (2000 steps), no-op observer.
    let f = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
    let y0 = Vector::new([1.0, 0.0]);
    let steps = 2000usize;
    let dt = core::f64::consts::TAU / steps as f64;
    c.bench_function("rk4_integrate", |b| {
        b.iter(|| Rk4::integrate(&f, 0.0, black_box(&y0), dt, steps, |_t, _y| {}).unwrap())
    });
}

fn bench_lev_marq(c: &mut Criterion) {
    // Sensor-calibration curve fit y = a·e^(b·t) to 8 samples, LM with autodiff Jacobians.
    let t: [f64; 8] = core::array::from_fn(|i| i as f64);
    let y: [f64; 8] = core::array::from_fn(|i| 100.0 * (-0.5 * i as f64).exp());
    let problem = SensorFit { t, y };
    c.bench_function("lev_marq", |b| {
        b.iter(|| {
            LevenbergMarquardt::<AutoDiffMulti>::default()
                .minimize(&problem, black_box(&[80.0, -0.3]))
                .unwrap()
        })
    });
}

fn bench_newton_system(crit: &mut Criterion) {
    // x^2 + y^2 = 4 and x*y = 1 (circle ∩ hyperbola).
    let system =
        scalar_fn_vec!(
            |v: &[f64; 2]| [c(-4.0) + v[0] * v[0] + v[1] * v[1], c(-1.0) + v[0] * v[1],]
        );
    crit.bench_function("newton_system", |b| {
        b.iter(|| {
            NewtonSystem::<AutoDiffMulti>::default()
                .solve(&system, black_box(&[1.5, 0.8]))
                .unwrap()
        })
    });
}

fn main() {
    let mut c = Criterion::default().configure_from_args();

    bench_derivative(&mut c);
    bench_jacobian_small(&mut c);
    bench_jacobian_large(&mut c);
    bench_gauss_quad(&mut c);
    bench_lu_solve(&mut c);
    bench_svd_solve(&mut c);
    bench_expm(&mut c);
    bench_rk45_solve(&mut c);
    bench_rk4_integrate(&mut c);
    bench_lev_marq(&mut c);
    bench_newton_system(&mut c);

    c.final_summary();

    render_latency_md();
}

// =================== latency.md rendering ===================

/// (bench id, equation) — decides which estimates.json files are read and the table rows.
const BENCHES: &[(&str, &str)] = &[
    ("derivative", "d³/dx³(x²·sin x) at x = 1"),
    ("jacobian_small", "Jacobian of (x·y·z, x²+y²)"),
    ("jacobian_large", "Jacobian of a 6-in/6-out map"),
    ("gauss_quad", "∫₀¹ (sin x − √x)·e⁻ˣ dx"),
    ("lu_solve", "solve A·x = b (10×10)"),
    ("svd_solve", "least-squares fit (30×3)"),
    ("expm", "matrix exponential eᴬ (6×6)"),
    ("rk45_solve", "y″ = −y, adaptive to 2π"),
    ("rk4_integrate", "y″ = −y, fixed-step to 2π"),
    ("lev_marq", "fit y = a·eᵇᵗ (8 points)"),
    ("newton_system", "x²+y² = 4, x·y = 1"),
];

const BEGIN: &str = "<!-- BEGIN generated: latency -->";
const END: &str = "<!-- END generated -->";

/// Locate <target>/criterion: honor $CARGO_TARGET_DIR, else workspace target at repo root.
fn criterion_dir() -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(t) => PathBuf::from(t).join("criterion"),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion"),
    }
}

/// Read median & mean point estimates (ns) for one bench id from its estimates.json.
fn read_estimate(dir: &std::path::Path, id: &str) -> Option<(f64, f64)> {
    let path = dir.join(id).join("new").join("estimates.json");
    let text = std::fs::read_to_string(path).ok()?;
    let j: serde_json::Value = serde_json::from_str(&text).ok()?;
    let median = j["median"]["point_estimate"].as_f64()?;
    let mean = j["mean"]["point_estimate"].as_f64()?;
    Some((median, mean))
}

/// Human-readable ns → "123.4 ns" / "12.3 µs" / "1.23 ms".
fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else {
        format!("{:.2} ms", ns / 1_000_000.0)
    }
}

fn render_latency_md() {
    use std::fmt::Write as _;

    let dir = criterion_dir();
    let mut table = String::from("| Operation | Equation | Median | Mean |\n");
    table.push_str("|-----------|----------|-------:|-----:|\n");
    for (id, equation) in BENCHES {
        let cell = match read_estimate(&dir, id) {
            Some((median, mean)) => format!("{} | {}", fmt_ns(median), fmt_ns(mean)),
            None => "n/a | n/a".to_string(),
        };
        let _ = writeln!(table, "| `{id}` | {equation} | {cell} |");
    }

    let doc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/latency.md");
    let content =
        std::fs::read_to_string(&doc).unwrap_or_else(|e| unreachable!("read {doc:?}: {e}"));
    let begin = content
        .find(BEGIN)
        .unwrap_or_else(|| unreachable!("no BEGIN in {doc:?}"));
    let end = content
        .find(END)
        .unwrap_or_else(|| unreachable!("no END in {doc:?}"));
    let new = format!(
        "{}\n{}\n{}",
        &content[..begin + BEGIN.len()],
        table,
        &content[end..]
    );
    std::fs::write(&doc, new).unwrap_or_else(|e| unreachable!("write {doc:?}: {e}"));
    println!("updated {}", doc.display());
}
