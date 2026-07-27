#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::error::{LinalgError, SolveError};
use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::numerical_derivative::FiniteDifferenceMulti;
use multicalc::numerical_derivative::Jacobian;
use multicalc::optimization::{
    GaussNewton, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use multicalc::scalar::{Numeric, VectorFn, c};
use multicalc::scalar_fn_vec;
use multicalc_testkit::problems::Rosenbrock;

// ----- Levenberg-Marquardt solver -----

#[test]
fn lm_solves_rosenbrock() {
    let rosenbrock = Rosenbrock;
    let report: MinimizationReport<2> = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&rosenbrock, &[-1.2, 1.0])
        .unwrap();
    assert!((report.solution[0] - 1.0).abs() < 1e-8);
    assert!((report.solution[1] - 1.0).abs() < 1e-8);
    assert!(report.objective_function < 1e-16);
}

#[test]
fn lm_recovers_linear_least_squares() {
    // Fit a*t + b to points lying exactly on y = 2t + 1.
    let linear_residual = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[1],
        c(-3.0) + v[0] + v[1],
        c(-5.0) + c(2.0) * v[0] + v[1],
    ]);
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&linear_residual, &[0.0, 0.0])
        .unwrap();
    assert!((report.solution[0] - 2.0).abs() < 1e-12);
    assert!((report.solution[1] - 1.0).abs() < 1e-12);
    assert!(report.objective_function < 1e-16);
}

#[test]
fn lm_fits_exponential_decay() {
    // a*e^(b*t) through (0,100), (1,50), (2,25): a = 100, b = -ln 2.
    let exponential_decay = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-100.0) + v[0],
        c(-50.0) + v[0] * v[1].exp(),
        c(-25.0) + v[0] * (c(2.0) * v[1]).exp(),
    ]);
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&exponential_decay, &[80.0, -0.3])
        .unwrap();
    assert!((report.solution[0] - 100.0).abs() < 1e-7);
    assert!((report.solution[1] + 2.0_f64.ln()).abs() < 1e-8);
    assert!(report.objective_function < 1e-12);
    assert!(matches!(
        report.termination,
        TerminationReason::Ftol | TerminationReason::Xtol | TerminationReason::Gtol
    ));
}

#[test]
fn lm_solves_rosenbrock_f32() {
    // One residual definition drives both precisions; eval is generic over the scalar.
    let rosenbrock = Rosenbrock;
    let report = LevenbergMarquardt::<AutoDiffMulti<f32>>::default()
        .minimize(&rosenbrock, &[-1.2_f32, 1.0])
        .unwrap();
    assert!((report.solution[0] - 1.0).abs() < 1e-3);
    assert!((report.solution[1] - 1.0).abs() < 1e-3);
}

#[test]
fn lm_rejects_underdetermined() {
    // Two residuals, three parameters.
    let underdetermined = scalar_fn_vec!(|v: &[f64; 3]| [c(-1.0) + v[0] + v[1], c(-2.0) + v[2]]);
    let result =
        LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&underdetermined, &[0.0, 0.0, 0.0]);
    assert!(matches!(
        result,
        Err(SolveError::Linalg(LinalgError::Underdetermined))
    ));
}

#[test]
fn lm_reports_non_finite() {
    // The residual is infinite at the starting point.
    let residual = scalar_fn_vec!(|v: &[f64; 1]| [c(1.0) / v[0], v[0]]);
    let result = LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&residual, &[0.0]);
    assert!(matches!(result, Err(SolveError::NonFinite)));
}

#[test]
fn lm_reports_did_not_converge() {
    // A one-iteration budget is too small for Rosenbrock, so the solver runs out.
    let rosenbrock = Rosenbrock;
    let result = LevenbergMarquardt::<AutoDiffMulti>::default()
        .with_patience(1)
        .minimize(&rosenbrock, &[-1.2, 1.0]);
    assert!(matches!(result, Err(SolveError::DidNotConverge { .. })));
}

// Sum of three damped sinusoids sampled at 60 points, with parameters [A, lambda, omega, phi]
// per component. Holds the sample times and targets so the residual is model - target.
struct DampedSinusoids {
    sample_times: [f64; 60],
    targets: [f64; 60],
}

impl VectorFn<12, 60> for DampedSinusoids {
    fn eval<S: Numeric>(&self, parameters: &[S; 12]) -> [S; 60] {
        core::array::from_fn(|sample_index| {
            let time = S::from_f64(self.sample_times[sample_index]);
            let mut model = S::ZERO;
            for component in 0..3 {
                let amplitude = parameters[4 * component];
                let decay_rate = parameters[4 * component + 1];
                let frequency = parameters[4 * component + 2];
                let phase = parameters[4 * component + 3];
                model +=
                    amplitude * (-(decay_rate * time)).exp() * (frequency * time + phase).sin();
            }
            model - S::from_f64(self.targets[sample_index])
        })
    }
}

#[test]
fn lm_fits_damped_sinusoids() {
    // Three well-separated components (distinct amplitudes, decays, frequencies, phases).
    let truth = [
        1.0, 0.5, 2.0, 0.3, // A, lambda, omega, phi
        0.7, 0.2, 5.0, 1.1, //
        1.3, 0.8, 8.5, -0.5, //
    ];
    let sample_times: [f64; 60] = core::array::from_fn(|index| index as f64 * 6.0 / 59.0);
    let mut problem = DampedSinusoids {
        sample_times,
        targets: [0.0; 60],
    };
    // Generate noiseless targets: with y = 0 the residual is exactly the model.
    problem.targets = problem.eval(&truth);

    // Start a modest step away (frequencies kept close, the rest looser).
    let start = [
        1.15, 0.55, 2.05, 0.2, //
        0.6, 0.18, 5.08, 1.2, //
        1.45, 0.72, 8.42, -0.65, //
    ];
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&problem, &start)
        .unwrap();

    // The fit reaches machine precision and every one of the 12 parameters is recovered.
    assert!(report.objective_function < 1e-16);
    for (got, want) in report.solution.iter().zip(truth.iter()) {
        assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
    }
}

// The Moré-Garbow-Hillstrom trigonometric function (problem 26) in N variables. Its global
// minimum is zero.
struct Trigonometric<const N: usize>;

impl<const N: usize> VectorFn<N, N> for Trigonometric<N> {
    fn eval<S: Numeric>(&self, point: &[S; N]) -> [S; N] {
        let dimension = S::from_f64(N as f64);
        let mut cos_sum = S::ZERO;
        for &component in point {
            cos_sum += component.cos();
        }
        core::array::from_fn(|index| {
            dimension - cos_sum + S::from_f64((index + 1) as f64) * (S::ONE - point[index].cos())
                - point[index].sin()
        })
    }
}

#[test]
fn lm_solves_trigonometric() {
    // From the standard start x_j = 1/n, the solver drives the 6-variable trigonometric
    // function all the way to its global minimum of zero (to machine precision).
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&Trigonometric::<6>, &[1.0 / 6.0; 6])
        .unwrap();
    assert!(
        report.objective_function < 1e-16,
        "objective {}",
        report.objective_function
    );
}

// ----- Gauss-Newton solver -----

#[test]
fn gn_recovers_linear_least_squares() {
    // A linear residual: Gauss-Newton reaches the exact least-squares solution.
    let linear_residual = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[1],
        c(-3.0) + v[0] + v[1],
        c(-5.0) + c(2.0) * v[0] + v[1],
    ]);
    let report = GaussNewton::<AutoDiffMulti>::default()
        .minimize(&linear_residual, &[0.0, 0.0])
        .unwrap();
    assert!((report.solution[0] - 2.0).abs() < 1e-12);
    assert!((report.solution[1] - 1.0).abs() < 1e-12);
    assert!(report.objective_function < 1e-16);
}

#[test]
fn gn_solves_rosenbrock() {
    // From a near guess, Gauss-Newton converges quadratically on this zero-residual problem.
    let rosenbrock = Rosenbrock;
    let report = GaussNewton::<AutoDiffMulti>::default()
        .minimize(&rosenbrock, &[0.9, 0.9])
        .unwrap();
    assert!((report.solution[0] - 1.0).abs() < 1e-9);
    assert!((report.solution[1] - 1.0).abs() < 1e-9);
    assert!(report.objective_function < 1e-16);
}

#[test]
fn gn_reports_singular() {
    // The two residuals are proportional, so the Jacobian is rank-deficient.
    let singular_residual = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[0] - v[1],
        c(-2.0) + c(2.0) * v[0] - c(2.0) * v[1],
    ]);
    let result = GaussNewton::<AutoDiffMulti>::default().minimize(&singular_residual, &[0.0, 0.0]);
    assert!(matches!(
        result,
        Err(SolveError::Linalg(LinalgError::Singular))
    ));
}

#[test]
fn gn_rejects_underdetermined() {
    let underdetermined = scalar_fn_vec!(|v: &[f64; 3]| [c(-1.0) + v[0] + v[1], c(-2.0) + v[2]]);
    let result =
        GaussNewton::<AutoDiffMulti>::default().minimize(&underdetermined, &[0.0, 0.0, 0.0]);
    assert!(matches!(
        result,
        Err(SolveError::Linalg(LinalgError::Underdetermined))
    ));
}

#[test]
fn gn_reports_non_finite() {
    let residual = scalar_fn_vec!(|v: &[f64; 1]| [c(1.0) / v[0], v[0]]);
    let result = GaussNewton::<AutoDiffMulti>::default().minimize(&residual, &[0.0]);
    assert!(matches!(result, Err(SolveError::NonFinite)));
}

// Fit a circle (center cx, cy, radius r) to points, minimizing the geometric distance residual
// sqrt((x-cx)^2 + (y-cy)^2) - r. Holds the measured points.
struct CircleFit {
    measured_x: [f64; 40],
    measured_y: [f64; 40],
}

impl VectorFn<3, 40> for CircleFit {
    fn eval<S: Numeric>(&self, parameters: &[S; 3]) -> [S; 40] {
        let centre_x = parameters[0];
        let centre_y = parameters[1];
        let radius = parameters[2];
        core::array::from_fn(|index| {
            let offset_x = S::from_f64(self.measured_x[index]) - centre_x;
            let offset_y = S::from_f64(self.measured_y[index]) - centre_y;
            (offset_x * offset_x + offset_y * offset_y).sqrt() - radius
        })
    }
}

#[test]
fn gn_fits_circle() {
    // Points sampled exactly on a circle of center (2, -1), radius 3.
    let angle = |index: usize| core::f64::consts::TAU * index as f64 / 40.0;
    let measured_x = core::array::from_fn(|index| 2.0 + 3.0 * angle(index).cos());
    let measured_y = core::array::from_fn(|index| -1.0 + 3.0 * angle(index).sin());
    let problem = CircleFit {
        measured_x,
        measured_y,
    };

    // A near guess for the geometry.
    let report = GaussNewton::<AutoDiffMulti>::default()
        .minimize(&problem, &[2.4, -0.6, 3.5])
        .unwrap();
    assert!((report.solution[0] - 2.0).abs() < 1e-9);
    assert!((report.solution[1] + 1.0).abs() < 1e-9);
    assert!((report.solution[2] - 3.0).abs() < 1e-9);
    assert!(report.objective_function < 1e-16);
}

// Fit two Gaussian peaks [a, mu, sigma] each to a sampled spectrum.
struct GaussianPeaks {
    sample_times: [f64; 50],
    targets: [f64; 50],
}

impl VectorFn<6, 50> for GaussianPeaks {
    fn eval<S: Numeric>(&self, parameters: &[S; 6]) -> [S; 50] {
        core::array::from_fn(|sample_index| {
            let time = S::from_f64(self.sample_times[sample_index]);
            let mut model = S::ZERO;
            for peak in 0..2 {
                let amplitude = parameters[3 * peak];
                let centre = parameters[3 * peak + 1];
                let width = parameters[3 * peak + 2];
                let standardized = (time - centre) / width;
                model += amplitude * (-(standardized * standardized)).exp();
            }
            model - S::from_f64(self.targets[sample_index])
        })
    }
}

#[test]
fn gn_fits_gaussian_peaks() {
    // Two well-separated peaks: a=2 at mu=3 (sigma 0.8), a=1.5 at mu=7 (sigma 1.2).
    let truth = [2.0, 3.0, 0.8, 1.5, 7.0, 1.2];
    let sample_times: [f64; 50] = core::array::from_fn(|index| index as f64 * 10.0 / 49.0);
    let mut problem = GaussianPeaks {
        sample_times,
        targets: [0.0; 50],
    };
    problem.targets = problem.eval(&truth);

    // A near start recovers all six parameters.
    let start = [2.2, 3.2, 0.7, 1.3, 6.8, 1.3];
    let report = GaussNewton::<AutoDiffMulti>::default()
        .minimize(&problem, &start)
        .unwrap();
    assert!(report.objective_function < 1e-16);
    for (got, want) in report.solution.iter().zip(truth.iter()) {
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }
}

#[test]
fn gn_backtracking_rescues_far_start() {
    // r(x) = x / sqrt(1 + x^2): a bounded sigmoid whose Gauss-Newton step map is x -> -x^3.
    // The minimum is x = 0, but past |x| = 1 the full step cubes and flings the iterate off to
    // the saturated tail (|r| -> 1). Backtracking halves the overshoot until the residual drops,
    // landing back in the basin.
    let residual = || scalar_fn_vec!(|v: &[f64; 1]| [v[0] / (c(1.0) + v[0] * v[0]).sqrt()]);
    let far = [2.0];

    // Plain Gauss-Newton overshoots: it either overflows or stalls on the tail far from x = 0.
    let plain = GaussNewton::<AutoDiffMulti>::default().minimize(&residual(), &far);
    let plain_missed = match &plain {
        Ok(report) => report.objective_function > 0.4,
        Err(_) => true,
    };
    assert!(
        plain_missed,
        "plain GN unexpectedly reached the minimum: {plain:?}"
    );

    // Backtracking rescues the same start and converges to x = 0.
    let guarded = GaussNewton::<AutoDiffMulti>::default()
        .with_backtracking(true)
        .minimize(&residual(), &far)
        .unwrap();
    assert!(guarded.objective_function < 1e-12, "{guarded:?}");
    assert!(guarded.solution[0].abs() < 1e-6, "{guarded:?}");
}

// ----- Embedded artifact regression -----

struct SensorFit<const M: usize> {
    sample_times: [f64; M],
    targets: [f64; M],
}

impl<const M: usize> VectorFn<2, M> for SensorFit<M> {
    fn eval<S: Numeric>(&self, parameters: &[S; 2]) -> [S; M] {
        let amplitude = parameters[0];
        let decay_rate = parameters[1];
        core::array::from_fn(|sample_index| {
            amplitude * (decay_rate * S::from_f64(self.sample_times[sample_index])).exp()
                - S::from_f64(self.targets[sample_index])
        })
    }
}

#[test]
fn embedded_artifact_fit_recovers_parameters() {
    let sample_times: [f64; 8] = core::array::from_fn(|index| index as f64);
    let targets: [f64; 8] = core::array::from_fn(|index| 100.0 * (-0.5 * index as f64).exp());
    let problem = SensorFit {
        sample_times,
        targets,
    };

    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .with_patience(50)
        .minimize(&problem, &[80.0, -0.3])
        .unwrap();

    assert!((report.solution[0] - 100.0).abs() < 1e-9);
    assert!((report.solution[1] + 0.5).abs() < 1e-9);
    assert!(report.objective_function < 1e-12);
}

// ----- Residual / Jacobian API -----

// Largest entrywise gap between the exact autodiff Jacobian and a central finite-difference
// Jacobian of `function` at `point`. A small value confirms the autodiff derivatives the solvers
// rely on match an independent finite-difference estimate.
fn check_jacobian<F: VectorFn<N, M>, const N: usize, const M: usize>(
    function: &F,
    point: &[f64; N],
) -> f64 {
    let autodiff = Jacobian::<AutoDiffMulti>::default()
        .evaluate(function, point)
        .unwrap();
    let finite_difference = Jacobian::from_derivator(FiniteDifferenceMulti::<f64>::default())
        .evaluate(function, point)
        .unwrap();
    let mut worst = 0.0_f64;
    for row in 0..M {
        for column in 0..N {
            worst = worst.max((autodiff[(row, column)] - finite_difference[(row, column)]).abs());
        }
    }
    worst
}

#[test]
fn autodiff_jacobian_matches_finite_differences() {
    // Rosenbrock residual: a low-degree polynomial, so central differences are near exact.
    let rosenbrock = Rosenbrock;
    assert!(check_jacobian(&rosenbrock, &[-1.2, 1.0]) < 1e-6);

    // A transcendental residual exercises the sin and exp derivatives.
    let mixed = scalar_fn_vec!(|v: &[f64; 2]| [v[0].sin() * v[1], c(2.0) * v[0] + v[1].exp()]);
    assert!(check_jacobian(&mixed, &[0.7, -0.4]) < 1e-6);
}

#[test]
fn solvers_accept_a_finite_difference_backend() {
    // Swap the autodiff default for a finite-difference Jacobian; both solvers still converge on
    // the zero-residual Rosenbrock problem.
    let rosenbrock = Rosenbrock;

    let levenberg_marquardt =
        LevenbergMarquardt::from_derivator(FiniteDifferenceMulti::<f64>::default())
            .minimize(&rosenbrock, &[-1.2, 1.0])
            .unwrap();
    assert!(
        (levenberg_marquardt.solution[0] - 1.0).abs() < 1e-5,
        "{levenberg_marquardt:?}"
    );
    assert!(
        (levenberg_marquardt.solution[1] - 1.0).abs() < 1e-5,
        "{levenberg_marquardt:?}"
    );

    let gauss_newton = GaussNewton::from_derivator(FiniteDifferenceMulti::<f64>::default())
        .minimize(&rosenbrock, &[0.9, 0.9])
        .unwrap();
    assert!(
        (gauss_newton.solution[0] - 1.0).abs() < 1e-5,
        "{gauss_newton:?}"
    );
    assert!(
        (gauss_newton.solution[1] - 1.0).abs() < 1e-5,
        "{gauss_newton:?}"
    );
}
