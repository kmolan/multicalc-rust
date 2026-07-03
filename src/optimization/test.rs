use crate::linear_algebra::{Matrix, PivotedQr, Vector};
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::optimization::trust_region::determine_lambda_and_parameter_update;
use crate::optimization::{LevenbergMarquardt, MinimizationReport, TerminationReason};
use crate::scalar::{Numeric, VectorFn, c};
use crate::scalar_fn_vec;
use crate::utils::error_codes::CalcError;

// ----- LMPAR trust-region parameter -----

// A full-rank 4x3 Jacobian and residual for the trust-region tests.
fn sample_jacobian() -> (Matrix<4, 3>, Vector<4>) {
    let j = Matrix::<4, 3>::new([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 1.0, 1.0],
        [1.0, 0.0, 2.0],
    ]);
    let residual = Vector::new([1.0, 2.0, 3.0, 4.0]);
    (j, residual)
}

#[test]
fn lmpar_accepts_gauss_newton_inside_region() {
    let (j, b) = sample_jacobian();
    let diag = [1.0, 1.0, 1.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);

    // A trust region larger than the Gauss-Newton step keeps the step undamped.
    let result = determine_lambda_and_parameter_update(&dls, &diag, 100.0, 0.0);
    assert_eq!(result.lambda, 0.0);

    let (gn, _) = dls.solve_with_zero_diagonal();
    for i in 0..3 {
        assert!((result.step[i] - gn[i]).abs() < 1e-12);
    }
}

#[test]
fn lmpar_hits_trust_region_boundary() {
    let (j, b) = sample_jacobian();
    let diag = [1.0, 1.0, 1.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);

    // Shrink the region below the Gauss-Newton length so damping is required.
    let (gn, _) = dls.solve_with_zero_diagonal();
    let delta = 0.5 * gn.norm();
    let result = determine_lambda_and_parameter_update(&dls, &diag, delta, 0.0);

    // Damping is positive and the step lands within 10% of the boundary.
    assert!(result.lambda > 0.0);
    let step_norm = result.step.norm();
    assert!((step_norm - delta).abs() <= 0.1 * delta);

    // The step solves the damped normal equations (JᵀJ + λI) p = Jᵀb (D = I here).
    let jtj = j.transpose() * j;
    let jtb = j.transpose() * b;
    let lhs = Matrix::<3, 3>::from_fn(|r, c| {
        jtj[(r, c)] + if r == c { result.lambda } else { 0.0 }
    }) * result.step;
    for i in 0..3 {
        assert!((lhs[i] - jtb[i]).abs() < 1e-10);
    }
}

#[test]
fn lmpar_stronger_damping_shortens_step() {
    let (j, b) = sample_jacobian();
    let diag = [1.0, 1.0, 1.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let (gn, _) = dls.solve_with_zero_diagonal();

    // A tighter trust region yields a larger λ and a shorter step.
    let loose = determine_lambda_and_parameter_update(&dls, &diag, 0.6 * gn.norm(), 0.0);
    let tight = determine_lambda_and_parameter_update(&dls, &diag, 0.3 * gn.norm(), 0.0);
    assert!(tight.lambda > loose.lambda);
    assert!(tight.step.norm() < loose.step.norm());
}

// ----- Levenberg-Marquardt solver -----

#[test]
fn lm_solves_rosenbrock() {
    let f = scalar_fn_vec!(|v: &[f64; 2]| [c(10.0) * (v[1] - v[0] * v[0]), c(1.0) - v[0]]);
    let report: MinimizationReport<2> = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&f, &[-1.2, 1.0])
        .unwrap();
    assert!((report.solution[0] - 1.0).abs() < 1e-6);
    assert!((report.solution[1] - 1.0).abs() < 1e-6);
    assert!(report.objective_function < 1e-10);
}

#[test]
fn lm_recovers_linear_least_squares() {
    // Fit a*t + b to points lying exactly on y = 2t + 1.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[1],
        c(-3.0) + v[0] + v[1],
        c(-5.0) + c(2.0) * v[0] + v[1],
    ]);
    let report = LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&f, &[0.0, 0.0]).unwrap();
    assert!((report.solution[0] - 2.0).abs() < 1e-9);
    assert!((report.solution[1] - 1.0).abs() < 1e-9);
}

#[test]
fn lm_fits_exponential_decay() {
    // a*e^(b*t) through (0,100), (1,50), (2,25): a = 100, b = -ln 2.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-100.0) + v[0],
        c(-50.0) + v[0] * v[1].exp(),
        c(-25.0) + v[0] * (c(2.0) * v[1]).exp(),
    ]);
    let report = LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&f, &[80.0, -0.3]).unwrap();
    assert!((report.solution[0] - 100.0).abs() < 1e-5);
    assert!((report.solution[1] + 2.0_f64.ln()).abs() < 1e-6);
    assert!(matches!(
        report.termination,
        TerminationReason::Ftol | TerminationReason::Xtol | TerminationReason::Gtol
    ));
}

#[test]
fn lm_solves_rosenbrock_f32() {
    // One residual definition drives both precisions; eval is generic over the scalar.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [c(10.0) * (v[1] - v[0] * v[0]), c(1.0) - v[0]]);
    let report = LevenbergMarquardt::<AutoDiffMulti<f32>>::default()
        .minimize(&f, &[-1.2_f32, 1.0])
        .unwrap();
    assert!((report.solution[0] - 1.0).abs() < 1e-3);
    assert!((report.solution[1] - 1.0).abs() < 1e-3);
}

#[test]
fn lm_rejects_underdetermined() {
    // Two residuals, three parameters.
    let f = scalar_fn_vec!(|v: &[f64; 3]| [c(-1.0) + v[0] + v[1], c(-2.0) + v[2]]);
    let result = LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&f, &[0.0, 0.0, 0.0]);
    assert!(matches!(result, Err(CalcError::Underdetermined)));
}

#[test]
fn lm_reports_non_finite() {
    // The residual is infinite at the starting point.
    let f = scalar_fn_vec!(|v: &[f64; 1]| [c(1.0) / v[0], v[0]]);
    let result = LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&f, &[0.0]);
    assert!(matches!(result, Err(CalcError::NonFiniteValue)));
}

// Sum of three damped sinusoids sampled at 60 points, with parameters [A, lambda, omega, phi]
// per component. Holds the sample times and targets so the residual is model - target.
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

#[test]
fn lm_fits_damped_sinusoids() {
    // Three well-separated components (distinct amplitudes, decays, frequencies, phases).
    let truth = [
        1.0, 0.5, 2.0, 0.3, // A, lambda, omega, phi
        0.7, 0.2, 5.0, 1.1, //
        1.3, 0.8, 8.5, -0.5, //
    ];
    let t: [f64; 60] = core::array::from_fn(|i| i as f64 * 6.0 / 59.0);
    let mut problem = DampedSinusoids { t, y: [0.0; 60] };
    // Generate noiseless targets: with y = 0 the residual is exactly the model.
    problem.y = problem.eval(&truth);

    // Start a modest step away (frequencies kept close, the rest looser).
    let start = [
        1.15, 0.55, 2.05, 0.2, //
        0.6, 0.18, 5.08, 1.2, //
        1.45, 0.72, 8.42, -0.65, //
    ];
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&problem, &start)
        .unwrap();

    // The fit is essentially perfect and every one of the 12 parameters is recovered.
    assert!(report.objective_function < 1e-12);
    for (got, want) in report.solution.iter().zip(truth.iter()) {
        assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
    }
}

// The Moré-Garbow-Hillstrom trigonometric function (problem 26) in N variables. Its global
// minimum is zero.
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

#[test]
fn lm_solves_trigonometric() {
    // From the standard start x_j = 1/n, the solver drives the 6-variable trigonometric
    // function all the way to its global minimum of zero (to machine precision).
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&Trigonometric::<6>, &[1.0 / 6.0; 6])
        .unwrap();
    assert!(
        report.objective_function < 1e-12,
        "objective {}",
        report.objective_function
    );
 }
