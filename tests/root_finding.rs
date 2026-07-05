use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::finite_difference::FiniteDifferenceSingle;
use multicalc::root_finding::{Bisection, Newton, NewtonSystem, RootTermination};
use multicalc::scalar::{Numeric, VectorFn, c};
use multicalc::scalar_fn;
use multicalc::scalar_fn_vec;
use multicalc::utils::error_codes::CalcError;

// ----- Bisection -----

#[test]
fn bisection_sqrt2() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let report = Bisection::default().solve(&f, 0.0_f64, 2.0).unwrap();
    assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-9);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance | RootTermination::BracketWidth
    ));
}

#[test]
fn bisection_dottie_number() {
    // cos(x) = x, the fixed point of cosine ≈ 0.7390851332151607.
    let f = scalar_fn!(|x| x.cos() - x);
    let report = Bisection::default().solve(&f, 0.0_f64, 1.0).unwrap();
    assert!((report.root - 0.7390851332151607).abs() < 1e-9);
}

#[test]
fn bisection_wien_displacement() {
    // Wien's displacement law: x - 5 + 5*e^(-x) = 0, constant ≈ 4.965114231744276.
    let f = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = Bisection::default().solve(&f, 1.0_f64, 10.0).unwrap();
    assert!((report.root - 4.965114231744276).abs() < 1e-9);
}

#[test]
fn bisection_invalid_bracket() {
    // x² - 2 on [2, 3]: both values positive, no sign change.
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let result = Bisection::default().solve(&f, 2.0_f64, 3.0);
    assert!(matches!(result, Err(CalcError::InvalidBracket)));
}

#[test]
fn bisection_non_finite() {
    // 1/x on [-1, 1]: f(-1) and f(1) have opposite signs, but f(0) = +∞.
    let f = scalar_fn!(|x| c(1.0) / x);
    let result = Bisection::default().solve(&f, -1.0_f64, 1.0);
    assert!(matches!(result, Err(CalcError::NonFiniteValue)));
}

#[test]
fn bisection_budget_exhausted() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let result = Bisection::default()
        .with_max_iterations(2)
        .solve(&f, 0.0_f64, 2.0);
    assert!(matches!(result, Err(CalcError::DidNotConverge)));
}

#[test]
fn bisection_exact_endpoint_root() {
    // f(0) = 0 exactly; the solver returns before the first iteration.
    let f = scalar_fn!(|x| x);
    let report = Bisection::default().solve(&f, 0.0_f64, 1.0).unwrap();
    assert_eq!(report.root, 0.0_f64);
    assert!(matches!(report.termination, RootTermination::ResidualTolerance));
}

// ----- Scalar Newton and damped Newton -----

#[test]
fn newton_sqrt2() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let solver: Newton = Newton::default();
    let report = solver.solve(&f, 2.0_f64).unwrap();
    assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-12);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance | RootTermination::StepTolerance
    ));
}

#[test]
fn newton_cbrt2() {
    // x³ - 2 = 0, root at 2^(1/3) ≈ 1.2599210498948732.
    let f = scalar_fn!(|x| c(-2.0) + x.powi(3));
    let solver: Newton = Newton::default();
    let report = solver.solve(&f, 1.0_f64).unwrap();
    assert!((report.root - 2.0_f64.powf(1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn newton_wien_displacement() {
    let f = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let solver: Newton = Newton::default();
    let report = solver.solve(&f, 5.0_f64).unwrap();
    assert!((report.root - 4.965114231744276).abs() < 1e-12);
}

#[test]
fn newton_finite_difference_backend() {
    // Any DerivatorSingleVariable works in place of the autodiff default.
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let solver = Newton::from_derivator(FiniteDifferenceSingle::<f64>::default());
    let report = solver.solve(&f, 2.0_f64).unwrap();
    assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-6);
}

#[test]
fn newton_vanishing_derivative() {
    // f'(0) = 0 for x² - 2, so the first step is undefined.
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let solver: Newton = Newton::default();
    let result = solver.solve(&f, 0.0_f64);
    assert!(matches!(result, Err(CalcError::SingularMatrix)));
}

#[test]
fn newton_budget_exhausted() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let solver: Newton = Newton::default().with_max_iterations(1);
    // One step from x0=2 is not enough to satisfy either tolerance.
    let result = solver.solve(&f, 2.0_f64);
    assert!(matches!(result, Err(CalcError::DidNotConverge)));
}

#[test]
fn newton_damped_rescues_far_start() {
    // f(x) = x / sqrt(1 + x²), root at 0. The Newton map is x → −x³, so from x0 = 2.0
    // plain Newton diverges immediately. Backtracking halves the step until |f| decreases,
    // which is enough to land back in the basin of the root.
    let f = scalar_fn!(|x| x / (c(1.0) + x * x).sqrt());

    let plain: Newton = Newton::default();
    let plain_result = plain.solve(&f, 2.0_f64);
    let plain_missed = match &plain_result {
        Ok(r) => r.root.abs() > 0.1,
        Err(_) => true,
    };
    assert!(
        plain_missed,
        "plain Newton unexpectedly converged: {plain_result:?}"
    );

    let damped: Newton = Newton::default().with_backtracking(true);
    let report = damped.solve(&f, 2.0_f64).unwrap();
    assert!(report.root.abs() < 1e-6, "{report:?}");
}

// ----- Vector Newton and damped Newton -----

// Two-link planar arm forward kinematics. Holds the link lengths and target tip position.
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

#[test]
fn newton_system_circle_hyperbola() {
    // x² + y² = 4 and xy = 1; root near (1.932, 0.518).
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-4.0) + v[0] * v[0] + v[1] * v[1],
        c(-1.0) + v[0] * v[1],
    ]);
    let solver: NewtonSystem = NewtonSystem::default();
    let report = solver.solve(&f, &[1.5_f64, 0.8]).unwrap();
    assert!(report.residual_norm < 1e-12);
    let [x, y] = report.root;
    assert!((x * x + y * y - 4.0).abs() < 1e-12);
    assert!((x * y - 1.0).abs() < 1e-12);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance | RootTermination::StepTolerance
    ));
}

#[test]
fn newton_system_two_link_ik() {
    // Two-link arm (l1 = l2 = 1) with true joint angles (0.5 rad, 0.8 rad).
    // Tip position is computed from the truth; the solver recovers the angles from a near start.
    let (l1, l2) = (1.0_f64, 1.0_f64);
    let (t1_true, t2_true) = (0.5_f64, 0.8_f64);
    let px = l1 * t1_true.cos() + l2 * (t1_true + t2_true).cos();
    let py = l1 * t1_true.sin() + l2 * (t1_true + t2_true).sin();

    let arm = TwoLinkArm { l1, l2, px, py };
    let solver: NewtonSystem = NewtonSystem::default();
    let report = solver.solve(&arm, &[0.4_f64, 0.9]).unwrap();
    assert!(report.residual_norm < 1e-12, "{report:?}");
    let [t1, t2] = report.root;
    assert!((t1 - t1_true).abs() < 1e-10, "theta1: got {t1}, want {t1_true}");
    assert!((t2 - t2_true).abs() < 1e-10, "theta2: got {t2}, want {t2_true}");
}

#[test]
fn newton_system_singular_jacobian() {
    // The two equations are proportional, so the Jacobian is rank-deficient.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[0] + c(-1.0) * v[1],
        c(-2.0) + c(2.0) * v[0] + c(-2.0) * v[1],
    ]);
    let solver: NewtonSystem = NewtonSystem::default();
    let result = solver.solve(&f, &[0.0_f64, 0.0]);
    assert!(matches!(result, Err(CalcError::SingularMatrix)));
}

#[test]
fn newton_system_non_finite() {
    // First component is 1/v[0], which is infinite at the starting point.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [c(1.0) / v[0], v[1]]);
    let solver: NewtonSystem = NewtonSystem::default();
    let result = solver.solve(&f, &[0.0_f64, 0.0]);
    assert!(matches!(result, Err(CalcError::NonFiniteValue)));
}

#[test]
fn newton_system_budget_exhausted() {
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-4.0) + v[0] * v[0] + v[1] * v[1],
        c(-1.0) + v[0] * v[1],
    ]);
    let solver: NewtonSystem = NewtonSystem::default().with_max_iterations(1);
    let result = solver.solve(&f, &[1.5_f64, 0.8]);
    assert!(matches!(result, Err(CalcError::DidNotConverge)));
}

#[test]
fn newton_system_damped_rescues_far_start() {
    // F(v) = [v[0]/sqrt(1+v[0]²), v[1]/sqrt(1+v[1]²)], root at (0, 0).
    // Each component has the Newton map x → −x³, so from (3, 3) the plain solver
    // diverges. Backtracking halves the step length until ‖F‖ decreases.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        v[0] / (c(1.0) + v[0] * v[0]).sqrt(),
        v[1] / (c(1.0) + v[1] * v[1]).sqrt(),
    ]);
    let far = [3.0_f64, 3.0];

    let plain: NewtonSystem = NewtonSystem::default();
    let plain_result = plain.solve(&f, &far);
    let plain_missed = match &plain_result {
        Ok(r) => r.residual_norm > 0.1,
        Err(_) => true,
    };
    assert!(
        plain_missed,
        "plain NewtonSystem unexpectedly converged: {plain_result:?}"
    );

    let damped: NewtonSystem = NewtonSystem::default().with_backtracking(true);
    let report = damped.solve(&f, &far).unwrap();
    assert!(report.residual_norm < 1e-10, "{report:?}");
}

// ----- f32 coverage -----

#[test]
fn newton_sqrt2_f32() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let solver = Newton::<AutoDiffSingle<f32>>::default();
    let report = solver.solve(&f, 2.0_f32).unwrap();
    assert!((report.root - 2.0_f32.sqrt()).abs() < 1e-3);
}

#[test]
fn newton_system_circle_hyperbola_f32() {
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-4.0) + v[0] * v[0] + v[1] * v[1],
        c(-1.0) + v[0] * v[1],
    ]);
    let solver = NewtonSystem::<AutoDiffMulti<f32>>::default();
    let report = solver.solve(&f, &[1.5_f32, 0.8]).unwrap();
    assert!(report.residual_norm < 1e-3);
}
