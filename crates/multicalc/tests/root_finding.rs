#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::finite_difference::FiniteDifferenceSingle;
use multicalc::root_finding::{
    Bisection, Newton, NewtonSystem, RootReport, RootReportN, RootTermination,
};
use multicalc::scalar::{Numeric, ScalarFn, VectorFn, c};
use multicalc::scalar_fn;
use multicalc::scalar_fn_vec;
use multicalc::error::{LinalgError, SolveError};

fn bisect<F: ScalarFn>(f: &F, a: f64, b: f64) -> Result<RootReport<f64>, SolveError> {
    Bisection::default().solve(f, a, b)
}

fn newton<F: ScalarFn>(f: &F, x0: f64) -> Result<RootReport<f64>, SolveError> {
    let s: Newton = Newton::default();
    s.solve(f, x0)
}

fn nsystem<F: VectorFn<2, 2>>(f: &F, x0: &[f64; 2]) -> Result<RootReportN<2>, SolveError> {
    let s: NewtonSystem = NewtonSystem::default();
    s.solve(f, x0)
}

// x² + y² = 4, xy = 1; roots near (±1.932, ±0.518).
struct CircleHyperbola;

impl VectorFn<2, 2> for CircleHyperbola {
    fn eval<S: Numeric>(&self, v: &[S; 2]) -> [S; 2] {
        [c(-4.0) + v[0] * v[0] + v[1] * v[1], c(-1.0) + v[0] * v[1]]
    }
}

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

// ----- Bisection -----

#[test]
fn bisection_sqrt2() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let report = bisect(&f, 0.0_f64, 2.0).unwrap();
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
    let report = bisect(&f, 0.0_f64, 1.0).unwrap();
    assert!((report.root - 0.7390851332151607).abs() < 1e-9);
}

#[test]
fn bisection_wien_displacement() {
    // Wien's displacement law: x - 5 + 5*e^(-x) = 0, constant ≈ 4.965114231744276.
    let f = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = bisect(&f, 1.0_f64, 10.0).unwrap();
    assert!((report.root - 4.965114231744276).abs() < 1e-9);
}

#[test]
fn bisection_invalid_bracket() {
    // x² - 2 on [2, 3]: both values positive, no sign change.
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    assert!(matches!(
        bisect(&f, 2.0_f64, 3.0),
        Err(SolveError::InvalidBracket)
    ));
}

#[test]
fn bisection_non_finite() {
    // 1/x on [-1, 1]: f(-1) and f(1) have opposite signs, but f(0) = +∞.
    let f = scalar_fn!(|x| c(1.0) / x);
    assert!(matches!(
        bisect(&f, -1.0_f64, 1.0),
        Err(SolveError::NonFinite)
    ));
}

#[test]
fn bisection_budget_exhausted() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let result = Bisection::default()
        .with_max_iterations(2)
        .solve(&f, 0.0_f64, 2.0);
    assert!(matches!(result, Err(SolveError::DidNotConverge { .. })));
}

#[test]
fn bisection_exact_endpoint_root() {
    // f(0) = 0 exactly; the solver returns before the first iteration.
    let f = scalar_fn!(|x| x);
    let report = bisect(&f, 0.0_f64, 1.0).unwrap();
    assert_eq!(report.root, 0.0_f64);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance
    ));
}

// ----- Scalar Newton and damped Newton -----

#[test]
fn newton_sqrt2() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let report = newton(&f, 2.0_f64).unwrap();
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
    let report = newton(&f, 1.0_f64).unwrap();
    assert!((report.root - 2.0_f64.powf(1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn newton_wien_displacement() {
    let f = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = newton(&f, 5.0_f64).unwrap();
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
    assert!(matches!(
        newton(&f, 0.0_f64),
        Err(SolveError::Linalg(LinalgError::Singular))
    ));
}

#[test]
fn newton_budget_exhausted() {
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let s: Newton = Newton::default().with_max_iterations(1);
    // One step from x0=2 is not enough to satisfy either tolerance.
    assert!(matches!(
        s.solve(&f, 2.0_f64),
        Err(SolveError::DidNotConverge { .. })
    ));
}

#[test]
fn newton_damped_rescues_far_start() {
    // f(x) = x / sqrt(1 + x²), root at 0. The Newton map is x → −x³, so from x0 = 2.0
    // plain Newton diverges immediately. Backtracking halves the step until |f| decreases,
    // which is enough to land back in the basin of the root.
    let f = scalar_fn!(|x| x / (c(1.0) + x * x).sqrt());

    let plain_result = newton(&f, 2.0_f64);
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

#[test]
fn newton_system_circle_hyperbola() {
    // x² + y² = 4 and xy = 1; root near (1.932, 0.518).
    let report = nsystem(&CircleHyperbola, &[1.5_f64, 0.8]).unwrap();
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
    let report = nsystem(&TwoLinkArm { l1, l2, px, py }, &[0.4_f64, 0.9]).unwrap();
    assert!(report.residual_norm < 1e-12, "{report:?}");
    let [t1, t2] = report.root;
    assert!(
        (t1 - t1_true).abs() < 1e-10,
        "theta1: got {t1}, want {t1_true}"
    );
    assert!(
        (t2 - t2_true).abs() < 1e-10,
        "theta2: got {t2}, want {t2_true}"
    );
}

#[test]
fn newton_system_singular_jacobian() {
    // The two equations are proportional, so the Jacobian is rank-deficient.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[0] + c(-1.0) * v[1],
        c(-2.0) + c(2.0) * v[0] + c(-2.0) * v[1],
    ]);
    assert!(matches!(
        nsystem(&f, &[0.0_f64, 0.0]),
        Err(SolveError::Linalg(LinalgError::Singular))
    ));
}

#[test]
fn newton_system_non_finite() {
    // First component is 1/v[0], which is infinite at the starting point.
    let f = scalar_fn_vec!(|v: &[f64; 2]| [c(1.0) / v[0], v[1]]);
    assert!(matches!(
        nsystem(&f, &[0.0_f64, 0.0]),
        Err(SolveError::NonFinite)
    ));
}

#[test]
fn newton_system_budget_exhausted() {
    let s: NewtonSystem = NewtonSystem::default().with_max_iterations(1);
    assert!(matches!(
        s.solve(&CircleHyperbola, &[1.5_f64, 0.8]),
        Err(SolveError::DidNotConverge { .. })
    ));
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

    let plain_result = nsystem(&f, &far);
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
    let solver = NewtonSystem::<AutoDiffMulti<f32>>::default();
    let report = solver.solve(&CircleHyperbola, &[1.5_f32, 0.8]).unwrap();
    assert!(report.residual_norm < 1e-3);
}

// ======================================================================
// Real-life stress tests
//
// Each solver runs on an engineering, physics, or finance equation. Every
// case has a known root: either a documented physical constant, or an
// input generated from a chosen answer so the solver recovers it (pick the
// root, compute the parameters, solve).
// ======================================================================

// Kepler's equation E - e*sin(E) = M, relating the mean anomaly M to the
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

#[test]
fn kepler_equation_moderate_eccentricity() {
    // Orbit with e = 0.8. Pick a true eccentric anomaly, form the mean
    // anomaly from it, then recover E by Newton starting at x0 = M.
    let e = 0.8_f64;
    let e_true = 1.0_f64;
    let m = e_true - e * e_true.sin();
    let report = newton(&Kepler { e, m }, m).unwrap();
    assert!((report.root - e_true).abs() < 1e-12, "{report:?}");
    // The solved equation holds: E - e*sin(E) == M.
    assert!((report.root - e * report.root.sin() - m).abs() < 1e-12);
}

#[test]
fn kepler_equation_high_eccentricity() {
    // e = 0.99 with M near zero makes 1 - e*cos(E) ~ 0.01 near the root, so
    // the Newton derivative is small and the plain step is fragile. Bisection
    // on [0, π] is guaranteed; damped Newton also recovers the root.
    let e = 0.99_f64;
    let e_true = 0.5_f64;
    let m = e_true - e * e_true.sin();
    let f = Kepler { e, m };

    let bracketed = bisect(&f, 0.0_f64, core::f64::consts::PI).unwrap();
    assert!((bracketed.root - e_true).abs() < 1e-9, "{bracketed:?}");

    let damped: Newton = Newton::default().with_backtracking(true);
    let stepped = damped.solve(&f, m).unwrap();
    assert!((stepped.root - e_true).abs() < 1e-9, "{stepped:?}");
}

// Colebrook–White equation for the Darcy friction factor f of turbulent
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

#[test]
fn colebrook_white_friction_factor() {
    // Water in a commercial-steel pipe: Re = 1e5, relative roughness 1e-4.
    let f = Colebrook {
        reynolds: 1.0e5,
        rel_roughness: 1.0e-4,
    };
    let report = newton(&f, 0.02_f64).unwrap();
    assert!(report.residual.abs() < 1e-10, "{report:?}");
    // Physical friction factors sit in this range.
    assert!(report.root > 0.01 && report.root < 0.05, "{report:?}");
}

// Bond pricing: the present value of the cash flows discounted at yield r
// equals the market price. Solving for r gives the yield to maturity.
struct BondYield {
    cashflows: [f64; 5],
    times: [i32; 5],
    price: f64,
}

impl ScalarFn for BondYield {
    fn eval<S: Numeric>(&self, r: S) -> S {
        let mut pv = S::ZERO;
        for (cash, t) in self.cashflows.iter().zip(self.times.iter()) {
            pv += S::from_f64(*cash) / (S::ONE + r).powi(*t);
        }
        pv - S::from_f64(self.price)
    }
}

#[test]
fn bond_internal_rate_of_return() {
    // Five-year bond, 5% annual coupon on 100 face value. Choose the true
    // yield, price the bond at it, then recover the yield by Newton.
    let cashflows = [5.0, 5.0, 5.0, 5.0, 105.0];
    let times = [1, 2, 3, 4, 5];
    let r_true = 0.04_f64;
    let price: f64 = cashflows
        .iter()
        .zip(times.iter())
        .map(|(cash, t)| cash / (1.0_f64 + r_true).powi(*t))
        .sum();
    let report = newton(
        &BondYield {
            cashflows,
            times,
            price,
        },
        0.1_f64,
    )
    .unwrap();
    assert!((report.root - r_true).abs() < 1e-10, "{report:?}");
}

// Catenary: a uniform cable of length `length` hung between supports a
// horizontal distance `span` apart satisfies length = 2a*sinh(span/(2a))
// for the catenary constant a.
struct Catenary {
    span: f64,
    length: f64,
}

impl ScalarFn for Catenary {
    fn eval<S: Numeric>(&self, a: S) -> S {
        let z = S::from_f64(self.span) / (S::TWO * a);
        let sinh = (z.exp() - (-z).exp()) * S::HALF;
        S::TWO * a * sinh - S::from_f64(self.length)
    }
}

#[test]
fn catenary_constant() {
    // Choose the catenary constant, derive the cable length for a 4 m span,
    // then recover the constant by Newton.
    let span = 4.0_f64;
    let a_true = 2.0_f64;
    let z = span / (2.0 * a_true);
    let length = 2.0 * a_true * z.sinh();
    let report = newton(&Catenary { span, length }, 1.0_f64).unwrap();
    assert!((report.root - a_true).abs() < 1e-10, "{report:?}");
}

// Diode load line: the node voltage V where the resistor current (Vs-V)/R
// equals the Shockley diode current Is*(exp(V/Vt) - 1).
struct DiodeLoadLine {
    vs: f64,
    r: f64,
    is: f64,
    vt: f64,
}

impl ScalarFn for DiodeLoadLine {
    fn eval<S: Numeric>(&self, v: S) -> S {
        let vs = S::from_f64(self.vs);
        let r = S::from_f64(self.r);
        let is = S::from_f64(self.is);
        let vt = S::from_f64(self.vt);
        (vs - v) / r - is * ((v / vt).exp() - S::ONE)
    }
}

#[test]
fn diode_load_line_voltage() {
    // 5 V source, 1 kΩ resistor, thermal voltage 25.852 mV. Pick the true
    // node voltage and back out the saturation current so it is the exact
    // root, then bracket the stiff exponential equation with bisection.
    let vs = 5.0_f64;
    let r = 1000.0_f64;
    let vt = 0.025852_f64;
    let v_true = 0.6_f64;
    let is = ((vs - v_true) / r) / ((v_true / vt).exp() - 1.0);
    let diode = DiodeLoadLine { vs, r, is, vt };
    let report = bisect(&diode, 0.0_f64, 1.0).unwrap();
    assert!((report.root - v_true).abs() < 1e-9, "{report:?}");
}

#[test]
fn wien_displacement_constant_from_blackbody_peak() {
    // The wavelength peak of the Planck blackbody spectrum solves
    // x - 5 + 5*e^(-x) = 0. Its root yields Wien's displacement constant
    // b = h*c/(x*k_B) ≈ 2.897771955e-3 m·K.
    let f = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = newton(&f, 5.0_f64).unwrap();
    let x = report.root;
    let h = 6.62607015e-34_f64;
    let c_light = 299_792_458.0_f64;
    let k_b = 1.380649e-23_f64;
    let b = h * c_light / (x * k_b);
    assert!((b - 2.897771955e-3).abs() < 1e-9, "b = {b}");
}

// ----- Systems -----

#[test]
fn chemical_equilibrium_three_species() {
    // Mass balance A + B + C = 1 coupled with two equilibria B = K1*A² and
    // C = K2*A*B. The constants are chosen so the solution is (0.4, 0.2, 0.4).
    let f = scalar_fn_vec!(|v: &[f64; 3]| [
        c(-1.0) + v[0] + v[1] + v[2],
        v[1] - c(1.25) * v[0] * v[0],
        v[2] - c(5.0) * v[0] * v[1],
    ]);
    let solver: NewtonSystem = NewtonSystem::default();
    let report = solver.solve(&f, &[0.5_f64, 0.25, 0.25]).unwrap();
    assert!(report.residual_norm < 1e-12, "{report:?}");
    let [a, b, conc_c] = report.root;
    assert!((a - 0.4).abs() < 1e-10, "{report:?}");
    assert!((b - 0.2).abs() < 1e-10, "{report:?}");
    assert!((conc_c - 0.4).abs() < 1e-10, "{report:?}");
}

#[test]
fn two_link_arm_far_start_damped() {
    // A 2 m + 1 m planar arm reaching a target built from true joint angles.
    // Starting far from the solution, damped Newton pulls the tip onto the
    // target; the residual is the tip-to-target error.
    let (l1, l2) = (2.0_f64, 1.0_f64);
    let (t1_true, t2_true) = (0.6_f64, 0.9_f64);
    let px = l1 * t1_true.cos() + l2 * (t1_true + t2_true).cos();
    let py = l1 * t1_true.sin() + l2 * (t1_true + t2_true).sin();
    let arm = TwoLinkArm { l1, l2, px, py };

    let solver: NewtonSystem = NewtonSystem::default().with_backtracking(true);
    let report = solver.solve(&arm, &[0.1_f64, 0.5]).unwrap();
    assert!(report.residual_norm < 1e-10, "{report:?}");
    // The recovered angles place the tip on the target.
    let [t1, t2] = report.root;
    let tip_x = l1 * t1.cos() + l2 * (t1 + t2).cos();
    let tip_y = l1 * t1.sin() + l2 * (t1 + t2).sin();
    assert!(
        (tip_x - px).abs() < 1e-9 && (tip_y - py).abs() < 1e-9,
        "{report:?}"
    );
}
