#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::error::{LinalgError, SolveError};
use multicalc::numerical_derivative::FiniteDifferenceSingle;
use multicalc::numerical_derivative::{AutoDiffMulti, AutoDiffSingle};
use multicalc::root_finding::{
    Bisection, Newton, NewtonSystem, RootReport, RootReportN, RootTermination,
};
use multicalc::scalar::{Numeric, ScalarFn, VectorFn, c};
use multicalc::scalar_fn;
use multicalc::scalar_fn_vec;

fn bisect<F: ScalarFn>(
    function: &F,
    lower_bound: f64,
    upper_bound: f64,
) -> Result<RootReport<f64>, SolveError> {
    Bisection::default().solve(function, lower_bound, upper_bound)
}

fn newton<F: ScalarFn>(function: &F, initial_guess: f64) -> Result<RootReport<f64>, SolveError> {
    let solver: Newton = Newton::default();
    solver.solve(function, initial_guess)
}

fn newton_system<F: VectorFn<2, 2>>(
    function: &F,
    initial_guess: &[f64; 2],
) -> Result<RootReportN<2>, SolveError> {
    let solver: NewtonSystem = NewtonSystem::default();
    solver.solve(function, initial_guess)
}

// x² + y² = 4, xy = 1; roots near (±1.932, ±0.518).
struct CircleHyperbola;

impl VectorFn<2, 2> for CircleHyperbola {
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> [S; 2] {
        [
            c(-4.0) + point[0] * point[0] + point[1] * point[1],
            c(-1.0) + point[0] * point[1],
        ]
    }
}

// Two-link planar arm forward kinematics. Holds the link lengths and target tip position.
struct TwoLinkArm {
    first_link: f64,
    second_link: f64,
    target_x: f64,
    target_y: f64,
}

impl VectorFn<2, 2> for TwoLinkArm {
    fn eval<S: Numeric>(&self, angles: &[S; 2]) -> [S; 2] {
        let first_link = S::from_f64(self.first_link);
        let second_link = S::from_f64(self.second_link);
        let target_x = S::from_f64(self.target_x);
        let target_y = S::from_f64(self.target_y);
        [
            first_link * angles[0].cos() + second_link * (angles[0] + angles[1]).cos() - target_x,
            first_link * angles[0].sin() + second_link * (angles[0] + angles[1]).sin() - target_y,
        ]
    }
}

// ----- Bisection -----

#[test]
fn bisection_sqrt2() {
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    let report = bisect(&sqrt_two, 0.0_f64, 2.0).unwrap();
    assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-9);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance | RootTermination::BracketWidth
    ));
}

#[test]
fn bisection_dottie_number() {
    // cos(x) = x, the fixed point of cosine ≈ 0.7390851332151607.
    let dottie = scalar_fn!(|x| x.cos() - x);
    let report = bisect(&dottie, 0.0_f64, 1.0).unwrap();
    assert!((report.root - 0.7390851332151607).abs() < 1e-9);
}

#[test]
fn bisection_wien_displacement() {
    // Wien's displacement law: x - 5 + 5*e^(-x) = 0, constant ≈ 4.965114231744276.
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = bisect(&wien, 1.0_f64, 10.0).unwrap();
    assert!((report.root - 4.965114231744276).abs() < 1e-9);
}

#[test]
fn bisection_invalid_bracket() {
    // x² - 2 on [2, 3]: both values positive, no sign change.
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    assert!(matches!(
        bisect(&sqrt_two, 2.0_f64, 3.0),
        Err(SolveError::InvalidBracket)
    ));
}

#[test]
fn bisection_non_finite() {
    // 1/x on [-1, 1]: f(-1) and f(1) have opposite signs, but f(0) = +∞.
    let reciprocal = scalar_fn!(|x| c(1.0) / x);
    assert!(matches!(
        bisect(&reciprocal, -1.0_f64, 1.0),
        Err(SolveError::NonFinite)
    ));
}

#[test]
fn bisection_budget_exhausted() {
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    let result = Bisection::default()
        .with_max_iterations(2)
        .solve(&sqrt_two, 0.0_f64, 2.0);
    assert!(matches!(result, Err(SolveError::DidNotConverge { .. })));
}

#[test]
fn bisection_exact_endpoint_root() {
    // f(0) = 0 exactly; the solver returns before the first iteration.
    let line = scalar_fn!(|x| x);
    let report = bisect(&line, 0.0_f64, 1.0).unwrap();
    assert_eq!(report.root, 0.0_f64);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance
    ));
}

// ----- Scalar Newton and damped Newton -----

#[test]
fn newton_sqrt2() {
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    let report = newton(&sqrt_two, 2.0_f64).unwrap();
    assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-12);
    assert!(matches!(
        report.termination,
        RootTermination::ResidualTolerance | RootTermination::StepTolerance
    ));
}

#[test]
fn newton_cbrt2() {
    // x³ - 2 = 0, root at 2^(1/3) ≈ 1.2599210498948732.
    let cbrt_two = scalar_fn!(|x| c(-2.0) + x.powi(3));
    let report = newton(&cbrt_two, 1.0_f64).unwrap();
    assert!((report.root - 2.0_f64.powf(1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn newton_wien_displacement() {
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = newton(&wien, 5.0_f64).unwrap();
    assert!((report.root - 4.965114231744276).abs() < 1e-12);
}

#[test]
fn newton_finite_difference_backend() {
    // Any DerivatorSingleVariable works in place of the autodiff default.
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    let solver = Newton::from_derivator(FiniteDifferenceSingle::<f64>::default());
    let report = solver.solve(&sqrt_two, 2.0_f64).unwrap();
    assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-6);
}

#[test]
fn newton_vanishing_derivative() {
    // f'(0) = 0 for x² - 2, so the first step is undefined.
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    assert!(matches!(
        newton(&sqrt_two, 0.0_f64),
        Err(SolveError::Linalg(LinalgError::Singular))
    ));
}

#[test]
fn newton_budget_exhausted() {
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    let solver: Newton = Newton::default().with_max_iterations(1);
    // One step from an initial guess of 2 is not enough to satisfy either tolerance.
    assert!(matches!(
        solver.solve(&sqrt_two, 2.0_f64),
        Err(SolveError::DidNotConverge { .. })
    ));
}

#[test]
fn newton_damped_rescues_far_start() {
    // f(x) = x / sqrt(1 + x²), root at 0. The Newton map is x → −x³, so from an initial guess
    // of 2.0 plain Newton diverges immediately. Backtracking halves the step until |f|
    // decreases, which is enough to land back in the basin of the root.
    let bounded_sigmoid = scalar_fn!(|x| x / (c(1.0) + x * x).sqrt());

    let plain_result = newton(&bounded_sigmoid, 2.0_f64);
    let plain_missed = match &plain_result {
        Ok(report) => report.root.abs() > 0.1,
        Err(_) => true,
    };
    assert!(
        plain_missed,
        "plain Newton unexpectedly converged: {plain_result:?}"
    );

    let damped: Newton = Newton::default().with_backtracking(true);
    let report = damped.solve(&bounded_sigmoid, 2.0_f64).unwrap();
    assert!(report.root.abs() < 1e-6, "{report:?}");
}

// ----- Vector Newton and damped Newton -----

#[test]
fn newton_system_circle_hyperbola() {
    // x² + y² = 4 and xy = 1; root near (1.932, 0.518).
    let report = newton_system(&CircleHyperbola, &[1.5_f64, 0.8]).unwrap();
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
    // Two-link arm (both links 1 m) with true joint angles (0.5 rad, 0.8 rad).
    // Tip position is computed from the truth; the solver recovers the angles from a near start.
    let first_link = 1.0_f64;
    let second_link = 1.0_f64;
    let true_first_angle = 0.5_f64;
    let true_second_angle = 0.8_f64;
    let target_x = first_link * true_first_angle.cos()
        + second_link * (true_first_angle + true_second_angle).cos();
    let target_y = first_link * true_first_angle.sin()
        + second_link * (true_first_angle + true_second_angle).sin();
    let report = newton_system(
        &TwoLinkArm {
            first_link,
            second_link,
            target_x,
            target_y,
        },
        &[0.4_f64, 0.9],
    )
    .unwrap();
    assert!(report.residual_norm < 1e-12, "{report:?}");
    let [first_angle, second_angle] = report.root;
    assert!(
        (first_angle - true_first_angle).abs() < 1e-10,
        "theta1: got {first_angle}, want {true_first_angle}"
    );
    assert!(
        (second_angle - true_second_angle).abs() < 1e-10,
        "theta2: got {second_angle}, want {true_second_angle}"
    );
}

#[test]
fn newton_system_singular_jacobian() {
    // The two equations are proportional, so the Jacobian is rank-deficient.
    let proportional_equations = scalar_fn_vec!(|v: &[f64; 2]| [
        c(-1.0) + v[0] + c(-1.0) * v[1],
        c(-2.0) + c(2.0) * v[0] + c(-2.0) * v[1],
    ]);
    assert!(matches!(
        newton_system(&proportional_equations, &[0.0_f64, 0.0]),
        Err(SolveError::Linalg(LinalgError::Singular))
    ));
}

#[test]
fn newton_system_non_finite() {
    // First component is 1/v[0], which is infinite at the starting point.
    let pole_at_origin = scalar_fn_vec!(|v: &[f64; 2]| [c(1.0) / v[0], v[1]]);
    assert!(matches!(
        newton_system(&pole_at_origin, &[0.0_f64, 0.0]),
        Err(SolveError::NonFinite)
    ));
}

#[test]
fn newton_system_budget_exhausted() {
    let solver: NewtonSystem = NewtonSystem::default().with_max_iterations(1);
    assert!(matches!(
        solver.solve(&CircleHyperbola, &[1.5_f64, 0.8]),
        Err(SolveError::DidNotConverge { .. })
    ));
}

#[test]
fn newton_system_damped_rescues_far_start() {
    // F(v) = [v[0]/sqrt(1+v[0]²), v[1]/sqrt(1+v[1]²)], root at (0, 0).
    // Each component has the Newton map x → −x³, so from (3, 3) the plain solver
    // diverges. Backtracking halves the step length until ‖F‖ decreases.
    let bounded_sigmoids = scalar_fn_vec!(|v: &[f64; 2]| [
        v[0] / (c(1.0) + v[0] * v[0]).sqrt(),
        v[1] / (c(1.0) + v[1] * v[1]).sqrt(),
    ]);
    let far = [3.0_f64, 3.0];

    let plain_result = newton_system(&bounded_sigmoids, &far);
    let plain_missed = match &plain_result {
        Ok(report) => report.residual_norm > 0.1,
        Err(_) => true,
    };
    assert!(
        plain_missed,
        "plain NewtonSystem unexpectedly converged: {plain_result:?}"
    );

    let damped: NewtonSystem = NewtonSystem::default().with_backtracking(true);
    let report = damped.solve(&bounded_sigmoids, &far).unwrap();
    assert!(report.residual_norm < 1e-10, "{report:?}");
}

// ----- f32 coverage -----

#[test]
fn newton_sqrt2_f32() {
    let sqrt_two = scalar_fn!(|x| c(-2.0) + x * x);
    let solver = Newton::<AutoDiffSingle<f32>>::default();
    let report = solver.solve(&sqrt_two, 2.0_f32).unwrap();
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
    eccentricity: f64,
    mean_anomaly: f64,
}

impl ScalarFn for Kepler {
    fn eval<S: Numeric>(&self, eccentric_anomaly: S) -> S {
        eccentric_anomaly
            - S::from_f64(self.eccentricity) * eccentric_anomaly.sin()
            - S::from_f64(self.mean_anomaly)
    }
}

#[test]
fn kepler_equation_moderate_eccentricity() {
    // Orbit with e = 0.8. Pick a true eccentric anomaly, form the mean
    // anomaly from it, then recover E by Newton starting at the mean anomaly.
    let eccentricity = 0.8_f64;
    let true_eccentric_anomaly = 1.0_f64;
    let mean_anomaly = true_eccentric_anomaly - eccentricity * true_eccentric_anomaly.sin();
    let report = newton(
        &Kepler {
            eccentricity,
            mean_anomaly,
        },
        mean_anomaly,
    )
    .unwrap();
    assert!(
        (report.root - true_eccentric_anomaly).abs() < 1e-12,
        "{report:?}"
    );
    // The solved equation holds: E - e*sin(E) == M.
    assert!((report.root - eccentricity * report.root.sin() - mean_anomaly).abs() < 1e-12);
}

#[test]
fn kepler_equation_high_eccentricity() {
    // e = 0.99 with M near zero makes 1 - e*cos(E) ~ 0.01 near the root, so
    // the Newton derivative is small and the plain step is fragile. Bisection
    // on [0, π] is guaranteed; damped Newton also recovers the root.
    let eccentricity = 0.99_f64;
    let true_eccentric_anomaly = 0.5_f64;
    let mean_anomaly = true_eccentric_anomaly - eccentricity * true_eccentric_anomaly.sin();
    let kepler = Kepler {
        eccentricity,
        mean_anomaly,
    };

    let bracketed = bisect(&kepler, 0.0_f64, core::f64::consts::PI).unwrap();
    assert!(
        (bracketed.root - true_eccentric_anomaly).abs() < 1e-9,
        "{bracketed:?}"
    );

    let damped: Newton = Newton::default().with_backtracking(true);
    let stepped = damped.solve(&kepler, mean_anomaly).unwrap();
    assert!(
        (stepped.root - true_eccentric_anomaly).abs() < 1e-9,
        "{stepped:?}"
    );
}

// Colebrook–White equation for the Darcy friction factor f of turbulent pipe flow:
// 1/√f + 2*log10(relative_roughness/3.7 + 2.51/(Re*√f)) = 0.
struct Colebrook {
    reynolds_number: f64,
    relative_roughness: f64,
}

impl ScalarFn for Colebrook {
    fn eval<S: Numeric>(&self, friction_factor: S) -> S {
        let reynolds_number = S::from_f64(self.reynolds_number);
        let relative_roughness = S::from_f64(self.relative_roughness);
        let root_friction_factor = friction_factor.sqrt();
        let inner = relative_roughness / S::from_f64(3.7)
            + S::from_f64(2.51) / (reynolds_number * root_friction_factor);
        let base_ten_log = inner.ln() / S::from_f64(10.0).ln();
        S::ONE / root_friction_factor + S::TWO * base_ten_log
    }
}

#[test]
fn colebrook_white_friction_factor() {
    // Water in a commercial-steel pipe: Re = 1e5, relative roughness 1e-4.
    let colebrook = Colebrook {
        reynolds_number: 1.0e5,
        relative_roughness: 1.0e-4,
    };
    let report = newton(&colebrook, 0.02_f64).unwrap();
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
    fn eval<S: Numeric>(&self, yield_rate: S) -> S {
        let mut present_value = S::ZERO;
        for (cash, time) in self.cashflows.iter().zip(self.times.iter()) {
            present_value += S::from_f64(*cash) / (S::ONE + yield_rate).powi(*time);
        }
        present_value - S::from_f64(self.price)
    }
}

#[test]
fn bond_internal_rate_of_return() {
    // Five-year bond, 5% annual coupon on 100 face value. Choose the true
    // yield, price the bond at it, then recover the yield by Newton.
    let cashflows = [5.0, 5.0, 5.0, 5.0, 105.0];
    let times = [1, 2, 3, 4, 5];
    let true_yield = 0.04_f64;
    let price: f64 = cashflows
        .iter()
        .zip(times.iter())
        .map(|(cash, time)| cash / (1.0_f64 + true_yield).powi(*time))
        .sum();
    let bond = BondYield {
        cashflows,
        times,
        price,
    };
    let report = newton(&bond, 0.1_f64).unwrap();
    assert!((report.root - true_yield).abs() < 1e-10, "{report:?}");
}

// Catenary: a uniform cable of length `length` hung between supports a
// horizontal distance `span` apart satisfies length = 2a*sinh(span/(2a))
// for the catenary constant a.
struct Catenary {
    span: f64,
    length: f64,
}

impl ScalarFn for Catenary {
    fn eval<S: Numeric>(&self, catenary_constant: S) -> S {
        let scaled_half_span = S::from_f64(self.span) / (S::TWO * catenary_constant);
        let sinh = (scaled_half_span.exp() - (-scaled_half_span).exp()) * S::HALF;
        S::TWO * catenary_constant * sinh - S::from_f64(self.length)
    }
}

#[test]
fn catenary_constant() {
    // Choose the catenary constant, derive the cable length for a 4 m span,
    // then recover the constant by Newton.
    let span = 4.0_f64;
    let true_catenary_constant = 2.0_f64;
    let scaled_half_span = span / (2.0 * true_catenary_constant);
    let length = 2.0 * true_catenary_constant * scaled_half_span.sinh();
    let catenary = Catenary { span, length };
    let report = newton(&catenary, 1.0_f64).unwrap();
    assert!(
        (report.root - true_catenary_constant).abs() < 1e-10,
        "{report:?}"
    );
}

// Diode load line: the node voltage V where the resistor current (Vs-V)/R
// equals the Shockley diode current Is*(exp(V/Vt) - 1).
struct DiodeLoadLine {
    source_voltage: f64,
    resistance: f64,
    saturation_current: f64,
    thermal_voltage: f64,
}

impl ScalarFn for DiodeLoadLine {
    fn eval<S: Numeric>(&self, node_voltage: S) -> S {
        let source_voltage = S::from_f64(self.source_voltage);
        let resistance = S::from_f64(self.resistance);
        let saturation_current = S::from_f64(self.saturation_current);
        let thermal_voltage = S::from_f64(self.thermal_voltage);
        (source_voltage - node_voltage) / resistance
            - saturation_current * ((node_voltage / thermal_voltage).exp() - S::ONE)
    }
}

#[test]
fn diode_load_line_voltage() {
    // 5 V source, 1 kΩ resistor, thermal voltage 25.852 mV. Pick the true
    // node voltage and back out the saturation current so it is the exact
    // root, then bracket the stiff exponential equation with bisection.
    let source_voltage = 5.0_f64;
    let resistance = 1000.0_f64;
    let thermal_voltage = 0.025852_f64;
    let true_node_voltage = 0.6_f64;
    let saturation_current = ((source_voltage - true_node_voltage) / resistance)
        / ((true_node_voltage / thermal_voltage).exp() - 1.0);
    let diode = DiodeLoadLine {
        source_voltage,
        resistance,
        saturation_current,
        thermal_voltage,
    };
    let report = bisect(&diode, 0.0_f64, 1.0).unwrap();
    assert!((report.root - true_node_voltage).abs() < 1e-9, "{report:?}");
}

#[test]
fn wien_displacement_constant_from_blackbody_peak() {
    // The wavelength peak of the Planck blackbody spectrum solves
    // x - 5 + 5*e^(-x) = 0. Its root yields Wien's displacement constant
    // b = h*c/(x*k_B) ≈ 2.897771955e-3 m·K.
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let report = newton(&wien, 5.0_f64).unwrap();
    let x = report.root;
    let planck_constant = 6.62607015e-34_f64;
    let speed_of_light = 299_792_458.0_f64;
    let boltzmann_constant = 1.380649e-23_f64;
    let wien_constant = planck_constant * speed_of_light / (x * boltzmann_constant);
    assert!(
        (wien_constant - 2.897771955e-3).abs() < 1e-9,
        "b = {wien_constant}"
    );
}

// ----- Systems -----

#[test]
fn chemical_equilibrium_three_species() {
    // Mass balance A + B + C = 1 coupled with two equilibria B = K1*A² and
    // C = K2*A*B. The constants are chosen so the solution is (0.4, 0.2, 0.4).
    let equilibrium = scalar_fn_vec!(|v: &[f64; 3]| [
        c(-1.0) + v[0] + v[1] + v[2],
        v[1] - c(1.25) * v[0] * v[0],
        v[2] - c(5.0) * v[0] * v[1],
    ]);
    let solver: NewtonSystem = NewtonSystem::default();
    let report = solver.solve(&equilibrium, &[0.5_f64, 0.25, 0.25]).unwrap();
    assert!(report.residual_norm < 1e-12, "{report:?}");
    let [concentration_a, concentration_b, concentration_c] = report.root;
    assert!((concentration_a - 0.4).abs() < 1e-10, "{report:?}");
    assert!((concentration_b - 0.2).abs() < 1e-10, "{report:?}");
    assert!((concentration_c - 0.4).abs() < 1e-10, "{report:?}");
}

#[test]
fn two_link_arm_far_start_damped() {
    // A 2 m + 1 m planar arm reaching a target built from true joint angles.
    // Starting far from the solution, damped Newton pulls the tip onto the
    // target; the residual is the tip-to-target error.
    let first_link = 2.0_f64;
    let second_link = 1.0_f64;
    let true_first_angle = 0.6_f64;
    let true_second_angle = 0.9_f64;
    let target_x = first_link * true_first_angle.cos()
        + second_link * (true_first_angle + true_second_angle).cos();
    let target_y = first_link * true_first_angle.sin()
        + second_link * (true_first_angle + true_second_angle).sin();
    let arm = TwoLinkArm {
        first_link,
        second_link,
        target_x,
        target_y,
    };

    let solver: NewtonSystem = NewtonSystem::default().with_backtracking(true);
    let report = solver.solve(&arm, &[0.1_f64, 0.5]).unwrap();
    assert!(report.residual_norm < 1e-10, "{report:?}");
    // The recovered angles place the tip on the target.
    let [first_angle, second_angle] = report.root;
    let tip_x = first_link * first_angle.cos() + second_link * (first_angle + second_angle).cos();
    let tip_y = first_link * first_angle.sin() + second_link * (first_angle + second_angle).sin();
    assert!(
        (tip_x - target_x).abs() < 1e-9 && (tip_y - target_y).abs() < 1e-9,
        "{report:?}"
    );
}
