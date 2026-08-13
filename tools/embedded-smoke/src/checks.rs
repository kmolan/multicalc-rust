//! Tiny on-target math checks. Each asserts a known answer to a tolerance.
//!
//! Golden checks assert against values taken from the host QA fixtures (see
//! `fixtures.rs`), so the target and the host share one source of truth. Identity
//! checks assert a mathematical identity that needs no fixture. Every assertion is
//! a hard failure: a wrong answer panics, which the runner turns into a non-zero
//! QEMU exit.
//!
//! Every check passes its inputs and its result through `core::hint::black_box` so
//! the compiler cannot const-fold the work away — the point is to run instructions on
//! target, not to prove a constant at build time.

use core::hint::black_box;

use multicalc::LevenbergMarquardt;
use multicalc::SO3;
use multicalc::control::{
    FollowTheGap, GeometricAttitudeController, Lqr, Pid, pure_pursuit_curvature,
};
use multicalc::error::LinalgError;
use multicalc::estimation::{
    ConstantTurnAndSpeed, ExtendedKalmanFilter, KalmanFilter, KalmanModel,
};
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Vector, Vector2D};
use multicalc::mapping::{MutableOccupancyMap, OccupancyMap};
use multicalc::numerical_derivative::DerivatorSingleVariable;
use multicalc::numerical_derivative::Jacobian;
use multicalc::numerical_derivative::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_integration::GaussianQuadratureMethod;
use multicalc::numerical_integration::GaussianSingle;
use multicalc::numerical_integration::IntegratorSingleVariable;
use multicalc::polynomial::{
    MultivariatePolynomial, MultivariateTerm, PiecewisePolynomial, Polynomial,
};
use multicalc::root_finding::Newton;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::scalar_fn;
use multicalc::spatial::SE2;
use multicalc::vector_field::{curl_3d, divergence_3d};
use multicalc_testkit::problems::{Jac23, Rosenbrock, VField3d, Wien};

use crate::fixtures;

/// Assert `got` is within `abs + rel * max(|got|, |want|)` of `want`, printing both
/// values and the tolerance over semihosting before panicking on failure. Use this for
/// identity checks so a QEMU-only failure shows the diverging number, not just a line.
macro_rules! assert_close {
    ($name:expr, $got:expr, $want:expr, $abs:expr, $rel:expr) => {{
        let got: f64 = $got;
        let want: f64 = $want;
        let ok = (got - want).abs() <= $abs + $rel * got.abs().max(want.abs());
        if !ok {
            let _ = crate::hprintln!(
                "CHECK {} FAIL got={:e} want={:e} abs={:e} rel={:e}",
                $name,
                got,
                want,
                $abs as f64,
                $rel as f64
            );
        }
        assert!(ok, "{}", $name);
    }};
}

/// Golden: the Rosenbrock least-squares minimizer must match the host QA golden
/// (optimization/rosenbrock). Returns `solution[0]` for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lm_fit() -> f64 {
    let x0 = black_box(fixtures::ROSENBROCK_X0);
    let solver = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(100);
    let report = solver.minimize(&Rosenbrock, &x0).expect("fit converges");
    for i in 0..2 {
        assert_close!(
            "lm_fit",
            black_box(report.solution[i]),
            fixtures::ROSENBROCK_SOLUTION[i],
            fixtures::ROSENBROCK_ABS,
            fixtures::ROSENBROCK_REL
        );
    }
    black_box(report.solution[0])
}

/// Identity: differentiate x^3 at x = 2 by autodiff. Exact derivative is 12. Returns the
/// value for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn autodiff_derivative() -> f64 {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value = d
        .differentiate(1, &f, black_box(2.0_f64))
        .expect("derivative");
    assert_close!("autodiff_derivative", black_box(value), 12.0, 1e-12, 0.0);
    black_box(value)
}

/// Identity in f32: differentiate x^3 at x = 2, exact derivative 12. f32 arithmetic is where
/// soft-float (eabi) and the hardware FPU (eabihf) diverge. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn autodiff_derivative_f32() -> f32 {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value: f32 = d
        .differentiate(1, &f, black_box(2.0_f32))
        .expect("derivative f32");
    // f32 tolerance is looser than f64; assert in f32 space directly.
    let ok = (value - 12.0_f32).abs() <= 1e-4;
    if !ok {
        let _ = crate::hprintln!("CHECK autodiff_f32 FAIL got={:e}", value);
    }
    assert!(ok, "autodiff_derivative_f32");
    black_box(value)
}

/// Real portable-path library call for the Cortex-M0 canary: a vector dot product through
/// `multicalc`. `[1,2,3,4] · [4,3,2,1] = 20`. Exercises a no-atomics, no-alloc library symbol
/// (unlike a plain array fold, which touches no `multicalc` code).
pub fn portable_path() {
    use multicalc::linear_algebra::Vector;
    let a = black_box(Vector::new([1.0_f64, 2.0, 3.0, 4.0]));
    let b = black_box(Vector::new([4.0_f64, 3.0, 2.0, 1.0]));
    let dot = black_box(a.dot(b));
    assert_close!("portable_path", dot, 20.0, 1e-12, 0.0);
}

/// No-panic negative path: a fallible decomposition returns a typed `Err` on bad input
/// instead of crashing.
pub fn error_path_returns_err() {
    let singular = black_box(Matrix3D::<f64>::zeros());
    assert!(matches!(singular.lu(), Err(LinalgError::Singular)));
    let indefinite = black_box(Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]));
    assert!(matches!(
        indefinite.cholesky(),
        Err(LinalgError::NotPositiveDefinite)
    ));
}

/// Golden: singular values of a fixture matrix must match the host QA golden (linalg/svd_3x3).
/// Returns the values for the cross-ABI guard (emitted on every target).
pub fn svd_golden() -> [f64; 3] {
    let a: Matrix3D = black_box(Matrix::new(fixtures::SVD_3X3_INPUT));
    let sv = a.svd().expect("svd").singular_values();
    for i in 0..3 {
        assert_close!(
            "svd_golden",
            black_box(sv[i]),
            fixtures::SVD_3X3_SINGULAR_VALUES[i],
            fixtures::SVD_3X3_ABS,
            fixtures::SVD_3X3_REL
        );
    }
    [black_box(sv[0]), black_box(sv[1]), black_box(sv[2])]
}

/// Identity: SO(3)/SE(3) exp/log round trips and one known rotation. Returns the first SE(3)
/// log component for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lie_group_identity() -> f64 {
    use multicalc::linear_algebra::Vector;
    use multicalc::spatial::{SE3, SO3};

    // A 90° rotation about z maps x -> y.
    let rz = SO3::<f64>::exp(black_box(Vector::new([
        0.0,
        0.0,
        core::f64::consts::FRAC_PI_2,
    ])));
    let p = rz.act(black_box(Vector::new([1.0, 0.0, 0.0])));
    assert_close!("lie_rot_x", black_box(p[0]), 0.0, 1e-12, 0.0);
    assert_close!("lie_rot_y", black_box(p[1]), 1.0, 1e-12, 0.0);
    assert_close!("lie_rot_z", black_box(p[2]), 0.0, 1e-12, 0.0);

    // SO(3) exp/log round trip.
    let phi = black_box(Vector::new([0.3, -0.6, 0.2]));
    let back = SO3::exp(phi).log();
    for i in 0..3 {
        assert_close!("lie_so3", black_box(back[i]), phi[i], 1e-9, 0.0);
    }

    // SE(3) exp/log round trip (exercises the left Jacobian and its inverse).
    let xi = black_box(Vector::new([0.5, -0.2, 0.1, 0.3, -0.6, 0.2]));
    let back6 = SE3::exp(xi).log();
    for i in 0..6 {
        assert_close!("lie_se3", black_box(back6[i]), xi[i], 1e-9, 0.0);
    }
    black_box(back6[0])
}

/// Identity: RK4 integrates the harmonic oscillator over one period back to its start; RK45
/// integrates y' = -y to e^{-1}. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn ode_identity() {
    use multicalc::linear_algebra::Vector;
    use multicalc::ode::{Rk4, Rk45};

    let f = |_t: f64, y: &Vector2D| Vector::new([y[1], -y[0]]);
    let steps = 2000;
    let dt = core::f64::consts::TAU / steps as f64;
    let yf = Rk4::integrate(
        &f,
        0.0,
        &black_box(Vector::new([1.0, 0.0])),
        dt,
        steps,
        |_, _| {},
    );
    assert_close!("ode_rk4_x", black_box(yf[0]), 1.0, 1e-4, 0.0);
    assert_close!("ode_rk4_v", black_box(yf[1]), 0.0, 1e-4, 0.0);

    let g = |_t: f64, y: &Vector<1, f64>| -*y;
    let e = Rk45::default()
        .solve(&g, 0.0, &black_box(Vector::new([1.0])), 1.0)
        .expect("rk45 solve");
    assert_close!(
        "ode_rk45",
        black_box(e[0]),
        multicalc::libm::exp(-1.0),
        1e-6,
        0.0
    );
}

/// Identity: Gauss-Legendre order 4 integrates `2x` on `[0, 2]` to `4`. Returns the value for
/// the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn quadrature_identity() -> f64 {
    let f = |x: f64| 2.0 * x;
    let quad = GaussianSingle::<f64>::from_parameters(4, GaussianQuadratureMethod::GaussLegendre);
    let value = quad
        .single_integral(&f, &black_box([0.0, 2.0]))
        .expect("quadrature");
    assert_close!("quadrature", black_box(value), 4.0, 1e-12, 0.0);
    black_box(value)
}

/// Identity: the Jacobian of `[x*y*z, x^2 + y^2]` at `(1, 2, 3)` is `[[6,3,2],[2,4,0]]`. Returns
/// the `(0, 0)` entry for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn jacobian_identity() -> f64 {
    let j = Jacobian::<AutoDiffMulti>::default()
        .evaluate(&Jac23, &black_box([1.0, 2.0, 3.0]))
        .expect("jacobian");
    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for r in 0..2 {
        for c in 0..3 {
            assert_close!("jacobian", black_box(j[(r, c)]), expected[r][c], 1e-12, 0.0);
        }
    }
    black_box(j[(0, 0)])
}

/// Identity: the field `[y, -x, 2z]` at `(1, 2, 3)` has curl `[0, 0, -2]` and divergence `2`.
/// Returns the divergence for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn vector_field_identity() -> f64 {
    let point = black_box([1.0, 2.0, 3.0]);
    let c = curl_3d(AutoDiffMulti::default(), &VField3d, &point).expect("curl");
    let expected_curl = [0.0, 0.0, -2.0];
    for i in 0..3 {
        assert_close!("vfield_curl", black_box(c[i]), expected_curl[i], 1e-12, 0.0);
    }
    let d = divergence_3d(AutoDiffMulti::default(), &VField3d, &point).expect("divergence");
    assert_close!("vfield_div", black_box(d), 2.0, 1e-12, 0.0);
    black_box(d)
}

/// Golden: Newton on Wien's displacement equation must match the host QA golden root
/// (root_finding/wien_newton). Returns the root for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn root_finding_golden() -> f64 {
    let report = Newton::<AutoDiffSingle>::default()
        .solve(&Wien, black_box(fixtures::ROOT_WIEN_X0))
        .expect("newton solve");
    assert_close!(
        "root_finding",
        black_box(report.root),
        fixtures::ROOT_WIEN_ROOT,
        fixtures::ROOT_WIEN_ABS,
        fixtures::ROOT_WIEN_REL
    );
    black_box(report.root)
}

/// A position fix: the sensor sees the first two state components.
struct GlobalPosition;
impl VectorFn<5, 2> for GlobalPosition {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// Golden: the linear Kalman filter's final state must match the host QA golden
/// (estimation/kalman_filter_constant_velocity_one_dimensional). Returns `state[0]` for the
/// cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn kalman_filter_golden() -> f64 {
    let mut filter = KalmanFilter::<2, 1>::new(
        black_box(Vector::new(fixtures::KALMAN_INITIAL_STATE)),
        Matrix::new(fixtures::KALMAN_INITIAL_COVARIANCE),
        KalmanModel {
            state_transition: Matrix::new(fixtures::KALMAN_STATE_TRANSITION),
            measurement_model: Matrix::new(fixtures::KALMAN_MEASUREMENT_MODEL),
            process_noise: Matrix::new(fixtures::KALMAN_PROCESS_NOISE),
            measurement_noise: Matrix::new(fixtures::KALMAN_MEASUREMENT_NOISE),
        },
    );
    for row in fixtures::KALMAN_MEASUREMENTS {
        filter.predict();
        filter
            .update(black_box(Vector::new(row)))
            .expect("kalman update");
    }
    let state = filter.state();
    for i in 0..2 {
        assert_close!(
            "kalman_filter",
            black_box(state[i]),
            fixtures::KALMAN_EXPECTED_STATE[i],
            fixtures::KALMAN_ABS,
            fixtures::KALMAN_REL
        );
    }
    black_box(state[0])
}

/// Golden: the extended filter tracking a turning vehicle from position fixes must match the
/// host QA golden (estimation/extended_kalman_filter_coordinated_turn_fusion). This is the only
/// check that drives autodiff Jacobians at 5x5 on target. Returns `state[0]`. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn extended_kalman_filter_golden() -> f64 {
    let motion = ConstantTurnAndSpeed {
        timestep: fixtures::COORDINATED_TURN_TIMESTEP,
    };
    let mut filter = ExtendedKalmanFilter::<5, 2>::new(
        black_box(Vector::new(fixtures::COORDINATED_TURN_INITIAL_STATE)),
        Matrix::new(fixtures::COORDINATED_TURN_INITIAL_COVARIANCE),
        Matrix::new(fixtures::COORDINATED_TURN_PROCESS_NOISE),
        Matrix::new(fixtures::COORDINATED_TURN_MEASUREMENT_NOISE),
    );
    for row in fixtures::COORDINATED_TURN_MEASUREMENTS {
        filter.predict(&motion).expect("extended predict");
        filter
            .update(&GlobalPosition, black_box(Vector::new(row)))
            .expect("extended update");
    }
    let state = filter.state();
    for i in 0..5 {
        assert_close!(
            "extended_kalman_filter",
            black_box(state[i]),
            fixtures::COORDINATED_TURN_EXPECTED_STATE[i],
            fixtures::COORDINATED_TURN_ABS,
            fixtures::COORDINATED_TURN_REL
        );
    }
    black_box(state[0])
}

/// Identity: pure pursuit steers straight at a point dead ahead, and returns the exact
/// `2*lateral/L^2` for one off-axis. Returns the off-axis curvature. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn pure_pursuit_identity() -> f64 {
    let pose = SE2::<f64>::identity();
    let ahead = pure_pursuit_curvature(pose, black_box(Vector::new([2.0, 0.0])), 2.0)
        .expect("pure pursuit ahead");
    assert_close!(
        "pure_pursuit_ahead",
        black_box(ahead.value()),
        0.0,
        1e-12,
        0.0
    );

    let left = pure_pursuit_curvature(pose, black_box(Vector::new([2.0, 1.0])), 2.0)
        .expect("pure pursuit left");
    assert_close!(
        "pure_pursuit_left",
        black_box(left.value()),
        0.5,
        1e-12,
        0.0
    );
    black_box(left.value())
}

/// A map of `COLUMNS` by `ROWS` cells held in a plain array: what a board with no heap uses.
struct FixedMap<const COLUMNS: usize, const ROWS: usize> {
    cells: [[bool; COLUMNS]; ROWS],
}

impl<const COLUMNS: usize, const ROWS: usize> OccupancyMap for FixedMap<COLUMNS, ROWS> {
    fn columns(&self) -> usize {
        COLUMNS
    }
    fn rows(&self) -> usize {
        ROWS
    }
    fn resolution(&self) -> f64 {
        0.25
    }
    fn origin(&self) -> [f64; 2] {
        [0.0, 0.0]
    }
    fn is_occupied(&self, row: usize, column: usize) -> bool {
        self.cells
            .get(row)
            .and_then(|row| row.get(column))
            .copied()
            .unwrap_or(false)
    }
}

impl<const COLUMNS: usize, const ROWS: usize> MutableOccupancyMap for FixedMap<COLUMNS, ROWS> {
    fn set_cell(&mut self, row: usize, column: usize, occupied: bool) {
        if let Some(cell) = self.cells.get_mut(row).and_then(|row| row.get_mut(column)) {
            *cell = occupied;
        }
    }
    fn clear(&mut self) {
        self.cells = [[false; COLUMNS]; ROWS];
    }
}

/// Identity: a wall drawn at x = 1.25 on a 16x16 map of 0.25 m cells is met at exactly 0.75 m by a
/// beam fired along the row from x = 0.5, and a beam fired the other way meets nothing. Covers the
/// rasterizing and the cell walk together, over a map held in a plain array. Returns the distance.
/// Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn occupancy_ray_cast_identity() -> f64 {
    let mut map: FixedMap<16, 16> = FixedMap {
        cells: [[false; 16]; 16],
    };

    // A wall up the map, drawn from world points rather than marked cell by cell.
    let wall_x = 1.25;
    let wall = [[wall_x, 0.0], [wall_x, 4.0]];
    let open_ended = false;
    map.occupy_polyline(&wall, open_ended);

    let sensor_position = [0.5, 0.5];
    let along_the_row = 0.0;
    let maximum_range = 4.0;
    let distance = map
        .cast_ray(black_box(sensor_position), along_the_row, maximum_range)
        .expect("wall");
    let expected = wall_x - sensor_position[0];
    assert_close!(
        "occupancy_ray_cast",
        black_box(distance),
        expected,
        1e-12,
        0.0
    );

    let back_along_the_row = core::f64::consts::PI;
    assert!(
        map.cast_ray(
            black_box(sensor_position),
            back_along_the_row,
            maximum_range
        )
        .is_none(),
        "occupancy_ray_cast_behind"
    );
    black_box(distance)
}

/// Identity: an unobstructed scan drives straight ahead at cruise speed, and a wall all round
/// stops the robot and reports it. Returns the clear-scan linear speed. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn follow_the_gap_identity() -> f64 {
    const BEAMS: usize = 31;
    let field_of_view = 2.0 * core::f64::consts::PI / 3.0;
    let follower: FollowTheGap<BEAMS, f64> =
        FollowTheGap::try_new(field_of_view, 4.0, 0.50, 0.60, 0.40).expect("follower");

    let clear = follower
        .compute(&black_box([4.0; BEAMS]), 0.0)
        .expect("clear scan");
    assert_close!(
        "follow_the_gap_heading",
        black_box(clear.heading()),
        0.0,
        1e-12,
        0.0
    );
    assert_close!(
        "follow_the_gap_speed",
        black_box(clear.body_twist().linear()),
        0.40,
        1e-12,
        0.0
    );

    let blocked = follower
        .compute(&black_box([0.2; BEAMS]), 0.0)
        .expect("blocked scan");
    assert!(blocked.is_blocked(), "follow_the_gap_blocked");
    assert_close!(
        "follow_the_gap_stopped",
        black_box(blocked.body_twist().linear()),
        0.0,
        0.0,
        0.0
    );
    black_box(clear.body_twist().linear())
}

/// Identity in f32: two unit-measurement steps of a constant-velocity filter land on the exact
/// [5/3, 2/3]. f32 arithmetic is where soft-float (eabi) and the hardware FPU (eabihf) diverge.
/// Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn kalman_filter_identity_f32() -> f32 {
    let mut filter = KalmanFilter::<2, 1, f32>::new(
        black_box(Vector::new([0.0_f32, 0.0])),
        Matrix::identity(),
        KalmanModel {
            state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
            measurement_model: Matrix::new([[1.0, 0.0]]),
            process_noise: Matrix::zeros(),
            measurement_noise: Matrix::new([[1.0]]),
        },
    );
    for measurement in [1.0_f32, 2.0] {
        filter.predict();
        filter
            .update(black_box(Vector::new([measurement])))
            .expect("kalman update f32");
    }
    let state = filter.state();
    let expected = [5.0_f32 / 3.0, 2.0_f32 / 3.0];
    for i in 0..2 {
        let scale = expected[i].abs().max(1.0);
        let ok = (state[i] - expected[i]).abs() <= 128.0 * f32::EPSILON * scale;
        if !ok {
            let _ = crate::hprintln!(
                "CHECK kalman_f32 FAIL got={:e} want={:e}",
                state[i],
                expected[i]
            );
        }
        assert!(ok, "kalman_filter_identity_f32");
    }
    black_box(state[0])
}

/// Real library call for the canary tier: a PID controller drives a plant that adds each output
/// straight onto its measurement. The first output is the exact `kp*e + ki*e*dt`, and over the
/// next hundred ticks the plant climbs steadily to within one percent of the setpoint. Returns
/// the measurement it reaches.
pub fn pid_step() -> f64 {
    let dt = 0.01_f64;
    let setpoint = 1.0_f64;
    let mut controller = Pid::new(2.0, 1.0, 0.0, dt).expect("pid");

    let first = controller.update(setpoint, black_box(0.0));
    assert_close!("pid_first_output", black_box(first), 2.01, 1e-12, 0.0);

    let mut measurement = dt * first;
    for _ in 0..100 {
        let previous = measurement;
        let output = controller.update(setpoint, measurement);
        measurement += dt * output;
        assert!(measurement > previous, "pid_climbs");
    }
    assert_close!("pid_settled", black_box(measurement), setpoint, 0.01, 0.0);
    black_box(measurement)
}

/// Identity: an optimal feedback law built for a cart carrying its speed forward brings a state
/// started away from zero back home, and the loop it closes certifies as settling. Returns the
/// distance left at the end. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lqr_identity() -> f64 {
    let state_transition = Matrix::<2, 2>::new([[1.0, 0.1], [0.0, 1.0]]);
    let input_model = Matrix::<2, 1>::new([[0.005], [0.1]]);
    let controller = Lqr::new(
        black_box(state_transition),
        input_model,
        Matrix::<2, 2>::identity(),
        Matrix::<1, 1>::identity(),
    )
    .expect("lqr design");
    let _ = controller.certify_stability().expect("lqr certificate");

    let mut state = Vector::new([1.0, 0.0]);
    for _ in 0..400 {
        let input = controller.control(state);
        state = state_transition * state + input_model * input;
    }
    assert_close!("lqr_settled", black_box(state.norm()), 0.0, 1e-6, 0.0);
    black_box(state.norm())
}

/// Identity: a body already pointing the right way and not turning needs no torque, and a body
/// tipped about x is pushed back the other way. Returns the x torque of the tipped case. Full set
/// only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn geometric_attitude_identity() -> f64 {
    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let controller =
        GeometricAttitudeController::new(6.0, 1.2, inertia).expect("attitude controller");
    let level = SO3::<f64>::identity();
    let still = Vector::new([0.0, 0.0, 0.0]);

    let at_target = controller.torque(level, still, level, still, still);
    assert_close!(
        "attitude_at_target",
        black_box(at_target.norm()),
        0.0,
        1e-12,
        0.0
    );

    let tipped = SO3::exp(black_box(Vector::new([0.1, 0.0, 0.0])));
    let pushed_back = controller.torque(tipped, still, level, still, still);
    assert!(pushed_back[0] < 0.0, "attitude_pushes_back");
    black_box(pushed_back[0])
}

/// Identity: a degree-7 polynomial's value and first two derivatives at `x = 0.5`, against
/// hand-computed constants. Repeated multiply-and-add over eight coefficients, so it adds
/// almost nothing to `.text` and no new library surface. Returns the value. Canary set.
pub fn polynomial_evaluate() -> f64 {
    // 1 + 2x + 3x² + 4x³ + 5x⁴ + 6x⁵ + 7x⁶ + 8x⁷ at x = 1/2, where every power of a half is
    // exact, so the three answers below are exact too.
    let polynomial = Polynomial::<8, f64>::new(black_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    let [value, slope, bend] = polynomial.evaluate_with_derivatives(black_box(0.5));

    assert_close!("polynomial_value", black_box(value), 3.921875, 1e-12, 0.0);
    assert_close!("polynomial_slope", black_box(slope), 14.5625, 1e-12, 0.0);
    assert_close!("polynomial_bend", black_box(bend), 71.625, 1e-12, 0.0);
    black_box(value)
}

/// Identity: the roots of `(x − 1)(x − 2)` are 1 and 2. Pulls in `sqrt` and `copysign`, both
/// already linked by the canary's SVD check, so it adds no new library surface on the M0.
/// Returns the first root. Canary set.
pub fn polynomial_quadratic_roots() -> f64 {
    let quadratic = Polynomial::<3, f64>::new(black_box([2.0, -3.0, 1.0]));
    let roots = quadratic.real_roots().expect("quadratic roots");

    assert!(roots.len() == 2, "polynomial_quadratic_root_count");
    let found = roots.as_slice();
    assert_close!("polynomial_root_0", black_box(found[0]), 1.0, 1e-12, 0.0);
    assert_close!("polynomial_root_1", black_box(found[1]), 2.0, 1e-12, 0.0);
    black_box(found[0])
}

/// Golden: the four real roots of `x(x − 1)(x − 2)(x − 4)` against the host QA golden
/// (polynomial/roots_quartic_four_real). Returns root 0. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn polynomial_roots_golden() -> f64 {
    let quartic = Polynomial::<5, f64>::new(black_box(fixtures::POLYNOMIAL_QUARTIC_COEFFICIENTS));
    let roots = quartic.real_roots().expect("quartic roots");

    assert!(
        roots.len() == fixtures::POLYNOMIAL_QUARTIC_ROOTS.len(),
        "polynomial_quartic_root_count"
    );
    for (index, want) in fixtures::POLYNOMIAL_QUARTIC_ROOTS.iter().enumerate() {
        assert_close!(
            "polynomial_quartic_root",
            black_box(roots.as_slice()[index]),
            *want,
            fixtures::POLYNOMIAL_QUARTIC_ABS,
            fixtures::POLYNOMIAL_QUARTIC_REL
        );
    }
    black_box(roots.as_slice()[0])
}

/// Identity: the six roots of `(x + 3)(x + 1)(x − 0.5)(x − 2)(x − 4)(x − 7)`, found by counting
/// sign changes and halving rather than by a formula. The deepest polynomial path on target —
/// a chain of seven polynomials plus the range stack. Returns root 0. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn polynomial_sturm_identity() -> f64 {
    let degree6 =
        Polynomial::<7, f64>::new(black_box([84.0, -131.0, -126.5, 104.5, 5.5, -9.5, 1.0]));
    let bound = degree6.cauchy_root_bound().expect("root bound");
    let roots = degree6
        .real_roots_in(black_box(-bound), black_box(bound), 1e-9, 200)
        .expect("roots by halving");

    assert!(roots.len() == 6, "polynomial_sturm_root_count");
    let expected = [-3.0, -1.0, 0.5, 2.0, 4.0, 7.0];
    for (index, want) in expected.iter().enumerate() {
        assert_close!(
            "polynomial_sturm_root",
            black_box(roots.as_slice()[index]),
            *want,
            1e-8,
            0.0
        );
    }
    black_box(roots.as_slice()[0])
}

/// Golden: the planned trajectory a controller follows, evaluated on target. Builds the
/// piecewise curve straight from the host golden's coefficients and compares position, velocity
/// and acceleration at the fixture's own sample times. **This is the hot-loop path** — planning
/// stays host-side. Returns the first sampled position component. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn piecewise_polynomial_golden() -> f64 {
    let mut pieces = [[Polynomial::<8, f64>::zeros(); 3]; 3];
    for (segment, piece) in pieces.iter_mut().enumerate() {
        for (axis, slot) in piece.iter_mut().enumerate() {
            *slot = Polynomial::new(fixtures::MINIMUM_SNAP_COEFFICIENTS[segment * 3 + axis]);
        }
    }
    let trajectory = PiecewisePolynomial::<3, 8, 3, f64>::try_from_pieces(
        black_box(&pieces),
        black_box(&fixtures::MINIMUM_SNAP_DURATIONS),
    )
    .expect("trajectory");

    let mut first = 0.0;
    for (sample, time) in fixtures::MINIMUM_SNAP_SAMPLE_TIMES.iter().enumerate() {
        let orders = trajectory
            .evaluate_with_derivatives::<3>(black_box(*time))
            .expect("trajectory state");
        for (order, found) in orders.iter().enumerate() {
            let want = fixtures::MINIMUM_SNAP_SAMPLED_STATES[sample * 3 + order];
            for axis in 0..3 {
                assert_close!(
                    "piecewise_state",
                    black_box(found[axis]),
                    want[axis],
                    fixtures::MINIMUM_SNAP_ABS,
                    fixtures::MINIMUM_SNAP_REL
                );
            }
        }
        if sample == 0 {
            first = orders[0][0];
        }
    }
    black_box(first)
}

/// Identity: `3x²y + 2xy − 1` at `(1.5, −2.0)` and its slopes there, against hand-computed
/// values. Exercises raising to a whole-number power on target. Returns the value. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn multivariate_identity() -> f64 {
    let polynomial = MultivariatePolynomial::<2, 3, f64>::try_from_terms(&[
        MultivariateTerm::new(3.0, [2, 1]),
        MultivariateTerm::new(2.0, [1, 1]),
        MultivariateTerm::new(-1.0, [0, 0]),
    ])
    .expect("multivariate");

    let point = black_box([1.5, -2.0]);
    // 3·2.25·(−2) + 2·1.5·(−2) − 1
    let value = polynomial.evaluate(&point);
    assert_close!("multivariate_value", black_box(value), -20.5, 1e-12, 0.0);

    // 6xy + 2y, and 3x² + 2x.
    let gradient = polynomial.gradient_at(&point);
    assert_close!(
        "multivariate_slope_x",
        black_box(gradient[0]),
        -22.0,
        1e-12,
        0.0
    );
    assert_close!(
        "multivariate_slope_y",
        black_box(gradient[1]),
        9.75,
        1e-12,
        0.0
    );
    black_box(value)
}

/// The degree-7 evaluation again in `f32`, widened on the way out. Single precision is where
/// soft-float and a hardware FPU are most likely to disagree, which is what the cross-ABI guard
/// exists to catch. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn polynomial_evaluate_identity_f32() -> f64 {
    let polynomial = Polynomial::<8, f32>::new(black_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    let [value, slope]: [f32; 2] = polynomial.evaluate_with_derivatives(black_box(0.5_f32));

    assert_close!(
        "polynomial_value_f32",
        black_box(f64::from(value)),
        3.921875,
        1e-5,
        0.0
    );
    assert_close!(
        "polynomial_slope_f32",
        black_box(f64::from(slope)),
        14.5625,
        1e-5,
        0.0
    );
    black_box(f64::from(value))
}

/// A two-joint planar arm of unit links, welded to a tool frame one link further out.
/// Shared by the kinematics checks below. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
fn planar_arm() -> multicalc::kinematics::KinematicTree<3, f64> {
    use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    use multicalc::spatial::{SE3, SO3};

    let about_z = Vector::new([0.0, 0.0, 1.0]);
    let along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    let joints = [
        Joint::revolute(about_z, SE3::identity()),
        Joint::revolute(about_z, along_x),
        Joint::fixed(along_x),
    ];
    let parents = [
        JointParent::World,
        JointParent::Joint(0),
        JointParent::Joint(1),
    ];
    KinematicTree::<3, f64>::try_from_joints(&joints, &parents)
        .unwrap_or_else(|_| unreachable!("two-joint planar arm is a valid tree"))
}

/// Identity: a two-joint planar arm of unit links puts its tool at
/// `[cos a + cos(a+b), sin a + sin(a+b), 0]`. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn kinematic_tree_identity() -> f64 {
    let tree = planar_arm();
    let (a, b) = (0.3_f64, -0.7);
    let state = tree
        .forward_kinematics(&black_box(Vector::new([a, b, 0.0])))
        .unwrap_or_else(|_| unreachable!("finite readings"));
    let tool = state
        .pose(2)
        .unwrap_or_else(|| unreachable!("three joints were settled"))
        .translation();

    assert_close!(
        "fk_x",
        black_box(tool[0]),
        a.cos() + (a + b).cos(),
        1e-12,
        0.0
    );
    assert_close!(
        "fk_y",
        black_box(tool[1]),
        a.sin() + (a + b).sin(),
        1e-12,
        0.0
    );
    assert_close!("fk_z", black_box(tool[2]), 0.0, 1e-12, 0.0);
    black_box(tool[0])
}

/// Identity: on the same arm stretched along x, the tool sits 2 units out, so turning the
/// shoulder sweeps it sideways at twice the rate turning the elbow does, and the weld
/// contributes nothing. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn geometric_jacobian_identity() -> f64 {
    use multicalc::kinematics::JacobianFrame;

    let tree = planar_arm();
    let jacobian = tree
        .geometric_jacobian_at(&black_box(Vector::zeros()), 2, JacobianFrame::World)
        .unwrap_or_else(|_| unreachable!("finite readings, valid tool index"));

    let shoulder = jacobian
        .column(0)
        .unwrap_or_else(|| unreachable!("column 0 is active"));
    let elbow = jacobian
        .column(1)
        .unwrap_or_else(|| unreachable!("column 1 is active"));
    let weld = jacobian
        .column(2)
        .unwrap_or_else(|| unreachable!("column 2 is active"));

    assert_close!(
        "jac_shoulder_v",
        black_box(shoulder.linear()[1]),
        2.0,
        1e-12,
        0.0
    );
    assert_close!("jac_elbow_v", black_box(elbow.linear()[1]), 1.0, 1e-12, 0.0);
    assert_close!(
        "jac_shoulder_w",
        black_box(shoulder.angular()[2]),
        1.0,
        1e-12,
        0.0
    );
    assert_close!("jac_weld", black_box(weld.linear().norm()), 0.0, 1e-12, 0.0);
    black_box(shoulder.linear()[1])
}
