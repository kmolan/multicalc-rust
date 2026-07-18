//! Estimation: the linear and extended Kalman filters. The linear filter tracks a constant-velocity
//! target — an exact two-step hand check, a noisy track, Joseph vs naive covariance, a control
//! input, innovation gating, and one `Dual` derivative through an update. The extended filter adds a
//! nonlinear landmark range/bearing sighting, its reduction to the linear filter, and angle wrapping.
//!
//! Run with: `cargo run -p multicalc-demos --example estimation`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::discretization::q_discrete_white_noise;
use multicalc::estimation::{CovarianceUpdate, ExtendedKalmanFilter, KalmanFilter};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric, VectorFn};

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<22} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

/// A constant-velocity tracker over a 1 s step: position integrates velocity, position is measured.
fn tracker<T: Numeric>(
    initial_covariance: Matrix<2, 2, T>,
    process_noise: Matrix<2, 2, T>,
    measurement_noise: Matrix<1, 1, T>,
) -> KalmanFilter<2, 1, T> {
    KalmanFilter::new(
        Vector::new([T::ZERO, T::ZERO]),
        initial_covariance,
        Matrix::new([[T::ONE, T::ONE], [T::ZERO, T::ONE]]),
        Matrix::new([[T::ONE, T::ZERO]]),
        process_noise,
        measurement_noise,
    )
}

fn trace<const N: usize>(m: Matrix<N, N>) -> f64 {
    (0..N).map(|i| m[(i, i)]).sum()
}

/// Range and bearing to a known landmark, from a [x, y, heading] pose. Written once; its Jacobian
/// is taken by automatic differentiation, never derived by hand.
struct LandmarkRangeAndBearing {
    landmark_x: f64,
    landmark_y: f64,
}
impl VectorFn<3, 2> for LandmarkRangeAndBearing {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 2] {
        let to_landmark_x = S::from_f64(self.landmark_x) - state[0];
        let to_landmark_y = S::from_f64(self.landmark_y) - state[1];
        [
            (to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt(),
            to_landmark_y.atan2(to_landmark_x) - state[2],
        ]
    }
}

/// The constant-velocity transition `[[1, 1], [0, 1]]` as a function.
struct ConstantVelocityMotion;
impl VectorFn<2, 2> for ConstantVelocityMotion {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0] + state[1], state[1]]
    }
}

/// Position is measured, velocity is not.
struct PositionMeasurement;
impl VectorFn<2, 1> for PositionMeasurement {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        [state[0]]
    }
}

/// A compass reading the heading directly.
struct Compass;
impl VectorFn<1, 1> for Compass {
    fn eval<S: Numeric>(&self, state: &[S; 1]) -> [S; 1] {
        [state[0]]
    }
}

/// Folds an angle into a ±π band by subtracting whole turns.
fn wrap_to_pi<T: Numeric>(angle: T) -> T {
    angle - T::TWO_PI * (angle / T::TWO_PI).round()
}

fn main() {
    // (1) Two steps with no process noise, worked by hand: with P0 = I, R = 1 and z = [1, 2],
    // the posterior is x = [5/3, 2/3] and P = [[2/3, 1/3], [1/3, 1/3]].
    let mut filter = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[1.0]]));
    for z in [1.0, 2.0] {
        filter.predict();
        filter.update(Vector::new([z])).unwrap();
    }
    println!("Exact two-step filter (P0 = I, Q = 0, R = 1, z = [1, 2])");
    report("position", filter.state()[0], 5.0 / 3.0);
    report("velocity", filter.state()[1], 2.0 / 3.0);
    report("P[0,0]", filter.covariance()[(0, 0)], 2.0 / 3.0);
    report("P[0,1] = P[1,0]", filter.covariance()[(0, 1)], 1.0 / 3.0);
    report(
        "innovation covariance",
        filter.innovation_covariance()[(0, 0)],
        3.0,
    );

    // (2) Track a target moving at 1 m/s from a standing start, measuring position only.
    // Truth is position = t; the measurements below are that track plus fixed noise.
    let measurements = [1.15, 1.87, 3.21, 3.94, 5.02, 6.13, 6.89, 8.05];
    let mut filter = tracker::<f64>(
        Matrix::new([[10.0, 0.0], [0.0, 10.0]]),
        q_discrete_white_noise::<2, f64>(1.0, 0.01),
        Matrix::new([[0.25]]),
    );
    let initial_uncertainty = trace(filter.covariance());
    println!("\nTracking a 1 m/s target from x = [0, 0] with a wide prior");
    for (step, z) in measurements.iter().enumerate() {
        filter.predict();
        filter.update(Vector::new([*z])).unwrap();
        println!(
            "  t = {:>2}s  measured {:>5.2}  ->  position {:>6.3}, velocity {:>6.3}  (trace P {:>7.4})",
            step + 1,
            z,
            filter.state()[0],
            filter.state()[1],
            trace(filter.covariance()),
        );
    }
    let (position, velocity) = (filter.state()[0], filter.state()[1]);
    assert!(
        (position - 8.0).abs() < 0.3,
        "position should converge on the truth track"
    );
    assert!(
        (velocity - 1.0).abs() < 0.3,
        "velocity should converge on 1 m/s"
    );
    assert!(
        trace(filter.covariance()) < initial_uncertainty,
        "measurements must reduce uncertainty"
    );
    println!("  velocity recovered from position measurements alone: {velocity:.4} m/s");

    // (3) Joseph vs naive. They agree in exact arithmetic; Joseph keeps symmetry under rounding.
    let mut naive = tracker::<f64>(
        Matrix::new([[10.0, 0.0], [0.0, 10.0]]),
        q_discrete_white_noise::<2, f64>(1.0, 0.01),
        Matrix::new([[0.25]]),
    )
    .with_covariance_update(CovarianceUpdate::Naive);
    for z in measurements.iter() {
        naive.predict();
        naive.update(Vector::new([*z])).unwrap();
    }
    println!("\nJoseph (default) vs naive covariance update");
    report("state agreement", (naive.state()[0] - position).abs(), 0.0);
    report(
        "Joseph symmetry err",
        (filter.covariance()[(0, 1)] - filter.covariance()[(1, 0)]).abs(),
        0.0,
    );
    assert!(
        filter.covariance().cholesky().is_ok(),
        "Joseph stays positive definite"
    );

    // (4) A driven system: constant acceleration enters through the control model.
    let mut driven = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[0.25]]));
    let control_model = Matrix::<2, 1>::new([[0.5], [1.0]]); // [dt²/2; dt] for dt = 1
    for z in [0.4, 2.1, 4.4, 8.2] {
        driven.predict_with_control(control_model, Vector::new([1.0])); // 1 m/s² command
        driven.update(Vector::new([z])).unwrap();
    }
    println!("\nDriven filter (1 m/s² command through the control model)");
    println!(
        "  after 4 s: position {:.3}, velocity {:.3}",
        driven.state()[0],
        driven.state()[1]
    );
    assert!(
        driven.state()[1] > 1.0,
        "a sustained acceleration command must build velocity"
    );

    // (5) Innovation gating: the normalized innovation squared flags an outlier measurement.
    let mut gated = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[0.25]]));
    gated.predict();
    gated.update(Vector::new([1.0])).unwrap();
    let consistent = gated.normalized_innovation_squared().unwrap();
    gated.predict();
    gated.update(Vector::new([50.0])).unwrap();
    let outlier = gated.normalized_innovation_squared().unwrap();
    println!("\nInnovation gating (yᵀ·S⁻¹·y)");
    println!("  consistent measurement: {consistent:.3}");
    println!("  outlier  measurement:   {outlier:.3}");
    assert!(
        outlier > consistent,
        "an outlier must score higher than a consistent measurement"
    );

    // (6) Autodiff: d(posterior position)/d(measurement) is the Kalman gain. Check it against the
    // gain formed independently from the prior covariance and the innovation covariance.
    let mut reference = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[0.25]]));
    reference.predict();
    let prior_position_variance = reference.covariance()[(0, 0)];
    reference.update(Vector::new([1.0])).unwrap();
    let gain = prior_position_variance / reference.innovation_covariance()[(0, 0)];

    let mut differentiated = tracker::<Dual<f64>>(
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::new([[Dual::constant(0.25)]]),
    );
    differentiated.predict();
    differentiated
        .update(Vector::new([Dual::variable(1.0)]))
        .unwrap();

    println!("\nAutodiff: d(position)/d(measurement) equals the Kalman gain");
    report("d(position)/dz", differentiated.state()[0].deriv, gain);

    // (7) The same tracking problem, made nonlinear: a range-and-bearing sighting of a known
    // landmark. The measurement model is written once as a plain function; its Jacobian is taken by
    // automatic differentiation, so there are no hand-derived Jacobians anywhere.
    let mut extended = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]), // pose [x, y, heading] at the origin
        Matrix::<3, 3>::identity(),   // a wide, uncertain prior
        Matrix::zeros(),
        Matrix::<2, 2>::identity().scale(0.01), // a precise range/bearing sensor
    );
    let landmark = LandmarkRangeAndBearing {
        landmark_x: 3.0,
        landmark_y: 4.0,
    };
    let uncertainty_before = trace(extended.covariance());
    // From the true pose the landmark is at range 5, bearing atan2(4, 3).
    extended
        .update(&landmark, Vector::new([5.0, 4.0_f64.atan2(3.0)]))
        .unwrap();
    let uncertainty_after = trace(extended.covariance());
    println!("\nExtended filter: one range/bearing sighting of a landmark at (3, 4)");
    println!("  uncertainty (trace P): {uncertainty_before:.3} -> {uncertainty_after:.3}");
    assert!(
        uncertainty_after < uncertainty_before,
        "a sighting must reduce uncertainty"
    );

    // (8) With linear models the extended filter reproduces the linear one exactly — that identity
    // is what "extended" means. Run the same constant-velocity models through both and compare.
    let mut reduced = ExtendedKalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        Matrix::identity(),
        Matrix::<2, 2>::identity().scale(0.01),
        Matrix::new([[0.5]]),
    );
    let mut linear = tracker::<f64>(
        Matrix::identity(),
        Matrix::<2, 2>::identity().scale(0.01),
        Matrix::new([[0.5]]),
    );
    for z in [0.5, 1.0, 1.5, 2.0] {
        reduced.predict(&ConstantVelocityMotion).unwrap();
        reduced
            .update(&PositionMeasurement, Vector::new([z]))
            .unwrap();
        linear.predict();
        linear.update(Vector::new([z])).unwrap();
    }
    println!("\nExtended filter reduces to the linear filter on linear models");
    report(
        "state agreement",
        (reduced.state()[0] - linear.state()[0]).abs(),
        0.0,
    );

    // (9) Angle wrapping: a heading just under +π measured by a compass reading just over −π. Plain
    // subtraction reads the ~0.08 rad error as nearly a full turn and throws the estimate the wrong
    // way; wrapping the residual to a ±π band fixes it. This is the trap `update_with_residual`
    // exists for.
    let measurement = -3.1; // compass reads just over −π; the state is just under +π

    let mut unwrapped = ExtendedKalmanFilter::<1, 1>::new(
        Vector::new([3.1]),
        Matrix::new([[0.1]]),
        Matrix::zeros(),
        Matrix::new([[0.05]]),
    );
    unwrapped
        .update(&Compass, Vector::new([measurement]))
        .unwrap();

    let mut wrapped = ExtendedKalmanFilter::<1, 1>::new(
        Vector::new([3.1]),
        Matrix::new([[0.1]]),
        Matrix::zeros(),
        Matrix::new([[0.05]]),
    );
    let predicted = Compass.eval(wrapped.state().as_array())[0];
    let residual = wrap_to_pi(measurement - predicted);
    wrapped
        .update_with_residual(&Compass, Vector::new([residual]))
        .unwrap();

    println!("\nAngle wrapping near ±π (true heading error ≈ 0.08 rad)");
    println!(
        "  update (plain subtraction): heading -> {:>7.3}",
        unwrapped.state()[0]
    );
    println!(
        "  update_with_residual (wrapped): heading -> {:>7.3}",
        wrapped.state()[0]
    );
    assert!(
        wrapped.state()[0] > 3.0,
        "the wrapped residual nudges the estimate a little past +π"
    );
    assert!(
        unwrapped.state()[0] < 0.0,
        "the plain residual throws the estimate across the circle"
    );
}
