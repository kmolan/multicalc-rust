//! Estimation: the linear and extended Kalman filters. The linear filter tracks a constant-velocity
//! target — an exact two-step hand check, a noisy track, Joseph vs naive covariance, a control
//! input, innovation gating, and one `Dual` derivative through an update. The extended filter adds a
//! nonlinear landmark range/bearing sighting, its reduction to the linear filter, and angle wrapping.
//! With `--features alloc`, a closing section locates a differential-drive robot from beacon ranges
//! with a particle filter.
//!
//! Run with: `cargo run -p multicalc-demos --example estimation`
//! (or `--no-default-features --features alloc` for the particle-filter section).

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
    (0..N).map(|i| m.get(i, i).copied().unwrap()).sum()
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

/// The beacons a robot ranges against to find itself, at the corners of a 10 m square room.
/// Example taken from https://github.com/destenson/kalman-filter-rs/blob/master/examples/particle_filter_robot.rs
#[cfg(feature = "alloc")]
const BEACONS: [(f64, f64); 4] = [(2.0, 8.0), (8.0, 8.0), (8.0, 2.0), (2.0, 2.0)];

/// A differential-drive robot rolling at a fixed forward and turn rate over one timestep: the pose
/// `[x, y, heading]` moves along the heading and the heading turns. The rates are fields the caller
/// sets between steps.
#[cfg(feature = "alloc")]
struct DifferentialDrive {
    forward_speed: f64,
    turn_rate: f64,
    timestep: f64,
}
#[cfg(feature = "alloc")]
impl VectorFn<3, 3> for DifferentialDrive {
    fn eval<S: Numeric>(&self, pose: &[S; 3]) -> [S; 3] {
        let forward = S::from_f64(self.forward_speed) * S::from_f64(self.timestep);
        let heading = pose[2];
        [
            pose[0] + forward * heading.cos(),
            pose[1] + forward * heading.sin(),
            heading + S::from_f64(self.turn_rate) * S::from_f64(self.timestep),
        ]
    }
}

/// The sensor: the straight-line distance from the robot's position to each beacon. Heading does not
/// affect a range, so this reads position only.
#[cfg(feature = "alloc")]
struct BeaconRanges;
#[cfg(feature = "alloc")]
impl VectorFn<3, 4> for BeaconRanges {
    fn eval<S: Numeric>(&self, pose: &[S; 3]) -> [S; 4] {
        core::array::from_fn(|i| {
            let to_beacon_x = pose[0] - S::from_f64(BEACONS[i].0);
            let to_beacon_y = pose[1] - S::from_f64(BEACONS[i].1);
            (to_beacon_x * to_beacon_x + to_beacon_y * to_beacon_y).sqrt()
        })
    }
}

/// Locates a differential-drive robot from beacon ranges with a particle filter. The robot knows how
/// fast it is driving (its odometry) but not where it started, so the cloud begins spread wide over
/// the room and pulls in as the beacon ranges rule out where the robot cannot be — the kind of
/// many-hypotheses belief a single Gaussian cannot hold.
#[cfg(feature = "alloc")]
fn particle_filter() {
    use multicalc::estimation::{GaussianLikelihood, ParticleFilter, ResamplingScheme};
    use multicalc::random::{Pcg32, RandomSource};

    let particle_count = 1500;
    let timestep = 1.0;
    let motion = DifferentialDrive {
        forward_speed: 0.4,
        turn_rate: 0.16,
        timestep,
    };
    let range_noise = 0.3; // metres, one standard deviation

    // The prior: the robot guesses it is near the middle of the room but could be a metre or two off
    // in any direction and pointing almost anywhere.
    let mut filter = ParticleFilter::<3, 4>::new(
        particle_count,
        Vector::new([4.0, 5.0, 0.0]), // initial mean, offset from the true start
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]), // wide prior
        Matrix::new([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.01]]), // odometry noise
        7,
    )
    .unwrap()
    .with_resampling(ResamplingScheme::Systematic);
    let sensor =
        GaussianLikelihood::new(Matrix::<4, 4>::identity().scale(range_noise * range_noise))
            .unwrap();

    // The true pose the robot actually follows; the filter never sees it. Each measurement is the
    // true ranges plus sensor noise, drawn from a seeded generator so the run reproduces exactly.
    let mut truth = [5.0_f64, 4.0, 0.3];
    let mut range_generator = Pcg32::new(20);

    println!("\nParticle filter: locating a robot from 4 beacon ranges with a 1500-sample cloud");
    println!("  step |  est x  est y  est θ |  ess");
    for step in 1..=16 {
        truth = motion.eval(&truth);
        let true_ranges = BeaconRanges.eval(&truth);
        let measurement = Vector::new(core::array::from_fn(|i| {
            true_ranges[i] + range_noise * range_generator.standard_normal()
        }));

        filter.predict(&motion).unwrap();
        filter.update(&BeaconRanges, &sensor, measurement).unwrap();

        let estimate = *filter.mean().as_array();
        println!(
            "  {step:4} | {:6.2} {:6.2} {:6.2} | {:5.0}",
            estimate[0],
            estimate[1],
            estimate[2],
            filter.effective_sample_size(),
        );
    }

    let estimate = *filter.mean().as_array();
    let best = *filter.maximum_a_posteriori_state().as_array();
    let heaviest_weight = filter.weights().iter().copied().fold(0.0_f64, f64::max);
    let position_spread = position_spread(&filter);
    println!(
        "  true position ({:.2}, {:.2})  ->  estimate ({:.2}, {:.2})",
        truth[0], truth[1], estimate[0], estimate[1]
    );
    println!(
        "  heaviest particle ({:.2}, {:.2}) carrying weight {:.4}",
        best[0], best[1], heaviest_weight
    );
    if position_spread > 1.0 {
        println!(
            "  the cloud is still spread ({position_spread:.2} m) — the robot is not sure yet"
        );
    } else {
        println!("  the cloud has pulled in ({position_spread:.2} m) — the robot is located");
    }

    let position_error =
        ((estimate[0] - truth[0]).powi(2) + (estimate[1] - truth[1]).powi(2)).sqrt();
    assert!(
        position_error < 0.4,
        "the cloud should settle onto the true position"
    );
    assert!(
        (1.0..=particle_count as f64 + 1e-6).contains(&filter.effective_sample_size()),
        "effective sample size stays between one and the particle count"
    );
}

/// How far the cloud reaches in position: the square root of the summed weighted variance of the x
/// and y coordinates, a plain-metres measure of how sure the filter is.
#[cfg(feature = "alloc")]
fn position_spread<R: multicalc::random::RandomSource>(
    filter: &multicalc::estimation::ParticleFilter<3, 4, f64, R>,
) -> f64 {
    let mean = *filter.mean().as_array();
    let mut variance = 0.0;
    for (particle, &weight) in filter.particles().iter().zip(filter.weights()) {
        let p = *particle.as_array();
        variance += weight * ((p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2));
    }
    variance.sqrt()
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
    report("position", filter.state().as_array()[0], 5.0 / 3.0);
    report("velocity", filter.state().as_array()[1], 2.0 / 3.0);
    report(
        "P[0,0]",
        filter.covariance().get(0, 0).copied().unwrap(),
        2.0 / 3.0,
    );
    report(
        "P[0,1] = P[1,0]",
        filter.covariance().get(0, 1).copied().unwrap(),
        1.0 / 3.0,
    );
    report(
        "innovation covariance",
        filter.innovation_covariance().get(0, 0).copied().unwrap(),
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
            filter.state().as_array()[0],
            filter.state().as_array()[1],
            trace(filter.covariance()),
        );
    }
    let (position, velocity) = (filter.state().as_array()[0], filter.state().as_array()[1]);
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
    report(
        "state agreement",
        (naive.state().as_array()[0] - position).abs(),
        0.0,
    );
    report(
        "Joseph symmetry err",
        (filter.covariance().get(0, 1).copied().unwrap()
            - filter.covariance().get(1, 0).copied().unwrap())
        .abs(),
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
        driven.state().as_array()[0],
        driven.state().as_array()[1]
    );
    assert!(
        driven.state().as_array()[1] > 1.0,
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
    let prior_position_variance = reference.covariance().get(0, 0).copied().unwrap();
    reference.update(Vector::new([1.0])).unwrap();
    let gain = prior_position_variance
        / reference
            .innovation_covariance()
            .get(0, 0)
            .copied()
            .unwrap();

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
    report(
        "d(position)/dz",
        differentiated.state().as_array()[0].deriv,
        gain,
    );

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
        (reduced.state().as_array()[0] - linear.state().as_array()[0]).abs(),
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
        unwrapped.state().as_array()[0]
    );
    println!(
        "  update_with_residual (wrapped): heading -> {:>7.3}",
        wrapped.state().as_array()[0]
    );
    assert!(
        wrapped.state().as_array()[0] > 3.0,
        "the wrapped residual nudges the estimate a little past +π"
    );
    assert!(
        unwrapped.state().as_array()[0] < 0.0,
        "the plain residual throws the estimate across the circle"
    );

    // (10) A particle filter, the nonlinear, non-Gaussian cousin of the Kalman filters: it carries a
    // cloud of weighted samples instead of one Gaussian. It is heap-backed, so it lives behind the
    // `alloc` feature; without it, the rest of the example still runs.
    #[cfg(feature = "alloc")]
    particle_filter();
    #[cfg(not(feature = "alloc"))]
    println!("\nParticle-filter section needs --features alloc");
}
