use multicalc::error::EstimationError;
use multicalc::estimation::{GaussianLikelihood, KalmanFilter, ParticleFilter, ResamplingScheme};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::random::{Pcg32, RandomSource};
use multicalc::scalar::{Numeric, VectorFn};

/// A point that stays put; its two coordinates carry over unchanged.
struct Stationary;
impl VectorFn<2, 2> for Stationary {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// Measures both coordinates of the point directly.
struct MeasureBoth;
impl VectorFn<2, 2> for MeasureBoth {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// Constant velocity: position moves by `dt · velocity`, velocity holds.
struct ConstantVelocity {
    dt: f64,
}
impl VectorFn<2, 2> for ConstantVelocity {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        let dt = S::from_f64(self.dt);
        [state[0] + dt * state[1], state[1]]
    }
}

/// Measures only the position of a position/velocity state.
struct MeasurePosition;
impl VectorFn<2, 1> for MeasurePosition {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        [state[0]]
    }
}

fn identity_covariance() -> Matrix<2, 2> {
    Matrix::new([[1.0, 0.0], [0.0, 1.0]])
}

fn small_noise() -> Matrix<2, 2> {
    Matrix::new([[0.01, 0.0], [0.0, 0.01]])
}

#[test]
fn weights_normalize_after_update() {
    let mut filter = ParticleFilter::<2, 2>::new(
        500,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        1,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(Matrix::new([[0.1, 0.0], [0.0, 0.1]])).unwrap();

    filter.predict(&Stationary).unwrap();
    filter
        .update(&MeasureBoth, &sensor, Vector::new([0.3, -0.2]))
        .unwrap();

    let total: f64 = filter.weights().iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-12,
        "weights should sum to one, got {total}"
    );
    assert!(
        filter.weights().iter().all(|&w| w >= 0.0),
        "weights should be non-negative"
    );
}

#[test]
fn effective_sample_size_stays_in_bounds() {
    let count = 400;
    let mut filter = ParticleFilter::<2, 2>::new(
        count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        2,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(Matrix::new([[0.2, 0.0], [0.0, 0.2]])).unwrap();

    for _ in 0..10 {
        filter.predict(&Stationary).unwrap();
        filter
            .update(&MeasureBoth, &sensor, Vector::new([0.5, 0.5]))
            .unwrap();
        let ess = filter.effective_sample_size();
        assert!(ess >= 1.0, "effective sample size below one: {ess}");
        assert!(
            ess <= count as f64 + 1e-6,
            "effective sample size above count: {ess}"
        );
    }

    // A fresh resample restores uniform weights, so the effective sample size returns to the count.
    filter.resample();
    let ess = filter.effective_sample_size();
    assert!(
        (ess - count as f64).abs() < 1e-9,
        "resampled cloud should have full sample size: {ess}"
    );
}

#[test]
fn same_seed_reproduces_estimate() {
    let build = || {
        ParticleFilter::<2, 2>::new(
            300,
            Vector::new([0.0, 0.0]),
            identity_covariance(),
            small_noise(),
            42,
        )
        .unwrap()
    };
    let sensor = GaussianLikelihood::new(Matrix::new([[0.1, 0.0], [0.0, 0.1]])).unwrap();

    let mut first = build();
    let mut second = build();
    for _ in 0..15 {
        first.predict(&Stationary).unwrap();
        first
            .update(&MeasureBoth, &sensor, Vector::new([0.4, 0.1]))
            .unwrap();
        second.predict(&Stationary).unwrap();
        second
            .update(&MeasureBoth, &sensor, Vector::new([0.4, 0.1]))
            .unwrap();
    }

    // Identical seeds and inputs produce a bit-identical estimate.
    assert_eq!(first.mean().into_array(), second.mean().into_array());
}

#[test]
fn every_scheme_covers_heavy_particles() {
    let schemes = [
        ResamplingScheme::Systematic,
        ResamplingScheme::Stratified,
        ResamplingScheme::Multinomial,
        ResamplingScheme::Residual,
    ];
    let weights = [0.1_f64, 0.7, 0.2];

    for scheme in schemes {
        let mut random = Pcg32::new(5);
        let mut counts = [0usize; 3];
        let mut indices = [0usize; 3];
        for _ in 0..10_000 {
            scheme.resample_indices(&weights, &mut random, &mut indices);
            for &index in &indices {
                assert!(index < 3, "index out of range for {scheme:?}: {index}");
                counts[index] += 1;
            }
        }
        assert!(
            counts[1] > counts[0] && counts[1] > counts[2],
            "heavy particle should appear most for {scheme:?}: {counts:?}"
        );
    }
}

#[test]
fn incompatible_measurement_degenerates() {
    let mut filter = ParticleFilter::<2, 2>::new(
        200,
        Vector::new([0.0, 0.0]),
        small_noise(),
        small_noise(),
        3,
    )
    .unwrap();

    // A tight sensor and a measurement so far away that every squared mismatch overflows: no
    // particle can explain it, so the whole cloud dies rather than the filter panicking.
    let sensor = GaussianLikelihood::new(Matrix::new([[0.001, 0.0], [0.0, 0.001]])).unwrap();
    let result = filter.update(&MeasureBoth, &sensor, Vector::new([1e200, 1e200]));
    assert_eq!(result, Err(EstimationError::WeightsDegenerate));
}

#[test]
fn non_positive_definite_noise_is_rejected() {
    let not_positive_definite = Matrix::new([[1.0, 2.0], [2.0, 1.0]]);

    let filter = ParticleFilter::<2, 2>::new(
        100,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        not_positive_definite,
        4,
    );
    assert_eq!(filter.err(), Some(EstimationError::NotPositiveDefinite));

    let likelihood = GaussianLikelihood::<2>::new(not_positive_definite);
    assert_eq!(likelihood.err(), Some(EstimationError::NotPositiveDefinite));
}

#[test]
fn zero_particle_count_is_rejected() {
    let filter = ParticleFilter::<2, 2>::new(
        0,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        4,
    );
    assert_eq!(filter.err(), Some(EstimationError::WeightsDegenerate));
}

#[test]
fn converges_to_kalman_on_linear_gaussian_model() {
    let dt = 1.0;
    let process_noise = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
    let measurement_noise = Matrix::new([[0.09]]);
    let measurement_standard_deviation = 0.3;

    // The two filters share one initial guess and one measurement sequence.
    let mut kalman = KalmanFilter::new(
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        Matrix::new([[1.0, dt], [0.0, 1.0]]),
        Matrix::new([[1.0, 0.0]]),
        process_noise,
        measurement_noise,
    );
    let mut particle = ParticleFilter::<2, 1>::new(
        20_000,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        process_noise,
        7,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(measurement_noise).unwrap();

    let process = ConstantVelocity { dt };
    let mut measurement_random = Pcg32::new(99);
    let mut truth = [0.0_f64, 1.0];

    for _ in 0..40 {
        truth = process.eval(&truth);
        let measurement =
            truth[0] + measurement_standard_deviation * measurement_random.standard_normal();

        kalman.predict();
        kalman.update(Vector::new([measurement])).unwrap();

        particle.predict(&process).unwrap();
        particle
            .update(&MeasurePosition, &sensor, Vector::new([measurement]))
            .unwrap();
    }

    let kalman_state = kalman.state();
    let particle_mean = particle.mean();
    assert!(
        (particle_mean[0] - kalman_state[0]).abs() < 0.05,
        "position off the Kalman estimate: {} vs {}",
        particle_mean[0],
        kalman_state[0]
    );
    assert!(
        (particle_mean[1] - kalman_state[1]).abs() < 0.10,
        "velocity off the Kalman estimate: {} vs {}",
        particle_mean[1],
        kalman_state[1]
    );
    assert!(
        (particle_mean[0] - truth[0]).abs() < 0.15,
        "position off the truth: {} vs {}",
        particle_mean[0],
        truth[0]
    );
}

#[test]
fn closure_update_matches_the_model_update() {
    // The same stationary point, scored two ways from one seed: through the measurement model and
    // its Gaussian likelihood, and through a closure computing the same Gaussian log-weight by hand.
    // Both paths should land on the same estimate.
    let measurement = Vector::new([0.3, -0.2]);
    let noise = 0.1;

    let build = || {
        ParticleFilter::<2, 2>::new(
            1000,
            Vector::new([0.0, 0.0]),
            identity_covariance(),
            small_noise(),
            11,
        )
        .unwrap()
    };

    let mut through_model = build();
    let sensor = GaussianLikelihood::new(Matrix::new([[noise, 0.0], [0.0, noise]])).unwrap();

    let mut through_closure = build();

    for _ in 0..15 {
        through_model.predict(&Stationary).unwrap();
        through_model
            .update(&MeasureBoth, &sensor, measurement)
            .unwrap();

        through_closure.predict(&Stationary).unwrap();
        through_closure
            .update_with_log_weights(|particle| {
                // The Gaussian log-weight for isotropic noise: −½ · |measurement − particle|² / σ².
                let dx = measurement[0] - particle[0];
                let dy = measurement[1] - particle[1];
                -0.5 * (dx * dx + dy * dy) / noise
            })
            .unwrap();
    }

    let model_mean = through_model.mean();
    let closure_mean = through_closure.mean();
    assert_eq!(
        model_mean.into_array(),
        closure_mean.into_array(),
        "closure scoring should match the model update exactly"
    );
}

#[test]
fn a_closure_that_favours_one_region_moves_the_mean() {
    let mut filter = ParticleFilter::<2, 2>::new(
        2000,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        22,
    )
    .unwrap();

    let target = [1.5, -1.0];
    for _ in 0..15 {
        filter.predict(&Stationary).unwrap();
        filter
            .update_with_log_weights(|particle| {
                let dx = target[0] - particle[0];
                let dy = target[1] - particle[1];
                -0.5 * (dx * dx + dy * dy) / 0.05
            })
            .unwrap();
    }

    let mean = filter.mean();
    assert!(
        (mean[0] - target[0]).abs() < 0.2 && (mean[1] - target[1]).abs() < 0.2,
        "the mean should follow the favoured region: {mean:?}"
    );
}

#[test]
fn a_zero_score_closure_leaves_the_weights_uniform() {
    let count = 500;
    let mut filter = ParticleFilter::<2, 2>::new(
        count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        33,
    )
    .unwrap();

    // Scoring every particle the same leaves the weights uniform, so nothing resamples and the
    // effective sample size stays at the full count.
    filter.update_with_log_weights(|_| 0.0).unwrap();

    let ess = filter.effective_sample_size();
    assert!(
        (ess - count as f64).abs() < 1e-9,
        "a flat score should leave the full sample size: {ess}"
    );
}
