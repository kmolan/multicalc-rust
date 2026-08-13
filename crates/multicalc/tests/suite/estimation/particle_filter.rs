use multicalc::error::EstimationError;
use multicalc::estimation::{
    GaussianLikelihood, KalmanFilter, KalmanModel, Likelihood, ParticleFilter, ResamplingScheme,
};
use multicalc::linear_algebra::{Matrix, Matrix2D, Vector};
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

/// Constant velocity: position moves by `timestep · velocity`, velocity holds.
struct ConstantVelocity {
    timestep: f64,
}
impl VectorFn<2, 2> for ConstantVelocity {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        let timestep = S::from_f64(self.timestep);
        [state[0] + timestep * state[1], state[1]]
    }
}

/// Measures only the position of a position/velocity state.
struct MeasurePosition;
impl VectorFn<2, 1> for MeasurePosition {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        [state[0]]
    }
}

fn identity_covariance() -> Matrix2D {
    Matrix::new([[1.0, 0.0], [0.0, 1.0]])
}

fn small_noise() -> Matrix2D {
    Matrix::new([[0.01, 0.0], [0.0, 0.01]])
}

#[test]
fn weights_normalize_after_update() {
    let particle_count = 500;
    let seed = 1;
    let mut filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        seed,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(Matrix::new([[0.1, 0.0], [0.0, 0.1]])).unwrap();

    filter.predict(&Stationary).unwrap();
    filter
        .update(&MeasureBoth, &sensor, Vector::new([0.3, -0.2]))
        .unwrap();

    let weight_sum: f64 = filter.weights().iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-12,
        "weights should sum to one, got {weight_sum}"
    );
    assert!(
        filter.weights().iter().all(|&weight| weight >= 0.0),
        "weights should be non-negative"
    );
}

#[test]
fn effective_sample_size_stays_in_bounds() {
    let particle_count = 400;
    let seed = 2;
    let mut filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        seed,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(Matrix::new([[0.2, 0.0], [0.0, 0.2]])).unwrap();

    for _ in 0..10 {
        filter.predict(&Stationary).unwrap();
        filter
            .update(&MeasureBoth, &sensor, Vector::new([0.5, 0.5]))
            .unwrap();
        let effective_sample_size = filter.effective_sample_size();
        assert!(
            effective_sample_size >= 1.0,
            "effective sample size below one: {effective_sample_size}"
        );
        assert!(
            effective_sample_size <= particle_count as f64 + 1e-6,
            "effective sample size above count: {effective_sample_size}"
        );
    }

    // A fresh resample restores uniform weights, so the effective sample size returns to the count.
    filter.resample();
    let effective_sample_size = filter.effective_sample_size();
    assert!(
        (effective_sample_size - particle_count as f64).abs() < 1e-9,
        "resampled cloud should have full sample size: {effective_sample_size}"
    );
}

#[test]
fn same_seed_reproduces_estimate() {
    let particle_count = 300;
    let seed = 42;
    let build = || {
        ParticleFilter::<2, 2>::new(
            particle_count,
            Vector::new([0.0, 0.0]),
            identity_covariance(),
            small_noise(),
            seed,
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

    let seed = 5;

    for scheme in schemes {
        let mut random = Pcg32::new(seed);
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
fn every_scheme_leaves_indices_unchanged_for_empty_weights() {
    let schemes = [
        ResamplingScheme::Systematic,
        ResamplingScheme::Stratified,
        ResamplingScheme::Multinomial,
        ResamplingScheme::Residual,
    ];
    let weights: [f64; 0] = [];

    for scheme in schemes {
        let mut random = Pcg32::new(5);
        let initial_random = random.clone();
        let mut indices = [7usize, 8, 9];

        scheme.resample_indices(&weights, &mut random, &mut indices);

        assert_eq!(random, initial_random);
        assert_eq!(
            indices,
            [7, 8, 9],
            "empty weights should leave indices untouched for {scheme:?}"
        );
    }
}

#[test]
fn incompatible_measurement_degenerates() {
    let particle_count = 200;
    let seed = 3;
    let mut filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        small_noise(),
        small_noise(),
        seed,
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

    let particle_count = 100;
    let seed = 4;
    let filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        not_positive_definite,
        seed,
    );
    assert_eq!(filter.err(), Some(EstimationError::NotPositiveDefinite));

    let likelihood = GaussianLikelihood::<2>::new(not_positive_definite);
    assert_eq!(likelihood.err(), Some(EstimationError::NotPositiveDefinite));
}

#[test]
fn zero_particle_count_is_rejected() {
    let particle_count = 0;
    let seed = 4;
    let filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        seed,
    );
    assert_eq!(filter.err(), Some(EstimationError::WeightsDegenerate));
}

#[test]
fn converges_to_kalman_on_linear_gaussian_model() {
    let timestep = 1.0;
    let process_noise = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
    let measurement_noise = Matrix::new([[0.09]]);
    let measurement_standard_deviation = 0.3;

    // The two filters share one initial guess and one measurement sequence.
    let mut kalman = KalmanFilter::new(
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        KalmanModel {
            state_transition: Matrix::new([[1.0, timestep], [0.0, 1.0]]),
            measurement_model: Matrix::new([[1.0, 0.0]]),
            process_noise,
            measurement_noise,
        },
    );
    let particle_count = 20_000;
    let seed = 7;
    let mut particle = ParticleFilter::<2, 1>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        process_noise,
        seed,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(measurement_noise).unwrap();

    let process = ConstantVelocity { timestep };
    let measurement_seed = 113;
    let mut measurement_random = Pcg32::<f64>::new(measurement_seed);
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

    assert!(
        (particle.mean()[0] - kalman.state()[0]).abs() < 0.05,
        "position off the Kalman estimate: {} vs {}",
        particle.mean()[0],
        kalman.state()[0]
    );
    assert!(
        (particle.mean()[1] - kalman.state()[1]).abs() < 0.10,
        "velocity off the Kalman estimate: {} vs {}",
        particle.mean()[1],
        kalman.state()[1]
    );
    assert!(
        (particle.mean()[0] - truth[0]).abs() < 0.15,
        "position off the truth: {} vs {}",
        particle.mean()[0],
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

    let particle_count = 1000;
    let seed = 11;
    let build = || {
        ParticleFilter::<2, 2>::new(
            particle_count,
            Vector::new([0.0, 0.0]),
            identity_covariance(),
            small_noise(),
            seed,
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
                let offset_x = measurement[0] - particle[0];
                let offset_y = measurement[1] - particle[1];
                -0.5 * (offset_x * offset_x + offset_y * offset_y) / noise
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
    let particle_count = 2000;
    let seed = 22;
    let mut filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        seed,
    )
    .unwrap();

    let target = [1.5, -1.0];
    for _ in 0..15 {
        filter.predict(&Stationary).unwrap();
        filter
            .update_with_log_weights(|particle| {
                let offset_x = target[0] - particle[0];
                let offset_y = target[1] - particle[1];
                -0.5 * (offset_x * offset_x + offset_y * offset_y) / 0.05
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
    let particle_count = 500;
    let seed = 33;
    let mut filter = ParticleFilter::<2, 2>::new(
        particle_count,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        seed,
    )
    .unwrap();

    // Scoring every particle the same leaves the weights uniform, so nothing resamples and the
    // effective sample size stays at the full count.
    filter.update_with_log_weights(|_| 0.0).unwrap();

    let effective_sample_size = filter.effective_sample_size();
    assert!(
        (effective_sample_size - particle_count as f64).abs() < 1e-9,
        "a flat score should leave the full sample size: {effective_sample_size}"
    );
}

#[test]
fn particle_recovers_after_its_exported_weight_underflows() {
    let mut filter = ParticleFilter::<2, 2>::new(
        2,
        Vector::new([0.0, 0.0]),
        identity_covariance(),
        small_noise(),
        44,
    )
    .unwrap()
    .with_resample_threshold(0.0);
    let recovering_particle = filter.particles()[0].into_array();

    struct Scores {
        recovering_particle: [f64; 2],
        recovering_score: f64,
        other_score: f64,
    }

    impl Likelihood<2, f64> for Scores {
        fn log_weight(&self, predicted: &[f64; 2], _measurement: &[f64; 2]) -> f64 {
            if *predicted == self.recovering_particle {
                self.recovering_score
            } else {
                self.other_score
            }
        }
    }

    // Push one displayed weight below f64's range without resampling the particle away.
    filter
        .update(
            &MeasureBoth,
            &Scores {
                recovering_particle,
                recovering_score: -1000.0,
                other_score: 0.0,
            },
            Vector::new([0.0, 0.0]),
        )
        .unwrap();
    assert_eq!(filter.weights()[0], 0.0);

    // A later observation reverses the evidence. The particle must still have a finite internal
    // prior even though its exported linear weight rounded to zero.
    filter
        .update(
            &MeasureBoth,
            &Scores {
                recovering_particle,
                recovering_score: 0.0,
                other_score: -2000.0,
            },
            Vector::new([0.0, 0.0]),
        )
        .unwrap();
    assert!(
        filter.weights()[0] > 1.0 - 1e-12,
        "the previously underflowed particle should recover: {:?}",
        filter.weights()
    );
}
