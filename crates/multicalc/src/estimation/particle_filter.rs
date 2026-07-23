//! Particle filtering for nonlinear, non-Gaussian estimation.
//!
//! Where the Kalman filters carry a single Gaussian belief, the [`ParticleFilter`] carries a cloud of
//! weighted state samples, so it can track a belief with several peaks or a strongly nonlinear one.
//! [`predict`](ParticleFilter::predict) pushes every sample through the process model and adds a draw
//! of process noise; [`update`](ParticleFilter::update) reweights each sample by how well it explains
//! the measurement and normalizes; when the cloud degenerates the filter resamples, drawing a fresh
//! equally-weighted cloud that favors the heavy samples.
//!
//! Non-finite policy: every step is checked. `predict` returns
//! [`NonFinite`](EstimationError::NonFinite) when any propagated sample holds an infinity or NaN, and
//! `update` does the same for the measurement, returning
//! [`WeightsDegenerate`](EstimationError::WeightsDegenerate) when every weight underflows to zero — a
//! measurement no sample can explain.
//!
//! Memory: `2·particle_count` state vectors plus `particle_count` weights, log-weights, and indices
//! live on the heap; the second state buffer is reused scratch for resampling, so a steady run
//! allocates nothing per step. A particle filter wants hundreds to thousands of particles, so this is
//! a heap, std/`alloc` type — the bare-metal target does not build it.
//!

use alloc::vec;
use alloc::vec::Vec;

use crate::error::EstimationError;
use crate::linear_algebra::{Cholesky, Matrix, Vector};
use crate::random::{Pcg32, RandomSource};
use crate::scalar::{Numeric, VectorFn};

/// How the filter draws a fresh, equally-weighted cloud from the current weighted one.
///
/// Each scheme consumes uniform draws in a fixed order and count, so a recorded draw sequence
/// reproduces its result exactly. Changing that order would invalidate any recorded fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ResamplingScheme {
    /// One random offset, then evenly spaced picks. Lowest variance, the usual default.
    #[default]
    Systematic,
    /// One random pick inside each of the evenly spaced strata.
    Stratified,
    /// Each pick drawn independently from the weight distribution.
    Multinomial,
    /// Keep the whole-number share of each weight, then fill the rest by independent draws on the
    /// leftover fractions.
    Residual,
}

impl ResamplingScheme {
    /// Fills `indices` with the chosen sample positions for `weights`.
    ///
    /// `weights` must be normalized and the same length as `indices`. Uniforms are drawn from
    /// `random` in a fixed order and count so a recorded draw sequence reproduces the result exactly:
    /// one draw for [`Systematic`](Self::Systematic); one per element for
    /// [`Stratified`](Self::Stratified) and [`Multinomial`](Self::Multinomial); and, for
    /// [`Residual`](Self::Residual), one per leftover slot after the whole-number copies are laid
    /// down.
    pub fn resample_indices<T: Numeric, R: RandomSource>(
        self,
        weights: &[T],
        random: &mut R,
        indices: &mut [usize],
    ) {
        match self {
            ResamplingScheme::Systematic => systematic(weights, random, indices),
            ResamplingScheme::Stratified => stratified(weights, random, indices),
            ResamplingScheme::Multinomial => multinomial(weights, random, indices),
            ResamplingScheme::Residual => residual(weights, random, indices),
        }
    }
}

/// One random offset shared by every pick, spaced one stratum apart. Draws a single uniform.
fn systematic<T: Numeric, R: RandomSource>(weights: &[T], random: &mut R, indices: &mut [usize]) {
    let count = indices.len();
    let count_scalar = T::from_usize(count);
    let offset = T::from_f64(random.next_unit_f64());

    // Picks sit at one-stratum spacing, all shifted by the same offset. Walk a running total of the
    // weights and place each pick on the first source whose total reaches it, so heavier sources,
    // which span more of the line, catch more picks.
    let mut running_sum = weights[0];
    let mut source = 0;
    for (step, slot) in indices.iter_mut().enumerate() {
        let position = (offset + T::from_usize(step)) / count_scalar;
        while position >= running_sum && source + 1 < count {
            source += 1;
            running_sum += weights[source];
        }
        *slot = source;
    }
}

/// A fresh random pick inside each evenly spaced stratum. Draws one uniform per pick.
fn stratified<T: Numeric, R: RandomSource>(weights: &[T], random: &mut R, indices: &mut [usize]) {
    let count = indices.len();
    let count_scalar = T::from_usize(count);

    // Same walk as systematic, but each stratum gets its own random offset instead of sharing one,
    // so the picks are still one per stratum but jittered independently within them.
    let mut running_sum = weights[0];
    let mut source = 0;
    for (step, slot) in indices.iter_mut().enumerate() {
        let offset = T::from_f64(random.next_unit_f64());
        let position = (offset + T::from_usize(step)) / count_scalar;
        while position >= running_sum && source + 1 < count {
            source += 1;
            running_sum += weights[source];
        }
        *slot = source;
    }
}

/// Each pick drawn independently from the weight distribution. Draws one uniform per pick.
fn multinomial<T: Numeric, R: RandomSource>(weights: &[T], random: &mut R, indices: &mut [usize]) {
    // Every pick is an independent throw at the 0-to-1 line; each lands on the source whose weight
    // covers where it fell. Simple, but the picks clump more than the spaced schemes above.
    for slot in indices.iter_mut() {
        let draw = T::from_f64(random.next_unit_f64());
        *slot = draw_from_weights(weights, draw);
    }
}

/// The whole-number share of each weight first, then the leftover slots filled by independent draws
/// on the normalized fractional remainders. Draws one uniform per leftover slot.
fn residual<T: Numeric, R: RandomSource>(weights: &[T], random: &mut R, indices: &mut [usize]) {
    let count = indices.len();
    let count_scalar = T::from_usize(count);

    // Lay down floor(count · weight) copies of each source, in order.
    let mut filled = 0;
    for (source, &weight) in weights.iter().enumerate() {
        let target = (count_scalar * weight).floor();
        let mut laid = T::ZERO;
        while laid < target && filled < count {
            indices[filled] = source;
            filled += 1;
            laid += T::ONE;
        }
    }

    // Fill the rest from the fractional remainders, normalized to sum to one.
    let mut remainder_sum = T::ZERO;
    for &weight in weights {
        let scaled = count_scalar * weight;
        remainder_sum += scaled - scaled.floor();
    }

    for slot in indices.iter_mut().skip(filled) {
        let draw = T::from_f64(random.next_unit_f64());
        *slot = draw_from_remainders(weights, count_scalar, remainder_sum, draw);
    }
}

/// The smallest source whose cumulative weight reaches `draw`.
fn draw_from_weights<T: Numeric>(weights: &[T], draw: T) -> usize {
    let mut running_sum = T::ZERO;
    for (source, &weight) in weights.iter().enumerate() {
        running_sum += weight;
        if running_sum >= draw {
            return source;
        }
    }
    weights.len() - 1
}

/// The smallest source whose cumulative fractional remainder reaches `draw`.
fn draw_from_remainders<T: Numeric>(
    weights: &[T],
    count_scalar: T,
    remainder_sum: T,
    draw: T,
) -> usize {
    let mut running_sum = T::ZERO;
    for (source, &weight) in weights.iter().enumerate() {
        let scaled = count_scalar * weight;
        running_sum += (scaled - scaled.floor()) / remainder_sum;
        if running_sum >= draw {
            return source;
        }
    }
    weights.len() - 1
}

/// Scores how well a particle's predicted measurement matches the real one, as a log-weight.
///
/// Implement this for a custom sensor; [`GaussianLikelihood`] is the ready-made default.
pub trait Likelihood<const MEASUREMENT_DIMENSION: usize, T: Numeric> {
    /// The log of the (unnormalized) weight for a particle whose model predicts `predicted` when the
    /// sensor read `measurement`. Larger means a better match.
    fn log_weight(
        &self,
        predicted: &[T; MEASUREMENT_DIMENSION],
        measurement: &[T; MEASUREMENT_DIMENSION],
    ) -> T;
}

/// A measurement model with additive Gaussian noise: the log-weight is the negative half of the
/// squared mismatch, measured in units of the noise covariance.
///
/// Plain subtraction forms the mismatch, so a measurement with an angular component needs a custom
/// [`Likelihood`] that folds the angle into a ±π band first.
#[derive(Debug, Clone, Copy)]
pub struct GaussianLikelihood<const MEASUREMENT_DIMENSION: usize, T = f64> {
    noise_factor: Cholesky<MEASUREMENT_DIMENSION, T>,
}

impl<const MEASUREMENT_DIMENSION: usize, T: Numeric> GaussianLikelihood<MEASUREMENT_DIMENSION, T> {
    /// Builds the likelihood from the measurement-noise covariance.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) if the covariance cannot
    /// be factorized.
    pub fn new(
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) -> Result<Self, EstimationError> {
        let noise_factor = measurement_noise
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?;
        Ok(GaussianLikelihood { noise_factor })
    }
}

impl<const MEASUREMENT_DIMENSION: usize, T: Numeric> Likelihood<MEASUREMENT_DIMENSION, T>
    for GaussianLikelihood<MEASUREMENT_DIMENSION, T>
{
    fn log_weight(
        &self,
        predicted: &[T; MEASUREMENT_DIMENSION],
        measurement: &[T; MEASUREMENT_DIMENSION],
    ) -> T {
        // residual = measurement − predicted; weight ∝ −½ · residualᵀ · R⁻¹ · residual. The shared
        // constant (2π)^{M/2}·√det(R) is dropped: it is the same for every particle and cancels in
        // normalization.
        let residual = Vector::from_fn(|i| measurement[i] - predicted[i]);
        let solved = self.noise_factor.solve(residual);
        -T::HALF * residual.dot(solved)
    }
}

/// A particle filter over a `STATE_DIMENSION`-state model with `MEASUREMENT_DIMENSION` measurements.
///
/// Holds a cloud of weighted state samples. [`predict`](Self::predict) pushes each through the
/// process model and adds sampled process noise; [`update`](Self::update) reweights each by a
/// measurement likelihood and normalizes; resampling refreshes the cloud when it degenerates.
///
/// The random backend `R` defaults to the built-in [`Pcg32`]; [`new`](Self::new) seeds one, and
/// [`from_random`](Self::from_random) takes any [`RandomSource`]. `particle_count` must be at least
/// one — a zero cloud carries no weight and is rejected as
/// [`WeightsDegenerate`](EstimationError::WeightsDegenerate).
///
/// # Examples
/// ```
/// use multicalc::estimation::{GaussianLikelihood, ParticleFilter};
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::scalar::{Numeric, VectorFn};
/// # fn main() -> Result<(), multicalc::error::EstimationError> {
/// // A stationary 2-D point, measured directly with a little noise.
/// struct Stationary;
/// impl VectorFn<2, 2> for Stationary {
///     fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
///         [state[0], state[1]]
///     }
/// }
///
/// let mut filter = ParticleFilter::<2, 2>::new(
///     1000,
///     Vector::new([0.0, 0.0]),                  // initial mean
///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),    // initial covariance
///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),  // process noise
///     7,                                        // seed
/// )?;
/// let sensor = GaussianLikelihood::new(Matrix::new([[0.05, 0.0], [0.0, 0.05]]))?;
///
/// for _ in 0..20 {
///     filter.predict(&Stationary)?;
///     filter.update(&Stationary, &sensor, Vector::new([1.0, 2.0]))?;
/// }
/// let mean = *filter.mean().as_array();
/// assert!((mean[0] - 1.0).abs() < 0.2);
/// assert!((mean[1] - 2.0).abs() < 0.2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ParticleFilter<
    const STATE_DIMENSION: usize,
    const MEASUREMENT_DIMENSION: usize,
    T = f64,
    R = Pcg32,
> {
    particles: Vec<Vector<STATE_DIMENSION, T>>,
    weights: Vec<T>,
    process_noise_factor: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    resampling: ResamplingScheme,
    resample_threshold: T,
    roughening: T,
    random: R,
    particle_scratch: Vec<Vector<STATE_DIMENSION, T>>,
    index_scratch: Vec<usize>,
    log_weight_scratch: Vec<T>,
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T: Numeric>
    ParticleFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T, Pcg32>
{
    /// Builds a filter with a [`Pcg32`] seeded from `seed`, sampling the initial cloud from the
    /// Gaussian `initial_mean`/`initial_covariance`.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) if either covariance
    /// cannot be factorized, and [`WeightsDegenerate`](EstimationError::WeightsDegenerate) if
    /// `particle_count` is zero.
    pub fn new(
        particle_count: usize,
        initial_mean: Vector<STATE_DIMENSION, T>,
        initial_covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        seed: u64,
    ) -> Result<Self, EstimationError> {
        Self::from_random(
            particle_count,
            initial_mean,
            initial_covariance,
            process_noise,
            Pcg32::new(seed),
        )
    }
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T: Numeric, R>
    ParticleFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T, R>
where
    R: RandomSource,
{
    /// Builds a filter with an explicit random backend — the built-in [`Pcg32`] or your own
    /// [`RandomSource`]. [`new`](Self::new) is the seeded-`Pcg32` default.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) if either covariance
    /// cannot be factorized, and [`WeightsDegenerate`](EstimationError::WeightsDegenerate) if
    /// `particle_count` is zero.
    pub fn from_random(
        particle_count: usize,
        initial_mean: Vector<STATE_DIMENSION, T>,
        initial_covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        mut random: R,
    ) -> Result<Self, EstimationError> {
        const {
            assert!(
                STATE_DIMENSION > 0,
                "ParticleFilter: STATE_DIMENSION must be non-zero"
            )
        };
        const {
            assert!(
                MEASUREMENT_DIMENSION > 0,
                "ParticleFilter: MEASUREMENT_DIMENSION must be non-zero"
            )
        };

        // A zero cloud carries no weight, so reject it rather than build an empty filter.
        if particle_count == 0 {
            return Err(EstimationError::WeightsDegenerate);
        }

        // Factor both covariances up front. The initial factor shapes the starting spread; the
        // process-noise factor is kept to add noise every predict step. A covariance that will not
        // factorize is not a valid spread.
        let initial_factor = initial_covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .l();

        let process_noise_factor = process_noise
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .l();

        // Scatter the starting cloud around the mean: each particle is the mean plus the initial
        // factor applied to a vector of standard normal draws, which reshapes those draws to match
        // the requested covariance.
        let mut particles = Vec::with_capacity(particle_count);
        for _ in 0..particle_count {
            let sample = Vector::from_fn(|_| T::from_f64(random.standard_normal()));
            particles.push(initial_mean + initial_factor * sample);
        }

        // Start every particle equally likely, and size the reusable scratch buffers once so a
        // steady run never allocates again.
        let weights = vec![T::ONE / T::from_usize(particle_count); particle_count];
        let particle_scratch = particles.clone();
        let index_scratch = vec![0usize; particle_count];
        let log_weight_scratch = vec![T::ZERO; particle_count];

        Ok(ParticleFilter {
            particles,
            weights,
            process_noise_factor,
            // Defaults: resample once half the cloud is effectively dead, with roughening off. The
            // builder methods change any of these.
            resampling: ResamplingScheme::Systematic,
            resample_threshold: T::from_usize(particle_count) * T::HALF,
            roughening: T::ZERO,
            random,
            particle_scratch,
            index_scratch,
            log_weight_scratch,
        })
    }

    /// Chooses how the filter refreshes the cloud when it degenerates. Defaults to
    /// [`Systematic`](ResamplingScheme::Systematic).
    #[must_use]
    pub const fn with_resampling(mut self, scheme: ResamplingScheme) -> Self {
        self.resampling = scheme;
        self
    }

    /// Sets the effective-sample-size below which [`update`](Self::update) resamples. Defaults to
    /// half the particle count.
    #[must_use]
    pub fn with_resample_threshold(mut self, threshold: T) -> Self {
        self.resample_threshold = threshold;
        self
    }

    /// Sets the extra jitter added to each coordinate after a resample, in units of that
    /// coordinate's spread across the cloud. Defaults to zero (off).
    #[must_use]
    pub fn with_roughening(mut self, scale: T) -> Self {
        self.roughening = scale;
        self
    }

    /// Replaces the process noise, refactorizing it.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) if it cannot be
    /// factorized.
    pub fn set_process_noise(
        &mut self,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    ) -> Result<(), EstimationError> {
        self.process_noise_factor = process_noise
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .l();
        Ok(())
    }

    /// Rolls every particle forward through `process_model` and adds a draw of process noise.
    ///
    /// The weights are left unchanged. Returns [`NonFinite`](EstimationError::NonFinite) if any
    /// particle ends up holding an infinity or NaN.
    pub fn predict<ProcessModel>(
        &mut self,
        process_model: &ProcessModel,
    ) -> Result<(), EstimationError>
    where
        ProcessModel: VectorFn<STATE_DIMENSION, STATE_DIMENSION>,
    {
        // Move each particle on its own: run it through the model, then add a random kick shaped by
        // the process-noise factor so the cloud spreads the way the noise says it should. One bad
        // particle does not stop the others, so finish the whole pass and report afterwards.
        let mut finite = true;
        for particle in self.particles.iter_mut() {
            let propagated = Vector::new(process_model.eval(particle.as_array()));
            let noise = Vector::from_fn(|_| T::from_f64(self.random.standard_normal()));
            *particle = propagated + self.process_noise_factor * noise;
            if !particle.is_finite() {
                finite = false;
            }
        }
        if finite {
            Ok(())
        } else {
            Err(EstimationError::NonFinite)
        }
    }

    /// Reweights every particle by how well it explains `measurement`, normalizes, and resamples if
    /// the cloud has degenerated below the threshold.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) if the measurement holds an infinity or NaN,
    /// and [`WeightsDegenerate`](EstimationError::WeightsDegenerate) if every weight underflows to
    /// zero — no particle can explain the measurement.
    pub fn update<MeasurementModel, L>(
        &mut self,
        measurement_model: &MeasurementModel,
        likelihood: &L,
        measurement: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError>
    where
        MeasurementModel: VectorFn<STATE_DIMENSION, MEASUREMENT_DIMENSION>,
        L: Likelihood<MEASUREMENT_DIMENSION, T>,
    {
        if !measurement.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        // Score every particle in log space: ask the model what this particle would have measured,
        // let the likelihood rate how well that matches the real reading, and add it to the
        // particle's current log-weight. Working in logs keeps tiny probabilities from underflowing.
        for i in 0..self.particles.len() {
            let predicted = measurement_model.eval(self.particles[i].as_array());
            let score = likelihood.log_weight(&predicted, measurement.as_array());
            self.log_weight_scratch[i] = self.weights[i].ln() + score;
        }

        // Normalize by subtracting the largest log-weight before exponentiating, so a peaked cloud
        // does not underflow every weight to zero at once. A non-finite largest means every particle
        // is impossible, so the cloud is dead.
        let mut largest = T::NEG_INFINITY;
        for &log_weight in &self.log_weight_scratch {
            if log_weight > largest {
                largest = log_weight;
            }
        }
        if !largest.is_finite() {
            return Err(EstimationError::WeightsDegenerate);
        }

        // Back to plain weights, measured relative to the heaviest particle, and total them.
        let mut sum = T::ZERO;
        for i in 0..self.weights.len() {
            let weight = (self.log_weight_scratch[i] - largest).exp();
            self.weights[i] = weight;
            sum += weight;
        }

        // A zero or non-finite total leaves nothing to divide by, so the cloud has degenerated.
        if sum <= T::ZERO || !sum.is_finite() {
            return Err(EstimationError::WeightsDegenerate);
        }

        // Scale so the weights sum to one.
        for weight in self.weights.iter_mut() {
            *weight /= sum;
        }

        // If the weight has piled onto too few particles, refresh the cloud before it collapses.
        if self.effective_sample_size() < self.resample_threshold {
            self.resample();
        }
        Ok(())
    }

    /// Draws a fresh, equally-weighted cloud from the current weighted one, regardless of the
    /// threshold, then applies roughening if it is on.
    pub fn resample(&mut self) {
        // Ask the scheme which source particle fills each new slot; heavy particles get picked
        // repeatedly, light ones drop out.
        self.resampling
            .resample_indices(&self.weights, &mut self.random, &mut self.index_scratch);
        // Copy the chosen particles into the spare buffer, then swap it in as the live cloud. A
        // source can appear several times, so this needs a separate buffer rather than an in-place
        // shuffle.
        for (destination, &source) in self.index_scratch.iter().enumerate() {
            self.particle_scratch[destination] = self.particles[source];
        }
        core::mem::swap(&mut self.particles, &mut self.particle_scratch);

        // The fresh cloud is equally likely again.
        let uniform = T::ONE / T::from_usize(self.weights.len());
        for weight in self.weights.iter_mut() {
            *weight = uniform;
        }

        // Resampling makes copies, so optionally jitter them apart again.
        if self.roughening > T::ZERO {
            self.roughen();
        }
    }

    /// Adds a little noise to each coordinate after a resample, scaled by how far the samples reach
    /// along that axis, to keep the cloud from collapsing onto a few repeated points.
    fn roughen(&mut self) {
        // Handle each coordinate on its own, since axes can reach across very different distances.
        for axis in 0..STATE_DIMENSION {
            // Measure how far the samples reach along this axis (lowest to highest), a cheap,
            // scale-aware stand-in for the spread.
            let mut lowest = *self.particles[0].get(axis).unwrap_or(&T::ZERO);
            let mut highest = lowest;
            for particle in &self.particles {
                let value = *particle.get(axis).unwrap_or(&T::ZERO);
                if value < lowest {
                    lowest = value;
                }
                if value > highest {
                    highest = value;
                }
            }
            // Size the jitter to that reach; if the samples all share this coordinate there is no
            // reach and nothing to jitter.
            let scale = self.roughening * (highest - lowest);
            if scale <= T::ZERO {
                continue;
            }
            // Nudge every particle along this axis by an independent draw at that scale.
            for particle in self.particles.iter_mut() {
                if let Some(x) = particle.get_mut(axis) {
                    *x += scale * T::from_f64(self.random.standard_normal());
                }
            }
        }
    }

    // ----- Accessors -----

    /// The current cloud of state samples.
    pub fn particles(&self) -> &[Vector<STATE_DIMENSION, T>] {
        &self.particles
    }

    /// The current normalized weights, one per particle.
    #[must_use]
    pub fn weights(&self) -> &[T] {
        &self.weights
    }

    /// The effective sample size, `1 / Σ weightᵢ²`: near the particle count when the weight is spread
    /// evenly, near one when a single particle carries it all.
    #[must_use]
    pub fn effective_sample_size(&self) -> T {
        let mut sum_of_squares = T::ZERO;
        for &weight in &self.weights {
            sum_of_squares += weight * weight;
        }
        T::ONE / sum_of_squares
    }

    /// The weighted mean of the cloud, `Σ weightᵢ · particleᵢ` — the usual state estimate.
    pub fn mean(&self) -> Vector<STATE_DIMENSION, T> {
        let mut accumulated = Vector::zeros();
        for (particle, &weight) in self.particles.iter().zip(&self.weights) {
            accumulated += particle.scale(weight);
        }
        accumulated
    }

    /// The single heaviest particle — the state the cloud considers most likely, useful when the
    /// belief has several peaks and the mean would fall between them.
    pub fn maximum_a_posteriori_state(&self) -> Vector<STATE_DIMENSION, T> {
        let mut best = 0;
        let mut best_weight = self.weights[0];
        for (index, &weight) in self.weights.iter().enumerate() {
            if weight > best_weight {
                best_weight = weight;
                best = index;
            }
        }
        self.particles[best]
    }
}
