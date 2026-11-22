//! Monte Carlo Localization using particle filter

use crate::error::EstimationError;
use crate::estimation::ParticleFilter;
use crate::estimation::likelihood_field::LikelihoodFieldModel;
use crate::linear_algebra::{Matrix, Vector};
use crate::mapping::{DistanceField, OccupancyMap, ScanGeometry};
use crate::random::RandomScalar;
use crate::scalar::{Numeric, Primal, VectorFn};

/// How the guesses are scattered before the robot has seen anything.
///
/// Both spreads are variances, which is what the particle filter takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialParticleCloud<T: Numeric = f64> {
    /// How many guesses to carry.
    pub particle_count: usize,
    /// How far the guesses spread in position.
    pub position_variance: T,
    /// How far the guesses spread in heading.
    pub heading_variance: T,
}

impl<T: Numeric> Default for InitialParticleCloud<T> {
    /// A cloud wide enough that only the position is really being hinted at: `0.16` spreads the
    /// position about `0.4 m`, and `4.0` spreads the heading about `2 rad`, which is as good as
    /// unknown — and a robot standing still cannot work its heading out for itself.
    fn default() -> Self {
        InitialParticleCloud {
            particle_count: 2000,
            position_variance: T::from_f64(0.16),
            heading_variance: T::from_f64(4.0),
        }
    }
}

/// What one beam agreeing or disagreeing with the map counts for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamModel<T: Numeric = f64> {
    /// How far a reading may sit from the map before it counts against a guess.
    pub range_deviation: T,
    /// What a beam is worth when the reading and the map both see nothing.
    pub agreement_reward: T,
    /// What a beam costs when one of the two sees a wall and the other open space.
    pub mismatch_penalty: T,
}

impl<T: Numeric> Default for BeamModel<T> {
    fn default() -> Self {
        BeamModel {
            range_deviation: T::from_f64(0.10),
            agreement_reward: T::from_f64(0.1),
            mismatch_penalty: T::from_f64(-5.0),
        }
    }
}

/// Finds a robot on a map it already has: every guess casts the beams it would see, and the
/// guesses matching the real scan carry more weight.
///
/// Each guess is a pose `[x, y, heading]`. Feed it travel and turn with [`predict`](Self::predict),
/// feed it scans with [`update`](Self::update), and read the answer with
/// [`estimate`](Self::estimate).
///
/// ```
/// use multicalc::estimation::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};
/// use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap, ScanGeometry};
///
/// // A 6 m by 6 m room of 20 cm cells, walled all the way round.
/// let cell_size = 0.2_f64;
/// let mut room = DynamicOccupancyGrid::try_new(30, 30, cell_size, [0.0, 0.0])?;
/// let walls = [[0.4, 0.4], [5.6, 0.4], [5.6, 5.6], [0.4, 5.6]];
/// room.occupy_polyline(&walls, true);
///
/// // An eight-beam scan across a half turn.
/// const NUM_BEAMS: usize = 8;
/// let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(core::f64::consts::PI, 8.0)?;
///
/// // The robot is really here; the localizer is only told roughly where to look.
/// let truth = [2.0, 3.0, 0.4];
/// let hint = [2.3, 2.7, 0.4];
/// let cloud = InitialParticleCloud { particle_count: 300, ..Default::default() };
/// let beam_model = BeamModel { range_deviation: 0.3, ..Default::default() };
/// let seed = 20260802;
/// let mut localizer = MonteCarloLocalizer::<NUM_BEAMS>::new(hint, cloud, beam_model, seed)?;
///
/// // Standing still, taking the same reading a few times over.
/// let reading: [f64; NUM_BEAMS] = core::array::from_fn(|beam| {
///     let offset = scan.beam_angle(beam).unwrap_or(0.0);
///     room.cast_ray([truth[0], truth[1]], truth[2] + offset, scan.maximum_range())
///         .unwrap_or(f64::INFINITY)
/// });
/// for _ in 0..6 {
///     localizer.update(&reading, &room, &scan)?;
/// }
///
/// // The cloud has settled onto the robot.
/// let (pose, _spread) = localizer.estimate();
/// assert!((pose[0] - truth[0]).abs() < 0.5);
/// assert!((pose[1] - truth[1]).abs() < 0.5);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MonteCarloLocalizer<const NUM_BEAMS: usize, T: RandomScalar + Primal = f64> {
    filter: ParticleFilter<3, 1, T>,
    beam_model: BeamModel<T>,
}

impl<const NUM_BEAMS: usize, T: RandomScalar + Primal> MonteCarloLocalizer<NUM_BEAMS, T> {
    /// A localizer seeded from a rough guess, with the guesses scattered as `cloud` says.
    ///
    /// The spread added on every [`predict`](Self::predict) starts at `1e-4` on each of `x`, `y`,
    /// and heading; [`set_motion_noise`](Self::set_motion_noise) changes it.
    ///
    /// Returns [`EstimationError::NotPositiveDefinite`] if the spreads do not describe a usable
    /// cloud, and [`EstimationError::WeightsDegenerate`] if the particle count is zero.
    pub fn new(
        hint: [T; 3],
        cloud: InitialParticleCloud<T>,
        beam_model: BeamModel<T>,
        seed: u64,
    ) -> Result<Self, EstimationError> {
        let spread = Matrix::from_diagonal([
            cloud.position_variance,
            cloud.position_variance,
            cloud.heading_variance,
        ]);
        let filter = ParticleFilter::<3, 1, T>::new(
            cloud.particle_count,
            Vector::new(hint),
            spread,
            Matrix::from_diagonal([T::from_f64(1e-4); 3]),
            seed,
        )?;
        Ok(MonteCarloLocalizer { filter, beam_model })
    }

    /// Sets how far the guesses spread on every predict, as a variance on `x`, `y`, and heading.
    ///
    /// Returns [`EstimationError::NotPositiveDefinite`] if the spread cannot be factorized.
    pub fn set_motion_noise(&mut self, variances: [T; 3]) -> Result<(), EstimationError> {
        self.filter
            .set_process_noise(Matrix::from_diagonal(variances))
    }

    /// Moves every guess by one step of travel and turn, plus the filter's own spread.
    ///
    /// Returns [`EstimationError::NonFinite`] if a guess ends up holding an infinity or NaN.
    pub fn predict(
        &mut self,
        delta_arc_length: T,
        delta_heading: T,
    ) -> Result<(), EstimationError> {
        self.filter.predict(&LocalizationMotion {
            delta_arc_length: delta_arc_length.to_f64(),
            delta_heading: delta_heading.to_f64(),
        })
    }

    /// Scores every guess by casting its beams against `map`, then reweights and resamples.
    ///
    /// A reading the scan would not believe — no return, or one outside the range it can see — is
    /// scored as seeing nothing rather than as a distance.
    ///
    /// Returns [`EstimationError::WeightsDegenerate`] if no guess can explain the scan at all.
    pub fn update<M: OccupancyMap<T>>(
        &mut self,
        scan: &[T; NUM_BEAMS],
        map: &M,
        geometry: &ScanGeometry<NUM_BEAMS, T>,
    ) -> Result<(), EstimationError> {
        let beam_model = self.beam_model;
        let maximum_range = geometry.maximum_range();
        self.filter.update_with_log_weights(|guess| {
            let mut score = T::ZERO;
            for (index, &measured) in scan.iter().enumerate() {
                let Some(offset) = geometry.beam_angle(index) else {
                    continue;
                };
                let from_the_map =
                    map.cast_ray([guess[0], guess[1]], guess[2] + offset, maximum_range);
                let believed = geometry.range_is_valid(measured);
                score += beam_score(from_the_map, measured, believed, beam_model);
            }
            score
        })
    }

    /// Reweights the cloud against a distance field instead of casting a ray per beam per particle.
    ///
    /// One interpolated lookup replaces one DDA walk per beam per particle, and the score is
    /// smoother in the pose than [`update`](Self::update)'s, whose likelihood is jagged because it
    /// depends on map resolution. An endpoint falling outside the field contributes the model's
    /// pure-noise term alone.
    ///
    /// The field ignores occlusion, so a pose can score highly by seeing through a wall; keep
    /// [`update`](Self::update) where that matters.
    ///
    /// Returns [`EstimationError::InvalidTuning`] for a non-finite or non-positive
    /// `measurement_deviation`, or a `random_measurement_weight` outside zero to one, and
    /// [`EstimationError::WeightsDegenerate`] if no guess can explain the scan at all.
    pub fn update_against_field<const NUM_ROWS: usize, const NUM_COLUMNS: usize>(
        &mut self,
        field: &DistanceField<NUM_ROWS, NUM_COLUMNS, T>,
        scan: &ScanGeometry<NUM_BEAMS, T>,
        ranges: &[T; NUM_BEAMS],
        model: LikelihoodFieldModel<T>,
    ) -> Result<(), EstimationError> {
        let deviation = model.measurement_deviation;
        let random_weight = model.random_measurement_weight;
        if !deviation.is_finite() || deviation <= T::ZERO {
            return Err(EstimationError::InvalidTuning);
        }
        if !random_weight.is_finite() || random_weight < T::ZERO || random_weight > T::ONE {
            return Err(EstimationError::InvalidTuning);
        }

        let maximum_range = scan.maximum_range();
        let noise_floor = random_weight / maximum_range;
        let twice_variance = T::TWO * deviation * deviation;

        self.filter.update_with_log_weights(|guess| {
            let mut score = T::ZERO;
            for (beam, &measured) in ranges.iter().enumerate() {
                let Some(offset) = scan.beam_angle(beam) else {
                    continue;
                };
                if !scan.range_is_valid(measured) {
                    continue;
                }
                let bearing = guess[2] + offset;
                let endpoint = [
                    guess[0] + measured * bearing.cos(),
                    guess[1] + measured * bearing.sin(),
                ];
                // An endpoint off the field is infinitely far from any obstacle, which leaves the
                // noise term alone.
                let distance = field.distance_at(endpoint).unwrap_or(T::INFINITY);
                let hit = (-(distance * distance) / twice_variance).exp();
                score += ((T::ONE - random_weight) * hit + noise_floor).log();
            }
            score
        })
    }

    /// The best pose the cloud can offer, and how tightly it is holding to it.
    ///
    /// The spread comes back as a covariance over `[x, y, heading]`, with the position part filled
    /// in and the heading part on its own diagonal entry.
    ///
    /// ```
    /// use multicalc::estimation::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};
    ///
    /// // A cloud scattered tightly around a rough guess.
    /// let hint = [2.0, 3.0, 0.5];
    /// let cloud = InitialParticleCloud {
    ///     particle_count: 500,
    ///     position_variance: 0.01,
    ///     heading_variance: 0.01,
    /// };
    /// let seed = 20260804;
    /// let localizer = MonteCarloLocalizer::<8>::new(hint, cloud, BeamModel::default(), seed)?;
    ///
    /// // Before it has seen anything, the answer is the guess it was handed.
    /// let (pose, spread) = localizer.estimate();
    /// assert!((pose[0] - hint[0]).abs() < 0.1);
    /// assert!((pose[2] - hint[2]).abs() < 0.1);
    ///
    /// // And it is holding that answer as loosely as it was told to.
    /// assert!(spread[(0, 0)] < 0.05);
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    pub fn estimate(&self) -> (Vector<3, T>, Matrix<3, 3, T>) {
        let particles = self.filter.particles();
        let weights = self.filter.weights();

        // The position is a plain weighted mean, but a heading cannot be averaged that way: half a
        // turn either side of the mark would average to straight ahead. Averaging the directions
        // themselves and taking the angle back off avoids that.
        let mut mean_x = T::ZERO;
        let mut mean_y = T::ZERO;
        let mut sine_sum = T::ZERO;
        let mut cosine_sum = T::ZERO;
        for (guess, &weight) in particles.iter().zip(weights) {
            mean_x += weight * guess[0];
            mean_y += weight * guess[1];
            sine_sum += weight * guess[2].sin();
            cosine_sum += weight * guess[2].cos();
        }
        let heading = sine_sum.atan2(cosine_sum);

        // How tightly the headings agree: a long resultant means they point the same way.
        let resultant = sine_sum.hypot(cosine_sum).min(T::ONE);
        let heading_spread = ((T::ONE - resultant) * T::TWO).max(T::from_f64(1e-6));

        let mut spread_xx = T::ZERO;
        let mut spread_yy = T::ZERO;
        let mut spread_xy = T::ZERO;
        for (guess, &weight) in particles.iter().zip(weights) {
            let offset_x = guess[0] - mean_x;
            let offset_y = guess[1] - mean_y;
            spread_xx += weight * offset_x * offset_x;
            spread_yy += weight * offset_y * offset_y;
            spread_xy += weight * offset_x * offset_y;
        }
        let spread = Matrix::from_fn(|row, column| match (row, column) {
            (0, 0) => spread_xx,
            (1, 1) => spread_yy,
            (0, 1) | (1, 0) => spread_xy,
            (2, 2) => heading_spread,
            _ => T::ZERO,
        });
        (Vector::new([mean_x, mean_y, heading]), spread)
    }

    /// Whether the guesses have settled tightly enough to trust the answer. The position limit
    /// covers both axes together.
    #[must_use]
    pub fn is_converged(&self, position_spread_limit: T, heading_spread_limit: T) -> bool {
        let (_, spread) = self.estimate();
        spread[(0, 0)] + spread[(1, 1)] < position_spread_limit
            && spread[(2, 2)] < heading_spread_limit
    }

    /// How many guesses are still pulling their weight: near the particle count when the weight is
    /// spread evenly, near one when a single guess carries it all.
    #[must_use]
    pub fn effective_sample_size(&self) -> T {
        self.filter.effective_sample_size()
    }

    /// How many guesses the cloud carries.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.filter.particles().len()
    }

    /// The guesses themselves, for drawing the cloud.
    pub fn particles(&self) -> &[Vector<3, T>] {
        self.filter.particles()
    }
}

/// Moves a guess `[x, y, heading]` by one step of travel and turn.
struct LocalizationMotion {
    delta_arc_length: f64,
    delta_heading: f64,
}

impl VectorFn<3, 3> for LocalizationMotion {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 3] {
        let heading = state[2];
        let step = S::from_f64(self.delta_arc_length);
        [
            state[0] + step * heading.cos(),
            state[1] + step * heading.sin(),
            (heading + S::from_f64(self.delta_heading)).wrap_to_pi(),
        ]
    }
}

/// Scores one beam: how well the distance a guess would see matches the real reading.
fn beam_score<T: Numeric>(
    from_the_map: Option<T>,
    measured: T,
    believed: bool,
    model: BeamModel<T>,
) -> T {
    match (from_the_map, believed) {
        (Some(range), true) => {
            let error = (range - measured) / model.range_deviation;
            T::from_f64(-0.5) * error * error
        }
        // Neither the guess nor the sensor found anything, which is agreement of a sort.
        (None, false) => model.agreement_reward,
        // One sees a wall where the other sees open space.
        _ => model.mismatch_penalty,
    }
}
