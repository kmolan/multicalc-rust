//! Startup global localization: a particle filter matching lidar against a known grid.

use multicalc::error::EstimationError;
use multicalc::estimation::ParticleFilter;
use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::scalar::{Numeric, VectorFn};

use super::kalman_filter_models::diagonal;
use super::lidar::Lidar2d;
use super::occupancy_grid::OccupancyGrid;

/// How the starting guesses are scattered before the robot has seen anything.
///
/// The two spreads are variances, matching what the particle filter takes: `0.16` is a position
/// scattered about `0.4 m`, and `4.0` is a heading scattered about `2 rad`, which is as good as
/// unknown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialParticleCloud {
    /// How many guesses to carry.
    pub particle_count: usize,
    /// How far the guesses spread in position.
    pub position_variance: f64,
    /// How far the guesses spread in heading.
    pub heading_variance: f64,
}

impl Default for InitialParticleCloud {
    /// A cloud wide enough that only the position is really being hinted at: the heading is left
    /// open, since that is what a robot at a standstill cannot work out for itself.
    fn default() -> Self {
        InitialParticleCloud {
            particle_count: 2000,
            position_variance: 0.16,
            heading_variance: 4.0,
        }
    }
}

/// Finds the robot on a known map at startup: a particle filter scoring lidar against the grid.
pub struct GlobalLocalizer<const BEAMS: usize> {
    filter: ParticleFilter<3, 1>,
    /// The lidar's range, so a hypothesis casts the same distance the real sensor does.
    maximum_range: f64,
    /// How much a range reading is expected to differ from the map, for scoring.
    beam_deviation: f64,
}

impl<const BEAMS: usize> GlobalLocalizer<BEAMS> {
    /// A localizer seeded from a rough guess, with the guesses scattered as `cloud` says.
    pub fn new(
        hint: [f64; 3],
        cloud: InitialParticleCloud,
        maximum_range: f64,
        beam_deviation: f64,
        seed: u64,
    ) -> Result<Self, EstimationError> {
        let filter = ParticleFilter::<3, 1>::new(
            cloud.particle_count,
            Vector::new(hint),
            diagonal([
                cloud.position_variance,
                cloud.position_variance,
                cloud.heading_variance,
            ]),
            diagonal([1e-4, 1e-4, 1e-4]), // motion noise added each predict
            seed,
        )?;
        Ok(GlobalLocalizer {
            filter,
            maximum_range,
            beam_deviation,
        })
    }

    /// Rolls every hypothesis forward by one step of the slow startup turn, plus the filter's noise.
    pub fn predict(
        &mut self,
        delta_arc_length: f64,
        delta_heading: f64,
    ) -> Result<(), EstimationError> {
        self.filter.predict(&LocalizationMotion {
            delta_arc_length,
            delta_heading,
        })
    }

    /// Scores each hypothesis by casting the beams against the grid, then normalizes and resamples.
    pub fn update(
        &mut self,
        scan: &[f64; BEAMS],
        grid: &OccupancyGrid,
        lidar: &Lidar2d<BEAMS>,
    ) -> Result<(), EstimationError> {
        let maximum_range = self.maximum_range;
        let beam_deviation = self.beam_deviation;
        self.filter.update_with_log_weights(|particle| {
            let (x, y, heading) = (particle[0], particle[1], particle[2]);
            let mut log_weight = 0.0;
            for (index, &measured) in scan.iter().enumerate() {
                let Some(offset) = lidar.beam_angle(index) else {
                    continue;
                };
                let predicted = grid.cast_ray([x, y], heading + offset, maximum_range);
                log_weight += beam_log_likelihood(predicted, measured, beam_deviation);
            }
            log_weight
        })
    }

    /// The best current pose and the spread of the position and heading around it.
    pub fn estimate(&self) -> (Vector3D, Matrix3D) {
        let particles = self.filter.particles();
        let weights = self.filter.weights();
        let (mut mean_x, mut mean_y, mut sin_sum, mut cos_sum) = (0.0, 0.0, 0.0, 0.0);
        for (p, &w) in particles.iter().zip(weights) {
            mean_x += w * p[0];
            mean_y += w * p[1];
            sin_sum += w * p[2].sin();
            cos_sum += w * p[2].cos();
        }
        let heading = sin_sum.atan2(cos_sum);
        // How tightly the headings agree, turned into a rough heading spread.
        let resultant = sin_sum.hypot(cos_sum).min(1.0);
        let heading_variance = ((1.0 - resultant) * 2.0).max(1e-6);
        let (mut cxx, mut cyy, mut cxy) = (0.0, 0.0, 0.0);
        for (p, &w) in particles.iter().zip(weights) {
            let (dx, dy) = (p[0] - mean_x, p[1] - mean_y);
            cxx += w * dx * dx;
            cyy += w * dy * dy;
            cxy += w * dx * dy;
        }
        let covariance = Matrix::from_fn(|row, column| match (row, column) {
            (0, 0) => cxx,
            (1, 1) => cyy,
            (0, 1) | (1, 0) => cxy,
            (2, 2) => heading_variance,
            _ => 0.0,
        });
        (Vector::new([mean_x, mean_y, heading]), covariance)
    }

    /// True once the guesses have settled tightly enough, in both position and heading, to trust the
    /// fix. The position limit covers both axes together.
    #[must_use]
    pub fn is_converged(&self, position_spread_limit: f64, heading_spread_limit: f64) -> bool {
        let (_, covariance) = self.estimate();
        covariance[(0, 0)] + covariance[(1, 1)] < position_spread_limit
            && covariance[(2, 2)] < heading_spread_limit
    }

    #[must_use]
    pub fn effective_sample_size(&self) -> f64 {
        self.filter.effective_sample_size()
    }

    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.filter.particles().len()
    }

    pub fn particles(&self) -> &[Vector3D] {
        self.filter.particles()
    }
}

/// Moves a hypothesis `[x, y, heading]` by one step of the startup turn.
struct LocalizationMotion {
    delta_arc_length: f64,
    delta_heading: f64,
}
impl VectorFn<3, 3> for LocalizationMotion {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 3] {
        let heading = state[2];
        let next = heading + S::from_f64(self.delta_heading);
        let step = S::from_f64(self.delta_arc_length);
        let wrapped = next.wrap_to_pi();
        [
            state[0] + step * heading.cos(),
            state[1] + step * heading.sin(),
            wrapped,
        ]
    }
}

/// Scores one beam: how well a hypothesis's cast range matches the real reading.
#[must_use]
fn beam_log_likelihood(predicted: Option<f64>, measured: f64, deviation: f64) -> f64 {
    match (predicted, measured.is_finite()) {
        (Some(range), true) => {
            let error = (range - measured) / deviation;
            -0.5 * error * error
        }
        (None, false) => 0.1, // both see nothing: a small agreement reward
        _ => -5.0,            // one sees a wall where the other sees space: a penalty
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    // Scoring a single beam is private to this module, so its test stays here; everything that goes
    // through the public localizer is tested in the suite beside `src`.
    #[test]
    fn beam_log_likelihood_rewards_matches() {
        let exact = beam_log_likelihood(Some(2.0), 2.0, 0.1);
        let off = beam_log_likelihood(Some(2.0), 3.0, 0.1);
        assert_eq!(exact, 0.0);
        assert!(
            exact > off,
            "an exact match {exact} should beat a metre-off one {off}"
        );
        // Both seeing nothing is a small reward, above the presence-mismatch penalty.
        let agree = beam_log_likelihood(None, f64::INFINITY, 0.1);
        let mismatch = beam_log_likelihood(Some(1.0), f64::INFINITY, 0.1);
        assert!(
            agree > 0.0 && agree > mismatch,
            "agree {agree}, mismatch {mismatch}"
        );
    }
}
