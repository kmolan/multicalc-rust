//! A forward-arc 2D lidar over an [`OccupancyGrid`], with Gaussian range noise and beam dropout.

use rand::RngExt;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

use super::occupancy_grid::OccupancyGrid;

/// A forward-arc lidar with `BEAMS` beams uniformly spaced across its field of view.
///
/// Beam angles are measured from the robot's forward axis, positive to the left, matching what
/// `multicalc::control::FollowTheGap` expects. A beam that hits nothing — out of range or
/// dropped — reads as `f64::INFINITY`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lidar2d<const BEAMS: usize> {
    field_of_view: f64,
    maximum_range: f64,
    range_standard_deviation: f64,
    dropout_probability: f64,
}

impl<const BEAMS: usize> Lidar2d<BEAMS> {
    /// A lidar with the given arc, range, range noise, and per-beam dropout probability.
    #[must_use]
    pub fn new(
        field_of_view: f64,
        maximum_range: f64,
        range_standard_deviation: f64,
        dropout_probability: f64,
    ) -> Self {
        debug_assert!(BEAMS >= 2, "a lidar needs at least two beams");
        debug_assert!(
            field_of_view.is_finite() && field_of_view >= 0.0,
            "field of view must be finite and non-negative"
        );
        debug_assert!(
            maximum_range.is_finite() && maximum_range > 0.0,
            "maximum range must be finite and positive"
        );
        debug_assert!(
            range_standard_deviation >= 0.0,
            "range standard deviation cannot be negative"
        );
        debug_assert!(
            (0.0..=1.0).contains(&dropout_probability),
            "dropout probability must be within [0, 1]"
        );
        Lidar2d {
            field_of_view,
            maximum_range,
            range_standard_deviation,
            dropout_probability,
        }
    }

    /// The direction beam `index` points, measured from straight ahead, or `None` if the index is
    /// out of range.
    ///
    /// Uses the same formula as the gap-follower, so the two agree beam for beam.
    #[must_use]
    pub fn beam_angle(&self, index: usize) -> Option<f64> {
        if BEAMS < 2 || index >= BEAMS {
            return None;
        }
        let span = (BEAMS - 1) as f64;
        Some(-self.field_of_view / 2.0 + self.field_of_view * index as f64 / span)
    }

    /// One scan from `pose = [x, y, heading]` against `grid`.
    ///
    /// The dropout draw runs for every beam whether or not the ray hits anything, so the generator
    /// advances by the same amount per beam and a scan stays reproducible when the grid changes.
    #[must_use]
    pub fn simulate(&self, grid: &OccupancyGrid, pose: [f64; 3], rng: &mut Pcg32) -> [f64; BEAMS] {
        // With zero noise the lidar returns exact geometry, which the tests rely on; `Normal::new`
        // rejects a zero deviation, so the noise draw is skipped in that case.
        let noise = (self.range_standard_deviation > 0.0)
            .then(|| Normal::new(0.0, self.range_standard_deviation).ok())
            .flatten();

        // Build one range reading per beam.
        core::array::from_fn(|index| {
            // This beam was randomly dropped; report no return.
            if rng.random::<f64>() < self.dropout_probability {
                return f64::INFINITY;
            }
            // Out-of-range index; report no return.
            let Some(angle) = self.beam_angle(index) else {
                return f64::INFINITY;
            };
            // The beam's direction in world coordinates: robot heading plus the beam's own offset.
            let world_angle = pose[2] + angle;
            match grid.cast_ray([pose[0], pose[1]], world_angle, self.maximum_range) {
                // Hit something: add the range noise, but never report a negative distance.
                Some(distance) => match noise {
                    Some(normal) => (distance + normal.sample(rng)).max(0.0),
                    None => distance,
                },
                // Nothing within range; report no return.
                None => f64::INFINITY,
            }
        })
    }
}
