//! A forward-arc 2D lidar over an occupancy map, with Gaussian range noise and beam dropout.

use multicalc::error::MappingError;
use multicalc::mapping::{OccupancyMap, ScanGeometry};
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

/// A forward-arc lidar with `NUM_BEAMS` beams uniformly spaced across its field of view.
///
/// The beam directions come from a [`ScanGeometry`], the same type
/// `multicalc::control::FollowTheGap` numbers its beams with, so a scan and the steering worked out
/// from it always agree. A beam that hits nothing — out of range or dropped — reads as
/// `f64::INFINITY`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lidar2d<const NUM_BEAMS: usize> {
    geometry: ScanGeometry<NUM_BEAMS, f64>,
    range_standard_deviation: f64,
    dropout_probability: f64,
}

impl<const NUM_BEAMS: usize> Lidar2d<NUM_BEAMS> {
    /// A lidar with the given arc, range, range noise, and per-beam dropout probability.
    ///
    /// Returns whatever [`ScanGeometry::try_new`] rejects the arc or range with.
    pub fn new(
        field_of_view: f64,
        maximum_range: f64,
        range_standard_deviation: f64,
        dropout_probability: f64,
    ) -> Result<Self, MappingError> {
        debug_assert!(
            range_standard_deviation >= 0.0,
            "range standard deviation cannot be negative"
        );
        debug_assert!(
            (0.0..=1.0).contains(&dropout_probability),
            "dropout probability must be within [0, 1]"
        );
        Ok(Lidar2d {
            geometry: ScanGeometry::try_new(field_of_view, maximum_range)?,
            range_standard_deviation,
            dropout_probability,
        })
    }

    /// Where this lidar's beams point, for anything that has to line up with them.
    #[must_use]
    pub fn geometry(&self) -> ScanGeometry<NUM_BEAMS, f64> {
        self.geometry
    }

    /// The direction beam `index` points, measured from straight ahead, or `None` if the index is
    /// out of range.
    #[must_use]
    pub fn beam_angle(&self, index: usize) -> Option<f64> {
        self.geometry.beam_angle(index)
    }

    /// One scan from `pose = [x, y, heading]` against `map`.
    ///
    /// The dropout draw runs for every beam whether or not the ray hits anything, so the generator
    /// advances by the same amount per beam and a scan stays reproducible when the map changes.
    #[must_use]
    pub fn simulate<M: OccupancyMap>(
        &self,
        map: &M,
        pose: [f64; 3],
        rng: &mut Pcg32,
    ) -> [f64; NUM_BEAMS] {
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
            let Some(angle) = self.geometry.beam_angle(index) else {
                return f64::INFINITY;
            };
            // The beam's direction in world coordinates: robot heading plus the beam's own offset.
            let world_angle = pose[2] + angle;
            match map.cast_ray(
                [pose[0], pose[1]],
                world_angle,
                self.geometry.maximum_range(),
            ) {
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
