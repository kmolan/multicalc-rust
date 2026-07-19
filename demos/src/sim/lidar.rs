//! A forward-arc 2D lidar over a [`Map`], with Gaussian range noise and beam dropout.

use rand::RngExt;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

use super::map::Map;

/// A forward-arc lidar with `BEAMS` beams uniformly spaced across its field of view.
///
/// Beam bearings are measured from the robot's forward axis, positive counter-clockwise, matching
/// the convention `multicalc::control::FollowTheGap` expects. A beam with no return — out of range,
/// or dropped — reads as `f64::INFINITY`.
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
        Lidar2d {
            field_of_view,
            maximum_range,
            range_standard_deviation,
            dropout_probability,
        }
    }

    /// The body-frame bearing of beam `index`, or `None` if the index is out of range.
    ///
    /// Uses the same formula as the gap-follower, so the two agree beam for beam.
    #[must_use]
    pub fn beam_bearing(&self, index: usize) -> Option<f64> {
        if BEAMS < 2 || index >= BEAMS {
            return None;
        }
        let span = (BEAMS - 1) as f64;
        Some(-self.field_of_view / 2.0 + self.field_of_view * index as f64 / span)
    }

    /// One scan from `pose = [x, y, heading]` against `map`.
    ///
    /// The dropout draw happens for every beam whether or not the cast hits, so the generator
    /// advances by a fixed amount per beam and a scan stays reproducible when the map changes.
    pub fn simulate(&self, map: &Map, pose: [f64; 3], rng: &mut Pcg32) -> [f64; BEAMS] {
        // A zero-σ lidar is the exact-geometry mode the tests use; `Normal::new` rejects it, so
        // the noise draw is skipped entirely in that case.
        let noise = (self.range_standard_deviation > 0.0)
            .then(|| Normal::new(0.0, self.range_standard_deviation).ok())
            .flatten();

        core::array::from_fn(|index| {
            if rng.random::<f64>() < self.dropout_probability {
                return f64::INFINITY;
            }
            let Some(bearing) = self.beam_bearing(index) else {
                return f64::INFINITY;
            };
            let world_bearing = pose[2] + bearing;
            match map.cast_ray([pose[0], pose[1]], world_bearing, self.maximum_range) {
                Some(distance) => match noise {
                    Some(normal) => (distance + normal.sample(rng)).max(0.0),
                    None => distance,
                },
                None => f64::INFINITY,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    fn wall() -> Map {
        Map::new().with_segment([2.0, -5.0], [2.0, 5.0])
    }

    #[test]
    fn noiseless_scan_matches_closed_form() {
        // Three beams at -π/4, 0, +π/4 against a wall at x = 2: the oblique beams travel
        // 2 / cos(π/4) to reach it.
        let lidar = Lidar2d::<3>::new(PI / 2.0, 10.0, 0.0, 0.0);
        let mut rng = Pcg32::seed_from_u64(1);
        let scan = lidar.simulate(&wall(), [0.0, 0.0, 0.0], &mut rng);
        let oblique = 2.0 / (PI / 4.0).cos();
        assert!((scan[0] - oblique).abs() < 1e-9, "{}", scan[0]);
        assert!((scan[1] - 2.0).abs() < 1e-9, "{}", scan[1]);
        assert!((scan[2] - oblique).abs() < 1e-9, "{}", scan[2]);
    }

    #[test]
    fn empty_map_reads_as_no_return() {
        let lidar = Lidar2d::<5>::new(PI / 2.0, 10.0, 0.0, 0.0);
        let mut rng = Pcg32::seed_from_u64(1);
        let scan = lidar.simulate(&Map::new(), [0.0, 0.0, 0.0], &mut rng);
        assert!(scan.iter().all(|range| range.is_infinite()));
    }

    #[test]
    fn certain_dropout_drops_every_beam() {
        let lidar = Lidar2d::<5>::new(PI / 2.0, 10.0, 0.0, 1.0);
        let mut rng = Pcg32::seed_from_u64(1);
        let scan = lidar.simulate(&wall(), [0.0, 0.0, 0.0], &mut rng);
        assert!(scan.iter().all(|range| range.is_infinite()));
    }

    #[test]
    fn a_fixed_seed_reproduces_the_scan() {
        let lidar = Lidar2d::<9>::new(PI / 2.0, 10.0, 0.03, 0.01);
        let map = wall();
        let mut first_rng = Pcg32::seed_from_u64(7);
        let mut second_rng = Pcg32::seed_from_u64(7);
        let first = lidar.simulate(&map, [0.0, 0.0, 0.0], &mut first_rng);
        let second = lidar.simulate(&map, [0.0, 0.0, 0.0], &mut second_rng);
        assert_eq!(first, second);
    }

    #[test]
    fn range_noise_has_about_the_right_spread() {
        // One beam straight at the wall, 10 000 scans from a single seeded generator. Fixed seed,
        // so this is a regression check rather than a flaky statistical test.
        let lidar = Lidar2d::<2>::new(0.0, 10.0, 0.03, 0.0);
        let map = wall();
        let mut rng = Pcg32::seed_from_u64(11);
        let samples = 10_000;
        let mut readings = Vec::with_capacity(samples);
        for _ in 0..samples {
            let scan = lidar.simulate(&map, [0.0, 0.0, 0.0], &mut rng);
            readings.push(scan[0]);
        }
        let mean = readings.iter().sum::<f64>() / samples as f64;
        let variance = readings
            .iter()
            .map(|r| (r - mean) * (r - mean))
            .sum::<f64>()
            / samples as f64;
        let deviation = variance.sqrt();
        assert!((mean - 2.0).abs() < 0.003, "mean {mean}");
        assert!((deviation - 0.03).abs() < 0.003, "deviation {deviation}");
    }
}
