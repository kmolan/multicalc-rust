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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    // A grid with one occupied column whose left face sits exactly at x = 2, tall enough that the
    // oblique beams meet it away from any row boundary. This keeps the closed-form scan exact.
    fn wall() -> OccupancyGrid {
        let mut grid = OccupancyGrid::new(15, 12, 1.0, [-5.0, -5.25]);
        for row in 0..grid.rows() {
            grid.set_cell(7, row, true);
        }
        grid
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
        let empty = OccupancyGrid::new(15, 12, 1.0, [-5.0, -5.25]);
        let scan = lidar.simulate(&empty, [0.0, 0.0, 0.0], &mut rng);
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
