//! Range and bearing to the nearest known beacon — the measurement an EKF update consumes.

use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

use super::wrap_angle;

/// A landmark at a known position, with a known identity (no data association).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beacon {
    pub identifier: usize,
    pub position: [f64; 2],
}

/// A noisy range and bearing to one beacon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeaconMeasurement {
    pub identifier: usize,
    /// Distance to the beacon, in metres.
    pub range: f64,
    /// Bearing to the beacon relative to the robot heading, wrapped to `(-π, π]`.
    pub bearing: f64,
}

/// The nearest beacon within `maximum_range` of `pose = [x, y, heading]`, as a noisy range and
/// bearing, or `None` if no beacon is in range.
///
/// Selection uses the *true* distance, so which beacon is reported does not depend on the noise
/// draw. The bearing is wrapped to `(-π, π]` after the noise is added.
pub fn nearest_beacon_measurement(
    beacons: &[Beacon],
    pose: [f64; 3],
    maximum_range: f64,
    range_standard_deviation: f64,
    bearing_standard_deviation: f64,
    rng: &mut Pcg32,
) -> Option<BeaconMeasurement> {
    let mut nearest: Option<(&Beacon, f64, f64)> = None;
    for beacon in beacons {
        let dx = beacon.position[0] - pose[0];
        let dy = beacon.position[1] - pose[1];
        let distance = dx.hypot(dy);
        if nearest.is_none_or(|(_, current, _)| distance < current) {
            nearest = Some((beacon, distance, dy.atan2(dx)));
        }
    }

    let (beacon, distance, world_bearing) = nearest?;
    if distance > maximum_range {
        return None;
    }

    // A zero σ is the exact-geometry mode the tests use; `Normal::new` rejects it, so the draw is
    // skipped entirely in that case.
    let range_noise = sample_noise(range_standard_deviation, rng);
    let bearing_noise = sample_noise(bearing_standard_deviation, rng);

    Some(BeaconMeasurement {
        identifier: beacon.identifier,
        range: (distance + range_noise).max(0.0),
        bearing: wrap_angle(wrap_angle(world_bearing - pose[2]) + bearing_noise),
    })
}

/// One zero-mean Gaussian draw, or exactly zero when the deviation is not positive.
fn sample_noise(standard_deviation: f64, rng: &mut Pcg32) -> f64 {
    if standard_deviation <= 0.0 {
        return 0.0;
    }
    match Normal::new(0.0, standard_deviation) {
        Ok(normal) => normal.sample(rng),
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    fn rng() -> Pcg32 {
        Pcg32::seed_from_u64(3)
    }

    #[test]
    fn nearest_beacon_is_selected() {
        let beacons = [
            Beacon {
                identifier: 0,
                position: [1.0, 0.0],
            },
            Beacon {
                identifier: 1,
                position: [5.0, 0.0],
            },
        ];
        let measurement =
            nearest_beacon_measurement(&beacons, [0.0, 0.0, 0.0], 10.0, 0.0, 0.0, &mut rng())
                .unwrap();
        assert_eq!(measurement.identifier, 0);
        assert!((measurement.range - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_noise_measurement_matches_closed_form() {
        let beacons = [Beacon {
            identifier: 0,
            position: [3.0, 4.0],
        }];
        let measurement =
            nearest_beacon_measurement(&beacons, [0.0, 0.0, 0.0], 10.0, 0.0, 0.0, &mut rng())
                .unwrap();
        assert!((measurement.range - 5.0).abs() < 1e-12);
        assert!((measurement.bearing - 4.0_f64.atan2(3.0)).abs() < 1e-12);
    }

    #[test]
    fn bearing_is_relative_to_heading() {
        let beacons = [Beacon {
            identifier: 0,
            position: [1.0, 0.0],
        }];
        let measurement =
            nearest_beacon_measurement(&beacons, [0.0, 0.0, PI / 2.0], 10.0, 0.0, 0.0, &mut rng())
                .unwrap();
        assert!((measurement.bearing + PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn bearing_behind_the_robot_is_wrapped() {
        let beacons = [Beacon {
            identifier: 0,
            position: [-1.0, 0.0],
        }];
        let behind =
            nearest_beacon_measurement(&beacons, [0.0, 0.0, 0.0], 10.0, 0.0, 0.0, &mut rng())
                .unwrap();
        assert!((behind.bearing.abs() - PI).abs() < 1e-12);

        let facing =
            nearest_beacon_measurement(&beacons, [0.0, 0.0, PI], 10.0, 0.0, 0.0, &mut rng())
                .unwrap();
        assert!(facing.bearing.abs() < 1e-12);
    }

    #[test]
    fn out_of_range_returns_none() {
        let beacons = [Beacon {
            identifier: 0,
            position: [5.0, 0.0],
        }];
        assert!(
            nearest_beacon_measurement(&beacons, [0.0, 0.0, 0.0], 4.0, 0.0, 0.0, &mut rng())
                .is_none()
        );
        assert!(
            nearest_beacon_measurement(&[], [0.0, 0.0, 0.0], 10.0, 0.0, 0.0, &mut rng()).is_none()
        );
    }
}
