//! An attitude-and-heading sensor: a noisy absolute heading and turn rate, standing in for an
//! AHRS-class inertial unit.

use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

use super::wrap_angle;

/// An attitude-and-heading sensor: it reports the vehicle's facing direction and turn rate, both
/// noisy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialMeasurementUnit {
    /// How much the reported heading jitters.
    heading_noise: f64,
    /// How much the reported turn rate jitters.
    yaw_rate_noise: f64,
    /// A small fixed offset the reported heading always carries.
    heading_bias: f64,
}

/// A reading: the vehicle's facing direction and its turn rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialReading {
    pub heading: f64,
    pub yaw_rate: f64,
}

impl InertialMeasurementUnit {
    pub fn new(heading_noise: f64, yaw_rate_noise: f64, heading_bias: f64) -> Self {
        InertialMeasurementUnit {
            heading_noise,
            yaw_rate_noise,
            heading_bias,
        }
    }

    /// Reads the true heading and turn rate, each with added noise; the heading also carries the
    /// fixed offset. The reported heading is folded back into (-π, π].
    pub fn read(&self, true_heading: f64, true_yaw_rate: f64, rng: &mut Pcg32) -> InertialReading {
        InertialReading {
            heading: wrap_angle(true_heading + self.heading_bias + noise(self.heading_noise, rng)),
            yaw_rate: true_yaw_rate + noise(self.yaw_rate_noise, rng),
        }
    }
}

/// One Gaussian draw, or zero when there is no noise to add.
fn noise(deviation: f64, rng: &mut Pcg32) -> f64 {
    if deviation <= 0.0 {
        return 0.0;
    }
    Normal::new(0.0, deviation)
        .map(|n| n.sample(rng))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    #[test]
    fn zero_noise_and_offset_returns_the_truth() {
        let mut rng = Pcg32::seed_from_u64(1);
        let unit = InertialMeasurementUnit::new(0.0, 0.0, 0.0);
        // A heading past π is folded back into range; the turn rate passes through unchanged.
        let reading = unit.read(4.0, 0.2, &mut rng);
        assert!(
            (reading.heading - wrap_angle(4.0)).abs() < 1e-12,
            "heading: {}",
            reading.heading
        );
        assert_eq!(reading.yaw_rate, 0.2);
    }

    #[test]
    fn a_heading_near_pi_plus_offset_wraps_past_pi() {
        let mut rng = Pcg32::seed_from_u64(2);
        let unit = InertialMeasurementUnit::new(0.0, 0.0, 0.1);
        // Just under π plus the offset lands past π, so it should fold to the negative side rather
        // than report a value above π.
        let reading = unit.read(PI - 0.05, 0.0, &mut rng);
        assert!(
            reading.heading < 0.0,
            "should wrap to negative: {}",
            reading.heading
        );
        assert!(
            reading.heading > -PI && reading.heading <= PI,
            "out of range: {}",
            reading.heading
        );
        assert!(
            (reading.heading - wrap_angle(PI - 0.05 + 0.1)).abs() < 1e-12,
            "heading: {}",
            reading.heading
        );
    }

    #[test]
    fn a_fixed_seed_reproduces_the_reading() {
        let unit = InertialMeasurementUnit::new(0.02, 0.01, 0.01);
        let mut first = Pcg32::seed_from_u64(7);
        let mut second = Pcg32::seed_from_u64(7);
        assert_eq!(
            unit.read(0.3, -0.4, &mut first),
            unit.read(0.3, -0.4, &mut second)
        );
    }
}
