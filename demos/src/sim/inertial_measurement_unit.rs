//! An attitude-and-heading sensor: a noisy absolute heading and turn rate, standing in for an
//! AHRS-class inertial unit.

use rand_pcg::Pcg32;

use super::geometry::wrap_angle;
use super::sensor_noise::gaussian_noise;

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
    #[must_use]
    pub fn new(heading_noise: f64, yaw_rate_noise: f64, heading_bias: f64) -> Self {
        InertialMeasurementUnit {
            heading_noise,
            yaw_rate_noise,
            heading_bias,
        }
    }

    /// Reads the true heading and turn rate, each with added noise; the heading also carries the
    /// fixed offset. The reported heading is folded back into (-π, π].
    #[must_use]
    pub fn read(&self, true_heading: f64, true_yaw_rate: f64, rng: &mut Pcg32) -> InertialReading {
        InertialReading {
            heading: wrap_angle(
                true_heading + self.heading_bias + gaussian_noise(self.heading_noise, rng),
            ),
            yaw_rate: true_yaw_rate + gaussian_noise(self.yaw_rate_noise, rng),
        }
    }
}
