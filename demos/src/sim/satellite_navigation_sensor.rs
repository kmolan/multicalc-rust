//! A satellite receiver: where the body is and how fast it is going, both noisy, and one of the two
//! far better than the other.

use rand_pcg::Pcg32;

use multicalc::linear_algebra::{Vector, Vector3D};

use super::sensor_noise::gaussian_noise;

/// A satellite receiver: it reports where the body is and how fast it is moving, and it is wrong
/// about the first of those by rather more than most people expect.
///
/// Height is the worse half of the position, by about a factor of two. Every satellite in view sits
/// above the receiver, so they all pull on the height answer from broadly the same direction and
/// none of them pins it down from the side — which is why a machine that has to hold a height
/// carries something else to hold it with.
///
/// Speed is a different matter entirely, and far better known. It is not worked out by watching the
/// position change; it comes from how the satellites' signals are pitched up or down by the
/// receiver's own motion, which is a much finer thing to measure. A machine that only ever asks
/// this box where it is throws away the better half of what it can say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatelliteNavigationSensor {
    horizontal_noise: f64,
    vertical_noise: f64,
    horizontal_speed_noise: f64,
    vertical_speed_noise: f64,
}

impl SatelliteNavigationSensor {
    /// A receiver from how far its position answer wanders across the ground and up and down.
    ///
    /// Its speed answer starts perfect; [`SatelliteNavigationSensor::with_speed_noise`] gives that
    /// half its own wander.
    #[must_use]
    pub fn new(horizontal_noise: f64, vertical_noise: f64) -> Self {
        SatelliteNavigationSensor {
            horizontal_noise,
            vertical_noise,
            horizontal_speed_noise: 0.0,
            vertical_speed_noise: 0.0,
        }
    }

    /// Sets how far the speed answer wanders, across the ground and up and down.
    #[inline]
    #[must_use]
    pub fn with_speed_noise(mut self, horizontal: f64, vertical: f64) -> Self {
        self.horizontal_speed_noise = horizontal;
        self.vertical_speed_noise = vertical;
        self
    }

    /// How far the position answer wanders across the ground.
    #[inline]
    #[must_use]
    pub fn horizontal_noise(&self) -> f64 {
        self.horizontal_noise
    }

    /// How far it wanders up and down.
    #[inline]
    #[must_use]
    pub fn vertical_noise(&self) -> f64 {
        self.vertical_noise
    }

    /// How far the speed answer wanders across the ground.
    #[inline]
    #[must_use]
    pub fn horizontal_speed_noise(&self) -> f64 {
        self.horizontal_speed_noise
    }

    /// How far the speed answer wanders up and down.
    #[inline]
    #[must_use]
    pub fn vertical_speed_noise(&self) -> f64 {
        self.vertical_speed_noise
    }

    /// A fix of the true position, each part with its own noise draw.
    pub fn read_position(&self, true_position: Vector3D<f64>, rng: &mut Pcg32) -> Vector3D<f64> {
        Vector::new([
            true_position[0] + gaussian_noise(self.horizontal_noise, rng),
            true_position[1] + gaussian_noise(self.horizontal_noise, rng),
            true_position[2] + gaussian_noise(self.vertical_noise, rng),
        ])
    }

    /// A reading of the true velocity, each part with its own noise draw.
    pub fn read_velocity(&self, true_velocity: Vector3D<f64>, rng: &mut Pcg32) -> Vector3D<f64> {
        Vector::new([
            true_velocity[0] + gaussian_noise(self.horizontal_speed_noise, rng),
            true_velocity[1] + gaussian_noise(self.horizontal_speed_noise, rng),
            true_velocity[2] + gaussian_noise(self.vertical_speed_noise, rng),
        ])
    }
}
