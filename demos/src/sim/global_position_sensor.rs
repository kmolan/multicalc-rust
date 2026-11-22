//! A GPS-class sensor: a noisy absolute position.

use rand_pcg::Pcg32;

use super::sensor_noise::gaussian_noise;

/// A GPS-class sensor: it reports a noisy absolute position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalPositionSensor {
    /// How much each reported coordinate jitters.
    position_noise: f64,
}

impl GlobalPositionSensor {
    #[must_use]
    pub fn new(position_noise: f64) -> Self {
        GlobalPositionSensor { position_noise }
    }

    /// A noisy fix of the true position. Each coordinate gets its own noise draw.
    #[must_use]
    pub fn read(&self, true_position: [f64; 2], rng: &mut Pcg32) -> [f64; 2] {
        [
            true_position[0] + gaussian_noise(self.position_noise, rng),
            true_position[1] + gaussian_noise(self.position_noise, rng),
        ]
    }
}
