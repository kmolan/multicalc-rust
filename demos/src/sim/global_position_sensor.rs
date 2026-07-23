//! A GPS-class sensor: a noisy absolute position.

use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

/// A GPS-class sensor: it reports a noisy absolute position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalPositionSensor {
    /// How much each reported coordinate jitters.
    position_noise: f64,
}

impl GlobalPositionSensor {
    pub fn new(position_noise: f64) -> Self {
        GlobalPositionSensor { position_noise }
    }

    /// A noisy fix of the true position. Each coordinate gets its own noise draw.
    pub fn read(&self, true_position: [f64; 2], rng: &mut Pcg32) -> [f64; 2] {
        [
            true_position[0] + noise(self.position_noise, rng),
            true_position[1] + noise(self.position_noise, rng),
        ]
    }
}

/// One Gaussian draw, or zero when there is no noise to add.
fn noise(deviation: f64, rng: &mut Pcg32) -> f64 {
    if deviation <= 0.0 {
        return 0.0;
    }
    Normal::new(0.0, deviation).map(|n| n.sample(rng)).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rand::SeedableRng;

    #[test]
    fn zero_noise_reads_the_truth() {
        let sensor = GlobalPositionSensor::new(0.0);
        let mut rng = Pcg32::seed_from_u64(2);
        assert_eq!(sensor.read([3.0, -4.0], &mut rng), [3.0, -4.0]);
    }

    #[test]
    fn the_spread_matches_the_noise() {
        let deviation = 0.3;
        let sensor = GlobalPositionSensor::new(deviation);
        let mut rng = Pcg32::seed_from_u64(3);
        let count = 4000;
        let (mut sum, mut sum_of_squares) = (0.0, 0.0);
        for _ in 0..count {
            let reading = sensor.read([1.0, -2.0], &mut rng);
            let residual = reading[0] - 1.0;
            sum += residual;
            sum_of_squares += residual * residual;
        }
        let mean = sum / count as f64;
        let spread = (sum_of_squares / count as f64 - mean * mean).sqrt();
        assert!(mean.abs() < 0.02, "residuals should centre on zero: {mean}");
        assert!((spread - deviation).abs() < 0.03, "spread off the noise: {spread}");
    }

    #[test]
    fn a_fixed_seed_reproduces_the_reading() {
        let sensor = GlobalPositionSensor::new(0.3);
        let mut first = Pcg32::seed_from_u64(8);
        let mut second = Pcg32::seed_from_u64(8);
        assert_eq!(sensor.read([1.0, 2.0], &mut first), sensor.read([1.0, 2.0], &mut second));
    }
}
