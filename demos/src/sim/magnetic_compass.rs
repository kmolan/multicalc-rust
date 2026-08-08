//! A compass: a noisy reading of which way the body's nose points.

use rand_pcg::Pcg32;

use multicalc::scalar::Numeric;

use super::sensor_noise::gaussian_noise;

/// A compass, read level: which way the nose points about the world's upright axis.
///
/// The reading is taken in the level plane rather than along whatever the body's own axes happen to
/// be, so it is never asked which way it is facing while pointing straight up — a question with no
/// answer, and one a machine that banks hard would otherwise ask several times a flight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagneticCompass {
    heading_noise: f64,
}

impl MagneticCompass {
    /// A compass from how much its answer jitters, in radians.
    #[must_use]
    pub fn new(heading_noise: f64) -> Self {
        MagneticCompass { heading_noise }
    }

    /// How much the answer jitters.
    #[inline]
    #[must_use]
    pub fn heading_noise(&self) -> f64 {
        self.heading_noise
    }

    /// A noisy reading of the true heading, folded back into the half turn either side of zero.
    #[must_use]
    pub fn read(&self, true_heading: f64, rng: &mut Pcg32) -> f64 {
        (true_heading + gaussian_noise(self.heading_noise, rng)).wrap_to_pi()
    }
}
