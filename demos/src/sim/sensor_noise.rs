//! The jitter every simulated sensor adds to its reading.

use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

/// One Gaussian draw, or zero when there is no noise to add.
///
/// A zero or negative deviation means a perfect sensor, which the draw itself cannot express, so it
/// is answered directly.
pub fn gaussian_noise(deviation: f64, rng: &mut Pcg32) -> f64 {
    if deviation <= 0.0 {
        return 0.0;
    }
    Normal::new(0.0, deviation)
        .map(|distribution| distribution.sample(rng))
        .unwrap_or(0.0)
}
