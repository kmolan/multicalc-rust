//! A downward beam that measures how far the ground is below.

use rand_pcg::Pcg32;

use super::sensor_noise::gaussian_noise;

/// How far the body may be leaning before the beam's answer is thrown away, in radians.
///
/// The correction from a slanted beam to a height divides by the cosine of the lean, which grows
/// without bound as the body tips toward its side. Well before that it is measuring the ground
/// somewhere off to one side rather than the ground underneath, so the honest answer past this
/// point is no answer.
const MAXIMUM_TRUSTED_LEAN: f64 = 0.6;

/// A beam pointed straight down the body's own axis, reporting how high the body is.
///
/// The beam measures along itself, so a leaning body reads long: the ground is further away down a
/// slanted line than it is straight below. Correcting for that needs the lean, which the machine
/// only has an estimate of — so this is a sensor whose answer is only as good as the attitude that
/// corrects it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightRangefinder {
    range_noise: f64,
    maximum_range: f64,
}

impl HeightRangefinder {
    /// A beam from how much its answer jitters and how far it reaches, both in metres.
    #[must_use]
    pub fn new(range_noise: f64, maximum_range: f64) -> Self {
        HeightRangefinder {
            range_noise,
            maximum_range,
        }
    }

    /// How much the answer jitters.
    #[inline]
    #[must_use]
    pub fn range_noise(&self) -> f64 {
        self.range_noise
    }

    /// How high the body is above flat ground, or nothing at all when the beam cannot answer.
    ///
    /// There is no answer when the ground is further away than the beam reaches, when the body is
    /// leaning too far for the slant to be corrected, or when the body is at or below the ground.
    /// `lean` is the angle between the beam and straight down, which is the body's own lean.
    #[must_use]
    pub fn read(&self, true_height: f64, lean: f64, rng: &mut Pcg32) -> Option<f64> {
        if true_height <= 0.0 || lean.abs() > MAXIMUM_TRUSTED_LEAN {
            return None;
        }
        let upright_share = lean.cos();
        let along_the_beam = true_height / upright_share + gaussian_noise(self.range_noise, rng);
        if along_the_beam > self.maximum_range || along_the_beam <= 0.0 {
            return None;
        }
        Some(along_the_beam * upright_share)
    }
}
