//! A downward beam that measures how far the ground is below.

use rand_pcg::Pcg32;

use super::sensor_noise::gaussian_noise;

/// How far the body may be leaning before the beam's answer is thrown away, in radians.
///
/// The reading grows without bound as the body tips toward its side, and whatever undoes that has
/// to divide by the cosine of a lean it only knows approximately. Well before that the beam is
/// measuring the ground somewhere off to one side rather than the ground underneath, so the honest
/// answer past this point is no answer.
const MAXIMUM_TRUSTED_LEAN: f64 = 0.6;

/// A beam pointed straight down the body's own axis, reporting how far away the ground is along it.
///
/// The beam measures along itself, so a leaning body reads long: the ground is further away down a
/// slanted line than it is straight below. What comes back is that slanted distance and nothing
/// else. Turning it into a height needs the lean, which the machine only has an estimate of — so
/// this is a sensor whose answer is only as good as the attitude that corrects it, and the
/// correcting is left to whoever holds an attitude to correct it with.
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

    /// How far the ground is down the beam, or nothing at all when the beam cannot answer.
    ///
    /// There is no answer when the ground is further away than the beam reaches, when the body is
    /// leaning so far that the beam is measuring the ground somewhere off to one side rather than
    /// the ground underneath, or when the body is at or below the ground. `lean` is the true angle
    /// between the beam and straight down, which is what makes the reading long; it is the geometry
    /// the beam is in and not something the answer has been corrected by.
    #[must_use]
    pub fn read(&self, true_height: f64, lean: f64, rng: &mut Pcg32) -> Option<f64> {
        if true_height <= 0.0 || lean.abs() > MAXIMUM_TRUSTED_LEAN {
            return None;
        }
        let along_the_beam = true_height / lean.cos() + gaussian_noise(self.range_noise, rng);
        if along_the_beam > self.maximum_range || along_the_beam <= 0.0 {
            return None;
        }
        Some(along_the_beam)
    }
}
