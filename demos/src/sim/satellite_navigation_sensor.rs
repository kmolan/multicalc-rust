//! A satellite receiver: where the body is and how fast it is going, both noisy, and one of the two
//! far better than the other.
//!
//! There are no satellites in here, no signals and no geometry. Each of the four ways the answer is
//! wrong — the jitter that averages away, the drift that does not, the height being the worse half,
//! and the occasional answer that is simply wrong by metres — is drawn at the size a real receiver's
//! answer carries it, because the size and the shape are what a filter downstream has to cope with
//! and where it came from is not. A receiver worked out from the sky above it would be a different
//! and much larger piece of work, and it would tell the filter nothing new.

use rand::RngExt;
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
///
/// How the position is wrong matters as much as how far. Part of the error is fresh in every fix
/// and averages away if you watch long enough; the rest drifts, holds where it has drifted to for
/// the best part of a minute, and averages away not at all — that part comes from the path the
/// signals took through the air and past whatever they bounced off on the way, and none of that
/// changes just because another fix was asked for. And every so often the answer is simply wrong,
/// by metres, because the receiver has taken a reflection for the real thing. Both are set up with
/// [`SatelliteNavigationSensor::with_wandering_share`] and
/// [`SatelliteNavigationSensor::with_outliers`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatelliteNavigationSensor {
    horizontal_noise: f64,
    vertical_noise: f64,
    horizontal_speed_noise: f64,
    vertical_speed_noise: f64,
    wandering_share: f64,
    wander_settling_seconds: f64,
    outlier_chance: f64,
    outlier_size: f64,
    wander: Vector3D<f64>,
}

impl SatelliteNavigationSensor {
    /// A receiver from how far its position answer wanders across the ground and up and down.
    ///
    /// Those two are the whole of how far off the position is, however it is made up. A plain
    /// receiver draws all of it fresh in every fix; [`SatelliteNavigationSensor::with_wandering_share`]
    /// moves some of it into a slow drift without making the total any wider.
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
            wandering_share: 0.0,
            wander_settling_seconds: 1.0,
            outlier_chance: 0.0,
            outlier_size: 0.0,
            wander: Vector::zeros(),
        }
    }

    /// Makes part of the position error a slow drift instead of fresh noise in every fix.
    ///
    /// `share` is how much of the declared spread drifts, as a share of it, and
    /// `settling_seconds` is roughly how long the drift takes to forget where it was. The two
    /// halves are drawn so the total spread stays exactly what was declared — nothing here makes
    /// the receiver worse, it makes it wrong in a way that lasts.
    ///
    /// That distinction is the whole point. Fresh noise averages away, so a filter folding in a
    /// fix every twentieth of a second can grow far surer of where it is than one fix would let it.
    /// Against a drift that holds for a minute it cannot: the twentieth fix says almost exactly
    /// what the first one did.
    #[inline]
    #[must_use]
    pub fn with_wandering_share(mut self, share: f64, settling_seconds: f64) -> Self {
        self.wandering_share = share.clamp(0.0, 1.0);
        self.wander_settling_seconds = settling_seconds.max(f64::EPSILON);
        self
    }

    /// Lets the odd fix come back plainly wrong.
    ///
    /// `chance` is how often that happens, as a share of fixes, and `size` is the spread the wrong
    /// one is drawn from — many times the receiver's own, because this is not a bad fix but a
    /// different one entirely, taken off a reflection. It is what the check that throws a fix away
    /// exists for.
    #[inline]
    #[must_use]
    pub fn with_outliers(mut self, chance: f64, size: f64) -> Self {
        self.outlier_chance = chance;
        self.outlier_size = size;
        self
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

    /// A fix of the true position: the slow drift moved on by `seconds_since_the_last_fix`, fresh
    /// noise on top of it, and once in a while a wrong answer instead of either.
    pub fn read_position(
        &mut self,
        true_position: Vector3D<f64>,
        seconds_since_the_last_fix: f64,
        rng: &mut Pcg32,
    ) -> Vector3D<f64> {
        let spread = Vector::new([
            self.horizontal_noise,
            self.horizontal_noise,
            self.vertical_noise,
        ]);

        // The drift forgets a little of where it was and picks up a little that is new, by amounts
        // that leave it the same size however long it is watched.
        let forgotten = (-seconds_since_the_last_fix.max(0.0) / self.wander_settling_seconds).exp();
        let picked_up = (1.0 - forgotten * forgotten).max(0.0).sqrt();
        self.wander = Vector::from_fn(|axis| {
            self.wander[axis] * forgotten
                + gaussian_noise(spread[axis] * self.wandering_share * picked_up, rng)
        });

        // What is left of the spread once the drift has taken its share, so the two together come
        // to exactly what the receiver claims.
        let fresh_share = (1.0 - self.wandering_share * self.wandering_share)
            .max(0.0)
            .sqrt();
        let fix = Vector::from_fn(|axis| {
            true_position[axis]
                + self.wander[axis]
                + gaussian_noise(spread[axis] * fresh_share, rng)
        });

        if rng.random::<f64>() < self.outlier_chance {
            return Vector::from_fn(|axis| fix[axis] + gaussian_noise(self.outlier_size, rng));
        }
        fix
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
