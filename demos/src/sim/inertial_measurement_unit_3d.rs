//! A three-axis inertial unit: what a body feels as it turns and is pushed about, reported the way
//! a real one reports it — late, offset, jittery, and shaking along with the machine it is bolted
//! to.

use rand_pcg::Pcg32;

use multicalc::linear_algebra::{Vector, Vector3D};

use super::sensor_noise::gaussian_noise;

/// How far round the shake is turned on each axis, so the three do not all swing together.
///
/// A unit whose three axes shook in step would leave a wobble that points in one fixed direction,
/// which is not what a machine bolted to a spinning motor feels and is far easier to filter out
/// than the real thing.
const SHAKE_PHASES: [f64; 3] = [
    0.0,
    std::f64::consts::TAU / 3.0,
    2.0 * std::f64::consts::TAU / 3.0,
];

/// What the unit reports, both in the body's own axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialReading3d {
    /// How fast the body is turning about each of its own axes, in radians per second.
    pub angular_rate: Vector3D<f64>,
    /// The push the body feels, gravity left out, in metres per second squared.
    ///
    /// A machine sitting still on the ground reads this as pointing straight up at one gravity: the
    /// unit feels what is holding the body up, not the pull it is being held up against.
    pub proper_acceleration: Vector3D<f64>,
}

impl InertialReading3d {
    /// A reading from its two parts.
    #[inline]
    #[must_use]
    pub fn new(angular_rate: Vector3D<f64>, proper_acceleration: Vector3D<f64>) -> Self {
        InertialReading3d {
            angular_rate,
            proper_acceleration,
        }
    }
}

/// A three-axis inertial unit, with everything that stands between what the body does and what the
/// unit says it did.
///
/// Three things are added to the truth on both the turn rate and the push: a steady offset that
/// wanders slowly over a flight, jitter that is different in every single reading, and — while the
/// rotors are turning — a shake at the rotors' own rate and at twice that again. The offset is the
/// one that hurts, because it does not average away however long it is watched; the shake is the one
/// that is easiest to remove, because it sits at a frequency that is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialMeasurementUnit3d {
    angular_rate_jitter: f64,
    acceleration_jitter: f64,
    angular_rate_offset_wander: f64,
    acceleration_offset_wander: f64,
    rotor_tone_hertz: f64,
    angular_rate_shake: f64,
    acceleration_shake: f64,
    second_harmonic_share: f64,
    angular_rate_offset: Vector3D<f64>,
    acceleration_offset: Vector3D<f64>,
    time: f64,
}

impl InertialMeasurementUnit3d {
    /// A plain unit: one that jitters by this much on the turn rate and on the push, and does
    /// nothing else wrong.
    ///
    /// The jitter is how far a single reading wanders from the truth, and it is the one fault every
    /// unit has. The rest — an offset that drifts, and the machine's own shaking — are added on top
    /// with [`InertialMeasurementUnit3d::with_offset_wander`],
    /// [`InertialMeasurementUnit3d::with_shake`] and
    /// [`InertialMeasurementUnit3d::with_starting_offsets`].
    #[must_use]
    pub fn new(angular_rate_jitter: f64, acceleration_jitter: f64) -> Self {
        InertialMeasurementUnit3d {
            angular_rate_jitter,
            acceleration_jitter,
            angular_rate_offset_wander: 0.0,
            acceleration_offset_wander: 0.0,
            rotor_tone_hertz: 0.0,
            angular_rate_shake: 0.0,
            acceleration_shake: 0.0,
            second_harmonic_share: 0.0,
            angular_rate_offset: Vector::zeros(),
            acceleration_offset: Vector::zeros(),
            time: 0.0,
        }
    }

    /// Lets the steady offsets drift, by this much over a second of flight.
    ///
    /// This is what makes an offset watched for a long time no better known than one watched
    /// briefly, and it is why an estimate has to keep working the offsets out rather than measuring
    /// them once on the pad.
    #[inline]
    #[must_use]
    pub fn with_offset_wander(mut self, angular_rate: f64, acceleration: f64) -> Self {
        self.angular_rate_offset_wander = angular_rate;
        self.acceleration_offset_wander = acceleration;
        self
    }

    /// Shakes the unit along with the machine it is bolted to.
    ///
    /// `rotor_tone_hertz` is how fast the rotors come round, `angular_rate` and `acceleration` are
    /// how hard the shake is felt on each half of the reading, and `second_harmonic_share` is how
    /// much of that again comes back at twice the rate.
    #[inline]
    #[must_use]
    pub fn with_shake(
        mut self,
        rotor_tone_hertz: f64,
        angular_rate: f64,
        acceleration: f64,
        second_harmonic_share: f64,
    ) -> Self {
        self.rotor_tone_hertz = rotor_tone_hertz;
        self.angular_rate_shake = angular_rate;
        self.acceleration_shake = acceleration;
        self.second_harmonic_share = second_harmonic_share;
        self
    }

    /// Starts the unit already carrying an offset, rather than at a perfect zero.
    ///
    /// Every real unit is switched on with one, and no amount of watching a still machine takes it
    /// away, so a demo that starts at zero is easier than the thing it stands for.
    #[inline]
    #[must_use]
    pub fn with_starting_offsets(
        mut self,
        angular_rate_offset: Vector3D<f64>,
        acceleration_offset: Vector3D<f64>,
    ) -> Self {
        self.angular_rate_offset = angular_rate_offset;
        self.acceleration_offset = acceleration_offset;
        self
    }

    /// The offsets the unit is carrying at this moment, as a reading of its own.
    ///
    /// This is the thing an estimate has to work out for itself. Nothing that flies may look at it;
    /// it is here so a measurement can say how much of an error was the offset and how much was
    /// everything else.
    #[inline]
    #[must_use]
    pub fn offsets(&self) -> InertialReading3d {
        InertialReading3d::new(self.angular_rate_offset, self.acceleration_offset)
    }

    /// Reads the truth once, and moves the offsets on by one tick's worth of wandering.
    ///
    /// `true_angular_rate` and `true_proper_acceleration` are both in the body's own axes.
    /// `rotors_running` says whether the machine is shaking at all — a unit on a still machine sits
    /// quiet, and the shake appears the moment the rotors are turning.
    pub fn read(
        &mut self,
        true_angular_rate: Vector3D<f64>,
        true_proper_acceleration: Vector3D<f64>,
        timestep: f64,
        rotors_running: bool,
        rng: &mut Pcg32,
    ) -> InertialReading3d {
        let time = self.time;
        let angular_rate = Vector::from_fn(|axis| {
            true_angular_rate[axis]
                + self.angular_rate_offset[axis]
                + gaussian_noise(self.angular_rate_jitter, rng)
                + self.shake(axis, time, self.angular_rate_shake, rotors_running)
        });
        let proper_acceleration = Vector::from_fn(|axis| {
            true_proper_acceleration[axis]
                + self.acceleration_offset[axis]
                + gaussian_noise(self.acceleration_jitter, rng)
                + self.shake(axis, time, self.acceleration_shake, rotors_running)
        });

        // The offsets take a small random step every tick, so how far they have wandered grows with
        // the square root of the time flown rather than with the time itself.
        let wander = timestep.max(0.0).sqrt();
        self.angular_rate_offset = Vector::from_fn(|axis| {
            self.angular_rate_offset[axis]
                + gaussian_noise(self.angular_rate_offset_wander * wander, rng)
        });
        self.acceleration_offset = Vector::from_fn(|axis| {
            self.acceleration_offset[axis]
                + gaussian_noise(self.acceleration_offset_wander * wander, rng)
        });
        self.time = time + timestep;

        InertialReading3d::new(angular_rate, proper_acceleration)
    }

    /// How far the shake has carried one axis at one moment.
    fn shake(&self, axis: usize, time: f64, amplitude: f64, rotors_running: bool) -> f64 {
        if !rotors_running || amplitude == 0.0 {
            return 0.0;
        }
        let phase = SHAKE_PHASES[axis % SHAKE_PHASES.len()];
        let turn = std::f64::consts::TAU * self.rotor_tone_hertz * time;
        amplitude * ((turn + phase).sin() + self.second_harmonic_share * (2.0 * turn + phase).sin())
    }
}
