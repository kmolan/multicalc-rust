//! A three-axis inertial unit: what a body feels as it turns and is pushed about, reported the way
//! a real one reports it — late, offset, jittery, and shaking along with the machine it is bolted
//! to.
//!
//! Two faults of a real unit are deliberately left out, and it is worth saying so rather than
//! leaving it to be noticed. Each axis here is exactly as sensitive as it claims to be, and the
//! three sit exactly square to the body and to each other. A real unit is out on both counts: an
//! axis reads a fraction of a per cent high or low, and it is bolted a fraction of a degree away
//! from where it is meant to point, so a little of what one axis feels shows up on the others. They
//! are left out because they are a different kind of fault from everything above — what they add
//! grows with the size of the reading instead of sitting at a level of its own, so the steady
//! offsets the filter works out cannot stand in for them, and putting them in would mean widening
//! what the filter claims about itself until the claim covered them. Nothing here measures what
//! they would cost.

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

/// A reading the unit has taken but not yet handed out, with the offsets that went into it and how
/// far round the shake was when it was taken.
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeldBack {
    reading: InertialReading3d,
    offsets: InertialReading3d,
    shake_phase: f64,
}

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
/// that is easiest to remove, because it sits at a frequency that is known. On top of all three the
/// answer arrives a tick late: what the unit hands over is the sample it took last time it was
/// asked, because a reading has to be taken, converted and carried across a wire before anything
/// can act on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialMeasurementUnit3d {
    angular_rate_jitter: f64,
    acceleration_jitter: f64,
    angular_rate_offset_wander: f64,
    acceleration_offset_wander: f64,
    angular_rate_shake: f64,
    acceleration_shake: f64,
    second_harmonic_share: f64,
    angular_rate_offset: Vector3D<f64>,
    acceleration_offset: Vector3D<f64>,
    held_back: Option<HeldBack>,
    shake_phase: f64,
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
            angular_rate_shake: 0.0,
            acceleration_shake: 0.0,
            second_harmonic_share: 0.0,
            angular_rate_offset: Vector::zeros(),
            acceleration_offset: Vector::zeros(),
            held_back: None,
            shake_phase: 0.0,
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
    /// `angular_rate` and `acceleration` are how hard the shake is felt on each half of the
    /// reading, and `second_harmonic_share` is how much of that again comes back at twice the rate.
    /// How fast the rotors are coming round is not fixed here: it is handed to
    /// [`InertialMeasurementUnit3d::read`] afresh every tick, because it moves with how hard the
    /// rotors are pushing.
    #[inline]
    #[must_use]
    pub fn with_shake(
        mut self,
        angular_rate: f64,
        acceleration: f64,
        second_harmonic_share: f64,
    ) -> Self {
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

    /// The offsets that went into the reading the unit is about to hand over, as a reading of its
    /// own.
    ///
    /// This is the thing an estimate has to work out for itself. Nothing that flies may look at it;
    /// it is here so a measurement can say how much of an error was the offset and how much was
    /// everything else. It follows the reading rather than the moment, so what it says always
    /// belongs to the sample that is being reported and not to the one still being taken.
    #[inline]
    #[must_use]
    pub fn offsets(&self) -> InertialReading3d {
        match self.held_back {
            Some(held) => held.offsets,
            None => InertialReading3d::new(self.angular_rate_offset, self.acceleration_offset),
        }
    }

    /// How far round the shake had come when the reading the unit is about to hand over was taken,
    /// in radians.
    ///
    /// Nothing that flies may look at this either. It is here so a measurement can pick the shake
    /// out of a reading by matching it against the wave that put it there, which is a thing no
    /// amount of jitter can imitate — and a wave whose rate keeps moving cannot be matched by
    /// guessing at the rate afterwards.
    #[inline]
    #[must_use]
    pub fn shake_phase(&self) -> f64 {
        match self.held_back {
            Some(held) => held.shake_phase,
            None => self.shake_phase,
        }
    }

    /// Takes a reading of the truth, hands back the one taken a tick ago, and moves the offsets and
    /// the shake on by one tick.
    ///
    /// `true_angular_rate` and `true_proper_acceleration` are both in the body's own axes.
    /// `rotor_tone_hertz` is how fast the rotors are coming round at this moment, or nothing at all
    /// when they are not turning — a unit on a still machine sits quiet, and the shake appears the
    /// moment the rotors are running.
    ///
    /// What comes back is the previous sample, not the one just taken: a real unit measures,
    /// converts and sends, and whatever acts on the answer is therefore always acting on a body as
    /// it was a moment ago. The very first call has nothing older to give and hands back what it
    /// just took.
    pub fn read(
        &mut self,
        true_angular_rate: Vector3D<f64>,
        true_proper_acceleration: Vector3D<f64>,
        timestep: f64,
        rotor_tone_hertz: Option<f64>,
        rng: &mut Pcg32,
    ) -> InertialReading3d {
        let phase = self.shake_phase;
        let running = rotor_tone_hertz.is_some();
        let offsets = InertialReading3d::new(self.angular_rate_offset, self.acceleration_offset);
        let angular_rate = Vector::from_fn(|axis| {
            true_angular_rate[axis]
                + self.angular_rate_offset[axis]
                + gaussian_noise(self.angular_rate_jitter, rng)
                + self.shake(axis, phase, self.angular_rate_shake, running)
        });
        let proper_acceleration = Vector::from_fn(|axis| {
            true_proper_acceleration[axis]
                + self.acceleration_offset[axis]
                + gaussian_noise(self.acceleration_jitter, rng)
                + self.shake(axis, phase, self.acceleration_shake, running)
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
        // The shake is carried as how far round it has come rather than worked out from the time,
        // because the rate it comes round at keeps changing and only the phase joins one tick's
        // wave smoothly onto the next.
        let full_turn = std::f64::consts::TAU;
        let turned = full_turn * rotor_tone_hertz.unwrap_or(0.0) * timestep.max(0.0);
        self.shake_phase = (phase + turned) % full_turn;

        let just_taken = HeldBack {
            reading: InertialReading3d::new(angular_rate, proper_acceleration),
            offsets,
            shake_phase: phase,
        };
        match self.held_back.replace(just_taken) {
            Some(waiting) => waiting.reading,
            None => just_taken.reading,
        }
    }

    /// How far the shake has carried one axis at one moment.
    fn shake(&self, axis: usize, phase: f64, amplitude: f64, rotors_running: bool) -> f64 {
        if !rotors_running || amplitude == 0.0 {
            return 0.0;
        }
        let offset = SHAKE_PHASES[axis % SHAKE_PHASES.len()];
        amplitude
            * ((phase + offset).sin() + self.second_harmonic_share * (2.0 * phase + offset).sin())
    }
}
