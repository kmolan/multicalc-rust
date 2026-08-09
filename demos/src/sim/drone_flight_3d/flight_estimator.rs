//! What the machine works out about its own motion from what it can feel.
//!
//! The inertial unit reports the truth with a shake at the rotors' turning rate sitting on top of
//! it. That shake is far too fast for the loops to answer and far too big to hand them anyway, so
//! it is taken out first, at the frequency it is known to sit at. Everything downstream — the
//! estimate and both loops — is fed the cleaned-up reading and never the raw one.
//!
//! Two things are then done with the cleaned-up reading. One is the simplest thing there is:
//! integrate it and nothing else. That answer is right for a moment and hopeless after a minute,
//! which is exactly what makes it worth having on screen — it is what the machine would believe
//! with nothing to check itself against.
//!
//! The other is the fifteen-number filter, which carries the same integration but keeps a spread
//! alongside it and lets a satellite fix, a downward beam and a compass pull it back toward what
//! they see. It also works out, and keeps working out, what the unit's own steady offsets are —
//! which is the whole reason it beats the first answer, because those offsets are what makes plain
//! integration fail.

use multicalc::error::SignalError;
use multicalc::estimation::{
    ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn, residual_with_wrapped_angles,
};
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::ode::ExponentialMap;
use multicalc::scalar::Numeric;
use multicalc::signal_processing::{
    BiquadCoefficients, MultiChannelBiquad, harmonic_notch_coefficients,
};
use multicalc::spatial::{FreeJointState, SO3};

use crate::sim::inertial_measurement_unit_3d::InertialReading3d;

use super::flight_plant::level_heading;
use super::x2_model::GRAVITY_STRENGTH;

/// How many notches are stacked up: one on the rotors' own turning rate, one on twice it.
///
/// A third would sit at three times the rate, which for this machine is past half of how often the
/// unit is read, and the builder refuses it rather than quietly filtering something else.
const NOTCH_SECTIONS: usize = 2;

/// How narrow each notch is. High enough that the notch takes out the shake without taking the
/// body's own motion with it, low enough that it still catches a rotor rate that is a little off.
const NOTCH_QUALITY: f64 = 8.0;

/// What the machine believes about its own motion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimatedState {
    /// Where the body is, in world axes.
    pub position: Vector3D<f64>,
    /// How fast it is moving, in world axes.
    pub velocity: Vector3D<f64>,
    /// Which way it is facing.
    pub orientation: SO3<f64>,
    /// How fast it is turning, about its own axes.
    pub angular_rate: Vector3D<f64>,
}

impl EstimatedState {
    /// What a body in `state` really is, for handing the loops the truth instead of a belief.
    #[must_use]
    pub fn from_truth(state: FreeJointState<f64>) -> Self {
        EstimatedState {
            position: state.pose().translation(),
            velocity: state.velocity().linear(),
            orientation: state.pose().rotation(),
            angular_rate: state.velocity().angular(),
        }
    }
}

/// Where the controller's idea of the body comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSource {
    /// The controller is handed the truth. What the loops can do at their best.
    Truth,
    /// The controller is handed the inertial reading integrated on its own, with no aiding.
    DeadReckoning,
    /// The controller is handed the filter's answer. What the demo actually flies on.
    Filter,
}

/// How far off a satellite fix may be, measured against how far off the filter expects it to be,
/// before it is thrown away rather than believed.
///
/// Three numbers come in with every fix, and this is the value a sum of three such gaps stays under
/// ninety-nine times in a hundred when everything is behaving. A fix past it is either a bad fix or
/// a filter that has lost track of itself, and neither is worth folding in blind.
const WORST_BELIEVABLE_FIX: f64 = 11.34;

/// How many fixes may be thrown away in a row before the next one is taken anyway.
///
/// A filter that is genuinely lost thinks every fix disagrees with it, and would go on refusing the
/// very readings that would put it right. Taking one after a run of refusals is what stops a wrong
/// answer defending itself forever.
const MOST_REFUSALS_IN_A_ROW: u32 = 5;

/// How far the filter may believe the body is leaning before the downward beam is left alone, in
/// radians.
///
/// Standing a slanted reading up into a height divides by how much of the beam still points at the
/// ground, and that share falls away to nothing as the body tips toward its side: past a point, a
/// small error in the believed lean becomes a large one in the height. The beam itself stops
/// answering at much the same lean, for the same reason.
const WORST_LEAN_FOR_THE_BEAM: f64 = 0.6;

/// How sure the filter is of each part of its answer when it is switched on, as a spread.
///
/// These are not free numbers to tune. Each one is what its own part of the answer is really
/// expected to be out by on the pad, and telling the filter anything else is telling it a lie about
/// itself: too wide and it stays needlessly unsure and ignores readings it should be taking, too
/// narrow and it grows sure of itself faster than it has earned. Whether it was told the truth is
/// exactly what the consistency check measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartingSpreads {
    /// How far off the surveyed position might be, in metres.
    pub position: f64,
    /// How far off the starting speed might be, in metres per second.
    pub velocity: f64,
    /// How far from level the facing might be, in radians. This comes from the push sensor's own
    /// steady offset, which reads as a lean that is not there.
    pub tilt: f64,
    /// How far round the facing might be, in radians — the compass's own jitter.
    pub heading: f64,
    /// What the turn-rate sensor might be reading with the body held still, in radians per second.
    pub turn_rate_offset: f64,
    /// What the push sensor might be reading beyond the real push, in metres per second squared.
    pub push_offset: f64,
}

/// Where the machine is taken to have been set down, and how fast it is taken to have been moving.
///
/// Both are surveyed rather than measured, so both are out by something — and they have to be. A
/// filter handed the truth begins with no error at all while claiming the spread in
/// [`StartingSpreads`], which is the one moment in a flight where what it claims and what is so are
/// knowingly different. Drawing these from those same spreads is what keeps the two agreeing from
/// the first tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartingBelief {
    /// Where the machine is taken to be, in world axes.
    pub position: Vector3D<f64>,
    /// How fast it is taken to be moving, in world axes.
    pub velocity: Vector3D<f64>,
}

impl StartingSpreads {
    /// The fifteen numbers laid out in the order the filter carries them.
    fn as_diagonal(self) -> [f64; 15] {
        let mut spreads = [0.0; 15];
        for axis in 0..3 {
            spreads[axis] = self.position;
            spreads[3 + axis] = self.velocity;
            spreads[9 + axis] = self.turn_rate_offset;
            spreads[12 + axis] = self.push_offset;
        }
        // The facing is not equally known about all three axes: the push sensor pins down which way
        // is up far more tightly than the compass pins down which way is north.
        spreads[6] = self.tilt;
        spreads[7] = self.tilt;
        spreads[8] = self.heading;
        spreads.map(|spread| spread * spread)
    }
}

/// Where the satellite fix says the body is.
struct SatelliteFix;

impl NominalStateFn<3> for SatelliteFix {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.position().as_array()
    }
}

/// How fast the satellite receiver says the body is going.
struct SatelliteVelocity;

impl NominalStateFn<3> for SatelliteVelocity {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.velocity().as_array()
    }
}

/// How far the ground would be down the beam, for a body where the filter believes it is and
/// leaning the way the filter believes it leans.
///
/// The beam points straight down the body's own axis, so a leaning body reads long by exactly as
/// much as its own up direction has tipped away from the world's. Working the reading out this way
/// round, rather than levelling it first and handing over a height, is what lets the filter see
/// that a beam reading disagreeing with it might be its attitude that is wrong and not its height.
struct RangeDownTheBeam;

impl NominalStateFn<1> for RangeDownTheBeam {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 1] {
        let up = state
            .orientation()
            .act(Vector::new([S::ZERO, S::ZERO, S::ONE]));
        [state.position()[2] / up[2]]
    }
}

/// Which way the compass says the nose points, taken in the level plane.
struct CompassHeading;

impl NominalStateFn<1> for CompassHeading {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 1] {
        [level_heading(state.orientation())]
    }
}

/// The notches that make the inertial reading usable, and the plainest possible estimate built on
/// what comes out of them.
pub struct FlightEstimator {
    timestep: f64,
    sections: [BiquadCoefficients<f64>; NOTCH_SECTIONS],
    angular_rate_notches: [MultiChannelBiquad<3, f64>; NOTCH_SECTIONS],
    acceleration_notches: [MultiChannelBiquad<3, f64>; NOTCH_SECTIONS],
    notched: InertialReading3d,
    dead_reckoned: EstimatedState,
    filter: ErrorStateKalmanFilter<3, f64>,
    gravity: Vector3D<f64>,
    fixes_offered: u64,
    fixes_thrown_away: u64,
    refusals_in_a_row: u32,
    filter_faults: u64,
}

impl FlightEstimator {
    /// Builds the notches for a machine whose rotors turn at `rotor_tone_hertz`, read every
    /// `timestep` seconds.
    ///
    /// `imu_noise` is what the filter is told the inertial unit does between one correction and the
    /// next, and `satellite_spread` is how far each part of a fix wanders. The filter is declared
    /// three wide because the fix is the widest reading it takes, and because the check that throws
    /// a bad fix away needs the spread of the reading it is checking, which the filter only keeps at
    /// its own declared width.
    ///
    /// Returns whatever the notch weights refuse on — which is what happens if the rotor rate is
    /// too close to how often the unit is read for the stack to mean anything.
    pub fn new(
        rotor_tone_hertz: f64,
        timestep: f64,
        imu_noise: ImuNoise<f64>,
        satellite_spread: Vector3D<f64>,
        starting_spreads: StartingSpreads,
    ) -> Result<Self, SignalError> {
        let sections = harmonic_notch_coefficients::<NOTCH_SECTIONS, f64>(
            rotor_tone_hertz,
            NOTCH_QUALITY,
            timestep,
        )?;
        let gravity = Vector::new([0.0, 0.0, -GRAVITY_STRENGTH]);
        let starting_covariance = Matrix::from_diagonal(starting_spreads.as_diagonal());
        let filter = ErrorStateKalmanFilter::<3, f64>::new(
            NominalState::at_rest(SO3::identity()),
            starting_covariance,
            imu_noise,
            Matrix::from_diagonal([
                satellite_spread[0] * satellite_spread[0],
                satellite_spread[1] * satellite_spread[1],
                satellite_spread[2] * satellite_spread[2],
            ]),
        )
        .with_gravity(gravity);

        Ok(FlightEstimator {
            timestep,
            sections,
            angular_rate_notches: sections.map(MultiChannelBiquad::new),
            acceleration_notches: sections.map(MultiChannelBiquad::new),
            notched: InertialReading3d::new(Vector::zeros(), Vector::zeros()),
            dead_reckoned: EstimatedState {
                position: Vector::zeros(),
                velocity: Vector::zeros(),
                orientation: SO3::identity(),
                angular_rate: Vector::zeros(),
            },
            filter,
            gravity,
            fixes_offered: 0,
            fixes_thrown_away: 0,
            refusals_in_a_row: 0,
            filter_faults: 0,
        })
    }

    /// Starts both estimates off, with the notches already sitting where a long run of `reading`
    /// would have left them.
    ///
    /// Settling the notches rather than starting them empty means the first reading of a flight is
    /// not mistaken for a sudden change, which would otherwise ring through the first few
    /// hundredths of a second.
    ///
    /// The two estimates start from different things, deliberately. Dead reckoning is handed the
    /// truth in `state`, because the only interesting thing about it is how quickly it leaves. The
    /// filter is handed `pad`, which is where the machine was taken to have been set down and is
    /// out by about as much as the filter is about to claim it is. It works its own facing out from
    /// the two things it can see there: a body that is not moving feels a push straight up, and the
    /// compass says which way north is. Two directions settle a facing outright. The push is the
    /// primary of the pair, because whichever direction goes in second only settles the spin left
    /// over about the first, and that is the place for the noisier reading.
    ///
    /// No amount of standing still takes the push sensor's own offset out of that, so the facing
    /// comes out something like a fiftieth of a radian off level. That is what a real machine starts
    /// from, and working the offset out from there is the filter's job.
    pub fn start_from(
        &mut self,
        state: FreeJointState<f64>,
        pad: StartingBelief,
        reading: InertialReading3d,
        heading: f64,
    ) {
        for section in &mut self.angular_rate_notches {
            section.settle_to(reading.angular_rate);
        }
        for section in &mut self.acceleration_notches {
            section.settle_to(reading.proper_acceleration);
        }
        self.notched = reading;
        self.dead_reckoned = EstimatedState::from_truth(state);

        let up_in_the_world = Vector::new([0.0, 0.0, 1.0]);
        let north_in_the_world = Vector::new([1.0, 0.0, 0.0]);
        let north_in_the_body = Vector::new([heading.cos(), -heading.sin(), 0.0]);
        let facing = SO3::from_two_direction_pairs(
            reading.proper_acceleration,
            north_in_the_body,
            up_in_the_world,
            north_in_the_world,
        )
        // A push reading pointing along north leaves the spin unsettled and there is no facing to
        // be had from the pair. What the compass says on its own is still better than nothing.
        .unwrap_or_else(|| SO3::exp(Vector::new([0.0, 0.0, heading])));

        self.filter.set_nominal_state(NominalState::new(
            pad.position,
            pad.velocity,
            facing,
            Vector::zeros(),
            Vector::zeros(),
        ));
        self.fixes_offered = 0;
        self.fixes_thrown_away = 0;
        self.refusals_in_a_row = 0;
        self.filter_faults = 0;
    }

    /// Moves the notches onto `rotor_tone_hertz`, keeping what they remember of the last few
    /// readings.
    ///
    /// The rotors come round faster when they are pushing harder, and the shake they put into the
    /// unit moves with them. A notch left sitting where the rotors were at a hover therefore stops
    /// being on the shake the moment the machine climbs or leans, and what it lets through is far
    /// worse than the jitter it was there to beat. Following the rate is the entire reason a notch
    /// stack is built on one rather than on a frequency written down once.
    ///
    /// The rate handed in is worked out from what the rotors were last asked for, not from what
    /// they are really doing — that is all a machine has, and the gap between the two is the rotors
    /// still spinning up.
    ///
    /// A rate the notches cannot be placed on leaves them where they are, which is the nearest
    /// placement that still means something.
    pub fn track_rotor_tone(&mut self, rotor_tone_hertz: f64) {
        let Ok(sections) = harmonic_notch_coefficients::<NOTCH_SECTIONS, f64>(
            rotor_tone_hertz,
            NOTCH_QUALITY,
            self.timestep,
        ) else {
            return;
        };
        self.sections = sections;
        for (notch, section) in self.angular_rate_notches.iter_mut().zip(sections) {
            notch.set_coefficients(section);
        }
        for (notch, section) in self.acceleration_notches.iter_mut().zip(sections) {
            notch.set_coefficients(section);
        }
    }

    /// Runs one reading through the notches alone, touching neither estimate.
    ///
    /// This is for the moment before anything flies. A notch fed a shake that appears out of
    /// nothing in a single tick passes most of that first shake straight through and only cancels
    /// it once it has seen a cycle or two — and a control loop handed that reads it as the body
    /// suddenly turning at most of a radian a second, which is enough to slam the rotors onto their
    /// limits. A real machine is armed, its rotors come up, and its reading is watched for a moment
    /// before anything acts on it. This is that moment.
    pub fn settle_notches(&mut self, reading: InertialReading3d) {
        let mut angular_rate = reading.angular_rate;
        for section in &mut self.angular_rate_notches {
            angular_rate = section.filter(angular_rate);
        }
        let mut proper_acceleration = reading.proper_acceleration;
        for section in &mut self.acceleration_notches {
            proper_acceleration = section.filter(proper_acceleration);
        }
        self.notched = InertialReading3d::new(angular_rate, proper_acceleration);
    }

    /// What the filter believes about the body's motion.
    ///
    /// The turn rate is the reading with the offset the filter has worked out taken back off it,
    /// which is the one part of the answer that comes straight from the unit rather than from the
    /// filter's own carrying-forward.
    #[must_use]
    pub fn believed(&self) -> EstimatedState {
        let state = self.filter.nominal_state();
        EstimatedState {
            position: state.position(),
            velocity: state.velocity(),
            orientation: state.orientation(),
            angular_rate: self.notched.angular_rate - state.gyroscope_bias(),
        }
    }

    /// How far the filter's answer sits from `truth`, measured against the spread the filter itself
    /// claims. Nothing at all if that spread cannot be factorized.
    ///
    /// Only a simulation can ask this, because only a simulation has the truth. Over a long flight
    /// it should sit near fifteen, one for each number the filter carries: much larger and the
    /// filter is more wrong than it admits, much smaller and it is being needlessly unsure.
    #[must_use]
    pub fn consistency_against(&self, truth: NominalState<f64>) -> Option<f64> {
        self.filter.normalized_estimation_error_squared(truth).ok()
    }

    /// How many satellite fixes have been offered, and how many of those were thrown away.
    #[inline]
    #[must_use]
    pub fn fixes_offered(&self) -> u64 {
        self.fixes_offered
    }

    /// How many fixes the filter did not believe enough to fold in.
    #[inline]
    #[must_use]
    pub fn fixes_thrown_away(&self) -> u64 {
        self.fixes_thrown_away
    }

    /// How many times the filter refused a step outright and the estimate was held instead.
    #[inline]
    #[must_use]
    pub fn filter_faults(&self) -> u64 {
        self.filter_faults
    }

    /// Offers the filter a satellite fix, and says whether it was taken.
    ///
    /// A fix is checked before it is believed: how far it sits from where the filter expected it,
    /// measured against how far the filter thought it might sit, has to come in under
    /// [`WORST_BELIEVABLE_FIX`]. The check is run by folding the fix into a copy of the filter and
    /// asking the copy — the number that answers it is formed from the spread as it stood before
    /// the fix, so the copy has it and the original is untouched if the answer is no.
    pub fn fold_in_satellite_fix(&mut self, fix: Vector3D<f64>) -> bool {
        self.fixes_offered += 1;
        let mut tried = self.filter;
        if tried.update(&SatelliteFix, fix).is_err() {
            self.filter_faults += 1;
            return false;
        }
        let believable = tried
            .normalized_innovation_squared()
            .is_ok_and(|gap| gap <= WORST_BELIEVABLE_FIX);
        let taken_anyway = self.refusals_in_a_row >= MOST_REFUSALS_IN_A_ROW;

        if believable || taken_anyway {
            self.filter = tried;
            self.refusals_in_a_row = 0;
            true
        } else {
            self.refusals_in_a_row += 1;
            self.fixes_thrown_away += 1;
            false
        }
    }

    /// Folds in how fast the receiver says the body is going, with `spread` for how far each part
    /// of that wanders.
    ///
    /// This is the reading that steadies the speed. Between fixes the only thing the machine has to
    /// say how fast it is going is its own push reading, integrated — and the small errors in that
    /// pile up into exactly the wander that the position loop then turns into a push nothing asked
    /// for. A direct reading of the speed, even a rough one, cuts that off at the source.
    pub fn fold_in_satellite_velocity(&mut self, velocity: Vector3D<f64>, spread: Vector3D<f64>) {
        let predicted = Vector::new(SatelliteVelocity.eval(&self.filter.nominal_state()));
        if self
            .filter
            .update_other::<3, _>(
                &SatelliteVelocity,
                velocity - predicted,
                Matrix::from_diagonal([
                    spread[0] * spread[0],
                    spread[1] * spread[1],
                    spread[2] * spread[2],
                ]),
            )
            .is_err()
        {
            self.filter_faults += 1;
        }
    }

    /// Folds in how far the downward beam says the ground is, with `spread` for how far that
    /// wanders.
    ///
    /// The ground is taken as flat and at nothing, so how high the body is above it is the height
    /// in world axes outright — but the beam does not report that height. It reports the slanted
    /// distance it measured, and what stands the two apart is the lean, which is the filter's own
    /// belief and comes with the filter's own error on it. A leaning body that has its lean wrong
    /// therefore gets its height wrong too, which is the honest arrangement and the reason the
    /// reading is folded in as a range rather than as a height.
    ///
    /// Nothing is folded in at all past [`WORST_LEAN_FOR_THE_BEAM`], where a small error in the
    /// believed lean turns into a large one in the height.
    pub fn fold_in_beam_range(&mut self, range: f64, spread: f64) {
        let state = self.filter.nominal_state();
        let up: Vector3D<f64> = state.orientation().act(Vector::new([0.0, 0.0, 1.0]));
        if up[2] < WORST_LEAN_FOR_THE_BEAM.cos() {
            return;
        }
        let predicted = RangeDownTheBeam.eval(&state);
        let residual = Vector::new([range - predicted[0]]);
        if self
            .filter
            .update_other::<1, _>(
                &RangeDownTheBeam,
                residual,
                Matrix::from_diagonal([spread * spread]),
            )
            .is_err()
        {
            self.filter_faults += 1;
        }
    }

    /// Folds in which way the compass says the nose points, with `spread` for how far that wanders.
    ///
    /// The difference between two headings is taken the short way round rather than by subtracting:
    /// a reading just the other side of the half turn is a small disagreement, and subtracting makes
    /// it look like most of a full one.
    pub fn fold_in_heading(&mut self, heading: f64, spread: f64) {
        let predicted = Vector::new(CompassHeading.eval(&self.filter.nominal_state()));
        let residual = residual_with_wrapped_angles(Vector::new([heading]), predicted, &[0]);
        if self
            .filter
            .update_other::<1, _>(
                &CompassHeading,
                residual,
                Matrix::from_diagonal([spread * spread]),
            )
            .is_err()
        {
            self.filter_faults += 1;
        }
    }

    /// How far behind the truth the notches put something happening at `frequency_hertz`, in
    /// seconds.
    ///
    /// A notch is a trade: it takes the shake out, and what it charges is a delay on everything
    /// else. This is that charge, worked out from the notches themselves rather than from a
    /// recording, and it is the half of the trade that a control loop feels.
    #[must_use]
    pub fn notch_delay_at(&self, frequency_hertz: f64) -> f64 {
        self.sections
            .iter()
            .map(|section| section.delay_at(frequency_hertz))
            .sum()
    }

    /// The reading with the shake taken out. This is what anything that flies is allowed to see.
    #[inline]
    #[must_use]
    pub fn notched(&self) -> InertialReading3d {
        self.notched
    }

    /// What comes of integrating that reading and nothing else.
    #[inline]
    #[must_use]
    pub fn dead_reckoned(&self) -> EstimatedState {
        self.dead_reckoned
    }

    /// Takes one reading in: cleans it up, then carries the estimate forward by one tick.
    pub fn update(&mut self, reading: InertialReading3d, timestep: f64) {
        let mut angular_rate = reading.angular_rate;
        for section in &mut self.angular_rate_notches {
            angular_rate = section.filter(angular_rate);
        }
        let mut proper_acceleration = reading.proper_acceleration;
        for section in &mut self.acceleration_notches {
            proper_acceleration = section.filter(proper_acceleration);
        }
        self.notched = InertialReading3d::new(angular_rate, proper_acceleration);

        // The filter carries the same reading forward, with a spread beside it and the unit's own
        // steady offsets taken off as it goes. A step it refuses holds the estimate where it is.
        if self
            .filter
            .predict(angular_rate, proper_acceleration, timestep)
            .is_err()
        {
            self.filter_faults += 1;
        }

        // The push is read in the body's own axes, so which way the body is facing is what turns it
        // into a push on the world. Gravity is what the unit cannot feel, and so is the one thing
        // that has to be added back by hand.
        let facing = self.dead_reckoned.orientation;
        let acceleration = facing.act(proper_acceleration) + self.gravity;
        let velocity = self.dead_reckoned.velocity;

        self.dead_reckoned = EstimatedState {
            position: self.dead_reckoned.position
                + velocity * timestep
                + acceleration * (0.5 * timestep * timestep),
            velocity: velocity + acceleration * timestep,
            orientation: ExponentialMap::attitude_step(facing, angular_rate, timestep),
            angular_rate,
        };
    }
}
