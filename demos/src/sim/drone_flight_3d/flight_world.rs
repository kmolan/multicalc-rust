//! The spine of the demo: one tick in, one record out. Only the flight stack is timed, so the cost
//! it reports is the library's work and not the simulation's.

use std::error::Error;
use std::time::Instant;

use multicalc::error::SignalError;
use multicalc::estimation::{ImuNoise, MonteCarloLocalizer, NominalState};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::spatial::{FreeJointState, SE3, SO3, Twist};
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::sim::height_rangefinder::HeightRangefinder;
use crate::sim::inertial_measurement_unit_3d::{InertialMeasurementUnit3d, InertialReading3d};
use crate::sim::magnetic_compass::MagneticCompass;
use crate::sim::satellite_navigation_sensor::SatelliteNavigationSensor;
use crate::sim::sensor_noise::gaussian_noise;

use super::flight_controller::{FlightCommand, FlightController};
use super::flight_estimator::{
    EstimatedState, FlightEstimator, StartingBelief, StartingSpreads, StateSource,
};
use super::flight_hangar::{FlightHangar, flight_hangar};
use super::flight_localization::{LIDAR_BEAMS, PAD_HEIGHT, SCAN_PERIOD_TICKS, StartupLocalization};
use super::flight_plant::{FlightPlant, angle_from_upright, level_heading};
use super::flight_reference::{FlightReference, ReferenceSample};
use super::x2_model::{GRAVITY_STRENGTH, ROTOR_COUNT, ROTOR_TONE_HERTZ, X2Model};

/// How long one tick lasts, in seconds.
///
/// The only place the tick is written down. Everything that needs to know it — the plant, both
/// loops, the notches, the filter — is handed this one, so changing it changes all of them together
/// rather than leaving one of them designed against a length nothing runs at.
pub const TIMESTEP: f64 = 0.001;

/// How many ticks at the start of a run are left out of the timing statistics, so a cold start does
/// not colour them.
pub const WARMUP_TICKS: u64 = 500;

/// How long the machine sits with its rotors turning, being read, before anything acts on the
/// reading, in ticks.
///
/// This is the arming: a tenth of a second is several times what the notches take to learn the
/// shake, and until they have, what comes out of them carries most of that shake straight through.
const ARMED_TICKS: u64 = 100;

/// The point the body is asked to hold.
pub const HOVER_POINT: [f64; 3] = [0.0, 0.0, 1.5];

/// The run the demo flies.
///
/// One seed settles every draw a flight makes — the offsets the unit is switched on with, every
/// reading it gives afterwards, where each satellite fix lands, and where the guesses about the room
/// are scattered. Fixing it means what is watched is the same flight every time, so something that
/// looks wrong can be looked at twice.
pub const SEED: u64 = 20260807;

// Where the numbers below come from.
//
// The model file describes an airframe and says nothing whatever about what is bolted to it, so
// unlike the numbers in `x2_model` not one of these can be read off it and none of them cites a
// line of anything. They are three different kinds of number, and knowing which kind one is matters
// more than its value before changing it:
//
// - **Descriptions of a part.** Every jitter, offset, wander, shake, spread and rate from here down
//   to the compass. Each is the size or speed a part of that class carries, and each one's own doc
//   says what makes it that size rather than another. Moving one is claiming a different part, and
//   the demo should be told so in the doc beside it.
// - **What the filter is told about all that**, which is not the same thing:
//   `OFFSET_WANDER_ALLOWANCE`, `FACING_ERROR_ALLOWED_FOR` and `SATELLITE_WANDER_HOLDS_OVER_SECONDS`.
//   These describe nothing. They are how a fault whose shape the filter cannot be told about is
//   declared to it anyway, each argued for where it is defined. Whether the argument was right is
//   what a consistency check answers, and nothing measures that any more.
// - **Staging.** `SEED`, `HOVER_POINT`, `WARMUP_TICKS`, `ARMED_TICKS`, `RUNNING_THRUST_MARGIN`.
//   Where the demo puts the machine and how it starts it, chosen to make the thing legible and
//   measurable.
//
// What none of them is: a number turned until a gate went green. If one ever becomes that, say so
// where it is defined — that is the single most useful thing a later reader can be told.

/// What the inertial unit does to a reading.
///
/// The jitter is a single reading's wander about the truth, the wander is how far the steady offset
/// drifts over a second of flight, and the shake is how hard the machine rattles the unit at the
/// rotors' own turning rate. The offset a unit is switched on with is drawn once at that same size.
pub const TURN_RATE_JITTER: f64 = 0.02;
pub const PUSH_JITTER: f64 = 0.05;
const TURN_RATE_OFFSET_WANDER: f64 = 0.001;
const PUSH_OFFSET_WANDER: f64 = 0.01;
const TURN_RATE_STARTING_OFFSET: f64 = 0.002;
const PUSH_STARTING_OFFSET: f64 = 0.02;
const TURN_RATE_SHAKE: f64 = 0.8;
const PUSH_SHAKE: f64 = 2.0;
const SECOND_HARMONIC_SHARE: f64 = 0.4;

/// How far above its floor a rotor has to be pushing before the machine counts as running, and
/// therefore as shaking, in newtons.
const RUNNING_THRUST_MARGIN: f64 = 1e-6;

/// What the sensors that aid the estimate do, and how often each one answers.
///
/// The receiver is worse up and down than across the ground, which is why the beam is on the
/// machine at all: it holds the one channel the satellites are weakest on, twelve times more finely
/// and twice as often. The compass holds the one thing neither of the others can say anything
/// about, which is which way the nose is pointing.
///
/// The receiver's speed answer is a great deal better than its position answer — five metres of
/// position error would be a broken receiver, five centimetres a second of speed error is ordinary
/// — because the two are measured by different means entirely. That better half is what keeps the
/// speed steady between fixes.
const SATELLITE_HORIZONTAL_NOISE: f64 = 0.25;
const SATELLITE_VERTICAL_NOISE: f64 = 0.5;
const SATELLITE_HORIZONTAL_SPEED_NOISE: f64 = 0.05;
const SATELLITE_VERTICAL_SPEED_NOISE: f64 = 0.10;
pub const SATELLITE_PERIOD_TICKS: u64 = 50;

/// How much of the receiver's position error drifts slowly instead of being drawn afresh in every
/// fix, as a share of the spread, and how long that drift takes to forget where it was.
///
/// The receiver is no worse for this — the two halves together come to the same spread it declares.
/// What it takes away is the ability to average the error down by asking often, which is the more
/// flattering of the two arrangements and not the one a real machine gets: half a metre of it comes
/// from the path the signals took, and that path is the same on the next fix as it was on this one.
const SATELLITE_WANDERING_SHARE: f64 = 0.5;
const SATELLITE_WANDER_SETTLING_SECONDS: f64 = 60.0;

/// Over how long a fix error that lasts does its damage, in seconds.
///
/// The filter is told a fix is drawn afresh every time, because that is the only shape it can be
/// told. Half of this one is not: it holds where it is for a minute, so the fix a second from now
/// carries the same error as this one and averaging the two takes nothing out. Left told as fresh
/// noise, the filter would grow surer of where it is with every fix that agreed with the last for
/// the wrong reason. So the drifting half is widened by the square root of how many fixes arrive
/// while it holds still — the same allowance the push reading gets for the gravity a facing that is
/// slightly out leaks. A second is how long the filter takes to grow sure of a position, and so how
/// long an error has to last to be believed.
const SATELLITE_WANDER_HOLDS_OVER_SECONDS: f64 = 1.0;

/// How often a fix comes back plainly wrong, and the spread the wrong one is drawn from.
///
/// About one a lap. Without it the check that throws a fix away has nothing to throw away but the
/// tails of its own noise, which is a machine defending itself against nothing.
const SATELLITE_OUTLIER_CHANCE: f64 = 0.002;
const SATELLITE_OUTLIER_SIZE: f64 = 5.0;
const RANGEFINDER_NOISE: f64 = 0.02;
const RANGEFINDER_MAXIMUM_RANGE: f64 = 6.0;
const RANGEFINDER_PERIOD_TICKS: u64 = 20;
const COMPASS_NOISE: f64 = 0.052;
const COMPASS_PERIOD_TICKS: u64 = 100;

/// How much faster than the truth the filter is told the steady offsets wander.
const OFFSET_WANDER_ALLOWANCE: f64 = 1.0;

/// How well the machine knows where it is and how it is moving when it is switched on.
///
/// It is placed rather than flown to its starting point, so both are known well — but not exactly,
/// and saying exactly would leave the filter refusing every reading that disagreed with it.
const PAD_POSITION_SPREAD: f64 = 0.05;
const PAD_VELOCITY_SPREAD: f64 = 0.05;

/// How far the facing may be out and still be counted as level, in radians.
///
/// This is not a guess about the machine but a statement about what the filter has to allow for.
/// A facing that is out by this much leaks that much of gravity into the sideways channels, and
/// that leak is far larger than anything the push sensor itself does. Telling the filter only the
/// sensor's own jitter would have it growing sure of itself faster than it has earned, and then the
/// check on the fixes starts throwing away the very readings that would put it right.
const FACING_ERROR_ALLOWED_FOR: f64 = 0.01;

/// What the machine is doing with itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightPhase {
    /// Hovering and turning on the spot, working out where in the room it is.
    FindingItself,
    /// Following the path it was given.
    Flying,
}

/// One tick: what the reference asked for, what the body did, and what it cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickRecord {
    /// Which tick this is, counting from one.
    pub tick: u64,
    /// How far into the flight it is, in seconds.
    pub time: f64,
    /// How far into the path it is, in seconds — what the reference was asked for.
    ///
    /// Not the same as `time`. The path's clock starts when the flying does, so a machine that
    /// spent three seconds working out where it was is three seconds into the flight and nothing at
    /// all into the path.
    pub reference_time: f64,
    /// Where the body is and which way it faces.
    pub pose: SE3<f64>,
    /// How fast it is moving and turning.
    pub velocity: Twist<f64>,
    /// What the reference asked for.
    pub reference: ReferenceSample,
    /// What the controller decided.
    pub command: FlightCommand,
    /// What each rotor actually pushed with over the tick, in newtons.
    pub delivered_thrusts: Vector<ROTOR_COUNT, f64>,
    /// What the body really did, as the inertial unit would report it if it were perfect.
    pub truth_inertial: InertialReading3d,
    /// What the unit actually reported.
    pub raw_inertial: InertialReading3d,
    /// The same reading with the shake taken out — the only one anything that flies may see.
    pub notched_inertial: InertialReading3d,
    /// The steady offsets the unit was carrying, which nothing flying is allowed to look at.
    pub inertial_offsets: InertialReading3d,
    /// How far round the rotors' shake had come when that reading was taken, in radians.
    ///
    /// Nothing flying may look at this either. It is here so a measurement can pick the shake out
    /// of a reading by matching it against the wave that put it there — which cannot be done by
    /// assuming a rate afterwards, because the rate moves with what the rotors are doing.
    pub shake_phase: f64,
    /// What the machine believes about its own motion, from the reading and nothing else.
    pub dead_reckoned: EstimatedState,
    /// What the filter believes about it, with everything it can see folded in.
    pub believed: EstimatedState,
    /// Whether a satellite fix was thrown away this tick because the filter did not believe it.
    pub fix_thrown_away: bool,
    /// What the machine is doing with itself.
    pub phase: FlightPhase,
    /// How far the furthest-behind rotor was from what it was asked for, in newtons.
    pub rotor_thrust_gap: f64,
    /// How far the body is from the point the reference asked for, in metres.
    pub position_error: f64,
    /// How far the body is leaning away from upright, in radians.
    pub tilt: f64,
    /// What the flight stack cost this tick, in microseconds.
    pub math_microseconds: f64,
}

/// Running totals over a whole flight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightMetrics {
    /// How many ticks have been flown.
    pub ticks: u64,
    /// How many of those had a rotor sitting on a limit.
    pub ticks_with_rotor_at_limit: u64,
    /// The worst lean away from upright seen, in radians.
    pub worst_tilt: f64,
    /// The worst distance from the reference seen, in metres.
    pub worst_position_error: f64,
    /// The worst a rotor has been behind what it was asked for, in newtons.
    pub worst_rotor_thrust_gap: f64,
    /// How many ticks held the previous command rather than acting on an impossible one.
    pub controller_holds: u64,
    /// How many ticks the adaptive step gave up on and the fixed step took over.
    pub fixed_step_fallbacks: u64,
    math_microseconds_total: f64,
    math_ticks: u64,
}

impl FlightMetrics {
    /// The share of ticks that had a rotor sitting on a limit.
    #[must_use]
    pub fn rotor_limit_fraction(&self) -> f64 {
        if self.ticks == 0 {
            0.0
        } else {
            self.ticks_with_rotor_at_limit as f64 / self.ticks as f64
        }
    }

    /// What the flight stack costs on an average tick, warm-up excluded.
    #[must_use]
    pub fn mean_math_microseconds(&self) -> f64 {
        if self.math_ticks == 0 {
            0.0
        } else {
            self.math_microseconds_total / self.math_ticks as f64
        }
    }
}

/// The whole demo: the machine, the truth, the loops that fly it, and where it is being asked to go.
pub struct FlightWorld {
    model: X2Model,
    plant: FlightPlant,
    controller: FlightController,
    reference: FlightReference,
    inertial: InertialMeasurementUnit3d,
    hangar: FlightHangar,
    localization: Option<StartupLocalization>,
    phase: FlightPhase,
    flight_reference: FlightReference,
    satellite: SatelliteNavigationSensor,
    rangefinder: HeightRangefinder,
    compass: MagneticCompass,
    estimator: FlightEstimator,
    last_commanded_thrusts: Vector<ROTOR_COUNT, f64>,
    state_source: StateSource,
    rng: Pcg32,
    start_offset: Vector3D<f64>,
    reference_started_at: f64,
    seed: u64,
    tick: u64,
    metrics: FlightMetrics,
}

impl FlightWorld {
    /// Builds the world with the body sitting still on the point it is asked to hold.
    ///
    /// `seed` settles every random draw in the run: the offsets the inertial unit is switched on
    /// with, and every reading it gives afterwards.
    ///
    /// Returns whatever the model file, the body, the rotor layout, or either loop refuses on.
    pub fn new(seed: u64) -> Result<Self, Box<dyn Error>> {
        let model = X2Model::load()?;
        let hover_thrust_per_rotor = model.hover_thrust_per_rotor();
        let hover_point = Vector::new(HOVER_POINT);
        // The wanted facing starts at the body's own, so the first tick asks for no turn at all.
        let controller = FlightController::new(&model, TIMESTEP, 0.0)?;
        let plant = FlightPlant::new(&model, FreeJointState::identity(), TIMESTEP)?;

        // What the filter is told the unit does between one correction and the next. The jitter
        // numbers are the unit's own; the push one is widened for the gravity a facing that is
        // slightly out leaks into the sideways channels, which is the larger of the two by a good
        // margin. The two are added in quadrature, being unrelated to each other.
        // A facing that is slightly out leaks gravity into the sideways channels, and unlike jitter
        // it leaks the same way for as long as the facing stays out — which is until the next fix
        // arrives. Told to the filter as if it were jitter it would count for almost nothing, since
        // jitter that changes every tick largely cancels itself over a hundred of them. Spreading a
        // leak that lasts the whole gap between fixes across those ticks is what this factor does.
        let leaked_gravity =
            GRAVITY_STRENGTH * FACING_ERROR_ALLOWED_FOR * (SATELLITE_PERIOD_TICKS as f64).sqrt();
        let allowed_push_jitter = PUSH_JITTER.hypot(leaked_gravity);
        let imu_noise = ImuNoise {
            gyroscope_noise_density: TURN_RATE_JITTER,
            accelerometer_noise_density: allowed_push_jitter,
            gyroscope_bias_random_walk: TURN_RATE_OFFSET_WANDER * OFFSET_WANDER_ALLOWANCE,
            accelerometer_bias_random_walk: PUSH_OFFSET_WANDER * OFFSET_WANDER_ALLOWANCE,
        };
        let satellite =
            SatelliteNavigationSensor::new(SATELLITE_HORIZONTAL_NOISE, SATELLITE_VERTICAL_NOISE)
                .with_speed_noise(
                    SATELLITE_HORIZONTAL_SPEED_NOISE,
                    SATELLITE_VERTICAL_SPEED_NOISE,
                )
                .with_wandering_share(SATELLITE_WANDERING_SHARE, SATELLITE_WANDER_SETTLING_SECONDS)
                .with_outliers(SATELLITE_OUTLIER_CHANCE, SATELLITE_OUTLIER_SIZE);
        let rangefinder = HeightRangefinder::new(RANGEFINDER_NOISE, RANGEFINDER_MAXIMUM_RANGE);
        let compass = MagneticCompass::new(COMPASS_NOISE);

        // What the filter is told a fix might be out by: the half of the receiver's spread that is
        // drawn afresh every time, and the half that drifts widened by how many fixes it holds
        // still across.
        let fixes_while_it_holds =
            SATELLITE_WANDER_HOLDS_OVER_SECONDS / (SATELLITE_PERIOD_TICKS as f64 * TIMESTEP);
        let allowed_fix_spread = |spread: f64| {
            let fresh = spread
                * (1.0 - SATELLITE_WANDERING_SHARE * SATELLITE_WANDERING_SHARE)
                    .max(0.0)
                    .sqrt();
            let drifting = spread * SATELLITE_WANDERING_SHARE * fixes_while_it_holds.sqrt();
            fresh.hypot(drifting)
        };
        let estimator = FlightEstimator::new(
            ROTOR_TONE_HERTZ,
            TIMESTEP,
            imu_noise,
            Vector::new([
                allowed_fix_spread(satellite.horizontal_noise()),
                allowed_fix_spread(satellite.horizontal_noise()),
                allowed_fix_spread(satellite.vertical_noise()),
            ]),
            StartingSpreads {
                position: PAD_POSITION_SPREAD,
                velocity: PAD_VELOCITY_SPREAD,
                // A push sensor reading a little off reads as a body leaning a little, and there is
                // nothing on a still machine that can tell the two apart.
                tilt: PUSH_STARTING_OFFSET / GRAVITY_STRENGTH,
                heading: COMPASS_NOISE,
                turn_rate_offset: TURN_RATE_STARTING_OFFSET,
                push_offset: PUSH_STARTING_OFFSET,
            },
        )?;

        // The unit is switched on already carrying an offset, drawn once from the run's own seed.
        let mut rng = Pcg32::seed_from_u64(seed);
        let starting_turn_rate_offset =
            Vector::from_fn(|_| gaussian_noise(TURN_RATE_STARTING_OFFSET, &mut rng));
        let starting_push_offset =
            Vector::from_fn(|_| gaussian_noise(PUSH_STARTING_OFFSET, &mut rng));
        let inertial = InertialMeasurementUnit3d::new(TURN_RATE_JITTER, PUSH_JITTER)
            .with_offset_wander(TURN_RATE_OFFSET_WANDER, PUSH_OFFSET_WANDER)
            .with_shake(TURN_RATE_SHAKE, PUSH_SHAKE, SECOND_HARMONIC_SHARE)
            .with_starting_offsets(starting_turn_rate_offset, starting_push_offset);

        let hangar = flight_hangar()?;
        let mut world = FlightWorld {
            model,
            plant,
            controller,
            reference: FlightReference::hover(hover_point),
            inertial,
            hangar,
            localization: None,
            phase: FlightPhase::Flying,
            flight_reference: FlightReference::hover(hover_point),
            satellite,
            rangefinder,
            compass,
            estimator,
            // Before anything has been commanded, the rotors are turning at the rate that holds the
            // machine up, which is where the notches start out placed.
            last_commanded_thrusts: Vector::from_fn(|_| hover_thrust_per_rotor),
            state_source: StateSource::Truth,
            rng,
            start_offset: Vector::zeros(),
            reference_started_at: 0.0,
            seed,
            tick: 0,
            metrics: FlightMetrics {
                ticks: 0,
                ticks_with_rotor_at_limit: 0,
                worst_tilt: 0.0,
                worst_position_error: 0.0,
                worst_rotor_thrust_gap: 0.0,
                controller_holds: 0,
                fixed_step_fallbacks: 0,
                math_microseconds_total: 0.0,
                math_ticks: 0,
            },
        };
        world.place_on_the_reference();
        Ok(world)
    }

    /// Moves the body's starting point by `offset` from where the reference asks it to be.
    ///
    /// Starting away from the reference is how the climb is checked: the body has to fly to the
    /// point and settle on it, rather than being handed it.
    #[must_use]
    pub fn with_start_offset(mut self, offset: Vector3D<f64>) -> Self {
        self.start_offset = offset;
        self.place_on_the_reference();
        self
    }

    /// Asks the body to follow `reference` instead of holding the point it started on.
    #[must_use]
    pub fn with_reference(mut self, reference: FlightReference) -> Self {
        self.reference = reference;
        self.place_on_the_reference();
        self
    }

    /// Has the machine work out where in the room it is before it flies anywhere.
    ///
    /// It hovers on the spot and turns slowly while a cloud of guesses is scored against the floor
    /// plan, and only once the cloud has settled on one place does it start following the path it
    /// was given. Call this last, after the path is set, because the path is what it holds onto
    /// while it looks around.
    ///
    /// The turning is not decoration. The guesses are moved by how far the machine believes it has
    /// travelled and turned, which is a model that only holds while it hovers and spins — a banking
    /// machine slides sideways and breaks it outright — so this happens before the flight and never
    /// during it.
    /// Returns whatever the cloud of guesses refuses to be scattered as.
    pub fn with_startup_localization(mut self) -> Result<Self, Box<dyn Error>> {
        self.flight_reference = self.reference;
        let start = self.reference.sample(0.0).position();
        let pad = Vector::new([start[0], start[1], PAD_HEIGHT]);
        self.reference = FlightReference::hover(pad);
        self.phase = FlightPhase::FindingItself;
        self.place_on_the_reference();

        // The search starts from where the machine was set down, not from where it turns out to be.
        let facing = level_heading(self.plant.state().pose().rotation());
        self.localization = Some(StartupLocalization::new(pad, facing, self.seed)?);
        Ok(self)
    }

    /// What the machine is doing with itself.
    #[inline]
    #[must_use]
    pub fn phase(&self) -> FlightPhase {
        self.phase
    }

    /// The room, for drawing and for casting beams against.
    #[inline]
    #[must_use]
    pub fn hangar(&self) -> &FlightHangar {
        &self.hangar
    }

    /// The cloud of guesses about where in the room the machine is, while it is still looking.
    #[inline]
    #[must_use]
    pub fn localizer(&self) -> Option<&MonteCarloLocalizer<LIDAR_BEAMS>> {
        self.localization.as_ref().map(StartupLocalization::guesses)
    }

    /// How many scans were matched against the floor plan.
    #[inline]
    #[must_use]
    pub fn scans_matched(&self) -> u64 {
        self.localization
            .as_ref()
            .map_or(0, StartupLocalization::scans_matched)
    }

    /// How many scans were thrown away, most of them for being taken at too much of a lean.
    #[inline]
    #[must_use]
    pub fn scans_refused(&self) -> u64 {
        self.localization
            .as_ref()
            .map_or(0, StartupLocalization::scans_refused)
    }

    /// Chooses what the controller is told about where the body is.
    ///
    /// The gap between one source and another, flown otherwise identically, is what that source
    /// costs the flying — and it is the only way to tell a control problem from an estimate that
    /// was never good enough to fly on.
    #[must_use]
    pub fn with_state_source(mut self, state_source: StateSource) -> Self {
        self.state_source = state_source;
        self
    }

    /// Where the controller's idea of the body is coming from.
    #[inline]
    #[must_use]
    pub fn state_source(&self) -> StateSource {
        self.state_source
    }

    /// What the machine believes about its own motion, from the reading and nothing else.
    #[inline]
    #[must_use]
    pub fn dead_reckoned(&self) -> EstimatedState {
        self.estimator.dead_reckoned()
    }

    /// What the filter believes about the body's motion.
    #[inline]
    #[must_use]
    pub fn believed(&self) -> EstimatedState {
        self.estimator.believed()
    }

    /// How far the filter's answer sits from the truth, measured against the spread it claims for
    /// itself. Nothing at all if that spread cannot be factorized.
    ///
    /// The truth handed in is the whole of it, the unit's own steady offsets included, because
    /// those are two of the five things the filter is carrying and leaving them out would ask the
    /// check a different question.
    ///
    /// Ask it about a second apart rather than every tick, and pool the answers across several
    /// runs. The reason is in [`FlightEstimator::consistency_against`], and it is the difference
    /// between a number that reads the tuning and a number that reads nothing at all.
    #[must_use]
    pub fn filter_consistency(&self) -> Option<f64> {
        let state = self.plant.state();
        let offsets = self.inertial.offsets();
        self.estimator.consistency_against(NominalState::new(
            state.pose().translation(),
            state.velocity().linear(),
            state.pose().rotation(),
            offsets.angular_rate,
            offsets.proper_acceleration,
        ))
    }

    /// How many satellite fixes have been offered so far.
    #[inline]
    #[must_use]
    pub fn fixes_offered(&self) -> u64 {
        self.estimator.fixes_offered()
    }

    /// How many of those the filter threw away.
    #[inline]
    #[must_use]
    pub fn fixes_thrown_away(&self) -> u64 {
        self.estimator.fixes_thrown_away()
    }

    /// How many times the filter refused a step outright and the estimate was held instead.
    #[inline]
    #[must_use]
    pub fn filter_faults(&self) -> u64 {
        self.estimator.filter_faults()
    }

    /// How far behind the truth the notches put something happening at `frequency_hertz`, in
    /// seconds.
    ///
    /// Returns the same errors as
    /// [`FlightEstimator::notch_delay_at`](super::flight_estimator::FlightEstimator::notch_delay_at).
    #[inline]
    pub fn notch_delay_at(&self, frequency_hertz: f64) -> Result<f64, SignalError> {
        self.estimator.notch_delay_at(frequency_hertz)
    }

    /// Puts the body where the reference starts, moving the way it starts out moving and facing
    /// the way it starts out going, plus whatever starting offset was asked for.
    ///
    /// Handing the body the reference's own motion at the start is what makes a moving reference
    /// measurable: begun from a standstill, the first seconds are the body catching up rather than
    /// following, and that is a different thing to measure. A reference that holds one point hands
    /// over nothing, so this leaves the body still and level.
    fn place_on_the_reference(&mut self) {
        let start = self.reference.sample(0.0);
        let heading = start.travel_heading().unwrap_or(0.0);
        let orientation = SO3::exp(Vector::new([0.0, 0.0, heading]));
        let pose = SE3::from_parts(orientation, start.position() + self.start_offset);
        let state = FreeJointState::new(pose, Twist::new(start.velocity(), Vector::zeros()));
        self.plant.reset(state);
        self.controller.set_desired_heading(heading);
        // On the pad: the position and the speed are surveyed rather than measured, so each is
        // handed over with a draw from the very spread the filter is told to claim about it. The
        // facing is worked out from the push the body feels and what the compass says, both read
        // once with their own noise on them.
        let pad = StartingBelief {
            position: state.pose().translation()
                + Vector::from_fn(|_| gaussian_noise(PAD_POSITION_SPREAD, &mut self.rng)),
            velocity: state.velocity().linear()
                + Vector::from_fn(|_| gaussian_noise(PAD_VELOCITY_SPREAD, &mut self.rng)),
        };
        let reading = self.truth_reading(state);
        let heading = self
            .compass
            .read(level_heading(state.pose().rotation()), &mut self.rng);
        self.estimator.start_from(state, pad, reading, heading);

        // The rotors are already turning when the machine is placed, so the unit is already
        // shaking. The notches are watched through a moment of that before anything acts on what
        // comes out of them, because a notch handed a shake that appears from nothing passes the
        // first of it straight through.
        for _ in 0..ARMED_TICKS {
            let shaking = self.inertial.read(
                reading.angular_rate,
                reading.proper_acceleration,
                TIMESTEP,
                Some(ROTOR_TONE_HERTZ),
                &mut self.rng,
            );
            self.estimator.settle_notches(shaking);
        }
    }

    /// Hovers, turns on the spot, and lets the search score one scan when a scan is due.
    ///
    /// Once the guesses have agreed on one place — or once the search has had long enough and
    /// clearly is not going to — the machine stops turning and starts following the path it was
    /// given.
    fn look_around(&mut self) {
        let truth = self.plant.state();
        let believed = self.estimator.believed();
        let tick = self.tick;

        let Some(search) = self.localization.as_mut() else {
            return;
        };
        let wanted_heading = search.turn_on_the_spot(TIMESTEP);
        if tick.is_multiple_of(SCAN_PERIOD_TICKS) {
            search.match_one_scan(&self.hangar, truth, believed, &mut self.rng);
        }
        let finished = search.is_finished(tick as f64 * TIMESTEP);

        self.controller.set_desired_heading(wanted_heading);
        if finished {
            self.phase = FlightPhase::Flying;
            self.reference = self.flight_reference;
            // The path starts when the flying does, not when the machine was switched on, so the
            // first thing asked for is the beginning of the path rather than wherever along it the
            // looking around happened to finish.
            self.reference_started_at = tick as f64 * TIMESTEP;
        }
    }

    /// How fast the rotors are really coming round at this moment, in turns a second, or nothing at
    /// all when they are not turning and the machine is therefore not shaking.
    ///
    /// This is the truth of it, read off what the rotors are actually delivering, and it is the
    /// rate the unit is shaken at. What the machine believes the rate to be is worked out from what
    /// it last asked the rotors for, which is a different number while they are still spinning up.
    fn rotor_tone(&self) -> Option<f64> {
        let floor = self.model.mixer().minimum_thrust() + RUNNING_THRUST_MARGIN;
        let delivered = self.plant.delivered_thrusts();
        if !delivered.as_array().iter().any(|thrust| *thrust > floor) {
            return None;
        }
        Some(rotor_tone_for(delivered, self.hovering_thrust()))
    }

    /// What the whole set of rotors pushes with while the machine merely holds itself up, in
    /// newtons.
    fn hovering_thrust(&self) -> f64 {
        self.model.hover_thrust_per_rotor() * ROTOR_COUNT as f64
    }

    /// What a perfect unit bolted at the body's origin would report about `state`.
    fn truth_reading(&self, state: FreeJointState<f64>) -> InertialReading3d {
        InertialReading3d::new(state.velocity().angular(), self.plant.proper_acceleration())
    }

    /// Where the body is being asked to go.
    #[inline]
    #[must_use]
    pub fn reference(&self) -> FlightReference {
        self.reference
    }

    /// The machine's own numbers.
    #[inline]
    #[must_use]
    pub fn model(&self) -> &X2Model {
        &self.model
    }

    /// The loops flying it.
    #[inline]
    #[must_use]
    pub fn controller(&self) -> &FlightController {
        &self.controller
    }

    /// Which run this is.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The running totals so far.
    #[inline]
    #[must_use]
    pub fn metrics(&self) -> FlightMetrics {
        self.metrics
    }

    /// Flies one millisecond.
    ///
    /// The order is load-bearing, and every part of it is a decision rather than an accident:
    ///
    /// 1. **Look around, if it is still doing that.** This moves the wanted facing and may hand the
    ///    flight over to the path, so it has to happen before the reference is asked for anything.
    /// 2. **Ask the reference where the body should be.** On the path's own clock, not the flight's.
    /// 3. **Read every sensor**, against the truth as the last tick left it. This is the demo's own
    ///    work rather than the machine's, so it sits outside the timed block — a reported cost with
    ///    the simulation's noise draws folded into it would be a cost of nothing in particular. Each
    ///    sensor answers on its own beat, and the compass is deliberately offered on a different
    ///    tick from the satellite so the two never land together.
    /// 4. **The timed block**: move the notches onto where the rotors are reckoned to be, carry the
    ///    estimate forward, fold in whatever answered this tick, and run both loops against whatever
    ///    the state source says. Nothing but the flight stack goes in here.
    /// 5. **Step the truth** under the command that was just decided — after it, because a body
    ///    moved before the command would be answering one that had not been issued yet.
    /// 6. **Fold the metrics** last, when both the command and what the body did with it are known.
    ///
    /// A later change that moves the truth up, or slips a sensor into the timed block, breaks
    /// something the numbers will not obviously show.
    pub fn step(&mut self) -> TickRecord {
        self.tick += 1;
        let time = self.tick as f64 * TIMESTEP;
        if self.phase == FlightPhase::FindingItself {
            self.look_around();
        }
        let reference_time = time - self.reference_started_at;
        let reference = self.reference.sample(reference_time);

        let state = self.plant.state();
        let pose = state.pose();
        let velocity = state.velocity();

        // Simulating the unit is the demo's work, not the machine's, so it sits outside the timed
        // block. The reading describes the truth as the last tick left it, which is what a unit
        // read at the top of a tick reports — and what comes back is the sample before that again,
        // because the unit answers a tick late.
        let truth_inertial = self.truth_reading(state);
        let inertial_offsets = self.inertial.offsets();
        let shake_phase = self.inertial.shake_phase();
        let hovering = self.hovering_thrust();
        let raw_inertial = self.inertial.read(
            truth_inertial.angular_rate,
            truth_inertial.proper_acceleration,
            TIMESTEP,
            self.rotor_tone(),
            &mut self.rng,
        );

        // The aiding sensors answer on their own beats, and each is simulated out here with the
        // rest of the demo's work. The compass is deliberately offered on a different tick from the
        // fix, so the two never land on the same one.
        let receiver_answered = self.tick.is_multiple_of(SATELLITE_PERIOD_TICKS);
        let satellite_fix = receiver_answered.then(|| {
            self.satellite.read_position(
                pose.translation(),
                SATELLITE_PERIOD_TICKS as f64 * TIMESTEP,
                &mut self.rng,
            )
        });
        let satellite_velocity = receiver_answered.then(|| {
            self.satellite
                .read_velocity(velocity.linear(), &mut self.rng)
        });
        // The beam hands back the slanted distance it measured and nothing else. Standing that up
        // into a height is the filter's job, and it has only its own belief about the lean to do it
        // with — the true lean goes into the geometry the beam is in, never into its answer.
        let beam_range = self
            .tick
            .is_multiple_of(RANGEFINDER_PERIOD_TICKS)
            .then(|| {
                self.rangefinder.read(
                    pose.translation()[2],
                    angle_from_upright(state),
                    &mut self.rng,
                )
            })
            .flatten();
        let compass_heading =
            (self.tick % COMPASS_PERIOD_TICKS == COMPASS_PERIOD_TICKS / 2).then(|| {
                self.compass
                    .read(level_heading(pose.rotation()), &mut self.rng)
            });

        // The timed block: everything the flight stack does, and nothing the simulation does.
        let started = Instant::now();
        // The notches are moved onto where the machine reckons the rotors are turning, which it
        // works out from what it last asked them for. That is all it has: what they are really
        // doing is the truth and nobody flying may look at it.
        self.estimator
            .track_rotor_tone(rotor_tone_for(self.last_commanded_thrusts, hovering));
        self.estimator.update(raw_inertial, TIMESTEP);
        let mut fix_thrown_away = false;
        if let Some(fix) = satellite_fix {
            fix_thrown_away = !self.estimator.fold_in_satellite_fix(fix);
        }
        if let Some(speed) = satellite_velocity {
            self.estimator.fold_in_satellite_velocity(
                speed,
                Vector::new([
                    self.satellite.horizontal_speed_noise(),
                    self.satellite.horizontal_speed_noise(),
                    self.satellite.vertical_speed_noise(),
                ]),
            );
        }
        if let Some(range) = beam_range {
            self.estimator
                .fold_in_beam_range(range, self.rangefinder.range_noise());
        }
        if let Some(heading) = compass_heading {
            self.estimator
                .fold_in_heading(heading, self.compass.heading_noise());
        }
        let believed = match self.state_source {
            StateSource::Truth => EstimatedState::from_truth(state),
            StateSource::DeadReckoning => self.estimator.dead_reckoned(),
            StateSource::Filter => self.estimator.believed(),
        };
        let command = self.controller.update(
            reference,
            believed.position,
            believed.velocity,
            believed.orientation,
            believed.angular_rate,
        );
        let math_microseconds = started.elapsed().as_nanos() as f64 / 1000.0;
        let notched_inertial = self.estimator.notched();
        let dead_reckoned = self.estimator.dead_reckoned();
        let filter_belief = self.estimator.believed();

        self.last_commanded_thrusts = command.rotor_thrusts;
        self.plant.step(command.rotor_thrusts, TIMESTEP);

        // What the rotors gave over the tick, against what they were asked for. The gap is what the
        // spin-up costs, and it is the one number that says whether a demand is being followed or
        // merely issued.
        let delivered_thrusts = self.plant.delivered_thrusts();
        let mut rotor_thrust_gap = 0.0_f64;
        for rotor in 0..ROTOR_COUNT {
            rotor_thrust_gap = rotor_thrust_gap
                .max((command.rotor_thrusts[rotor] - delivered_thrusts[rotor]).abs());
        }

        let position_error = (pose.translation() - reference.position()).norm();
        let tilt = angle_from_upright(state);

        self.metrics.ticks += 1;
        if command.at_limit {
            self.metrics.ticks_with_rotor_at_limit += 1;
        }
        self.metrics.worst_tilt = self.metrics.worst_tilt.max(tilt);
        self.metrics.worst_position_error = self.metrics.worst_position_error.max(position_error);
        self.metrics.worst_rotor_thrust_gap =
            self.metrics.worst_rotor_thrust_gap.max(rotor_thrust_gap);
        self.metrics.controller_holds = self.controller.holds();
        self.metrics.fixed_step_fallbacks = self.plant.fixed_step_fallbacks();
        if self.tick > WARMUP_TICKS {
            self.metrics.math_microseconds_total += math_microseconds;
            self.metrics.math_ticks += 1;
        }

        TickRecord {
            tick: self.tick,
            time,
            reference_time,
            pose,
            velocity,
            reference,
            command,
            phase: self.phase,
            delivered_thrusts,
            truth_inertial,
            raw_inertial,
            notched_inertial,
            inertial_offsets,
            shake_phase,
            dead_reckoned,
            believed: filter_belief,
            fix_thrown_away,
            rotor_thrust_gap,
            position_error,
            tilt,
            math_microseconds,
        }
    }
}

/// How fast rotors pushing with `thrusts` come round, in turns a second.
///
/// A rotor's push grows with the square of how fast it turns, so the rate follows the square root of
/// the push. `hovering` is what the whole set pushes with while the machine merely holds itself up,
/// which is the push [`ROTOR_TONE_HERTZ`] belongs to; a machine climbing or leaning is pushing
/// harder than that and shakes a few per cent faster for it.
#[must_use]
pub fn rotor_tone_for(thrusts: Vector<ROTOR_COUNT, f64>, hovering: f64) -> f64 {
    let pushing: f64 = thrusts.as_array().iter().sum();
    ROTOR_TONE_HERTZ * (pushing / hovering).max(0.0).sqrt()
}

/// Where each rotor sits in the world, for drawing the thrust it is producing.
#[must_use]
pub fn rotor_positions_in_world(model: &X2Model, pose: SE3<f64>) -> [[f64; 3]; ROTOR_COUNT] {
    model
        .rotor_positions()
        .map(|position| pose.act(position).into_array())
}
