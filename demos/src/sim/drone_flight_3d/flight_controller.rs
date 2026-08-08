//! The two loops that fly the machine: a slow one that decides where to push, and a fast one that
//! decides how to point.
//!
//! The outer loop sees the body as a point being pushed around and answers with the acceleration
//! that closes the gap to the reference. That acceleration is turned into a direction to lean and a
//! push to apply, and the inner loop closes the gap to that lean every tick. The last step shares
//! the push and twist out across the four rotors.

use multicalc::control::{GeometricAttitudeController, Lqr, thrust_command_from_acceleration};
use multicalc::error::ControlError;
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::plant::MultirotorMixer;
use multicalc::spatial::SO3;

use super::flight_reference::ReferenceSample;
use super::x2_model::{GRAVITY_STRENGTH, ROTOR_COUNT, X2Model};

/// How many numbers the outer loop carries: where the body is, and how fast it is going.
const POSITION_LOOP_DIMENSION: usize = 6;

/// How often the outer loop runs, in ticks — one update every 10 ms, so 100 Hz.
const POSITION_LOOP_PERIOD_TICKS: u64 = 10;

/// The step the outer loop is designed against, matching that period.
const POSITION_LOOP_TIMESTEP: f64 = 0.01;

/// What being away from the reference costs the outer loop, against what asking for acceleration
/// costs it. The two together set how hard the loop pulls: about 7 m/s² of push per metre out of
/// place, damped just short of the point where it would start to overshoot.
const POSITION_ERROR_COST: f64 = 49.0;
const VELOCITY_ERROR_COST: f64 = 11.0;
const ACCELERATION_COST: f64 = 1.0;

/// The most the outer loop may ask for sideways, and up and down, in metres per second squared.
///
/// Down is held well short of a free fall so the rotors never have to stop to deliver it.
const MAXIMUM_SIDEWAYS_ACCELERATION: f64 = 6.0;
const MAXIMUM_CLIMB_ACCELERATION: f64 = 6.0;
const MAXIMUM_DESCENT_ACCELERATION: f64 = 4.0;

/// How far the body may be asked to lean, in radians.
const MAXIMUM_LEAN_ANGLE: f64 = 0.45;

/// How long the wanted acceleration takes to follow a change in what the outer loop asked for, in
/// seconds.
///
/// A jump in the reference is a jump in the wanted acceleration, and a jump in the wanted
/// acceleration is a jump in the direction the body is told to lean. Handed that straight through,
/// the body is asked to turn far faster than its rotors can turn it, and it answers by sitting at
/// its limits until the demand comes back within reach. Easing the whole wanted acceleration in
/// over about a twentieth of a second gives the body something it can actually follow, and is still
/// quick enough to stay well clear of the inner loop.
const ACCELERATION_SMOOTHING_SECONDS: f64 = 0.05;

/// How long one tick lasts, in seconds — the beat the smoothing and the inner loop run on.
const TICK_SECONDS: f64 = 0.001;

/// How quickly the wanted facing may be brought round to catch up with where the body is going, in
/// radians per second.
///
/// This is a limit on catching up, not on following. Once the nose is pointing along travel it
/// simply keeps up with the reference's own turning, however fast that is; the limit only bounds
/// how hard the nose is swung when the two are apart. Capping the following as well would leave the
/// nose falling further behind every second on any circle tighter than the cap allows.
const HEADING_CATCH_UP_RATE: f64 = 0.5;

/// How hard the inner loop pushes on the gap in pointing, and on the gap in turn rate.
///
/// The body resists being spun by about 0.02 to 0.06 kg m², so these settle a lean back in roughly
/// a twentieth of a second without ringing.
const ATTITUDE_GAIN: f64 = 12.0;
const ANGULAR_RATE_GAIN: f64 = 1.2;

/// How close a rotor has to come to a limit before it counts as sitting on one.
const LIMIT_TOLERANCE: f64 = 1e-9;

/// What the controller decided this tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightCommand {
    /// The acceleration the outer loop asked for, after the limits were applied.
    pub wanted_acceleration: Vector3D<f64>,
    /// Which way the body should point to deliver it.
    pub desired_attitude: SO3<f64>,
    /// The total push wanted along the body's up axis, in newtons.
    pub collective_thrust: f64,
    /// The twist wanted about the body's own axes, in newton metres.
    pub torque: Vector3D<f64>,
    /// What each rotor was asked to push with, in newtons.
    pub rotor_thrusts: Vector<ROTOR_COUNT, f64>,
    /// Whether any rotor ended up sitting on a limit, or the twist had to be scaled back to fit.
    pub at_limit: bool,
}

/// The flight stack: the outer loop, the inner loop, and the sharing out across the rotors.
pub struct FlightController {
    position_loop: Lqr<POSITION_LOOP_DIMENSION, 3, f64>,
    attitude_loop: GeometricAttitudeController<f64>,
    stability_certificate: Matrix<POSITION_LOOP_DIMENSION, POSITION_LOOP_DIMENSION, f64>,
    mixer: MultirotorMixer<ROTOR_COUNT, f64>,
    distribution: Matrix<ROTOR_COUNT, 4, f64>,
    mass: f64,
    desired_heading: f64,
    previous_travel_heading: Option<f64>,
    held_acceleration: Vector3D<f64>,
    smoothed_acceleration: Vector3D<f64>,
    held_command: FlightCommand,
    holds: u64,
    tick: u64,
}

impl FlightController {
    /// Builds both loops for a given machine, and proves the outer one settles.
    ///
    /// `starting_heading` is which way the body's nose already points, as an angle in the level
    /// plane. Starting the wanted facing anywhere else asks for a twist far past what the rotors
    /// have, and the body answers by sitting at its limits for the first second of the flight.
    ///
    /// Returns whatever the outer loop, its stability proof, or the inner loop refuses on.
    pub fn new(model: &X2Model, starting_heading: f64) -> Result<Self, ControlError> {
        // The body seen as a point being pushed around: position carries velocity forward, and the
        // input is the acceleration.
        let timestep = POSITION_LOOP_TIMESTEP;
        let state_transition =
            Matrix::<POSITION_LOOP_DIMENSION, POSITION_LOOP_DIMENSION, f64>::from_fn(
                |row, column| {
                    if row == column {
                        1.0
                    } else if row < 3 && column == row + 3 {
                        timestep
                    } else {
                        0.0
                    }
                },
            );
        let input_model = Matrix::<POSITION_LOOP_DIMENSION, 3, f64>::from_fn(|row, column| {
            if row < 3 && row == column {
                0.5 * timestep * timestep
            } else if row >= 3 && row - 3 == column {
                timestep
            } else {
                0.0
            }
        });
        let state_cost = Matrix::from_diagonal([
            POSITION_ERROR_COST,
            POSITION_ERROR_COST,
            POSITION_ERROR_COST,
            VELOCITY_ERROR_COST,
            VELOCITY_ERROR_COST,
            VELOCITY_ERROR_COST,
        ]);
        let input_cost =
            Matrix::from_diagonal([ACCELERATION_COST, ACCELERATION_COST, ACCELERATION_COST]);

        let position_loop = Lqr::new(state_transition, input_model, state_cost, input_cost)?;
        let stability_certificate = position_loop.certify_stability()?;
        let attitude_loop = GeometricAttitudeController::new(
            ATTITUDE_GAIN,
            ANGULAR_RATE_GAIN,
            model.inertia().rotational_inertia(),
        )?;

        // Sitting still and level, holding itself up: the command the loops fall back on if the
        // wanted acceleration ever comes out as one no lean can deliver.
        let hovering = model.mass() * GRAVITY_STRENGTH;
        let held_command = FlightCommand {
            wanted_acceleration: Vector::zeros(),
            desired_attitude: SO3::identity(),
            collective_thrust: hovering,
            torque: Vector::zeros(),
            rotor_thrusts: Vector::from_fn(|_| model.hover_thrust_per_rotor()),
            at_limit: false,
        };

        Ok(FlightController {
            position_loop,
            attitude_loop,
            stability_certificate,
            mixer: model.mixer(),
            distribution: model.distribution(),
            mass: model.mass(),
            desired_heading: starting_heading,
            previous_travel_heading: None,
            held_acceleration: Vector::zeros(),
            smoothed_acceleration: Vector::zeros(),
            held_command,
            holds: 0,
            tick: 0,
        })
    }

    /// The proof that the outer loop settles, worked out once when it was built.
    #[inline]
    pub fn stability_certificate(
        &self,
    ) -> Matrix<POSITION_LOOP_DIMENSION, POSITION_LOOP_DIMENSION, f64> {
        self.stability_certificate
    }

    /// The matrix that turns being out of place into the acceleration that answers it.
    #[inline]
    pub fn position_gain(&self) -> Matrix<3, POSITION_LOOP_DIMENSION, f64> {
        self.position_loop.gain()
    }

    /// How many ticks held the previous command because the wanted acceleration named no direction
    /// to push in.
    #[inline]
    #[must_use]
    pub fn holds(&self) -> u64 {
        self.holds
    }

    /// Which way the nose should point, as an angle in the level plane.
    #[inline]
    #[must_use]
    pub fn desired_heading(&self) -> f64 {
        self.desired_heading
    }

    /// Points the wanted facing straight at `heading`, rather than bringing it round.
    ///
    /// For placing the body before it flies and nothing else. The wanted facing has to start at the
    /// body's own, or the first tick asks for a twist far past what the rotors have.
    #[inline]
    pub fn set_desired_heading(&mut self, heading: f64) {
        self.desired_heading = heading;
        self.previous_travel_heading = None;
    }

    /// Runs one tick of both loops and shares the answer out across the rotors.
    ///
    /// The outer loop only runs on its own slower beat and its answer is held in between; the inner
    /// loop and the sharing out run every tick.
    pub fn update(
        &mut self,
        reference: ReferenceSample,
        position: Vector3D<f64>,
        velocity: Vector3D<f64>,
        orientation: SO3<f64>,
        angular_rate: Vector3D<f64>,
    ) -> FlightCommand {
        self.tick += 1;
        self.bring_the_nose_round(reference);
        if self.tick.is_multiple_of(POSITION_LOOP_PERIOD_TICKS) || self.tick == 1 {
            let state = Vector::new([
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ]);
            let wanted = reference.position();
            let wanted_velocity = reference.velocity();
            let target = Vector::new([
                wanted[0],
                wanted[1],
                wanted[2],
                wanted_velocity[0],
                wanted_velocity[1],
                wanted_velocity[2],
            ]);
            let answer =
                self.position_loop
                    .control_tracking(state, target, reference.acceleration());
            self.held_acceleration = held_inside_reach(answer);
        }

        // Ease the wanted acceleration toward what the outer loop asked for rather than jumping to
        // it. Every value being eased toward is one the body can hold, and the ones in between are
        // blends of those, so the smoothing never asks for anything past its reach.
        let catch_up = TICK_SECONDS / (ACCELERATION_SMOOTHING_SECONDS + TICK_SECONDS);
        self.smoothed_acceleration = self.smoothed_acceleration
            + (self.held_acceleration - self.smoothed_acceleration).scale(catch_up);

        let Ok(thrust_command) = thrust_command_from_acceleration(
            self.smoothed_acceleration,
            self.desired_heading,
            GRAVITY_STRENGTH,
        ) else {
            self.holds += 1;
            return self.held_command;
        };

        let desired_attitude = thrust_command.attitude();
        // Push only as hard as actually goes the way the push is wanted. A body still turning into
        // its lean points somewhere else, and the full push applied there goes somewhere else too —
        // sideways at full power, or straight up when neither was asked for. How nearly the two up
        // directions agree scales the push down to the part that lands where it was meant to, and
        // costs nothing once the body is pointing the right way.
        let up: Vector3D<f64> = Vector::new([0.0, 0.0, 1.0]);
        let alignment = orientation.act(up).dot(desired_attitude.act(up)).max(0.0);
        let collective_thrust = thrust_command.thrust_force(self.mass) * alignment;
        let still = Vector::zeros();
        let torque =
            self.attitude_loop
                .torque(orientation, angular_rate, desired_attitude, still, still);

        let (rotor_thrusts, at_limit) = self.share_out(collective_thrust, torque);
        self.held_command = FlightCommand {
            wanted_acceleration: self.smoothed_acceleration,
            desired_attitude,
            collective_thrust,
            torque,
            rotor_thrusts,
            at_limit,
        };
        self.held_command
    }

    /// Moves the wanted facing a tick's worth toward where the reference is travelling.
    ///
    /// The direction comes from the reference's own velocity rather than from where the body is
    /// pointing, because while the nose is still coming round the two are not the same and it is
    /// the reference that is right.
    ///
    /// The facing is carried along by however far the reference's own direction turned this tick,
    /// and only what is left over — the gap the nose is still behind by — is limited. So a nose
    /// already pointing along travel stays there however tight the circle, while one that is a
    /// long way off is swung round at a pace the rotors can hold rather than snapped.
    fn bring_the_nose_round(&mut self, reference: ReferenceSample) {
        let Some(travel) = reference.travel_heading() else {
            self.previous_travel_heading = None;
            return;
        };
        let carried = match self.previous_travel_heading {
            Some(previous) => shortest_way_round(travel - previous),
            None => 0.0,
        };
        self.previous_travel_heading = Some(travel);

        let catch_up = HEADING_CATCH_UP_RATE * TICK_SECONDS;
        let behind = shortest_way_round(travel - self.desired_heading - carried);
        self.desired_heading =
            shortest_way_round(self.desired_heading + carried + behind.clamp(-catch_up, catch_up));
    }

    /// Shares a wanted push and twist out across the rotors, holding each one inside its limits
    /// without giving up any more of the twist than it has to.
    ///
    /// Moving all four rotors by the same amount changes only their total push, because this layout
    /// is even enough that the four contributions to each part of the twist cancel. So the push
    /// fixes the level the four sit at, and the twist is the spread about that level. When the
    /// spread does not fit between the level and a rotor's limits, the twist is scaled back until
    /// it does — the push is what is kept, and the twist is what gives way.
    ///
    /// Which of the two gives way matters more than it looks. Sliding the level up instead, to keep
    /// the lowest rotor turning, delivers the whole twist and hands the body a climb nothing asked
    /// for — and the demand most likely to ask for a twist that does not fit is the one about the
    /// body's own up axis, which this layout can only produce by playing the rotors' own drag
    /// against each other and is therefore weak at. A body a couple of degrees off which way it is
    /// pointing would swap that for a push of twice its own weight. A twist scaled back is only a
    /// slower turn, which is much the cheaper mistake.
    ///
    /// Holding each rotor inside its limits on its own instead would change the twist as well as
    /// the total push, and the body would get neither of the two things it was asked for.
    fn share_out(
        &self,
        collective_thrust: f64,
        torque: Vector3D<f64>,
    ) -> (Vector<ROTOR_COUNT, f64>, bool) {
        let twist_share = self.distribution * Vector::new([0.0, torque[0], torque[1], torque[2]]);
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for share in twist_share.as_array() {
            lowest = lowest.min(*share);
            highest = highest.max(*share);
        }

        // The level that gives the wanted total push, and how much room is left either side of it.
        let minimum = self.mixer.minimum_thrust();
        let maximum = self.mixer.maximum_thrust();
        let level = (collective_thrust / ROTOR_COUNT as f64).clamp(minimum, maximum);
        let room_below = level - minimum;
        let room_above = maximum - level;

        let mut twist_scale = 1.0_f64;
        if lowest < 0.0 {
            twist_scale = twist_scale.min(room_below / -lowest);
        }
        if highest > 0.0 {
            twist_scale = twist_scale.min(room_above / highest);
        }
        let twist = twist_share.scale(twist_scale);
        let thrusts = Vector::from_fn(|rotor| level + twist[rotor]);

        let at_limit = twist_scale < 1.0
            || thrusts.as_array().iter().any(|thrust| {
                *thrust <= minimum + LIMIT_TOLERANCE || *thrust >= maximum - LIMIT_TOLERANCE
            });
        (thrusts, at_limit)
    }
}

/// The same angle brought into the half turn either side of zero, so the difference between two
/// facings is the short way round rather than the long one.
fn shortest_way_round(angle: f64) -> f64 {
    let full_turn = std::f64::consts::TAU;
    let brought_in = angle % full_turn;
    if brought_in > std::f64::consts::PI {
        brought_in - full_turn
    } else if brought_in < -std::f64::consts::PI {
        brought_in + full_turn
    } else {
        brought_in
    }
}

/// Holds a wanted acceleration inside what the machine can actually deliver.
///
/// Sideways and up-and-down are held separately first, then both together: pushing sideways at all
/// means leaning, and how far the body has to lean depends on how hard it is pushing upward at the
/// same time, so a sideways demand that would need an impossible lean is cut back to the one that
/// does not.
fn held_inside_reach(wanted: Vector3D<f64>) -> Vector3D<f64> {
    let vertical = wanted[2].clamp(-MAXIMUM_DESCENT_ACCELERATION, MAXIMUM_CLIMB_ACCELERATION);
    let sideways = wanted[0].hypot(wanted[1]);
    if sideways <= 0.0 {
        return Vector::new([0.0, 0.0, vertical]);
    }
    let leaning_reach = MAXIMUM_LEAN_ANGLE.tan() * (GRAVITY_STRENGTH + vertical);
    let allowed = sideways
        .min(MAXIMUM_SIDEWAYS_ACCELERATION)
        .min(leaning_reach);
    let scale = allowed / sideways;
    Vector::new([wanted[0] * scale, wanted[1] * scale, vertical])
}
