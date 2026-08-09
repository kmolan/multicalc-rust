//! The truth motion: what the machine actually does when its rotors are asked for a set of thrusts.
//!
//! Nothing here knows about the controller. It takes what each rotor was asked for and answers with
//! where the body has got to and how it is moving. In between sit the two things that stop the body
//! doing exactly what it is told: rotors that take time to spin up, and air that pushes back.

use multicalc::dynamics::{RigidBody, RigidBodyAcceleration};
use multicalc::error::PlantError;
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::ode::{ExponentialMap, Rk45};
use multicalc::plant::{MultirotorMixer, RotorLag};
use multicalc::scalar::Numeric;
use multicalc::spatial::{FreeJointState, SE3, SO3, Twist, Wrench};

use super::x2_model::{ROTOR_COUNT, X2Model};

/// How many numbers the straight-line part of the motion takes: where the body is, and how fast it
/// is going.
const TRANSLATION_DIMENSION: usize = 6;

/// How long a rotor takes to close a little under two thirds of the gap to the thrust it was asked
/// for, in seconds.
///
/// A rotor has to spin up or slow down before its push changes, and on a machine this size that
/// takes a few hundredths of a second — long enough that a demand which swings about faster than
/// this simply never arrives.
const ROTOR_LAG_SECONDS: f64 = 0.025;

/// How hard the air pushes back, in newtons per squared metre per second of speed.
///
/// The push grows with the square of the speed and always opposes travel, so it is nothing at all
/// in a hover and worth a few per cent of the machine's weight at the speeds this demo flies.
const DRAG_COEFFICIENT: f64 = 0.1;

/// The moving body, the rotors driving it, and the two integrators that carry it forward.
///
/// Which way the body faces is carried as a turn rather than as four loose numbers, so it stays a
/// true rotation with no drift to scale away. Where the body is goes through an adaptive step,
/// which picks its own step sizes inside the tick and falls back to a plain fixed step if it cannot
/// settle — a stalled integrator holds up the demo, it does not stop it.
pub struct FlightPlant {
    body: RigidBody<f64>,
    mixer: MultirotorMixer<ROTOR_COUNT, f64>,
    rotors: RotorLag<ROTOR_COUNT, f64>,
    hover_thrust_per_rotor: f64,
    integrator: Rk45<f64>,
    state: FreeJointState<f64>,
    acceleration: RigidBodyAcceleration<f64>,
    proper_acceleration: Vector3D<f64>,
    fixed_step_fallbacks: u64,
}

impl FlightPlant {
    /// A machine placed where `start` says, moving however `start` says, with its rotors already
    /// carrying their share of a hover.
    ///
    /// `timestep` is how long one tick lasts. The rotors' catching up is worked out once against
    /// it, so every tick after that is a couple of multiplies.
    ///
    /// Starting the rotors at a hover rather than at a standstill is what a machine already in the
    /// air is doing: started from nothing they would take a few hundredths of a second to take the
    /// weight, and the body would drop through the floor of the demo before it ever flew.
    ///
    /// Returns whatever the rotors' spin-up model refuses on.
    pub fn new(
        model: &X2Model,
        start: FreeJointState<f64>,
        timestep: f64,
    ) -> Result<Self, PlantError> {
        let body = model.body();
        let hover_thrust_per_rotor = model.hover_thrust_per_rotor();
        let rotors = RotorLag::<ROTOR_COUNT, f64>::new(ROTOR_LAG_SECONDS, timestep)?
            .with_thrusts(Vector::from_fn(|_| hover_thrust_per_rotor));

        let mut plant = FlightPlant {
            body,
            mixer: model.mixer(),
            rotors,
            hover_thrust_per_rotor,
            integrator: Rk45::default(),
            state: start,
            acceleration: RigidBodyAcceleration::new(Vector::zeros(), Vector::zeros()),
            proper_acceleration: Vector::zeros(),
            fixed_step_fallbacks: 0,
        };
        plant.reset(start);
        Ok(plant)
    }

    /// Puts the machine back where `start` says, with its rotors back at a hover and the run's
    /// tallies cleared, ready to fly again.
    pub fn reset(&mut self, start: FreeJointState<f64>) {
        self.state = start;
        self.rotors = self
            .rotors
            .with_thrusts(Vector::from_fn(|_| self.hover_thrust_per_rotor));
        self.fixed_step_fallbacks = 0;

        // What the machine is doing before it has flown a tick: rotors already carrying the weight,
        // and the air pushing back on whatever motion it was placed with.
        let orientation = start.pose().rotation();
        let applied =
            self.mixer.wrench(self.rotors.thrusts()) + drag(orientation, start.velocity().linear());
        self.acceleration =
            self.body
                .accelerations(orientation, start.velocity().angular(), applied);
        self.proper_acceleration = self.felt_at_the_origin(orientation);
    }

    /// What a sensor bolted at the body's origin felt over the tick just gone: everything acting on
    /// the body except gravity, read in the body's own axes.
    ///
    /// This is not the applied force divided by the mass. The origin is not the point the body
    /// balances on, so as the body turns the origin swings around that point and picks up
    /// acceleration the balance point never sees — a third of a gravity of it in a hard turn. What
    /// comes out here is what the body's origin actually did, which is what a unit bolted there
    /// feels.
    #[inline]
    pub fn proper_acceleration(&self) -> Vector3D<f64> {
        self.proper_acceleration
    }

    /// What each rotor is actually pushing with, which is not what it was last asked for.
    #[inline]
    pub fn delivered_thrusts(&self) -> Vector<ROTOR_COUNT, f64> {
        self.rotors.thrusts()
    }

    /// Where the body is and how it is moving.
    #[inline]
    #[must_use]
    pub fn state(&self) -> FreeJointState<f64> {
        self.state
    }

    /// How quickly the motion was changing at the start of the last tick.
    ///
    /// The straight-line part is the acceleration of the body frame's **origin**, not of the point
    /// the body balances on — the two differ as the body turns, and the origin's is what a sensor
    /// bolted there feels.
    #[inline]
    #[must_use]
    pub fn acceleration(&self) -> RigidBodyAcceleration<f64> {
        self.acceleration
    }

    /// How many times the adaptive step gave up and the fixed step took over.
    #[inline]
    #[must_use]
    pub fn fixed_step_fallbacks(&self) -> u64 {
        self.fixed_step_fallbacks
    }

    /// What the body's origin is doing, minus gravity, read in the body's own axes.
    ///
    /// Gravity is taken off because it is the one push a falling sensor cannot feel: left in, a
    /// machine hanging still in the air would read nothing at all, and one in free fall would read
    /// a gravity.
    fn felt_at_the_origin(&self, orientation: SO3<f64>) -> Vector3D<f64> {
        orientation
            .inverse()
            .act(self.acceleration.linear() - self.body.gravity())
    }

    /// Moves the machine one tick, with `commanded_thrusts` asked of the rotors across it.
    ///
    /// What the rotors give is not what they were asked for: each one is a little behind, and the
    /// air is pushing back on the body the whole time. Both are worked out once at the start of the
    /// tick and held across it, which is what the rotors' own model expects and is close enough for
    /// a tick a thousandth of a second long.
    pub fn step(&mut self, commanded_thrusts: Vector<ROTOR_COUNT, f64>, timestep: f64) {
        let delivered = self.rotors.stepped(commanded_thrusts);
        let applied_wrench = self.mixer.wrench(delivered)
            + drag(self.state.pose().rotation(), self.state.velocity().linear());

        let pose = self.state.pose();
        let orientation = pose.rotation();
        let position = pose.translation();
        let linear_velocity = self.state.velocity().linear();
        let angular_rate = self.state.velocity().angular();
        let half_step = timestep * 0.5;

        // The turning half, read once at the start and once half way through, then carried forward
        // with the half-way values.
        let at_start = self
            .body
            .accelerations(orientation, angular_rate, applied_wrench);
        let half_way_orientation =
            ExponentialMap::attitude_step(orientation, angular_rate, half_step);
        let half_way_rate = angular_rate + at_start.angular() * half_step;
        let half_way = self
            .body
            .accelerations(half_way_orientation, half_way_rate, applied_wrench);
        let next_orientation = ExponentialMap::attitude_step(orientation, half_way_rate, timestep);
        let next_angular_rate = angular_rate + half_way.angular() * timestep;

        // The straight-line half. How fast the origin picks up speed depends on which way the body
        // points and how fast it is turning, and both of those move across the tick, so the rate
        // being integrated carries them forward as it goes.
        let body = self.body;
        let rate_of_change = |time: f64, motion: &Vector<TRANSLATION_DIMENSION, f64>| {
            let orientation_then = ExponentialMap::attitude_step(orientation, half_way_rate, time);
            let rate_then = angular_rate + at_start.angular() * time;
            let linear = body
                .accelerations(orientation_then, rate_then, applied_wrench)
                .linear();
            Vector::new([
                motion[3], motion[4], motion[5], linear[0], linear[1], linear[2],
            ])
        };
        let start_motion = Vector::new([
            position[0],
            position[1],
            position[2],
            linear_velocity[0],
            linear_velocity[1],
            linear_velocity[2],
        ]);
        let (next_position, next_linear_velocity) =
            match self
                .integrator
                .solve(&rate_of_change, 0.0, &start_motion, timestep)
            {
                Ok(motion) => (
                    Vector::new([motion[0], motion[1], motion[2]]),
                    Vector::new([motion[3], motion[4], motion[5]]),
                ),
                Err(_) => {
                    self.fixed_step_fallbacks += 1;
                    let linear = half_way.linear();
                    (
                        position + linear_velocity * timestep + linear * (half_step * timestep),
                        linear_velocity + linear * timestep,
                    )
                }
            };

        self.acceleration = at_start;
        self.proper_acceleration = self.felt_at_the_origin(orientation);
        self.state = FreeJointState::new(
            SE3::from_parts(next_orientation, next_position),
            Twist::new(next_linear_velocity, next_angular_rate),
        );
    }
}

/// How hard the air pushes back on a body moving at `velocity`, ready to be added to what the
/// rotors are doing.
///
/// The push grows with the square of the speed and points straight against travel. It is taken as
/// acting on the body's origin, so it slows the machine down without twisting it — the twist a real
/// airframe picks up from the air depends on its shape, which is not something a model file says
/// anything about.
///
/// The answer comes back in the body's own axes, because that is where everything acting on the
/// body is read.
fn drag(orientation: SO3<f64>, velocity: Vector3D<f64>) -> Wrench<f64> {
    let speed = velocity.norm();
    if speed == 0.0 {
        return Wrench::zeros();
    }
    let in_the_world = velocity.scale(-DRAG_COEFFICIENT * speed);
    Wrench::new(orientation.inverse().act(in_the_world), Vector::zeros())
}

/// Which way a facing points in the level plane, as an angle from the world's +x axis.
///
/// Written over any kind of number rather than over plain ones alone, because the filter asks the
/// very same question of its own state, in a scalar that carries the slope of every answer along
/// with it. One question, one answer, one place it is worked out.
#[must_use]
pub fn level_heading<S: Numeric>(orientation: SO3<S>) -> S {
    let forward: Vector3D<S> = orientation.act(Vector::new([S::ONE, S::ZERO, S::ZERO]));
    forward[1].atan2(forward[0])
}

/// The angle between the body's own up direction and the world's, in radians.
#[must_use]
pub fn angle_from_upright(state: FreeJointState<f64>) -> f64 {
    let world_up: Vector3D<f64> = Vector::new([0.0, 0.0, 1.0]);
    let body_up = state.pose().rotation().act(world_up);
    body_up[2].clamp(-1.0, 1.0).acos()
}
