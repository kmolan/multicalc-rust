//! Where the body should be, how fast it should be going, and how quickly that should be changing.

use multicalc::error::MotionError;
use multicalc::motion::{
    BoundaryDerivatives, MinimumSnapPlanner, PiecewisePolynomial, durations_from_average_speed,
};

use multicalc::linear_algebra::{Vector, Vector3D};

use super::x2_model::GRAVITY_STRENGTH;

/// How fast the reference has to be travelling before the direction it is travelling in is worth
/// reading, in metres per second. Below this the direction is all rounding error.
const TRAVELLING_SPEED: f64 = 0.05;

/// What the reference asks for at one moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceSample {
    position: Vector3D<f64>,
    velocity: Vector3D<f64>,
    acceleration: Vector3D<f64>,
}

impl ReferenceSample {
    /// A sample from its three parts.
    #[inline]
    #[must_use]
    pub fn new(
        position: Vector3D<f64>,
        velocity: Vector3D<f64>,
        acceleration: Vector3D<f64>,
    ) -> Self {
        ReferenceSample {
            position,
            velocity,
            acceleration,
        }
    }

    /// Where the body should be.
    #[inline]
    pub fn position(self) -> Vector3D<f64> {
        self.position
    }

    /// How fast it should be going.
    #[inline]
    pub fn velocity(self) -> Vector3D<f64> {
        self.velocity
    }

    /// How quickly that speed should be changing.
    #[inline]
    pub fn acceleration(self) -> Vector3D<f64> {
        self.acceleration
    }

    /// Which way the reference is travelling in the level plane, as an angle from the world's +x
    /// axis, or nothing at all while it is barely moving.
    ///
    /// This is where the reference is going, not where the body is facing. The two come apart
    /// whenever the nose is still being brought round, and it is the reference that is right.
    #[inline]
    #[must_use]
    pub fn travel_heading(self) -> Option<f64> {
        let level_speed = self.velocity[0].hypot(self.velocity[1]);
        if level_speed < TRAVELLING_SPEED {
            None
        } else {
            Some(self.velocity[1].atan2(self.velocity[0]))
        }
    }
}

/// How many corners the planned loop is built around. The loop closes back onto the first, so there
/// is one more waypoint than that, and one segment per corner.
pub const PLAN_WAYPOINTS: usize = 10;
const PLAN_SEGMENTS: usize = PLAN_WAYPOINTS;

/// How much room the solve needs to work in.
///
/// Three values — how fast, how fast that is changing, and how fast that is changing — are chosen at
/// every waypoint except the two ends, and the ends are given. Asking for less than this comes back
/// as a refusal rather than a wrong answer, but it is a refusal at startup for something that could
/// have been counted, so it is counted.
const PLAN_FREE_DERIVATIVES: usize = 3 * (PLAN_SEGMENTS - 1);

/// How many coefficients each piece of the planned path carries — fixed by what the planner builds.
const PLAN_COEFFICIENTS: usize = 8;

/// How wide the planned loop is, how much that width swings in and out, how high it flies, and how
/// far it climbs and dives, all in metres.
///
/// The width and the height each swing twice round the loop, which makes a closed line that leans in
/// and out and climbs and dives twice a lap without ever turning through more than a gentle corner.
/// A waypoint the path has to turn through much more than a right angle forces the curve to nearly
/// stop there, and no amount of re-timing fixes that — it is the corner, not the timing.
const PLAN_RADIUS: f64 = 3.5;
const PLAN_RADIUS_SWING: f64 = 0.8;
const PLAN_HEIGHT: f64 = 1.5;
const PLAN_HEIGHT_SWING: f64 = 0.6;

/// How many times the timing is worked out again from the curve the last one produced.
///
/// The first timing can only measure the straight lines between waypoints, and the path through a
/// waypoint bends, so it is longer than the line and gets flown faster than the pace asked for.
/// Walking the curve gives the length it really is. One round settles it; the second is cheap
/// insurance.
const TIMING_ROUNDS: usize = 2;

/// How finely the curve is walked when measuring how long each of its pieces really is.
const ARC_LENGTH_STEPS: usize = 200;

/// How wide the circle the body is asked to fly is, in metres.
pub const CIRCLE_RADIUS: f64 = 3.0;

/// How fast it is asked to fly it, in metres per second.
pub const CIRCLE_SPEED: f64 = 2.0;

/// How far each jump moves the body, in metres.
pub const STEP_DISTANCE: f64 = 2.0;

/// How long the body is left on each point before the next jump, in seconds.
pub const STEP_HOLD_SECONDS: f64 = 10.0;

/// Where each point of the stepping cycle sits, relative to the point the cycle is built around.
///
/// The body is left where it is, then jumped out along one axis and back again, one axis at a time.
/// Every jump — including the one that wraps the end of the list round to the start — moves it
/// exactly [`STEP_DISTANCE`] along a single axis, so every step is the same size and the same shape.
const STEP_OFFSETS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [STEP_DISTANCE, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, STEP_DISTANCE, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, STEP_DISTANCE],
];

/// The path the body is asked to follow.
///
/// The planned line carries its whole solved curve inside it, which makes it far larger than the
/// others — a couple of kilobytes against a couple of dozen bytes. It is kept inline anyway: exactly
/// one of these exists per flight, it is copied two or three times at startup and never again, and
/// putting it behind a pointer would cost a hop on every tick to save nothing that can be measured.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlightReference {
    /// One point, held, with nothing asked of the body but to sit on it.
    Hover {
        /// The point to sit on.
        point: Vector3D<f64>,
    },
    /// A point that jumps two metres along one axis every ten seconds.
    ///
    /// The jump is deliberately a jump: the reference says nothing about how fast the body should
    /// be going or how quickly it should be speeding up, so the loop gets no help at all and has to
    /// answer the whole move out of the gap it can see.
    Stepping {
        /// The point the cycle is built around.
        base: Vector3D<f64>,
    },
    /// A planned line through a ring of waypoints, flown over and over.
    ///
    /// Where the circle is a shape with a formula, this is a shape with a solve behind it: the
    /// smoothest line through a set of corners, worked out once before the flight and never touched
    /// again. It is the first reference the body has to work at, because the plan itself speeds up
    /// and slows down and climbs and dives, and none of that is the loop's doing.
    Planned {
        /// The solved curve, one piece per pair of corners.
        path: PiecewisePolynomial<PLAN_SEGMENTS, PLAN_COEFFICIENTS, 3, f64>,
    },
    /// A circle flown level at a steady speed.
    ///
    /// The one path whose right answer is known outright: the speed never changes, the push toward
    /// the centre is exactly `speed²/radius` the whole way round, and the lean that delivers it is
    /// exactly [`lean_for_circle`]. Every part of it — where to be, how fast to be going, and how
    /// quickly that should be changing — is written down here, so the loop only has to answer the
    /// gap rather than work the whole manoeuvre out from it.
    Circle {
        /// The point the circle is drawn around, at the height it is flown at.
        center: Vector3D<f64>,
        /// How far from that point the body flies, in metres.
        radius: f64,
        /// How fast it travels along the circle, in metres per second.
        speed: f64,
    },
}

impl FlightReference {
    /// A reference that holds one point.
    #[inline]
    #[must_use]
    pub fn hover(point: Vector3D<f64>) -> Self {
        FlightReference::Hover { point }
    }

    /// A reference that jumps two metres along one axis every ten seconds, around `base`.
    #[inline]
    #[must_use]
    pub fn stepping(base: Vector3D<f64>) -> Self {
        FlightReference::Stepping { base }
    }

    /// Works out the smoothest line through the ring of waypoints and hands it back as a reference.
    ///
    /// This is the whole of the planning, done once. Nothing here runs again while the machine
    /// flies: what comes back is a set of curve pieces, and following it afterwards is a handful of
    /// multiplies.
    ///
    /// The two ends of the loop are given the same motion — running through the join at `pace`, with
    /// nothing speeding up or changing there — so flying it over and over by wrapping the clock is
    /// smooth across the join. Left to itself the planner would bring the path to a dead stop at
    /// both ends, and the machine would halt and set off again every lap, which is by far the
    /// largest change of speed anywhere on the loop.
    ///
    /// Returns whatever the planner refuses on.
    pub fn planned_loop(pace: f64) -> Result<Self, MotionError> {
        let waypoints = planned_waypoints();
        let mut durations = [0.0; PLAN_SEGMENTS];
        durations_from_average_speed(&waypoints, pace, &mut durations)?;

        // How the path is travelling as it runs through the join, taken from the corner before it to
        // the corner after.
        let across_the_join = waypoints[1] - waypoints[PLAN_SEGMENTS - 1];
        let length = across_the_join.norm();
        if length <= 0.0 || !length.is_finite() {
            return Err(MotionError::NonFinite);
        }
        let through_the_join = BoundaryDerivatives {
            velocity: across_the_join.scale(pace / length),
            acceleration: Vector::zeros(),
            jerk: Vector::zeros(),
        };
        let planner = MinimumSnapPlanner::<PLAN_SEGMENTS, PLAN_FREE_DERIVATIVES, 3, f64>::new()
            .with_start(through_the_join)
            .with_end(through_the_join);

        let mut path = planner.plan(&waypoints, &durations)?;
        for _ in 0..TIMING_ROUNDS {
            let mut start = 0.0;
            for (segment, duration) in durations.iter_mut().enumerate() {
                let span = path.span(segment).unwrap_or(*duration);
                *duration = walked_length(&path, start, span) / pace;
                start += span;
            }
            path = planner.plan(&waypoints, &durations)?;
        }
        Ok(FlightReference::Planned { path })
    }

    /// A reference that goes round a circle of `radius` at a steady `speed`, around `center`.
    #[inline]
    #[must_use]
    pub fn circle(center: Vector3D<f64>, radius: f64, speed: f64) -> Self {
        FlightReference::Circle {
            center,
            radius,
            speed,
        }
    }

    /// How long one time round takes, in seconds. Zero for a reference that does not go round.
    #[must_use]
    pub fn lap_seconds(&self) -> f64 {
        match self {
            FlightReference::Hover { .. } | FlightReference::Stepping { .. } => 0.0,
            FlightReference::Planned { path } => path.total_span(),
            FlightReference::Circle { radius, speed, .. } => std::f64::consts::TAU * radius / speed,
        }
    }

    /// What the reference asks for at `time` seconds into the flight.
    #[must_use]
    pub fn sample(&self, time: f64) -> ReferenceSample {
        match *self {
            FlightReference::Hover { point } => {
                ReferenceSample::new(point, Vector::zeros(), Vector::zeros())
            }
            FlightReference::Stepping { base } => {
                let point = base + Vector::new(STEP_OFFSETS[step_index(time)]);
                ReferenceSample::new(point, Vector::zeros(), Vector::zeros())
            }
            FlightReference::Planned { path } => {
                // The clock is wrapped rather than the path being flown once: what comes back at the
                // very end of a lap is what comes back at the start of the next one, because both
                // ends were given the same motion when it was solved.
                let lap = path.total_span();
                let along = if lap > 0.0 { time.rem_euclid(lap) } else { 0.0 };
                match path.evaluate_with_derivatives::<3>(along) {
                    Ok([position, velocity, acceleration]) => {
                        ReferenceSample::new(position, velocity, acceleration)
                    }
                    // A curve with no pieces in it has nowhere to be; holding still is the one
                    // answer that cannot make things worse.
                    Err(_) => {
                        ReferenceSample::new(Vector::zeros(), Vector::zeros(), Vector::zeros())
                    }
                }
            }
            FlightReference::Circle {
                center,
                radius,
                speed,
            } => {
                let turn_rate = speed / radius;
                let (sine, cosine) = (turn_rate * time).sin_cos();
                let inward = speed * turn_rate;
                ReferenceSample::new(
                    center + Vector::new([radius * cosine, radius * sine, 0.0]),
                    Vector::new([-speed * sine, speed * cosine, 0.0]),
                    Vector::new([-inward * cosine, -inward * sine, 0.0]),
                )
            }
        }
    }

    /// How far `position` sits off the circle, as how far off the radius it is and how far off the
    /// height, both in metres. Nothing at all for a reference that is not a circle.
    ///
    /// The two are kept apart because they are held by different things — the lean holds the
    /// radius and the total push holds the height — so one being wrong says something the other
    /// does not.
    #[must_use]
    pub fn offset_from_the_circle(&self, position: Vector3D<f64>) -> Option<(f64, f64)> {
        let FlightReference::Circle {
            center,
            radius,
            speed: _,
        } = *self
        else {
            return None;
        };
        let from_the_centre = (position[0] - center[0]).hypot(position[1] - center[1]);
        Some((from_the_centre - radius, position[2] - center[2]))
    }

    /// The line the reference traces out over one full time round, as `samples` even steps that
    /// close back on where they started. Empty for a reference that does not go round.
    #[must_use]
    pub fn traced_ring(&self, samples: usize) -> Vec<[f64; 3]> {
        let time_round = self.lap_seconds();
        if samples == 0 || time_round <= 0.0 {
            return Vec::new();
        }
        (0..=samples)
            .map(|step| {
                let time = time_round * step as f64 / samples as f64;
                self.sample(time).position().into_array()
            })
            .collect()
    }
}

/// Where the corners of the planned loop sit, with the last one back on the first so the line
/// closes.
pub fn planned_waypoints() -> [Vector3D<f64>; PLAN_SEGMENTS + 1] {
    std::array::from_fn(|corner| {
        let angle = std::f64::consts::TAU * corner as f64 / PLAN_WAYPOINTS as f64;
        let radius = PLAN_RADIUS + PLAN_RADIUS_SWING * (2.0 * angle).cos();
        let height = PLAN_HEIGHT + PLAN_HEIGHT_SWING * (2.0 * angle).sin();
        Vector::new([radius * angle.cos(), radius * angle.sin(), height])
    })
}

/// How long one piece of the curve really is, walked rather than measured across.
fn walked_length(
    path: &PiecewisePolynomial<PLAN_SEGMENTS, PLAN_COEFFICIENTS, 3, f64>,
    start: f64,
    span: f64,
) -> f64 {
    let mut walked = 0.0;
    let mut previous = path.evaluate(start).unwrap_or(Vector::zeros());
    for step in 1..=ARC_LENGTH_STEPS {
        let along = start + span * step as f64 / ARC_LENGTH_STEPS as f64;
        let here = path.evaluate(along).unwrap_or(previous);
        walked += (here - previous).norm();
        previous = here;
    }
    walked
}

/// How far the body has to lean to fly a circle of `radius` at `speed`, in radians.
///
/// Going round in a circle means being pushed toward the centre the whole way, and the only way
/// this machine pushes sideways is by leaning. That ties the two together exactly, so this is the
/// lean the flight has to come out at — a fact to check the demo against, not a number to tune
/// toward. A measured lean that does not match it means the tilt conversion or the rotor sharing
/// is wrong, and no amount of loop tuning will cover for that.
#[must_use]
pub fn lean_for_circle(radius: f64, speed: f64) -> f64 {
    (speed * speed / (radius * GRAVITY_STRENGTH)).atan()
}

/// Which point of the stepping cycle `time` falls on.
#[must_use]
fn step_index(time: f64) -> usize {
    let elapsed_holds = (time / STEP_HOLD_SECONDS).floor().max(0.0) as usize;
    elapsed_holds % STEP_OFFSETS.len()
}
