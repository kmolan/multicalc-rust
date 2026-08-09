//! What `--check` runs: the two flights that say the demo still does what it claims.
//!
//! A gate is a fault report, not a target — a bound that fails means something is wrong, not that
//! the bound is.
//!
//! This is deliberately not a test suite. The demo is the deliverable, and these two flights are
//! what is worth re-measuring every time it is touched:
//!
//! - **What the estimate costs the flying.** The same circle flown twice, once with the loops handed
//!   the truth and once with them handed the filter's answer. Neither row means much alone; the gap
//!   between them is the whole point, because it is the only number that separates a control problem
//!   from an estimate that was never good enough to fly on.
//! - **Whether it can find itself and then fly.** Set down on the pad knowing roughly where it was
//!   put and nothing about which way it faces, left to work the rest out against a floor plan, and
//!   then asked to climb onto the planned loop and follow it. An answer nobody can fly from is not
//!   an answer, so the handover is measured too.
//!
//! Both run on one seed. That makes this a check that the demo still behaves, not a claim that it
//! behaves on every seed.

use std::error::Error;

use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::motion::PolylinePath;
use multicalc::scalar::Numeric;

use super::flight_estimator::StateSource;
use super::flight_plant::level_heading;
use super::flight_reference::{CIRCLE_RADIUS, CIRCLE_SPEED, FlightReference};
use super::flight_world::{FlightPhase, FlightWorld, HOVER_POINT, TIMESTEP};

/// The run every gate flies, so a failure is always the same flight.
pub const SEED: u64 = 20260807;

/// How long a circling run is measured over, in seconds, and how much of its start is left out
/// while the body settles onto the circle.
const CIRCLE_SECONDS: f64 = 60.0;
const CIRCLE_SETTLE_SECONDS: f64 = 2.0;

/// How long the machine is given to work out where in the room it is before the run is called off,
/// in seconds. It is well past what it takes; a run that reaches this has not settled at all.
const FINDING_ITSELF_SECONDS: f64 = 40.0;

/// How long the machine is given to climb from the pad onto the path and settle onto it, in
/// seconds, before how well it is following it is read.
const HANDOVER_SECONDS: f64 = 8.0;

/// How many points the planned line is chopped into when working out how far the body is from it.
///
/// The loop is a little over twenty metres round, so this puts a corner every nine centimetres. A
/// straight line drawn across nine centimetres of a curve this gentle sits under half a millimetre
/// inside it, which is nothing against what is being measured.
const PLANNED_LINE_POINTS: usize = 256;

/// One measured quantity and what it had to clear.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    /// Which gate the quantity belongs to.
    pub gate: u32,
    /// The flight it was measured on.
    pub scenario: &'static str,
    /// What was measured.
    pub quantity: &'static str,
    /// What it came out as.
    pub measured: String,
    /// What it had to clear.
    pub bound: &'static str,
    /// Whether it did.
    pub passed: bool,
}

/// Runs both gates and hands back every quantity either of them measured.
///
/// Returns whatever building a world refuses on.
pub fn run_ladder() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut outcomes = Vec::new();
    outcomes.extend(gate_seven()?);
    outcomes.extend(gate_nine()?);
    Ok(outcomes)
}

/// Gate 7: it flies on what it believes. One circle flown twice — once with the controller handed
/// the truth, once with it handed the filter's answer.
///
/// Nothing new is built for this gate and nothing is tuned for it. The two rows are the same flight
/// with one thing changed, so the difference between them is what the estimate costs the flying and
/// cannot be anything else. Tracking is expected to get visibly worse on the filter: the position
/// loop answers being out of place with about seven metres per second squared of push for every
/// metre, so a tenth of a metre of estimate wander becomes a real push in a direction nothing asked
/// for. That is the gate working, not failing.
fn gate_seven() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let on_truth = fly_circle(SEED, StateSource::Truth)?;
    let on_filter = fly_circle(SEED, StateSource::Filter)?;
    let gap = on_filter.radius_error - on_truth.radius_error;
    let rotor_limit = on_truth
        .rotor_limit_fraction
        .max(on_filter.rotor_limit_fraction);

    let scenario = "circle at 2 m/s";
    Ok(vec![
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "radius error, on truth (m)",
            measured: format!("{:.5}", on_truth.radius_error),
            bound: "< 0.10",
            passed: on_truth.radius_error < 0.10,
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "radius error, on filter (m)",
            measured: format!("{:.5}", on_filter.radius_error),
            bound: "< 0.30",
            passed: on_filter.radius_error < 0.30,
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "what the estimate costs (m)",
            measured: format!("{gap:.5}"),
            bound: "recorded",
            passed: gap.is_finite(),
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "speed ripple, on filter (%)",
            measured: format!("{:.3}", 100.0 * on_filter.speed_ripple_fraction),
            bound: "< 20",
            passed: on_filter.speed_ripple_fraction < 0.20,
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "rotors at a limit (%)",
            measured: format!("{:.3}", 100.0 * rotor_limit),
            bound: "< 3",
            passed: rotor_limit < 0.03,
        },
    ])
}

/// What one circling flight measured.
struct CircleRun {
    speed_ripple_fraction: f64,
    radius_error: f64,
    rotor_limit_fraction: f64,
}

/// Flies the circle once and reads how well it was held.
///
/// `state_source` is what the controller is told about where the body is, which is the one thing
/// that turns this from a measurement of the loops into a measurement of the whole machine.
fn fly_circle(seed: u64, state_source: StateSource) -> Result<CircleRun, Box<dyn Error>> {
    let reference = FlightReference::circle(Vector::new(HOVER_POINT), CIRCLE_RADIUS, CIRCLE_SPEED);
    let mut world = FlightWorld::new(seed)?
        .with_reference(reference)
        .with_state_source(state_source);
    let settle_ticks = (CIRCLE_SETTLE_SECONDS / TIMESTEP) as u64;
    let total_ticks = settle_ticks + (CIRCLE_SECONDS / TIMESTEP) as u64;

    let mut measured_ticks = 0u64;
    let mut speed_total = 0.0;
    let mut speed_gap_squares = 0.0;
    let mut radius_squares = 0.0;
    let mut ticks_at_a_limit = 0u64;

    for _ in 0..total_ticks {
        let record = world.step();
        if record.tick <= settle_ticks {
            continue;
        }
        let (off_the_radius, _) = reference
            .offset_from_the_circle(record.pose.translation())
            .unwrap_or((0.0, 0.0));
        let flown = record.velocity.linear().norm();

        measured_ticks += 1;
        speed_total += flown;
        speed_gap_squares += (flown - CIRCLE_SPEED) * (flown - CIRCLE_SPEED);
        radius_squares += off_the_radius * off_the_radius;
        if record.command.at_limit {
            ticks_at_a_limit += 1;
        }
    }

    let counted = measured_ticks.max(1) as f64;
    let mean_speed = speed_total / counted;
    Ok(CircleRun {
        speed_ripple_fraction: if mean_speed > 0.0 {
            (speed_gap_squares / counted).sqrt() / mean_speed
        } else {
            0.0
        },
        radius_error: (radius_squares / counted).sqrt(),
        rotor_limit_fraction: ticks_at_a_limit as f64 / counted,
    })
}

/// Gate 9: it finds itself on a floor plan, and then flies. Set down on the pad knowing roughly
/// where it is and nothing about which way it is pointing, and left to work the rest out.
///
/// The machine hovers and turns on the spot while a cloud of guesses is scored against the plan.
/// Turning is what makes it work: held still it sees the same walls for ever, and a guess pointing
/// the wrong way stays as good as the right one. Nothing here happens once it is flying — the way
/// the guesses are moved assumes a body that hovers and spins, and a banking machine sliding
/// sideways breaks that outright.
fn gate_nine() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let found = fly_finding_itself(SEED)?;

    let scenario = "the hangar";
    Ok(vec![
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "found itself",
            measured: format!("{}", u8::from(found.settled)),
            bound: "1",
            passed: found.settled,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "how far out (m)",
            measured: format!("{:.4}", found.position_error),
            bound: "< 0.15",
            passed: found.position_error < 0.15,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "how far round out (rad)",
            measured: format!("{:.4}", found.heading_error),
            bound: "< 0.03",
            passed: found.heading_error < 0.03,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "time to settle (s)",
            measured: format!("{:.2}", found.seconds),
            bound: "recorded",
            passed: found.seconds.is_finite(),
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "scans matched",
            measured: format!("{}", found.scans_matched),
            bound: "recorded",
            passed: found.scans_matched > 0,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "then onto the path (m)",
            measured: format!("{:.5}", found.off_the_line),
            bound: "< 0.30 RMS",
            passed: found.off_the_line < 0.30,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "scans refused, leaning",
            measured: format!("{}", found.scans_refused),
            bound: "recorded",
            passed: true,
        },
    ])
}

/// What one finding-itself run measured.
struct FindingItselfRun {
    settled: bool,
    position_error: f64,
    heading_error: f64,
    seconds: f64,
    off_the_line: f64,
    scans_matched: u64,
    scans_refused: u64,
}

/// Sets the machine down on the pad, leaves it to work out where in the room it is, and then flies
/// it a lap of the planned loop.
fn fly_finding_itself(seed: u64) -> Result<FindingItselfRun, Box<dyn Error>> {
    let reference = FlightReference::planned_loop(CIRCLE_SPEED)?;
    let mut world = FlightWorld::new(seed)?
        .with_reference(reference)
        .with_state_source(StateSource::Filter)
        .with_startup_localization()?;

    let total_ticks = (FINDING_ITSELF_SECONDS / TIMESTEP) as u64;
    let mut settled_at = None;
    let mut truth = Vector::zeros();
    let mut true_heading = 0.0;
    for _ in 0..total_ticks {
        let record = world.step();
        if record.phase == FlightPhase::Flying {
            truth = record.pose.translation();
            true_heading = level_heading(record.pose.rotation());
            settled_at = Some(record.time);
            break;
        }
    }

    let (guess, _) = match world.localizer() {
        Some(localizer) => localizer.estimate(),
        None => (Vector::zeros(), Matrix::zeros()),
    };

    // Having found itself, it still has to climb off the pad and onto the path. That handover is
    // the other half of this gate: an answer nobody can fly from is not an answer.
    let line = planned_line(&reference)?;
    let settle_ticks = (HANDOVER_SECONDS / TIMESTEP) as u64;
    let flying_ticks = settle_ticks + (reference.lap_seconds() / TIMESTEP) as u64;
    let mut measured_ticks = 0u64;
    let mut off_the_line_squares = 0.0;
    for tick in 0..flying_ticks {
        let record = world.step();
        if tick < settle_ticks {
            continue;
        }
        let off = line
            .closest_point(record.pose.translation())
            .map(|nearest| nearest.distance())
            .unwrap_or(0.0);
        measured_ticks += 1;
        off_the_line_squares += off * off;
    }
    Ok(FindingItselfRun {
        settled: settled_at.is_some(),
        position_error: (guess[0] - truth[0]).hypot(guess[1] - truth[1]),
        heading_error: (guess[2] - true_heading).wrap_to_pi().abs(),
        seconds: settled_at.unwrap_or(FINDING_ITSELF_SECONDS),
        off_the_line: (off_the_line_squares / measured_ticks.max(1) as f64).sqrt(),
        scans_matched: world.scans_matched(),
        scans_refused: world.scans_refused(),
    })
}

/// The planned line chopped into straight pieces, for asking how far the body is from it.
fn planned_line(
    reference: &FlightReference,
) -> Result<PolylinePath<PLANNED_LINE_POINTS, 3, f64>, Box<dyn Error>> {
    let lap = reference.lap_seconds();
    let points: Vec<Vector3D<f64>> = (0..PLANNED_LINE_POINTS)
        .map(|point| {
            reference
                .sample(lap * point as f64 / (PLANNED_LINE_POINTS - 1) as f64)
                .position()
        })
        .collect();
    Ok(PolylinePath::try_from_points(&points)?)
}
