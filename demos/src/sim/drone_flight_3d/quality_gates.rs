//! What `--check` runs: every bound the demo has to clear, measured rather than eyeballed.
//!
//! Each gate is a short headless flight with the numbers read straight off it. A gate is a fault
//! report, not a target — a bound that fails means something is wrong, not that the bound is.

use std::error::Error;
use std::time::Instant;

use multicalc::linear_algebra::{Matrix, Vector, Vector3D};

use super::flight_estimator::StateSource;
use super::flight_reference::{
    CIRCLE_RADIUS, CIRCLE_SPEED, FlightReference, STEP_DISTANCE, STEP_HOLD_SECONDS, lean_for_circle,
};
use super::flight_world::{
    FlightPhase, FlightWorld, HOVER_POINT, SATELLITE_PERIOD_TICKS, TIMESTEP, TURN_RATE_JITTER,
    level_heading,
};
use super::x2_model::GRAVITY_STRENGTH;
use super::x2_model::ROTOR_TONE_HERTZ;
use multicalc::motion::PolylinePath;

/// The run every gate flies, so a failure is always the same flight.
pub const SEED: u64 = 20260807;

/// How long a hovering run lasts, in seconds.
const HOVER_SECONDS: f64 = 60.0;

/// How long the body is given to settle before the steady numbers start being read, in seconds.
const SETTLE_SECONDS: f64 = 10.0;

/// How far below the reference the climbing run starts, in metres.
const CLIMB_START_DEPTH: f64 = 1.0;

/// How long a stepping run lasts, in seconds.
///
/// The cycle jumps six times before it repeats, and the last of those jumps needs a hold of its own
/// to settle in, so a run has to cover seven holds for every step to be measured. Stopping at six
/// would leave the descent — the one step the rotors have the least room for, since they can only
/// push one way — never measured at all.
const STEPPING_SECONDS: f64 = 70.0;

/// How close to the reference the body has to get, and stay, before a step counts as settled, in
/// metres.
const SETTLED_BAND: f64 = 0.05;

/// How long before the next step the steady offset is read over, in seconds.
const STEADY_SECONDS: f64 = 5.0;

/// How far the distance from the reference has to climb back above the closest it has been before
/// that counts as swinging back out, in metres.
///
/// Some number is needed, because a trace read a thousand times a second wanders by a rounding
/// error constantly. A tenth of the settled band is far below anything a real swing would be and
/// far above the wander.
const REBOUND_BAND: f64 = 0.005;

/// How long a circling run is measured over, in seconds, and how much of its start is left out
/// while the body settles onto the circle.
const CIRCLE_SECONDS: f64 = 60.0;
const CIRCLE_SETTLE_SECONDS: f64 = 2.0;

/// The speeds the circle is flown at, and what each run is called.
///
/// The middle one is the circle every later stage comes back to. The other two are there because
/// the lean a circle needs is fixed by the speed and the radius alone: three speeds giving three
/// leans that all match what the arithmetic says is a far stronger statement than one that matches
/// once, and it is the check that catches a tilt conversion or a rotor sharing that is quietly
/// wrong by a scale factor.
const CIRCLE_SPEEDS: [(&str, f64); 3] = [
    ("gate 3 circle at 1 m/s", 1.0),
    ("gate 3 circle at 2 m/s", CIRCLE_SPEED),
    ("gate 3 circle at 3 m/s", 3.0),
];

/// How many seeds the sensing gate flies, so a number that came out well by luck cannot pass.
const SENSING_SEEDS: u64 = 5;

/// How long a sensing run lasts, and how much of its start is left out while the notches settle and
/// the body finds the circle, both in seconds.
const SENSING_SECONDS: f64 = 22.0;
const SENSING_SETTLE_SECONDS: f64 = 2.0;

/// The fastest thing the body itself does, in cycles a second, and how finely the notches' delay is
/// checked across everything up to it.
///
/// The outer loop runs fifty times a second and the inner one settles a lean in about a twentieth
/// of a second, so nothing the machine does of its own accord lives above this. What the notches do
/// to anything faster does not matter, because there is nothing up there but the shake they are
/// placed on.
const BODY_MOTION_HERTZ: f64 = 50.0;
const DELAY_CHECK_STEPS: usize = 50;

/// When dead reckoning is asked how far it has wandered.
struct DriftMark {
    name: &'static str,
    seconds: f64,
}

/// The three moments the drift is written down at: one second in, five, and twenty. Nothing here is
/// a bound — integrating a reading with nothing to check it against goes wrong, and the point is to
/// record how quickly.
const DEAD_RECKONING_MARKS: [DriftMark; 3] = [
    DriftMark {
        name: "dead reckoning at 1 s (m)",
        seconds: 1.0,
    },
    DriftMark {
        name: "dead reckoning at 5 s (m)",
        seconds: 5.0,
    },
    DriftMark {
        name: "dead reckoning at 20 s (m)",
        seconds: 20.0,
    },
];

/// How long the machine is given to climb from the pad onto the path and settle onto it, in
/// seconds, before how well it is following it is read.
const HANDOVER_SECONDS: f64 = 8.0;

/// How long the machine is given to work out where in the room it is before the run is called off,
/// in seconds. It is well past what it takes; a run that reaches this has not settled at all.
const FINDING_ITSELF_SECONDS: f64 = 40.0;

/// How long an estimating run lasts, and how much of its start is left out while the filter works
/// out what the unit's steady offsets are, both in seconds.
const ESTIMATING_SECONDS: f64 = 35.0;
const ESTIMATING_SETTLE_SECONDS: f64 = 5.0;

/// How far apart the filter is checked against the truth, in ticks.
///
/// Far apart on purpose. The band below is where a check lands nineteen times in twenty *when each
/// check is a fresh roll of the dice*, and two checks a hundredth of a second apart are nothing of
/// the sort — the filter's error barely moves in that time, so a thousand of them are one answer
/// written out a thousand times. Checked that often, a filter that is exactly right about itself
/// sits quietly in the middle of the band and scores a hundred per cent, while a badly wrong one
/// sits quietly outside it: the test stops measuring what it is for. A second apart is longer than
/// the filter takes to forget, so the checks stand on their own.
const CONSISTENCY_PERIOD_TICKS: u64 = 1000;

/// The band a check of fifteen numbers against their own claimed spread falls inside nineteen times
/// in twenty when the filter is telling the truth about itself.
///
/// What a check should average is fifteen, one for each number the filter carries, and that average
/// is the more telling of the two: the share in band says whether the filter is roughly honest, the
/// average says which way it is wrong when it is not.
///
/// Landing inside it far too often means the filter is holding a spread wider than its errors
/// deserve and is ignoring readings it should be taking; far too rarely means it is surer of itself
/// than it has earned, which is the failure that ends with a confident wrong answer.
const CONSISTENCY_BAND: [f64; 2] = [6.262, 27.488];

/// How many laps of the planned loop a run flies, after the filter has settled.
const PLANNED_LAPS: f64 = 3.0;

/// How many points the planned line is chopped into when working out how far the body is from it.
///
/// The loop is a little over twenty metres round, so this puts a corner every nine centimetres. A
/// straight line drawn across nine centimetres of a curve this gentle sits under half a millimetre
/// inside it, which is nothing against what is being measured.
const PLANNED_LINE_POINTS: usize = 256;

/// How far the measured mean lean may sit from what the arithmetic says it must be, in radians.
///
/// The air pushing back has to be leaned into as well as the turn, so the flight now sits a little
/// further over than a circle alone would need — a few thousandths of a radian at these speeds.
const LEAN_TOLERANCE: f64 = 0.03;

/// What one measured quantity came out as, and whether it cleared its bound.
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

/// What one hovering flight measured.
struct HoverRun {
    steady_offset: f64,
    drift: f64,
    worst_tilt: f64,
    rotor_limit_fraction: f64,
    mean_math_microseconds: f64,
    stability_proof: bool,
}

/// Runs the whole ladder, from gate 1 up to the last gate built so far.
///
/// Returns whatever building a world refuses on.
pub fn run_ladder() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut outcomes = Vec::new();
    outcomes.extend(gate_one()?);
    outcomes.extend(gate_two()?);
    outcomes.extend(gate_four()?);
    outcomes.extend(gate_five()?);
    outcomes.extend(gate_six()?);
    outcomes.extend(gate_seven()?);
    outcomes.extend(gate_eight()?);
    outcomes.extend(gate_nine()?);
    Ok(outcomes)
}

/// Gate 1: it hovers. Sixty seconds from a standstill on the reference point, then the same again
/// having started a metre below it.
fn gate_one() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let held = fly_hover(0.0)?;
    let climbed = fly_hover(-CLIMB_START_DEPTH)?;
    let mut outcomes = Vec::new();
    for (scenario, run) in [("on the point", held), ("from a metre below", climbed)] {
        outcomes.push(GateOutcome {
            gate: 1,
            scenario,
            quantity: "steady offset (m)",
            measured: format!("{:.5}", run.steady_offset),
            bound: "< 0.01",
            passed: run.steady_offset < 0.01,
        });
        outcomes.push(GateOutcome {
            gate: 1,
            scenario,
            quantity: "drift (m)",
            measured: format!("{:.5}", run.drift),
            bound: "< 0.01",
            passed: run.drift < 0.01,
        });
        outcomes.push(GateOutcome {
            gate: 1,
            scenario,
            quantity: "worst tilt (rad)",
            measured: format!("{:.5}", run.worst_tilt),
            bound: "< 0.02",
            passed: run.worst_tilt < 0.02,
        });
        outcomes.push(GateOutcome {
            gate: 1,
            scenario,
            quantity: "rotors at a limit (%)",
            measured: format!("{:.3}", 100.0 * run.rotor_limit_fraction),
            bound: "0",
            passed: run.rotor_limit_fraction == 0.0,
        });
        outcomes.push(GateOutcome {
            gate: 1,
            scenario,
            quantity: "flight stack cost (us)",
            measured: format!("{:.2}", run.mean_math_microseconds),
            bound: "< 50 mean",
            passed: run.mean_math_microseconds < 50.0,
        });
        outcomes.push(GateOutcome {
            gate: 1,
            scenario,
            quantity: "stability proof",
            measured: if run.stability_proof {
                "found".to_owned()
            } else {
                "missing".to_owned()
            },
            bound: "exists",
            passed: run.stability_proof,
        });
    }
    Ok(outcomes)
}

/// Flies one hovering run, starting `start_height_offset` metres above the reference point.
fn fly_hover(start_height_offset: f64) -> Result<HoverRun, Box<dyn Error>> {
    let mut world =
        FlightWorld::new(SEED)?.with_start_offset(Vector::new([0.0, 0.0, start_height_offset]));
    let total_ticks = (HOVER_SECONDS / TIMESTEP) as u64;
    let settled_tick = (SETTLE_SECONDS / TIMESTEP) as u64;

    let mut steady_error_total = 0.0;
    let mut steady_ticks = 0u64;
    let mut worst_tilt = 0.0_f64;
    let mut error_when_settled = 0.0;
    let mut error_at_the_end = 0.0;

    for _ in 0..total_ticks {
        let record = world.step();
        if record.tick == settled_tick {
            error_when_settled = record.position_error;
        }
        if record.tick == total_ticks {
            error_at_the_end = record.position_error;
        }
        if record.tick > settled_tick {
            steady_error_total += record.position_error;
            steady_ticks += 1;
            worst_tilt = worst_tilt.max(record.tilt);
        }
    }

    let metrics = world.metrics();
    let stability_proof = world
        .controller()
        .stability_certificate()
        .diagonal()
        .iter()
        .all(|entry| entry.is_finite());

    Ok(HoverRun {
        steady_offset: if steady_ticks == 0 {
            0.0
        } else {
            steady_error_total / steady_ticks as f64
        },
        drift: (error_at_the_end - error_when_settled).abs(),
        worst_tilt,
        rotor_limit_fraction: metrics.rotor_limit_fraction(),
        mean_math_microseconds: metrics.mean_math_microseconds(),
        stability_proof,
    })
}

/// Gate 2: it moves and stops. Sixty seconds of a reference that jumps two metres every ten
/// seconds, with every number read per step and the worst one reported.
fn gate_two() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let run = fly_stepping()?;
    let scenario = "two-metre steps";
    // A run this long holds one full window after each of the cycle's six jumps. Reporting the
    // count keeps a windowing fault from passing every bound by measuring nothing at all.
    Ok(vec![
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "steps measured",
            measured: format!("{}", run.steps_measured),
            bound: "6",
            passed: run.steps_measured == 6,
        },
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "overshoot (% of step)",
            measured: format!("{:.2}", 100.0 * run.worst_overshoot_fraction),
            bound: "< 10",
            passed: run.worst_overshoot_fraction < 0.10,
        },
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "settling (s)",
            measured: format!("{:.3}", run.worst_settling_seconds),
            bound: "< 4",
            passed: run.worst_settling_seconds < 4.0,
        },
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "steady offset (m)",
            measured: format!("{:.5}", run.worst_steady_offset),
            bound: "< 0.01",
            passed: run.worst_steady_offset < 0.01,
        },
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "peak lean (rad)",
            measured: format!("{:.4}", run.peak_lean),
            bound: "< 0.5",
            passed: run.peak_lean < 0.5,
        },
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "rotors at a limit (%)",
            measured: format!("{:.3}", 100.0 * run.rotor_limit_fraction),
            bound: "< 1",
            passed: run.rotor_limit_fraction < 0.01,
        },
        GateOutcome {
            gate: 2,
            scenario,
            quantity: "swings back out (count)",
            measured: format!("{}", run.rebounds),
            bound: "0",
            passed: run.rebounds == 0,
        },
    ])
}

/// What one stepping flight measured, worst-case across its steps.
struct SteppingRun {
    steps_measured: usize,
    worst_overshoot_fraction: f64,
    worst_settling_seconds: f64,
    worst_steady_offset: f64,
    peak_lean: f64,
    rotor_limit_fraction: f64,
    rebounds: usize,
}

/// What one tick of a stepping flight looked like, kept so each step can be read back afterwards.
struct SteppingSample {
    position: Vector3D<f64>,
    reference: Vector3D<f64>,
    distance: f64,
}

/// Flies the stepping reference and reads every step out of the trace it leaves.
fn fly_stepping() -> Result<SteppingRun, Box<dyn Error>> {
    let base = Vector::new(HOVER_POINT);
    let mut world = FlightWorld::new(SEED)?.with_reference(FlightReference::stepping(base));
    let total_ticks = (STEPPING_SECONDS / TIMESTEP) as usize;

    let mut trace = Vec::with_capacity(total_ticks);
    let mut peak_lean = 0.0_f64;
    for _ in 0..total_ticks {
        let record = world.step();
        peak_lean = peak_lean.max(record.tilt);
        trace.push(SteppingSample {
            position: record.pose.translation(),
            reference: record.reference.position(),
            distance: record.position_error,
        });
    }

    // Where the steps are is read back off the trace rather than worked out from the clock, so a
    // window can never be one tick out of step with the jump it belongs to. A window runs from the
    // tick the reference moved to the tick before it moves again; the first has no step into it and
    // the last is cut short by the end of the run, so neither is measured.
    let hold_ticks = (STEP_HOLD_SECONDS / TIMESTEP) as usize;
    let mut jumps = Vec::new();
    for index in 1..trace.len() {
        if trace[index].reference != trace[index - 1].reference {
            jumps.push(index);
        }
    }

    let mut worst_overshoot_fraction = 0.0_f64;
    let mut worst_settling_seconds = 0.0_f64;
    let mut worst_steady_offset = 0.0_f64;
    let mut rebounds = 0;
    let mut steps_measured = 0;
    for (which, &start) in jumps.iter().enumerate() {
        let end = jumps.get(which + 1).copied().unwrap_or(trace.len());
        if end - start < hold_ticks {
            continue;
        }
        let response = read_step(&trace[start..end], trace[start - 1].reference);
        worst_overshoot_fraction = worst_overshoot_fraction.max(response.overshoot_fraction);
        worst_settling_seconds = worst_settling_seconds.max(response.settling_seconds);
        worst_steady_offset = worst_steady_offset.max(response.steady_offset);
        rebounds += response.rebounds;
        steps_measured += 1;
    }

    Ok(SteppingRun {
        steps_measured,
        worst_overshoot_fraction,
        worst_settling_seconds,
        worst_steady_offset,
        peak_lean,
        rotor_limit_fraction: world.metrics().rotor_limit_fraction(),
        rebounds,
    })
}

/// How one step went.
struct StepResponse {
    overshoot_fraction: f64,
    settling_seconds: f64,
    steady_offset: f64,
    rebounds: usize,
}

/// Reads one step out of the ticks it covers.
///
/// `window` starts on the tick the reference jumped and runs to the tick before the next jump;
/// `previous_reference` is the point the body was sitting on before it.
fn read_step(window: &[SteppingSample], previous_reference: Vector3D<f64>) -> StepResponse {
    let Some(first) = window.first() else {
        return StepResponse {
            overshoot_fraction: 0.0,
            settling_seconds: 0.0,
            steady_offset: 0.0,
            rebounds: 0,
        };
    };
    let target = first.reference;
    let along_the_step = (target - previous_reference).scale(1.0 / STEP_DISTANCE);

    // How far past the target the body ever got, measured along the way it was asked to travel.
    let mut furthest_past = f64::NEG_INFINITY;
    for sample in window {
        furthest_past = furthest_past.max((sample.position - target).dot(along_the_step));
    }

    // It has settled once it is inside the band and never leaves again.
    let mut settled_index = 0;
    for (index, sample) in window.iter().enumerate() {
        if sample.distance > SETTLED_BAND {
            settled_index = index + 1;
        }
    }

    // Once settled, the distance should only ever shrink. A climb back above the closest it has
    // been is the body swinging out again.
    let mut closest_so_far = f64::INFINITY;
    let mut climbing = false;
    let mut rebounds = 0;
    for sample in &window[settled_index.min(window.len())..] {
        closest_so_far = closest_so_far.min(sample.distance);
        let above = sample.distance > closest_so_far + REBOUND_BAND;
        if above && !climbing {
            rebounds += 1;
        }
        climbing = above;
    }

    let steady_ticks = (STEADY_SECONDS / TIMESTEP) as usize;
    let steady_from = window.len().saturating_sub(steady_ticks);
    let steady = &window[steady_from..];
    let steady_offset = if steady.is_empty() {
        0.0
    } else {
        steady.iter().map(|sample| sample.distance).sum::<f64>() / steady.len() as f64
    };

    StepResponse {
        overshoot_fraction: furthest_past.max(0.0) / STEP_DISTANCE,
        settling_seconds: settled_index as f64 * TIMESTEP,
        steady_offset,
        rebounds,
    }
}

/// Gate 4: it flies a circle with real actuators. Three minutes of circling in all, at three
/// speeds, with everything measured against what a circle flown properly must come out at rather
/// than against a target.
///
/// A circle is the one path whose right answer is known outright, so every number below is a
/// statement about the flight and nothing else: the speed is meant to be flat, the distance from
/// the centre is meant to be the radius, and the lean is meant to be exactly what
/// [`lean_for_circle`] says. Anything left over is the loop or the machine.
///
/// This is the flight gate 3 measured, flown now by a machine whose rotors take time to spin up and
/// whose air pushes back. Those two cost the flight something, so the bounds here are the wider ones
/// that allow for them; what the same flight came out at before they existed is written down beside
/// these numbers in the plan, and the difference between the two is what the actuators cost.
fn gate_four() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut outcomes = Vec::new();
    for (scenario, speed) in CIRCLE_SPEEDS {
        let run = fly_circle(speed, SEED, StateSource::Truth)?;
        let wanted_lean = lean_for_circle(CIRCLE_RADIUS, speed);
        let lean_gap = (run.mean_lean - wanted_lean).abs();

        // The lean is checked at every speed, because it is the closed form the whole stage rests
        // on. The rest of the table belongs to the circle the later stages come back to.
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "mean lean (rad)",
            measured: format!("{:.4} vs {wanted_lean:.4}", run.mean_lean),
            bound: "within 0.03",
            passed: lean_gap < LEAN_TOLERANCE,
        });
        if speed != CIRCLE_SPEED {
            continue;
        }
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "speed ripple (% of mean)",
            measured: format!("{:.3}", 100.0 * run.speed_ripple_fraction),
            bound: "< 5",
            passed: run.speed_ripple_fraction < 0.05,
        });
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "radius error (m)",
            measured: format!("{:.5}", run.radius_error),
            bound: "< 0.10",
            passed: run.radius_error < 0.10,
        });
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "height hold (m)",
            measured: format!("{:.5}", run.height_error),
            bound: "< 0.02",
            passed: run.height_error < 0.02,
        });
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "lean ripple (rad)",
            measured: format!("{:.5}", run.lean_ripple),
            bound: "< 0.02",
            passed: run.lean_ripple < 0.02,
        });
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "rotors at a limit (%)",
            measured: format!("{:.3}", 100.0 * run.rotor_limit_fraction),
            bound: "< 1",
            passed: run.rotor_limit_fraction < 0.01,
        });
        // Not a bound but a check that the spin-up really sits between the command and the machine:
        // a gap of nothing at all means the rotors are being handed straight through, and every
        // number above would then belong to a machine that does not exist.
        outcomes.push(GateOutcome {
            gate: 4,
            scenario,
            quantity: "worst thrust gap (N)",
            measured: format!("{:.4}", run.worst_thrust_gap),
            bound: "recorded, > 0",
            passed: run.worst_thrust_gap > 0.0,
        });
    }
    Ok(outcomes)
}

/// What one circling flight measured, all of it after the settling window.
struct CircleRun {
    speed_ripple_fraction: f64,
    radius_error: f64,
    height_error: f64,
    mean_lean: f64,
    lean_ripple: f64,
    rotor_limit_fraction: f64,
    worst_thrust_gap: f64,
}

/// Flies the circle at `speed` and reads the run out tick by tick.
///
/// `state_source` is what the controller is told about where the body is, which is the one thing
/// that turns this from a measurement of the loops into a measurement of the whole machine.
fn fly_circle(
    speed: f64,
    seed: u64,
    state_source: StateSource,
) -> Result<CircleRun, Box<dyn Error>> {
    let reference = FlightReference::circle(Vector::new(HOVER_POINT), CIRCLE_RADIUS, speed);
    let mut world = FlightWorld::new(seed)?
        .with_reference(reference)
        .with_state_source(state_source);
    let settle_ticks = (CIRCLE_SETTLE_SECONDS / TIMESTEP) as u64;
    let total_ticks = settle_ticks + (CIRCLE_SECONDS / TIMESTEP) as u64;

    let mut measured_ticks = 0u64;
    let mut speed_total = 0.0;
    let mut speed_gap_squares = 0.0;
    let mut radius_squares = 0.0;
    let mut height_squares = 0.0;
    let mut lean_total = 0.0;
    let mut lean_squares = 0.0;
    let mut ticks_at_a_limit = 0u64;
    let mut worst_thrust_gap = 0.0_f64;

    for _ in 0..total_ticks {
        let record = world.step();
        if record.tick <= settle_ticks {
            continue;
        }
        let (off_the_radius, off_the_height) = reference
            .offset_from_the_circle(record.pose.translation())
            .unwrap_or((0.0, 0.0));
        let flown = record.velocity.linear().norm();

        measured_ticks += 1;
        speed_total += flown;
        speed_gap_squares += (flown - speed) * (flown - speed);
        radius_squares += off_the_radius * off_the_radius;
        height_squares += off_the_height * off_the_height;
        lean_total += record.tilt;
        lean_squares += record.tilt * record.tilt;
        worst_thrust_gap = worst_thrust_gap.max(record.rotor_thrust_gap);
        if record.command.at_limit {
            ticks_at_a_limit += 1;
        }
    }

    let counted = measured_ticks.max(1) as f64;
    let mean_speed = speed_total / counted;
    let mean_lean = lean_total / counted;
    // The spread about the mean lean, worked out from the mean of the squares rather than by
    // walking the run twice.
    let lean_ripple = (lean_squares / counted - mean_lean * mean_lean)
        .max(0.0)
        .sqrt();

    Ok(CircleRun {
        speed_ripple_fraction: if mean_speed > 0.0 {
            (speed_gap_squares / counted).sqrt() / mean_speed
        } else {
            0.0
        },
        radius_error: (radius_squares / counted).sqrt(),
        height_error: (height_squares / counted).sqrt(),
        mean_lean,
        lean_ripple,
        rotor_limit_fraction: ticks_at_a_limit as f64 / counted,
        worst_thrust_gap,
    })
}

/// Gate 5: it senses its own motion. The stage 3 circle again, on five seeds, with the controller
/// fed the truth throughout — nothing the unit says reaches the loops, so this measures the sensing
/// on its own before anything depends on it.
///
/// Three of the four numbers are about the notches, because a notch is a trade: it takes the shake
/// out, and what it charges for that is a delay on everything else and whatever it leaves behind.
/// Both halves have to be measured, or a filter that quietly delays the whole reading by five
/// milliseconds passes for a good one right up until it is put in the loop.
fn gate_five() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut worst_attenuation = f64::INFINITY;
    let mut worst_delay_milliseconds = 0.0_f64;
    let mut worst_residual = 0.0_f64;
    let mut drift = [0.0_f64; DEAD_RECKONING_MARKS.len()];

    for offset in 0..SENSING_SEEDS {
        let run = fly_sensing(SEED + offset)?;
        worst_attenuation = worst_attenuation.min(run.attenuation_decibels);
        worst_delay_milliseconds = worst_delay_milliseconds.max(run.worst_delay_milliseconds);
        worst_residual = worst_residual.max(run.worst_residual);
        for (total, seen) in drift.iter_mut().zip(run.drift) {
            *total += seen / SENSING_SEEDS as f64;
        }
    }

    let scenario = "gate 3 circle, 5 seeds";
    let mut outcomes = vec![
        GateOutcome {
            gate: 5,
            scenario,
            quantity: "notch attenuation (dB)",
            measured: format!("{worst_attenuation:.1}"),
            bound: ">= 20",
            passed: worst_attenuation >= 20.0,
        },
        GateOutcome {
            gate: 5,
            scenario,
            quantity: "notch delay (ms)",
            measured: format!("{worst_delay_milliseconds:.4}"),
            bound: "<= 1",
            passed: worst_delay_milliseconds <= 1.0,
        },
        GateOutcome {
            gate: 5,
            scenario,
            quantity: "notches leave (rad/s)",
            measured: format!("{worst_residual:.5}"),
            bound: "<= 1.5x jitter",
            passed: worst_residual <= 1.5 * TURN_RATE_JITTER,
        },
    ];
    for (mark, seen) in DEAD_RECKONING_MARKS.iter().zip(drift) {
        outcomes.push(GateOutcome {
            gate: 5,
            scenario,
            quantity: mark.name,
            measured: format!("{seen:.3}"),
            bound: "recorded",
            passed: seen.is_finite(),
        });
    }
    Ok(outcomes)
}

/// What one sensing flight measured.
struct SensingRun {
    attenuation_decibels: f64,
    worst_delay_milliseconds: f64,
    worst_residual: f64,
    drift: [f64; DEAD_RECKONING_MARKS.len()],
}

/// One tick of a sensing flight, kept so the whole window can be read back afterwards.
struct SensingSample {
    truth_turn_rate: Vector3D<f64>,
    raw_turn_rate: Vector3D<f64>,
    notched_turn_rate: Vector3D<f64>,
    turn_rate_offset: Vector3D<f64>,
    time: f64,
}

/// Flies the circle on one seed and reads the unit, the notches and the dead reckoning out of it.
fn fly_sensing(seed: u64) -> Result<SensingRun, Box<dyn Error>> {
    let reference = FlightReference::circle(Vector::new(HOVER_POINT), CIRCLE_RADIUS, CIRCLE_SPEED);
    let mut world = FlightWorld::new(seed)?
        .with_reference(reference)
        .with_state_source(StateSource::Truth);
    let settle_ticks = (SENSING_SETTLE_SECONDS / TIMESTEP) as u64;
    let total_ticks = (SENSING_SECONDS / TIMESTEP) as u64;

    let mut window = Vec::with_capacity((total_ticks - settle_ticks) as usize);
    let mut drift = [0.0; DEAD_RECKONING_MARKS.len()];
    for _ in 0..total_ticks {
        let record = world.step();

        // Dead reckoning is on its own from the first tick, so how far it has wandered is read from
        // the start of the flight rather than from the end of the settling window.
        for (mark, seen) in DEAD_RECKONING_MARKS.iter().zip(drift.iter_mut()) {
            if record.tick == (mark.seconds / TIMESTEP) as u64 {
                *seen = (record.dead_reckoned.position - record.pose.translation()).norm();
            }
        }

        if record.tick > settle_ticks {
            window.push(SensingSample {
                truth_turn_rate: record.truth_inertial.angular_rate,
                raw_turn_rate: record.raw_inertial.angular_rate,
                notched_turn_rate: record.notched_inertial.angular_rate,
                turn_rate_offset: record.inertial_offsets.angular_rate,
                time: record.time,
            });
        }
    }

    // The delay is asked of the notches themselves. A recording of this flight cannot answer it:
    // the body's own turning is so smooth that lining the reading up against the truth a tick at a
    // time matches almost equally well at every lag, and what tips the balance is jitter.
    let mut worst_delay_milliseconds = 0.0_f64;
    for step in 0..=DELAY_CHECK_STEPS {
        let frequency = BODY_MOTION_HERTZ * step as f64 / DELAY_CHECK_STEPS as f64;
        worst_delay_milliseconds =
            worst_delay_milliseconds.max(1000.0 * world.notch_delay_at(frequency));
    }

    Ok(SensingRun {
        attenuation_decibels: notch_attenuation(&window),
        worst_delay_milliseconds,
        worst_residual: worst_residual(&window),
        drift,
    })
}

/// How far down the notches put the rotors' own shake, in decibels.
///
/// The size of the tone is read out of each signal by matching it against a wave of exactly that
/// frequency, which is a thing no amount of jitter can fake: jitter is spread across every
/// frequency, so what it leaves behind in the match shrinks the longer the window is.
fn notch_attenuation(window: &[SensingSample]) -> f64 {
    let raw = tone_amplitude(window, |sample| sample.raw_turn_rate);
    let notched = tone_amplitude(window, |sample| sample.notched_turn_rate);
    if notched <= 0.0 || raw <= 0.0 {
        return f64::INFINITY;
    }
    20.0 * (raw / notched).log10()
}

/// How big the shake at the rotors' turning rate is in one signal, across all three axes.
fn tone_amplitude(window: &[SensingSample], pick: fn(&SensingSample) -> Vector3D<f64>) -> f64 {
    let mut squares = 0.0;
    for axis in 0..3 {
        let mut along_the_wave = 0.0;
        let mut across_it = 0.0;
        for sample in window {
            let turn = std::f64::consts::TAU * ROTOR_TONE_HERTZ * sample.time;
            along_the_wave += pick(sample)[axis] * turn.sin();
            across_it += pick(sample)[axis] * turn.cos();
        }
        let scale = 2.0 / window.len().max(1) as f64;
        squares += (scale * along_the_wave).powi(2) + (scale * across_it).powi(2);
    }
    squares.sqrt()
}

/// What the notches leave behind on the worst of the three axes: everything that is neither the
/// truth nor the unit's own steady offset.
fn worst_residual(window: &[SensingSample]) -> f64 {
    let mut worst = 0.0_f64;
    for axis in 0..3 {
        let mut squares = 0.0;
        for sample in window {
            let left_over = sample.notched_turn_rate[axis]
                - sample.truth_turn_rate[axis]
                - sample.turn_rate_offset[axis];
            squares += left_over * left_over;
        }
        worst = worst.max((squares / window.len().max(1) as f64).sqrt());
    }
    worst
}

/// Gate 6: it works out where it is. The stage 3 circle on five seeds, controller still fed the
/// truth, with everything the filter does measured on its own.
///
/// Keeping the loops on the truth is the whole point of the stage. An estimate wired into the
/// controller and then measured tells you only how the two behave together, and when tracking goes
/// wrong there is no way to say which half caused it. Measured here, before anything depends on it,
/// the estimate's own error is a fact that later stages can be compared against.
fn gate_six() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut worst_position_error = 0.0_f64;
    let mut worst_horizontal = 0.0_f64;
    let mut worst_vertical = 0.0_f64;
    let mut worst_facing_error = 0.0_f64;
    let mut fixes_offered = 0u64;
    let mut fixes_thrown_away = 0u64;
    let mut worst_cost = 0.0_f64;
    let mut checks = 0u64;
    let mut checks_in_band = 0u64;
    let mut check_total = 0.0;
    let mut faults = 0u64;

    for offset in 0..SENSING_SEEDS {
        let run = fly_estimating(SEED + offset)?;
        worst_position_error = worst_position_error.max(run.position_error);
        worst_horizontal = worst_horizontal.max(run.horizontal_error);
        worst_vertical = worst_vertical.max(run.vertical_error);
        worst_facing_error = worst_facing_error.max(run.facing_error);
        fixes_offered += run.fixes_offered;
        fixes_thrown_away += run.fixes_thrown_away;
        worst_cost = worst_cost.max(run.mean_math_microseconds);
        checks += run.checks;
        checks_in_band += run.checks_in_band;
        check_total += run.check_total;
        faults += run.filter_faults;
    }

    // Thrown-away fixes are counted across all five flights rather than seed by seed. Everything
    // else here is an average over thirty seconds and settles to much the same number on every
    // seed; this is a count of something that happens three or four times in a flight, and the
    // worst of five such counts says more about which seed drew the unluckiest handful of fixes
    // than about the filter.
    let thrown_away_share = fixes_thrown_away as f64 / fixes_offered.max(1) as f64;
    // The checks are pooled across the five flights for the same reason: a second apart, one flight
    // holds only thirty of them, and a share worked out from thirty is mostly noise.
    let in_band_share = checks_in_band as f64 / checks.max(1) as f64;
    let scenario = "gate 3 circle, 5 seeds";
    Ok(vec![
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "estimate error (m)",
            measured: format!("{worst_position_error:.4}"),
            bound: "< 0.25 RMS",
            passed: worst_position_error < 0.25,
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "of that, across (m)",
            measured: format!("{worst_horizontal:.4}"),
            bound: "recorded",
            passed: worst_horizontal.is_finite(),
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "of that, up and down (m)",
            measured: format!("{worst_vertical:.4}"),
            bound: "recorded",
            passed: worst_vertical.is_finite(),
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "facing error (rad)",
            measured: format!("{worst_facing_error:.4}"),
            bound: "< 0.03",
            passed: worst_facing_error < 0.03,
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "in its own band (%)",
            measured: format!("{:.1}", 100.0 * in_band_share),
            bound: "70 to 95",
            passed: (0.70..=0.95).contains(&in_band_share),
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "what a check averages",
            measured: format!("{:.2}", check_total / checks.max(1) as f64),
            bound: "recorded, 15 is right",
            passed: check_total.is_finite(),
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "fixes thrown away (%)",
            measured: format!("{:.2}", 100.0 * thrown_away_share),
            bound: "< 2",
            passed: thrown_away_share < 0.02,
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "filter refusals (count)",
            measured: format!("{faults}"),
            bound: "0",
            passed: faults == 0,
        },
        GateOutcome {
            gate: 6,
            scenario,
            quantity: "flight stack cost (us)",
            measured: format!("{worst_cost:.2}"),
            bound: "< 100 mean",
            passed: worst_cost < 100.0,
        },
    ])
}

/// What one estimating flight measured.
struct EstimatingRun {
    position_error: f64,
    horizontal_error: f64,
    vertical_error: f64,
    facing_error: f64,
    checks: u64,
    checks_in_band: u64,
    check_total: f64,
    fixes_offered: u64,
    fixes_thrown_away: u64,
    filter_faults: u64,
    mean_math_microseconds: f64,
}

/// Flies the circle on one seed and reads the filter out of it.
fn fly_estimating(seed: u64) -> Result<EstimatingRun, Box<dyn Error>> {
    let reference = FlightReference::circle(Vector::new(HOVER_POINT), CIRCLE_RADIUS, CIRCLE_SPEED);
    let mut world = FlightWorld::new(seed)?
        .with_reference(reference)
        .with_state_source(StateSource::Truth);
    let settle_ticks = (ESTIMATING_SETTLE_SECONDS / TIMESTEP) as u64;
    let total_ticks = (ESTIMATING_SECONDS / TIMESTEP) as u64;

    let mut measured_ticks = 0u64;
    let mut position_squares = 0.0;
    let mut horizontal_squares = 0.0;
    let mut vertical_squares = 0.0;
    let mut facing_squares = 0.0;
    let mut checks = 0u64;
    let mut checks_in_band = 0u64;
    let mut fixes_offered = 0u64;
    let mut fixes_thrown_away = 0u64;
    let mut check_total = 0.0;

    for _ in 0..total_ticks {
        let record = world.step();
        if record.tick <= settle_ticks {
            continue;
        }
        if record.tick.is_multiple_of(SATELLITE_PERIOD_TICKS) {
            fixes_offered += 1;
            if record.fix_thrown_away {
                fixes_thrown_away += 1;
            }
        }

        let truth = record.pose.translation();
        let believed = record.believed.position;
        let across = (believed[0] - truth[0]).hypot(believed[1] - truth[1]);
        let up_and_down = believed[2] - truth[2];
        // The gap between two facings is the turn that takes one to the other, which is a thing
        // subtraction cannot give.
        let facing_error = record
            .believed
            .orientation
            .inverse()
            .compose(record.pose.rotation())
            .log()
            .norm();

        measured_ticks += 1;
        position_squares += (believed - truth).norm().powi(2);
        horizontal_squares += across * across;
        vertical_squares += up_and_down * up_and_down;
        facing_squares += facing_error * facing_error;

        if record.tick.is_multiple_of(CONSISTENCY_PERIOD_TICKS)
            && let Some(check) = world.filter_consistency()
        {
            checks += 1;
            check_total += check;
            if (CONSISTENCY_BAND[0]..=CONSISTENCY_BAND[1]).contains(&check) {
                checks_in_band += 1;
            }
        }
    }

    let counted = measured_ticks.max(1) as f64;
    Ok(EstimatingRun {
        position_error: (position_squares / counted).sqrt(),
        horizontal_error: (horizontal_squares / counted).sqrt(),
        vertical_error: (vertical_squares / counted).sqrt(),
        facing_error: (facing_squares / counted).sqrt(),
        checks,
        checks_in_band,
        check_total,
        fixes_offered,
        fixes_thrown_away,
        filter_faults: world.filter_faults(),
        mean_math_microseconds: world.metrics().mean_math_microseconds(),
    })
}

/// Gate 7: it flies on what it believes. The stage 3 circle on five seeds, flown twice — once with
/// the controller handed the truth, once with it handed the filter's answer.
///
/// Nothing new is built for this gate and nothing is tuned for it. The two rows are the same flight
/// with one thing changed, so the difference between them is what the estimate costs the flying and
/// cannot be anything else. Tracking is expected to get visibly worse here: the position loop
/// answers being out of place with about seven metres per second squared of push for every metre,
/// so a tenth of a metre of estimate wander becomes a real push in a direction nothing asked for.
/// That is the stage working, not failing.
fn gate_seven() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut worst_on_truth = 0.0_f64;
    let mut worst_on_filter = 0.0_f64;
    let mut worst_ripple_on_filter = 0.0_f64;
    let mut worst_gap = 0.0_f64;
    let mut worst_rotor_limit = 0.0_f64;

    for offset in 0..SENSING_SEEDS {
        let on_truth = fly_circle(CIRCLE_SPEED, SEED + offset, StateSource::Truth)?;
        let on_filter = fly_circle(CIRCLE_SPEED, SEED + offset, StateSource::Filter)?;
        worst_on_truth = worst_on_truth.max(on_truth.radius_error);
        worst_on_filter = worst_on_filter.max(on_filter.radius_error);
        worst_ripple_on_filter = worst_ripple_on_filter.max(on_filter.speed_ripple_fraction);
        worst_gap = worst_gap.max(on_filter.radius_error - on_truth.radius_error);
        worst_rotor_limit = worst_rotor_limit
            .max(on_truth.rotor_limit_fraction)
            .max(on_filter.rotor_limit_fraction);
    }

    let scenario = "gate 3 circle, 5 seeds";
    Ok(vec![
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "radius error, on truth (m)",
            measured: format!("{worst_on_truth:.5}"),
            bound: "< 0.10, as gate 4",
            passed: worst_on_truth < 0.10,
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "radius error, on filter (m)",
            measured: format!("{worst_on_filter:.5}"),
            bound: "< 0.30",
            passed: worst_on_filter < 0.30,
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "what the estimate costs (m)",
            measured: format!("{worst_gap:.5}"),
            bound: "recorded",
            passed: worst_gap.is_finite(),
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "speed ripple, on filter (%)",
            measured: format!("{:.3}", 100.0 * worst_ripple_on_filter),
            bound: "< 20",
            passed: worst_ripple_on_filter < 0.20,
        },
        GateOutcome {
            gate: 7,
            scenario,
            quantity: "rotors at a limit (%)",
            measured: format!("{:.3}", 100.0 * worst_rotor_limit),
            bound: "< 3",
            passed: worst_rotor_limit < 0.03,
        },
    ])
}

/// Gate 8: it flies a planned line. Three laps of the planned loop on five seeds, flown on the truth
/// and on the filter, with what the plan itself asks for reported apart from how well it was flown.
///
/// Those two are different quantities and mixing them is how a steady controller gets blamed for a
/// lumpy plan. The plan speeds up and slows down and climbs and dives all by itself; that variation
/// is the first three rows and no flight is involved in measuring it. What the body did about it is
/// everything below.
fn gate_eight() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let planned_at = Instant::now();
    let reference = FlightReference::planned_loop(CIRCLE_SPEED)?;
    let planning_microseconds = planned_at.elapsed().as_nanos() as f64 / 1000.0;

    let plan = read_the_plan(&reference);
    let line = planned_line(&reference)?;

    let mut worst_on_truth = 0.0_f64;
    let mut worst_on_filter = 0.0_f64;
    let mut worst_gap = 0.0_f64;
    let mut worst_cost = 0.0_f64;
    for offset in 0..SENSING_SEEDS {
        let on_truth = fly_planned(&reference, &line, SEED + offset, StateSource::Truth)?;
        let on_filter = fly_planned(&reference, &line, SEED + offset, StateSource::Filter)?;
        worst_on_truth = worst_on_truth.max(on_truth.off_the_line);
        worst_on_filter = worst_on_filter.max(on_filter.off_the_line);
        worst_gap = worst_gap.max(on_filter.off_the_line - on_truth.off_the_line);
        worst_cost = worst_cost
            .max(on_truth.mean_math_microseconds)
            .max(on_filter.mean_math_microseconds);
    }

    let scenario = "planned loop, 5 seeds";
    Ok(vec![
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "the plan's own speed (m/s)",
            measured: format!("{:.3} +/- {:.3}", plan.mean_speed, plan.speed_variation),
            bound: "recorded",
            passed: plan.speed_variation.is_finite(),
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "speed across the wrap (%)",
            measured: format!("{:.4}", 100.0 * plan.wrap_speed_gap),
            bound: "< 1",
            passed: plan.wrap_speed_gap < 0.01,
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "lean the plan demands (rad)",
            measured: format!("{:.4}", plan.worst_lean),
            bound: "< 0.35",
            passed: plan.worst_lean < 0.35,
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "off the line, on truth (m)",
            measured: format!("{worst_on_truth:.5}"),
            bound: "< 0.10 RMS",
            passed: worst_on_truth < 0.10,
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "off the line, on filter (m)",
            measured: format!("{worst_on_filter:.5}"),
            bound: "< 0.30 RMS",
            passed: worst_on_filter < 0.30,
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "what the estimate costs (m)",
            measured: format!("{worst_gap:.5}"),
            bound: "recorded",
            passed: worst_gap.is_finite(),
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "flight stack cost (us)",
            measured: format!("{worst_cost:.2}"),
            bound: "< 100 mean",
            passed: worst_cost < 100.0,
        },
        GateOutcome {
            gate: 8,
            scenario,
            quantity: "planning cost (us)",
            measured: format!("{planning_microseconds:.1}"),
            bound: "recorded",
            passed: planning_microseconds.is_finite(),
        },
    ])
}

/// What the plan asks for, before anything tries to fly it.
struct PlanShape {
    mean_speed: f64,
    speed_variation: f64,
    wrap_speed_gap: f64,
    worst_lean: f64,
}

/// Reads the plan's own behaviour off it, over one lap, with no flight involved.
fn read_the_plan(reference: &FlightReference) -> PlanShape {
    let lap = reference.lap_seconds();
    let steps = (lap / TIMESTEP) as u64;
    let mut speed_total = 0.0;
    let mut speed_squares = 0.0;
    let mut worst_lean = 0.0_f64;
    for step in 0..steps {
        let sample = reference.sample(step as f64 * TIMESTEP);
        let speed = sample.velocity().norm();
        speed_total += speed;
        speed_squares += speed * speed;

        // What the plan asks the body to lean, from the acceleration alone: the push has to hold the
        // machine up and shove it sideways at the same time, and the angle between those two is the
        // lean. Nothing about how well it is flown comes into it.
        let wanted = sample.acceleration();
        let sideways = wanted[0].hypot(wanted[1]);
        worst_lean = worst_lean.max(sideways.atan2(GRAVITY_STRENGTH + wanted[2]));
    }

    let counted = steps.max(1) as f64;
    let mean_speed = speed_total / counted;
    // Either side of the join, where a lap ends and the next one begins.
    let before = reference.sample(lap - f64::EPSILON).velocity().norm();
    let after = reference.sample(0.0).velocity().norm();
    PlanShape {
        mean_speed,
        speed_variation: (speed_squares / counted - mean_speed * mean_speed)
            .max(0.0)
            .sqrt(),
        wrap_speed_gap: if mean_speed > 0.0 {
            (after - before).abs() / mean_speed
        } else {
            0.0
        },
        worst_lean,
    }
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

/// What one flight of the planned loop measured.
struct PlannedRun {
    off_the_line: f64,
    mean_math_microseconds: f64,
}

/// Flies three laps of the planned loop and reads how far off the line the body stayed.
fn fly_planned(
    reference: &FlightReference,
    line: &PolylinePath<PLANNED_LINE_POINTS, 3, f64>,
    seed: u64,
    state_source: StateSource,
) -> Result<PlannedRun, Box<dyn Error>> {
    let mut world = FlightWorld::new(seed)?
        .with_reference(*reference)
        .with_state_source(state_source);
    let settle_ticks = (ESTIMATING_SETTLE_SECONDS / TIMESTEP) as u64;
    let total_ticks = settle_ticks + (PLANNED_LAPS * reference.lap_seconds() / TIMESTEP) as u64;

    let mut measured_ticks = 0u64;
    let mut off_the_line_squares = 0.0;
    for _ in 0..total_ticks {
        let record = world.step();
        if record.tick <= settle_ticks {
            continue;
        }
        let off = line
            .closest_point(record.pose.translation())
            .map(|nearest| nearest.distance())
            .unwrap_or(0.0);
        measured_ticks += 1;
        off_the_line_squares += off * off;
    }

    Ok(PlannedRun {
        off_the_line: (off_the_line_squares / measured_ticks.max(1) as f64).sqrt(),
        mean_math_microseconds: world.metrics().mean_math_microseconds(),
    })
}

/// Gate 9: it finds itself on a floor plan. Five seeds, each set down on the pad knowing roughly
/// where it is and nothing about which way it is pointing, and left to work the rest out.
///
/// The machine hovers and turns on the spot while a cloud of guesses is scored against the plan.
/// Turning is what makes it work: held still it sees the same walls for ever, and a guess pointing
/// the wrong way stays as good as the right one. Nothing here happens once it is flying — the way
/// the guesses are moved assumes a body that hovers and spins, and a banking machine sliding
/// sideways breaks that outright.
fn gate_nine() -> Result<Vec<GateOutcome>, Box<dyn Error>> {
    let mut settled = 0u64;
    let mut worst_position_error = 0.0_f64;
    let mut worst_heading_error = 0.0_f64;
    let mut worst_seconds = 0.0_f64;
    let mut scans_matched = 0u64;
    let mut scans_refused = 0u64;
    let mut worst_off_the_line = 0.0_f64;

    for offset in 0..SENSING_SEEDS {
        let found = fly_finding_itself(SEED + offset)?;
        worst_off_the_line = worst_off_the_line.max(found.off_the_line);
        if found.settled {
            settled += 1;
        }
        worst_position_error = worst_position_error.max(found.position_error);
        worst_heading_error = worst_heading_error.max(found.heading_error);
        worst_seconds = worst_seconds.max(found.seconds);
        scans_matched += found.scans_matched;
        scans_refused += found.scans_refused;
    }

    let scenario = "the hangar, 5 seeds";
    Ok(vec![
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "found itself (of 5)",
            measured: format!("{settled}"),
            bound: "5",
            passed: settled == SENSING_SEEDS,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "how far out (m)",
            measured: format!("{worst_position_error:.4}"),
            bound: "< 0.15",
            passed: worst_position_error < 0.15,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "how far round out (rad)",
            measured: format!("{worst_heading_error:.4}"),
            bound: "< 0.03",
            passed: worst_heading_error < 0.03,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "time to settle (s)",
            measured: format!("{worst_seconds:.2}"),
            bound: "recorded",
            passed: worst_seconds.is_finite(),
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "scans matched",
            measured: format!("{scans_matched}"),
            bound: "recorded",
            passed: scans_matched > 0,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "then onto the path (m)",
            measured: format!("{worst_off_the_line:.5}"),
            bound: "< 0.30 RMS",
            passed: worst_off_the_line < 0.30,
        },
        GateOutcome {
            gate: 9,
            scenario,
            quantity: "scans refused, leaning",
            measured: format!("{scans_refused}"),
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

/// Sets the machine down on the pad and leaves it to work out where in the room it is.
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
        heading_error: wrapped_to_half_turn(guess[2] - true_heading).abs(),
        seconds: settled_at.unwrap_or(FINDING_ITSELF_SECONDS),
        off_the_line: (off_the_line_squares / measured_ticks.max(1) as f64).sqrt(),
        scans_matched: world.scans_matched(),
        scans_refused: world.scans_refused(),
    })
}

/// The same angle brought into the half turn either side of zero.
fn wrapped_to_half_turn(angle: f64) -> f64 {
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
