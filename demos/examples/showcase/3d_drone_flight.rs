//! 3D drone flight (dynamics + control + estimation showcase). A Skydio X2 read out of its own
//! model file finds itself on a floor plan, then flies a planned loop on its own estimate: it is
//! drawn where it believes it is, and the trail is where it really went. Streams to a Rerun viewer.
//! Run with: cargo run --release -p multicalc-demos --example 3d_drone_flight

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::VecDeque;
use std::error::Error;
use std::time::Instant;

use multicalc::mapping::OccupancyMap;
use multicalc_demos::loop_util::{LatencyRing, Pacer};
use multicalc_demos::sim::drone_flight_3d::{
    CIRCLE_SPEED, FlightPhase, FlightReference, FlightWorld, ROTOR_TONE_HERTZ, SEED, StateSource,
    WARMUP_TICKS, planned_waypoints,
};
use multicalc_demos::visualization::ground_grid;
use multicalc_demos::{RerunSink, Rgba, VizSink};

// Palette, sRGB with alpha.
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // the body
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // the planned line, its corners, the reference marker
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // where a satellite fix was thrown away
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 0xff]; // the trail and the guess cloud, both set apart by shape
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // the ground and the walls

const GEOMETRY_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const PANEL_EVERY: i64 = 1000; // text cadence (1 Hz)

/// How many true positions the trail keeps — a few seconds of them at the geometry cadence.
const TRAIL_MAX: usize = 240;
/// How far the ground grid reaches, and how big one of its cells is, in metres.
const GROUND_HALF_EXTENT: f64 = 6.5;
const GROUND_CELL: f64 = 0.5;
/// How many points the planned line is drawn from — enough that its own corners do not show.
const RING_POINTS: usize = 512;
/// How high up the walls of the floor plan are drawn, in metres — the height the beams sweep at.
const WALL_HEIGHT: f64 = 1.5;
/// How many thrown-away fixes are left on screen before the oldest is dropped.
const REFUSED_FIX_MARKS: usize = 32;

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 3d_drone_flight"
        );
    }
    run_live()
}

/// Runs the demo at a thousand ticks a second, streaming to a viewer.
fn run_live() -> Result<(), Box<dyn Error>> {
    // The whole of the planning, done once and timed where it happens.
    let started_planning = Instant::now();
    let plan = FlightReference::planned_loop(CIRCLE_SPEED)?;
    let planning_microseconds = started_planning.elapsed().as_nanos() as f64 / 1000.0;

    // The demo flies on what it works out for itself, not on the truth. The truth is still simulated
    // and still drawn, but nothing that decides anything is allowed to look at it.
    // It is set down knowing roughly where it is and nothing about which way it faces, and has to
    // work the rest out from the walls before it flies anywhere.
    let mut world = FlightWorld::new(SEED)?
        .with_reference(plan)
        .with_state_source(StateSource::Filter)
        .with_startup_localization()?;
    println!("\nThe machine, as read and transcribed:\n");
    print!("{}", indented(&world.model().report()));
    println!(
        "  the line through {} corners was solved in {planning_microseconds:.0} us, once, and one lap of it takes {:.2} s\n",
        planned_waypoints().len() - 1,
        plan.lap_seconds(),
    );
    let drawn_size = world.model().mesh_size();
    println!(
        "  the machine is drawn as its own shape, {:.3} x {:.3} x {:.3} m once shrunk and stood up",
        drawn_size[0], drawn_size[1], drawn_size[2],
    );
    // The proof the outer loop settles, worked out once when it was built. Its diagonal is the
    // quantity the loop drives down every step, so a finite one is the proof itself.
    println!(
        "  the position loop was proved to settle before it ever flew: the quantity it drives down starts at {:.1}\n",
        world
            .controller()
            .stability_certificate()
            .diagonal()
            .iter()
            .sum::<f64>(),
    );

    let mut viewer = RerunSink::live("multicalc-demos/3d-drone-flight")?;

    // The ground and the circle belong to the whole run rather than to the tick they were logged
    // on. The circle never moves, so it is drawn once and left alone.
    viewer.set_static(true);
    viewer.line_strips3d(
        "world/ground",
        &ground_grid(GROUND_HALF_EXTENT, GROUND_CELL),
        &[CHROME],
        &[0.004],
    )?;
    viewer.line_strips3d(
        "world/plan",
        &[plan.traced_ring(RING_POINTS)],
        &[TARGET],
        &[0.02],
    )?;
    // The corners the line was solved through. The line does not pass through them in a straight
    // way, so seeing both at once is what makes the solve visible.
    let corners: Vec<[f64; 3]> = planned_waypoints()
        .iter()
        .map(|corner| corner.into_array())
        .collect();
    viewer.points3d_styled("world/plan_corners", &corners, &[TARGET], &[0.05])?;
    // The room, as the floor plan has it: one point per wall cell, at the height the beams sweep.
    viewer.points3d_styled(
        "world/hangar",
        &wall_points(world.hangar().grid()),
        &[CHROME],
        &[0.04],
    )?;
    // The machine drawn as itself, out of the shape file that ships with the model. The file is in
    // centimetres and is drawn lying on a different axis, so it is shrunk and stood up on the way
    // in; both numbers come from the model rather than from a guess, and the size they produce is
    // printed above so a wrong one is caught before anything is drawn rather than after.
    viewer.model3d("world/body/shape", &world.model().mesh_path())?;
    viewer.transform3d_scaled(
        "world/body/shape",
        [0.0, 0.0, 0.0],
        world.model().mesh_rotation(),
        world.model().mesh_scale(),
    )?;
    viewer.set_static(false);
    // One plot each. A metre of height and a metre a second of speed are different quantities, so
    // sharing an axis lets whichever happens to be larger set the scale for both.
    viewer.series_style("plots/height", HERO, "height (m)", 2.0)?;
    viewer.series_style("plots/speed", TARGET, "speed (m/s)", 2.0)?;

    let mut pacer = Pacer::new();
    let mut cost_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);
    let mut refused_fixes: VecDeque<[f64; 3]> = VecDeque::with_capacity(REFUSED_FIX_MARKS);

    let mut tick: i64 = 0;
    loop {
        let _ = pacer.wait();
        tick += 1;
        viewer.set_sequence("tick", tick);

        let record = world.step();
        if record.tick > WARMUP_TICKS {
            cost_ring.push(record.math_microseconds);
        }

        let flown_speed = record.velocity.linear().norm();

        viewer.scalar("plots/height", record.pose.translation()[2])?;
        viewer.scalar("plots/speed", flown_speed)?;

        if record.fix_thrown_away {
            if refused_fixes.len() == REFUSED_FIX_MARKS {
                refused_fixes.pop_front();
            }
            refused_fixes.push_back(record.pose.translation().into_array());
        }

        if tick % GEOMETRY_EVERY == 0 {
            // The body is drawn where the machine believes it is, because that is the only place it
            // knows about and every decision it makes is made from there.
            viewer.transform3d(
                "world/body",
                record.believed.position.into_array(),
                record.believed.orientation.quaternion().as_array(),
            )?;
            viewer.points3d_styled(
                "world/reference",
                &[record.reference.position().into_array()],
                &[TARGET],
                &[0.06],
            )?;
            // The cloud of guesses about where in the room it is, which empties itself the moment it
            // settles on one.
            let guesses: Vec<[f64; 3]> = match world.localizer() {
                Some(localizer) if record.phase == FlightPhase::FindingItself => localizer
                    .particles()
                    .iter()
                    .map(|guess| [guess[0], guess[1], WALL_HEIGHT])
                    .collect(),
                _ => Vec::new(),
            };
            viewer.points3d_styled("world/guesses", &guesses, &[ACCENT], &[0.05])?;
            viewer.points3d_styled(
                "world/refused_fixes",
                &refused_fixes.iter().copied().collect::<Vec<_>>(),
                &[ERROR],
                &[0.05],
            )?;

            if trail.len() == TRAIL_MAX {
                trail.pop_front();
            }
            trail.push_back(record.pose.translation().into_array());
            viewer.line_strips3d(
                "world/trail",
                &[trail.iter().copied().collect()],
                &[ACCENT],
                &[0.01],
            )?;
        }

        if tick % PANEL_EVERY == 0 {
            let metrics = world.metrics();
            let cost = match cost_ring.summary() {
                Some(summary) => format!("median {:.1} µs of the 1 ms tick", summary.median),
                None => "warming up".to_owned(),
            };
            let panel = if record.phase == FlightPhase::FindingItself {
                let spread = world
                    .localizer()
                    .map_or(0.0, |localizer| localizer.effective_sample_size());
                format!(
                    "## 3d_drone_flight — multicalc live demo\n\
                     ### finding itself on the floor plan: {} guesses, {spread:.0} still worth having\n\
                     ### hovering and turning on the spot · {} scans matched against the walls",
                    world
                        .localizer()
                        .map_or(0, |localizer| localizer.particle_count()),
                    world.scans_matched(),
                )
            } else {
                // Counted off the path's own clock, which starts at the handover: counting from
                // switch-on would leave the looking-around time in it for the rest of the flight.
                let lap = 1 + (record.reference_time / plan.lap_seconds()).floor() as u64;
                format!(
                    "## 3d_drone_flight — multicalc live demo\n\
                     ### minimum-snap plan · LQR · geometric attitude · 15-number filter · {ROTOR_TONE_HERTZ:.0} Hz notches: {cost}\n\
                     ### lap {lap} · {flown_speed:.2} m/s · {:.1} % at a rotor limit",
                    100.0 * metrics.rotor_limit_fraction(),
                )
            };
            viewer.text("hud/stats", &panel)?;
        }
    }
}

/// The middle of every wall cell of the floor plan, at the height the beams sweep through.
fn wall_points<M: OccupancyMap>(grid: &M) -> Vec<[f64; 3]> {
    let origin = grid.origin();
    let cell = grid.resolution();
    let mut points = Vec::new();
    for row in 0..grid.rows() {
        for column in 0..grid.columns() {
            if grid.is_occupied(row, column) {
                points.push([
                    origin[0] + (column as f64 + 0.5) * cell,
                    origin[1] + (row as f64 + 0.5) * cell,
                    WALL_HEIGHT,
                ]);
            }
        }
    }
    points
}

/// The same text with every line pushed in two spaces, for printing under a heading.
fn indented(text: &str) -> String {
    text.lines()
        .map(|line| format!("  {line}\n"))
        .collect::<String>()
}
