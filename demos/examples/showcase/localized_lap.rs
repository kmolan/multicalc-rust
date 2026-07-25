//! The localized lap (estimation + control showcase): "Know where you are."
//!
//! A differential-drive robot boots not knowing where it is. A particle filter matches its lidar to a
//! known map to find itself, then an extended Kalman filter fuses wheel odometry, an IMU, and GPS to
//! hold a centimetre-level global pose while Follow-the-Gap laps a course of obstacles on lidar alone.
//! A dead-reckoning foil drifts away beside the fused estimate, most visibly through the wheel-slip zone.
//!
//! The startup localizer runs before the timed loop and is not part of the per-tick cost; the lidar
//! reports every tick, a property of this simulation, not a hardware claim.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example localized_lap

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use multicalc::linear_algebra::Matrix;
use multicalc_demos::loop_util::{LatencyRing, Pacer};
use multicalc_demos::sim::localization_obstacle_avoidance_2d::{
    GPS_NIS_LOWER_BOUND, GPS_NIS_UPPER_BOUND, LapWorld, Phase,
};
use multicalc_demos::sim::{OccupancyGrid, circle_outline, wrap_angle};
use multicalc_demos::{RerunSink, Rgba, VizError, VizSink};

const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // the fused estimate, its ellipse and trail
const TRUTH: Rgba = [0xc9, 0x85, 0x00, 0xff]; // the true robot body and heading
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // the dead-reckoning foil
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 0xff]; // the localization cloud and GPS fixes
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // the map walls
const RAY: Rgba = [0x89, 0x87, 0x81, 70]; // lidar rays, faint

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 300;
const FOOTPRINT_RADIUS: f64 = 0.17;
const LIDAR_RANGE: f64 = 4.0;
const FIELD_OF_VIEW: f64 = 2.0 * PI / 3.0;
const BUDGET_MICROSECONDS: f64 = 1000.0; // the 1 ms tick the per-tick math must fit inside

/// The 2σ ellipse of the estimate's position spread — the top-left 2×2 block of the covariance.
///
/// For [[a, b], [b, c]] the two spreads are (a + c)/2 ± sqrt(((a − c)/2)² + b²), the long axis
/// points at 0.5·atan2(2b, a − c), and the semi-axes are 2·sqrt(spread) for the 2σ contour.
fn covariance_ellipse(
    center: [f64; 2],
    covariance: &Matrix<5, 5, f64>,
    segments: usize,
) -> Vec<[f64; 2]> {
    let (a, b, c) = (covariance[(0, 0)], covariance[(0, 1)], covariance[(1, 1)]);
    let mean = 0.5 * (a + c);
    let spread = (0.5 * (a - c)).hypot(b);
    let major = 2.0 * (mean + spread).max(0.0).sqrt();
    let minor = 2.0 * (mean - spread).max(0.0).sqrt();
    let (sin, cos) = (0.5 * (2.0 * b).atan2(a - c)).sin_cos();
    (0..=segments)
        .map(|i| {
            let t = TAU * i as f64 / segments as f64;
            let (ex, ey) = (major * t.cos(), minor * t.sin());
            [
                center[0] + cos * ex - sin * ey,
                center[1] + sin * ex + cos * ey,
            ]
        })
        .collect()
}

/// A short strip from `pose` in its heading direction, to show which way the robot faces.
fn heading_tick(pose: [f64; 3], length: f64) -> Vec<[f64; 2]> {
    vec![
        [pose[0], pose[1]],
        [
            pose[0] + length * pose[2].cos(),
            pose[1] + length * pose[2].sin(),
        ],
    ]
}

/// The centres of every occupied cell, for drawing the map walls as a point cloud.
fn wall_points(grid: &OccupancyGrid) -> Vec<[f64; 2]> {
    let origin = grid.origin();
    let cell = grid.resolution();
    let mut points = Vec::new();
    for row in 0..grid.rows() {
        for column in 0..grid.columns() {
            if grid.is_occupied(column, row) {
                points.push([
                    origin[0] + (column as f64 + 0.5) * cell,
                    origin[1] + (row as f64 + 0.5) * cell,
                ]);
            }
        }
    }
    points
}

/// The direction beam `index` points, from straight ahead, positive to the left.
fn beam_angle(index: usize, beams: usize) -> f64 {
    -FIELD_OF_VIEW / 2.0 + FIELD_OF_VIEW * index as f64 / (beams - 1) as f64
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example localized_lap"
        );
    }

    let mut rr = RerunSink::live("multicalc-demos/localized-lap")?;
    let mut world = LapWorld::new(20260722).expect("the pinned configuration is valid");

    // Statics at tick 0 so they forward-fill across the run.
    rr.set_sequence("tick", 0);
    rr.points2d_styled(
        "world/map",
        &wall_points(&world.track().grid),
        &[CHROME],
        &[0.025],
    )?;
    rr.series_style("plots/tick_us", HERO, "tick math (us)", 1.5)?;
    rr.series_style("plots/jitter_us", CHROME, "schedule jitter (us)", 1.0)?;
    rr.series_style("plots/pos_err_fused", HERO, "fused error (m)", 1.5)?;
    rr.series_style(
        "plots/pos_err_dead_reckoned",
        ERROR,
        "dead-reckoned error (m)",
        1.5,
    )?;
    rr.series_style("plots/gps_nis", ACCENT, "GPS NIS", 1.5)?;
    rr.series_style("plots/gps_nis_lower", ERROR, "chi-square lower", 1.0)?;
    rr.series_style("plots/gps_nis_upper", ERROR, "chi-square upper", 1.0)?;

    let mut pacer = Pacer::new();
    let mut math_ring = LatencyRing::new(1024);
    let mut fused_trail: VecDeque<[f64; 2]> = VecDeque::with_capacity(TRAIL_MAX);
    let mut dead_trail: VecDeque<[f64; 2]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut n: i64 = 0;
    loop {
        let late_us = pacer.wait();
        n += 1;
        rr.set_sequence("tick", n);
        let record = world.step();

        if n > WARMUP_TICKS && record.phase == Phase::Driving {
            math_ring.push(record.math_microseconds);
        }

        let pose = record.pose.into_array();
        let position = [pose[0], pose[1]];

        // Spatial geometry every GEOM_EVERY ticks.
        if n % GEOM_EVERY == 0 {
            match record.phase {
                Phase::Localizing => {
                    let cloud: Vec<[f64; 2]> = world
                        .localizer()
                        .particles()
                        .iter()
                        .map(|particle| [particle[0], particle[1]])
                        .collect();
                    rr.points2d_styled("world/cloud", &cloud, &[ACCENT], &[0.02])?;
                    rr.line_strips2d(
                        "world/truth",
                        &[circle_outline(position, FOOTPRINT_RADIUS, 24)],
                        &[TRUTH],
                        &[0.02],
                    )?;
                    rr.line_strips2d(
                        "world/truth/heading",
                        &[heading_tick(pose, 0.3)],
                        &[TRUTH],
                        &[0.02],
                    )?;
                }
                Phase::Driving => {
                    let fused = [record.estimate[0], record.estimate[1]];
                    let dead = [record.dead_reckoned[0], record.dead_reckoned[1]];
                    if fused_trail.len() == TRAIL_MAX {
                        fused_trail.pop_front();
                    }
                    fused_trail.push_back(fused);
                    if dead_trail.len() == TRAIL_MAX {
                        dead_trail.pop_front();
                    }
                    dead_trail.push_back(dead);

                    // The true robot.
                    rr.line_strips2d(
                        "world/truth",
                        &[circle_outline(position, FOOTPRINT_RADIUS, 24)],
                        &[TRUTH],
                        &[0.02],
                    )?;
                    rr.line_strips2d(
                        "world/truth/heading",
                        &[heading_tick(pose, 0.3)],
                        &[TRUTH],
                        &[0.02],
                    )?;

                    // The fused estimate: a dot, its 2σ ellipse, and its trail.
                    rr.points2d_styled("world/fused", &[fused], &[HERO], &[0.06])?;
                    rr.line_strips2d(
                        "world/fused/ellipse",
                        &[covariance_ellipse(fused, &record.covariance, 32)],
                        &[HERO],
                        &[0.012],
                    )?;
                    rr.line_strips2d(
                        "world/fused/trail",
                        &[fused_trail.iter().copied().collect()],
                        &[HERO],
                        &[0.01],
                    )?;

                    // The dead-reckoning foil: a dot and its trail.
                    rr.points2d_styled("world/dead", &[dead], &[ERROR], &[0.05])?;
                    rr.line_strips2d(
                        "world/dead/trail",
                        &[dead_trail.iter().copied().collect()],
                        &[ERROR],
                        &[0.01],
                    )?;

                    // The most recent GPS fix.
                    if let Some(fix) = record.gps_fix {
                        rr.points2d_styled("world/gps", &[fix], &[ACCENT], &[0.05])?;
                    }

                    // The lidar: a faint ray per beam, and the hit points.
                    let mut rays = Vec::with_capacity(record.scan.len());
                    let mut hits = Vec::new();
                    for (index, &range) in record.scan.iter().enumerate() {
                        let angle = pose[2] + beam_angle(index, record.scan.len());
                        let reach = if range.is_finite() {
                            range
                        } else {
                            LIDAR_RANGE
                        };
                        let end = [
                            position[0] + reach * angle.cos(),
                            position[1] + reach * angle.sin(),
                        ];
                        rays.push(vec![position, end]);
                        if range.is_finite() {
                            hits.push(end);
                        }
                    }
                    rr.line_strips2d("world/lidar/rays", &rays, &[RAY], &[0.003])?;
                    rr.points2d_styled("world/lidar/hits", &hits, &[TRUTH], &[0.02])?;
                }
            }
        }

        // Scalars every tick while driving.
        if record.phase == Phase::Driving {
            let fused_error = (record.estimate[0] - pose[0]).hypot(record.estimate[1] - pose[1]);
            let dead_error =
                (record.dead_reckoned[0] - pose[0]).hypot(record.dead_reckoned[1] - pose[1]);
            let heading_error = wrap_angle(record.estimate[2] - pose[2]).abs();
            rr.scalar("plots/tick_us", record.math_microseconds)?;
            rr.scalar("plots/jitter_us", late_us as f64)?;
            rr.scalar("plots/speed", record.twist.linear())?;
            rr.scalar("plots/laps", f64::from(record.laps))?;
            rr.scalar("plots/pos_err_fused", fused_error)?;
            rr.scalar("plots/pos_err_dead_reckoned", dead_error)?;
            rr.scalar("plots/heading_err", heading_error)?;
            if let Some(nis) = record.gps_nis {
                rr.scalar("plots/gps_nis", nis)?;
                rr.scalar("plots/gps_nis_lower", GPS_NIS_LOWER_BOUND)?;
                rr.scalar("plots/gps_nis_upper", GPS_NIS_UPPER_BOUND)?;
            }
        }

        // Hud every HUD_EVERY ticks.
        if n % HUD_EVERY == 0 {
            let markdown = match record.phase {
                Phase::Localizing => format!(
                    "## localized_lap — multicalc live demo\n\
                     ### localizing: effective sample size {:.0} of {}",
                    world.localizer().effective_sample_size(),
                    world.localizer().particle_count()
                ),
                Phase::Driving => {
                    let metrics = world.metrics();
                    let flex = match math_ring.summary() {
                        Some(summary) => format!(
                            "### localize + fuse(odom/IMU/GPS) + FTG: median {:.1} µs · p99 {:.1} µs of {:.0} µs ({:.1} %)",
                            summary.median,
                            summary.p99,
                            BUDGET_MICROSECONDS,
                            100.0 * summary.p99 / BUDGET_MICROSECONDS,
                        ),
                        None => "### warming up".to_string(),
                    };
                    format!(
                        "## localized_lap — multicalc live demo\n\
                         {flex}\n\
                         ### fused RMS {:.0} mm vs dead-reckoning {:.0} mm · heading {:.2} ° · GPS NIS in χ²(2) bounds {:.0} % of {} · laps {}",
                        1000.0 * metrics.fused_position_rms_error(),
                        1000.0 * metrics.dead_reckoned_position_rms_error(),
                        metrics.heading_rms_error().to_degrees(),
                        100.0 * metrics.gps_nis_in_bounds_fraction(),
                        metrics.gps_updates,
                        metrics.laps,
                    )
                }
            };
            rr.text("hud/stats", &markdown)?;
        }
    }
}
