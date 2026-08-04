//! 2D localization and obstacle avoidance (estimation + control showcase): "Know where you are."
//!
//! A differential-drive robot boots not knowing where it is. A particle filter matches its lidar to a
//! known map to find itself, then an extended Kalman filter fuses wheel odometry, an IMU, and GPS to
//! hold a centimetre-level global pose while Follow-the-Gap laps a course of obstacles on lidar alone.
//!
//! The startup localizer runs before the timed loop and is not part of the per-tick cost; the lidar
//! reports every tick, a property of this simulation, not a hardware claim.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example 2d_localization_obstacle_avoidance

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use multicalc::linear_algebra::Matrix;
use multicalc::mapping::OccupancyMap;
use multicalc_demos::loop_util::{LatencyRing, Pacer, commas};
use multicalc_demos::sim::circle_outline;
use multicalc_demos::sim::localization_obstacle_avoidance_2d::{LapWorld, Phase};
use multicalc_demos::{RerunSink, Rgba, VizError, VizSink};

const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // the fused estimate, its ellipse and trail
const TRUTH: Rgba = [0xc9, 0x85, 0x00, 0xff]; // the true robot body and heading
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 0xff]; // the localization cloud and GPS fixes
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // the map walls
const RAY: Rgba = [0x89, 0x87, 0x81, 70]; // lidar rays, faint

// The legend sits off to the right of the course (which spans x 0..6, y 0..4), level with it and
// centred on its height, so it never covers anything the demo is showing. Label text is drawn at a
// fixed size on screen rather than in course units, so it spreads further the more the view is
// zoomed out; the gap to the track is wide enough to keep even long labels off it. The view draws y
// downward, so the top row takes the smallest y and each row below it adds a step.
const LEGEND_X: f64 = 8.5;
const LEGEND_TOP_Y: f64 = 1.25;
const LEGEND_ROW_STEP: f64 = 0.3;

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 300;
const FOOTPRINT_RADIUS: f64 = 0.17;
const LIDAR_RANGE: f64 = 4.0;
const FIELD_OF_VIEW: f64 = 2.0 * PI / 3.0;
const BUDGET_MICROSECONDS: f64 = 1000.0; // the 1 ms tick the per-tick math must fit inside
const TICK_RATE_HERTZ: f64 = 1000.0; // the rate that budget comes from: one tick every millisecond

/// The 2σ ellipse of the estimate's position spread — the top-left 2×2 block of the covariance.
///
/// For [[a, b], [b, c]] the two spreads are (a + c)/2 ± sqrt(((a − c)/2)² + b²), the long axis
/// points at 0.5·atan2(2b, a − c), and the semi-axes are 2·sqrt(spread) for the 2σ contour.
#[must_use]
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
#[must_use]
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
#[must_use]
fn wall_points<M: OccupancyMap>(grid: &M) -> Vec<[f64; 2]> {
    let origin = grid.origin();
    let cell = grid.resolution();
    let mut points = Vec::new();
    for row in 0..grid.rows() {
        for column in 0..grid.columns() {
            if grid.is_occupied(row, column) {
                points.push([
                    origin[0] + (column as f64 + 0.5) * cell,
                    origin[1] + (row as f64 + 0.5) * cell,
                ]);
            }
        }
    }
    points
}

/// The two wheels of the robot: each one's rim, and the tread marks that show it rolling.
///
/// Seen from straight above, a rolling wheel looks perfectly still — nothing about its outline
/// changes as it turns. The tread marks carry the motion instead: they slide along the wheel by
/// exactly the distance its rim has rolled, so the wheels visibly drive the robot forward, and on a
/// turn on the spot the two sets of marks run opposite ways.
#[must_use]
fn wheel_shapes(
    pose: [f64; 3],
    wheelbase: f64,
    wheel_radius: f64,
    wheel_angles: [f64; 2],
) -> Vec<Vec<[f64; 2]>> {
    const TREADS: usize = 3;
    let half_width = 0.4 * wheel_radius;
    let (sin, cos) = pose[2].sin_cos();
    // Forward along the heading, and left across it.
    let forward = [cos, sin];
    let left = [-sin, cos];
    let place = |centre: [f64; 2], along: f64, across: f64| {
        [
            centre[0] + along * forward[0] + across * left[0],
            centre[1] + along * forward[1] + across * left[1],
        ]
    };

    let mut shapes = Vec::with_capacity(2 * (1 + TREADS));
    for (side, angle) in [(1.0, wheel_angles[0]), (-1.0, wheel_angles[1])] {
        let hub = place([pose[0], pose[1]], 0.0, side * 0.5 * wheelbase);
        // The rim, as a rectangle lying along the direction of travel.
        shapes.push(vec![
            place(hub, -wheel_radius, -half_width),
            place(hub, wheel_radius, -half_width),
            place(hub, wheel_radius, half_width),
            place(hub, -wheel_radius, half_width),
            place(hub, -wheel_radius, -half_width),
        ]);
        // Tread marks, spaced evenly and shifted back by how far the rim has rolled.
        let spacing = 2.0 * wheel_radius / TREADS as f64;
        let rolled = (-angle * wheel_radius).rem_euclid(spacing);
        for index in 0..TREADS {
            let along = -wheel_radius + rolled + index as f64 * spacing;
            if along <= wheel_radius {
                shapes.push(vec![
                    place(hub, along, -half_width),
                    place(hub, along, half_width),
                ]);
            }
        }
    }
    shapes
}

/// The direction beam `index` points, from straight ahead, positive to the left.
#[must_use]
fn beam_angle(index: usize, beams: usize) -> f64 {
    -FIELD_OF_VIEW / 2.0 + FIELD_OF_VIEW * index as f64 / (beams - 1) as f64
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 2d_localization_obstacle_avoidance"
        );
    }

    let mut rr = RerunSink::live("multicalc-demos/2d-localization-obstacle-avoidance")?;
    let mut world = LapWorld::new(20260722).expect("the pinned configuration is valid");

    // Statics at tick 0 so they forward-fill across the run.
    rr.set_sequence("tick", 0);
    rr.points2d_styled(
        "world/map",
        &wall_points(&world.track().grid),
        &[CHROME],
        &[0.025],
    )?;

    // A colour key for the scene, so a viewer can read every element without guessing.
    let legend: [(&str, Rgba); 5] = [
        ("true robot · heading · lidar hits", TRUTH),
        ("fused estimate · 2σ ellipse · trail", HERO),
        ("particle cloud · GPS fix", ACCENT),
        ("map walls", CHROME),
        ("lidar rays", RAY),
    ];
    let legend_points: Vec<[f64; 2]> = (0..legend.len())
        .map(|row| [LEGEND_X, LEGEND_TOP_Y + row as f64 * LEGEND_ROW_STEP])
        .collect();
    let legend_colors: Vec<Rgba> = legend.iter().map(|(_, color)| *color).collect();
    let legend_labels: Vec<&str> = legend.iter().map(|(label, _)| *label).collect();
    rr.points2d_labeled(
        "world/legend",
        &legend_points,
        &legend_colors,
        &[0.07],
        &legend_labels,
    )?;

    let mut pacer = Pacer::new();
    let mut math_ring = LatencyRing::new(1024);
    let mut fused_trail: VecDeque<[f64; 2]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut n: i64 = 0;
    let mut missed_deadlines: u64 = 0;
    loop {
        let late_us = pacer.wait();
        n += 1;
        rr.set_sequence("tick", n);
        let record = world.step();

        if record.phase == Phase::Driving {
            // A tick that woke a whole millisecond past its slot missed its deadline; the sleep
            // itself always overshoots by a few tens of microseconds, which does not.
            if late_us as f64 > BUDGET_MICROSECONDS {
                missed_deadlines += 1;
            }
            if n > WARMUP_TICKS {
                math_ring.push(record.math_microseconds);
            }
        }

        let pose = record.pose.into_array();
        let position = [pose[0], pose[1]];

        // Spatial geometry every GEOM_EVERY ticks.
        if n % GEOM_EVERY == 0 {
            // The true robot: its body, the way it faces, and the two wheels carrying it there.
            let drive = world.vehicle().drive();
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
            rr.line_strips2d(
                "world/truth/wheels",
                &wheel_shapes(
                    pose,
                    drive.wheelbase(),
                    drive.wheel_radius(),
                    record.wheel_angles,
                ),
                &[TRUTH],
                &[0.012],
            )?;

            match record.phase {
                Phase::Localizing => {
                    let cloud: Vec<[f64; 2]> = world
                        .localizer()
                        .particles()
                        .iter()
                        .map(|particle| [particle[0], particle[1]])
                        .collect();
                    rr.points2d_styled("world/cloud", &cloud, &[ACCENT], &[0.02])?;
                }
                Phase::Driving => {
                    let fused = [record.estimate[0], record.estimate[1]];
                    if fused_trail.len() == TRAIL_MAX {
                        fused_trail.pop_front();
                    }
                    fused_trail.push_back(fused);

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
            rr.scalar("plots/speed", record.twist.linear())?;
        }

        // Hud every HUD_EVERY ticks.
        if n % HUD_EVERY == 0 {
            let markdown = match record.phase {
                Phase::Localizing => format!(
                    "## 2d_localization_obstacle_avoidance — multicalc live demo\n\
                     ### finding itself on the map: {} guesses, {:.0} still in play",
                    world.localizer().particle_count(),
                    world.localizer().effective_sample_size(),
                ),
                Phase::Driving => {
                    let metrics = world.metrics();
                    let flex = match math_ring.summary() {
                        Some(summary) => format!(
                            "### particle filter localization (noisy lidar) + EKF fusion (noisy odom+imu+gps) + obstacle avoidance controller (noisy lidar): median {:.1} µs · {:.0} Hz · {} missed deadlines in {}",
                            summary.median,
                            TICK_RATE_HERTZ,
                            missed_deadlines,
                            commas(metrics.driving_ticks),
                        ),
                        None => "### warming up".to_string(),
                    };
                    format!(
                        "## 2d_localization_obstacle_avoidance — multicalc live demo\n\
                         {flex}\n\
                         ### {} laps · {:.0} m driven · {:.2} m/s now · {} collisions · closest pass {:.1} cm · found itself in {} scans",
                        metrics.laps,
                        metrics.distance_travelled,
                        record.twist.linear(),
                        metrics.contacts,
                        100.0 * metrics.minimum_clearance,
                        metrics.localization_ticks,
                    )
                }
            };
            rr.text("hud/stats", &markdown)?;
        }
    }
}
