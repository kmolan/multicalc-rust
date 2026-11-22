//! The localized lap: find the robot on the map, then fuse odometry, an IMU, and GPS while driving.
//!
//! The lidar reports on every 1 ms tick here. A real 2D lidar runs at 10–40 Hz; the rate is a
//! property of this simulation, chosen so the whole chain runs every tick, not a claim about hardware.

use std::time::Instant;

use multicalc::control::FollowTheGap;
use multicalc::estimation::{
    BeamModel, ConstantTurnAndSpeed, DirectMeasurement, ExtendedKalmanFilter, InitialParticleCloud,
    MonteCarloLocalizer, residual_with_wrapped_angles,
};
use multicalc::kinematics::{BodyTwist, Unicycle, WheelRotations};
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::ode::Rk4;
use multicalc::scalar::{Numeric, VectorFn};
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::sim::global_position_sensor::GlobalPositionSensor;
use crate::sim::inertial_measurement_unit::InertialMeasurementUnit;
use crate::sim::lidar::Lidar2d;
use crate::sim::wheeled_vehicle::WheeledVehicle;

use super::lap_track_2d::{LapTrack2D, lap_track_2d};

pub const BEAMS: usize = 61;
const TIMESTEP: f64 = 0.001;
const FOOTPRINT_RADIUS: f64 = 0.17;
const WHEELBASE: f64 = 0.235;
const WHEEL_RADIUS: f64 = 0.036;
const LIDAR_RANGE: f64 = 4.0;
const STARTUP_YAW_RATE: f64 = 0.8; // in-place turn while localizing
const RECOVERY_YAW_RATE: f64 = 1.2; // in-place turn when the planner is boxed in
const MAXIMUM_ACCELERATION: f64 = 1.0; // how fast the commanded speed may rise, m/s²
const MAXIMUM_DECELERATION: f64 = 3.0; // and fall — braking beats driving, as on a real vehicle
/// How long the way ahead must stay shut before the robot gives up and turns on the spot.
///
/// Reading a noisy scan a thousand times a second, the planner reports no way through for a tick or
/// two quite often — a dropped beam, a wall clipping the edge of the arc — and clears again straight
/// away. Reacting to every one of those stops the robot dead several times a second for no reason.
/// Waiting for the report to persist ignores the flickers and still catches a real dead end within a
/// fiftieth of a second, in which the robot travels under a centimetre.
const BLOCKED_TICKS_BEFORE_RECOVERY: u32 = 20;
/// Give up localizing after 3 s and drive anyway.
pub const LOCALIZE_CAP_TICKS: u64 = 3000;
const ODOMETRY_EVERY: u64 = 10; // fold in odometry at 100 Hz
const INERTIAL_EVERY: u64 = 5; // fold in the IMU at 200 Hz
const SLIP_SPEED_FACTOR: f64 = 1.20; // measured speed over-reports by this while slipping
const SPEED_NOISE: f64 = 0.02; // odometry speed jitter, m/s
const YAW_RATE_NOISE: f64 = 0.02; // odometry turn-rate jitter, rad/s
const IMU_HEADING_NOISE: f64 = 0.02; // rad
const IMU_YAW_RATE_NOISE: f64 = 0.01; // rad/s
const IMU_HEADING_BIAS: f64 = 0.01; // rad
const GPS_POSITION_NOISE: f64 = 0.30; // m per axis
const GPS_PERIOD_TICKS: u64 = 50; // one GPS fix every 50 ticks (20 Hz)
const LIDAR_RANGE_NOISE: f64 = 0.03; // m
const LIDAR_DROPOUT: f64 = 0.01;
const BEAM_MATCH_DEVIATION: f64 = 0.10; // localizer scan-match looseness, m
/// How tightly the localizer's position guesses must agree before its fix is used, both axes together.
const LOCALIZED_POSITION_SPREAD: f64 = 0.04;
/// How tightly its heading guesses must agree — the slower of the two to settle, since the robot has
/// to turn before heading means anything.
const LOCALIZED_HEADING_SPREAD: f64 = 0.01;
/// The χ²(2) 95 % band the GPS innovation should fall inside when the filter is honest.
pub const GPS_NIS_LOWER_BOUND: f64 = 0.0506;
pub const GPS_NIS_UPPER_BOUND: f64 = 7.378;

/// Which phase the run is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Localizing,
    Driving,
}

/// One tick: the truth, the fused estimate, the odometry-only foil, and what was measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickRecord {
    pub phase: Phase,
    pub pose: Vector3D,           // truth [x, y, heading]
    pub estimate: Vector<5, f64>, // fused [x, y, heading, speed, turn_rate]
    pub covariance: Matrix<5, 5, f64>,
    pub dead_reckoned: Vector3D, // odometry-only foil
    pub scan: [f64; BEAMS],
    pub gps_fix: Option<[f64; 2]>,
    pub twist: BodyTwist<f64>,
    /// How far the left and right wheel have turned in total, in radians, folded into (-π, π].
    pub wheel_angles: [f64; 2],
    /// Distance from the robot's rim to the nearest wall. Negative means contact.
    pub clearance: f64,
    pub contact: bool,
    pub slipping: bool,
    pub laps: u32,
    /// How consistent the GPS reading was with the filter's belief, when one arrived.
    pub gps_nis: Option<f64>,
    /// Cost of the fusion, planning, and control this tick, in microseconds (Driving only).
    pub math_microseconds: f64,
}

/// Running totals over the driving phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LapMetrics {
    pub driving_ticks: u64,
    pub laps: u32,
    pub contacts: u64,
    pub minimum_clearance: f64,
    /// How many ticks the startup localization took.
    pub localization_ticks: u64,
    pub gps_updates: u64,
    pub gps_nis_in_bounds: u64,
    fused_position_error_sum_squares: f64,
    dead_reckoned_position_error_sum_squares: f64,
    heading_error_sum_squares: f64,
    pub distance_travelled: f64,
}

impl LapMetrics {
    /// Typical distance between the fused position and the truth.
    #[must_use]
    pub fn fused_position_rms_error(&self) -> f64 {
        rms(self.fused_position_error_sum_squares, self.driving_ticks)
    }
    /// Typical distance between the odometry-only foil and the truth.
    #[must_use]
    pub fn dead_reckoned_position_rms_error(&self) -> f64 {
        rms(
            self.dead_reckoned_position_error_sum_squares,
            self.driving_ticks,
        )
    }
    /// Typical heading error of the fused estimate.
    #[must_use]
    pub fn heading_rms_error(&self) -> f64 {
        rms(self.heading_error_sum_squares, self.driving_ticks)
    }
    /// The share of GPS readings whose consistency check fell inside the χ²(2) band.
    #[must_use]
    pub fn gps_nis_in_bounds_fraction(&self) -> f64 {
        if self.gps_updates == 0 {
            1.0
        } else {
            self.gps_nis_in_bounds as f64 / self.gps_updates as f64
        }
    }
}

#[must_use]
fn rms(sum_of_squares: f64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        (sum_of_squares / count as f64).sqrt()
    }
}

pub struct LapWorld {
    track: LapTrack2D,
    vehicle: WheeledVehicle,
    lidar: Lidar2d<BEAMS>,
    inertial: InertialMeasurementUnit,
    gps: GlobalPositionSensor,
    follower: FollowTheGap<BEAMS, f64>,
    localizer: MonteCarloLocalizer<BEAMS>,
    // What each sensor sees of the filter's [x, y, heading, speed, turn rate] state.
    wheel_odometry: DirectMeasurement<5, 2>,
    attitude: DirectMeasurement<5, 2>,
    global_position: DirectMeasurement<5, 2>,
    filter: Option<ExtendedKalmanFilter<5, 2>>,
    phase: Phase,
    rng: Pcg32,
    pose: Vector3D,
    dead_reckoned: Vector3D,
    twist: BodyTwist<f64>,
    tick: u64,
    wheel_angles: [f64; 2], // how far each wheel has turned, for the display
    blocked_ticks: u32,     // consecutive ticks the planner has reported no way through
    winding: f64,           // total angle wound around the island, for lap counting
    last_bearing: f64,      // last angle from the island to the robot
    metrics: LapMetrics,
}

impl LapWorld {
    pub fn new(seed: u64) -> Result<Self, Box<dyn std::error::Error>> {
        let field_of_view = 2.0 * std::f64::consts::PI / 3.0;
        let track = lap_track_2d()?;
        let vehicle = WheeledVehicle::new(
            WHEEL_RADIUS,
            WHEELBASE,
            SPEED_NOISE,
            YAW_RATE_NOISE,
            SLIP_SPEED_FACTOR,
        )?;
        let lidar =
            Lidar2d::<BEAMS>::new(field_of_view, LIDAR_RANGE, LIDAR_RANGE_NOISE, LIDAR_DROPOUT)?;
        let inertial =
            InertialMeasurementUnit::new(IMU_HEADING_NOISE, IMU_YAW_RATE_NOISE, IMU_HEADING_BIAS);
        let gps = GlobalPositionSensor::new(GPS_POSITION_NOISE);
        // The follower's chassis width is a planning margin, not the physical footprint: inflating
        // it to the chassis plus the clearance bar on each side keeps the planned path off the walls.
        // The speed ramps from a stop at 0.15 m of clear space ahead to cruise at 0.70 m — short
        // distances, because a 0.9 m corridor rarely opens up more than that.
        let follower =
            FollowTheGap::<BEAMS, f64>::try_new(field_of_view, LIDAR_RANGE, 0.60, 0.62, 0.40)?
                .with_frontal_half_angle(0.35)?
                .with_speed_scaling(0.15, 0.70)?;
        let localizer = MonteCarloLocalizer::<BEAMS>::new(
            track.localization_hint,
            InitialParticleCloud::default(),
            BeamModel {
                range_deviation: BEAM_MATCH_DEVIATION,
                ..Default::default()
            },
            seed,
        )?;

        let start = track.start_pose;
        let pose = Vector::new(start);
        let last_bearing =
            (start[1] - track.island_center[1]).atan2(start[0] - track.island_center[0]);

        Ok(LapWorld {
            track,
            vehicle,
            lidar,
            inertial,
            gps,
            follower,
            localizer,
            wheel_odometry: DirectMeasurement::try_new([3, 4])?,
            attitude: DirectMeasurement::try_new([2, 4])?,
            global_position: DirectMeasurement::try_new([0, 1])?,
            filter: None,
            phase: Phase::Localizing,
            rng: Pcg32::seed_from_u64(seed),
            pose,
            dead_reckoned: pose,
            twist: BodyTwist::new(0.0, 0.0),
            tick: 0,
            wheel_angles: [0.0, 0.0],
            blocked_ticks: 0,
            winding: 0.0,
            last_bearing,
            metrics: LapMetrics {
                driving_ticks: 0,
                laps: 0,
                contacts: 0,
                minimum_clearance: f64::INFINITY,
                localization_ticks: 0,
                gps_updates: 0,
                gps_nis_in_bounds: 0,
                fused_position_error_sum_squares: 0.0,
                dead_reckoned_position_error_sum_squares: 0.0,
                heading_error_sum_squares: 0.0,
                distance_travelled: 0.0,
            },
        })
    }

    #[must_use]
    pub fn step(&mut self) -> TickRecord {
        self.tick += 1;
        match self.phase {
            Phase::Localizing => self.step_localizing(),
            Phase::Driving => self.step_driving(),
        }
    }

    #[must_use]
    fn step_localizing(&mut self) -> TickRecord {
        let command = BodyTwist::new(0.0, STARTUP_YAW_RATE);
        let truth = self
            .vehicle
            .step(self.pose, command, TIMESTEP, false, &mut self.rng);
        self.pose = truth.pose;
        self.turn_wheels(truth.wheel_rotations);
        let scan = self.lidar.simulate(
            &self.track.grid,
            [self.pose[0], self.pose[1], self.pose[2]],
            &mut self.rng,
        );
        // On a degenerate-weights error, hold the cloud and keep turning rather than panic.
        let _ = self.localizer.predict(0.0, STARTUP_YAW_RATE * TIMESTEP);
        let _ = self
            .localizer
            .update(&scan, &self.track.grid, &self.lidar.geometry());
        self.metrics.localization_ticks += 1;

        let (loc_pose, loc_cov) = self.localizer.estimate();
        // Wait for both the position and the heading to settle: the position hint is near the truth
        // from the start, but heading is the real unknown and only resolves as the robot turns.
        let resolved = self
            .localizer
            .is_converged(LOCALIZED_POSITION_SPREAD, LOCALIZED_HEADING_SPREAD);
        let converged = resolved || self.metrics.localization_ticks >= LOCALIZE_CAP_TICKS;

        let estimate = Vector::new([loc_pose[0], loc_pose[1], loc_pose[2], 0.0, 0.0]);
        let covariance = Matrix::from_fn(|row, column| {
            if row < 3 && column < 3 {
                loc_cov[(row, column)]
            } else {
                0.0
            }
        });

        if converged {
            let initial_state = estimate;
            let initial_covariance = Matrix::from_fn(|row, column| match (row, column) {
                // Distrust the localizer's overconfident position spread — a straight corridor is
                // ambiguous along its axis — so GPS pulls the estimate onto the truth in the first
                // fixes rather than over tens of seconds.
                (0, 0) | (1, 1) => 0.25,
                (2, 2) => loc_cov[(2, 2)].max(0.02),
                (3, 3) => 0.25,
                (4, 4) => 0.10,
                _ => 0.0,
            });
            let filter = ExtendedKalmanFilter::<5, 2>::new(
                initial_state,
                initial_covariance,
                Matrix::from_diagonal([1e-7, 1e-7, 1e-7, 4e-4, 4e-4]), // process noise, fixed for the run
                Matrix::from_diagonal([1.0, 1.0]), // any 2×2; overwritten before each update
            );
            self.filter = Some(filter);
            self.dead_reckoned = Vector::new([loc_pose[0], loc_pose[1], loc_pose[2]]);
            self.twist = BodyTwist::new(0.0, 0.0);
            self.phase = Phase::Driving;
        }

        let position = [self.pose[0], self.pose[1]];
        let clearance = nearest_wall_distance(&self.track, position) - FOOTPRINT_RADIUS;
        let contact = clearance <= 0.0;
        if contact {
            self.metrics.contacts += 1;
        }

        TickRecord {
            phase: Phase::Localizing,
            pose: self.pose,
            estimate,
            covariance,
            dead_reckoned: self.dead_reckoned,
            scan,
            gps_fix: None,
            twist: command,
            wheel_angles: self.wheel_angles,
            clearance,
            contact,
            slipping: false,
            laps: self.metrics.laps,
            gps_nis: None,
            math_microseconds: 0.0,
        }
    }

    #[must_use]
    fn step_driving(&mut self) -> TickRecord {
        let commanded = self.twist;

        // 1. Truth step under the current command.
        let slipping = self.track.inside_slip_zone([self.pose[0], self.pose[1]]);
        let truth = self
            .vehicle
            .step(self.pose, commanded, TIMESTEP, slipping, &mut self.rng);
        self.pose = truth.pose;
        self.turn_wheels(truth.wheel_rotations);

        // 2. Sensor reads (the simulator, not the library, so outside the timed block).
        let imu_reading = if self.tick.is_multiple_of(INERTIAL_EVERY) {
            Some(
                self.inertial
                    .read(self.pose[2], commanded.angular(), &mut self.rng),
            )
        } else {
            None
        };
        let gps_fix = if self.tick.is_multiple_of(GPS_PERIOD_TICKS) {
            Some(self.gps.read([self.pose[0], self.pose[1]], &mut self.rng))
        } else {
            None
        };
        let scan = self.lidar.simulate(
            &self.track.grid,
            [self.pose[0], self.pose[1], self.pose[2]],
            &mut self.rng,
        );

        // 3. The timed multicalc block: predict, the sensor updates that fired, and the plan.
        let started = Instant::now();
        let mut gps_nis = None;
        let tick = self.tick;
        let measured_speed = truth.measured_speed;
        let measured_yaw_rate = truth.measured_yaw_rate;
        if let Some(filter) = self.filter.as_mut() {
            let _ = filter.predict(&ConstantTurnAndSpeed { timestep: TIMESTEP });
            rewrap_heading(filter);

            if tick.is_multiple_of(ODOMETRY_EVERY) {
                filter.set_measurement_noise(Matrix::from_diagonal([
                    SPEED_NOISE * SPEED_NOISE,
                    YAW_RATE_NOISE * YAW_RATE_NOISE,
                ]));
                let _ = filter.update(
                    &self.wheel_odometry,
                    Vector::new([measured_speed, measured_yaw_rate]),
                );
            }

            if let Some(reading) = imu_reading {
                filter.set_measurement_noise(Matrix::from_diagonal([
                    IMU_HEADING_NOISE * IMU_HEADING_NOISE,
                    IMU_YAW_RATE_NOISE * IMU_YAW_RATE_NOISE,
                ]));
                let predicted = self.attitude.eval(filter.state().as_array());
                let residual = residual_with_wrapped_angles(
                    Vector::new([reading.heading, reading.yaw_rate]),
                    Vector::new(predicted),
                    &[0],
                );
                let _ = filter.update_with_residual(&self.attitude, residual);
                rewrap_heading(filter);
            }

            if let Some(fix) = gps_fix {
                filter.set_measurement_noise(Matrix::from_diagonal([
                    GPS_POSITION_NOISE * GPS_POSITION_NOISE,
                    GPS_POSITION_NOISE * GPS_POSITION_NOISE,
                ]));
                let _ = filter.update(&self.global_position, Vector::new(fix));
                gps_nis = filter.normalized_innovation_squared().ok();
            }
        }
        let plan = self.follower.compute(&scan, 0.0);
        let math_microseconds = started.elapsed().as_nanos() as f64 / 1000.0;

        // 5. Recovery, then the next command eased in at a rate the drive could really manage.
        let (blocked, planned_twist) = match &plan {
            Ok(output) => (output.is_blocked(), output.body_twist()),
            Err(_) => (true, BodyTwist::new(0.0, 0.0)),
        };
        self.blocked_ticks = if blocked { self.blocked_ticks + 1 } else { 0 };
        let wanted = if self.blocked_ticks >= BLOCKED_TICKS_BEFORE_RECOVERY {
            // Shut for long enough to believe: stop and turn toward the open side.
            BodyTwist::new(0.0, RECOVERY_YAW_RATE * free_side_sign(&scan))
        } else if blocked {
            // A flicker: carry on as before rather than stamping on the brakes.
            commanded
        } else {
            planned_twist
        };
        let command = BodyTwist::new(
            reachable_speed(commanded.linear(), wanted.linear()),
            wanted.angular(),
        );
        self.twist = command;

        // 6. Dead-reckoned foil: integrate the measured odometry with no correction.
        self.dead_reckoned = Rk4::step(
            &Unicycle::new(BodyTwist::new(measured_speed, measured_yaw_rate)).field(),
            0.0,
            &self.dead_reckoned,
            TIMESTEP,
        );

        // 7. Metrics and the record.
        let (estimate, covariance) = match self.filter.as_ref() {
            Some(filter) => (filter.state(), filter.covariance()),
            None => (
                Vector::new([self.pose[0], self.pose[1], self.pose[2], 0.0, 0.0]),
                Matrix::from_diagonal([0.0; 5]),
            ),
        };
        let position = [self.pose[0], self.pose[1]];
        let clearance = nearest_wall_distance(&self.track, position) - FOOTPRINT_RADIUS;
        let contact = clearance <= 0.0;

        let bearing = (position[1] - self.track.island_center[1])
            .atan2(position[0] - self.track.island_center[0]);
        self.winding += (bearing - self.last_bearing).wrap_to_pi();
        self.last_bearing = bearing;
        if self.winding.abs() >= std::f64::consts::TAU {
            self.metrics.laps += 1;
            self.winding -= std::f64::consts::TAU.copysign(self.winding);
        }

        self.metrics.driving_ticks += 1;
        if contact {
            self.metrics.contacts += 1;
        }
        if !self.track.inside_pocket(position) {
            self.metrics.minimum_clearance = self.metrics.minimum_clearance.min(clearance);
        }
        let fused_dx = estimate[0] - self.pose[0];
        let fused_dy = estimate[1] - self.pose[1];
        self.metrics.fused_position_error_sum_squares += fused_dx * fused_dx + fused_dy * fused_dy;
        let dead_dx = self.dead_reckoned[0] - self.pose[0];
        let dead_dy = self.dead_reckoned[1] - self.pose[1];
        self.metrics.dead_reckoned_position_error_sum_squares +=
            dead_dx * dead_dx + dead_dy * dead_dy;
        let heading_error = (estimate[2] - self.pose[2]).wrap_to_pi();
        self.metrics.heading_error_sum_squares += heading_error * heading_error;
        self.metrics.distance_travelled += commanded.linear().abs() * TIMESTEP;
        if let Some(nis) = gps_nis {
            self.metrics.gps_updates += 1;
            if (GPS_NIS_LOWER_BOUND..=GPS_NIS_UPPER_BOUND).contains(&nis) {
                self.metrics.gps_nis_in_bounds += 1;
            }
        }

        TickRecord {
            phase: Phase::Driving,
            pose: self.pose,
            estimate,
            covariance,
            dead_reckoned: self.dead_reckoned,
            scan,
            gps_fix,
            twist: command,
            wheel_angles: self.wheel_angles,
            clearance,
            contact,
            slipping,
            laps: self.metrics.laps,
            gps_nis,
            math_microseconds,
        }
    }

    #[must_use]
    pub fn metrics(&self) -> LapMetrics {
        self.metrics
    }
    #[must_use]
    pub fn track(&self) -> &LapTrack2D {
        &self.track
    }
    #[must_use]
    pub fn follower(&self) -> &FollowTheGap<BEAMS, f64> {
        &self.follower
    }
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }
    #[must_use]
    pub fn localizer(&self) -> &MonteCarloLocalizer<BEAMS> {
        &self.localizer
    }
    #[must_use]
    pub fn vehicle(&self) -> &WheeledVehicle {
        &self.vehicle
    }

    /// Rolls the wheels on by the turn they just made, keeping the angles folded into (-π, π].
    fn turn_wheels(&mut self, rotations: WheelRotations<f64>) {
        self.wheel_angles[0] = (self.wheel_angles[0] + rotations.left()).wrap_to_pi();
        self.wheel_angles[1] = (self.wheel_angles[1] + rotations.right()).wrap_to_pi();
    }
}

/// Folds the filter's heading state back into range after a predict or an angular update.
fn rewrap_heading(filter: &mut ExtendedKalmanFilter<5, 2>) {
    let mut state = filter.state();
    state[2] = state[2].wrap_to_pi();
    filter.set_state(state);
}

/// How much of the wanted speed change the drive can actually deliver in one tick.
///
/// The planner reads a fresh, noisy scan every tick and its speed follows the nearest thing in front,
/// so the raw command jumps around far faster than wheels and motors ever could. Easing toward it —
/// slower when speeding up than when slowing down, as on a real vehicle — is what the machine itself
/// would do, and it leaves the planner untouched.
#[must_use]
fn reachable_speed(previous: f64, wanted: f64) -> f64 {
    let rise = MAXIMUM_ACCELERATION * TIMESTEP;
    let fall = MAXIMUM_DECELERATION * TIMESTEP;
    wanted.clamp(previous - fall, previous + rise)
}

/// +1.0 to turn toward the left half of the scan when it is the more open one, else -1.0.
///
/// Beams run from the rightmost to the leftmost, so the back half of the array is the left side. A
/// no-return reads as the full range, so a dropped beam does not swamp the sum. A tie turns left.
#[must_use]
fn free_side_sign(scan: &[f64; BEAMS]) -> f64 {
    let middle = BEAMS / 2;
    let capped = |range: &f64| range.min(LIDAR_RANGE);
    let right: f64 = scan.iter().take(middle).map(capped).sum();
    let left: f64 = scan.iter().skip(BEAMS - middle).map(capped).sum();
    if left >= right { 1.0 } else { -1.0 }
}

/// The distance from `point` to the nearest wall, read from the track's distance field.
///
/// One interpolated lookup, where a ring of 64 rays used to approximate the same number: the field
/// is the exact Euclidean distance, so it cannot miss a wall that falls between two rays. A point
/// in the outer half-cell rim has no four cells to blend, and reads zero — treated as contact,
/// which is the conservative answer and unreachable on this track anyway.
#[must_use]
fn nearest_wall_distance(track: &LapTrack2D, point: [f64; 2]) -> f64 {
    track.distance_field.distance_at(point).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    // Picking the open half of a scan is private to this module, so its test stays here; everything
    // that goes through the public world is tested in the suite beside `src`.
    #[test]
    fn free_side_sign_reads_the_open_half() {
        let mut scan = [0.2; BEAMS];
        for beam in scan.iter_mut().skip(BEAMS - BEAMS / 2) {
            *beam = LIDAR_RANGE;
        }
        assert_eq!(free_side_sign(&scan), 1.0);

        let mut scan = [0.2; BEAMS];
        for beam in scan.iter_mut().take(BEAMS / 2) {
            *beam = LIDAR_RANGE;
        }
        assert_eq!(free_side_sign(&scan), -1.0);
    }
}
