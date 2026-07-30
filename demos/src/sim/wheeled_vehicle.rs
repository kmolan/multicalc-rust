//! The differential-drive truth model: exact pose propagation and noisy wheel odometry with a slip
//! mode.

use multicalc::error::KinematicsError;
use multicalc::kinematics::{BodyTwist, DifferentialDrive, Unicycle, WheelRotations};
use multicalc::linear_algebra::Vector3D;
use multicalc::ode::Rk4;
use rand_pcg::Pcg32;

use super::sensor_noise::gaussian_noise;

/// A differential-drive wheeled vehicle: it moves the true pose, turns its two wheels, and reports
/// what noisy wheel encoders saw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheeledVehicle {
    /// The wheel geometry: how a body motion divides between the two wheels.
    drive: DifferentialDrive<f64>,
    /// How much the reported forward speed jitters.
    speed_noise: f64,
    /// How much the reported turn rate jitters.
    yaw_rate_noise: f64,
    /// The reported speed is scaled by this while a wheel is slipping.
    slip_speed_factor: f64,
}

/// One truth step: where the vehicle actually went, how far each wheel turned, and what the encoders
/// reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthStep {
    pub pose: Vector3D,
    /// How far each wheel turned over the step, in radians.
    pub wheel_rotations: WheelRotations<f64>,
    /// Forward speed the encoders reported over the step.
    pub measured_speed: f64,
    /// Turn rate the encoders reported over the step.
    pub measured_yaw_rate: f64,
}

impl WheeledVehicle {
    /// Builds a vehicle, rejecting an invalid geometry.
    pub fn new(
        wheel_radius: f64,
        wheelbase: f64,
        speed_noise: f64,
        yaw_rate_noise: f64,
        slip_speed_factor: f64,
    ) -> Result<Self, KinematicsError> {
        Ok(WheeledVehicle {
            drive: DifferentialDrive::new(wheel_radius, wheelbase)?,
            speed_noise,
            yaw_rate_noise,
            slip_speed_factor,
        })
    }

    /// The wheel geometry, for anything that needs to draw or reason about the wheels.
    #[must_use]
    pub fn drive(&self) -> DifferentialDrive<f64> {
        self.drive
    }

    #[must_use]
    pub fn wheelbase(&self) -> f64 {
        self.drive.wheelbase()
    }

    /// Advances the true pose by `dt` under `command`, works out how far each wheel turned to do it,
    /// and reports the encoder speeds. When `slipping`, only the reported speed is scaled — the true
    /// motion is untouched.
    pub fn step(
        &self,
        pose: Vector3D,
        command: BodyTwist<f64>,
        dt: f64,
        slipping: bool,
        rng: &mut Pcg32,
    ) -> TruthStep {
        let next = Rk4::step(&Unicycle::new(command).field(), 0.0, &pose, dt);
        // The arc the body traced this step, split into the turn each wheel had to make for it.
        let wheel_rotations = self.drive.inverse_arc(command.integrate_over(dt));
        let speed_factor = if slipping {
            self.slip_speed_factor
        } else {
            1.0
        };
        TruthStep {
            pose: next,
            wheel_rotations,
            measured_speed: command.linear() * speed_factor + gaussian_noise(self.speed_noise, rng),
            measured_yaw_rate: command.angular() + gaussian_noise(self.yaw_rate_noise, rng),
        }
    }
}
