//! The differential-drive truth model: exact pose propagation and noisy wheel odometry with a slip
//! mode.

use multicalc::error::KinematicsError;
use multicalc::kinematics::{BodyTwist, DifferentialDrive, Unicycle};
use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk4;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

/// A differential-drive wheeled vehicle: it moves the true pose and reports what noisy wheel encoders
/// saw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheeledVehicle {
    wheelbase: f64,
    /// How much the reported forward speed jitters.
    speed_noise: f64,
    /// How much the reported turn rate jitters.
    yaw_rate_noise: f64,
    /// The reported speed is scaled by this while a wheel is slipping.
    slip_speed_factor: f64,
}

/// One truth step: where the vehicle actually went, and what the encoders reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthStep {
    pub pose: Vector<3, f64>,
    /// Forward speed the encoders reported over the step.
    pub measured_speed: f64,
    /// Turn rate the encoders reported over the step.
    pub measured_yaw_rate: f64,
}

impl WheeledVehicle {
    /// Builds a vehicle, rejecting an invalid geometry.
    pub fn new(
        wheelbase: f64,
        speed_noise: f64,
        yaw_rate_noise: f64,
        slip_speed_factor: f64,
    ) -> Result<Self, KinematicsError> {
        // Validate the geometry through the shipped constructor; the wheel radius is unused here
        // because the encoders report body speeds directly, so any positive value is alright.
        DifferentialDrive::new(0.036, wheelbase)?;
        Ok(WheeledVehicle {
            wheelbase,
            speed_noise,
            yaw_rate_noise,
            slip_speed_factor,
        })
    }

    #[must_use]
    pub fn wheelbase(&self) -> f64 {
        self.wheelbase
    }

    /// Advances the true pose by `dt` under `command` and reports the encoder speeds. When
    /// `slipping`, only the reported speed is scaled — the true motion is untouched.
    pub fn step(
        &self,
        pose: Vector<3, f64>,
        command: BodyTwist<f64>,
        dt: f64,
        slipping: bool,
        rng: &mut Pcg32,
    ) -> TruthStep {
        let next = Rk4::step(&Unicycle::new(command).field(), 0.0, &pose, dt);
        let speed_factor = if slipping {
            self.slip_speed_factor
        } else {
            1.0
        };
        TruthStep {
            pose: next,
            measured_speed: command.linear() * speed_factor + noise(self.speed_noise, rng),
            measured_yaw_rate: command.angular() + noise(self.yaw_rate_noise, rng),
        }
    }
}

/// One Gaussian draw, or zero when there is no noise to add.
fn noise(deviation: f64, rng: &mut Pcg32) -> f64 {
    if deviation <= 0.0 {
        return 0.0;
    }
    Normal::new(0.0, deviation)
        .map(|n| n.sample(rng))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rand::SeedableRng;

    const WHEELBASE: f64 = 0.235;

    fn noiseless() -> WheeledVehicle {
        WheeledVehicle::new(WHEELBASE, 0.0, 0.0, 1.2).unwrap()
    }

    #[test]
    fn straight_motion_advances_x_only() {
        let mut rng = Pcg32::seed_from_u64(1);
        let step = noiseless().step(
            Vector::new([0.0, 0.0, 0.0]),
            BodyTwist::new(1.0, 0.0),
            0.1,
            false,
            &mut rng,
        );
        assert!(step.pose[0] > 0.0, "x should advance: {}", step.pose[0]);
        assert!(
            step.pose[1].abs() < 1e-12,
            "y should not move: {}",
            step.pose[1]
        );
        assert!(
            step.pose[2].abs() < 1e-12,
            "heading should not turn: {}",
            step.pose[2]
        );
    }

    #[test]
    fn a_positive_turn_rate_steers_left() {
        let mut rng = Pcg32::seed_from_u64(2);
        let dt = 0.2;
        let yaw_rate = 1.0;
        let step = noiseless().step(
            Vector::new([0.0, 0.0, 0.0]),
            BodyTwist::new(0.5, yaw_rate),
            dt,
            false,
            &mut rng,
        );
        // Heading integrates the turn rate exactly, and turning left moves the vehicle up.
        assert!(
            (step.pose[2] - yaw_rate * dt).abs() < 1e-12,
            "heading: {}",
            step.pose[2]
        );
        assert!(step.pose[1] > 0.0, "should steer left: {}", step.pose[1]);
    }

    #[test]
    fn zero_noise_reports_the_command() {
        let mut rng = Pcg32::seed_from_u64(3);
        let step = noiseless().step(
            Vector::new([0.0, 0.0, 0.0]),
            BodyTwist::new(0.8, -0.3),
            0.1,
            false,
            &mut rng,
        );
        assert_eq!(step.measured_speed, 0.8);
        assert_eq!(step.measured_yaw_rate, -0.3);
    }

    #[test]
    fn slipping_scales_only_the_measured_speed() {
        let pose = Vector::new([0.2, 0.1, 0.3]);
        let command = BodyTwist::new(1.0, 0.4);

        let mut rng = Pcg32::seed_from_u64(4);
        let slipping = noiseless().step(pose, command, 0.1, true, &mut rng);
        let mut rng = Pcg32::seed_from_u64(4);
        let rolling = noiseless().step(pose, command, 0.1, false, &mut rng);

        // The reported speed is scaled up, but the truth and the reported turn rate are untouched.
        assert_eq!(slipping.measured_speed, 1.2);
        assert_eq!(slipping.pose.into_array(), rolling.pose.into_array());
        assert_eq!(slipping.measured_yaw_rate, rolling.measured_yaw_rate);
    }

    #[test]
    fn a_fixed_seed_reproduces_the_step() {
        let vehicle = WheeledVehicle::new(WHEELBASE, 0.05, 0.05, 1.2).unwrap();
        let pose = Vector::new([0.0, 0.0, 0.0]);
        let command = BodyTwist::new(1.0, 0.2);

        let mut first = Pcg32::seed_from_u64(9);
        let mut second = Pcg32::seed_from_u64(9);
        assert_eq!(
            vehicle.step(pose, command, 0.1, false, &mut first),
            vehicle.step(pose, command, 0.1, false, &mut second)
        );
    }
}
