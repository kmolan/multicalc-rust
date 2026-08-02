//! Light attitude filters: a body's facing from a turn-rate sensor pulled onto a push sensor and a
//! magnetometer, without carrying any spread. Reads a starting facing off a still body from which
//! way is down and which way is north, then flies a tumble with a turn-rate sensor that reads a
//! steady offset it was never told about, and watches both filters find the facing and the offset
//! from readings alone. The two are run side by side on exactly the same readings, so the
//! difference on show is the filters and nothing else.
//!
//! Run with: `cargo run -p multicalc-demos --example attitude_filter`

use multicalc::error::SpatialError;
use multicalc::ode::ExponentialMap;
use multicalc::{CalcError, MadgwickFilter, MahonyFilter, SO3, Vector, Vector3D};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg32;

fn report(label: &str, value: f64, exact: f64, allowed: f64) {
    assert!((value - exact).abs() < allowed, "{label}: |err| too large");
    println!(
        "  {label:<28} = {value:>10.5}   (truth {exact:>10.5}, |err| {:.2e})",
        (value - exact).abs()
    );
}

#[must_use]
fn draw(spread: f64, generator: &mut Pcg32) -> f64 {
    Normal::new(0.0, spread)
        .map(|distribution| distribution.sample(generator))
        .unwrap_or(0.0)
}

fn draw_vector(spread: f64, generator: &mut Pcg32) -> Vector3D<f64> {
    Vector::new([
        draw(spread, generator),
        draw(spread, generator),
        draw(spread, generator),
    ])
}

/// How far one facing is from another, in degrees.
#[must_use]
fn degrees_between(a: SO3<f64>, b: SO3<f64>) -> f64 {
    (a.inverse() * b).log().norm().to_degrees()
}

/// The world the tumble happens in, and what a body's own sensors would read from it.
struct Tumble {
    facing: SO3<f64>,
    gyroscope_bias: Vector3D<f64>,
    upward_in_world: Vector3D<f64>,
    field_in_world: Vector3D<f64>,
    gravity_strength: f64,
    elapsed: f64,
}

impl Tumble {
    #[must_use]
    fn new(facing: SO3<f64>, gyroscope_bias: Vector3D<f64>) -> Self {
        // A field pointing north and 60 degrees into the ground, which is nothing like the level
        // north the filters reference by default. That is the point: it must not lean them over.
        let dip = 60.0_f64.to_radians();
        Tumble {
            facing,
            gyroscope_bias,
            upward_in_world: Vector::new([0.0, 0.0, 1.0]),
            field_in_world: Vector::new([dip.cos(), 0.0, -dip.sin()]),
            gravity_strength: 9.81,
            elapsed: 0.0,
        }
    }

    /// How fast the body is really turning right now, about its own axes.
    fn turn_rate(&self) -> Vector3D<f64> {
        Vector::new([
            0.6 * (0.7 * self.elapsed).sin(),
            -0.4 * (0.5 * self.elapsed).cos(),
            0.9,
        ])
    }

    /// Rolls the body forward one tick and returns what its three sensors read: the turn rate
    /// through the sensor's steady offset, the push it feels, and the field it sits in, all with
    /// jitter on top.
    fn step(
        &mut self,
        timestep: f64,
        generator: &mut Pcg32,
    ) -> (Vector3D<f64>, Vector3D<f64>, Vector3D<f64>) {
        let gyroscope_spread = 0.02;
        let accelerometer_spread = 0.08;
        let magnetometer_spread = 0.01;

        let turn_rate = self.turn_rate();
        let gyroscope_reading =
            turn_rate + self.gyroscope_bias + draw_vector(gyroscope_spread, generator);
        // Not accelerating, so the only push the body feels is the ground holding it up.
        let accelerometer_reading = self
            .facing
            .inverse()
            .act(self.upward_in_world * self.gravity_strength)
            + draw_vector(accelerometer_spread, generator);
        let magnetometer_reading = self.facing.inverse().act(self.field_in_world)
            + draw_vector(magnetometer_spread, generator);

        self.facing = ExponentialMap::attitude_step(self.facing, turn_rate, timestep);
        self.elapsed += timestep;

        (
            gyroscope_reading,
            accelerometer_reading,
            magnetometer_reading,
        )
    }
}

fn main() -> Result<(), CalcError> {
    let timestep = 0.005;
    let run_time = 20.0;
    let gyroscope_bias = Vector::new([0.03, -0.02, 0.015]);

    println!("\nReading the starting facing off a still body");
    // A body turned some known way, standing still. Its push sensor tells it which way is down and
    // its magnetometer roughly which way is north.
    let true_facing = SO3::exp(Vector::new([0.1, -0.2, 0.9]));
    let down_in_world = Vector::new([0.0, 0.0, -1.0]);
    let north_in_world = Vector::new([1.0, 0.0, 0.0]);
    let seeding_spread = 0.01;
    let mut generator = Pcg32::seed_from_u64(20260802);
    let down_in_body =
        true_facing.inverse().act(down_in_world) + draw_vector(seeding_spread, &mut generator);
    let north_in_body =
        true_facing.inverse().act(north_in_world) + draw_vector(seeding_spread, &mut generator);

    // Down and north are not parallel, so this always answers; a pair that were would not.
    let read_off =
        SO3::from_two_direction_pairs(down_in_body, north_in_body, down_in_world, north_in_world)
            .ok_or(CalcError::Spatial(SpatialError::NonFinite))?;
    let exactly_recovered = 0.0;
    report(
        "facing error, radians",
        (read_off.inverse() * true_facing).log().norm(),
        exactly_recovered,
        0.05,
    );
    println!("  two noisy directions and their world counterparts start both filters off");

    println!("\nTwenty seconds of tumbling, both filters on exactly the same readings");
    println!("  the turn-rate sensor reads a steady offset neither filter was told about");
    let mut tumble = Tumble::new(true_facing, gyroscope_bias);
    // Each filter's gains are tuned for this run rather than left at their defaults; how fast a
    // filter should chase its sensors is a property of the vehicle, not of the algorithm.
    let mut mahony = MahonyFilter::new(read_off)
        .with_proportional_gain(1.0)
        .with_integral_gain(0.3);
    let mut madgwick = MadgwickFilter::new(read_off)
        .with_correction_gain(0.3)
        .with_bias_gain(0.05);

    println!(
        "\n  {:>6}  {:>14}  {:>14}  {:>26}  {:>26}",
        "time s",
        "Mahony error°",
        "Madgwick error°",
        "Mahony offset rad/s",
        "Madgwick offset rad/s"
    );
    let step_count = (run_time / timestep) as usize;
    let report_period = (2.0 / timestep) as usize;
    for step in 0..step_count {
        let (gyroscope_reading, accelerometer_reading, magnetometer_reading) =
            tumble.step(timestep, &mut generator);
        mahony.step(
            gyroscope_reading,
            accelerometer_reading,
            Some(magnetometer_reading),
            timestep,
        )?;
        madgwick.step(
            gyroscope_reading,
            accelerometer_reading,
            Some(magnetometer_reading),
            timestep,
        )?;

        if (step + 1) % report_period == 0 {
            let learned_mahony = mahony.gyroscope_bias();
            let learned_madgwick = madgwick.gyroscope_bias();
            println!(
                "  {:>6.1}  {:>14.3}  {:>14.3}  {:>26}  {:>26}",
                (step + 1) as f64 * timestep,
                degrees_between(mahony.orientation(), tumble.facing),
                degrees_between(madgwick.orientation(), tumble.facing),
                format!(
                    "{:>7.4} {:>7.4} {:>7.4}",
                    learned_mahony[0], learned_mahony[1], learned_mahony[2]
                ),
                format!(
                    "{:>7.4} {:>7.4} {:>7.4}",
                    learned_madgwick[0], learned_madgwick[1], learned_madgwick[2]
                ),
            );
        }
    }
    println!(
        "  {:>6}  {:>14}  {:>14}  {:>26}  {:>26}",
        "truth",
        "-",
        "-",
        format!(
            "{:>7.4} {:>7.4} {:>7.4}",
            gyroscope_bias[0], gyroscope_bias[1], gyroscope_bias[2]
        ),
        "same"
    );

    println!("\nWhere the two ended up");
    let exactly_right = 0.0;
    let facing_allowance = 0.05;
    report(
        "Mahony facing error, rad",
        (mahony.orientation().inverse() * tumble.facing)
            .log()
            .norm(),
        exactly_right,
        facing_allowance,
    );
    report(
        "Madgwick facing error, rad",
        (madgwick.orientation().inverse() * tumble.facing)
            .log()
            .norm(),
        exactly_right,
        facing_allowance,
    );

    let offset_allowance = 0.01;
    let learned_mahony = mahony.gyroscope_bias();
    let learned_madgwick = madgwick.gyroscope_bias();
    for (axis, name) in ["x", "y", "z"].iter().enumerate() {
        report(
            &format!("Mahony offset {name}, rad/s"),
            learned_mahony[axis],
            gyroscope_bias[axis],
            offset_allowance,
        );
    }
    for (axis, name) in ["x", "y", "z"].iter().enumerate() {
        report(
            &format!("Madgwick offset {name}, rad/s"),
            learned_madgwick[axis],
            gyroscope_bias[axis],
            offset_allowance,
        );
    }

    println!("\n  Both filters found the facing and the turn-rate sensor's steady offset from");
    println!("  readings that were never told either, carrying nothing but a facing and three");
    println!("  numbers, and the steeply dipping field never leaned them over.");

    Ok(())
}
