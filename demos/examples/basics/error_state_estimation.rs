//! Error-state estimation: an IMU-driven filter finding a body's place, motion, and facing while
//! working out what its own sensors are getting steadily wrong. Starts by reading the orientation
//! off a still body from which way is down and which way is north, then flies a short tumble on
//! turn-rate readings alone, then folds in a room tracker and a heading aid and watches the
//! sensor offsets converge on the values that were injected.
//!
//! Run with: `cargo run -p multicalc-demos --example error_state_estimation`

use multicalc::error::SpatialError;
use multicalc::{
    CalcError, ErrorStateKalmanFilter, ImuNoise, Matrix, NominalState, NominalStateFn, Numeric,
    SO3, Vector, Vector3D,
};
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

/// A tracker in the room that reports where the body is, and nothing else.
struct RoomTracker;

impl NominalStateFn<3> for RoomTracker {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.position().as_array()
    }
}

/// A heading aid that reports which way the body is pointing about the vertical.
struct HeadingAid;

impl NominalStateFn<1> for HeadingAid {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 1] {
        let (_, _, heading) = state.orientation().quaternion().to_euler_zyx();
        [heading]
    }
}

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

/// What the world does and what the sensors say about it.
struct Flight {
    truth: NominalState<f64>,
    turn_rate: Vector3D<f64>,
    gravity: Vector3D<f64>,
    step: usize,
}

impl Flight {
    fn new(gyroscope_bias: Vector3D<f64>, accelerometer_bias: Vector3D<f64>) -> Self {
        let gravity_strength = 9.81;
        Flight {
            // The truth carries the sensors' real offsets, because those are among the things the
            // filter is trying to work out.
            truth: NominalState::new(
                Vector::zeros(),
                Vector::zeros(),
                SO3::identity(),
                gyroscope_bias,
                accelerometer_bias,
            ),
            turn_rate: Vector::new([0.3, -0.2, 0.5]),
            gravity: Vector::new([0.0, 0.0, -gravity_strength]),
            step: 0,
        }
    }

    /// Rolls the world forward one tick and returns what the two sensors read: the real motion,
    /// seen through their steady offsets, with jitter on top.
    fn step(
        &mut self,
        timestep: f64,
        imu_noise: &ImuNoise<f64>,
        generator: &mut Pcg32,
    ) -> (Vector3D<f64>, Vector3D<f64>) {
        // A slow sway along the world's x axis, so the push is not the same every tick.
        let sway_size = 0.5;
        let sway_rate = 0.5;
        let time = self.step as f64 * timestep;
        let world_push = Vector::new([
            sway_size * (2.0 * core::f64::consts::PI * sway_rate * time).sin(),
            0.0,
            0.0,
        ]);
        let proper_push = self
            .truth
            .orientation()
            .inverse()
            .act(world_push - self.gravity);

        let clean_gyroscope = self.turn_rate + self.truth.gyroscope_bias();
        let clean_accelerometer = proper_push + self.truth.accelerometer_bias();
        self.truth =
            self.truth
                .propagated(clean_gyroscope, clean_accelerometer, timestep, self.gravity);
        self.step += 1;

        (
            clean_gyroscope + draw_vector(imu_noise.gyroscope_noise_density, generator),
            clean_accelerometer + draw_vector(imu_noise.accelerometer_noise_density, generator),
        )
    }
}

fn imu_noise() -> ImuNoise<f64> {
    ImuNoise {
        gyroscope_noise_density: 0.02,
        accelerometer_noise_density: 0.05,
        gyroscope_bias_random_walk: 1e-4,
        accelerometer_bias_random_walk: 1e-3,
    }
}

fn new_filter(measurement_spread: f64) -> ErrorStateKalmanFilter<3, f64> {
    let position_spread = 0.1;
    let velocity_spread = 0.1;
    let tilt_spread = 0.05;
    let gyroscope_bias_spread = 0.05;
    let accelerometer_bias_spread = 0.2;
    let mut starting_spread = [0.0; 15];
    for axis in 0..3 {
        starting_spread[axis] = position_spread * position_spread;
        starting_spread[3 + axis] = velocity_spread * velocity_spread;
        starting_spread[6 + axis] = tilt_spread * tilt_spread;
        starting_spread[9 + axis] = gyroscope_bias_spread * gyroscope_bias_spread;
        starting_spread[12 + axis] = accelerometer_bias_spread * accelerometer_bias_spread;
    }

    // The filter starts believing its sensors are honest, which is exactly the thing it is wrong
    // about.
    ErrorStateKalmanFilter::new(
        NominalState::at_rest(SO3::identity()),
        Matrix::from_diagonal(starting_spread),
        imu_noise(),
        Matrix::from_diagonal([measurement_spread * measurement_spread; 3]),
    )
}

/// Flies the whole run and returns how far the filter's position ended up from the truth, plus the
/// filter itself. With `corrected` false nothing is folded in and the estimate is left to drift.
fn fly(
    corrected: bool,
    run_time: f64,
    timestep: f64,
    gyroscope_bias: Vector3D<f64>,
    accelerometer_bias: Vector3D<f64>,
    generator: &mut Pcg32,
) -> Result<(ErrorStateKalmanFilter<3, f64>, f64), CalcError> {
    let tracker_spread = 0.03;
    let heading_aid_spread = 2.0_f64.to_radians();
    let heading_aid_noise = Matrix::from_diagonal([heading_aid_spread * heading_aid_spread; 1]);

    let settings = imu_noise();
    let mut filter = new_filter(tracker_spread);
    let mut flight = Flight::new(gyroscope_bias, accelerometer_bias);

    // A room tracker at 10 Hz and a heading aid at 5 Hz.
    let tracker_period = (0.1 / timestep) as usize;
    let heading_aid_period = (0.2 / timestep) as usize;
    let step_count = (run_time / timestep) as usize;

    for step in 0..step_count {
        let (gyroscope_reading, accelerometer_reading) =
            flight.step(timestep, &settings, generator);
        filter.predict(gyroscope_reading, accelerometer_reading, timestep)?;

        if !corrected {
            continue;
        }

        if (step + 1) % tracker_period == 0 {
            let fix = flight.truth.position() + draw_vector(tracker_spread, generator);
            filter.update(&RoomTracker, fix)?;
        }

        // The heading is an angle, so the difference is wrapped into a half turn before it goes in.
        if (step + 1) % heading_aid_period == 0 {
            let reading = HeadingAid.eval(&flight.truth)[0] + draw(heading_aid_spread, generator);
            let predicted = HeadingAid.eval(&filter.nominal_state())[0];
            let residual = Vector::new([(reading - predicted).wrap_to_pi()]);
            filter.update_other(&HeadingAid, residual, heading_aid_noise)?;
        }
    }

    let position_error = (filter.nominal_state().position() - flight.truth.position()).norm();
    Ok((filter, position_error))
}

fn main() -> Result<(), CalcError> {
    let timestep = 0.005;
    let run_time = 10.0;
    let gyroscope_bias = Vector::new([0.02, -0.015, 0.01]);
    let accelerometer_bias = Vector::new([0.15, -0.10, 0.05]);

    println!("\nReading the starting orientation off a still body");
    // A body turned some known way, standing still. Its push sensor tells it which way is down and
    // its compass tells it roughly which way is north.
    let true_facing = SO3::exp(Vector::new([0.1, -0.2, 0.9]));
    let down_in_world = Vector::new([0.0, 0.0, -1.0]);
    let north_in_world = Vector::new([1.0, 0.0, 0.0]);
    let down_in_body = true_facing.inverse().act(down_in_world);
    let north_in_body = true_facing.inverse().act(north_in_world);

    // Down and north are not parallel, so this always answers; a pair that were would not.
    let read_off =
        SO3::from_two_direction_pairs(down_in_body, north_in_body, down_in_world, north_in_world)
            .ok_or(CalcError::Spatial(SpatialError::NonFinite))?;
    let recovery_error = (read_off.inverse() * true_facing).log().norm();
    let exactly_recovered = 0.0;
    report(
        "facing error, radians",
        recovery_error,
        exactly_recovered,
        1e-12,
    );
    println!("  two directions and their world counterparts pin the facing with no filter at all");

    println!("\nFlying blind: ten seconds on turn-rate and push readings alone");
    let mut generator = Pcg32::seed_from_u64(20260802);
    let (_, drifted_error) = fly(
        false,
        run_time,
        timestep,
        gyroscope_bias,
        accelerometer_bias,
        &mut generator,
    )?;
    println!("  position error after {run_time:.0} s = {drifted_error:>10.5} m");
    // Dead reckoning has nothing pulling it back, so the offsets integrate into a large drift.
    let must_drift_past = 1.0;
    assert!(
        drifted_error > must_drift_past,
        "dead reckoning should drift well past {must_drift_past} m"
    );

    println!("\nFolding in a room tracker at 10 Hz and a heading aid at 5 Hz");
    let mut generator = Pcg32::seed_from_u64(20260802);
    let (filter, corrected_error) = fly(
        true,
        run_time,
        timestep,
        gyroscope_bias,
        accelerometer_bias,
        &mut generator,
    )?;
    println!("  position error after {run_time:.0} s = {corrected_error:>10.5} m");
    println!("  against {drifted_error:>10.5} m with no corrections at all");
    let allowed_error = 0.1;
    assert!(
        corrected_error < allowed_error,
        "corrected position error should stay under {allowed_error} m"
    );

    println!("\nWhat the filter worked out about its own sensors");
    let learned_gyroscope = filter.nominal_state().gyroscope_bias();
    let learned_accelerometer = filter.nominal_state().accelerometer_bias();
    let gyroscope_allowance = 0.2;
    let accelerometer_allowance = 0.25;
    for (axis, name) in ["x", "y", "z"].iter().enumerate() {
        report(
            &format!("turn-rate offset {name}, rad/s"),
            learned_gyroscope[axis],
            gyroscope_bias[axis],
            gyroscope_allowance * gyroscope_bias[axis].abs(),
        );
    }
    for (axis, name) in ["x", "y"].iter().enumerate() {
        report(
            &format!("push offset {name}, m/s²"),
            learned_accelerometer[axis],
            accelerometer_bias[axis],
            accelerometer_allowance * accelerometer_bias[axis].abs(),
        );
    }
    let vertical = 2;
    println!(
        "  push offset z, m/s²          = {:>10.5}   (truth {:>10.5}, not checked)",
        learned_accelerometer[vertical], accelerometer_bias[vertical]
    );
    println!("  the vertical one is left unchecked on purpose: with the push held steady, a body");
    println!("  pushed a little too hard upward looks exactly like a body tilted a little, and");
    println!("  nothing in this run can tell the two apart");

    Ok(())
}
