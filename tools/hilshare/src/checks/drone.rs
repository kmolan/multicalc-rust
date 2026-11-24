//! The Skydio X2 checks: the hover identity of the control loop in `f32`, and the ESKF over the
//! IMU trajectory of the QA fixtures.

use core::hint::black_box;

use multicalc::control::{GeometricAttitudeController, Lqr, thrust_command_from_acceleration};
use multicalc::estimation::{ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn};
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::plant::{MultirotorMixer, RotorCommands, RotorSpin};
use multicalc::scalar::Numeric;
use multicalc::spatial::SO3;

use super::{CheckOutcome, close, failed};
use crate::cycles;

/// The X2 of `third_party/menagerie/skydio_x2/x2.xml`: mass, inertia, rotors.
const MASS: f32 = 1.325;
const GRAVITY: f32 = 9.81;
/// `m·g/4`.
const HOVER_THRUST: f32 = MASS * GRAVITY / 4.0;
const INERTIA: [[f32; 3]; 3] = [
    [0.036_651_698, 0.0, -0.0021],
    [0.0, 0.025_411_698, 0.0],
    [-0.0021, 0.0, 0.060_528],
];
const ROTOR_POSITIONS: [[f32; 3]; 4] = [
    [-0.14, -0.18, 0.05],
    [-0.14, 0.18, 0.05],
    [0.14, 0.18, 0.08],
    [0.14, -0.18, 0.08],
];
const ROTOR_SPINS: [RotorSpin; 4] = [
    RotorSpin::CounterClockwise,
    RotorSpin::Clockwise,
    RotorSpin::CounterClockwise,
    RotorSpin::Clockwise,
];
const TORQUE_PER_THRUST: f32 = 0.0201;
const MAXIMUM_THRUST: f32 = 13.0;
/// The idle thrust of `3d_drone_flight`: a quarter of hover.
const MINIMUM_THRUST: f32 = 0.25 * HOVER_THRUST;
/// The gains of `3d_drone_flight`.
const ATTITUDE_GAIN: f32 = 12.0;
const RATE_GAIN: f32 = 1.2;
/// The position LQR step of `3d_drone_flight`, and its cost.
const POSITION_STEP: f32 = 0.01;
const POSITION_COST: [f32; 6] = [49.0, 49.0, 49.0, 11.0, 11.0, 11.0];
const TICK: f32 = 1e-3;
/// The hover point.
const HOLD: [f32; 3] = [0.0, 0.0, 1.0];
/// `128·ε_f32`, the bound of an `f32` identity.
const IDENTITY_F32: f64 = 128.0 * f32::EPSILON as f64;

fn mixer() -> Option<MultirotorMixer<4, f32>> {
    MultirotorMixer::new(
        ROTOR_POSITIONS.map(Vector::new),
        ROTOR_SPINS,
        TORQUE_PER_THRUST,
        MINIMUM_THRUST,
        MAXIMUM_THRUST,
    )
    .ok()
}

fn attitude_controller() -> Option<GeometricAttitudeController<f32>> {
    GeometricAttitudeController::new(ATTITUDE_GAIN, RATE_GAIN, Matrix::new(INERTIA)).ok()
}

/// Hover: each rotor at `m·g/4`, zero torque, no rotor at a limit.
fn hover_holds(commands: RotorCommands<4, f32>, torque: Vector3D<f32>) -> bool {
    let thrusts = commands.thrusts();
    !commands.saturated()
        && f64::from(torque.norm()) <= IDENTITY_F32
        && (0..4).all(|rotor| {
            close(
                f64::from(thrusts[rotor]),
                f64::from(HOVER_THRUST),
                0.0,
                IDENTITY_F32,
            )
        })
}

/// A tracker in the room: the nominal position.
struct RoomTracker;

impl NominalStateFn<3> for RoomTracker {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.position().as_array()
    }
}

/// Identity: one 1 kHz tick of the X2 at hover in `f32`: ESKF predict and position update,
/// position LQR, thrust command, geometric attitude, mixer. Zero error gives `m·g/4` on each
/// rotor and zero torque.
pub fn drone_control_tick_f32() -> CheckOutcome {
    const NAME: &str = "drone_control_tick_f32";
    let state_transition = Matrix::<6, 6, f32>::from_fn(|row, column| {
        if row == column {
            1.0
        } else if column == row + 3 {
            POSITION_STEP
        } else {
            0.0
        }
    });
    let input_model = Matrix::<6, 3, f32>::from_fn(|row, column| {
        if row == column {
            0.5 * POSITION_STEP * POSITION_STEP
        } else if row == column + 3 {
            POSITION_STEP
        } else {
            0.0
        }
    });
    let (Some(mixer), Some(attitude), Ok(position_controller)) = (
        mixer(),
        attitude_controller(),
        Lqr::new(
            state_transition,
            input_model,
            Matrix::from_diagonal(POSITION_COST),
            Matrix::identity(),
        ),
    ) else {
        return failed(NAME, 0);
    };
    let hold = Vector::new(HOLD);
    let still = Vector::<3, f32>::zeros();
    // The identity holds for any positive covariance and noise.
    let mut filter = ErrorStateKalmanFilter::<3, f32>::new(
        NominalState::new(hold, still, SO3::identity(), still, still),
        Matrix::from_diagonal([0.01; 15]),
        ImuNoise {
            gyroscope_noise_density: 1e-3,
            accelerometer_noise_density: 1e-2,
            gyroscope_bias_random_walk: 1e-5,
            accelerometer_bias_random_walk: 1e-4,
        },
        Matrix::from_diagonal([0.25; 3]),
    );
    let gyroscope_reading = black_box(still);
    let accelerometer_reading = black_box(Vector::new([0.0, 0.0, GRAVITY]));

    let start = cycles::read();
    let mut tick = || {
        filter
            .predict(gyroscope_reading, accelerometer_reading, TICK)
            .ok()?;
        filter.update(&RoomTracker, hold).ok()?;
        let estimate = filter.nominal_state();
        let (position, velocity) = (estimate.position(), estimate.velocity());
        let state = Vector::new([
            position[0],
            position[1],
            position[2],
            velocity[0],
            velocity[1],
            velocity[2],
        ]);
        let target = Vector::new([HOLD[0], HOLD[1], HOLD[2], 0.0, 0.0, 0.0]);
        let acceleration = position_controller.control_tracking(state, target, still);
        let thrust = thrust_command_from_acceleration(acceleration, 0.0, GRAVITY).ok()?;
        let torque = attitude.torque(
            estimate.orientation(),
            gyroscope_reading,
            thrust.attitude(),
            still,
            still,
        );
        Some((
            mixer.rotor_thrusts(thrust.thrust_force(MASS), torque),
            torque,
        ))
    };
    let result = black_box(tick());
    let elapsed = cycles::read().wrapping_sub(start);

    let Some((commands, torque)) = result else {
        return failed(NAME, elapsed);
    };
    CheckOutcome {
        name: NAME,
        passed: hover_holds(commands, torque),
        cycles: elapsed,
        value: f64::from(commands.thrusts()[0]),
    }
}

/// Identity: the 1 kHz part of the X2 loop in `f32`: geometric attitude and mixer at hover give
/// `m·g/4` on each rotor and zero torque.
pub fn attitude_and_mixer_f32() -> CheckOutcome {
    const NAME: &str = "attitude_and_mixer_f32";
    let (Some(mixer), Some(attitude)) = (mixer(), attitude_controller()) else {
        return failed(NAME, 0);
    };
    let level = black_box(SO3::<f32>::identity());
    let still = black_box(Vector::<3, f32>::zeros());

    let start = cycles::read();
    let torque = black_box(attitude.torque(level, still, level, still, still));
    let commands = black_box(mixer.rotor_thrusts(MASS * GRAVITY, torque));
    let elapsed = cycles::read().wrapping_sub(start);

    CheckOutcome {
        name: NAME,
        passed: hover_holds(commands, torque),
        cycles: elapsed,
        value: f64::from(commands.thrusts()[0]),
    }
}
