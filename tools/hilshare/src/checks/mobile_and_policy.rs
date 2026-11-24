//! The mobile-robot tick in `f32`, and the MLP policy in `f64` and `f32`.

use core::hint::black_box;

use multicalc::control::pure_pursuit_curvature;
use multicalc::estimation::{ConstantTurnAndSpeed, ExtendedKalmanFilter};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::mlp_inference::{Activation, ParameterCursor};
use multicalc::scalar::{Numeric, Primal, VectorFn};
use multicalc::spatial::{SE2, SO2};

use super::{CheckOutcome, close, failed};
use crate::cycles;
use crate::fixtures;

/// The pure-pursuit lookahead distance `L`.
const LOOKAHEAD_DISTANCE: f32 = 2.0;
/// `128·ε_f32`, the bound of an `f32` identity.
const IDENTITY_F32: f64 = 128.0 * f32::EPSILON as f64;

/// The policy layers: 22 observations, two hidden layers of 64, 4 actions.
const OBSERVATION: usize = 22;
const HIDDEN: usize = 64;
const ACTION: usize = 4;

/// A position fix: the sensor reads the first two state components.
struct GlobalPosition;

impl VectorFn<5, 2> for GlobalPosition {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// Golden and identity: the last of eight coordinated-turn EKF ticks in `f32` (predict, position
/// update, pure pursuit) against the `extended_kalman_filter_coordinated_turn_fusion` golden, and
/// `κ = 2·lateral/L²`.
pub fn mobile_robot_tick_f32() -> CheckOutcome {
    const NAME: &str = "mobile_robot_tick_f32";
    let narrow_matrix =
        |rows: [[f64; 5]; 5]| Matrix::new(rows.map(|row| row.map(|value| value as f32)));
    let motion = ConstantTurnAndSpeed {
        timestep: fixtures::COORDINATED_TURN_TIMESTEP,
    };
    let mut filter = ExtendedKalmanFilter::<5, 2, f32>::new(
        Vector::new(fixtures::COORDINATED_TURN_INITIAL_STATE.map(|value| value as f32)),
        narrow_matrix(fixtures::COORDINATED_TURN_INITIAL_COVARIANCE),
        narrow_matrix(fixtures::COORDINATED_TURN_PROCESS_NOISE),
        Matrix::new(
            fixtures::COORDINATED_TURN_MEASUREMENT_NOISE.map(|row| row.map(|value| value as f32)),
        ),
    );
    let [early @ .., last] = fixtures::COORDINATED_TURN_MEASUREMENTS;
    for fix in early {
        if filter.predict(&motion).is_err()
            || filter
                .update(&GlobalPosition, Vector::new(fix.map(|value| value as f32)))
                .is_err()
        {
            return failed(NAME, 0);
        }
    }
    // A fixed world point, 2 m ahead and 1 m left of the golden end state.
    let lookahead = [
        fixtures::COORDINATED_TURN_EXPECTED_STATE[0] + 2.0,
        fixtures::COORDINATED_TURN_EXPECTED_STATE[1] + 1.0,
    ];
    let lookahead_point = black_box(Vector::new(lookahead.map(|value| value as f32)));
    let last_fix = black_box(Vector::new(last.map(|value| value as f32)));

    let start = cycles::read();
    let mut tick = || {
        filter.predict(&motion).ok()?;
        filter.update(&GlobalPosition, last_fix).ok()?;
        let state = filter.state();
        let pose = SE2::from_parts(SO2::from_angle(state[2]), Vector::new([state[0], state[1]]));
        let curvature = pure_pursuit_curvature(pose, lookahead_point, LOOKAHEAD_DISTANCE).ok()?;
        Some((state, curvature.value()))
    };
    let result = black_box(tick());
    let elapsed = cycles::read().wrapping_sub(start);

    let Some((state, curvature)) = result else {
        return failed(NAME, elapsed);
    };
    let (x, y, heading) = (
        f64::from(state[0]),
        f64::from(state[1]),
        f64::from(state[2]),
    );
    let lateral =
        -Numeric::sin(heading) * (lookahead[0] - x) + Numeric::cos(heading) * (lookahead[1] - y);
    let lookahead_distance = f64::from(LOOKAHEAD_DISTANCE);
    let expected_curvature = 2.0 * lateral / (lookahead_distance * lookahead_distance);
    let passed = (0..5).all(|index| {
        close(
            f64::from(state[index]),
            fixtures::COORDINATED_TURN_EXPECTED_STATE[index],
            fixtures::COORDINATED_TURN_F32_ABS,
            fixtures::COORDINATED_TURN_F32_REL,
        )
    }) && close(
        f64::from(curvature),
        expected_curvature,
        IDENTITY_F32,
        IDENTITY_F32,
    );
    CheckOutcome {
        name: NAME,
        passed,
        cycles: elapsed,
        value: f64::from(curvature),
    }
}

/// Golden: one forward pass of the policy in `T` against numpy, sample 0.
fn policy_forward<T: Numeric + Primal>(
    name: &'static str,
    parameters: &'static [T],
    abs: f64,
    rel: f64,
) -> CheckOutcome {
    let mut cursor = ParameterCursor::new(black_box(parameters));
    let (Ok(first), Ok(second), Ok(output)) = (
        cursor.try_take_layer::<HIDDEN, OBSERVATION>(Activation::Relu),
        cursor.try_take_layer::<HIDDEN, HIDDEN>(Activation::Relu),
        cursor.try_take_layer::<ACTION, HIDDEN>(Activation::Tanh),
    ) else {
        return failed(name, 0);
    };
    let every_parameter_read = cursor.is_empty();
    let observation = black_box(Vector::<OBSERVATION, T>::new(
        fixtures::MLP_POLICY_OBSERVATION.map(T::from_f64),
    ));

    let start = cycles::read();
    let hidden_first = first.forward(observation.view());
    let hidden_second = second.forward(hidden_first.view());
    let action = black_box(output.forward(hidden_second.view()));
    let elapsed = cycles::read().wrapping_sub(start);

    let passed = every_parameter_read
        && (0..ACTION).all(|index| {
            close(
                action[index].to_f64(),
                fixtures::MLP_POLICY_ACTION[index],
                abs,
                rel,
            )
        });
    CheckOutcome {
        name,
        passed,
        cycles: elapsed,
        value: action[0].to_f64(),
    }
}

/// Golden: the policy in `f64`.
pub fn mlp_policy_forward() -> CheckOutcome {
    policy_forward::<f64>(
        "mlp_policy_forward",
        &fixtures::MLP_POLICY_PARAMETERS,
        fixtures::MLP_POLICY_ABS,
        fixtures::MLP_POLICY_REL,
    )
}

/// Golden: the policy in `f32`, the configuration of an FPU.
pub fn mlp_policy_forward_f32() -> CheckOutcome {
    policy_forward::<f32>(
        "mlp_policy_forward_f32",
        &fixtures::MLP_POLICY_PARAMETERS_F32,
        fixtures::MLP_POLICY_F32_ABS,
        fixtures::MLP_POLICY_F32_REL,
    )
}
