# Estimation

State estimation from noisy measurements. All filters are fixed-size and generic over the `Numeric`
scalar, so a `Dual` state differentiates the whole filter. `ParticleFilter` and
`MonteCarloLocalizer` are heap-backed and need the `alloc` feature.

## Kalman filter

`KalmanFilter` is the linear filter. `predict` propagates `x = F·x`, `P = F·P·Fᵀ + Q`; `update` folds
in a measurement and shrinks `P`.

- `KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>`: built from an initial estimate, an
  initial covariance and a `KalmanModel`.
- `KalmanModel<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>`: the four matrices `state_transition`,
  `measurement_model`, `process_noise` and `measurement_noise`. Three of them have the same shape, so
  with named fields, a swapped pair does not compile.
- `predict` / `predict_with_control`: the time step, without or with a `control_model ·
  control_input` term. `CONTROL_DIMENSION` is a parameter of the method, not of the type.
- `update`: the measurement step. It returns a `Result`; `predict` does not.
- `with_covariance_update(CovarianceUpdate)`: `Joseph` (the default) or `Naive`.
- `innovation` / `innovation_covariance` / `normalized_innovation_squared`: for measurement gating.
- `set_state_transition`, `set_process_noise`, `set_measurement_model`, `set_measurement_noise`: for
  a time-varying model, for example a variable timestep.

```rust
use multicalc::{KalmanFilter, KalmanModel};
use multicalc::{Matrix, Vector};

// Constant velocity: position integrates velocity over a 1 s step; position is measured.
let initial_state = Vector::new([0.0, 0.0]);   // [position, velocity]
let initial_covariance = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
let model = KalmanModel {
    state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
    measurement_model: Matrix::new([[1.0, 0.0]]),   // position only
    process_noise: Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
    measurement_noise: Matrix::new([[0.1]]),
};

let mut filter = KalmanFilter::new(initial_state, initial_covariance, model);

filter.predict();
let measurement = Vector::new([1.0]);
filter.update(measurement).unwrap();
let position = filter.state()[0];

// Gate an outlier before folding it in.
filter.predict();
let outlier = Vector::new([1.9]);
filter.update(outlier).unwrap();
let gate = filter.normalized_innovation_squared().unwrap();
```

The Joseph form `P = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ` keeps `P` symmetric positive definite by
construction. The naive `P = (I − K·H)·P` loses symmetry as rounding accumulates. Joseph form also
drifts over approximately 10⁷ single-precision updates; at that scale, symmetrize and clamp the
covariance.

`update` returns `EstimationError::NonFinite` for a non-finite measurement or innovation covariance,
and `EstimationError::NotPositiveDefinite` when the innovation covariance has no Cholesky factor, so
the gain is undefined. `predict` is an element-wise path and propagates non-finite values silently.

## Extended filter

`ExtendedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>` takes the process and measurement
models as functions (any `VectorFn`) and re-linearizes them at the current estimate on each step.
**Automatic differentiation gives exact Jacobians: write the model once, and no Jacobian is derived
by hand.**

- `new` / `from_derivator`: the autodiff default, or an explicit differentiation backend, for
  example `FiniteDifferenceMulti`.
- `predict(&process_model)` / `update(&measurement_model, measurement)`: the caller passes the models
  on each step, so the type stays `ExtendedKalmanFilter<3, 2>`. A control input or a variable
  timestep is a field of the model that the caller sets between steps. There is no
  `predict_with_control`.
- `update_with_residual(&measurement_model, residual)`: `update` with a caller-formed residual.
  Use it when a measurement component is an angle, because plain subtraction is wrong across the ±π
  wrap. Only the caller knows which components are angular.
- `CovarianceUpdate`, the accessors and `normalized_innovation_squared` are the same as on the
  linear filter.

`predict` evaluates and differentiates a caller model, so it returns a `Result`. Every step checks
for non-finite values: `predict` returns `EstimationError::NonFinite` for a non-finite propagated
state or Jacobian, and `update` does the same for the measurement, the residual and the innovation
covariance. Both also return `EstimationError::Diff` if a Jacobian step fails. Only a
finite-difference backend can cause `Diff`; the autodiff default cannot.

```rust
use multicalc::ExtendedKalmanFilter;
use multicalc::{Matrix, Vector};
use multicalc::{Numeric, VectorFn};

// Range to a landmark at (3, 4): nonlinear in the pose, so the linear filter cannot take it.
struct RangeToLandmark;
impl VectorFn<2, 1> for RangeToLandmark {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        let to_landmark_x = S::from_f64(3.0) - state[0];
        let to_landmark_y = S::from_f64(4.0) - state[1];
        [(to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt()]
    }
}

// A stationary target: the pose carries over unchanged.
struct Stationary;
impl VectorFn<2, 2> for Stationary {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

let mut filter = ExtendedKalmanFilter::<2, 1>::new(
    Vector::new([0.0, 0.0]),                  // initial pose, 5.0 from the landmark
    Matrix::new([[1.0, 0.0], [0.0, 1.0]]),    // initial covariance
    Matrix::new([[0.01, 0.0], [0.0, 0.01]]),  // process noise
    Matrix::new([[0.1]]),                     // measurement noise
);
filter.predict(&Stationary).unwrap();
filter.update(&RangeToLandmark, Vector::new([5.5])).unwrap();
```

## Unscented filter

`UnscentedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>` takes the same `VectorFn` models
as the extended filter. It propagates `2·STATE_DIMENSION + 1` sigma points through the model and
rebuilds the mean and covariance from them. **The filter never differentiates the model, so the
model does not have to be smooth**: a lookup table, a saturated actuator or a threshold branch is
valid. On a strongly curved model, the result is usually closer than one linearization gives.

- `new`: initial estimate, initial covariance, process noise and measurement noise.
- `with_scaling(alpha, beta, kappa)`: the sigma-point spread and the weight of the centre point.
  The defaults are `alpha` = 1e-3, `beta` = 2, `kappa` = 0. It returns
  `EstimationError::InvalidTuning` if a value is not finite or if `alpha² · (STATE_DIMENSION + kappa)
  ≤ 0`, because then the points have no spread.
- `with_regularization(epsilon)`: adds `epsilon` to the diagonal before factorization. It is off by
  default and the filter never applies it automatically. Without it, a covariance with no Cholesky
  factor returns `EstimationError::NotPositiveDefinite` from `predict`. A negative or non-finite
  `epsilon` returns `EstimationError::InvalidTuning`.
- `predict(&process_model)` / `update(&measurement_model, measurement)` /
  `update_with_residual(&measurement_model, residual)`: as on the extended filter. `update` uses the
  sigma points from the last `predict`. Before the first `predict`, the gain is zero and the estimate
  does not move.
- The accessors and `normalized_innovation_squared` are the same as on the other two filters. There
  is no `CovarianceUpdate`, because the filter has no `(I − K·H)` step. The filter symmetrizes the
  covariance each time it forms it.

```rust
use multicalc::UnscentedKalmanFilter;
use multicalc::{Matrix, Vector};
use multicalc::{Numeric, VectorFn};

// Range to a landmark at (3, 4), from a sensor that saturates at 6: the model has a corner, and
// no derivative describes it.
struct SaturatingRange;
impl VectorFn<2, 1> for SaturatingRange {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        let to_landmark_x = S::from_f64(3.0) - state[0];
        let to_landmark_y = S::from_f64(4.0) - state[1];
        let range = (to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt();
        let ceiling = S::from_f64(6.0);
        [if range > ceiling { ceiling } else { range }]
    }
}

// A stationary target: the state carries over unchanged.
struct Stationary;
impl VectorFn<2, 2> for Stationary {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

let mut filter = UnscentedKalmanFilter::<2, 1>::new(
    Vector::new([0.0, 0.0]),                  // initial state, 5.0 from the landmark
    Matrix::new([[1.0, 0.0], [0.0, 1.0]]),
    Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
    Matrix::new([[0.1]]),
)
.with_scaling(0.3, 2.0, 0.0)
.unwrap();

filter.predict(&Stationary).unwrap();
filter.update(&SaturatingRange, Vector::new([5.5])).unwrap();
let position = filter.state();
```

The filter averages the propagated sigma points. If the process model wraps an angle into ±π, two
adjacent points can land at +π and −π, and their mean is wrong. Let an angular state component run
past ±π inside the model, and wrap it after the step through `set_state`. The sigma points are a
fraction of a standard deviation apart, so no other average crosses the wrap. The innovation can
cross it; use `update_with_residual` for that, as on the extended filter.

Choice between the two nonlinear filters:

- The extended filter is cheaper when the model is cheap to differentiate and only slightly curved.
- The unscented filter costs `2·STATE_DIMENSION + 1` model evaluations per step and no derivative.
  It is better on a strongly curved model, or on a model that has no derivative.

## Error-state filter

`ErrorStateKalmanFilter<MEASUREMENT_DIMENSION, T>` fuses an IMU (gyroscope and accelerometer) with
aid measurements. It estimates position, velocity, orientation, gyroscope bias and accelerometer
bias.

The filter estimates the error of a nominal state, not the state itself. The orientation stays on
SO(3), with no wrap and no renormalization, and the covariance is over a flat 15-dimensional error.
After each update, the filter injects the error into the nominal state and resets the error to zero.
Thus the error stays small and the linearization stays valid.

The 15 error components, three each:

| Index range | Meaning |
| --- | --- |
| 0..3 | position error, world axes, m |
| 3..6 | velocity error, world axes, m/s |
| 6..9 | rotation from the estimated orientation to the true orientation, rad |
| 9..12 | gyroscope bias, rad/s |
| 12..15 | accelerometer bias, m/s² |

- `NominalState<T>`: position, velocity, orientation and the two biases. Build it with `new`, or with
  `at_rest` from an orientation alone. `plus_error` injects an error and `error_from` extracts one.
  The reset relies on the invariant that the two are exact inverses. `SO3::from_two_direction_pairs`
  usually gives the initial orientation.
- `ImuNoise<T>`: the IMU noise in datasheet terms (`gyroscope_noise_density`,
  `accelerometer_noise_density`, `gyroscope_bias_random_walk`, `accelerometer_bias_random_walk`), not
  as a raw noise matrix. The four fields have the same type, so the names stop a swapped pair.
- `NominalStateFn<MEASUREMENT_DIMENSION>`: a measurement model, written once against the named
  fields and evaluated at the scalar type the filter needs. **No derivative is coded by hand.**
- `new(initial_state, initial_covariance, imu_noise, measurement_noise)`. `with_gravity` replaces the
  default gravity `(0, 0, −9.81)` (z up), and `with_covariance_update` replaces the default `Joseph`.
- `predict(gyroscope_reading, accelerometer_reading, timestep)`: one IMU step. The error-state
  transition is in closed form; `error_state_transition` returns it.
- `update` / `update_with_residual` / `update_other`: one correction. Use `update_with_residual` when
  a measurement is an angle, because plain subtraction is wrong across the ±π wrap. Use
  `update_other` for a sensor of a different width from `MEASUREMENT_DIMENSION`, for example a
  3-number position fix and a 1-number heading aid on the same filter.
- `inject_error_and_reset(error)`: `update` calls it. It is public, so a test can apply a known
  correction.
- `condition_covariance(minimum_eigenvalue)`: see below.
- `normalized_estimation_error_squared(true_state)`: NEES against a known truth. Only a test or a
  simulation has the truth.

**Two generic parameters, where the extended filter has four.** The error state is fixed at 15. There
is no pluggable differentiation backend: the error is identically zero at the linearization point,
and a finite step on SO(3) is not distinguishable from signal in the measurement model. The filter
always takes the Jacobian exactly, with `Dual`.

Each `predict` and `update` symmetrizes the covariance, at almost no cost. `condition_covariance`
also clamps the eigenvalues to `minimum_eigenvalue`. The eigendecomposition costs much more than a
filter step, so call it on a slow schedule (for example, once per second or on a health check), not
on each tick. Joseph form plus symmetrization is sufficient for hours of operation, but not for a
10⁷-update single-precision duty cycle.

Two expected behaviors that are not bugs:

- The gyroscope bias about the vertical axis is observable only through a heading aid. Without one,
  it does not converge.
- The accelerometer bias along the vertical axis is observable only when the specific force
  changes. Otherwise, a small vertical bias is not distinguishable from a small tilt.

```rust
use multicalc::{ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn};
use multicalc::{Matrix, Numeric, SO3, Vector};

// A tracker in the room reports where the drone is, and nothing else.
struct RoomTracker;
impl NominalStateFn<3> for RoomTracker {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.position().as_array()
    }
}

let level = SO3::<f64>::identity();
let starting_spread = 0.1;
let imu_noise = ImuNoise {
    gyroscope_noise_density: 0.02,
    accelerometer_noise_density: 0.05,
    gyroscope_bias_random_walk: 1e-4,
    accelerometer_bias_random_walk: 1e-3,
};
let tracker_spread = 0.03;
let mut filter = ErrorStateKalmanFilter::<3>::new(
    NominalState::at_rest(level),
    Matrix::from_diagonal([starting_spread; 15]),
    imu_noise,
    Matrix::from_diagonal([tracker_spread * tracker_spread; 3]),
);

// At rest, the accelerometer reads one gravity upward.
let gravity_strength = 9.81;
let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
let accelerometer_reading = Vector::new([0.0, 0.0, gravity_strength]);
let timestep = 0.001;
filter.predict(gyroscope_reading, accelerometer_reading, timestep).unwrap();

// The tracker puts the drone a little east of the estimate.
let step_east = 0.1;
filter.update(&RoomTracker, Vector::new([step_east, 0.0, 0.0])).unwrap();
assert!(filter.nominal_state().position()[0] > 0.0);

// The biases start at zero and the corrections estimate them.
let learned = filter.nominal_state().accelerometer_bias();
assert!(learned.is_finite());
```

Full demo:
[error_state_estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/error_state_estimation.rs).

## Attitude filters

`MahonyFilter<T>` and `MadgwickFilter<T>` estimate orientation and gyroscope bias from a gyroscope,
an accelerometer and an optional magnetometer. They hold an orientation and a 3-vector bias only: no
position, no velocity, no covariance. For a covariance, use `ErrorStateKalmanFilter`, at
approximately 100 times the arithmetic. Each tick of these filters is a few cross products and one
exponential map, with constant work.

Both filters integrate the gyroscope and correct the orientation on each tick toward the
accelerometer (down) and the magnetometer (north). They differ in the correction law:

- `MahonyFilter`: a PI correction, proportional to the error. `with_proportional_gain` (default 1) is
  the proportional term. `with_integral_gain` (default 0.1) integrates the correction into the bias
  estimate. If the integral gain is zero, the filter estimates no bias and keeps a small steady tilt
  when the gyroscope has a bias.
- `MadgwickFilter`: a correction of fixed magnitude along the normalized error direction.
  `with_correction_gain` (rad/s, default 0.1) is the full tuning: the rate at which the filter moves
  toward the sensors, the same for a 1° error and a 90° error. `with_bias_gain` (default 0.01) sets
  how much of that correction goes into the bias. Set it to zero for the published filter, which
  estimates no bias.

Common to both:

- `new(initial_orientation)`: the only value with no default. `SO3::from_two_direction_pairs` on a
  body at rest usually gives it.
- `with_reference_directions(upward_reference, north_reference)`: up and north in world axes. The
  defaults are `(0, 0, 1)` and `(1, 0, 0)`. The method takes both together, so their order has no
  effect, and it orthogonalizes north against up before it stores them.
- `step(gyroscope_reading, accelerometer_reading, magnetometer_reading, timestep)`: one tick. The
  magnetometer is an `Option`; pass `None` for a tick with no magnetometer or an untrusted one.
  `step_without_magnetometer` is the shorter form.
- `orientation()` and `gyroscope_bias()`: the estimate. `set_orientation` and `set_gyroscope_bias`
  set a value, for example to re-seed from a fix at rest or to restore a saved bias at start-up.

On each tick, the filter rotates the measured magnetic field into world axes, keeps its vertical
component, and puts the horizontal remainder along north. Thus the magnetometer corrects heading
only, and an unknown magnetic dip angle causes no steady tilt.

Expected behaviors that are not bugs:

- Without a magnetometer, the heading integrates the gyroscope alone and drifts. Only tilt is
  corrected, because down is the only direction an accelerometer measures.
- In free fall, the accelerometer reads zero and gives no correction. The orientation integrates the
  gyroscope until the free fall stops.
- `MadgwickFilter` does not stop at the true orientation, because each tick applies a correction of
  fixed size. Its error floor is `with_correction_gain` times the timestep.
- Neither filter can separate a sustained acceleration from gravity. A steady acceleration gives a
  wrong down direction, and both filters follow it. Thus these filters complement the error-state
  filter and do not replace it.

The filters renormalize the quaternion on each tick. The step itself gives a valid rotation; the
renormalization is for runs of hours at kilohertz rates in single precision with no other drift
check.

```rust
use multicalc::{MadgwickFilter, MahonyFilter, SO3, Vector};

// Start approximately 0.1 rad off level, on a body that is at rest and level.
let tilt = Vector::new([0.1, -0.05, 0.0]);
let tilted = SO3::exp(tilt);
let mut mahony = MahonyFilter::new(tilted);
let mut madgwick = MadgwickFilter::new(tilted);

// At rest: one gravity upward, and a field that points north and 60 degrees down.
let gravity_strength = 9.81;
let not_turning = Vector::new([0.0, 0.0, 0.0]);
let one_gravity_up = Vector::new([0.0, 0.0, gravity_strength]);
let dip: f64 = 60.0_f64.to_radians();
let field = Vector::new([dip.cos(), 0.0, -dip.sin()]);
let timestep = 0.005;
let ticks = 12_000; // a minute at 200 Hz

for _ in 0..ticks {
    mahony.step(not_turning, one_gravity_up, Some(field), timestep).unwrap();
    madgwick.step(not_turning, one_gravity_up, Some(field), timestep).unwrap();
}

// Both converge to level, and the steep dip causes no tilt.
assert!(mahony.orientation().log().norm() < 1e-3);
assert!(madgwick.orientation().log().norm() < 1e-3);
```

Full demo:
[attitude_filter.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/attitude_filter.rs).

## Ready-made models

- `ConstantTurnAndSpeed { timestep }`: the coordinated-turn model over the state
  `[x, y, heading, speed, turn rate]`. It moves the state one tick along the arc, with speed and turn
  rate held constant.
- `DirectMeasurement<STATE_DIMENSION, MEASUREMENT_DIMENSION>`: a sensor that reads the state
  components at the listed indices, for example `[x, y]` from a position fix. `try_new` returns
  `EstimationError::StateIndexOutOfRange` for an index past the end of the state.
- `residual_with_wrapped_angles(reading, predicted, angle_indices)`: `reading − predicted`, with the
  listed components wrapped into (−π, π]. Without the wrap, a heading on the two sides of ±π gives an
  error of almost one full turn, and the filter makes a large wrong correction.

```rust
use multicalc::estimation::{
    ConstantTurnAndSpeed, DirectMeasurement, ExtendedKalmanFilter, residual_with_wrapped_angles,
};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::VectorFn;

// A vehicle tracked as [x, y, heading, speed, turn rate], corrected by position alone.
let timestep = 0.1;
let motion = ConstantTurnAndSpeed { timestep };
let position_fix = DirectMeasurement::<5, 2>::try_new([0, 1])?;

let initial_state = Vector::new([0.0, 0.0, 0.0, 1.0, 0.2]);
let initial_covariance = Matrix::from_diagonal([0.5; 5]);
let process_noise = Matrix::from_diagonal([1e-4, 1e-4, 1e-4, 1e-3, 1e-3]);
let measurement_noise = Matrix::from_diagonal([0.05, 0.05]);
let mut filter = ExtendedKalmanFilter::<5, 2>::new(
    initial_state,
    initial_covariance,
    process_noise,
    measurement_noise,
);

let fix = Vector::new([0.1, 0.02]);
filter.predict(&motion)?;
filter.update(&position_fix, fix)?;
assert!(filter.state()[0].is_finite());

// A heading sensor needs the wrapped residual.
let heading_sensor = DirectMeasurement::<5, 2>::try_new([2, 4])?;
let reading = Vector::new([3.10, 0.2]);
let predicted = Vector::new(heading_sensor.eval(filter.state().as_array()));
let residual = residual_with_wrapped_angles(reading, predicted, &[0]);
assert!(residual[0].abs() <= core::f64::consts::PI);
filter.update_with_residual(&heading_sensor, residual)?;
# Ok::<(), multicalc::CalcError>(())
```

Full demo (linear, extended and unscented filters, and a particle filter with `alloc`):
[estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/estimation.rs).

## Particle filter

`ParticleFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T, R>` (`alloc` only) represents the belief
as a set of weighted samples, not as one Gaussian. Thus it can track a nonlinear, non-Gaussian or
multimodal belief, for example a robot that can be in one of two corridors. The cost is hundreds to
thousands of samples per step.

- `new(particle_count, initial_mean, initial_covariance, process_noise, seed)`: samples the initial
  set from the Gaussian, with a seeded `Pcg32`. `from_random` takes any `RandomSource`.
- `predict(&process_model)`: propagates each sample through the model (any `VectorFn`) and adds a
  process-noise draw.
- `update(&measurement_model, &likelihood, measurement)`: reweights each sample by the likelihood of
  the measurement, normalizes, and resamples if the effective sample size is below the threshold.
- `Likelihood` gives the log-weight of a sample. `GaussianLikelihood::new(measurement_noise)` is the
  implementation for additive Gaussian noise. Write a custom `Likelihood` for other noise.
- `with_resampling(ResamplingScheme)`: `Systematic` (the default), `Stratified`, `Multinomial` or
  `Residual`. `with_resample_threshold` sets the effective-sample-size threshold (default: half the
  particle count). `with_roughening` adds jitter after a resample.
- `mean` (the usual estimate), `covariance` (the weighted spread `Σ wᵢ·(xᵢ − μ)(xᵢ − μ)ᵀ`),
  `maximum_a_posteriori_state` (the heaviest sample, for a multimodal belief where the mean falls
  between the modes), `effective_sample_size`, `particles` and `weights`.
- `reseed(mean, covariance)` draws a fresh cloud and resets the weights. Use it to recover when the
  robot moves and no motion model sees the move.

```rust
# use multicalc::{GaussianLikelihood, ParticleFilter};
# use multicalc::{Matrix, Vector};
# use multicalc::{Numeric, VectorFn};
// A stationary 2-D point, measured directly with a little noise.
struct Stationary;
impl VectorFn<2, 2> for Stationary {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

let particle_count = 1000;
let initial_mean = Vector::new([0.0, 0.0]);
let initial_covariance = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
let process_noise = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
let seed = 7;

let mut filter = ParticleFilter::<2, 2>::new(
    particle_count,
    initial_mean,
    initial_covariance,
    process_noise,
    seed,
)
.unwrap();

let measurement_noise = Matrix::new([[0.05, 0.0], [0.0, 0.05]]);
let sensor = GaussianLikelihood::new(measurement_noise).unwrap();
let measurement = Vector::new([1.0, 2.0]);

for _ in 0..20 {
    filter.predict(&Stationary).unwrap();
    filter.update(&Stationary, &sensor, measurement).unwrap();
}
assert!((filter.mean()[0] - 1.0).abs() < 0.2);
assert!(filter.covariance()[(0, 0)] < 1.0);
filter.reseed(Vector::new([0.0, 0.0]), initial_covariance).unwrap();
```

Errors:

- `new` returns `EstimationError::NotPositiveDefinite` if a covariance has no Cholesky factor, and
  `EstimationError::WeightsDegenerate` if `particle_count` is zero.
- `update` returns `EstimationError::NonFinite` for a non-finite measurement, and
  `EstimationError::WeightsDegenerate` when no sample explains the measurement.
- `reseed` returns `EstimationError::NonFinite` for a non-finite mean, and
  `EstimationError::NotPositiveDefinite` if the covariance has no Cholesky factor.

`GaussianLikelihood` forms the residual by plain subtraction. For a measurement with an angular
component, write a custom `Likelihood` that wraps the angle into ±π first, as with
`update_with_residual` on the extended filter.

## Monte Carlo Localization

`MonteCarloLocalizer<NUM_BEAMS>` (`alloc` only) solves global localization: it finds the pose of a
robot on a known map with no initial fix. It is a particle filter over `[x, y, heading]`. Each
particle casts the scan beams against the map, and the particles that match the real scan gain
weight.

- `new(hint, InitialParticleCloud, BeamModel, seed)`: the initial cloud around a rough pose.
  `InitialParticleCloud` and `BeamModel` implement `Default`.
- `predict(delta_arc_length, delta_heading)`: odometry. `set_motion_noise` sets its variance (default
  `1e-4` on each of `x`, `y` and heading).
- `update(&scan, &map, &geometry)`: reweights and resamples against any
  [`OccupancyMap`](mapping.md). The beam directions come from the `ScanGeometry` of the real scan.
  `update_against_field` scores against a `DistanceField` instead; see
  [Mapping](mapping.md#localizing-with-a-likelihood-field).
- `estimate()`: the pose and its 3×3 covariance. `is_converged(position_limit, heading_limit)`
  tells when the result is reliable, which is usually the time to give control to a Kalman filter.

`new` returns `EstimationError::NotPositiveDefinite` for an unusable cloud spread and
`EstimationError::WeightsDegenerate` for a zero particle count. `update` returns
`EstimationError::WeightsDegenerate` when no particle explains the scan.

```rust
use multicalc::estimation::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};
use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap, ScanGeometry};

// A walled room the robot already has a map of.
let cell_size = 0.1_f64;
let mut room = DynamicOccupancyGrid::try_new(40, 30, cell_size, [0.0, 0.0])?;
let walls = [[0.2, 0.2], [3.8, 0.2], [3.8, 2.8], [0.2, 2.8]];
room.occupy_polyline(&walls, true);
room.occupy_circle([2.8, 2.0], 0.25);

const NUM_BEAMS: usize = 16;
let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(2.0 * core::f64::consts::PI / 3.0, 6.0)?;

// The true pose, and the rough hint the localizer gets.
let truth = [1.2, 1.0, 0.3];
let hint = [1.45, 0.75, 0.35];
let cloud = InitialParticleCloud {
    particle_count: 1500,
    position_variance: 0.16,
    heading_variance: 0.25,
};
let beam_model = BeamModel { range_deviation: 0.15, ..Default::default() };
let seed = 20260804;
let mut localizer = MonteCarloLocalizer::<NUM_BEAMS>::new(hint, cloud, beam_model, seed)?;

// One scan from a robot at rest, given to the localizer a few times.
let reading: [f64; NUM_BEAMS] = core::array::from_fn(|beam| {
    let offset = scan.beam_angle(beam).unwrap_or(0.0);
    room.cast_ray([truth[0], truth[1]], truth[2] + offset, scan.maximum_range())
        .unwrap_or(f64::INFINITY)
});
for _ in 0..10 {
    localizer.update(&reading, &room, &scan)?;
}

// The cloud converges on the true pose.
let (pose, _spread) = localizer.estimate();
assert!((pose[0] - truth[0]).abs() < 0.25);
assert!((pose[1] - truth[1]).abs() < 0.25);
assert!(localizer.is_converged(0.05, 0.2));
# Ok::<(), multicalc::CalcError>(())
```

Full demo:
[localized_lap_check.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/localized_lap_check.rs).

---

[Back to the tutorial index](README.md)
