# Estimation

State estimation from noisy measurements. `KalmanFilter` is the linear filter: `predict` rolls the
state forward through a matrix model and grows the covariance by the process noise; `update` folds in
a measurement and shrinks it. Fixed-size, no allocation, and generic over the `Numeric` scalar, so a
`Dual` state differentiates the whole filter.

- `KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>`: built from an initial estimate and a
  `KalmanModel`.
- `KalmanModel<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>`: the four matrices that describe what the
  filter is tracking — transition, measurement model, process noise, measurement noise. Three of
  them are the same shape, so naming each field is what stops a swapped pair compiling.
- `predict` / `predict_with_control`: the time step, undriven or with a `control_model ·
  control_input` term. `CONTROL_DIMENSION` lives on the method, so undriven users never meet it.
- `update`: the measurement step. The only fallible operation in the module.
- `CovarianceUpdate`: `Joseph` (the default) or `Naive`.
- `innovation` / `innovation_covariance` / `normalized_innovation_squared`: for measurement gating.
- The setters (`set_state_transition`, `set_process_noise`, …) cover the time-varying case, where a
  changing timestep changes the model between steps.

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

The covariance update uses Joseph form by default — `(I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ` — which stays
symmetric and positive definite by construction, while the naive `(I − K·H)·P` loses symmetry as
rounding builds up. Joseph form is not a guarantee at every scale: over roughly 10⁷ single-precision
updates it drifts too, and the fix there is to symmetrize and clamp the covariance.

`update` returns `EstimationError::NonFinite` for a non-finite measurement or innovation covariance,
and `EstimationError::NotPositiveDefinite` when the innovation covariance cannot be factorized — the
gain is undefined. `predict` is a cheap element-wise path and propagates non-finite values silently.

`ExtendedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>` takes the process and measurement
models as functions rather than matrices — any `VectorFn` — and re-linearizes them at the current
estimate on every step. **The Jacobians come from automatic differentiation: write the model once
and its partial derivatives are exact, with no hand-derived Jacobians anywhere** — the classic source
of silent estimator bugs.

- `new` / `from_derivator`: the autodiff default, or an explicit differentiation backend (e.g.
  `FiniteDifferenceMulti`).
- `predict(&process_model)` / `update(&measurement_model, measurement)`: the models are passed per
  step, not stored, so the type stays `ExtendedKalmanFilter<3, 2>`. A control input or a changing
  timestep lives in the model as a field the caller sets between steps — there is no
  `predict_with_control`. Unlike the linear filter, `predict` here evaluates and differentiates a
  model, so it returns a `Result`.
- `update_with_residual(&measurement_model, residual)`: `update` with a caller-formed residual, for
  when a measurement component is an angle — plain subtraction is wrong across the ±π wrap, and only
  the caller knows which components are angular.
- `CovarianceUpdate`, the accessors, and `normalized_innovation_squared` are shared with the linear
  filter. `predict` and `update` also return `EstimationError::Diff` if a Jacobian step fails —
  reachable only with a finite-difference backend, as the autodiff default cannot.

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
the extended filter does, and handles their curvature a different way. Rather than flattening the
model to a straight line at the current estimate, it picks `2·STATE_DIMENSION + 1` points spread
around it, pushes each one through the model untouched, and rebuilds the estimate from where they
land. **The model is never differentiated, so it does not have to be smooth** — a lookup table, a
saturating actuator, or a branch on a threshold works here and does not work in a filter that needs
a derivative. On a strongly curved model the answer is usually closer than one straight-line fit
gets.

- `new`: the same four matrices as the extended filter — initial estimate, initial covariance,
  process noise, measurement noise.
- `with_scaling(alpha, beta, kappa)`: how far the points spread and how the middle one is weighted.
  `alpha` = 1e-3, `beta` = 2, `kappa` = 0 by default. It returns a `Result` rather than chaining,
  because a spread that works out to zero or less has no points to place and is worth catching where
  it is written rather than a step later.
- `with_regularization(epsilon)`: adds `epsilon` to the diagonal before the covariance is
  factorized. Off by default and never applied on its own — a covariance that cannot be factorized
  returns `EstimationError::NotPositiveDefinite`, and quietly nudging it would hide a filter that
  has gone wrong.
- `predict(&process_model)` / `update(&measurement_model, measurement)` /
  `update_with_residual(&measurement_model, residual)`: as on the extended filter. `update` works
  from the points `predict` left behind, so predict first — with no prediction the gain is zero and
  the estimate does not move.
- The accessors and `normalized_innovation_squared` are shared with the other two filters. There is
  no `CovarianceUpdate` here: Joseph and naive are two ways of writing one step this filter does not
  have. Its covariance is made exactly symmetric every time it is formed instead.

```rust
use multicalc::UnscentedKalmanFilter;
use multicalc::{Matrix, Vector};
use multicalc::{Numeric, VectorFn};

// Range to a landmark at (3, 4), measured by a sensor that saturates at 6 — a model with a corner
// in it, which no derivative describes and this filter does not need one for.
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

One thing this filter asks that the other two do not. It averages the points it gets back, so an
angle that the process model wraps into a ±π band is a trap: two points a hair apart end up at +π
and −π, and their average is nothing like either. Let an angular state component run past ±π inside
the model and wrap it afterwards through `set_state`. The points themselves sit a fraction of a
standard deviation apart, so nothing else this filter averages can straddle the boundary — but the
innovation can, which is what `update_with_residual` is for, exactly as on the extended filter.

Which of the two nonlinear filters to reach for comes down to the model. The extended filter is
cheaper when the model is cheap to differentiate and gently curved. This one costs
`2·STATE_DIMENSION + 1` evaluations per step instead of a derivative, and earns that back on a
sharply curved model, or on any model a derivative does not describe.

## Error-state filter

`ErrorStateKalmanFilter<MEASUREMENT_DIMENSION, T>` fuses an IMU — a turn-rate sensor and a push
sensor — with whatever corrections a vehicle can get, and tracks where a body is, how it is moving,
which way it faces, and what its own two sensors are getting steadily wrong.

It tracks the *correction* to a running guess rather than the guess itself. That is what lets the
facing live on the rotation group, where it can turn any distance without wrapping or needing
renormalization, while the uncertainty stays a plain flat fifteen numbers that ordinary matrix
arithmetic can carry forward. After every correction the error is folded back into the guess and
reset to zero, so it never grows large enough for the flat treatment to strain.

The fifteen numbers run in this order, three each:

| Index range | Meaning |
| --- | --- |
| 0..3 | where the estimate has the body, in world axes, metres |
| 3..6 | how wrong the estimated speed is, world axes, m/s |
| 6..9 | a small turn taking the estimated facing to the true one, radians |
| 9..12 | the turn-rate sensor's steady offset, rad/s |
| 12..15 | the push sensor's steady offset, m/s² |

- `NominalState<T>`: the running guess — place, motion, facing, and the two sensor offsets. Built
  with `new`, or with `at_rest` from a facing alone. `plus_error` folds a correction in and
  `error_from` takes one back out; the two are exact inverses, which is what the reset relies on.
  The starting facing usually comes from `SO3::from_two_direction_pairs`.
- `ImuNoise<T>`: how noisy the IMU is, in the figures a datasheet quotes rather than as a raw noise
  matrix. Four fields of the same type, so naming each is what stops a swapped pair compiling.
- `NominalStateFn<MEASUREMENT_DIMENSION>`: a sensor model, written once against named fields and
  evaluated at whatever kind of number the filter needs. **No derivative is ever coded by hand.**
- `new(initial_state, initial_covariance, imu_noise, measurement_noise)`, with
  `with_gravity` and `with_covariance_update` for the two settings that have defaults.
- `predict(gyroscope_reading, accelerometer_reading, timestep)`: one IMU step. The transition it
  uses is written in closed form and reachable as `error_state_transition`.
- `update` / `update_with_residual` / `update_other`: one correction. Use the second when a
  measurement is an angle, because plain subtraction is wrong across the ±π wrap. Use the third for a
  sensor of a different width from the one the filter is declared with — a three-number position fix
  and a one-number heading aid cannot both set the type's width.
- `inject_error_and_reset(error)`: `update` calls this itself; it is public so the step can be
  exercised with a known correction.
- `condition_covariance(minimum_eigenvalue)`: see below.
- `normalized_estimation_error_squared(true_state)`: how far the estimate is from a known truth,
  measured against its own claimed spread. Only a test or a simulation has the truth to pass in.

**Two generic parameters, where the extended filter has four.** The state width is fixed at fifteen
by the formulation. There is no pluggable differentiation backend either: a stepped difference over
an error state is not meaningful, because the error is identically zero and a finite step would move
a point on the rotation group by an amount the sensor model cannot tell from real signal. The
Jacobian is always taken exactly.

Evening the spread out across its diagonal happens on every predict and every update, and costs
almost nothing. Lifting a direction that rounding has pushed below zero is a different matter: it
means working out the spread's directions, which costs far more than a filter step does. So
`condition_covariance` is left to the caller's schedule — once a second, or on a health check, not
every tick. Joseph form plus the automatic evening is what you get otherwise, and that is good for
hours rather than for a ten-million-update single-precision duty cycle.

Two things will look like bugs and are not. The turn-rate offset about the vertical is only visible
through a heading aid, so without one it never settles. The push offset along the vertical is only
visible when the push itself varies, because a body pushed a little too hard upward looks exactly
like a body tilted a little.

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

// Sitting still, the push sensor reads a full gravity upward.
let gravity_strength = 9.81;
let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
let accelerometer_reading = Vector::new([0.0, 0.0, gravity_strength]);
let timestep = 0.001;
filter.predict(gyroscope_reading, accelerometer_reading, timestep).unwrap();

// The tracker says the drone is a little east of where the filter has it.
let step_east = 0.1;
filter.update(&RoomTracker, Vector::new([step_east, 0.0, 0.0])).unwrap();
assert!(filter.nominal_state().position()[0] > 0.0);

// The sensor offsets start at zero and are learned from corrections like that one.
let learned = filter.nominal_state().accelerometer_bias();
assert!(learned.is_finite());
```

Full demo:
[error_state_estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/error_state_estimation.rs).

## Attitude filters

`MahonyFilter<T>` and `MadgwickFilter<T>` work out which way a body is facing, and what its
turn-rate sensor reads when the body is not turning, from a turn-rate sensor, a push sensor, and
optionally a magnetometer. They carry a facing and a three-number offset and nothing else — no
place, no speed, no spread. When a spread is wanted, that is `ErrorStateKalmanFilter`'s job, at
roughly a hundred times the arithmetic; these are a handful of cross products and one exponential a
tick, the same work whatever the readings are.

A turn-rate sensor alone gives a smooth facing that slowly wanders. A push sensor alone says which
way is down but jumps about whenever the body is pushed; a magnetometer alone says which way is
north and is easily disturbed. Both filters take the turn rate as the answer and nudge it, every
tick, by however far the other two say it is off. They differ only in how they nudge:

- `MahonyFilter` nudges harder the more wrong it is — a `with_proportional_gain` term acting now,
  and a `with_integral_gain` term that turns the running total of those nudges into the offset
  estimate. Set the integral gain to zero and no offset is learned, at the price of a small
  permanent lean whenever the sensor really does have one.
- `MadgwickFilter` always nudges by the same amount and takes only the direction from the readings.
  That makes `with_correction_gain`, in radians per second, the whole tuning story: it is how fast
  the filter is willing to walk toward the sensors, and it does not change whether the facing is a
  degree out or ninety. `with_bias_gain` says how much of that walking to blame on a sensor offset;
  set it to zero for the published filter's behaviour, which learns none.

Shared by both:

- `new(initial_orientation)`: the only thing without a default. The starting facing usually comes
  from `SO3::from_two_direction_pairs` on a still body.
- `with_reference_directions(upward_reference, north_reference)`: which way is up and which way is
  north, in world axes. Starts at `(0, 0, 1)` and `(1, 0, 0)`. Both go in at once so their order
  cannot matter, and north is squared up against up before it is stored.
- `step(gyroscope_reading, accelerometer_reading, magnetometer_reading, timestep)`: one tick. The
  magnetometer is an `Option`, so a tick without one — or with one that is not to be trusted — is
  a `None` rather than a separate call. `step_without_magnetometer` is the same thing spelled
  shorter.
- `orientation()` and `gyroscope_bias()`: what it has worked out. `set_orientation` and
  `set_gyroscope_bias` put a value back, for re-seeding from a still-body fix or restoring a saved
  offset at start-up.

The magnetometer's world direction is worked out afresh each tick rather than taken as a setting:
the measured field is turned into world axes, its upward part is kept as it is, and everything left
over is laid along north. So the magnetometer only ever moves the heading, and a caller who does
not know how steeply the local field dips cannot get a lasting lean out of it.

Three things will look like bugs and are not. Without a magnetometer the heading rides on the
turn-rate sensor alone and slowly wanders — only the lean is pinned, because down is the only
direction a push sensor can see. A body in free fall has no usable down at all: the push reading
goes to nothing, contributes nothing, and the facing coasts on the turn rate until the body is
caught. And `MadgwickFilter` never quite stands still even when it is exactly right, because a
fixed-rate walk always takes a step; that step, `with_correction_gain` times the timestep, is its
error floor.

Neither filter can tell a sustained push from gravity. A body accelerating steadily gives a push
reading that is confidently wrong about down, and both will believe it — which is the honest reason
these are the error-state filter's complement rather than its replacement.

The facing is pulled back onto unit length every tick. The step itself already gives a true
rotation; the pull back is there because these filters are the ones expected to run for hours at a
kilohertz in single precision with nothing else watching for drift.

```rust
use multicalc::{MadgwickFilter, MahonyFilter, SO3, Vector};

// Starting off level by about a tenth of a radian, on a body that is in fact still and level.
let tilt = Vector::new([0.1, -0.05, 0.0]);
let tilted = SO3::exp(tilt);
let mut mahony = MahonyFilter::new(tilted);
let mut madgwick = MadgwickFilter::new(tilted);

// A still body reads one gravity upward, and a field pointing north and 60 degrees down.
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

// Both have found level, and neither was leaned over by the steeply dipping field.
assert!(mahony.orientation().log().norm() < 1e-3);
assert!(madgwick.orientation().log().norm() < 1e-3);
```

Full demo:
[attitude_filter.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/attitude_filter.rs).

## Particle filter

`ParticleFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T, R>` carries a cloud of weighted state
samples instead of a single Gaussian, so it can track a belief the Kalman filters cannot represent —
strongly nonlinear, non-Gaussian, or with several peaks at once (a robot that could be in one of two
corridors). It is the tool to reach for when a single-Gaussian belief is the thing that breaks, and
the price is running hundreds to thousands of samples every step.

- `new(particle_count, initial_mean, initial_covariance, process_noise, seed)`: samples the starting
  cloud from the given Gaussian, with a seeded built-in `Pcg32`. `from_random` takes any
  `RandomSource` instead. `particle_count` must be at least one.
- `predict(&process_model)`: pushes every sample through the model — any `VectorFn` — and adds a draw
  of process noise. `update(&measurement_model, &likelihood, measurement)`: reweights each sample by
  how well its predicted measurement matches, normalizes, and resamples if the cloud has degenerated.
- `Likelihood` scores a sample as a log-weight; `GaussianLikelihood::new(measurement_noise)` is the
  batteries-included default for additive Gaussian noise. Write your own for anything else.
- `ResamplingScheme`: `Systematic` (the default), `Stratified`, `Multinomial`, or `Residual`. Set it
  with `with_resampling`; tune when it fires with `with_resample_threshold`, and add post-resample
  jitter with `with_roughening`.
- `mean` (the usual estimate), `maximum_a_posteriori_state` (the single heaviest sample, for when the
  belief has several peaks and the mean falls between them), `effective_sample_size`, `particles`,
  and `weights`.

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
```

The particle filter is heap-backed, so it is behind the `alloc` feature and the bare-metal build does
not compile it. Its `update` returns `EstimationError::NonFinite` for a non-finite measurement and
`EstimationError::WeightsDegenerate` when no sample can explain the measurement. `GaussianLikelihood`
forms the mismatch by plain subtraction, so a measurement with an angular component needs a custom
`Likelihood` that folds the angle into a ±π band first — the same wrap the extended filter's
`update_with_residual` exists for.


---

[Back to the tutorial index](README.md)
