# Plant

Actuator models between a command and the wrench or joint motion that the body receives.

- `MultirotorMixer`: control allocation. It maps a collective thrust and a body torque to rotor
  thrusts, and maps rotor thrusts back to a `Wrench`.
- `RotorLag`: first-order rotor thrust response, discretized exactly.
- `PositionServo`: a position-controlled joint, modelled as its closed-loop second-order servo
  and discretized exactly by zero-order hold.

## Multirotor mixer

Each rotor `i` at body position `(x_i, y_i)` gives thrust `f_i` along body z. The allocation
matrix `A` maps the thrusts to `[thrust; τx; τy; τz]`, with column `i` equal to
`[1, y_i, −x_i, ±k]`. Here `k` is `torque_per_thrust`, and the sign comes from the `RotorSpin`.

- `wrench(thrusts)` returns `A·f` as a `Wrench`: force on body z, torque about the body-frame
  origin. `RigidBody` takes this form directly.
- `rotor_thrusts(collective_thrust, torque)` returns rotor thrusts inside the limits, with the
  torque before the collective. With `d = A⁺·e₁`, the collective `c` moves, up or down, into
  the interval where `c·d + A⁺·[0; τ]` fits the limits. If no `c` fits the full torque, yaw is
  scaled down first. If roll and pitch alone do not fit, yaw is zero and roll and pitch are
  scaled together. The work is `O(ROTOR_COUNT²)`, with no iteration.
- `MultirotorMixer::new` computes `A⁺` once, so each call is one small matrix product.
- `quadrotor_x` builds the X layout. The rotor order is front right, back left, front left, back
  right. The rotors on each diagonal turn the same way.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::plant::MultirotorMixer;

// X quadrotor: 0.15 m arms, yaw torque 0.016 N·m per N of thrust, 0 N to 5 N per rotor.
let arm_length = 0.15_f64;
let torque_per_thrust = 0.016;
let minimum_thrust = 0.0;
let maximum_thrust = 5.0;
let mixer = MultirotorMixer::<4, f64>::quadrotor_x(
    arm_length,
    torque_per_thrust,
    minimum_thrust,
    maximum_thrust,
)?;

// Hover at 0.8 kg with zero torque: equal thrusts, no saturation.
let mass = 0.8;
let gravity_strength = 9.81;
let weight = mass * gravity_strength;
let no_turn = Vector::new([0.0, 0.0, 0.0]);

let commands = mixer.rotor_thrusts(weight, no_turn);
assert!(!commands.saturated());
for rotor in 0..4 {
    assert!((commands.thrusts()[rotor] - weight / 4.0).abs() < 1e-12);
}

// A·f gives back the commanded wrench.
let wrench = mixer.wrench(commands.thrusts());
assert!((wrench.force()[2] - weight).abs() < 1e-12);
assert!(wrench.torque().norm() < 1e-12);

// A command past the rotor limits saturates.
let beyond_reach = 30.0;
let too_much = mixer.rotor_thrusts(beyond_reach, no_turn);
assert!(too_much.saturated());

// Near full collective, a roll command keeps its torque and the collective gives way.
let roll = Vector::new([0.2, 0.0, 0.0]);
let aggressive = mixer.rotor_thrusts(19.0, roll);
let produced = mixer.wrench(aggressive.thrusts());
assert!((produced.torque()[0] - 0.2).abs() < 1e-12);
assert!(aggressive.shortfall()[0] > 0.0);
# Ok::<(), multicalc::CalcError>(())
```

`RotorCommands::shortfall` is `[thrust; τ]` minus the command that the thrusts produce, so
`A·f = [thrust; τ] − shortfall`. A negative thrust component means that the collective went
up to keep the torque. `RotorCommands::saturated` is `true` when the shortfall is not zero. An
outer loop can use this flag for anti-windup.

## Rotor lag

`RotorLag` holds one thrust per rotor and follows the command by `ḟ = (f_cmd − f)/τ`, with `τ` the
time constant. After one `τ`, the gap closes by `1 − e⁻¹ ≈ 63 %`.

- `RotorLag::new(time_constant, timestep)` fixes the tick, so `stepped` is
  `f ← e^(−Δt/τ)·f + (1 − e^(−Δt/τ))·f_cmd`: two multiplies per rotor.
- The update is exact for a command held constant across the tick. The thrust cannot overshoot
  or oscillate at any tick length. A command that changes inside a tick is not followed exactly.
- `stepped_over(commanded, timestep)` and `try_stepped_over` take a variable tick. Use
  `try_stepped_over` if the tick is not already validated: a negative tick makes
  `e^(−Δt/τ) > 1` and the thrust grows without bound.
- `rate` returns `ḟ`, for a caller that integrates the thrusts in its own state. It is not
  clamped.
- `try_with_thrust_limits` gives the lag the mixer limits. Each step then clamps the thrusts, so a
  command that did not go through the mixer also stays inside the limits.

The chain is `rotor_thrusts` (commanded), `RotorLag::stepped` (delivered one tick later), then
`wrench`.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::plant::{MultirotorMixer, RotorLag};

// X quadrotor: 0.15 m arms, yaw torque 0.016 N·m per N of thrust, 0 N to 5 N per rotor.
let arm_length = 0.15_f64;
let torque_per_thrust = 0.016;
let minimum_thrust = 0.0;
let maximum_thrust = 5.0;
let mixer = MultirotorMixer::<4, f64>::quadrotor_x(
    arm_length,
    torque_per_thrust,
    minimum_thrust,
    maximum_thrust,
)?;

// τ = 20 ms, 1 kHz loop.
let lag_time = 0.02;
let tick = 0.001;
let mut rotors = RotorLag::<4, f64>::new(lag_time, tick)?
    .try_with_thrust_limits(minimum_thrust, maximum_thrust)?;
assert_eq!(rotors.maximum_thrust(), maximum_thrust);

// Hover command at 0.8 kg, from zero thrust.
let mass = 0.8;
let gravity_strength = 9.81;
let weight = mass * gravity_strength;
let no_turn = Vector::new([0.0, 0.0, 0.0]);
let asked_for = mixer.rotor_thrusts(weight, no_turn).thrusts();

let first_tick = rotors.stepped(asked_for);
assert!(first_tick[0] < asked_for[0]);

// After one τ, the closed fraction is 1 − e⁻¹.
let ticks_in_one_lag_time = (lag_time / tick) as usize;
for _ in 1..ticks_in_one_lag_time {
    let _ = rotors.stepped(asked_for);
}
let closed_fraction = rotors.thrusts()[0] / asked_for[0];
let closed_in_one_lag_time = 1.0 - (-1.0_f64).exp();
assert!((closed_fraction - closed_in_one_lag_time).abs() < 1e-12);

// At steady state, the delivered wrench equals the command.
let long_enough_to_settle = 2000;
for _ in 0..long_enough_to_settle {
    let _ = rotors.stepped(asked_for);
}
let felt = mixer.wrench(rotors.thrusts());
assert!((felt.force()[2] - weight).abs() < 1e-12);
# Ok::<(), multicalc::CalcError>(())
```

## Position servo

`PositionServo` models a joint that takes a commanded position, as on most industrial arms and
hobby servos. Per joint, the closed loop is

```text
q̈ = ω²·(q_cmd − q) − 2ζω·q̇
A = [[0, 1], [−ω², −2ζω]]      B = [[0], [ω²]]      u = q_cmd
```

- `PositionServo::new` computes the exact zero-order-hold pair `(F, G)` once. Each `stepped` tick
  is `[q; q̇] ← F·[q; q̇] + G·q_cmd`, one 2×2 product per joint.
- The exact discretization is unconditionally stable at any tick length. A tick of fifty natural
  periods gives the same result as a thousand short ticks.
- `stepped_over(cmd, timestep)` recomputes `(F, G)`, at the cost of one matrix exponential per
  joint.
- `uniform` uses the same `ω` and `ζ` on each joint. `with_state` sets a non-zero initial state.

The servo is the stiff linear half of an operator-split step. The nonlinear body is the other
half, advanced by the [`ode`](ode-integrators.md) RK4 or RK45, whose stable tick is set by the
fastest mode. An exact servo sub-step removes that mode from the explicit integrator.

| Splitting | Sequence | Order in `Δt` |
|---|---|---|
| Lie–Trotter | `stepped` over `Δt`, then the body over `Δt` | First |
| Strang | `stepped_over(cmd, Δt/2)`, the body over `Δt`, then `stepped_over(cmd, Δt/2)` | Second |

The unconditional stability applies to the servo sub-step only. The body keeps the stability of
its own integrator.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::plant::PositionServo;

// Two joints, ω = 50 rad/s, ζ = 1, 1 kHz loop.
let natural_frequency = 50.0_f64;
let tick = 0.001;
let mut joints = PositionServo::<2, f64>::uniform(natural_frequency, 1.0, tick)?;
let commanded = Vector::new([0.4, -0.2]);

// Critically damped from rest: q(t) = q_cmd·(1 − (1 + ωt)·e^(−ωt)).
let ticks = 20;
for _ in 0..ticks {
    let _ = joints.stepped(commanded);
}
let elapsed = tick * ticks as f64;
let settled = 1.0 - (1.0 + natural_frequency * elapsed) * (-natural_frequency * elapsed).exp();
assert!((joints.positions()[0] - 0.4 * settled).abs() < 1e-12);

// Steady state: q = q_cmd.
for _ in 0..5000 {
    let _ = joints.stepped(commanded);
}
assert!((joints.positions()[1] + 0.2).abs() < 1e-12);
# Ok::<(), multicalc::CalcError>(())
```

[`JointPdController`](control.md) is the torque-side model of the same closed loop: the caller
computes the torque, not the joint servo.

## Errors

All errors are [`PlantError`](error-handling.md).

| Call | Errors |
|---|---|
| `MultirotorMixer::new`, `quadrotor_x` | `NonFinite`, `NonPositiveArmLength`, `NonPositiveTorqueRatio`, `InvalidThrustLimits`, `RotorLayoutNotIndependent`, `Linalg` |
| `RotorLag::new` | `NonFinite`, `NonPositiveTimeConstant`, `NonPositiveTimestep` |
| `RotorLag::try_with_thrust_limits` | `NonFinite`, `InvalidThrustLimits` |
| `RotorLag::try_stepped_over` | `NonFinite`, `NonPositiveTimestep` |
| `PositionServo::new`, `uniform` | `NonFinite`, `NonPositiveNaturalFrequency`, `NegativeDampingRatio`, `NonPositiveTimestep`, `Linalg` |
| `PositionServo::stepped_over` | `NonFinite`, `NonPositiveTimestep`, `Linalg` |

`RotorLayoutNotIndependent` means that the layout cannot produce each thrust and torque direction,
for example fewer than four rotors or collinear rotors. `MultirotorMixer::new` finds the rank of
the allocation with the `matrix_rank` rule `σ > rotor_count·ε·σ_max`, so `f32` and `f64` accept
the same layouts. These per-tick calls are infallible: `rotor_thrusts`, `wrench`,
`RotorLag::stepped`, `RotorLag::stepped_over`, `rate`, and `PositionServo::stepped`. They do not
check a non-finite command; it gives a non-finite output.

[Rigid-body dynamics](rigid-body-dynamics.md) applies the wrench to the body. Demo:
[rigid_body_dynamics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/rigid_body_dynamics.rs).

---

[Back to the tutorial index](README.md)
