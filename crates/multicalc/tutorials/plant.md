# Plant

What sits between a command and the force a body actually feels.

- `MultirotorMixer`: shares a wanted push and turn out across a set of rotors, and works the other
  way too, saying what push and turn a set of rotor thrusts adds up to.
- `RotorLag`: how quickly a rotor catches up to the thrust it was asked for, since it cannot change
  what it is giving the moment it is asked.
- `PositionServo`: a joint that takes a commanded position rather than a torque, and how its own
  servo answers. Per joint the closed loop is second order, `q̈ = ω²·(q_cmd − q) − 2ζω·q̇`, and its
  exact zero-order-hold pair is worked out once, when the model is built — so a tick is one 2×2
  product per joint and no tick length makes it blow up. It is the stiff linear half of an
  operator-split step: Lie–Trotter is `stepped` over `Δt` then the body over `Δt`, first order;
  Strang is `stepped_over(cmd, Δt/2)`, the body over `Δt`, then `stepped_over(cmd, Δt/2)`, second
  order. The unconditional-stability claim covers this linear sub-step alone. Its torque-side
  counterpart is [`JointPdController`](control.md).

Each rotor pushes along the body's z axis; where it sits decides how much it tips the body, and
which way it turns decides how much it twists it about z. The mixer holds that relation and the way
back through it, both worked out once, so asking for a push and a turn costs one small matrix
product. Its answer comes back as a `Wrench`, which is exactly what
[`RigidBody`](rigid-body-dynamics.md) takes — so the two join up with nothing in between.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::plant::MultirotorMixer;

// Four rotors 15 cm out, each twisting the body 1.6 cm-worth per newton of push, able to give
// between nothing and 5 N.
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

// A 0.8 kg machine asked to carry its own weight and turn not at all: the push is shared out
// evenly and no rotor is stretched.
let mass = 0.8;
let gravity_strength = 9.81;
let weight = mass * gravity_strength;
let no_turn = Vector::new([0.0, 0.0, 0.0]);

let commands = mixer.rotor_thrusts(weight, no_turn);
assert!(!commands.saturated());
for rotor in 0..4 {
    assert!((commands.thrusts()[rotor] - weight / 4.0).abs() < 1e-12);
}

// Those thrusts add back up to the push that was asked for, in the form the body takes.
let wrench = mixer.wrench(commands.thrusts());
assert!((wrench.force()[2] - weight).abs() < 1e-12);
assert!(wrench.torque().norm() < 1e-12);

// Asked for far more than the rotors have, every one sits at its limit and says so.
let beyond_reach = 30.0;
let too_much = mixer.rotor_thrusts(beyond_reach, no_turn);
assert!(too_much.saturated());
# Ok::<(), multicalc::CalcError>(())
```

A saturated command does not produce the push and turn that was asked for, because what was left
over has nowhere to go. `RotorCommands::saturated` says when that happened, so an outer loop can
stop winding up against a limit it cannot reach past.

## The moment a rotor takes to catch up

What the mixer hands back is what the rotors were *asked* for. A real rotor cannot change its thrust
the moment it is asked to — it has to spin up or slow down first, and while it is doing that the body
feels less than the command. `RotorLag` holds that moment. It keeps one thrust per rotor and closes
the gap to the command a fixed share at a time: quickly at first, then more slowly, never quite
arriving but soon close enough to make no difference. The lag time is how long it takes to close a
little under two thirds of the gap, and the gap shrinks by that same share again over every lag time
after that.

The tick length is fixed when the model is built, so the two numbers a tick needs are worked out
once and each tick is a couple of multiplies per rotor. Where the thrust lands is worked out
exactly rather than stepped toward, so a long tick is as safe as a short one — the thrust can never
overshoot the command or swing about it. That holds as long as the command stays still across the
tick, which is what a loop running at a fixed rate does anyway.

It sits between the two mixer calls with nothing else in between: `rotor_thrusts` says what was
asked for, `RotorLag::stepped` says what is actually being given a tick later, and `wrench` turns
that into the push and turn the body feels.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::plant::{MultirotorMixer, RotorLag};

// Four rotors 15 cm out, each twisting the body 1.6 cm-worth per newton of push.
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

// They take 20 ms to catch up, driven by a loop running every millisecond.
let lag_time = 0.02;
let tick = 0.001;
let mut rotors = RotorLag::<4, f64>::new(lag_time, tick)?;

// A 0.8 kg machine asked to carry its own weight, from a standstill.
let mass = 0.8;
let gravity_strength = 9.81;
let weight = mass * gravity_strength;
let no_turn = Vector::new([0.0, 0.0, 0.0]);
let asked_for = mixer.rotor_thrusts(weight, no_turn).thrusts();

// The first tick delivers only a sliver of it.
let first_tick = rotors.stepped(asked_for);
assert!(first_tick[0] < asked_for[0]);

// One lag time in, a little under two thirds of the gap is closed.
let ticks_in_one_lag_time = (lag_time / tick) as usize;
for _ in 1..ticks_in_one_lag_time {
    let _ = rotors.stepped(asked_for);
}
let closed_fraction = rotors.thrusts()[0] / asked_for[0];
let closed_in_one_lag_time = 1.0 - (-1.0_f64).exp();
assert!((closed_fraction - closed_in_one_lag_time).abs() < 1e-12);

// Held there, they settle on exactly what was asked for, and that drives the body directly.
let long_enough_to_settle = 2000;
for _ in 0..long_enough_to_settle {
    let _ = rotors.stepped(asked_for);
}
let felt = mixer.wrench(rotors.thrusts());
assert!((felt.force()[2] - weight).abs() < 1e-12);
# Ok::<(), multicalc::CalcError>(())
```

## A joint that takes a position, not a torque

Not every actuator takes a torque. A position-controlled joint — most industrial arms, and any
hobby servo — takes a commanded angle and runs its own loop inside, and what the outside world gets
to see is how that loop answers. `PositionServo` holds that loop, one second-order system per joint,
and advances it exactly rather than stepping toward it. Because the discretization is exact, the
tick length is free: a tick fifty natural periods long lands in the same place as a thousand short
ones, where an explicit integrator would have diverged long before.

That matters because this is normally the stiff half of a split step. The arm's own nonlinear
dynamics go through [`ode`](ode-integrators.md)'s RK4 or RK45, whose stable tick length is set by the fastest
thing in the model — and the servo is usually that fastest thing. Splitting it out and solving it
exactly takes it off the integrator's plate. Lie–Trotter is `stepped` over `Δt` then the body over
`Δt`, first order in `Δt`; Strang is `stepped_over(cmd, Δt/2)`, the body over `Δt`, then
`stepped_over(cmd, Δt/2)`, second order. The unconditional-stability claim covers this linear
sub-step alone — the body keeps whatever stability its own integrator has.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::plant::PositionServo;

// Two joints whose servos run at 50 rad/s, critically damped, driven every millisecond.
let natural_frequency = 50.0_f64;
let tick = 0.001;
let mut joints = PositionServo::<2, f64>::uniform(natural_frequency, 1.0, tick)?;
let commanded = Vector::new([0.4, -0.2]);

// Critically damped from rest: q(t) = q_cmd*(1 - (1 + w*t)*exp(-w*t)).
let ticks = 20;
for _ in 0..ticks {
    let _ = joints.stepped(commanded);
}
let elapsed = tick * ticks as f64;
let settled = 1.0 - (1.0 + natural_frequency * elapsed) * (-natural_frequency * elapsed).exp();
assert!((joints.positions()[0] - 0.4 * settled).abs() < 1e-12);

// Held there, they arrive exactly.
for _ in 0..5000 {
    let _ = joints.stepped(commanded);
}
assert!((joints.positions()[1] + 0.2).abs() < 1e-12);
# Ok::<(), multicalc::CalcError>(())
```

The same hardware seen from the torque side is
[`JointPdController`](control.md): the same closed loop, driven by a torque the caller works out
rather than by a position the joint works out for itself.

Errors: `MultirotorMixer::new` and `quadrotor_x` return [`PlantError`](error-handling.md):
`NonFinite`, `NonPositiveArmLength`, `NonPositiveTorqueRatio`, `InvalidThrustLimits`,
`RotorLayoutNotIndependent`, or `Linalg`. `RotorLag::new` returns
[`PlantError`](error-handling.md): `NonFinite`, `NonPositiveTimeConstant`, or
`NonPositiveTimestep`. `PositionServo::new` and `uniform` return `NonFinite`,
`NonPositiveNaturalFrequency`, `NegativeDampingRatio`, `NonPositiveTimestep`, or `Linalg`, and
`PositionServo::stepped_over` returns `NonFinite`, `NonPositiveTimestep`, or `Linalg` because it
works the discretization out afresh. Everything else on the per-tick path — `rotor_thrusts`,
`wrench`, `RotorLag::stepped`, `stepped_over`, `rate`, and `PositionServo::stepped` — is
infallible.

What the wrench then does to the body is in [Rigid-body dynamics](rigid-body-dynamics.md). Full
demo:
[rigid_body_dynamics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/rigid_body_dynamics.rs).


---

[Back to the tutorial index](README.md)
