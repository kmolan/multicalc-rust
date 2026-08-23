# Motion

The path a controller follows. `PolylinePath` is an ordered set of waypoints joined by straight
segments, stored as a fixed array of `MAX_POINTS` with a runtime length, so it is stack-allocated
and needs no heap. It answers the two questions a path-following law asks every tick: where am I on
the path, and what point should I aim at.

- `PolylinePath<MAX_POINTS, DIMENSION, T>`: built with `try_from_points` from a slice, or `new` plus
  `push` one waypoint at a time. Duplicate consecutive waypoints are accepted; every query treats a
  zero-length segment as contributing no arc length.
- `total_arc_length`: the distance along the whole path, zero for fewer than two waypoints.
- `closest_point`: projects a query point onto the path, returning a `PathProjection` with the
  `point` found, its `segment_index`, its `arc_length` from the start, and the `distance` from the
  query point to it.
- `lookahead_point`: the point a given distance further along from a given arc length — the aim
  point for `pure_pursuit_curvature`.
- `EndOfPath`: what a lookahead does when it runs off the end, `Stop` (clamp to the last waypoint,
  the default) or `Loop` (wrap to the start). Set it with `with_end_of_path`.

```rust
use multicalc::{EndOfPath, PolylinePath};
use multicalc::Vector;

// An L-shaped path: three units east, then four units north.
let path: PolylinePath<3, 2, f64> = PolylinePath::try_from_points(&[
    Vector::new([0.0, 0.0]),
    Vector::new([3.0, 0.0]),
    Vector::new([3.0, 4.0]),
])
.unwrap()
.with_end_of_path(EndOfPath::Loop);

let total = path.total_arc_length();                        // 7.0

// Where is a robot sitting off to the side of the first leg?
let robot_position = Vector::new([2.0, 0.5]);
let here = path.closest_point(robot_position).unwrap();
let on_path = here.point();                                 // (2.0, 0.0)
let travelled = here.arc_length();                          // 2.0
let cross_track = here.distance();                          // 0.5

// Aim one unit further along than that.
let lookahead_distance = 1.0;
let aim = path.lookahead_point(travelled, lookahead_distance).unwrap();   // (3.0, 0.0)
```

`try_from_points` and `push` return [`MotionError::CapacityExceeded`] if there is no room for the
waypoint and [`MotionError::NonFinite`] if any coordinate is not finite. `closest_point` and
`lookahead_point` return [`MotionError::PathTooShort`] on an empty path.

Demo: `2d_localization_obstacle_avoidance` drives a lap of this kind of path under pure pursuit.

## Minimum-snap trajectories

`PolylinePath` says where to go. `MinimumSnapPlanner` says how to get there smoothly, and when.
Minimum snap means the path keeps the fourth rate of change of position as small as it can over the
whole route — that is the quantity a quadrotor's motors have to produce, which is why it is the usual
choice for one. The result passes through every waypoint exactly and is smooth in position, velocity,
acceleration and jerk everywhere, including across the joins.

The planner takes two capacity parameters. `MAX_SEGMENTS` is how many pairs of waypoints it can hold.
`MAX_FREE_DERIVATIVES` must be at least `3 × (MAX_SEGMENTS - 1)`, because three values are chosen at
each waypoint between the first and last; stable Rust cannot work that out from `MAX_SEGMENTS`, so it
is given separately and checked at runtime.

| Segments | `MAX_FREE_DERIVATIVES` needed |
|---|---|
| 4 | 9 |
| 8 | 21 |
| 12 | 33 |

Too small gives [`MotionError::WorkspaceTooSmall`] rather than a panic.

**Planning is a one-off cost; evaluating is the per-tick one.** Planning grows with the number of
waypoints and factorizes a matrix, so it is not bounded per tick, and it is not small on the stack
either — a three-segment path in three dimensions measures about 12.5 KB on a Cortex-M4, growing with
the square of `MAX_FREE_DERIVATIVES`. Evaluating the result is fixed work costing a few hundred bytes.
On a chip, plan on the host and ship the trajectory.

`durations_from_average_speed` gives a reasonable first set of segment times from a target speed. It
is a starting point for tuning rather than the fastest possible timing: a sharp corner needs more time
than its straight-line distance suggests, because the path has to slow down to turn.

The trajectory comes back as a [`PiecewisePolynomial`](polynomials.md) from the polynomial module, so
its calls report a [`PolynomialError`](error-handling.md) rather than a `MotionError`, and it calls a
segment's length a **span** where this module calls it a **duration**. They are the same number.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::motion::MinimumSnapPlanner;

// Three segments in three dimensions, from a standstill to a standstill.
let planner = MinimumSnapPlanner::<4, 9, 3, f64>::new();
let waypoints = [
    Vector::new([0.0, 0.0, 0.0]),
    Vector::new([1.0, 2.0, 0.5]),
    Vector::new([3.0, 1.0, 1.5]),
    Vector::new([4.0, 3.0, 1.0]),
];
let trajectory = planner.plan(&waypoints, &[1.0, 1.5, 1.2]).unwrap();

// Partway along: position, velocity and acceleration in one call.
let [position, velocity, _acceleration] = trajectory.evaluate_with_derivatives::<3>(1.75).unwrap();
assert!(position[0] > 1.0 && position[0] < 3.0);
assert!(velocity.norm() > 0.0);

// It arrives at the second waypoint at the end of the first segment.
let arrival = trajectory.evaluate(1.0).unwrap();
assert!((arrival[0] - 1.0).abs() < 1e-9);
```

`plan` reports [`MotionError::PathTooShort`] for fewer than two waypoints,
[`MotionError::CapacityExceeded`] for more segments than fit,
[`MotionError::SegmentCountMismatch`] when the duration count does not match,
[`MotionError::DurationNotPositive`], [`MotionError::NonFinite`], and
[`MotionError::WorkspaceTooSmall`].

Demo: [minimum_snap_trajectory.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/minimum_snap_trajectory.rs).

## Point-to-point profiles

`PolylinePath` says where to go and `MinimumSnapPlanner` says how to get smoothly through a set of
waypoints. A profile answers a different question: how to get from here to there as fast as the
hardware allows and no faster.

- `ProfileLimits::try_new(speed, acceleration, jerk)`: the limits, jerk optional. Any of them zero,
  negative or not finite gives [`MotionError::LimitNotPositive`].
- `ProfileStrategy`: `Automatic` (the default), `Trapezoidal`, `JerkLimited`, set with
  `with_strategy`. `Automatic` reads the limits — a jerk limit means seven phases, none means three.
- `MotionProfilePlanner::new(limits).plan(distance)`: the planned move. A negative distance moves the
  other way; zero gives a profile of no duration.
- `MotionProfile`: `duration`, `distance`, `peak_speed`, `phase_durations`, `state_at`,
  `stretched_to`.
- `SynchronizedProfile::from_profiles` / `try_from_profiles_over`: several axes finishing together.
  A requested duration shorter than the slowest axis needs is replaced by that axis's duration rather
  than refused, so read `duration()` back if it matters.

The seven phases are: ease the acceleration in, hold it, ease it out, hold the speed, ease the
deceleration in, hold it, ease it out. A trapezoid is the same seven with the four jerk phases
lasting no time, its acceleration stepping where the jerk-limited shape ramps. `state_at` is
right-continuous, so a trapezoid reports full acceleration at `t = 0`.

**Planning is the one-off cost, `state_at` the per-tick one** — but unlike minimum snap, both are
bounded: planning is a fixed handful of square and cube roots with nothing to factorize, and
`state_at` walks at most seven phases and evaluates one cubic. Either is safe on a chip. A
`MotionProfile<f64>` is 296 bytes.

```rust
use multicalc::motion::{MotionProfilePlanner, ProfileLimits, SynchronizedProfile};

// Ten metres, no faster than 2 m/s, no harder than 1 m/s², easing at 2 m/s³.
let limits = ProfileLimits::<f64>::try_new(2.0, 1.0, Some(2.0)).unwrap();
let profile = MotionProfilePlanner::new(limits).plan(10.0).unwrap();

assert!((profile.duration() - 7.5).abs() < 1e-12);          // 7.5 s in all
assert!((profile.peak_speed() - 2.0).abs() < 1e-12);        // at the speed limit

// What a control loop asks for, once a tick.
let state = profile.state_at(3.75).unwrap();
assert!((state.velocity - 2.0).abs() < 1e-12);              // mid-cruise
assert!(state.acceleration.abs() < 1e-12);

// Two joints with different limits, made to finish together.
let fast = MotionProfilePlanner::new(ProfileLimits::<f64>::try_new(2.0, 1.0, None).unwrap());
let slow = MotionProfilePlanner::new(ProfileLimits::<f64>::try_new(0.5, 1.0, None).unwrap());
let both = SynchronizedProfile::from_profiles([
    fast.plan(1.0).unwrap(),
    slow.plan(1.0).unwrap(),
]);

let end = both.state_at(both.duration()).unwrap();
assert!((end.position[0] - 1.0).abs() < 1e-12);
assert!((end.position[1] - 1.0).abs() < 1e-12);
```

`plan` reports [`MotionError::NonFinite`] for a distance that is not finite and
[`MotionError::JerkLimitRequired`] for `ProfileStrategy::JerkLimited` against limits with no jerk
ceiling; `state_at` reports [`MotionError::NonFinite`] for a time that is not finite;
`try_from_profiles_over` reports [`MotionError::DurationNotPositive`].

Demo: [motion_profiles.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/motion_profiles.rs).


---

[Back to the tutorial index](README.md)
