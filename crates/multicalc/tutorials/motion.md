# Motion

Paths, smooth trajectories and time-optimal profiles. Everything is fixed-size and stack-allocated.

## Paths

`PolylinePath` is an ordered set of waypoints joined by straight segments, stored as a fixed array of
`MAX_POINTS` with a runtime length. It answers the two queries of a path-following law: the current
position on the path, and the aim point.

- `PolylinePath<MAX_POINTS, DIMENSION, T>`: built with `try_from_points` from a slice, or with `new`
  and then `push` for each waypoint. Duplicate consecutive waypoints are accepted; a zero-length
  segment adds no arc length.
- `total_arc_length`: the length of the whole path, zero for fewer than two waypoints.
- `closest_point`: projects a query point onto the path. It returns a `PathProjection` with the
  `point`, its `segment_index`, its `arc_length` from the start, and the `distance` from the query
  point.
- `lookahead_point`: the point a given distance after a given arc length. It is the aim point for
  `pure_pursuit_curvature`.
- `EndOfPath`: the behavior of a lookahead past the end. `Stop` (the default) clamps to the last
  waypoint, and `Loop` wraps to the start. Set it with `with_end_of_path`.

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

// A robot to the side of the first leg.
let robot_position = Vector::new([2.0, 0.5]);
let here = path.closest_point(robot_position).unwrap();
let on_path = here.point();                                 // (2.0, 0.0)
let travelled = here.arc_length();                          // 2.0
let cross_track = here.distance();                          // 0.5

// Aim one unit further along.
let lookahead_distance = 1.0;
let aim = path.lookahead_point(travelled, lookahead_distance).unwrap();   // (3.0, 0.0)
```

`try_from_points` and `push` return `MotionError::CapacityExceeded` if there is no space for the
waypoint, and `MotionError::NonFinite` if a coordinate is not finite. `closest_point` and
`lookahead_point` return `MotionError::PathTooShort` on an empty path, and `MotionError::OutOfSync`
if the cumulative arc length is not available.

Demo:
[2d_localization_obstacle_avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/2d_localization_obstacle_avoidance.rs)
drives a lap of this type of path under pure pursuit.

## Minimum-snap trajectories

`PolylinePath` gives the geometry. `MinimumSnapPlanner` gives a smooth, timed trajectory through it.
It minimizes the integral of squared snap (the fourth derivative of position) over the route, with
one degree-7 polynomial per segment. Snap is what the motors of a multirotor must produce, so this
is the usual choice for one. The trajectory passes through each waypoint exactly and is continuous
in position, velocity, acceleration and jerk, also across the joins.

The boundary conditions default to rest at both ends. `with_start` and `with_end` take a
`BoundaryDerivatives` (velocity, acceleration, jerk; `Default` is all zeros) to set other values.

The planner has two capacity parameters. `MAX_SEGMENTS` is the maximum number of segments.
`MAX_FREE_DERIVATIVES` must be at least `3 × (MAX_SEGMENTS - 1)`, because the solve chooses three
values at each interior waypoint. Stable Rust cannot compute that from `MAX_SEGMENTS`, so it is a
separate parameter, checked at runtime.

| Segments | `MAX_FREE_DERIVATIVES` needed |
|---|---|
| 4 | 9 |
| 8 | 21 |
| 12 | 33 |

A value that is too small gives `MotionError::WorkspaceTooSmall`, not a panic.

**Planning is a one-off cost; evaluation is the per-tick cost.** Planning grows with the number of
waypoints and factorizes a matrix, so its time per call is not bounded. Its stack use is also large:
a three-segment path in three dimensions measures approximately 12.5 KB on a Cortex-M4, and it grows
with the square of `MAX_FREE_DERIVATIVES`. Evaluation is fixed work and costs a few hundred bytes. On
a microcontroller, plan on the host and send the trajectory.

`durations_from_average_speed` gives initial segment durations from a target speed. It is a start
point for tuning, not the time-optimal timing: a sharp corner needs more time than its straight-line
distance shows, because the trajectory must decelerate to turn.

The trajectory is a [`PiecewisePolynomial`](polynomials.md), so its methods return a
[`PolynomialError`](error-handling.md), not a `MotionError`. It calls the length of a segment a
**span**, where this module calls it a **duration**. They are the same number.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::motion::MinimumSnapPlanner;

// Three segments in three dimensions, from rest to rest.
let planner = MinimumSnapPlanner::<4, 9, 3, f64>::new();
let waypoints = [
    Vector::new([0.0, 0.0, 0.0]),
    Vector::new([1.0, 2.0, 0.5]),
    Vector::new([3.0, 1.0, 1.5]),
    Vector::new([4.0, 3.0, 1.0]),
];
let trajectory = planner.plan(&waypoints, &[1.0, 1.5, 1.2]).unwrap();

// Position, velocity and acceleration in one call.
let [position, velocity, _acceleration] = trajectory.evaluate_with_derivatives::<3>(1.75).unwrap();
assert!(position[0] > 1.0 && position[0] < 3.0);
assert!(velocity.norm() > 0.0);

// The trajectory reaches the second waypoint at the end of the first segment.
let arrival = trajectory.evaluate(1.0).unwrap();
assert!((arrival[0] - 1.0).abs() < 1e-9);
```

`plan` returns:

- `MotionError::PathTooShort` for fewer than two waypoints.
- `MotionError::CapacityExceeded` for more segments than `MAX_SEGMENTS`.
- `MotionError::SegmentCountMismatch` when the number of durations is not one per segment.
- `MotionError::DurationNotPositive` for a duration that is zero, negative or NaN.
- `MotionError::NonFinite` for a non-finite waypoint or boundary value.
- `MotionError::WorkspaceTooSmall` when `MAX_FREE_DERIVATIVES` is below `3 × (segments - 1)`.
- `MotionError::Linalg` when the linear system has no solution.

Demo: [minimum_snap_trajectory.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/minimum_snap_trajectory.rs).

## Point-to-point profiles

A profile gives the time-optimal move from one position to another within the speed, acceleration
and jerk limits.

- `ProfileLimits::try_new(speed, acceleration, jerk)`: the limits; `jerk` is an `Option`. A limit
  that is zero, negative or not finite gives `MotionError::LimitNotPositive`.
- `ProfileStrategy`: `Automatic` (the default), `Trapezoidal` or `JerkLimited`, set with
  `with_strategy`. `Automatic` uses seven phases if there is a jerk limit, and three if not.
- `MotionProfilePlanner::new(limits).plan(distance)`: the planned move. A negative distance moves in
  the opposite direction; zero gives a profile of zero duration.
- `MotionProfile`: `duration`, `distance`, `peak_speed`, `phase_durations`, `state_at`,
  `stretched_to`.
- `SynchronizedProfile::from_profiles` / `try_from_profiles_over`: time-scales several axes to finish
  together. If the requested duration is shorter than the slowest axis needs, the duration of that
  axis replaces it, with no error. Read `duration()` back to see which value applies.

The seven phases are: jerk up, constant acceleration, jerk down, cruise, jerk down, constant
deceleration, jerk up. A trapezoid is the same seven phases with zero-length jerk phases, so its
acceleration steps where the jerk-limited profile ramps. `state_at` is right-continuous, so a
trapezoid reports full acceleration at `t = 0`.

**Planning is the one-off cost, `state_at` the per-tick cost.** Unlike minimum snap, both are
bounded. Planning is a small fixed set of square and cube roots, with no factorization. `state_at`
walks a maximum of seven phases and evaluates one cubic. Both are safe on a microcontroller. A
`MotionProfile<f64>` is 296 bytes.

```rust
use multicalc::motion::{MotionProfilePlanner, ProfileLimits, SynchronizedProfile};

// 10 m, at a maximum of 2 m/s and 1 m/s², with a jerk limit of 2 m/s³.
let limits = ProfileLimits::<f64>::try_new(2.0, 1.0, Some(2.0)).unwrap();
let profile = MotionProfilePlanner::new(limits).plan(10.0).unwrap();

assert!((profile.duration() - 7.5).abs() < 1e-12);          // 7.5 s in total
assert!((profile.peak_speed() - 2.0).abs() < 1e-12);        // at the speed limit

// The per-tick query of a control loop.
let state = profile.state_at(3.75).unwrap();
assert!((state.velocity - 2.0).abs() < 1e-12);              // mid-cruise
assert!(state.acceleration.abs() < 1e-12);

// Two joints with different limits, synchronized to finish together.
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

`plan` returns `MotionError::NonFinite` for a non-finite distance, and
`MotionError::JerkLimitRequired` for `ProfileStrategy::JerkLimited` with no jerk limit. `state_at`
returns `MotionError::NonFinite` for a non-finite time. `try_from_profiles_over` returns
`MotionError::DurationNotPositive` for a duration that is zero, negative or not finite.

Demo: [motion_profiles.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/motion_profiles.rs).

---

[Back to the tutorial index](README.md)
