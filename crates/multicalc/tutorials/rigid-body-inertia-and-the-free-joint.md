# Rigid-body inertia and the free joint

How a single body's mass is spread out, and where that body is and how it is moving when nothing
holds it in place.

- `SpatialInertia`: a body's mass, the point it balances about, and how it resists being spun about
  that point. `inertia_about` asks the same question about a different reference point — moving away
  from the balance point always makes the body harder to spin, by the mass times how far the point
  moved. Building one is fallible: the mass has to be positive and finite and the resistance has to
  read the same across the diagonal, so a body that cannot exist is rejected at construction.
- `FreeJointState`: the pose and velocity of a body free to move in all six directions.

Both hand their numbers back as plain arrays rather than wrapper types, so the conventions matter.
The seven place numbers are position first, then orientation, as `[x, y, z, w, qx, qy, qz]` —
matching how MuJoCo writes a free joint. The six motion numbers are `[v; ω]`, linear first, the
ordering the rest of the crate uses.

```rust
use multicalc::{FreeJointState, SpatialInertia};
use multicalc::{Matrix, SE3, Twist, Vector};

// A 2 kg body balancing at its origin.
let inertia = SpatialInertia::new(
    2.0_f64,
    Vector::new([0.0, 0.0, 0.0]),
    Matrix::from_diagonal([1.0, 1.0, 1.0]),
)?;

// Spinning it about a point one metre away is harder, by mass times distance squared.
let about_offset = inertia.inertia_about(Vector::new([1.0, 0.0, 0.0]));
assert_eq!(about_offset[(1, 1)], 3.0);

// Where the body is and how it is moving.
let state = FreeJointState::new(SE3::<f64>::identity(), Twist::zeros());
assert_eq!(state.generalized_position(), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
# Ok::<(), multicalc::CalcError>(())
```

Errors: `SpatialInertia::new` returns [`SpatialError`](error-handling.md): `NonPositiveMass`,
`NonFinite`, `NotSymmetric`, or `NonPositiveInertia`.

These two types are what a model file loads into. The separate `multicalc-mjcf` crate reads one
rigid body out of a MuJoCo MJCF file — working its mass out from the shapes it is built from where
the file does not state it — and is checked against MuJoCo's own compile of the same file. Full
demo:
[model_ingestion.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/model_ingestion.rs).


---

[Back to the tutorial index](README.md)
