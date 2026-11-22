# Importing

There is one answer: glob the prelude for the traits and one-call functions, then name the types
you need from the crate root. Every public type lives at `multicalc::Type`, so you never have to
know which file it is declared in. The examples below spell out their imports in full, but
`use multicalc::prelude::*;` covers the traits in all of them.

```rust
use multicalc::prelude::*;
use multicalc::{KalmanFilter, Matrix, Vector};
```

The one exception is a handful of free functions that stay on their own module, because their
names only make sense next to each other — `multicalc::vector_field::curl_3d` reads better than
`multicalc::curl_3d`.

## The easy path and the configurable one

Most calculus work has two ways in, and the guide uses both:

- **The one-call functions** — `derivative`, `second_derivative`, `partial`, `integral`. They need
  no imported trait and no configuration, and they use exact automatic differentiation. Reach for
  these first.
- **The strategy objects** — `AutoDiffSingle`, `FiniteDifferenceSingle`, `IterativeSingle`,
  `GaussianSingle` and their multi-variable siblings. These are how you choose a different method,
  a step size, an iteration count, or a derivative order above the second.

Both compute the same answers; the objects just expose the knobs.


---

[Back to the tutorial index](README.md)
