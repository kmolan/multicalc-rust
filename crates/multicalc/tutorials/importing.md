# Importing

Glob-import the prelude for the traits and one-call functions. Then name the types from the crate
root. The examples in this guide spell out their imports, but `use multicalc::prelude::*;` covers
the traits in all of them.

```rust
use multicalc::prelude::*;
use multicalc::{KalmanFilter, Matrix, Vector};
```

Most public types are re-exported at `multicalc::Type`. Some items stay in their module only:

- Free functions whose names need the module as context, for example
  `multicalc::vector_field::curl_3d` and `multicalc::gaussian_tables::nodes`.
- Some types, for example `multicalc::mlp_inference::Layer` and
  `multicalc::linear_algebra::MatrixView`.

## One-call functions and strategy objects

The calculus modules give two ways to compute a result:

- **One-call functions:** `derivative`, `second_derivative`, `partial` and `integral`. They need no
  imported trait and no configuration. The derivatives use exact automatic differentiation. Use
  these first.
- **Strategy objects:** `AutoDiffSingle`, `FiniteDifferenceSingle`, `IterativeSingle`,
  `GaussianSingle` and their multi-variable forms. Use them to select the method, the step size,
  the iteration count, or a derivative order above the second.

Both give the same results. The strategy objects only expose the parameters.

---

[Back to the tutorial index](README.md)
