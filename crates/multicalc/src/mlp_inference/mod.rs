//! Forward-pass inference for a multi-layer perceptron, over borrowed parameters.
//!
//! A learned policy is a stack of dense layers. Each forms one weighted sum per output —
//! `weights · input + biases` — and passes every sum through a scalar [`Activation`]. One layer's
//! output is the next layer's input, and the last one's is the action the policy was trained to
//! produce. Only inference lives here; training belongs on a machine with room for it.
//!
//! The parameters are borrowed rather than owned, because a policy is large next to the board
//! running it: two 64-wide hidden layers over a 22-component observation is some 23 KB as `f32`,
//! against a small Cortex-M's 64 KB of RAM. A [`Layer`] holds a
//! [`MatrixView`](crate::linear_algebra::MatrixView) of its weights and a
//! [`VectorView`](crate::linear_algebra::VectorView) of its biases, so nothing is copied and only
//! the activations are written.
//!
//! A trained network arrives as one flat block of numbers, so [`ParameterCursor`] walks it. Each
//! layer's shape is declared once at the point it is read, and the offsets follow from the shapes
//! rather than being written out by hand.
//!
//! Widths are const parameters, so a mismatched chain is a build error. Nothing allocates and
//! nothing panics, so this runs under `no_std`.
//!
//! ```
//! use multicalc::linear_algebra::Vector;
//! use multicalc::mlp_inference::{Activation, ParameterCursor};
//!
//! // One flat block, the way a trained policy arrives: a 2 -> 3 -> 1 network.
//! let parameters = [
//!     0.5, -0.5, 1.0, 0.0, -1.0, 2.0, // 3x2 hidden weights, row-major
//!     0.0, 1.0, -1.0, // 3 hidden biases
//!     1.0, 1.0, 1.0, // 1x3 output weights
//!     0.5, // 1 output bias
//! ];
//! let mut cursor = ParameterCursor::new(&parameters);
//! let hidden = cursor.try_take_layer::<3, 2>(Activation::Relu)?;
//! let output = cursor.try_take_layer::<1, 3>(Activation::Identity)?;
//! assert!(cursor.is_empty());
//!
//! let observation = Vector::new([2.0, 1.0]);
//! let activations = hidden.forward(observation.view());
//!
//! // The third hidden unit sums to -1.0, so the rectifier switches it off.
//! assert_eq!(activations.into_array(), [0.5, 3.0, 0.0]);
//! assert_eq!(output.forward(activations.view()).into_array(), [4.0]);
//! # Ok::<(), multicalc::error::LinalgError>(())
//! ```
//!
//! # Non-finite values
//!
//! A layer holds no state, so a non-finite observation spoils one call and nothing after it.
//! What it does to that call depends on the activation:
//!
//! | Activation | NaN | Infinity |
//! |---|---|---|
//! | [`Relu`](Activation::Relu) | **`0`** — every comparison against NaN is false | `+inf` passes, `-inf` clamps to `0` |
//! | [`Tanh`](Activation::Tanh) | `NaN` | `±1` |
//! | [`Identity`](Activation::Identity) | `NaN` | `±inf` |
//!
//! The first row is the hazard: a NaN does not propagate through a rectifier, it becomes an
//! ordinary `0` and the layers after it compute normally.
//!
//! Parameters are checked once, at load, by [`Layer::try_from_slices`]; observations per call, on
//! request, by [`Layer::forward_checked`]. [`Layer::forward`] inspects nothing, and parameters that
//! reach a layer any other way are unchecked: [`Layer::new`] takes its views as they come, and
//! [`ParameterCursor::try_take_layer`] builds through it, so a network loaded from a flat export by
//! the cursor is never read for finiteness. No check promises a finite result.

mod layer;
pub use layer::{Activation, Layer};

mod cursor;
pub use cursor::ParameterCursor;
