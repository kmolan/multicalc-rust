# MLP inference

Forward-pass inference for a multi-layer perceptron, over borrowed parameters. Training is out of
scope.

- `Layer`: one dense layer, `y = activation(W·x + b)`. It holds borrowed views of `W` and `b`.
- `Activation`: the element-wise nonlinearity, `Relu`, `Tanh`, or `Identity`.
- `ParameterCursor`: a read position in a flat parameter buffer. It gives one layer at a time.

Row `i` of `W` and bias `b_i` give output `i`. Without the activation, depth adds nothing:
`W₂·(W₁·x + b₁) + b₂ = (W₂·W₁)·x + (W₂·b₁ + b₂)`.

## Borrowed parameters

Two 64-wide hidden layers over a 22-component observation are about 5,900 parameters, or 23 KB as
`f32`. A small Cortex-M has 64 KB of RAM, and the weights are in flash. An owned copy would put
23 KB on the stack in each cycle of a 1 kHz loop.

A `Layer` holds a [`MatrixView`](linear-algebra.md#borrowed-views) of `W` and a `VectorView` of
`b`: a slice, an offset, and a stride. A layer build copies nothing. `forward` writes only the
`OUTPUT` activations.

Widths are const parameters, so a mismatched chain is a compile error, not a runtime fault. There
is no allocation and no panic, so the module runs under `no_std`.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::mlp_inference::{Activation, ParameterCursor};

// A 2 -> 3 -> 1 network as one flat block: per layer, weights row-major, then biases.
let parameters = [
    0.5, -0.5, 1.0, 0.0, -1.0, 2.0, // 3x2 hidden weights
    0.0, 1.0, -1.0,                 // 3 hidden biases
    1.0, 1.0, 1.0,                  // 1x3 output weights
    0.5,                            // 1 output bias
];

let mut cursor = ParameterCursor::new(&parameters);
let hidden = cursor.try_take_layer::<3, 2>(Activation::Relu)?;
let output = cursor.try_take_layer::<1, 3>(Activation::Identity)?;

// Values left over mean that the declared shapes do not match the export.
assert!(cursor.is_empty());

let observation = Vector::new([2.0, 1.0]);
let activations = hidden.forward(observation.view());

// The third hidden pre-activation is -1.0, so Relu gives 0.
assert_eq!(activations.into_array(), [0.5, 3.0, 0.0]);
assert_eq!(output.forward(activations.view()).into_array(), [4.0]);
# Ok::<(), multicalc::CalcError>(())
```

If `try_take_layer` runs past the end of the buffer, it returns `OutOfBounds` and does not move
the cursor.

## Other parameter layouts

- `Layer::try_from_slices(weights, biases, activation)` reads row-major slices and rejects a
  non-finite parameter.
- `Layer::new(weights, biases, activation)` takes views directly, for example a block of a larger
  matrix or a transposed export. It does not check the values.

`ParameterCursor::try_take_layer` builds through `Layer::new`, so it does not check for non-finite
parameters. `try_take_matrix` and `try_take_vector` give the raw views.

```rust
use multicalc::linear_algebra::{MatrixView, Vector, VectorView};
use multicalc::mlp_inference::{Activation, Layer};

// A column-major export: `transposed` swaps the strides, with no copy.
let column_major = [1.0, 0.0, 0.0, 1.0];
let biases = [0.0, 0.0];
let layer = Layer::new(
    MatrixView::<2, 2>::try_from_row_major_slice(&column_major)?.transposed(),
    VectorView::<2>::try_from_slice(&biases)?,
    Activation::Identity,
);

let input = Vector::new([2.0, -3.0]);
assert_eq!(layer.forward(input.view()), input);
# Ok::<(), multicalc::CalcError>(())
```

A transposed weight matrix gives a network that runs without error and gives wrong outputs. A
square layer compiles either way. Check an export against known input/output pairs, not only
against the shapes.

## The activations

| Variant | Value | Use |
|---|---|---|
| `Relu` | `max(0, x)` | Hidden layers. One comparison, no `libm` call. |
| `Tanh` | `tanh(x)`, in `(-1, 1)` | A bounded output, at one `libm` call per component. |
| `Identity` | `x` | Output layers that give a physical quantity, such as a torque or a steering angle. |

`Activation` is `#[non_exhaustive]`, so a caller `match` needs a wildcard arm.

## Non-finite values

A layer has no state, so a non-finite observation affects only that call. The effect depends on
the activation:

| Activation | NaN | Infinity |
|---|---|---|
| `Relu` | `0`: each comparison against NaN is false | `+inf` passes, `-inf` clamps to `0` |
| `Tanh` | `NaN` | `±1` |
| `Identity` | `NaN` | `±inf` |

The `Relu` row is the hazard. A NaN does not propagate; it becomes `0`, and the later layers
compute normally on a meaningless value. `Layer::forward_checked` rejects a non-finite
observation before any arithmetic. `forward` checks nothing. No check guarantees a finite result,
because finite values can still overflow.

## Errors

All errors are [`LinalgError`](error-handling.md):

- `OutOfBounds`: a slice is shorter than the declared shape. A longer slice is valid; the layer
  ignores the trailing elements, so consecutive layers can share one buffer.
- `NonFinite`: from `try_from_slices` on a non-finite parameter, and from `forward_checked` on a
  non-finite observation.

---

[Back to the tutorial index](README.md)
