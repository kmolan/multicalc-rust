# MLP inference

Running a learned policy on the robot, over parameters that are never copied.

- `Layer`: one dense layer — `activation(weights · input + biases)` — holding borrowed views of its
  weights and biases rather than owning them.
- `Activation`: the scalar function applied to each output, `Relu`, `Tanh`, or `Identity`.

A multi-layer perceptron is a stack of dense layers. Each takes the vector below it, forms one
weighted sum per output, and passes every sum through an activation. Row `i` of the weight matrix is
the recipe for output `i`: it is multiplied component by component with the whole input, summed, and
offset by bias `i`. One layer's output is the next layer's input, and the last layer's output is
whatever the policy was trained to produce — joint torques, rotor commands, a steering angle.

The activation is the only nonlinear step, and without it depth buys nothing: two affine maps
composed are still one affine map, since `W₂·(W₁·x + b₁) + b₂` is `(W₂·W₁)·x + (W₂·b₁ + b₂)`.
`Relu` clamps a negative sum to zero, which costs one comparison and no `libm` call, and is the
usual choice inside a network. `Identity` is the usual choice on the output layer, where the value
is a physical quantity that should be reported rather than squashed into a range.

Only inference lives here. Training happens on a machine with room for it; what arrives on the robot
is a block of numbers read in order.

## Why the parameters are borrowed

Two hidden layers 64 units wide over a 22-component observation come to about 5,900 numbers, some
23 KB as `f32`. A small Cortex-M has 64 KB of RAM in total, and the weights are in flash. Owning
them would copy that 23 KB onto the stack, and a control loop running at a kilohertz would do it a
thousand times a second.

So a `Layer` holds a [`MatrixView`](linear-algebra.md) of its weights and a `VectorView` of its
biases: a slice, an offset, and a stride, pointing at wherever the parameters already are. Nothing
is copied to build a layer. Running one writes `OUTPUT` numbers — the activations — rather than
`OUTPUT × INPUT`.

Widths are const parameters, so the shape of a network is settled when it compiles. Feeding a layer
that produces three values into one that expects four does not build, rather than failing on the
robot. Nothing is allocated and nothing panics, so this runs under `no_std`.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::mlp_inference::{Activation, Layer};

// A trained policy arrives as one flat block. This is a 2 -> 3 -> 1 network: each layer's weights
// row-major, then its biases, in the order the layers run.
let parameters = [
    0.5, -0.5, 1.0, 0.0, -1.0, 2.0, // 3x2 hidden weights
    0.0, 1.0, -1.0,                 // 3 hidden biases
    1.0, 1.0, 1.0,                  // 1x3 output weights
    0.5,                            // 1 output bias
];

// Walking the block hands each layer its own run of it. Nothing is copied out.
let (hidden_weights, rest) = parameters.split_at(6);
let (hidden_biases, rest) = rest.split_at(3);
let (output_weights, output_biases) = rest.split_at(3);

let hidden = Layer::<3, 2>::try_from_slices(hidden_weights, hidden_biases, Activation::Relu)?;
let output = Layer::<1, 3>::try_from_slices(output_weights, output_biases, Activation::Identity)?;

// One control step: an observation in, an action out.
let observation = Vector::new([2.0, 1.0]);
let activations = hidden.forward(observation.view())?;

// The third hidden unit sums to -1.0, so the rectifier switches it off.
assert_eq!(activations.into_array(), [0.5, 3.0, 0.0]);
assert_eq!(output.forward(activations.view())?.into_array(), [4.0]);
# Ok::<(), multicalc::CalcError>(())
```

## Reading a layer's parameters yourself

`try_from_slices` reads a run of the buffer row-major and is the shortest path from a flat export.
When the parameters are already viewed — a block of a larger matrix, or a transposed export —
`Layer::new` takes the views directly, and every reshaping the views offer is available first:

```rust
use multicalc::linear_algebra::{MatrixView, Vector, VectorView};
use multicalc::mlp_inference::{Activation, Layer};

// An exporter that writes columns first leaves the weights transposed. A view fixes that by
// swapping the strides, without moving a number.
let column_major = [1.0, 0.0, 0.0, 1.0];
let biases = [0.0, 0.0];
let layer = Layer::new(
    MatrixView::<2, 2>::try_from_row_major_slice(&column_major)?.transposed(),
    VectorView::<2>::try_from_slice(&biases)?,
    Activation::Identity,
);

let input = Vector::new([2.0, -3.0]);
assert_eq!(layer.forward(input.view())?, input);
# Ok::<(), multicalc::CalcError>(())
```

Getting that orientation wrong is the failure worth guarding against: a transposed weight matrix
produces a network that runs cleanly and answers wrongly. A square layer will not even fail to
compile. Check an export against known input/output pairs rather than against the shapes alone.

## The activations

| Variant | Value | Where it belongs |
|---|---|---|
| `Relu` | `max(0, x)` | Hidden layers. One comparison, no `libm` call. |
| `Tanh` | `tanh(x)`, in `(-1, 1)` | Where a bounded output matters, at one `libm` call per component. |
| `Identity` | `x` | Output layers reporting a physical quantity. |

`Activation` is `#[non_exhaustive]`, so more can be added without breaking a caller's `match`.

Errors are [`LinalgError::OutOfBounds`](error-handling.md), returned when a slice is too short for
the shape a layer declares. A slice longer than the shape is fine — the trailing elements are simply
not part of the layer, which is what lets successive layers share one buffer.
