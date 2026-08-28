//! Forward-pass inference for a multi-layer perceptron, over borrowed parameters.
//!
//! A learned policy is a stack of dense layers. Each takes the vector below it, forms one weighted
//! sum per output — `weights · input + biases` — and passes every sum through a scalar
//! [`Activation`]. One layer's output is the next layer's input, and the last layer's output is the
//! answer the policy was trained to give: joint torques, rotor commands, a steering angle.
//!
//! Only the forward pass lives here. Training happens elsewhere, on a machine with room for it;
//! what reaches the robot is a block of numbers to be read in order.
//!
//! That block is why the parameters are borrowed rather than owned. Two hidden layers 64 units wide
//! over a 22-component observation come to roughly 5,900 numbers — about 23 KB as `f32`, against
//! the 64 KB of RAM a small Cortex-M has in total. A [`Layer`] holds a
//! [`MatrixView`](crate::linear_algebra::MatrixView) of its weights and a
//! [`VectorView`](crate::linear_algebra::VectorView) of its biases, so those coefficients are read
//! where they were stored and never copied onto the stack. Only the activations are written, and
//! there are `OUTPUT` of those per layer rather than `OUTPUT`×`INPUT`.
//!
//! Widths are const parameters, so a network's shape is settled when it compiles: feeding a layer
//! that produces three values into one that expects four is a build error, not a runtime one.
//! Nothing is allocated and nothing panics, so this runs under `no_std`.
//!
//! ```
//! use multicalc::linear_algebra::Vector;
//! use multicalc::mlp_inference::{Activation, Layer};
//!
//! // One flat block, the way a trained policy arrives: a 2 -> 3 -> 1 network.
//! let parameters = [
//!     0.5, -0.5, 1.0, 0.0, -1.0, 2.0, // 3x2 hidden weights, row-major
//!     0.0, 1.0, -1.0, // 3 hidden biases
//!     1.0, 1.0, 1.0, // 1x3 output weights
//!     0.5, // 1 output bias
//! ];
//! let (hidden_weights, rest) = parameters.split_at(6);
//! let (hidden_biases, rest) = rest.split_at(3);
//! let (output_weights, output_biases) = rest.split_at(3);
//!
//! let hidden = Layer::<3, 2>::try_from_slices(hidden_weights, hidden_biases, Activation::Relu)?;
//! let output =
//!     Layer::<1, 3>::try_from_slices(output_weights, output_biases, Activation::Identity)?;
//!
//! let observation = Vector::new([2.0, 1.0]);
//! let activations = hidden.forward(observation.view())?;
//!
//! // The third hidden unit sums to -1.0, so the rectifier switches it off.
//! assert_eq!(activations.into_array(), [0.5, 3.0, 0.0]);
//! assert_eq!(output.forward(activations.view())?.into_array(), [4.0]);
//! # Ok::<(), multicalc::error::LinalgError>(())
//! ```

use crate::Numeric;
use crate::error::LinalgError;
use crate::linear_algebra::{MatrixView, Vector, VectorView};

/// The scalar function a layer applies to each of its outputs, shaping the raw weighted sum
/// into the range the next layer expects.
///
/// It is also the layer's only nonlinear step: without one, a chain of layers collapses into a
/// single layer no matter how deep it is.
///
/// ```
/// use multicalc::mlp_inference::Activation;
/// assert_eq!(Activation::Relu.apply(-2.0), 0.0);
/// assert_eq!(Activation::Relu.apply(3.0), 3.0);
/// assert_eq!(Activation::Identity.apply(-2.0), -2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Activation {
    /// Clamps a negative sum to zero and passes a positive one through. The usual hidden-layer
    /// choice: one comparison, no `libm` call.
    Relu,
    /// Squashes into `(-1, 1)`. Bounded, which matters when the value drives an actuator, at the
    /// cost of a `libm` call per component.
    Tanh,
    /// Passes the value through unchanged. The usual output-layer choice, where the value is a
    /// physical quantity to report rather than squash.
    Identity,
}

impl Activation {
    /// Applies the activation to one value.
    ///
    /// ```
    /// use multicalc::mlp_inference::Activation;
    /// assert_eq!(Activation::Tanh.apply(0.0), 0.0);
    /// assert_eq!(Activation::Identity.apply(1.5), 1.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn apply<T: Numeric>(self, value: T) -> T {
        match self {
            Activation::Relu => {
                if value > T::ZERO {
                    value
                } else {
                    T::ZERO
                }
            }
            Activation::Tanh => value.tanh(),
            Activation::Identity => value,
        }
    }
}

/// One dense layer of a multi-layer perceptron: `activation(weights · input + biases)`.
///
/// The parameters are borrowed, not owned, so a policy exported as one flat buffer is read where
/// it sits. Only the intermediate activations are materialized, and those are `OUTPUT` values
/// rather than `OUTPUT`×`INPUT`.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::mlp_inference::{Activation, Layer};
/// let weights = [0.5, -0.5, 1.0, 0.0, -1.0, 2.0];
/// let biases = [0.0, 1.0, -1.0];
/// let hidden = Layer::<3, 2>::try_from_slices(&weights, &biases, Activation::Relu).unwrap();
/// let input = Vector::new([2.0, 1.0]);
/// assert_eq!(hidden.forward(input.view()).unwrap().into_array(), [0.5, 3.0, 0.0]);
/// ```
#[derive(Debug)]
#[must_use]
pub struct Layer<'data, const OUTPUT: usize, const INPUT: usize, T = f64> {
    weights: MatrixView<'data, OUTPUT, INPUT, T>,
    biases: VectorView<'data, OUTPUT, T>,
    activation: Activation,
}

// Written out rather than derived: a derive would demand `T: Copy`, but what is copied is the pair
// of handles, not the parameters they point at.
impl<'data, const OUTPUT: usize, const INPUT: usize, T> Clone for Layer<'data, OUTPUT, INPUT, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'data, const OUTPUT: usize, const INPUT: usize, T> Copy for Layer<'data, OUTPUT, INPUT, T> {}

impl<'data, const OUTPUT: usize, const INPUT: usize, T> Layer<'data, OUTPUT, INPUT, T> {
    /// A layer over parameters that are already viewed.
    ///
    /// ```
    /// use multicalc::linear_algebra::{MatrixView, Vector, VectorView};
    /// use multicalc::mlp_inference::{Activation, Layer};
    /// let weights = [1.0, 0.0, 0.0, 1.0];
    /// let biases = [0.0, 0.0];
    /// let layer = Layer::new(
    ///     MatrixView::<2, 2>::try_from_row_major_slice(&weights).unwrap(),
    ///     VectorView::<2>::try_from_slice(&biases).unwrap(),
    ///     Activation::Identity,
    /// );
    /// // Identity weights, no bias, and no squashing, so the input comes back unchanged.
    /// let input = Vector::new([2.0, -3.0]);
    /// assert_eq!(layer.forward(input.view()).unwrap(), input);
    /// ```
    #[inline]
    pub const fn new(
        weights: MatrixView<'data, OUTPUT, INPUT, T>,
        biases: VectorView<'data, OUTPUT, T>,
        activation: Activation,
    ) -> Self {
        Layer {
            weights,
            biases,
            activation,
        }
    }

    /// A layer over two runs of a parameter buffer: `weights` read row-major as
    /// `OUTPUT`×`INPUT`, `biases` as `OUTPUT` components. `OutOfBounds` if either is too short;
    /// trailing elements in either slice are ignored.
    ///
    /// ```
    /// use multicalc::mlp_inference::{Activation, Layer};
    /// let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let biases = [0.0, 0.0];
    /// assert!(Layer::<2, 3>::try_from_slices(&weights, &biases, Activation::Relu).is_ok());
    /// assert!(Layer::<3, 3>::try_from_slices(&weights, &biases, Activation::Relu).is_err());
    /// ```
    #[inline]
    pub fn try_from_slices(
        weights: &'data [T],
        biases: &'data [T],
        activation: Activation,
    ) -> Result<Self, LinalgError> {
        Ok(Layer::new(
            MatrixView::try_from_row_major_slice(weights)?,
            VectorView::try_from_slice(biases)?,
            activation,
        ))
    }
}

impl<'data, const OUTPUT: usize, const INPUT: usize, T: Numeric> Layer<'data, OUTPUT, INPUT, T> {
    /// `activation(weights · input + biases)`, one output component at a time.
    ///
    /// Each output reads the whole input and nothing else, so the components are independent and
    /// the order they are computed in does not matter. The activation is applied to the finished
    /// sum, never to the individual products.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::mlp_inference::{Activation, Layer};
    /// let weights = [1.0, 1.0, 1.0];
    /// let biases = [0.5];
    /// let output = Layer::<1, 3>::try_from_slices(&weights, &biases, Activation::Identity).unwrap();
    /// let hidden = Vector::new([0.5, 3.0, 0.0]);
    /// assert_eq!(output.forward(hidden.view()).unwrap().into_array(), [4.0]);
    /// ```
    #[inline]
    pub fn forward(
        &self,
        input: VectorView<'_, INPUT, T>,
    ) -> Result<Vector<OUTPUT, T>, LinalgError> {
        let mut result = Vector::<OUTPUT, T>::zeros();
        for row_index in 0..OUTPUT {
            let weighted_sum = self.weights.try_row(row_index)?.dot(input);
            let biased = weighted_sum + *self.biases.try_get(row_index)?;
            let slot = result.get_mut(row_index).ok_or(LinalgError::OutOfBounds)?;
            *slot = self.activation.apply(biased);
        }
        Ok(result)
    }
}
