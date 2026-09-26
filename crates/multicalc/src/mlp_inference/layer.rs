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
    /// Not every activation passes a non-finite value through; see
    /// [Non-finite values](crate::mlp_inference#non-finite-values).
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
/// assert_eq!(hidden.forward(input.view()).into_array(), [0.5, 3.0, 0.0]);
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
    /// The views are taken as they come. Unlike [`try_from_slices`](Self::try_from_slices) this
    /// never reads them, so a non-finite parameter passes through here unremarked.
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
    /// assert_eq!(layer.forward(input.view()), input);
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
}

impl<'data, const OUTPUT: usize, const INPUT: usize, T: Numeric> Layer<'data, OUTPUT, INPUT, T> {
    /// A layer over two runs of a parameter buffer: `weights` read row-major as
    /// `OUTPUT`×`INPUT`, `biases` as `OUTPUT` components. Trailing elements in either slice are
    /// ignored, so one flat export can be split across several layers.
    ///
    /// `OutOfBounds` if either slice is too short, `NonFinite` if a parameter the shape reaches
    /// is infinite or NaN. That check happens here, once, so [`forward`](Self::forward) can read
    /// the same buffer every control cycle without looking at it again.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::mlp_inference::{Activation, Layer};
    /// let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let biases = [0.0, 0.0];
    /// assert!(Layer::<2, 3>::try_from_slices(&weights, &biases, Activation::Relu).is_ok());
    /// assert!(Layer::<2, 4>::try_from_slices(&weights, &biases, Activation::Relu).is_err());
    ///
    /// // A policy that diverge in training is refused at load, not ar the actuator.
    /// let diverged = [1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0];
    /// assert_eq!(
    ///    Layer::<2, 3>::try_from_slices(&diverged, &biases, Activation::Relu).unwrap_err(),
    ///    LinalgError::NonFinite
    /// );
    /// ```
    #[inline]
    pub fn try_from_slices(
        weights: &'data [T],
        biases: &'data [T],
        activation: Activation,
    ) -> Result<Self, LinalgError> {
        let weight_view = MatrixView::try_from_row_major_slice(weights)?;
        let bias_view = VectorView::try_from_slice(biases)?;

        let weight_count = OUTPUT.saturating_mul(INPUT);
        let finite = weights
            .iter()
            .take(weight_count)
            .all(|value| value.is_finite())
            && biases.iter().take(OUTPUT).all(|value| value.is_finite());

        if !finite {
            return Err(LinalgError::NonFinite);
        }
        Ok(Layer::new(weight_view, bias_view, activation))
    }

    /// `activation(weights · input + biases)`, one output component at a time.
    ///
    /// Each output reads the whole input and nothing else, so the components are independent and
    /// the order they are computed in does not matter. The activation is applied to the finished
    /// sum, never to the individual products.
    ///
    /// This cannot fail. Both parameter views are `OUTPUT` tall by their own type parameters, and
    /// a view only exists once its storage has been measured against them.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::mlp_inference::{Activation, Layer};
    /// let weights = [1.0, 1.0, 1.0];
    /// let biases = [0.5];
    /// let output = Layer::<1, 3>::try_from_slices(&weights, &biases, Activation::Identity)?;
    /// let hidden = Vector::new([0.5, 3.0, 0.0]);
    /// assert_eq!(output.forward(hidden.view()).into_array(), [4.0]);
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    pub fn forward(&self, input: VectorView<'_, INPUT, T>) -> Vector<OUTPUT, T> {
        Vector::from_fn(|row_index| {
            let weighted_sum = self
                .weights
                .try_row(row_index)
                .map_or(T::ZERO, |row| row.dot(input));

            let bias = self.biases.try_get(row_index).copied().unwrap_or(T::ZERO);

            self.activation.apply(weighted_sum + bias)
        })
    }

    /// [`forward`](Self::forward), refusing an observation that is not finite.
    ///
    /// The check runs before any arithmetic, so a refused call reads no parameters at all. It
    /// covers the observation only: parameters loaded through
    /// [`try_from_slices`](Self::try_from_slices) were checked when the layer was built.
    ///
    /// Worth a separate call because [`Relu`](Activation::Relu) does not propagate a NaN. Every
    /// comparison against NaN is false, so the clamp takes it and returns an ordinary `0`, and
    /// every layer after that computes normally on a number that means nothing. It does not
    /// promise a finite result: a finite observation over finite parameters can still overflow.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::mlp_inference::{Activation, Layer};
    /// let weights = [1.0, 1.0];
    /// let biases = [0.0];
    /// let layer = Layer::<1, 2>::try_from_slices(&weights, &biases, Activation::Relu)?;
    ///
    /// let spoiled = Vector::new([f64::NAN, 1.0]);
    /// assert_eq!(layer.forward_checked(spoiled.view()), Err(LinalgError::NonFinite));
    /// // Unchecked, the rectifier turns that NaN into an ordinary `0.0`.
    /// assert_eq!(layer.forward(spoiled.view()).into_array(), [0.0]);
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    pub fn forward_checked(
        &self,
        input: VectorView<'_, INPUT, T>,
    ) -> Result<Vector<OUTPUT, T>, LinalgError> {
        for index in 0..INPUT {
            if !input.try_get(index).is_ok_and(|value| value.is_finite()) {
                return Err(LinalgError::NonFinite);
            }
        }
        Ok(self.forward(input))
    }
}
