//! Optimal linear state feedback for a system running at a fixed timestep.

use crate::error::ControlError;
use crate::linear_algebra::{Matrix, Vector, solve_discrete_lyapunov, solve_discrete_riccati};
use crate::scalar::Numeric;

/// A fixed feedback law that trades state error against input effort as cheaply as possible.
///
/// Give it how the state moves (`state_transition`), how the input pushes it (`input_model`), how
/// much state error costs (`state_cost`), and how much input effort costs (`input_cost`). The
/// expensive part — solving for the best trade-off — happens once when the controller is built. The
/// gain it produces is then a plain matrix, so every call after that is a matrix-vector product:
/// bounded work, no allocation, and cheap enough for a fast loop.
///
/// [`certify_stability`](Self::certify_stability) checks that the loop this closes actually settles.
/// It costs about as much as building the controller, so it belongs at startup or on the bench,
/// never inside the loop.
///
/// Every operation is generic over [`Numeric`](crate::Numeric).
///
/// ```
/// use multicalc::control::Lqr;
/// use multicalc::linear_algebra::{Matrix, Vector};
///
/// // A cart that carries its speed forward and is pushed by the input, at a 0.1 s timestep.
/// let state_transition = Matrix::<2, 2>::new([[1.0, 0.1], [0.0, 1.0]]);
/// let input_model = Matrix::<2, 1>::new([[0.005], [0.1]]);
/// let state_cost = Matrix::<2, 2>::identity();
/// let input_cost = Matrix::<1, 1>::new([[1.0]]);
///
/// let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
/// controller.certify_stability().unwrap();
///
/// // Started away from zero, the loop brings the cart home.
/// let mut state = Vector::new([1.0, 0.0]);
/// for _ in 0..400 {
///     let input = controller.control(state);
///     state = state_transition * state + input_model * input;
/// }
/// assert!(state.norm() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lqr<const N: usize, const M: usize, T: Numeric = f64> {
    gain: Matrix<M, N, T>,
    closed_loop: Matrix<N, N, T>,
    cost_to_go: Matrix<N, N, T>,
    state_cost: Matrix<N, N, T>,
}

impl<const N: usize, const M: usize, T: Numeric> Lqr<N, M, T> {
    /// Builds the controller by solving for the best trade-off once.
    ///
    /// `state_transition` and `input_model` describe one timestep of the system; `state_cost` and
    /// `input_cost` say how much state error and input effort are each worth. The caller has to
    /// supply a system whose runaway directions the input can reach and whose costly directions
    /// `state_cost` can see; without that there is no answer to find.
    ///
    /// This does `O(n³)` work per pass over a budget of passes, so build the controller once and
    /// reuse it rather than rebuilding it in a loop.
    ///
    /// Returns [`ControlError::Linalg`] carrying
    /// [`LinalgError::NonFinite`](crate::error::LinalgError::NonFinite) if any entry is not finite,
    /// [`LinalgError::NotSymmetric`](crate::error::LinalgError::NotSymmetric) if either cost does
    /// not read the same across the diagonal,
    /// [`LinalgError::NotPositiveDefinite`](crate::error::LinalgError::NotPositiveDefinite) if the
    /// input cost has no Cholesky factor, or
    /// [`LinalgError::DidNotConverge`](crate::error::LinalgError::DidNotConverge) if the solve did
    /// not settle.
    pub fn new(
        state_transition: Matrix<N, N, T>,
        input_model: Matrix<N, M, T>,
        state_cost: Matrix<N, N, T>,
        input_cost: Matrix<M, M, T>,
    ) -> Result<Self, ControlError> {
        let cost_to_go =
            solve_discrete_riccati(state_transition, input_model, state_cost, input_cost)?;

        // The gain is (R + BᵀPB)⁻¹·BᵀPA, applied through a factorization rather than by forming
        // the inverse.
        let input_weight = input_cost + input_model.transpose() * cost_to_go * input_model;
        let coupling = input_model.transpose() * cost_to_go * state_transition;
        let gain = input_weight.cholesky()?.solve_matrix::<N>(coupling);

        let closed_loop = state_transition - input_model * gain;
        Ok(Self {
            gain,
            closed_loop,
            cost_to_go,
            state_cost,
        })
    }

    /// Returns the feedback gain, the matrix that turns a state into the input that answers it.
    #[inline]
    pub fn gain(&self) -> Matrix<M, N, T> {
        self.gain
    }

    /// Returns how the state moves once the loop is closed.
    #[inline]
    pub fn closed_loop(&self) -> Matrix<N, N, T> {
        self.closed_loop
    }

    /// Returns the remaining cost of starting from a given state and following this law forever,
    /// as the matrix `P` in `xᵀ·P·x`.
    #[inline]
    pub fn cost_to_go(&self) -> Matrix<N, N, T> {
        self.cost_to_go
    }

    /// Returns the input that drives the state toward zero.
    #[inline]
    pub fn control(&self, state: Vector<N, T>) -> Vector<M, T> {
        -(self.gain * state)
    }

    /// Returns the input that drives the state toward `reference`, on top of the input that holds
    /// it there.
    ///
    /// `feedforward` is whatever input keeps the system sitting at `reference`; pass a zero vector
    /// when the reference is the origin or when nothing is needed to hold it.
    #[inline]
    pub fn control_tracking(
        &self,
        state: Vector<N, T>,
        reference: Vector<N, T>,
        feedforward: Vector<M, T>,
    ) -> Vector<M, T> {
        feedforward - self.gain * (state - reference)
    }

    /// Checks that the loop this controller closes actually settles, and returns the proof.
    ///
    /// It solves `A_clᵀ·P·A_cl − P + Q = 0` for the closed loop and confirms the answer is
    /// positive definite. A solution exists only when repeated application of the closed loop
    /// shrinks every direction, so failing to find one is the verdict that the loop does not
    /// settle. The returned matrix is the energy-like quantity that the loop drives down every
    /// step.
    ///
    /// This costs about as much as building the controller, so run it at startup or on the bench
    /// and never inside a control loop.
    ///
    /// Returns [`ControlError::Linalg`] carrying
    /// [`LinalgError::DidNotConverge`](crate::error::LinalgError::DidNotConverge) if the loop does
    /// not settle, or
    /// [`LinalgError::NotPositiveDefinite`](crate::error::LinalgError::NotPositiveDefinite) if the
    /// answer is not positive definite, which happens when the state cost cannot see every
    /// direction the state can move in.
    pub fn certify_stability(&self) -> Result<Matrix<N, N, T>, ControlError> {
        let certificate = solve_discrete_lyapunov(self.closed_loop, self.state_cost)?;
        // Only whether a factor exists matters here, not the factor itself.
        let _ = certificate.cholesky()?;
        Ok(certificate)
    }
}
