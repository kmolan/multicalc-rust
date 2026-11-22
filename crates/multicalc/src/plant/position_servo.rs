//! Position-controlled hardware: the joint's own servo, advanced exactly.
#![deny(clippy::indexing_slicing)]

use crate::discretization::zoh;
use crate::error::PlantError;
use crate::linear_algebra::{Matrix, Matrix2D, Vector};
use crate::scalar::Numeric;

/// A joint that takes a commanded position rather than a torque, and its internal servo's response.
///
/// Per joint the closed loop is second order, `q̈ = ω²·(q_cmd − q) − 2ζω·q̇`, i.e.
///
/// ```text
/// A = [[0, 1], [−ω², −2ζω]]      B = [[0], [ω²]]      u = q_cmd
/// ```
///
/// The exact zero-order-hold pair `(F, G)` is formed once at construction, so a tick is one 2×2
/// product per joint. Exact discretization makes the sub-step unconditionally stable at any tick
/// length — a claim about this linear sub-step alone, not the articulated body it is split from,
/// which keeps whatever stability its explicit integrator has.
///
/// **Splitting order.** This is the stiff linear half of an operator-split step; the nonlinear body
/// is the other half, advanced by [`ode`](crate::ode)'s RK4/RK45. Lie–Trotter is `stepped` over
/// `Δt` then the body over `Δt`, first order in `Δt`. Strang is `stepped_over(cmd, Δt/2)`, the body
/// over `Δt`, then `stepped_over(cmd, Δt/2)`, second order in `Δt`.
///
/// The torque-side view of the same hardware is
/// [`JointPdController`](crate::control::JointPdController).
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::plant::PositionServo;
/// # fn main() -> Result<(), multicalc::error::PlantError> {
/// // Two joints whose servos run at 50 rad/s, critically damped, driven every millisecond.
/// let mut joints = PositionServo::<2, f64>::uniform(50.0, 1.0, 0.001)?;
/// let commanded = Vector::new([0.4, -0.2]);
///
/// // Critically damped: q(t) = q_cmd·(1 − (1 + ωt)·e^(−ωt)) from rest.
/// let ticks = 20;
/// let _ = (0..ticks).fold(Vector::zeros(), |_, _| joints.stepped(commanded));
/// let elapsed = 0.001 * ticks as f64;
/// let settled = 1.0 - (1.0 + 50.0 * elapsed) * (-50.0 * elapsed).exp();
/// assert!((joints.positions()[0] - 0.4 * settled).abs() < 1e-12);
///
/// // Held there, they arrive exactly.
/// for _ in 0..5000 {
///     let _ = joints.stepped(commanded);
/// }
/// assert!((joints.positions()[1] + 0.2).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionServo<const JOINT_COUNT: usize, T: Numeric = f64> {
    timestep: T,
    natural_frequencies: Vector<JOINT_COUNT, T>,
    damping_ratios: Vector<JOINT_COUNT, T>,
    transitions: [Matrix2D<T>; JOINT_COUNT],
    inputs: [Vector<2, T>; JOINT_COUNT],
    positions: Vector<JOINT_COUNT, T>,
    velocities: Vector<JOINT_COUNT, T>,
}

/// The exact zero-order-hold pair of one joint's servo over `timestep`.
fn discretize<T: Numeric>(
    natural_frequency: T,
    damping_ratio: T,
    timestep: T,
) -> Result<(Matrix2D<T>, Vector<2, T>), PlantError> {
    let stiffness = natural_frequency * natural_frequency;
    let a = Matrix2D::new([
        [T::ZERO, T::ONE],
        [-stiffness, -T::TWO * damping_ratio * natural_frequency],
    ]);
    let b = Matrix::<2, 1, T>::new([[T::ZERO], [stiffness]]);
    let (transition, input) = zoh::<2, 1, 3, T>(a, b, timestep)?;
    let column = Vector::from_fn(|row| input.get(row, 0).copied().unwrap_or(T::ZERO));
    Ok((transition, column))
}

impl<const JOINT_COUNT: usize, T: Numeric> PositionServo<JOINT_COUNT, T> {
    /// Builds the model from a per-joint servo bandwidth and damping ratio, and a fixed tick
    /// length.
    ///
    /// Every joint starts at zero position and zero rate; [`PositionServo::with_state`] starts them
    /// somewhere else.
    ///
    /// Returns [`PlantError::NonFinite`] if any value is not finite,
    /// [`PlantError::NonPositiveNaturalFrequency`] if a bandwidth is zero or negative,
    /// [`PlantError::NegativeDampingRatio`] if a damping ratio is negative,
    /// [`PlantError::NonPositiveTimestep`] if the tick length is zero or negative, or
    /// [`PlantError::Linalg`] if the discretization fails.
    pub fn new(
        natural_frequencies: Vector<JOINT_COUNT, T>,
        damping_ratios: Vector<JOINT_COUNT, T>,
        timestep: T,
    ) -> Result<Self, PlantError> {
        if !natural_frequencies.is_finite() || !damping_ratios.is_finite() || !timestep.is_finite()
        {
            return Err(PlantError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(PlantError::NonPositiveTimestep);
        }

        let mut transitions = [Matrix2D::<T>::zeros(); JOINT_COUNT];
        let mut inputs = [Vector::<2, T>::zeros(); JOINT_COUNT];
        for joint in 0..JOINT_COUNT {
            let natural_frequency = natural_frequencies.get(joint).copied().unwrap_or(T::ZERO);
            let damping_ratio = damping_ratios.get(joint).copied().unwrap_or(T::ZERO);
            if natural_frequency <= T::ZERO {
                return Err(PlantError::NonPositiveNaturalFrequency);
            }
            if damping_ratio < T::ZERO {
                return Err(PlantError::NegativeDampingRatio);
            }
            let (transition, input) = discretize(natural_frequency, damping_ratio, timestep)?;
            if let Some(slot) = transitions.get_mut(joint) {
                *slot = transition;
            }
            if let Some(slot) = inputs.get_mut(joint) {
                *slot = input;
            }
        }

        Ok(Self {
            timestep,
            natural_frequencies,
            damping_ratios,
            transitions,
            inputs,
            positions: Vector::zeros(),
            velocities: Vector::zeros(),
        })
    }

    /// The same servo on every joint.
    ///
    /// Errors: as [`PositionServo::new`].
    pub fn uniform(
        natural_frequency: T,
        damping_ratio: T,
        timestep: T,
    ) -> Result<Self, PlantError> {
        Self::new(
            Vector::from_fn(|_| natural_frequency),
            Vector::from_fn(|_| damping_ratio),
            timestep,
        )
    }

    /// Starts the joints somewhere other than at rest at zero.
    #[inline]
    #[must_use]
    pub fn with_state(
        mut self,
        positions: Vector<JOINT_COUNT, T>,
        velocities: Vector<JOINT_COUNT, T>,
    ) -> Self {
        self.positions = positions;
        self.velocities = velocities;
        self
    }

    /// Advances every joint one tick toward its commanded position, and says where they landed.
    ///
    /// The tick is the one the model was built with; the discretization it needs was worked out
    /// then. A command that is not finite comes back not finite rather than being rejected — this
    /// runs every tick, so checking is the caller's job, once, upstream.
    pub fn stepped(&mut self, commanded: Vector<JOINT_COUNT, T>) -> Vector<JOINT_COUNT, T> {
        for joint in 0..JOINT_COUNT {
            let (Some(transition), Some(input)) =
                (self.transitions.get(joint), self.inputs.get(joint))
            else {
                continue;
            };
            self.advance(joint, *transition, *input, commanded);
        }
        self.positions
    }

    /// The same step, over a tick of some other length.
    ///
    /// Recomputes the discretization, so it costs a matrix exponential per joint; prefer
    /// [`PositionServo::stepped`] on a loop running at a fixed rate. This is the call Strang
    /// splitting's two half-steps need.
    ///
    /// Returns [`PlantError::NonPositiveTimestep`] if `timestep` is zero or negative, or
    /// [`PlantError::Linalg`] if the discretization fails.
    pub fn stepped_over(
        &mut self,
        commanded: Vector<JOINT_COUNT, T>,
        timestep: T,
    ) -> Result<Vector<JOINT_COUNT, T>, PlantError> {
        if !timestep.is_finite() {
            return Err(PlantError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(PlantError::NonPositiveTimestep);
        }
        for joint in 0..JOINT_COUNT {
            let natural_frequency = self
                .natural_frequencies
                .get(joint)
                .copied()
                .unwrap_or(T::ZERO);
            let damping_ratio = self.damping_ratios.get(joint).copied().unwrap_or(T::ZERO);
            let (transition, input) = discretize(natural_frequency, damping_ratio, timestep)?;
            self.advance(joint, transition, input, commanded);
        }
        Ok(self.positions)
    }

    /// `[q; q̇] ← F·[q; q̇] + G·q_cmd` on one joint.
    fn advance(
        &mut self,
        joint: usize,
        transition: Matrix2D<T>,
        input: Vector<2, T>,
        commanded: Vector<JOINT_COUNT, T>,
    ) {
        let position = self.positions.get(joint).copied().unwrap_or(T::ZERO);
        let velocity = self.velocities.get(joint).copied().unwrap_or(T::ZERO);
        let command = commanded.get(joint).copied().unwrap_or(T::ZERO);
        let state = transition * Vector::new([position, velocity]) + input.scale(command);
        if let Some(entry) = self.positions.get_mut(joint) {
            *entry = state.get(0).copied().unwrap_or(T::ZERO);
        }
        if let Some(entry) = self.velocities.get_mut(joint) {
            *entry = state.get(1).copied().unwrap_or(T::ZERO);
        }
    }

    /// Where each joint is now.
    #[inline]
    pub fn positions(&self) -> Vector<JOINT_COUNT, T> {
        self.positions
    }

    /// How fast each joint is moving now.
    #[inline]
    pub fn velocities(&self) -> Vector<JOINT_COUNT, T> {
        self.velocities
    }

    /// How long one tick of the loop lasts.
    #[inline]
    #[must_use]
    pub fn timestep(&self) -> T {
        self.timestep
    }

    /// Puts every joint back at rest at zero.
    #[inline]
    pub fn reset(&mut self) {
        self.positions = Vector::zeros();
        self.velocities = Vector::zeros();
    }
}
