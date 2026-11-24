//! The generalized-momentum observer for sensorless contact detection.
#![deny(clippy::indexing_slicing)]

use crate::dynamics::ArticulatedBody;
use crate::error::DynamicsError;
use crate::kinematics::KinematicTreeState;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// External joint torque from `q`, `q̇` and the applied torque alone: no force sensor, no `q̈`.
///
/// The residual `r` tracks `τ_ext` through a first-order lag of bandwidth `K` per joint:
///
/// ```text
/// r = K ⊙ (p − p₀ − ∫(τ − β + r) dt)        ṙ = K ⊙ (τ_ext − r)
/// ```
///
/// with `p = H(q)·q̇` and `β` from [`ArticulatedBody::generalized_momentum_and_bias`]. Per tick,
/// `τ` is held over the tick, `β` takes the trapezoid rule, and `r` is implicit, so the loop is
/// stable for every `K·dt > 0`:
///
/// ```text
/// u_k = τ_k − (β_{k−1} + β_k)/2
/// r_k = K/(1 + K·dt) ⊙ (p_k − p₀ − I_{k−1} − u_k·dt)
/// I_k = I_{k−1} + (u_k + r_k)·dt
/// ```
///
/// A step in `τ_ext` shows in `r` as `1 − (1 + K·dt)^−k` after `k` ticks. The first update after
/// [`new`](MomentumObserver::new) or [`reset`](MomentumObserver::reset) latches `p₀` and `β₀` and
/// returns zero. The contact threshold on `r` is the caller's.
///
/// ```
/// use multicalc::dynamics::{ArticulatedBody, MomentumObserver};
/// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::spatial::{SE3, SpatialInertia};
///
/// let hinge = Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::<f64>::identity());
/// let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])
///     .unwrap();
/// let link = SpatialInertia::new(
///     2.0,
///     Vector::new([0.5, 0.0, 0.0]),
///     Matrix::from_diagonal([0.01, 0.01, 0.01]),
/// )
/// .unwrap();
/// let body =
///     ArticulatedBody::new(tree, &[Some(link)], Vector::new([0.0, 0.0, -9.81])).unwrap();
///
/// // K = 50 rad/s at 1 kHz.
/// let mut observer = MomentumObserver::new(Vector::new([50.0]), 0.001).unwrap();
///
/// // Held level against a 0.8 N·m push: τ = G(q) − τ_ext at q̇ = 0.
/// let push = 0.8;
/// let level = Vector::zeros();
/// let still = Vector::zeros();
/// let torque = body.gravity_torque_at(&level).unwrap() - Vector::new([push]);
///
/// let mut residual = observer.update_at(&body, &level, &still, &torque).unwrap();
/// assert_eq!(residual[0], 0.0);
/// for _ in 0..500 {
///     residual = observer.update_at(&body, &level, &still, &torque).unwrap();
/// }
/// // r = τ_ext·(1 − 1.05^−500).
/// assert!((residual[0] - push).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumObserver<const MAX_CONFIG: usize, T: Numeric = f64> {
    /// Per-joint bandwidth `K`.
    gains: Vector<MAX_CONFIG, T>,
    /// Fixed tick length `dt`.
    timestep: T,
    /// `K/(1 + K·dt)` per joint.
    residual_factors: Vector<MAX_CONFIG, T>,
    /// `p₀`, latched on the first update.
    initial_momentum: Vector<MAX_CONFIG, T>,
    /// `I = ∫(τ − β + r) dt` since the latch.
    momentum_integral: Vector<MAX_CONFIG, T>,
    /// `β` at the last update, the left end of the next trapezoid.
    previous_bias: Vector<MAX_CONFIG, T>,
    /// `r` at the last update.
    residual: Vector<MAX_CONFIG, T>,
    /// Whether `p₀` and `β₀` are latched.
    started: bool,
}

impl<const MAX_CONFIG: usize, T: Numeric> MomentumObserver<MAX_CONFIG, T> {
    /// An observer with per-joint bandwidth `gains` at a fixed `timestep`.
    ///
    /// Every one of the `MAX_CONFIG` gains is checked, so fill unused slots with a positive value.
    ///
    /// Errors: [`NonFinite`](DynamicsError::NonFinite) on a non-finite gain or timestep,
    /// [`NonPositiveTimestep`](DynamicsError::NonPositiveTimestep) if `timestep ≤ 0`,
    /// [`NonPositiveGain`](DynamicsError::NonPositiveGain) if a gain is `≤ 0`.
    pub fn new(gains: Vector<MAX_CONFIG, T>, timestep: T) -> Result<Self, DynamicsError> {
        if !gains.is_finite() || !timestep.is_finite() {
            return Err(DynamicsError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(DynamicsError::NonPositiveTimestep);
        }
        for slot in 0..MAX_CONFIG {
            if gains.get(slot).copied().unwrap_or(T::ZERO) <= T::ZERO {
                return Err(DynamicsError::NonPositiveGain);
            }
        }
        let residual_factors = Vector::from_fn(|slot| {
            let gain = gains.get(slot).copied().unwrap_or(T::ZERO);
            gain / (T::ONE + gain * timestep)
        });
        Ok(Self {
            gains,
            timestep,
            residual_factors,
            initial_momentum: Vector::zeros(),
            momentum_integral: Vector::zeros(),
            previous_bias: Vector::zeros(),
            residual: Vector::zeros(),
            started: false,
        })
    }

    /// The per-joint bandwidths `K`.
    #[inline]
    pub fn gains(&self) -> Vector<MAX_CONFIG, T> {
        self.gains
    }

    /// The fixed tick length `dt`.
    #[inline]
    #[must_use]
    pub fn timestep(&self) -> T {
        self.timestep
    }

    /// The residual `r` from the last update: the estimate of `τ_ext`.
    #[inline]
    pub fn residual(&self) -> Vector<MAX_CONFIG, T> {
        self.residual
    }

    /// One tick for already-solved poses; returns `r_k`.
    ///
    /// `torque` is the torque the actuators applied over the tick that ends at
    /// `(state, velocity)`. Slots past the model's velocity count stay zero.
    ///
    /// Errors: [`NonFinite`](DynamicsError::NonFinite) on a non-finite torque, and as
    /// [`generalized_momentum_and_bias`](ArticulatedBody::generalized_momentum_and_bias). The
    /// observer does not change on an error.
    pub fn update<const MAX_JOINTS: usize>(
        &mut self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        torque: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        if !torque.is_finite() {
            return Err(DynamicsError::NonFinite);
        }
        let (momentum, bias) = body.generalized_momentum_and_bias(state, velocity)?;
        if !self.started {
            self.initial_momentum = momentum;
            self.momentum_integral = Vector::zeros();
            self.previous_bias = bias;
            self.residual = Vector::zeros();
            self.started = true;
            return Ok(self.residual);
        }

        let mut residual = Vector::<MAX_CONFIG, T>::zeros();
        let mut momentum_integral = Vector::<MAX_CONFIG, T>::zeros();
        for slot in 0..body.velocity_len().min(MAX_CONFIG) {
            let read =
                |vector: &Vector<MAX_CONFIG, T>| vector.get(slot).copied().unwrap_or(T::ZERO);
            let drive = read(torque) - T::HALF * (read(&self.previous_bias) + read(&bias));
            let integral = read(&self.momentum_integral);
            let entry = read(&self.residual_factors)
                * (read(&momentum)
                    - read(&self.initial_momentum)
                    - integral
                    - drive * self.timestep);
            if let Some(target) = residual.get_mut(slot) {
                *target = entry;
            }
            if let Some(target) = momentum_integral.get_mut(slot) {
                *target = integral + (drive + entry) * self.timestep;
            }
        }
        self.momentum_integral = momentum_integral;
        self.previous_bias = bias;
        self.residual = residual;
        Ok(residual)
    }

    /// One tick for configuration `configuration`.
    ///
    /// Solves the world poses first, then hands them to [`update`](MomentumObserver::update).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`Kinematics`](DynamicsError::Kinematics), and as [`update`](MomentumObserver::update).
    pub fn update_at<const MAX_JOINTS: usize>(
        &mut self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        configuration: &Vector<MAX_CONFIG, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        torque: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        let state = body.tree().forward_kinematics(configuration)?;
        self.update(body, &state, velocity, torque)
    }

    /// Restarts the observer: the next update latches `p₀` and `β₀` and returns zero.
    pub fn reset(&mut self) {
        self.initial_momentum = Vector::zeros();
        self.momentum_integral = Vector::zeros();
        self.previous_bias = Vector::zeros();
        self.residual = Vector::zeros();
        self.started = false;
    }
}
