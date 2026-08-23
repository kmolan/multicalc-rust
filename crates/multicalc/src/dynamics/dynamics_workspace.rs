//! Caller-owned scratch for the articulated-body algorithm.
#![deny(clippy::indexing_slicing)]

use crate::linear_algebra::{Matrix, Matrix6D};
use crate::scalar::Numeric;
use crate::spatial::{Twist, Wrench};

/// Caller-owned scratch for
/// [`ArticulatedBody::forward_dynamics_with_workspace`](crate::dynamics::ArticulatedBody::forward_dynamics_with_workspace).
///
/// Opaque: the fields are the articulated-body algorithm's per-joint registers and carry no meaning
/// between calls. Build one once and reuse it, so a control loop never puts a model-sized dynamics
/// frame on the stack.
#[derive(Debug, Clone, Copy)]
pub struct DynamicsWorkspace<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric = f64> {
    pub(crate) subspaces: [Twist<T>; MAX_JOINTS],
    pub(crate) velocities: [Twist<T>; MAX_JOINTS],
    pub(crate) bias_velocities: [Twist<T>; MAX_JOINTS],
    pub(crate) accelerations: [Twist<T>; MAX_JOINTS],
    pub(crate) articulated_inertias: [Matrix6D<T>; MAX_JOINTS],
    pub(crate) bias_wrenches: [Wrench<T>; MAX_JOINTS],
    pub(crate) axis_wrenches: [Wrench<T>; MAX_JOINTS],
    pub(crate) axis_inertias: [T; MAX_JOINTS],
    pub(crate) axis_residuals: [T; MAX_JOINTS],
}

impl<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric> Default
    for DynamicsWorkspace<MAX_JOINTS, MAX_CONFIG, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric>
    DynamicsWorkspace<MAX_JOINTS, MAX_CONFIG, T>
{
    /// A workspace with every register zeroed, sized for a model of `MAX_JOINTS` slots.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        DynamicsWorkspace {
            subspaces: [Twist::zeros(); MAX_JOINTS],
            velocities: [Twist::zeros(); MAX_JOINTS],
            bias_velocities: [Twist::zeros(); MAX_JOINTS],
            accelerations: [Twist::zeros(); MAX_JOINTS],
            articulated_inertias: [Matrix::zeros(); MAX_JOINTS],
            bias_wrenches: [Wrench::zeros(); MAX_JOINTS],
            axis_wrenches: [Wrench::zeros(); MAX_JOINTS],
            axis_inertias: [T::ZERO; MAX_JOINTS],
            axis_residuals: [T::ZERO; MAX_JOINTS],
        }
    }
}
