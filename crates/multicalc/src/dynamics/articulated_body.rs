//! Inverse and forward dynamics for a jointed model, and its joint-space inertia matrix.
#![deny(clippy::indexing_slicing)]

use crate::dynamics::DynamicsWorkspace;
use crate::error::DynamicsError;
use crate::kinematics::{Joint, JointKind, JointParent, KinematicTree, KinematicTreeState};
use crate::linear_algebra::{Matrix, Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::{SpatialInertia, Twist, Wrench};

/// The joint's motion subspace in its own body frame, as a `[v; ω]` twist about that frame's
/// origin. `None` for a joint with no single-axis freedom.
///
/// Revolute: a unit rate about the axis `n` through the point `p` moves the body-fixed point at the
/// origin at `p × n`. Prismatic: `[n; 0]`. The stored axis is already unit — the tree normalizes it
/// on `push`.
fn motion_subspace<T: Numeric>(joint: Joint<T>) -> Option<Twist<T>> {
    match joint.kind() {
        JointKind::Revolute | JointKind::Continuous => {
            Some(Twist::new(joint.anchor().cross(joint.axis()), joint.axis()))
        }
        JointKind::Prismatic => Some(Twist::new(joint.axis(), Vector::zeros())),
        JointKind::Fixed | JointKind::Floating => None,
    }
}

/// Whether the joint carries a degree of freedom, and so a torque entry and a row of `H`.
fn is_movable<T: Numeric>(joint: Joint<T>) -> bool {
    motion_subspace(joint).is_some()
}

/// `sign(rate)`, zero at zero on every scalar type.
///
/// Not [`Numeric::signum`]: the `f32`/`f64` impls override it to the primitive, where `0.0.signum()`
/// is `1.0` — which would break a standing joint away under its full friction loss.
fn coulomb_direction<T: Numeric>(rate: T) -> T {
    if rate > T::ZERO {
        T::ONE
    } else if rate < T::ZERO {
        -T::ONE
    } else {
        T::ZERO
    }
}

/// Two composite inertias summed, either side possibly massless.
///
/// `SpatialInertia::new` guarantees a strictly positive mass, so `combined`'s division by the summed
/// mass is always safe.
fn combine<T: Numeric>(
    left: Option<SpatialInertia<T>>,
    right: Option<SpatialInertia<T>>,
) -> Option<SpatialInertia<T>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.combined(right)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

/// A jointed robot with mass: the model forward kinematics walks, one spatial inertia per joint
/// slot, and gravity.
///
/// Slot `index`'s inertia is stated in the frame
/// [`KinematicTreeState::pose`](crate::kinematics::KinematicTreeState::pose) returns for that index
/// — the body frame after the joint has moved, which is the frame MJCF's `<inertial>` is relative
/// to. A slot with no mass carries `None`; a weld still carries its own.
///
/// Fixed-base only: a model whose first joint is floating is rejected at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArticulatedBody<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric = f64> {
    /// The jointed model the recursions walk.
    tree: KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
    /// Mass properties per joint slot, in the slot's own frame; `None` where the slot carries no
    /// mass. Slots past the tree's joint count are `None`.
    inertias: [Option<SpatialInertia<T>>; MAX_JOINTS],
    /// Uniform acceleration due to gravity, in world axes.
    gravity: Vector3D<T>,
}

impl<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric>
    ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>
{
    /// A model from a tree, one inertia per joint slot in tree order, and gravity in world axes.
    ///
    /// Errors: [`FloatingBaseUnsupported`](DynamicsError::FloatingBaseUnsupported) if the tree's
    /// first joint is floating, [`InertiaCountMismatch`](DynamicsError::InertiaCountMismatch) if
    /// `inertias` is not one entry per joint, [`NonFinite`](DynamicsError::NonFinite) on a
    /// non-finite gravity or inertia.
    ///
    /// ```
    /// use multicalc::dynamics::ArticulatedBody;
    /// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SE3, SpatialInertia};
    ///
    /// let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
    /// let hinge = Joint::revolute(hinge_axis, SE3::<f64>::identity());
    /// let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])
    ///     .unwrap();
    ///
    /// let mass = 2.0;
    /// let balance_point = Vector::new([0.5, 0.0, 0.0]);
    /// let resistance_to_spinning = Matrix::from_diagonal([0.01, 0.01, 0.01]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -9.81]);
    ///
    /// let link = SpatialInertia::new(mass, balance_point, resistance_to_spinning).unwrap();
    /// let body = ArticulatedBody::new(tree, &[Some(link)], earth_gravity).unwrap();
    /// assert_eq!(body.len(), 1);
    /// ```
    pub fn new(
        tree: KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        inertias: &[Option<SpatialInertia<T>>],
        gravity: Vector3D<T>,
    ) -> Result<Self, DynamicsError> {
        if tree.has_floating_base() {
            return Err(DynamicsError::FloatingBaseUnsupported);
        }
        if inertias.len() != tree.len() {
            return Err(DynamicsError::InertiaCountMismatch);
        }
        if !gravity.is_finite() {
            return Err(DynamicsError::NonFinite);
        }

        let mut stored = [None; MAX_JOINTS];
        for (index, inertia) in inertias.iter().enumerate() {
            if let Some(inertia) = inertia
                && !inertia.is_finite()
            {
                return Err(DynamicsError::NonFinite);
            }
            *stored
                .get_mut(index)
                .ok_or(DynamicsError::InertiaCountMismatch)? = *inertia;
        }

        Ok(ArticulatedBody {
            tree,
            inertias: stored,
            gravity,
        })
    }

    /// The jointed model the recursions walk.
    #[inline]
    #[must_use]
    pub fn tree(&self) -> &KinematicTree<MAX_JOINTS, MAX_CONFIG, T> {
        &self.tree
    }

    /// Slot `index`'s mass properties, or `None` where the slot carries no mass or does not exist.
    #[inline]
    #[must_use]
    pub fn inertia(&self, index: usize) -> Option<SpatialInertia<T>> {
        if index >= self.tree.len() {
            return None;
        }
        self.inertias.get(index).copied().flatten()
    }

    /// Uniform acceleration due to gravity, in world axes.
    #[inline]
    pub fn gravity(&self) -> Vector3D<T> {
        self.gravity
    }

    /// How many joint slots the model has.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// `true` when the model has no joints.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// How many velocity numbers the model uses.
    #[inline]
    #[must_use]
    pub fn velocity_len(&self) -> usize {
        self.tree.velocity_len()
    }

    /// Slot `index`'s joint, or [`StateShapeMismatch`](DynamicsError::StateShapeMismatch) past the
    /// joint count.
    #[inline]
    fn joint_at(&self, index: usize) -> Result<Joint<T>, DynamicsError> {
        self.tree
            .joint(index)
            .ok_or(DynamicsError::StateShapeMismatch)
    }

    /// Slot `index`'s parent joint index, or `None` where the slot hangs off the world.
    #[inline]
    fn parent_joint(&self, index: usize) -> Option<usize> {
        match self.tree.parent(index) {
            Some(JointParent::Joint(parent_index)) => Some(parent_index),
            _ => None,
        }
    }

    /// Slot `index`'s reading out of a `MAX_CONFIG`-wide vector, zero where the slot cannot move.
    #[inline]
    fn slot_reading(
        &self,
        index: usize,
        readings: &Vector<MAX_CONFIG, T>,
    ) -> Result<T, DynamicsError> {
        if !is_movable(self.joint_at(index)?) {
            return Ok(T::ZERO);
        }
        let slot = self
            .tree
            .velocity_offset(index)
            .ok_or(DynamicsError::StateShapeMismatch)?;
        readings
            .get(slot)
            .copied()
            .ok_or(DynamicsError::StateShapeMismatch)
    }

    /// Rejects poses that were not solved for this model.
    #[inline]
    fn check_state(&self, state: &KinematicTreeState<MAX_JOINTS, T>) -> Result<(), DynamicsError> {
        if state.len() == self.tree.len() {
            Ok(())
        } else {
            Err(DynamicsError::StateShapeMismatch)
        }
    }

    /// Rejects a rate or torque vector carrying a non-finite entry below the model's velocity count.
    fn check_finite(&self, readings: &Vector<MAX_CONFIG, T>) -> Result<(), DynamicsError> {
        for slot in 0..self.velocity_len() {
            let reading = readings
                .get(slot)
                .copied()
                .ok_or(DynamicsError::StateShapeMismatch)?;
            if !reading.is_finite() {
                return Err(DynamicsError::NonFinite);
            }
        }
        Ok(())
    }

    /// World-axes motion subspace and body velocity per slot, from a solved state and a joint-rate
    /// vector. The first half of every forward sweep in this module.
    fn body_velocities(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
    ) -> Result<([Twist<T>; MAX_JOINTS], [Twist<T>; MAX_JOINTS]), DynamicsError> {
        let mut subspaces = [Twist::<T>::zeros(); MAX_JOINTS];
        let mut velocities = [Twist::<T>::zeros(); MAX_JOINTS];

        for index in 0..self.tree.len() {
            let pose = state.pose(index).ok_or(DynamicsError::StateShapeMismatch)?;
            let subspace = motion_subspace(self.joint_at(index)?)
                .map_or(Twist::zeros(), |local| pose.act_twist(local));
            let parent_velocity = match self.parent_joint(index) {
                Some(parent_index) => *velocities
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?,
                None => Twist::zeros(),
            };
            let rate = self.slot_reading(index, velocity)?;

            *subspaces
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = subspace;
            *velocities
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = parent_velocity + subspace.scale(rate);
        }

        Ok((subspaces, velocities))
    }

    /// The rigid-body joint torque for a solved state, with no joint-dynamics terms.
    ///
    /// Two sweeps in world axes about the world origin. Forward, per slot `i` with parent `λ(i)`:
    ///
    /// ```text
    /// S_i = pose_i · S_i^body                      motion subspace in world axes
    /// v_i = v_λ + S_i·q̇_i
    /// a_i = a_λ + S_i·q̈_i + v_i × (S_i·q̇_i)
    /// f_i = I_i·a_i + v_i ×* (I_i·v_i)             I_i in world axes
    /// ```
    ///
    /// with the world seeded at `v = 0`, `a = [−g; 0]`, so each body's wrench carries its own
    /// weight. Backward, `τ_i = S_iᵀ·f_i` and `f_λ += f_i` — a plain add, since both are in world
    /// axes.
    fn rigid_body_torque(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        acceleration: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        let (subspaces, velocities) = self.body_velocities(state, velocity)?;
        let mut accelerations = [Twist::<T>::zeros(); MAX_JOINTS];
        let mut forces = [Wrench::<T>::zeros(); MAX_JOINTS];
        let base_acceleration = Twist::new(-self.gravity, Vector::zeros());

        for index in 0..self.tree.len() {
            let pose = state.pose(index).ok_or(DynamicsError::StateShapeMismatch)?;
            let subspace = *subspaces
                .get(index)
                .ok_or(DynamicsError::StateShapeMismatch)?;
            let body_velocity = *velocities
                .get(index)
                .ok_or(DynamicsError::StateShapeMismatch)?;
            let parent_acceleration = match self.parent_joint(index) {
                Some(parent_index) => *accelerations
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?,
                None => base_acceleration,
            };
            let rate = self.slot_reading(index, velocity)?;
            let change = self.slot_reading(index, acceleration)?;

            let body_acceleration = parent_acceleration
                + subspace.scale(change)
                + body_velocity.cross(subspace.scale(rate));
            let force = match self.inertia(index) {
                Some(inertia) => {
                    let world = pose.act_inertia(inertia);
                    world.momentum(body_acceleration) + world.bias_wrench(body_velocity)
                }
                None => Wrench::zeros(),
            };

            *accelerations
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = body_acceleration;
            *forces
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = force;
        }

        let mut torque = Vector::<MAX_CONFIG, T>::zeros();
        for index in (0..self.tree.len()).rev() {
            let force = *forces.get(index).ok_or(DynamicsError::StateShapeMismatch)?;
            if is_movable(self.joint_at(index)?) {
                let slot = self
                    .tree
                    .velocity_offset(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let subspace = *subspaces
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                *torque
                    .get_mut(slot)
                    .ok_or(DynamicsError::StateShapeMismatch)? = subspace.dot_wrench(force);
            }
            if let Some(parent_index) = self.parent_joint(index) {
                let carried = *forces
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                *forces
                    .get_mut(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = carried + force;
            }
        }

        Ok(torque)
    }

    /// Joint torque producing `acceleration` at `velocity`, for already-solved poses.
    ///
    /// `τ = H(q)·q̈ + C(q,q̇)·q̇ + G(q) + armature⊙q̈ + damping⊙q̇ + friction_loss⊙sign(q̇)`, by the
    /// recursive Newton-Euler algorithm. Entries for welds and slots past the model's velocity count
    /// stay zero.
    ///
    /// `sign` is zero at zero rate, so a standing joint carries no friction loss. It is not
    /// differentiable there: under [`Dual`](crate::Dual) the Coulomb term contributes nothing to
    /// `∂τ/∂q̇`.
    ///
    /// Errors: [`StateShapeMismatch`](DynamicsError::StateShapeMismatch) if the poses were solved
    /// for a different joint count, [`NonFinite`](DynamicsError::NonFinite) on a non-finite rate.
    pub fn inverse_dynamics(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        acceleration: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        self.check_state(state)?;
        self.check_finite(velocity)?;
        self.check_finite(acceleration)?;

        let mut torque = self.rigid_body_torque(state, velocity, acceleration)?;
        for index in 0..self.tree.len() {
            let joint = self.joint_at(index)?;
            if !is_movable(joint) {
                continue;
            }
            let slot = self
                .tree
                .velocity_offset(index)
                .ok_or(DynamicsError::StateShapeMismatch)?;
            let rate = self.slot_reading(index, velocity)?;
            let change = self.slot_reading(index, acceleration)?;
            let entry = torque
                .get_mut(slot)
                .ok_or(DynamicsError::StateShapeMismatch)?;
            *entry = *entry
                + joint.armature() * change
                + joint.damping() * rate
                + joint.friction_loss() * coulomb_direction(rate);
        }

        Ok(torque)
    }

    /// Joint torque producing `acceleration` at `velocity`, for configuration `configuration`.
    ///
    /// Solves the world poses first, then hands them to
    /// [`inverse_dynamics`](ArticulatedBody::inverse_dynamics).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`Kinematics`](DynamicsError::Kinematics), and as
    /// [`inverse_dynamics`](ArticulatedBody::inverse_dynamics).
    ///
    /// ```
    /// use multicalc::dynamics::ArticulatedBody;
    /// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SE3, SpatialInertia};
    ///
    /// // One hinge about y, the link balancing `moment_arm` out along x.
    /// let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
    /// let hinge = Joint::revolute(hinge_axis, SE3::<f64>::identity());
    /// let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])
    ///     .unwrap();
    ///
    /// let mass = 2.0;
    /// let moment_arm = 0.5;
    /// let gravity_strength = 9.81;
    /// let balance_point = Vector::new([moment_arm, 0.0, 0.0]);
    /// let resistance_to_spinning = Matrix::from_diagonal([0.01, 0.01, 0.01]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -gravity_strength]);
    ///
    /// let link = SpatialInertia::new(mass, balance_point, resistance_to_spinning).unwrap();
    /// let body = ArticulatedBody::new(tree, &[Some(link)], earth_gravity).unwrap();
    ///
    /// // Held level and still, the applied torque is `G(q) = −m·g·L·cos(q)` at `q = 0`.
    /// let torque = body
    ///     .inverse_dynamics_at(&Vector::zeros(), &Vector::zeros(), &Vector::zeros())
    ///     .unwrap();
    /// let holding = -mass * gravity_strength * moment_arm;
    /// assert!((torque[0] - holding).abs() < 1e-12);
    /// ```
    pub fn inverse_dynamics_at(
        &self,
        configuration: &Vector<MAX_CONFIG, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        acceleration: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        let state = self.tree.forward_kinematics(configuration)?;
        self.inverse_dynamics(&state, velocity, acceleration)
    }

    /// The gravity torque `G(q)` for already-solved poses:
    /// [`inverse_dynamics`](ArticulatedBody::inverse_dynamics) at rest.
    ///
    /// No armature (`q̈ = 0`) and no friction (`q̇ = 0`), so this is the gravity vector alone.
    ///
    /// Errors: as [`inverse_dynamics`](ArticulatedBody::inverse_dynamics).
    pub fn gravity_torque(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        self.inverse_dynamics(state, &Vector::zeros(), &Vector::zeros())
    }

    /// The gravity torque `G(q)` for configuration `configuration`.
    ///
    /// Errors: as [`inverse_dynamics_at`](ArticulatedBody::inverse_dynamics_at).
    pub fn gravity_torque_at(
        &self,
        configuration: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        self.inverse_dynamics_at(configuration, &Vector::zeros(), &Vector::zeros())
    }

    /// Everything but the inertial term: `C(q,q̇)·q̇ + G(q) + damping⊙q̇ + friction_loss⊙sign(q̇)`,
    /// for already-solved poses.
    ///
    /// [`inverse_dynamics`](ArticulatedBody::inverse_dynamics) at zero acceleration, so armature
    /// drops out and friction does not.
    ///
    /// Errors: as [`inverse_dynamics`](ArticulatedBody::inverse_dynamics).
    pub fn bias_torque(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        self.inverse_dynamics(state, velocity, &Vector::zeros())
    }

    /// Everything but the inertial term, for configuration `configuration`.
    ///
    /// Errors: as [`inverse_dynamics_at`](ArticulatedBody::inverse_dynamics_at).
    pub fn bias_torque_at(
        &self,
        configuration: &Vector<MAX_CONFIG, T>,
        velocity: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        self.inverse_dynamics_at(configuration, velocity, &Vector::zeros())
    }

    /// The joint-space inertia matrix `H(q)` for already-solved poses, by the composite-rigid-body
    /// algorithm.
    ///
    /// Symmetric positive definite over the model's leading `velocity_len()` block; zero outside it,
    /// and zero in a weld's row and column. Each joint's armature sits on its own diagonal entry and
    /// nowhere else.
    ///
    /// One backward pass. With `Iᶜ_i` the composite inertia of the subtree at `i`, all in world
    /// axes:
    ///
    /// ```text
    /// F            = Iᶜ_i · S_i
    /// H(i, i)      = S_iᵀ·F + armature_i
    /// H(i, j)      = H(j, i) = S_jᵀ·F      for every ancestor j of i
    /// Iᶜ_λ(i)     += Iᶜ_i
    /// ```
    ///
    /// `F` needs no transform as the walk climbs, since every subspace is already in world axes.
    ///
    /// Errors: [`StateShapeMismatch`](DynamicsError::StateShapeMismatch) if the poses were solved
    /// for a different joint count.
    pub fn joint_space_inertia(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
    ) -> Result<Matrix<MAX_CONFIG, MAX_CONFIG, T>, DynamicsError> {
        self.check_state(state)?;

        let mut subspaces = [Twist::<T>::zeros(); MAX_JOINTS];
        let mut composites = [None; MAX_JOINTS];
        for index in 0..self.tree.len() {
            let pose = state.pose(index).ok_or(DynamicsError::StateShapeMismatch)?;
            *subspaces
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = motion_subspace(self.joint_at(index)?)
                .map_or(Twist::zeros(), |local| pose.act_twist(local));
            *composites
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? =
                self.inertia(index).map(|inertia| pose.act_inertia(inertia));
        }

        let mut entries = Matrix::<MAX_CONFIG, MAX_CONFIG, T>::zeros();
        for index in (0..self.tree.len()).rev() {
            let joint = self.joint_at(index)?;
            if is_movable(joint) {
                let own = self
                    .tree
                    .velocity_offset(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let subspace = *subspaces
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let force = composites
                    .get(index)
                    .copied()
                    .flatten()
                    .map_or(Wrench::zeros(), |composite| composite.momentum(subspace));

                *entries
                    .get_mut(own, own)
                    .ok_or(DynamicsError::StateShapeMismatch)? =
                    subspace.dot_wrench(force) + joint.armature();

                // Bounded by the joint count, as the tree's other ancestor walks are.
                let mut ancestor = index;
                for _ in 0..MAX_JOINTS {
                    let Some(next) = self.parent_joint(ancestor) else {
                        break;
                    };
                    ancestor = next;
                    if !is_movable(self.joint_at(ancestor)?) {
                        continue;
                    }
                    let other = self
                        .tree
                        .velocity_offset(ancestor)
                        .ok_or(DynamicsError::StateShapeMismatch)?;
                    let coupling = subspaces
                        .get(ancestor)
                        .ok_or(DynamicsError::StateShapeMismatch)?
                        .dot_wrench(force);
                    *entries
                        .get_mut(own, other)
                        .ok_or(DynamicsError::StateShapeMismatch)? = coupling;
                    *entries
                        .get_mut(other, own)
                        .ok_or(DynamicsError::StateShapeMismatch)? = coupling;
                }
            }

            if let Some(parent_index) = self.parent_joint(index) {
                let carried = composites
                    .get(parent_index)
                    .copied()
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let own = composites
                    .get(index)
                    .copied()
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                *composites
                    .get_mut(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = combine(carried, own);
            }
        }

        Ok(entries)
    }

    /// The joint-space inertia matrix `H(q)` for configuration `configuration`.
    ///
    /// Solves the world poses first, then hands them to
    /// [`joint_space_inertia`](ArticulatedBody::joint_space_inertia).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`Kinematics`](DynamicsError::Kinematics), and as
    /// [`joint_space_inertia`](ArticulatedBody::joint_space_inertia).
    ///
    /// ```
    /// use multicalc::dynamics::ArticulatedBody;
    /// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SE3, SpatialInertia};
    ///
    /// // One hinge about y, with reflected rotor inertia on top of the link it carries.
    /// let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
    /// let armature = 0.03;
    /// let hinge = Joint::revolute(hinge_axis, SE3::<f64>::identity()).with_armature(armature);
    /// let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])
    ///     .unwrap();
    ///
    /// let mass = 2.0;
    /// let moment_arm = 0.5;
    /// let own_axis_inertia = 0.01;
    /// let balance_point = Vector::new([moment_arm, 0.0, 0.0]);
    /// let resistance_to_spinning =
    ///     Matrix::from_diagonal([own_axis_inertia, own_axis_inertia, own_axis_inertia]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -9.81]);
    ///
    /// let link = SpatialInertia::new(mass, balance_point, resistance_to_spinning).unwrap();
    /// let body = ArticulatedBody::new(tree, &[Some(link)], earth_gravity).unwrap();
    ///
    /// // Parallel axis, then the rotor.
    /// let inertia = body.joint_space_inertia_at(&Vector::zeros()).unwrap();
    /// let want = own_axis_inertia + mass * moment_arm * moment_arm + armature;
    /// assert!((inertia[(0, 0)] - want).abs() < 1e-12);
    /// ```
    pub fn joint_space_inertia_at(
        &self,
        configuration: &Vector<MAX_CONFIG, T>,
    ) -> Result<Matrix<MAX_CONFIG, MAX_CONFIG, T>, DynamicsError> {
        let state = self.tree.forward_kinematics(configuration)?;
        self.joint_space_inertia(&state)
    }

    /// Joint acceleration produced by `torque` at `velocity`, for already-solved poses, into a
    /// caller-owned workspace.
    ///
    /// The articulated-body algorithm, three sweeps in world axes about the world origin. Forward:
    ///
    /// ```text
    /// S_i   = pose_i · S_i^body
    /// v_i   = v_λ + S_i·q̇_i
    /// c_i   = v_i × (S_i·q̇_i)
    /// I^A_i = I_i^world ;  p^A_i = v_i ×* (I_i·v_i)
    /// ```
    ///
    /// Backward, per movable joint, then `I^A_λ += I^a` and `p^A_λ += p^a` as plain adds:
    ///
    /// ```text
    /// U_i = I^A_i·S_i
    /// D_i = S_iᵀ·U_i + armature_i
    /// u_i = τ_i − damping_i·q̇_i − friction_loss_i·sign(q̇_i) − S_iᵀ·p^A_i
    /// I^a = I^A_i − U_i·U_iᵀ/D_i
    /// p^a = p^A_i + I^a·c_i + U_i·u_i/D_i
    /// ```
    ///
    /// Forward again, with the world seeded at `a = [−g; 0]`:
    ///
    /// ```text
    /// a'_i = a_λ + c_i
    /// q̈_i  = (u_i − U_iᵀ·a'_i)/D_i
    /// a_i  = a'_i + S_i·q̈_i
    /// ```
    ///
    /// Errors: [`StateShapeMismatch`](DynamicsError::StateShapeMismatch) if the poses were solved
    /// for a different joint count, [`NonFinite`](DynamicsError::NonFinite) on a non-finite rate or
    /// torque, [`NonPositiveArticulatedInertia`](DynamicsError::NonPositiveArticulatedInertia)
    /// where a movable joint has no inertia behind it — a massless body on a joint with no
    /// armature.
    pub fn forward_dynamics_with_workspace(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        torque: &Vector<MAX_CONFIG, T>,
        workspace: &mut DynamicsWorkspace<MAX_JOINTS, MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        self.check_state(state)?;
        self.check_finite(velocity)?;
        self.check_finite(torque)?;

        for index in 0..self.tree.len() {
            let pose = state.pose(index).ok_or(DynamicsError::StateShapeMismatch)?;
            let subspace = motion_subspace(self.joint_at(index)?)
                .map_or(Twist::zeros(), |local| pose.act_twist(local));
            let parent_velocity = match self.parent_joint(index) {
                Some(parent_index) => *workspace
                    .velocities
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?,
                None => Twist::zeros(),
            };
            let rate = self.slot_reading(index, velocity)?;
            let body_velocity = parent_velocity + subspace.scale(rate);
            let inertia = self.inertia(index).map(|inertia| pose.act_inertia(inertia));

            *workspace
                .subspaces
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = subspace;
            *workspace
                .velocities
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = body_velocity;
            *workspace
                .bias_velocities
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? =
                body_velocity.cross(subspace.scale(rate));
            *workspace
                .articulated_inertias
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? =
                inertia.map_or(Matrix::zeros(), SpatialInertia::to_matrix);
            *workspace
                .bias_wrenches
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = inertia
                .map_or(Wrench::zeros(), |inertia| {
                    inertia.bias_wrench(body_velocity)
                });
        }

        for index in (0..self.tree.len()).rev() {
            let joint = self.joint_at(index)?;
            let articulated_inertia = *workspace
                .articulated_inertias
                .get(index)
                .ok_or(DynamicsError::StateShapeMismatch)?;
            let bias_wrench = *workspace
                .bias_wrenches
                .get(index)
                .ok_or(DynamicsError::StateShapeMismatch)?;

            let (carried_inertia, carried_wrench) = if is_movable(joint) {
                let subspace = *workspace
                    .subspaces
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let bias_velocity = *workspace
                    .bias_velocities
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let rate = self.slot_reading(index, velocity)?;
                let applied = self.slot_reading(index, torque)?;

                let axis_wrench = Wrench::from_vector(articulated_inertia * subspace.to_vector());
                let axis_inertia = subspace.dot_wrench(axis_wrench) + joint.armature();
                if axis_inertia <= T::ZERO {
                    return Err(DynamicsError::NonPositiveArticulatedInertia);
                }
                let axis_residual = applied
                    - joint.damping() * rate
                    - joint.friction_loss() * coulomb_direction(rate)
                    - subspace.dot_wrench(bias_wrench);

                // `U·Uᵀ/D`, the rank-one term the joint's own freedom takes out of `I^A`.
                let axis_column = axis_wrench.to_vector();
                let projected_inertia = articulated_inertia
                    - Matrix::from_fn(|row, column| {
                        *axis_column.get(row).unwrap_or(&T::ZERO)
                            * *axis_column.get(column).unwrap_or(&T::ZERO)
                            / axis_inertia
                    });
                let projected_wrench = bias_wrench
                    + Wrench::from_vector(projected_inertia * bias_velocity.to_vector())
                    + axis_wrench.scale(axis_residual / axis_inertia);

                *workspace
                    .axis_wrenches
                    .get_mut(index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = axis_wrench;
                *workspace
                    .axis_inertias
                    .get_mut(index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = axis_inertia;
                *workspace
                    .axis_residuals
                    .get_mut(index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = axis_residual;

                (projected_inertia, projected_wrench)
            } else {
                (articulated_inertia, bias_wrench)
            };

            if let Some(parent_index) = self.parent_joint(index) {
                let parent_inertia = *workspace
                    .articulated_inertias
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let parent_wrench = *workspace
                    .bias_wrenches
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                *workspace
                    .articulated_inertias
                    .get_mut(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = parent_inertia + carried_inertia;
                *workspace
                    .bias_wrenches
                    .get_mut(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)? = parent_wrench + carried_wrench;
            }
        }

        let base_acceleration = Twist::new(-self.gravity, Vector::zeros());
        let mut acceleration = Vector::<MAX_CONFIG, T>::zeros();
        for index in 0..self.tree.len() {
            let parent_acceleration = match self.parent_joint(index) {
                Some(parent_index) => *workspace
                    .accelerations
                    .get(parent_index)
                    .ok_or(DynamicsError::StateShapeMismatch)?,
                None => base_acceleration,
            };
            let carried = parent_acceleration
                + *workspace
                    .bias_velocities
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;

            let body_acceleration = if is_movable(self.joint_at(index)?) {
                let axis_wrench = *workspace
                    .axis_wrenches
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let axis_inertia = *workspace
                    .axis_inertias
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let axis_residual = *workspace
                    .axis_residuals
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                let subspace = *workspace
                    .subspaces
                    .get(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;

                let change = (axis_residual - carried.dot_wrench(axis_wrench)) / axis_inertia;
                let slot = self
                    .tree
                    .velocity_offset(index)
                    .ok_or(DynamicsError::StateShapeMismatch)?;
                *acceleration
                    .get_mut(slot)
                    .ok_or(DynamicsError::StateShapeMismatch)? = change;
                carried + subspace.scale(change)
            } else {
                carried
            };

            *workspace
                .accelerations
                .get_mut(index)
                .ok_or(DynamicsError::StateShapeMismatch)? = body_acceleration;
        }

        Ok(acceleration)
    }

    /// Joint acceleration produced by `torque` at `velocity`, for already-solved poses.
    ///
    /// Builds a workspace on the stack; use
    /// [`forward_dynamics_with_workspace`](ArticulatedBody::forward_dynamics_with_workspace) where
    /// a model-sized frame would not fit.
    ///
    /// Errors: as
    /// [`forward_dynamics_with_workspace`](ArticulatedBody::forward_dynamics_with_workspace).
    pub fn forward_dynamics(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        torque: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        let mut workspace = DynamicsWorkspace::<MAX_JOINTS, MAX_CONFIG, T>::new();
        self.forward_dynamics_with_workspace(state, velocity, torque, &mut workspace)
    }

    /// Joint acceleration produced by `torque` at `velocity`, for configuration `configuration`.
    ///
    /// Solves the world poses first, then hands them to
    /// [`forward_dynamics`](ArticulatedBody::forward_dynamics).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`Kinematics`](DynamicsError::Kinematics), and as
    /// [`forward_dynamics`](ArticulatedBody::forward_dynamics).
    ///
    /// ```
    /// use multicalc::dynamics::ArticulatedBody;
    /// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SE3, SpatialInertia};
    ///
    /// let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
    /// let hinge = Joint::revolute(hinge_axis, SE3::<f64>::identity());
    /// let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])
    ///     .unwrap();
    ///
    /// let mass = 2.0;
    /// let balance_point = Vector::new([0.5, 0.0, 0.0]);
    /// let resistance_to_spinning = Matrix::from_diagonal([0.01, 0.01, 0.01]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -9.81]);
    ///
    /// let link = SpatialInertia::new(mass, balance_point, resistance_to_spinning).unwrap();
    /// let body = ArticulatedBody::new(tree, &[Some(link)], earth_gravity).unwrap();
    ///
    /// // The torque that holds it still leaves it still.
    /// let holding = body.gravity_torque_at(&Vector::zeros()).unwrap();
    /// let acceleration = body
    ///     .forward_dynamics_at(&Vector::zeros(), &Vector::zeros(), &holding)
    ///     .unwrap();
    /// assert!(acceleration[0].abs() < 1e-12);
    /// ```
    pub fn forward_dynamics_at(
        &self,
        configuration: &Vector<MAX_CONFIG, T>,
        velocity: &Vector<MAX_CONFIG, T>,
        torque: &Vector<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, DynamicsError> {
        let state = self.tree.forward_kinematics(configuration)?;
        self.forward_dynamics(&state, velocity, torque)
    }

    /// Kinetic energy `Σᵢ ½·vᵢᵀ·Iᵢ·vᵢ`, summed over every body in world axes.
    ///
    /// Errors: [`StateShapeMismatch`](DynamicsError::StateShapeMismatch) if the poses were solved
    /// for a different joint count, [`NonFinite`](DynamicsError::NonFinite) on a non-finite rate.
    pub fn kinetic_energy(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        velocity: &Vector<MAX_CONFIG, T>,
    ) -> Result<T, DynamicsError> {
        self.check_state(state)?;
        self.check_finite(velocity)?;

        let (_, velocities) = self.body_velocities(state, velocity)?;
        let mut energy = T::ZERO;
        for index in 0..self.tree.len() {
            let Some(inertia) = self.inertia(index) else {
                continue;
            };
            let pose = state.pose(index).ok_or(DynamicsError::StateShapeMismatch)?;
            let body_velocity = *velocities
                .get(index)
                .ok_or(DynamicsError::StateShapeMismatch)?;
            energy += pose.act_inertia(inertia).kinetic_energy(body_velocity);
        }

        Ok(energy)
    }

    /// Potential energy `−Σᵢ mᵢ·g·cᵢ`, each body's centre of mass taken in world axes and
    /// referenced to the world origin.
    ///
    /// Errors: [`StateShapeMismatch`](DynamicsError::StateShapeMismatch) if the poses were solved
    /// for a different joint count.
    pub fn potential_energy(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
    ) -> Result<T, DynamicsError> {
        self.check_state(state)?;

        let mut energy = T::ZERO;
        for index in 0..self.tree.len() {
            let Some(inertia) = self.inertia(index) else {
                continue;
            };
            let pose = state.pose(index).ok_or(DynamicsError::StateShapeMismatch)?;
            let center = pose.act(inertia.center_of_mass());
            energy -= inertia.mass() * self.gravity.dot(center);
        }

        Ok(energy)
    }
}
