//! Multi-start inverse kinematics: `InverseKinematics::solve` run from several seeds, keeping the
//! distinct converged branches. Seeds come from the caller or from jittering a base configuration
//! with any `RandomSource`.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::kinematics::inverse_kinematics::{
    InverseKinematics, InverseKinematicsReport, InverseKinematicsTermination,
};
use crate::kinematics::joint::JointKind;
use crate::kinematics::kinematic_tree::KinematicTree;
use crate::linear_algebra::Vector;
use crate::random::{RandomScalar, RandomSource};
use crate::scalar::Numeric;
use crate::spatial::SE3;

/// Multi-start wrapper around [`InverseKinematics`], configured by a builder.
///
/// A damped least-squares solve converges to one branch, chosen by its seed. This runs up to
/// `MAX_STARTS` solves and collects the distinct converged configurations, which is how a
/// numerical solver enumerates branches without a closed form to enumerate from.
///
/// Holds gains only, so construction is free. The generator is passed per call rather than held,
/// keeping the wrapper `Copy` and its two entry points at different scalar bounds.
///
/// ```
/// use multicalc::kinematics::MultiStartInverseKinematics;
///
/// let _solver = MultiStartInverseKinematics::<4, 3, f64>::new().with_distinct_threshold(1e-2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MultiStartInverseKinematics<
    const MAX_STARTS: usize,
    const MAX_CONFIG: usize,
    T: Numeric = f64,
> {
    solver: InverseKinematics<MAX_CONFIG, T>,
    distinct_threshold: T,
    jitter_span: T,
}

impl<const MAX_STARTS: usize, const MAX_CONFIG: usize, T: Numeric> Default
    for MultiStartInverseKinematics<MAX_STARTS, MAX_CONFIG, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_STARTS: usize, const MAX_CONFIG: usize, T: Numeric>
    MultiStartInverseKinematics<MAX_STARTS, MAX_CONFIG, T>
{
    /// Defaults: a default-tuned [`InverseKinematics`], solutions within 1e-3 of
    /// [`KinematicTree::configuration_distance`] treated as the same branch, jittered draws at an
    /// unbounded joint spanning `base ± π`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            solver: InverseKinematics::new(),
            distinct_threshold: T::from_f64(1e-3),
            jitter_span: T::PI,
        }
    }

    /// Sets the inner solver every start runs.
    #[must_use]
    pub fn with_solver(mut self, solver: InverseKinematics<MAX_CONFIG, T>) -> Self {
        self.solver = solver;
        self
    }

    /// Sets the configuration distance below which two converged solutions count as one branch.
    #[must_use]
    pub fn with_distinct_threshold(mut self, distinct_threshold: T) -> Self {
        self.distinct_threshold = distinct_threshold;
        self
    }

    /// Sets the half-width a jittered draw at an unbounded joint spans about its seed reading.
    #[must_use]
    pub fn with_jitter_span(mut self, jitter_span: T) -> Self {
        self.jitter_span = jitter_span;
        self
    }

    /// Solves once per seed, keeping every `Converged` result at least `distinct_threshold` from
    /// each solution already kept — first found wins a near-duplicate. Every start counts toward
    /// [`MultiStartReport::attempts`], converged or not.
    ///
    /// Errors: [`StartCapacityExceeded`](KinematicsError::StartCapacityExceeded) if `seeds.len()`
    /// exceeds `MAX_STARTS`, otherwise as [`InverseKinematics::solve`].
    ///
    /// ```
    /// use multicalc::kinematics::{
    ///     Joint, JointParent, KinematicTree, MultiStartInverseKinematics,
    /// };
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    /// let tree = KinematicTree::<3, 3, f64>::try_from_joints(
    ///     &[
    ///         Joint::revolute(z, SE3::identity()),
    ///         Joint::revolute(z, link),
    ///         Joint::fixed(link),
    ///     ],
    ///     &[
    ///         JointParent::World,
    ///         JointParent::Joint(0),
    ///         JointParent::Joint(1),
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// // 2 DOF against a full SE(3) target: position and orientation together pin the
    /// // configuration, so both seeds converge to the one solution and it is kept once.
    /// let target = SE3::from_parts(SO3::identity(), Vector::new([1.0, 1.0, 0.0]));
    /// let seeds = [Vector::new([0.3, 0.3, 0.0]), Vector::new([1.2, -0.9, 0.0])];
    ///
    /// let solver = MultiStartInverseKinematics::<2, 3, f64>::new();
    /// let report = solver.solve_from_seeds(&tree, 2, target, &seeds).unwrap();
    ///
    /// assert_eq!(report.attempts(), 2);
    /// assert_eq!(report.len(), 1);
    /// ```
    pub fn solve_from_seeds<const MAX_JOINTS: usize>(
        &self,
        tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        tool_index: usize,
        target: SE3<T>,
        seeds: &[Vector<MAX_CONFIG, T>],
    ) -> Result<MultiStartReport<MAX_STARTS, MAX_CONFIG, T>, KinematicsError> {
        if seeds.len() > MAX_STARTS {
            return Err(KinematicsError::StartCapacityExceeded);
        }
        let mut report = MultiStartReport::empty();
        for seed in seeds {
            let attempt = self.solver.solve(tree, tool_index, target, seed)?;
            report.attempts += 1;
            if attempt.termination == InverseKinematicsTermination::Converged {
                report.insert(tree, attempt, self.distinct_threshold);
            }
        }
        Ok(report)
    }

    /// Solves once from `base_seed` unperturbed, then `count - 1` times from jittered draws: a
    /// joint with limits is redrawn uniformly across its range, an unbounded one uniformly across
    /// `base_reading ± jitter_span`. A floating joint's seven slots are copied through unperturbed
    /// — jitter targets branch ambiguity in the actuated chain, not base placement. Deduplicated
    /// as [`solve_from_seeds`](Self::solve_from_seeds).
    ///
    /// Errors: [`StartCapacityExceeded`](KinematicsError::StartCapacityExceeded) if `count`
    /// exceeds `MAX_STARTS`, otherwise as [`InverseKinematics::solve`].
    pub fn solve_seeded<const MAX_JOINTS: usize, R: RandomSource<T>>(
        &self,
        tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        tool_index: usize,
        target: SE3<T>,
        base_seed: &Vector<MAX_CONFIG, T>,
        source: &mut R,
        count: usize,
    ) -> Result<MultiStartReport<MAX_STARTS, MAX_CONFIG, T>, KinematicsError>
    where
        T: RandomScalar,
    {
        if count > MAX_STARTS {
            return Err(KinematicsError::StartCapacityExceeded);
        }
        let mut report = MultiStartReport::empty();
        for draw in 0..count {
            let seed = if draw == 0 {
                *base_seed
            } else {
                jittered_seed(tree, base_seed, source, self.jitter_span)
            };
            let attempt = self.solver.solve(tree, tool_index, target, &seed)?;
            report.attempts += 1;
            if attempt.termination == InverseKinematicsTermination::Converged {
                report.insert(tree, attempt, self.distinct_threshold);
            }
        }
        Ok(report)
    }
}

/// `base_seed` with every actuated joint redrawn: uniform across its limits where it has them,
/// else uniform across `base_reading ± jitter_span`. Fixed and floating joints pass through.
fn jittered_seed<
    const MAX_JOINTS: usize,
    const MAX_CONFIG: usize,
    T: RandomScalar,
    R: RandomSource<T>,
>(
    tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
    base_seed: &Vector<MAX_CONFIG, T>,
    source: &mut R,
    jitter_span: T,
) -> Vector<MAX_CONFIG, T> {
    let mut seed = *base_seed;
    for index in 0..tree.len() {
        let Some(joint) = tree.joint(index) else {
            continue;
        };
        if joint.kind() == JointKind::Fixed || joint.kind() == JointKind::Floating {
            continue;
        }
        let Some(offset) = tree.config_offset(index) else {
            continue;
        };
        let Some(base_reading) = base_seed.get(offset).copied() else {
            continue;
        };
        let draw = source.next_unit();
        let reading = match joint.limits() {
            Some((lower, upper)) => lower + draw * (upper - lower),
            None => base_reading + (draw * T::TWO - T::ONE) * jitter_span,
        };
        if let Some(slot) = seed.get_mut(offset) {
            *slot = reading;
        }
    }
    seed
}

/// The distinct converged branches a [`MultiStartInverseKinematics`] run found, deduplicated by
/// [`KinematicTree::configuration_distance`].
///
/// Not exhaustive: without a closed-form solver, multi-start finds some of the valid branches —
/// probabilistically from jittered seeds, deterministically from caller-chosen ones — never
/// provably all of them.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct MultiStartReport<const MAX_STARTS: usize, const MAX_CONFIG: usize, T: Numeric = f64> {
    solutions: [InverseKinematicsReport<MAX_CONFIG, T>; MAX_STARTS],
    solution_count: usize,
    attempts: usize,
}

impl<const MAX_STARTS: usize, const MAX_CONFIG: usize, T: Numeric>
    MultiStartReport<MAX_STARTS, MAX_CONFIG, T>
{
    fn empty() -> Self {
        let filler = InverseKinematicsReport {
            joint_positions: Vector::zeros(),
            position_error: T::ZERO,
            orientation_error: T::ZERO,
            iterations: 0,
            termination: InverseKinematicsTermination::IterationBudget,
        };
        Self {
            solutions: [filler; MAX_STARTS],
            solution_count: 0,
            attempts: 0,
        }
    }

    /// Keeps `candidate` unless it is within `distinct_threshold` of a solution already held.
    ///
    /// `solution_count` cannot reach `MAX_STARTS` unnoticed: both entry points bound their start
    /// count by `MAX_STARTS` first, and one start yields at most one solution.
    fn insert<const MAX_JOINTS: usize>(
        &mut self,
        tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        candidate: InverseKinematicsReport<MAX_CONFIG, T>,
        distinct_threshold: T,
    ) {
        for existing in self.solutions() {
            if tree.configuration_distance(&existing.joint_positions, &candidate.joint_positions)
                < distinct_threshold
            {
                return;
            }
        }
        if let Some(slot) = self.solutions.get_mut(self.solution_count) {
            *slot = candidate;
            self.solution_count += 1;
        }
    }

    /// The distinct converged solutions, in the order their seeds were tried.
    pub fn solutions(&self) -> &[InverseKinematicsReport<MAX_CONFIG, T>] {
        self.solutions.get(..self.solution_count).unwrap_or(&[])
    }

    /// Distinct solution count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.solution_count
    }

    /// Whether no start converged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solution_count == 0
    }

    /// Starts run, converged or not.
    #[must_use]
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    /// The kept solution nearest `reference` by [`KinematicTree::configuration_distance`], or
    /// `None` if nothing converged.
    #[must_use]
    pub fn closest_to<const MAX_JOINTS: usize>(
        &self,
        tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        reference: &Vector<MAX_CONFIG, T>,
    ) -> Option<&InverseKinematicsReport<MAX_CONFIG, T>> {
        let mut best: Option<(usize, T)> = None;
        for (index, solution) in self.solutions().iter().enumerate() {
            let distance = tree.configuration_distance(&solution.joint_positions, reference);
            let better = match best {
                Some((_, best_distance)) => distance < best_distance,
                None => true,
            };
            if better {
                best = Some((index, distance));
            }
        }
        best.and_then(|(index, _)| self.solutions().get(index))
    }
}
