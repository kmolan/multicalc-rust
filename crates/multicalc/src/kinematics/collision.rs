//! Sphere/capsule proximity queries: self-collision between primitives attached to a kinematic
//! tree's frames, and environment collision against world-fixed primitives.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::kinematics::kinematic_tree_state::KinematicTreeState;
use crate::linear_algebra::{Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::SE3;

/// A collision shape, posed by an `SE3` supplied alongside it.
///
/// A capsule's central segment runs `-half_length` to `+half_length` along its pose's local z,
/// matching MJCF's own capsule convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Primitive<T: Numeric = f64> {
    /// Ball of the given radius, centred on its pose's origin.
    Sphere {
        /// Ball radius.
        radius: T,
    },
    /// Swept sphere: a segment of the given half-length, thickened by the radius.
    Capsule {
        /// Swept-sphere radius.
        radius: T,
        /// Half the central segment's length.
        half_length: T,
    },
}

impl<T: Numeric> Primitive<T> {
    /// Surface-to-surface distance from `self` at `self_pose` to `other` at `other_pose`, negative
    /// on overlap. Dispatches to the closed form for the shape pair.
    ///
    /// ```
    /// use multicalc::kinematics::Primitive;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let at_origin = SE3::<f64>::identity();
    /// let along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
    /// // Radii 0.3 and 0.2, centres 1 m apart: 0.5 m of clearance.
    /// let ball = Primitive::Sphere { radius: 0.3 };
    /// let other = Primitive::Sphere { radius: 0.2 };
    /// assert!((ball.distance_to(at_origin, other, along_x) - 0.5).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn distance_to(self, self_pose: SE3<T>, other: Primitive<T>, other_pose: SE3<T>) -> T {
        match (self, other) {
            (Primitive::Sphere { radius: first }, Primitive::Sphere { radius: second }) => {
                sphere_sphere_distance(self_pose, first, other_pose, second)
            }
            (
                Primitive::Capsule {
                    radius: first_radius,
                    half_length: first_half_length,
                },
                Primitive::Capsule {
                    radius: second_radius,
                    half_length: second_half_length,
                },
            ) => capsule_capsule_distance(
                self_pose,
                first_radius,
                first_half_length,
                other_pose,
                second_radius,
                second_half_length,
            ),
            (
                Primitive::Sphere { radius },
                Primitive::Capsule {
                    radius: capsule_radius,
                    half_length,
                },
            ) => {
                sphere_capsule_distance(self_pose, radius, other_pose, capsule_radius, half_length)
            }
            (
                Primitive::Capsule {
                    radius: capsule_radius,
                    half_length,
                },
                Primitive::Sphere { radius },
            ) => {
                sphere_capsule_distance(other_pose, radius, self_pose, capsule_radius, half_length)
            }
        }
    }
}

/// Local z of a pose: a capsule's own axis in world axes.
fn capsule_axis<T: Numeric>(pose: SE3<T>) -> Vector3D<T> {
    pose.rotation().act(Vector::new([T::ZERO, T::ZERO, T::ONE]))
}

/// A capsule's central segment endpoints, in world coordinates.
fn capsule_segment<T: Numeric>(pose: SE3<T>, half_length: T) -> (Vector3D<T>, Vector3D<T>) {
    let axis = capsule_axis(pose);
    (
        pose.translation() - axis.scale(half_length),
        pose.translation() + axis.scale(half_length),
    )
}

/// Closest point on segment `[start, end]` to `point`, clamped to the segment. A degenerate
/// segment returns its start.
fn closest_point_on_segment<T: Numeric>(
    point: Vector3D<T>,
    start: Vector3D<T>,
    end: Vector3D<T>,
) -> Vector3D<T> {
    let segment = end - start;
    let length_squared = segment.dot(segment);
    if length_squared <= T::EPSILON {
        return start;
    }
    let parameter = ((point - start).dot(segment) / length_squared)
        .max(T::ZERO)
        .min(T::ONE);
    start + segment.scale(parameter)
}

/// Closest points between two segments, each clamped to its own. Parallel or degenerate segments
/// fall back to a clamped endpoint pair, so the result is finite for every input.
///
/// Ericson, *Real-Time Collision Detection*, §5.1.9.
fn closest_points_between_segments<T: Numeric>(
    first_start: Vector3D<T>,
    first_end: Vector3D<T>,
    second_start: Vector3D<T>,
    second_end: Vector3D<T>,
) -> (Vector3D<T>, Vector3D<T>) {
    let first_direction = first_end - first_start;
    let second_direction = second_end - second_start;
    let start_offset = first_start - second_start;
    let first_length_squared = first_direction.dot(first_direction);
    let second_length_squared = second_direction.dot(second_direction);
    let second_offset = second_direction.dot(start_offset);

    // Both degenerate: two points.
    if first_length_squared <= T::EPSILON && second_length_squared <= T::EPSILON {
        return (first_start, second_start);
    }

    let (first_parameter, second_parameter) = if first_length_squared <= T::EPSILON {
        // First degenerate: project its point onto the second.
        (
            T::ZERO,
            (second_offset / second_length_squared)
                .max(T::ZERO)
                .min(T::ONE),
        )
    } else {
        let first_offset = first_direction.dot(start_offset);
        if second_length_squared <= T::EPSILON {
            // Second degenerate: project its point onto the first.
            (
                (-first_offset / first_length_squared)
                    .max(T::ZERO)
                    .min(T::ONE),
                T::ZERO,
            )
        } else {
            let alignment = first_direction.dot(second_direction);
            let denominator = first_length_squared * second_length_squared - alignment * alignment;
            // Parallel: denominator vanishes, so pin the first parameter and solve the second.
            let candidate = if denominator > T::EPSILON {
                ((alignment * second_offset - first_offset * second_length_squared) / denominator)
                    .max(T::ZERO)
                    .min(T::ONE)
            } else {
                T::ZERO
            };
            let second_candidate = (alignment * candidate + second_offset) / second_length_squared;
            // Off the second segment: clamp it, then re-solve the first against the clamped end.
            if second_candidate < T::ZERO {
                (
                    (-first_offset / first_length_squared)
                        .max(T::ZERO)
                        .min(T::ONE),
                    T::ZERO,
                )
            } else if second_candidate > T::ONE {
                (
                    ((alignment - first_offset) / first_length_squared)
                        .max(T::ZERO)
                        .min(T::ONE),
                    T::ONE,
                )
            } else {
                (candidate, second_candidate)
            }
        }
    };

    (
        first_start + first_direction.scale(first_parameter),
        second_start + second_direction.scale(second_parameter),
    )
}

/// Surface-to-surface distance between two spheres, negative on overlap.
#[must_use]
pub fn sphere_sphere_distance<T: Numeric>(
    first_pose: SE3<T>,
    first_radius: T,
    second_pose: SE3<T>,
    second_radius: T,
) -> T {
    (first_pose.translation() - second_pose.translation()).norm() - first_radius - second_radius
}

/// Surface-to-surface distance between two capsules, negative on overlap: the distance between
/// their central segments, less both radii.
#[must_use]
pub fn capsule_capsule_distance<T: Numeric>(
    first_pose: SE3<T>,
    first_radius: T,
    first_half_length: T,
    second_pose: SE3<T>,
    second_radius: T,
    second_half_length: T,
) -> T {
    let (first_start, first_end) = capsule_segment(first_pose, first_half_length);
    let (second_start, second_end) = capsule_segment(second_pose, second_half_length);
    let (first_closest, second_closest) =
        closest_points_between_segments(first_start, first_end, second_start, second_end);
    (first_closest - second_closest).norm() - first_radius - second_radius
}

/// Surface-to-surface distance between a sphere and a capsule, negative on overlap: the distance
/// from the sphere centre to the capsule's central segment, less both radii.
#[must_use]
pub fn sphere_capsule_distance<T: Numeric>(
    sphere_pose: SE3<T>,
    sphere_radius: T,
    capsule_pose: SE3<T>,
    capsule_radius: T,
    capsule_half_length: T,
) -> T {
    let (start, end) = capsule_segment(capsule_pose, capsule_half_length);
    let centre = sphere_pose.translation();
    (centre - closest_point_on_segment(centre, start, end)).norm() - sphere_radius - capsule_radius
}

/// A primitive rigidly attached to a tree frame, posed in that frame's own coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AttachedPrimitive<T: Numeric = f64> {
    frame_index: usize,
    shape: Primitive<T>,
    local_pose: SE3<T>,
}

/// A world-fixed primitive: an environment obstacle.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WorldPrimitive<T: Numeric = f64> {
    shape: Primitive<T>,
    pose: SE3<T>,
}

/// Which list a [`CollisionReport`]'s closest pair came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionSource {
    /// Index into the query's self-primitive list.
    SelfPrimitive(usize),
    /// Index into the query's environment-primitive list.
    Environment(usize),
}

/// Result of a [`CollisionQuery::check`]: the tightest clearance found and the pair holding it.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct CollisionReport<T: Numeric = f64> {
    /// Smallest surface-to-surface clearance over every checked pair, negative on overlap. Zero
    /// where no pair was checked.
    pub minimum_clearance: T,
    /// The pair holding it, or `None` where no pair was checked.
    pub closest_pair: Option<(CollisionSource, CollisionSource)>,
}

impl<T: Numeric> CollisionReport<T> {
    /// Whether every checked pair met `required_clearance`. Vacuously true with no pairs checked.
    #[must_use]
    pub fn is_clear(&self, required_clearance: T) -> bool {
        self.closest_pair.is_none() || self.minimum_clearance >= required_clearance
    }
}

/// Fixed-capacity broad-phase-free proximity query: every self-primitive against every other
/// self-primitive and against every environment primitive, all pairs, no acceleration structure.
/// Cost is `O(self² + self · environment)`, so capacities are sized for link-count primitives, not
/// mesh-count.
///
/// Environment pairs are never checked against each other. Self-primitive poses come from a solved
/// [`KinematicTreeState`]; environment poses are world-fixed.
///
/// ```
/// use multicalc::kinematics::{CollisionQuery, Joint, JointParent, KinematicTree, Primitive};
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::{SE3, SO3};
///
/// let z = Vector::new([0.0, 0.0, 1.0]);
/// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
/// let tree = KinematicTree::<2, 2, f64>::try_from_joints(
///     &[Joint::revolute(z, SE3::identity()), Joint::fixed(link)],
///     &[JointParent::World, JointParent::Joint(0)],
/// )
/// .unwrap();
/// let state = tree.forward_kinematics(&Vector::zeros()).unwrap();
///
/// let mut query = CollisionQuery::<2, 0, 0, f64>::new();
/// query
///     .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
///     .unwrap();
/// query
///     .push_self_primitive(1, Primitive::Sphere { radius: 0.1 }, SE3::identity())
///     .unwrap();
///
/// // Frames 1 m apart, radii 0.1 each: 0.8 m of clearance.
/// let report = query.check(&state).unwrap();
/// assert!((report.minimum_clearance - 0.8).abs() < 1e-12);
/// assert!(report.is_clear(0.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CollisionQuery<
    const MAX_SELF: usize,
    const MAX_ENVIRONMENT: usize,
    const MAX_EXCLUDED: usize,
    T: Numeric = f64,
> {
    self_primitives: [AttachedPrimitive<T>; MAX_SELF],
    self_count: usize,
    environment_primitives: [WorldPrimitive<T>; MAX_ENVIRONMENT],
    environment_count: usize,
    excluded_pairs: [(usize, usize); MAX_EXCLUDED],
    excluded_count: usize,
}

impl<const MAX_SELF: usize, const MAX_ENVIRONMENT: usize, const MAX_EXCLUDED: usize, T: Numeric>
    Default for CollisionQuery<MAX_SELF, MAX_ENVIRONMENT, MAX_EXCLUDED, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_SELF: usize, const MAX_ENVIRONMENT: usize, const MAX_EXCLUDED: usize, T: Numeric>
    CollisionQuery<MAX_SELF, MAX_ENVIRONMENT, MAX_EXCLUDED, T>
{
    /// An empty query: no primitives, no exclusions.
    #[must_use]
    pub fn new() -> Self {
        let attached_filler = AttachedPrimitive {
            frame_index: 0,
            shape: Primitive::Sphere { radius: T::ZERO },
            local_pose: SE3::identity(),
        };
        let world_filler = WorldPrimitive {
            shape: Primitive::Sphere { radius: T::ZERO },
            pose: SE3::identity(),
        };
        Self {
            self_primitives: [attached_filler; MAX_SELF],
            self_count: 0,
            environment_primitives: [world_filler; MAX_ENVIRONMENT],
            environment_count: 0,
            excluded_pairs: [(0, 0); MAX_EXCLUDED],
            excluded_count: 0,
        }
    }

    /// Attaches `shape` to `frame_index` at `local_pose`, in that frame's own coordinates.
    /// Returns its index in the self list, as taken by [`exclude_pair`](Self::exclude_pair).
    ///
    /// Errors: [`CollisionCapacityExceeded`](KinematicsError::CollisionCapacityExceeded) past
    /// `MAX_SELF`.
    pub fn push_self_primitive(
        &mut self,
        frame_index: usize,
        shape: Primitive<T>,
        local_pose: SE3<T>,
    ) -> Result<usize, KinematicsError> {
        let index = self.self_count;
        let slot = self
            .self_primitives
            .get_mut(index)
            .ok_or(KinematicsError::CollisionCapacityExceeded)?;
        *slot = AttachedPrimitive {
            frame_index,
            shape,
            local_pose,
        };
        self.self_count += 1;
        Ok(index)
    }

    /// Places `shape` at a world-fixed `pose`. Returns its index in the environment list.
    ///
    /// Errors: [`CollisionCapacityExceeded`](KinematicsError::CollisionCapacityExceeded) past
    /// `MAX_ENVIRONMENT`.
    pub fn push_environment_primitive(
        &mut self,
        shape: Primitive<T>,
        pose: SE3<T>,
    ) -> Result<usize, KinematicsError> {
        let index = self.environment_count;
        let slot = self
            .environment_primitives
            .get_mut(index)
            .ok_or(KinematicsError::CollisionCapacityExceeded)?;
        *slot = WorldPrimitive { shape, pose };
        self.environment_count += 1;
        Ok(index)
    }

    /// Drops one self pair from checking, by the indices `push_self_primitive` returned — the
    /// adjacent-link pairs that always sit inside each other. Symmetric in its arguments.
    ///
    /// Errors: [`CollisionCapacityExceeded`](KinematicsError::CollisionCapacityExceeded) past
    /// `MAX_EXCLUDED`.
    pub fn exclude_pair(&mut self, first: usize, second: usize) -> Result<(), KinematicsError> {
        let slot = self
            .excluded_pairs
            .get_mut(self.excluded_count)
            .ok_or(KinematicsError::CollisionCapacityExceeded)?;
        *slot = (first, second);
        self.excluded_count += 1;
        Ok(())
    }

    /// Self-primitive count.
    #[must_use]
    pub fn self_primitive_count(&self) -> usize {
        self.self_count
    }

    /// Environment-primitive count.
    #[must_use]
    pub fn environment_primitive_count(&self) -> usize {
        self.environment_count
    }

    fn is_excluded(&self, first: usize, second: usize) -> bool {
        self.excluded_pairs
            .get(..self.excluded_count)
            .unwrap_or(&[])
            .iter()
            .any(|pair| *pair == (first, second) || *pair == (second, first))
    }

    /// Checks every unexcluded pair, self against self and self against environment. Each
    /// self-primitive is placed at `state.pose(frame_index) * local_pose`.
    ///
    /// Errors: [`ToolIndexOutOfRange`](KinematicsError::ToolIndexOutOfRange) where a primitive is
    /// attached to a frame `state` does not have.
    pub fn check<const MAX_JOINTS: usize>(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
    ) -> Result<CollisionReport<T>, KinematicsError> {
        let mut minimum_clearance = T::MAX;
        let mut closest_pair = None;

        let attached = self.self_primitives.get(..self.self_count).unwrap_or(&[]);
        let environment = self
            .environment_primitives
            .get(..self.environment_count)
            .unwrap_or(&[]);

        for (first_index, first) in attached.iter().enumerate() {
            let first_pose = state
                .pose(first.frame_index)
                .ok_or(KinematicsError::ToolIndexOutOfRange)?
                * first.local_pose;

            for (second_index, second) in attached.iter().enumerate().skip(first_index + 1) {
                if self.is_excluded(first_index, second_index) {
                    continue;
                }
                let second_pose = state
                    .pose(second.frame_index)
                    .ok_or(KinematicsError::ToolIndexOutOfRange)?
                    * second.local_pose;
                let clearance = first
                    .shape
                    .distance_to(first_pose, second.shape, second_pose);
                if clearance < minimum_clearance {
                    minimum_clearance = clearance;
                    closest_pair = Some((
                        CollisionSource::SelfPrimitive(first_index),
                        CollisionSource::SelfPrimitive(second_index),
                    ));
                }
            }

            for (obstacle_index, obstacle) in environment.iter().enumerate() {
                let clearance = first
                    .shape
                    .distance_to(first_pose, obstacle.shape, obstacle.pose);
                if clearance < minimum_clearance {
                    minimum_clearance = clearance;
                    closest_pair = Some((
                        CollisionSource::SelfPrimitive(first_index),
                        CollisionSource::Environment(obstacle_index),
                    ));
                }
            }
        }

        Ok(CollisionReport {
            minimum_clearance: if closest_pair.is_some() {
                minimum_clearance
            } else {
                T::ZERO
            },
            closest_pair,
        })
    }
}
