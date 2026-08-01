//! The state of a body that can move freely in space.

use crate::linear_algebra::Vector;
use crate::scalar::Numeric;
use crate::spatial::{Quaternion, SE3, SO3, Twist};

/// Where a freely moving body is and how it is moving.
///
/// A free joint is the connection a body has to the world when nothing constrains it: it can slide
/// in any direction and turn about any axis. Its position and orientation together take seven
/// numbers — three for where it is, four for which way it faces — while its motion takes six, one
/// per direction it is free to move in.
///
/// The seven place numbers read `[x, y, z, w, qx, qy, qz]`, position first and the orientation's
/// leading number next, which is how MuJoCo writes a free joint. The six motion numbers are in the
/// crate-wide `[v; ω]` order, linear part first.
///
/// ```
/// use multicalc::spatial::{FreeJointState, SE3, Twist};
/// let at_rest = FreeJointState::new(SE3::<f64>::identity(), Twist::zeros());
/// assert_eq!(
///     at_rest.generalized_position(),
///     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeJointState<T: Numeric = f64> {
    pose: SE3<T>,
    velocity: Twist<T>,
}

impl<T: Numeric> FreeJointState<T> {
    /// How many numbers it takes to say where the body is and which way it faces.
    pub const GENERALIZED_POSITION_DIMENSION: usize = 7;

    /// How many numbers it takes to say how the body is moving.
    pub const GENERALIZED_VELOCITY_DIMENSION: usize = 6;

    /// A state from a pose and a velocity.
    ///
    /// ```
    /// use multicalc::spatial::{FreeJointState, SE3, Twist};
    /// let state = FreeJointState::new(SE3::<f64>::identity(), Twist::zeros());
    /// assert_eq!(state.velocity(), Twist::zeros());
    /// ```
    #[inline]
    #[must_use]
    pub fn new(pose: SE3<T>, velocity: Twist<T>) -> Self {
        FreeJointState { pose, velocity }
    }

    /// A body sitting at the origin, facing along the world axes, not moving.
    ///
    /// ```
    /// use multicalc::spatial::FreeJointState;
    /// assert_eq!(
    ///     FreeJointState::<f64>::identity().generalized_position(),
    ///     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        FreeJointState {
            pose: SE3::identity(),
            velocity: Twist::zeros(),
        }
    }

    /// Where the body is and which way it faces.
    #[inline]
    #[must_use]
    pub fn pose(self) -> SE3<T> {
        self.pose
    }

    /// How the body is moving.
    #[inline]
    #[must_use]
    pub fn velocity(self) -> Twist<T> {
        self.velocity
    }

    /// A state from the numbers a free joint is written as.
    ///
    /// Returns `None` when the four orientation numbers are all zero, which names no direction.
    /// Anything else is scaled to unit length first, so a slightly drifted orientation is accepted.
    ///
    /// ```
    /// use multicalc::spatial::FreeJointState;
    /// let place = [1.0_f64, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0];
    /// let state = FreeJointState::from_generalized_vectors(place, [0.0; 6]).unwrap();
    /// assert_eq!(state.generalized_position(), place);
    ///
    /// assert!(FreeJointState::from_generalized_vectors([0.0_f64; 7], [0.0; 6]).is_none());
    /// ```
    #[must_use]
    pub fn from_generalized_vectors(position: [T; 7], velocity: [T; 6]) -> Option<Self> {
        let [x, y, z, w, qx, qy, qz] = position;
        let orientation = Quaternion::new(w, qx, qy, qz).try_normalized()?;
        Some(FreeJointState {
            pose: SE3::from_parts(SO3::from_quaternion(orientation), Vector::new([x, y, z])),
            velocity: Twist::from_array(velocity),
        })
    }

    /// The seven numbers saying where the body is and which way it faces.
    ///
    /// ```
    /// use multicalc::spatial::FreeJointState;
    /// let state = FreeJointState::<f64>::identity();
    /// assert_eq!(state.generalized_position(), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn generalized_position(self) -> [T; 7] {
        let [x, y, z] = *self.pose.translation().as_array();
        let [w, qx, qy, qz] = self.pose.rotation().quaternion().as_array();
        [x, y, z, w, qx, qy, qz]
    }

    /// The six numbers saying how the body is moving.
    ///
    /// The same motion [`velocity`](FreeJointState::velocity) reports, written as loose numbers
    /// rather than as a typed spatial velocity.
    ///
    /// ```
    /// use multicalc::spatial::FreeJointState;
    /// assert_eq!(FreeJointState::<f64>::identity().generalized_velocity(), [0.0; 6]);
    /// ```
    #[inline]
    #[must_use]
    pub fn generalized_velocity(self) -> [T; 6] {
        self.velocity.as_array()
    }
}
