//! A single body free to move in all six directions, and how it responds to the forces on it.

use crate::error::DynamicsError;
use crate::linear_algebra::{Matrix3D, Vector, Vector3D};
use crate::ode::ExponentialMap;
use crate::scalar::Numeric;
use crate::spatial::{FreeJointState, Quaternion, SE3, SO3, SpatialInertia, Twist, Wrench};

/// How many numbers it takes to say where a free body is and how it is moving.
/// index  0  1  2   3   4   5   6   7  8  9   10 11 12
///        px py pz  qw  qx  qy  qz  vx vy vz  ωx ωy ωz
/// p — position in world frame
/// q — orientation in body frame
/// v — linear velocity in world frame
/// ω — angular velocity in body frame
pub const STATE_DIMENSION: usize = 13;

/// A body free to move in all six directions, and what the forces on it do to it.
///
/// Give it the body's mass and how that mass is spread out, plus the pull of gravity, and it
/// answers the question an integrator keeps asking: given where the body is pointing, how fast it
/// is turning, and what is pushing on it, how quickly is its motion changing? The work needed
/// every time — inverting the body's resistance to spinning — is done once here, so each later
/// call is a fixed handful of small products with nothing that can fail.
///
/// Straight-line motion is measured in world axes; turning, and the forces applied to the body,
/// are in the body's own axes.
///
/// ```
/// use multicalc::dynamics::RigidBody;
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::SpatialInertia;
///
/// // A small flying machine that balances on its own origin.
/// let mass = 0.8_f64;
/// let balance_point = Vector::new([0.0, 0.0, 0.0]);
/// let resistance_to_spinning = Vector::new([0.005, 0.007, 0.009]);
/// let earth_gravity = Vector::new([0.0, 0.0, -9.81]);
///
/// let inertia =
///     SpatialInertia::from_diagonal_inertia(mass, balance_point, resistance_to_spinning).unwrap();
/// let body = RigidBody::new(inertia, earth_gravity).unwrap();
///
/// assert_eq!(body.inertia().mass(), mass);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBody<T: Numeric = f64> {
    inertia: SpatialInertia<T>,
    inverse_rotational_inertia: Matrix3D<T>,
    gravity: Vector3D<T>,
}

impl<T: Numeric> RigidBody<T> {
    /// Builds a body from how its mass is spread out and the pull of gravity.
    ///
    /// `gravity` is the acceleration gravity gives the body, in world axes — on Earth with the
    /// world's z axis pointing up, that is `[0, 0, -9.81]`.
    ///
    /// Returns [`DynamicsError::NonFinite`] if any value is not finite,
    /// [`DynamicsError::NonPositiveInertia`] if the body's resistance to spinning is not positive
    /// definite, or [`DynamicsError::Linalg`] if it cannot be inverted.
    pub fn new(inertia: SpatialInertia<T>, gravity: Vector3D<T>) -> Result<Self, DynamicsError> {
        if !inertia.is_finite() || !gravity.is_finite() {
            return Err(DynamicsError::NonFinite);
        }
        let rotational_inertia = inertia.rotational_inertia();
        if rotational_inertia.cholesky().is_err() {
            return Err(DynamicsError::NonPositiveInertia);
        }
        let inverse_rotational_inertia = rotational_inertia.inverse()?;
        Ok(RigidBody {
            inertia,
            inverse_rotational_inertia,
            gravity,
        })
    }

    /// How the body's mass is spread out.
    #[inline]
    #[must_use]
    pub fn inertia(self) -> SpatialInertia<T> {
        self.inertia
    }

    /// The acceleration gravity gives the body, in world axes.
    #[inline]
    pub fn gravity(self) -> Vector3D<T> {
        self.gravity
    }

    /// How quickly the body's motion changes under a given push and turn.
    ///
    /// `orientation` is which way the body is facing and `angular_rate` how fast it is turning,
    /// in the body's own axes. `applied_wrench` is everything pushing and turning the body apart
    /// from gravity — rotor thrust, drag, a gust — with both parts read in the body's own axes and
    /// the turning part taken about the body frame's origin. A force given in world axes is turned
    /// into body axes first with `orientation.inverse().act(force)`.
    ///
    /// The answer comes back as how fast the body frame's origin picks up speed, in world axes,
    /// and how fast the spin changes, in body axes.
    ///
    /// ```
    /// use multicalc::dynamics::RigidBody;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SO3, SpatialInertia, Wrench};
    ///
    /// let mass = 2.0_f64;
    /// let gravity_strength = 9.81;
    /// let balance_point = Vector::new([0.0, 0.0, 0.0]);
    /// let resistance_to_spinning = Vector::new([1.0, 1.0, 1.0]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -gravity_strength]);
    ///
    /// let inertia =
    ///     SpatialInertia::from_diagonal_inertia(mass, balance_point, resistance_to_spinning)
    ///         .unwrap();
    /// let body = RigidBody::new(inertia, earth_gravity).unwrap();
    ///
    /// // Level, not turning, nothing pushing on it: it falls.
    /// let level = SO3::identity();
    /// let not_turning = Vector::new([0.0, 0.0, 0.0]);
    /// let nothing_applied = Wrench::zeros();
    ///
    /// let falling = body.accelerations(level, not_turning, nothing_applied);
    /// assert!((falling.linear() - earth_gravity).norm() < 1e-12);
    /// assert!(falling.angular().norm() < 1e-12);
    ///
    /// // Pushing straight up with exactly its own weight holds it still.
    /// let weight = mass * gravity_strength;
    /// let push_up = Vector::new([0.0, 0.0, weight]);
    /// let no_turn = Vector::new([0.0, 0.0, 0.0]);
    /// let hover = Wrench::new(push_up, no_turn);
    ///
    /// let held = body.accelerations(level, not_turning, hover);
    /// assert!(held.linear().norm() < 1e-12);
    /// ```
    #[must_use]
    pub fn accelerations(
        self,
        orientation: SO3<T>,
        angular_rate: Vector3D<T>,
        applied_wrench: Wrench<T>,
    ) -> RigidBodyAcceleration<T> {
        let mass = self.inertia.mass();
        let balance_point = self.inertia.center_of_mass();
        let rotational_inertia = self.inertia.rotational_inertia();
        let force = applied_wrench.force();

        // The turn, taken about the point the body balances on rather than about its origin.
        let turn_about_balance_point = applied_wrench.torque() - balance_point.cross(force);
        // A spinning body resists having its axis moved, which eats part of the turn applied.
        let spin_resistance = angular_rate.cross(rotational_inertia * angular_rate);
        let angular =
            self.inverse_rotational_inertia * (turn_about_balance_point - spin_resistance);

        // Gravity acts through the balance point, so it never turns the body.
        let balance_point_acceleration = orientation.act(force).scale(T::ONE / mass) + self.gravity;
        // The origin swings around the balance point as the body turns, so it picks up speed the
        // balance point does not.
        let swing =
            angular.cross(balance_point) + angular_rate.cross(angular_rate.cross(balance_point));
        let linear = balance_point_acceleration - orientation.act(swing);

        RigidBodyAcceleration { linear, angular }
    }

    /// Moves a body one tick forward under the forces on it.
    ///
    /// `state` is where the body is and how it is moving, `applied_wrench` is everything pushing
    /// and turning it apart from gravity — read in the body's own axes, with the turning part taken
    /// about the body frame's origin — and `timestep` is how long the tick lasts.
    ///
    /// The direction the body faces is carried forward as a turn rather than as four loose numbers,
    /// so what comes back is still a true rotation to within rounding, with no drift to scale away.
    /// Everything is read once at the start of the tick and once half way through, and the whole
    /// state is moved with the half-way values, so the error shrinks with the square of the tick
    /// length. That is coarser than handing [`RigidBody::state_derivative`] to
    /// [`Rk4`](crate::ode::Rk4), which shrinks with the fourth power but lets the direction drift;
    /// pick by which of the two matters more for the run.
    ///
    /// Nothing here can fail and nothing is allocated: two goes at working out the accelerations
    /// and two turns composed on.
    ///
    /// ```
    /// use multicalc::dynamics::RigidBody;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{FreeJointState, SE3, SpatialInertia, Twist, Wrench};
    ///
    /// let mass = 1.0_f64;
    /// let gravity_strength = 9.81;
    /// let balance_point = Vector::new([0.0, 0.0, 0.0]);
    /// let resistance_to_spinning = Vector::new([0.01, 0.01, 0.02]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -gravity_strength]);
    ///
    /// let inertia =
    ///     SpatialInertia::from_diagonal_inertia(mass, balance_point, resistance_to_spinning)
    ///         .unwrap();
    /// let body = RigidBody::new(inertia, earth_gravity).unwrap();
    ///
    /// // Let go while spinning about its own z axis, with nothing pushing on it.
    /// let spinning = Twist::new(Vector::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 3.0]));
    /// let mut state = FreeJointState::new(SE3::identity(), spinning);
    /// let nothing_applied = Wrench::zeros();
    ///
    /// let step = 0.001;
    /// let step_count = 1000;
    /// for _ in 0..step_count {
    ///     state = body.stepped(state, nothing_applied, step);
    /// }
    ///
    /// // One second on it has fallen about 4.9 m and turned 3 rad about z.
    /// let fall_time = 1.0;
    /// let expected_fall = 0.5 * gravity_strength * fall_time * fall_time;
    /// assert!((state.pose().translation()[2] + expected_fall).abs() < 1e-9);
    /// assert!((state.pose().rotation().log()[2] - 3.0).abs() < 1e-9);
    /// // The direction it faces is still a true rotation.
    /// let facing = state.pose().rotation().quaternion();
    /// assert!((facing.norm() - 1.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn stepped(
        self,
        state: FreeJointState<T>,
        applied_wrench: Wrench<T>,
        timestep: T,
    ) -> FreeJointState<T> {
        let pose = state.pose();
        let orientation = pose.rotation();
        let position = pose.translation();
        let velocity = state.velocity();
        let linear_velocity = velocity.linear();
        let angular_rate = velocity.angular();
        let half = timestep * T::HALF;

        let at_start = self.accelerations(orientation, angular_rate, applied_wrench);
        let half_way_orientation = ExponentialMap::attitude_step(orientation, angular_rate, half);
        let half_way_linear_velocity = linear_velocity + at_start.linear() * half;
        let half_way_angular_rate = angular_rate + at_start.angular() * half;

        let half_way =
            self.accelerations(half_way_orientation, half_way_angular_rate, applied_wrench);

        let next_orientation =
            ExponentialMap::attitude_step(orientation, half_way_angular_rate, timestep);
        let next_position = position + half_way_linear_velocity * timestep;
        let next_linear_velocity = linear_velocity + half_way.linear() * timestep;
        let next_angular_rate = angular_rate + half_way.angular() * timestep;

        FreeJointState::new(
            SE3::from_parts(next_orientation, next_position),
            Twist::new(next_linear_velocity, next_angular_rate),
        )
    }

    /// How the thirteen numbers change with time, ready to hand to an integrator.
    ///
    /// index  0  1  2   3   4   5   6   7  8  9   10 11 12
    ///        px py pz  qw  qx  qy  qz  vx vy vz  ωx ωy ωz
    /// p — position in world frame
    /// q — orientation in body frame
    /// v — linear velocity in world frame
    /// ω — angular velocity in body frame
    ///
    /// The four orientation numbers drift away from unit length as an integrator steps, so they
    /// are scaled back to unit length before they are read as a direction. The four numbers
    /// themselves are left to drift and the caller scales them when it wants to. A state whose
    /// four orientation numbers are all zero names no direction, and the whole derivative comes
    /// back as zeros rather than a guess.
    ///
    /// ```
    /// use multicalc::dynamics::{RigidBody, state_vector_from_free_joint};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::ode::Rk4;
    /// use multicalc::spatial::{FreeJointState, SE3, SpatialInertia, Twist, Wrench};
    ///
    /// let mass = 1.0_f64;
    /// let gravity_strength = 9.81;
    /// let balance_point = Vector::new([0.0, 0.0, 0.0]);
    /// let resistance_to_spinning = Vector::new([0.01, 0.01, 0.02]);
    /// let earth_gravity = Vector::new([0.0, 0.0, -gravity_strength]);
    ///
    /// let inertia =
    ///     SpatialInertia::from_diagonal_inertia(mass, balance_point, resistance_to_spinning)
    ///         .unwrap();
    /// let body = RigidBody::new(inertia, earth_gravity).unwrap();
    ///
    /// // Sitting at the origin, not moving and not turning, with nothing pushing on it.
    /// let at_rest = FreeJointState::new(SE3::identity(), Twist::zeros());
    /// let start = state_vector_from_free_joint(at_rest);
    /// let nothing_applied = Wrench::zeros();
    /// let rate = |_time: f64, state: &Vector<13, f64>| {
    ///     body.state_derivative(state, nothing_applied)
    /// };
    ///
    /// // Dropped and left alone for one second.
    /// let start_time = 0.0;
    /// let step = 0.001;
    /// let step_count = 1000;
    /// let fall_time = 1.0;
    /// let after = Rk4::integrate(&rate, start_time, &start, step, step_count, |_time, _state| {});
    ///
    /// // It has fallen about 4.9 m and is heading down at about 9.8 m/s.
    /// let expected_fall = 0.5 * gravity_strength * fall_time * fall_time;
    /// let expected_speed = gravity_strength * fall_time;
    /// assert!((after[2] + expected_fall).abs() < 1e-9);
    /// assert!((after[9] + expected_speed).abs() < 1e-9);
    /// ```
    pub fn state_derivative(
        self,
        state: &Vector<STATE_DIMENSION, T>,
        applied_wrench: Wrench<T>,
    ) -> Vector<STATE_DIMENSION, T> {
        let stored = Quaternion::new(state[3], state[4], state[5], state[6]);
        let Some(unit) = stored.try_normalized() else {
            return Vector::zeros();
        };
        let angular_rate = Vector::new([state[10], state[11], state[12]]);
        let acceleration =
            self.accelerations(SO3::from_quaternion(unit), angular_rate, applied_wrench);

        let (w, x, y, z) = (stored.w(), stored.x(), stored.y(), stored.z());
        let [rate_x, rate_y, rate_z] = *angular_rate.as_array();
        let facing = [
            T::HALF * (-x * rate_x - y * rate_y - z * rate_z),
            T::HALF * (w * rate_x + y * rate_z - z * rate_y),
            T::HALF * (w * rate_y + z * rate_x - x * rate_z),
            T::HALF * (w * rate_z + x * rate_y - y * rate_x),
        ];
        let linear = acceleration.linear();
        let angular = acceleration.angular();

        Vector::new([
            state[7], state[8], state[9], facing[0], facing[1], facing[2], facing[3], linear[0],
            linear[1], linear[2], angular[0], angular[1], angular[2],
        ])
    }
}

/// The thirteen numbers an integrator carries, taken from a free body's pose and motion.
///
/// The order is where the body is (3), which way it faces as four numbers with the leading one
/// first (4), how fast it is moving in world axes (3), and how fast it is turning in its own axes
/// (3) — the free joint's own seven place numbers followed by its six motion numbers.
pub fn state_vector_from_free_joint<T: Numeric>(
    state: FreeJointState<T>,
) -> Vector<STATE_DIMENSION, T> {
    let place = state.generalized_position();
    let motion = state.generalized_velocity();
    Vector::from_fn(|index| {
        if index < 7 {
            place[index]
        } else {
            motion[index - 7]
        }
    })
}

/// A free body's pose and motion, read back out of the thirteen numbers.
///
/// Returns `None` when the four numbers saying which way the body faces are all zero, which names
/// no direction. Anything else is scaled to unit length first, so a slightly drifted orientation
/// is accepted.
#[must_use]
pub fn free_joint_from_state_vector<T: Numeric>(
    state: &Vector<STATE_DIMENSION, T>,
) -> Option<FreeJointState<T>> {
    let mut place = [T::ZERO; 7];
    let mut motion = [T::ZERO; 6];
    for (index, value) in place.iter_mut().enumerate() {
        *value = state[index];
    }
    for (index, value) in motion.iter_mut().enumerate() {
        *value = state[index + 7];
    }
    FreeJointState::from_generalized_vectors(place, motion)
}

/// How quickly a body's motion is changing.
///
/// The straight-line part is how fast the body frame's origin is picking up speed, in world axes.
/// The turning part is how fast its spin is changing, in the body's own axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyAcceleration<T: Numeric = f64> {
    linear: Vector3D<T>,
    angular: Vector3D<T>,
}

impl<T: Numeric> RigidBodyAcceleration<T> {
    /// A value from its straight-line and turning parts.
    #[inline]
    #[must_use]
    pub fn new(linear: Vector3D<T>, angular: Vector3D<T>) -> Self {
        RigidBodyAcceleration { linear, angular }
    }

    /// How fast the body frame's origin is picking up speed, in world axes.
    #[inline]
    pub fn linear(self) -> Vector3D<T> {
        self.linear
    }

    /// How fast the body's spin is changing, in the body's own axes.
    #[inline]
    pub fn angular(self) -> Vector3D<T> {
        self.angular
    }
}
