//! Sharing a wanted push and turn out across a set of rotors.

use crate::error::PlantError;
use crate::linear_algebra::{Matrix, Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::Wrench;

/// Which way a rotor turns, seen from above the body looking down its z axis.
///
/// A rotor drags the air around with it, and the air pushes back, so a rotor twists the body the
/// opposite way to its own turn. Neighbouring rotors are normally set to turn opposite ways so
/// those twists cancel while the body holds still.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorSpin {
    /// Turns clockwise seen from above, so it twists the body the other way, about +z.
    Clockwise,
    /// Turns counter-clockwise seen from above, so it twists the body about −z.
    CounterClockwise,
}

/// What each rotor was asked to push with, and whether any of them was asked for more than it can
/// give.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotorCommands<const ROTOR_COUNT: usize, T: Numeric = f64> {
    thrusts: Vector<ROTOR_COUNT, T>,
    saturated: bool,
}

impl<const ROTOR_COUNT: usize, T: Numeric> RotorCommands<ROTOR_COUNT, T> {
    /// What each rotor was asked to push with, already held inside its limits.
    #[inline]
    pub fn thrusts(self) -> Vector<ROTOR_COUNT, T> {
        self.thrusts
    }

    /// `true` when at least one rotor was asked for more, or less, than it can give.
    ///
    /// When this is `true` the body does not get the push and turn that was asked for, because
    /// what was left over has nowhere to go. A controller usually wants to know, so it can stop
    /// winding up against a limit it cannot reach past.
    #[inline]
    #[must_use]
    pub fn saturated(self) -> bool {
        self.saturated
    }
}

/// Turns a wanted push and turn into a thrust for each rotor, and back again.
///
/// Every rotor pushes along the body's own z axis. Where a rotor sits decides how much it tips the
/// body when it pushes harder, and which way it turns decides how much it twists the body about z.
/// Put those together for every rotor and there is one fixed relation between the set of rotor
/// thrusts and the push and turn the body feels; this type holds that relation and the way back
/// through it, both worked out once when it is built, so a tick is one small matrix product.
///
/// Rotor positions are in the body's own axes, measured from the body frame's origin, and so is
/// the turn the mixer is asked for.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::plant::MultirotorMixer;
///
/// // Four rotors 15 cm out, each twisting the body 1.6 cm-worth per newton of push, able to give
/// // between nothing and 5 N.
/// let arm_length = 0.15_f64;
/// let torque_per_thrust = 0.016;
/// let minimum_thrust = 0.0;
/// let maximum_thrust = 5.0;
/// let rotor_count = 4;
///
/// let mixer = MultirotorMixer::<4, f64>::quadrotor_x(
///     arm_length,
///     torque_per_thrust,
///     minimum_thrust,
///     maximum_thrust,
/// )
/// .unwrap();
///
/// // A 0.8 kg machine asked to carry its own weight and turn not at all: the push is shared out
/// // evenly and no rotor is stretched.
/// let mass = 0.8;
/// let gravity_strength = 9.81;
/// let weight = mass * gravity_strength;
/// let no_turn = Vector::new([0.0, 0.0, 0.0]);
///
/// let commands = mixer.rotor_thrusts(weight, no_turn);
/// let even_share = weight / 4.0;
/// assert!(!commands.saturated());
/// for rotor in 0..rotor_count {
///     assert!((commands.thrusts()[rotor] - even_share).abs() < 1e-12);
/// }
///
/// // Those four thrusts add back up to the push that was asked for, and to no turn at all.
/// let produced = mixer.wrench(commands.thrusts());
/// assert!((produced.force()[2] - weight).abs() < 1e-12);
/// assert!(produced.force()[0].abs() < 1e-12 && produced.force()[1].abs() < 1e-12);
/// assert!(produced.torque().norm() < 1e-12);
///
/// // Asked for far more than the rotors have, every one of them sits at its limit and says so.
/// let beyond_reach = 30.0;
/// let too_much = mixer.rotor_thrusts(beyond_reach, no_turn);
/// assert!(too_much.saturated());
/// for rotor in 0..rotor_count {
///     assert!((too_much.thrusts()[rotor] - maximum_thrust).abs() < 1e-12);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultirotorMixer<const ROTOR_COUNT: usize, T: Numeric = f64> {
    allocation: Matrix<4, ROTOR_COUNT, T>,
    distribution: Matrix<ROTOR_COUNT, 4, T>,
    minimum_thrust: T,
    maximum_thrust: T,
}

impl<const ROTOR_COUNT: usize, T: Numeric> MultirotorMixer<ROTOR_COUNT, T> {
    /// Builds a mixer from where the rotors sit and which way each one turns.
    ///
    /// `positions` are in the body's own axes, measured from the body frame's origin.
    /// `torque_per_thrust` is how much a rotor twists the body about z for each newton it pushes
    /// with. `minimum_thrust` and `maximum_thrust` are what one rotor can give.
    ///
    /// Returns [`PlantError::NonFinite`] if any value is not finite,
    /// [`PlantError::NonPositiveTorqueRatio`] if the twist per unit of push is zero or negative,
    /// [`PlantError::InvalidThrustLimits`] if the smallest thrust is not below the largest,
    /// [`PlantError::Linalg`] if the layout cannot be inverted, or
    /// [`PlantError::RotorLayoutNotIndependent`] if the rotors cannot produce every wanted push
    /// and turn — which is what happens with fewer than four rotors, or with rotors all in a line.
    pub fn new(
        positions: [Vector3D<T>; ROTOR_COUNT],
        spins: [RotorSpin; ROTOR_COUNT],
        torque_per_thrust: T,
        minimum_thrust: T,
        maximum_thrust: T,
    ) -> Result<Self, PlantError> {
        if !torque_per_thrust.is_finite()
            || !minimum_thrust.is_finite()
            || !maximum_thrust.is_finite()
            || positions.iter().any(|p| !p.is_finite())
        {
            return Err(PlantError::NonFinite);
        }
        if torque_per_thrust <= T::ZERO {
            return Err(PlantError::NonPositiveTorqueRatio);
        }
        if maximum_thrust <= minimum_thrust {
            return Err(PlantError::InvalidThrustLimits);
        }

        let allocation = Matrix::<4, ROTOR_COUNT, T>::from_fn(|row, rotor| {
            let position = positions[rotor];
            match row {
                0 => T::ONE,
                1 => position[1],
                2 => -position[0],
                _ => match spins[rotor] {
                    RotorSpin::Clockwise => torque_per_thrust,
                    RotorSpin::CounterClockwise => -torque_per_thrust,
                },
            }
        });
        let distribution = allocation.pseudo_inverse()?;

        // Going out to the rotors and back has to land where it started; when it does not, some
        // wanted push or turn is one the rotors simply cannot produce. The bar is loose enough
        // that single precision passes and tight enough that a missing direction, which is off by
        // about one, always fails.
        let round_trip = allocation * distribution;
        let bar = T::from_f64(1e-4);
        for row in 0..4 {
            for col in 0..4 {
                let wanted = if row == col { T::ONE } else { T::ZERO };
                if (round_trip[(row, col)] - wanted).abs() > bar {
                    return Err(PlantError::RotorLayoutNotIndependent);
                }
            }
        }

        Ok(MultirotorMixer {
            allocation,
            distribution,
            minimum_thrust,
            maximum_thrust,
        })
    }

    /// How a set of rotor thrusts turns into the push and turn the body feels.
    #[inline]
    pub fn allocation(self) -> Matrix<4, ROTOR_COUNT, T> {
        self.allocation
    }

    /// The least one rotor can push with.
    #[inline]
    #[must_use]
    pub fn minimum_thrust(self) -> T {
        self.minimum_thrust
    }

    /// The most one rotor can push with.
    #[inline]
    #[must_use]
    pub fn maximum_thrust(self) -> T {
        self.maximum_thrust
    }

    /// Works out what each rotor has to push with to give the body a wanted push and turn.
    ///
    /// `collective_thrust` is the total push wanted along the body's z axis, and `torque` the turn
    /// wanted about the body's own axes, taken about the body frame's origin. Each rotor's share
    /// is held inside its limits, and the result says whether any of them hit one.
    ///
    /// A command that is not finite comes back not finite rather than being rejected — this runs
    /// every tick, so checking is the caller's job, once, upstream.
    #[must_use]
    pub fn rotor_thrusts(
        self,
        collective_thrust: T,
        torque: Vector3D<T>,
    ) -> RotorCommands<ROTOR_COUNT, T> {
        let wanted = Vector::new([collective_thrust, torque[0], torque[1], torque[2]]);
        let share = self.distribution * wanted;
        let mut saturated = false;
        let thrusts = Vector::from_fn(|rotor| {
            let value = share[rotor];
            if value < self.minimum_thrust {
                saturated = true;
                self.minimum_thrust
            } else if value > self.maximum_thrust {
                saturated = true;
                self.maximum_thrust
            } else {
                value
            }
        });
        RotorCommands { thrusts, saturated }
    }

    /// The push and turn a set of rotor thrusts adds up to.
    ///
    /// The push is along the body's z axis and the turn is about the body's own axes, taken about
    /// the body frame's origin — the form [`RigidBody`](crate::dynamics::RigidBody) takes, so a
    /// mixer's output drives a body without anything in between.
    #[must_use]
    pub fn wrench(self, thrusts: Vector<ROTOR_COUNT, T>) -> Wrench<T> {
        let produced = self.allocation * thrusts;
        Wrench::new(
            Vector::new([T::ZERO, T::ZERO, produced[0]]),
            Vector::new([produced[1], produced[2], produced[3]]),
        )
    }
}

impl<T: Numeric> MultirotorMixer<4, T> {
    /// Builds a mixer for the usual four-rotor layout, with the rotors at the corners of an
    /// X and the body's nose pointing between two of them.
    ///
    /// `arm_length` is how far each rotor sits from the body frame's origin. The rotors come out
    /// in the order front right, back left, front left, back right; the two on each diagonal turn
    /// the same way, so their twists about z cancel while the body holds still.
    ///
    /// Returns [`PlantError::NonFinite`] if any value is not finite,
    /// [`PlantError::NonPositiveArmLength`] if the arm length is zero or negative, and otherwise
    /// whatever [`MultirotorMixer::new`] would return.
    pub fn quadrotor_x(
        arm_length: T,
        torque_per_thrust: T,
        minimum_thrust: T,
        maximum_thrust: T,
    ) -> Result<Self, PlantError> {
        if !arm_length.is_finite() {
            return Err(PlantError::NonFinite);
        }
        if arm_length <= T::ZERO {
            return Err(PlantError::NonPositiveArmLength);
        }
        let half = arm_length / T::TWO.sqrt();
        let positions = [
            Vector::new([half, -half, T::ZERO]),
            Vector::new([-half, half, T::ZERO]),
            Vector::new([half, half, T::ZERO]),
            Vector::new([-half, -half, T::ZERO]),
        ];
        let spins = [
            RotorSpin::Clockwise,
            RotorSpin::Clockwise,
            RotorSpin::CounterClockwise,
            RotorSpin::CounterClockwise,
        ];
        Self::new(
            positions,
            spins,
            torque_per_thrust,
            minimum_thrust,
            maximum_thrust,
        )
    }
}
