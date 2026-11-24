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

/// Rotor thrusts inside the limits, and the part of the command that they do not produce.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotorCommands<const ROTOR_COUNT: usize, T: Numeric = f64> {
    thrusts: Vector<ROTOR_COUNT, T>,
    shortfall: Vector<4, T>,
}

impl<const ROTOR_COUNT: usize, T: Numeric> RotorCommands<ROTOR_COUNT, T> {
    /// The thrust of each rotor, inside `[minimum_thrust, maximum_thrust]`.
    #[inline]
    pub fn thrusts(self) -> Vector<ROTOR_COUNT, T> {
        self.thrusts
    }

    /// `[collective, τx, τy, τz]` commanded minus the command that the thrusts produce.
    ///
    /// A negative collective component means that the collective went up to keep the torque.
    #[inline]
    pub fn shortfall(self) -> Vector<4, T> {
        self.shortfall
    }

    /// `true` when any component of [`shortfall`](Self::shortfall) is not zero.
    ///
    /// An outer loop can use this for anti-windup.
    #[inline]
    #[must_use]
    pub fn saturated(self) -> bool {
        self.shortfall
            .as_array()
            .iter()
            .any(|component| *component != T::ZERO)
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
/// // The collective gives way; no torque was asked for, so none is lost.
/// assert!((too_much.shortfall()[0] - (beyond_reach - 4.0 * maximum_thrust)).abs() < 1e-12);
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
            || positions.iter().any(|position| !position.is_finite())
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
        if ROTOR_COUNT < 4 {
            return Err(PlantError::RotorLayoutNotIndependent);
        }
        // Rank of `Aᵀ` (`ROTOR_COUNT × 4`) by the `matrix_rank` rule σ > ROTOR_COUNT·ε·σ_max.
        let decomposition = allocation.transpose().svd()?;
        let cutoff = T::from_usize(ROTOR_COUNT) * T::EPSILON * decomposition.singular_values()[0];
        if decomposition.rank(cutoff) < 4 {
            return Err(PlantError::RotorLayoutNotIndependent);
        }
        // (Aᵀ)⁺ = (A⁺)ᵀ.
        let distribution = decomposition.pseudo_inverse().transpose();

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

    /// Rotor thrusts for a collective thrust along body z and a torque about the body-frame
    /// origin, with the torque before the collective.
    ///
    /// With `d = A⁺·e₁`, the collective `c` moves, up or down, into the interval where
    /// `c·d + A⁺·[0; τ]` is inside the rotor limits. If no `c` fits the full torque, yaw is
    /// scaled down first. If roll and pitch alone do not fit, yaw is zero and roll and pitch are
    /// scaled together. [`RotorCommands::shortfall`] gives what the thrusts do not produce. The work
    /// is `O(ROTOR_COUNT²)`, with no iteration.
    ///
    /// A command that is not finite comes back not finite. This runs each tick, so the caller
    /// checks the input once, upstream.
    #[must_use]
    pub fn rotor_thrusts(
        self,
        collective_thrust: T,
        torque: Vector3D<T>,
    ) -> RotorCommands<ROTOR_COUNT, T> {
        let wanted = Vector::new([collective_thrust, torque[0], torque[1], torque[2]]);
        if !wanted.is_finite() {
            return RotorCommands {
                thrusts: self.distribution * wanted,
                shortfall: wanted,
            };
        }

        let roll_pitch_share = Vector::from_fn(|rotor| {
            self.distribution[(rotor, 1)] * torque[0] + self.distribution[(rotor, 2)] * torque[1]
        });
        let yaw_share = Vector::from_fn(|rotor| self.distribution[(rotor, 3)] * torque[2]);

        let mut produced_torque = torque;
        let (lower, upper) = self.collective_interval(roll_pitch_share + yaw_share);
        if lower > upper {
            let (roll_pitch_lower, roll_pitch_upper) = self.collective_interval(roll_pitch_share);
            if roll_pitch_lower <= roll_pitch_upper {
                let scale = self.largest_feasible_scale(roll_pitch_share, yaw_share);
                produced_torque = Vector::new([torque[0], torque[1], scale * torque[2]]);
            } else {
                let scale = self.largest_feasible_scale(Vector::zeros(), roll_pitch_share);
                produced_torque = Vector::new([scale * torque[0], scale * torque[1], T::ZERO]);
            }
        }

        let torque_share = Vector::from_fn(|rotor| {
            self.distribution[(rotor, 1)] * produced_torque[0]
                + self.distribution[(rotor, 2)] * produced_torque[1]
                + self.distribution[(rotor, 3)] * produced_torque[2]
        });
        // An interval that rounding left empty spans `[upper, lower]`; clamping into it stays
        // within rounding of each bound.
        let (lower, upper) = self.collective_interval(torque_share);
        let produced_collective =
            clamp_keeping_nan(collective_thrust, lower.min(upper), lower.max(upper));

        let thrusts = Vector::from_fn(|rotor| {
            clamp_keeping_nan(
                produced_collective * self.distribution[(rotor, 0)] + torque_share[rotor],
                self.minimum_thrust,
                self.maximum_thrust,
            )
        });
        let shortfall = Vector::new([
            collective_thrust - produced_collective,
            torque[0] - produced_torque[0],
            torque[1] - produced_torque[1],
            torque[2] - produced_torque[2],
        ]);
        RotorCommands { thrusts, shortfall }
    }

    /// The bounds `(lower, upper)` on a collective `c` that keep each `c·dᵢ + torque_shareᵢ` inside
    /// the rotor limits, with `d = A⁺·e₁`. Empty when `lower > upper`.
    fn collective_interval(self, torque_share: Vector<ROTOR_COUNT, T>) -> (T, T) {
        let mut lower = T::NEG_INFINITY;
        let mut upper = T::INFINITY;
        for rotor in 0..ROTOR_COUNT {
            let collective_share = self.distribution[(rotor, 0)];
            let share = torque_share[rotor];
            if collective_share == T::ZERO {
                if share < self.minimum_thrust || share > self.maximum_thrust {
                    return (T::INFINITY, T::NEG_INFINITY);
                }
                continue;
            }
            let (lower_limit, upper_limit) = if collective_share > T::ZERO {
                (self.minimum_thrust, self.maximum_thrust)
            } else {
                (self.maximum_thrust, self.minimum_thrust)
            };
            lower = lower.max((lower_limit - share) / collective_share);
            upper = upper.min((upper_limit - share) / collective_share);
        }
        (lower, upper)
    }

    /// The largest `s ∈ [0, 1]` with a collective `c` that keeps `c·d + base + s·extra` inside the
    /// rotor limits, for a `base` that fits.
    ///
    /// Fourier–Motzkin on `c`: rotor `i` gives `c ≥ αᵢ + s·βᵢ` and `c ≤ γᵢ + s·βᵢ`, with
    /// `βᵢ = −extraᵢ/dᵢ`, and each pair needs `αᵢ + s·βᵢ ≤ γⱼ + s·βⱼ`.
    fn largest_feasible_scale(
        self,
        base: Vector<ROTOR_COUNT, T>,
        extra: Vector<ROTOR_COUNT, T>,
    ) -> T {
        let mut scale = T::ONE;
        for lower_rotor in 0..ROTOR_COUNT {
            let lower_share = self.distribution[(lower_rotor, 0)];
            if lower_share == T::ZERO {
                // `minimum_thrust ≤ base + s·extra ≤ maximum_thrust` bounds `s` directly.
                if extra[lower_rotor] > T::ZERO {
                    scale =
                        scale.min((self.maximum_thrust - base[lower_rotor]) / extra[lower_rotor]);
                } else if extra[lower_rotor] < T::ZERO {
                    scale =
                        scale.min((self.minimum_thrust - base[lower_rotor]) / extra[lower_rotor]);
                }
                continue;
            }
            let lower_limit = if lower_share > T::ZERO {
                self.minimum_thrust
            } else {
                self.maximum_thrust
            };
            let lower_offset = (lower_limit - base[lower_rotor]) / lower_share;
            let lower_slope = -extra[lower_rotor] / lower_share;
            for upper_rotor in 0..ROTOR_COUNT {
                let upper_share = self.distribution[(upper_rotor, 0)];
                if upper_share == T::ZERO {
                    continue;
                }
                let upper_limit = if upper_share > T::ZERO {
                    self.maximum_thrust
                } else {
                    self.minimum_thrust
                };
                let upper_offset = (upper_limit - base[upper_rotor]) / upper_share;
                let upper_slope = -extra[upper_rotor] / upper_share;
                let slope = lower_slope - upper_slope;
                if slope > T::ZERO {
                    scale = scale.min((upper_offset - lower_offset) / slope);
                }
            }
        }
        scale.max(T::ZERO)
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

/// `value` held inside `[low, high]`. A NaN `value` comes back as NaN.
#[inline]
fn clamp_keeping_nan<T: Numeric>(value: T, low: T, high: T) -> T {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
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
