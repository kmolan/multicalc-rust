//! How quickly a rotor catches up to the thrust it was asked for.

use crate::error::PlantError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// Holds what each rotor is actually giving, and moves it toward what it was asked for.
///
/// A rotor cannot change its thrust the moment it is asked to — it has to spin up or slow down
/// first. Ask for more and it closes the gap quickly at first and then more slowly, never quite
/// arriving but getting close enough to make no difference. The lag time is how long it takes to
/// close a little under two thirds of the gap, and the gap shrinks by that same fraction again over
/// every lag time after that.
///
/// The tick length is fixed when the model is built, so the two numbers a tick needs are worked out
/// once and each tick is a couple of multiplies per rotor with nothing expensive in it. Where the
/// thrust lands is worked out exactly rather than stepped toward, so a long tick is as safe as a
/// short one: the thrust can never overshoot what was asked for, or swing about it. That holds as
/// long as the command stays still across the tick, which is what a loop running at a fixed rate
/// does anyway; a command that moves part way through a tick is not followed exactly.
///
/// Thrusts come out in the order the rotors went in, so
/// [`MultirotorMixer::rotor_thrusts`](crate::plant::MultirotorMixer::rotor_thrusts) feeds this and
/// this feeds [`MultirotorMixer::wrench`](crate::plant::MultirotorMixer::wrench), with nothing
/// needed in between.
///
/// Thrust limits are optional and a model starts without any, so it follows its command exactly as
/// it always has. [`RotorLag::try_with_thrust_limits`] gives the lag the same limits the mixer was
/// built with, and every step then holds the rotors inside them — a command that never went
/// through the mixer cannot leave the lag giving a thrust the rotor could not physically produce.
/// [`RotorLag::with_thrusts`] holds an already-spinning rotor inside them too, while
/// [`RotorLag::reset`] returns to the disarmed all-zero state, limits or not.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::plant::RotorLag;
/// # fn main() -> Result<(), multicalc::error::PlantError> {
/// // Four rotors that take 20 ms to catch up, driven by a loop running every millisecond.
/// let lag_time = 0.02_f64;
/// let tick = 0.001;
/// let mut rotors = RotorLag::<4, f64>::new(lag_time, tick)?;
///
/// // From a standstill, asked for 2 N each.
/// let wanted = 2.0;
/// let asked_for = Vector::new([wanted, wanted, wanted, wanted]);
/// assert_eq!(rotors.thrusts()[0], 0.0);
///
/// // One lag time in, a little under two thirds of the gap is closed.
/// let ticks_in_one_lag_time = (lag_time / tick) as usize;
/// for _ in 0..ticks_in_one_lag_time {
///     let _ = rotors.stepped(asked_for);
/// }
/// let closed_fraction = rotors.thrusts()[0] / wanted;
/// let closed_in_one_lag_time = 1.0 - (-1.0_f64).exp();
/// assert!((closed_fraction - closed_in_one_lag_time).abs() < 1e-12);
///
/// // Held there, they settle on exactly what was asked for.
/// let long_enough_to_settle = 2000;
/// for _ in 0..long_enough_to_settle {
///     let _ = rotors.stepped(asked_for);
/// }
/// for rotor in 0..4 {
///     assert!((rotors.thrusts()[rotor] - wanted).abs() < 1e-12);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotorLag<const ROTOR_COUNT: usize, T: Numeric = f64> {
    time_constant: T,
    timestep: T,
    carried_over: T,
    caught_up: T,
    minimum_thrust: T,
    maximum_thrust: T,
    thrusts: Vector<ROTOR_COUNT, T>,
}

impl<const ROTOR_COUNT: usize, T: Numeric> RotorLag<ROTOR_COUNT, T> {
    /// Builds a model from how long a rotor takes to catch up and how long one tick lasts.
    ///
    /// `time_constant` is how long the rotor takes to close a little under two thirds of the gap to
    /// what it was asked for. `timestep` is how long one tick of the loop lasts. Every rotor starts
    /// out giving nothing; [`RotorLag::with_thrusts`] starts them somewhere else.
    ///
    /// The share of the gap a tick closes is worked out here with `expm1` rather than by taking the
    /// leftover share away from one: for a tick much shorter than the lag time that leftover share
    /// sits just under one, and the subtraction would throw away the digits that matter.
    ///
    /// Returns [`PlantError::NonFinite`] if either value is not finite,
    /// [`PlantError::NonPositiveTimeConstant`] if the lag time is zero or negative, or
    /// [`PlantError::NonPositiveTimestep`] if the tick length is zero or negative.
    pub fn new(time_constant: T, timestep: T) -> Result<Self, PlantError> {
        if !time_constant.is_finite() || !timestep.is_finite() {
            return Err(PlantError::NonFinite);
        }
        if time_constant <= T::ZERO {
            return Err(PlantError::NonPositiveTimeConstant);
        }
        if timestep <= T::ZERO {
            return Err(PlantError::NonPositiveTimestep);
        }

        let ticks_of_lag = -timestep / time_constant;
        Ok(RotorLag {
            time_constant,
            timestep,
            carried_over: ticks_of_lag.exp(),
            caught_up: -ticks_of_lag.expm1(),
            minimum_thrust: T::NEG_INFINITY,
            maximum_thrust: T::INFINITY,
            thrusts: Vector::zeros(),
        })
    }

    /// Starts the rotors at thrusts they are already giving, rather than at nothing.
    ///
    /// For a machine that is already flying by the time the model is built. A rotor cannot be
    /// already giving a thrust it could not produce, so the thrusts are held inside the model's
    /// limits when it already has any.
    #[inline]
    #[must_use]
    pub fn with_thrusts(mut self, thrusts: Vector<ROTOR_COUNT, T>) -> Self {
        self.thrusts = self.held_inside_limits(thrusts);
        self
    }

    /// Holds the rotors inside the thrusts a real rotor could give.
    ///
    /// The mixer already holds its own answer inside its limits, so this is for a caller that
    /// steps the lag directly: given the same limits here, a command that never went through the
    /// mixer cannot leave the lag giving a thrust the rotor could not produce. Both
    /// [`RotorLag::stepped`] and [`RotorLag::stepped_over`] hold the state inside the limits from
    /// then on, and [`RotorLag::with_thrusts`] holds it on the way in.
    ///
    /// # Errors
    /// Returns [`PlantError::NonFinite`] if either limit is not finite, or
    /// [`PlantError::InvalidThrustLimits`] if `maximum_thrust` is at or below `minimum_thrust`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::plant::RotorLag;
    /// # fn main() -> Result<(), multicalc::error::PlantError> {
    /// // Four rotors that take 20 ms to catch up, giving between nothing and 5 N.
    /// let mut rotors = RotorLag::<4, f64>::new(0.02, 0.001)?.try_with_thrust_limits(0.0, 5.0)?;
    ///
    /// // Asked for far more than they have, they settle on the most they can give.
    /// let beyond_reach = 30.0;
    /// for _ in 0..2000 {
    ///     let _ = rotors.stepped(Vector::new([beyond_reach; 4]));
    /// }
    /// assert_eq!(rotors.thrusts()[0], 5.0);
    ///
    /// // Limits the wrong way round are refused.
    /// assert!(RotorLag::<4, f64>::new(0.02, 0.001)?
    ///     .try_with_thrust_limits(5.0, 0.0)
    ///     .is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_with_thrust_limits(
        mut self,
        minimum_thrust: T,
        maximum_thrust: T,
    ) -> Result<Self, PlantError> {
        if !minimum_thrust.is_finite() || !maximum_thrust.is_finite() {
            return Err(PlantError::NonFinite);
        }
        if maximum_thrust <= minimum_thrust {
            return Err(PlantError::InvalidThrustLimits);
        }
        self.minimum_thrust = minimum_thrust;
        self.maximum_thrust = maximum_thrust;
        Ok(self)
    }

    /// Moves every rotor one tick closer to what it was asked for, and says where they landed.
    ///
    /// The tick is the one the model was built with. Nothing expensive happens here — the two
    /// numbers this needs were worked out once, when the model was built. What each rotor lands on
    /// is held inside the model's thrust limits, if it has any.
    ///
    /// A command that is not finite comes back not finite rather than being rejected — this runs
    /// every tick, so checking is the caller's job, once, upstream.
    pub fn stepped(&mut self, commanded: Vector<ROTOR_COUNT, T>) -> Vector<ROTOR_COUNT, T> {
        let before = self.thrusts;
        self.thrusts = self.held_inside_limits(Vector::from_fn(|rotor| {
            self.carried_over * before[rotor] + self.caught_up * commanded[rotor]
        }));
        self.thrusts
    }

    /// The same step, over a tick of some other length — rejecting a timestep that is not finite
    /// or not strictly positive.
    ///
    /// [`RotorLag::stepped_over`]'s "checking is the caller's job, once, upstream" reasoning holds
    /// for a fixed-rate loop, whose tick length was already validated in [`RotorLag::new`]. It does
    /// not hold here: this step's whole point is a tick length the constructor never saw, such as
    /// one computed from a pair of timestamps that a variable-rate loop can get the wrong way
    /// round. A negative tick from that composes into runaway growth instead of a small backward
    /// step, since `(-timestep / time_constant).exp()` then exceeds one.
    ///
    /// Returns [`PlantError::NonFinite`] if `timestep` is not finite, or
    /// [`PlantError::NonPositiveTimestep`] if it is zero or negative. `self` is left unchanged on
    /// either error. The command is not checked, same as [`RotorLag::stepped_over`].
    pub fn try_stepped_over(
        &mut self,
        commanded: Vector<ROTOR_COUNT, T>,
        timestep: T,
    ) -> Result<Vector<ROTOR_COUNT, T>, PlantError> {
        if !timestep.is_finite() {
            return Err(PlantError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(PlantError::NonPositiveTimestep);
        }
        Ok(self.stepped_over(commanded, timestep))
    }

    /// The same step, over a tick of some other length.
    ///
    /// For a loop whose ticks are not all the same length. This works out afresh what one tick
    /// closes, so it costs more than [`RotorLag::stepped`]; prefer that one on a loop running at a
    /// fixed rate. What each rotor lands on is held inside the model's thrust limits, if it has
    /// any, exactly as in [`RotorLag::stepped`].
    ///
    /// A tick length or command that is not finite comes back as thrusts that are not finite,
    /// rather than being rejected. Prefer [`RotorLag::try_stepped_over`] when the tick length has
    /// not already been validated.
    pub fn stepped_over(
        &mut self,
        commanded: Vector<ROTOR_COUNT, T>,
        timestep: T,
    ) -> Vector<ROTOR_COUNT, T> {
        let ticks_of_lag = -timestep / self.time_constant;
        let carried_over = ticks_of_lag.exp();
        let caught_up = -ticks_of_lag.expm1();

        let before = self.thrusts;
        self.thrusts = self.held_inside_limits(Vector::from_fn(|rotor| {
            carried_over * before[rotor] + caught_up * commanded[rotor]
        }));
        self.thrusts
    }

    /// How fast each rotor's thrust is changing right now, given what it is being asked for.
    ///
    /// For a caller that would rather carry the rotor thrusts in its own state and hand the whole
    /// thing to an integrator than step the rotors on their own. This is a rate rather than the
    /// state itself, so it is not held inside the model's thrust limits.
    pub fn rate(&self, commanded: Vector<ROTOR_COUNT, T>) -> Vector<ROTOR_COUNT, T> {
        Vector::from_fn(|rotor| (commanded[rotor] - self.thrusts[rotor]) / self.time_constant)
    }

    /// What each rotor is giving now.
    #[inline]
    pub fn thrusts(&self) -> Vector<ROTOR_COUNT, T> {
        self.thrusts
    }

    /// The least one rotor can give, or negative infinity when the model is unbounded.
    #[inline]
    #[must_use]
    pub fn minimum_thrust(&self) -> T {
        self.minimum_thrust
    }

    /// The most one rotor can give, or infinity when the model is unbounded.
    #[inline]
    #[must_use]
    pub fn maximum_thrust(&self) -> T {
        self.maximum_thrust
    }

    /// How long a rotor takes to close a little under two thirds of the gap to what it was asked
    /// for.
    #[inline]
    #[must_use]
    pub fn time_constant(&self) -> T {
        self.time_constant
    }

    /// How long one tick of the loop lasts.
    #[inline]
    #[must_use]
    pub fn timestep(&self) -> T {
        self.timestep
    }

    /// Puts every rotor back to giving nothing.
    ///
    /// The all-zero state is the disarmed one, so it is not held inside any thrust limits: a model
    /// with a minimum above zero still comes back to nothing here, and climbs back inside the
    /// limits on its next step. [`RotorLag::new`] starts in that same disarmed state.
    #[inline]
    pub fn reset(&mut self) {
        self.thrusts = Vector::zeros();
    }

    /// Holds each rotor inside the limits, leaving a value that is not a number alone — the
    /// floating-point `min`/`max` would take the other operand instead, changing what an unchecked
    /// non-finite command comes back as.
    #[inline]
    fn held_inside_limits(&self, thrusts: Vector<ROTOR_COUNT, T>) -> Vector<ROTOR_COUNT, T> {
        Vector::from_fn(|rotor| {
            let thrust = thrusts[rotor];
            if thrust < self.minimum_thrust {
                self.minimum_thrust
            } else if thrust > self.maximum_thrust {
                self.maximum_thrust
            } else {
                thrust
            }
        })
    }
}
