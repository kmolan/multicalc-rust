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
            thrusts: Vector::zeros(),
        })
    }

    /// Starts the rotors at thrusts they are already giving, rather than at nothing.
    ///
    /// For a machine that is already flying by the time the model is built.
    #[inline]
    #[must_use]
    pub fn with_thrusts(mut self, thrusts: Vector<ROTOR_COUNT, T>) -> Self {
        self.thrusts = thrusts;
        self
    }

    /// Moves every rotor one tick closer to what it was asked for, and says where they landed.
    ///
    /// The tick is the one the model was built with. Nothing expensive happens here — the two
    /// numbers this needs were worked out once, when the model was built.
    ///
    /// A command that is not finite comes back not finite rather than being rejected — this runs
    /// every tick, so checking is the caller's job, once, upstream.
    pub fn stepped(&mut self, commanded: Vector<ROTOR_COUNT, T>) -> Vector<ROTOR_COUNT, T> {
        let before = self.thrusts;
        self.thrusts = Vector::from_fn(|rotor| {
            self.carried_over * before[rotor] + self.caught_up * commanded[rotor]
        });
        self.thrusts
    }

    /// The same step, over a tick of some other length.
    ///
    /// For a loop whose ticks are not all the same length. This works out afresh what one tick
    /// closes, so it costs more than [`RotorLag::stepped`]; prefer that one on a loop running at a
    /// fixed rate.
    ///
    /// A tick length or command that is not finite comes back as thrusts that are not finite,
    /// rather than being rejected.
    pub fn stepped_over(
        &mut self,
        commanded: Vector<ROTOR_COUNT, T>,
        timestep: T,
    ) -> Vector<ROTOR_COUNT, T> {
        let ticks_of_lag = -timestep / self.time_constant;
        let carried_over = ticks_of_lag.exp();
        let caught_up = -ticks_of_lag.expm1();

        let before = self.thrusts;
        self.thrusts =
            Vector::from_fn(|rotor| carried_over * before[rotor] + caught_up * commanded[rotor]);
        self.thrusts
    }

    /// How fast each rotor's thrust is changing right now, given what it is being asked for.
    ///
    /// For a caller that would rather carry the rotor thrusts in its own state and hand the whole
    /// thing to an integrator than step the rotors on their own.
    pub fn rate(&self, commanded: Vector<ROTOR_COUNT, T>) -> Vector<ROTOR_COUNT, T> {
        Vector::from_fn(|rotor| (commanded[rotor] - self.thrusts[rotor]) / self.time_constant)
    }

    /// What each rotor is giving now.
    #[inline]
    pub fn thrusts(&self) -> Vector<ROTOR_COUNT, T> {
        self.thrusts
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
    #[inline]
    pub fn reset(&mut self) {
        self.thrusts = Vector::zeros();
    }
}
