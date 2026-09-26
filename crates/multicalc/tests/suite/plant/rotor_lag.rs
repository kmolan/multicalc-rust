//! Rotor lag tests: settling on a steady command, matching the closed form tick by tick, the
//! point where two thirds of the gap is closed, a tick far longer than the lag time, the
//! variable-tick step agreeing with the fixed one, rotors not talking to each other, the checked
//! variable-tick step refusing a bad timestep, the values that are refused, the thrust limits
//! holding a command above or below them, and the lag and the mixer agreeing on the wrench once
//! a limited command settles.

use multicalc::error::PlantError;
use multicalc::linear_algebra::Vector;
use multicalc::plant::{MultirotorMixer, RotorLag};
use multicalc::scalar::Dual;

const LAG_TIME: f64 = 0.02;
const TICK: f64 = 0.001;
const COMMAND: f64 = 5.0;
const MINIMUM_THRUST: f64 = 0.0;
const MAXIMUM_THRUST: f64 = 5.0;
const ARM_LENGTH: f64 = 0.15;
const TORQUE_PER_THRUST: f64 = 0.016;

/// The four rotors every test below shares.
fn rotors() -> RotorLag<4, f64> {
    RotorLag::<4, f64>::new(LAG_TIME, TICK).unwrap()
}

fn all_four(thrust: f64) -> Vector<4, f64> {
    Vector::new([thrust, thrust, thrust, thrust])
}

/// How many ticks make up one lag time.
fn ticks_in_one_lag_time() -> usize {
    (LAG_TIME / TICK) as usize
}

/// What a rotor starting from a standstill is giving after `elapsed` seconds.
fn closed_form(command: f64, elapsed: f64) -> f64 {
    command * (1.0 - (-elapsed / LAG_TIME).exp())
}

#[test]
fn a_steady_command_is_settled_on() {
    let mut rotors = rotors();

    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let _ = rotors.stepped(all_four(COMMAND));
    }

    for rotor in 0..4 {
        assert!((rotors.thrusts()[rotor] - COMMAND).abs() < 1e-12);
        assert!(
            rotors.thrusts()[rotor] <= COMMAND,
            "a rotor must never push past what it was asked for"
        );
    }
}

#[test]
fn every_tick_matches_the_closed_form() {
    let mut rotors = rotors();

    let checkpoints = [1, 5, 20, 100, 400];
    let mut ticks_taken = 0;
    for checkpoint in checkpoints {
        while ticks_taken < checkpoint {
            let _ = rotors.stepped(all_four(COMMAND));
            ticks_taken += 1;
        }
        let elapsed = ticks_taken as f64 * TICK;
        assert!((rotors.thrusts()[0] - closed_form(COMMAND, elapsed)).abs() < 1e-12);
    }
}

#[test]
fn one_lag_time_closes_two_thirds_of_the_gap() {
    let mut rotors = rotors();

    for _ in 0..ticks_in_one_lag_time() {
        let _ = rotors.stepped(all_four(COMMAND));
    }

    let closed_fraction = rotors.thrusts()[0] / COMMAND;
    let closed_in_one_lag_time = 1.0 - (-1.0_f64).exp();
    assert!((closed_fraction - closed_in_one_lag_time).abs() < 1e-12);
}

#[test]
fn a_tick_far_longer_than_the_lag_time_still_lands_on_the_command() {
    let very_long_tick = 1.0;
    let mut rotors = RotorLag::<4, f64>::new(LAG_TIME, very_long_tick).unwrap();

    let landed = rotors.stepped(all_four(COMMAND));

    // Stepping toward the command rather than landing on it exactly would overshoot to
    // COMMAND * very_long_tick / LAG_TIME here, fifty times too far.
    for rotor in 0..4 {
        assert!((landed[rotor] - COMMAND).abs() < 1e-12);
        assert!(landed[rotor] <= COMMAND);
    }
}

#[test]
fn the_variable_tick_step_agrees_with_the_fixed_one() {
    // The same tick length, taken both ways, lands in the same place.
    let mut fixed = rotors();
    let mut variable = rotors();
    let by_the_fixed_tick = fixed.stepped(all_four(COMMAND));
    let by_a_stated_tick = variable.stepped_over(all_four(COMMAND), TICK);
    for rotor in 0..4 {
        assert!((by_the_fixed_tick[rotor] - by_a_stated_tick[rotor]).abs() < 1e-15);
    }

    // Splitting one tick into two halves lands in the same place too.
    let mut in_halves = rotors();
    let half_tick = TICK / 2.0;
    let _ = in_halves.stepped_over(all_four(COMMAND), half_tick);
    let after_both_halves = in_halves.stepped_over(all_four(COMMAND), half_tick);
    for rotor in 0..4 {
        assert!((after_both_halves[rotor] - by_the_fixed_tick[rotor]).abs() < 1e-14);
    }
}

#[test]
fn the_checked_variable_tick_step_matches_the_fixed_one_at_the_total_time() {
    // Three differing positive ticks, taken with the checked step, land in the same place as one
    // fixed-rate model whose tick is the total elapsed time.
    let ticks = [0.0006, 0.0011, 0.0003];
    let total: f64 = ticks.iter().sum();

    let mut variable = rotors();
    let mut landed = all_four(0.0);
    for &tick in &ticks {
        landed = variable.try_stepped_over(all_four(COMMAND), tick).unwrap();
    }

    let mut fixed = RotorLag::<4, f64>::new(LAG_TIME, total).unwrap();
    let want = fixed.stepped(all_four(COMMAND));

    for rotor in 0..4 {
        assert!((landed[rotor] - want[rotor]).abs() < 1e-12);
    }
}

#[test]
fn the_checked_step_rejects_a_non_positive_timestep() {
    let mut rotors = rotors();
    assert_eq!(
        rotors.try_stepped_over(all_four(COMMAND), 0.0).err(),
        Some(PlantError::NonPositiveTimestep)
    );
    assert_eq!(
        rotors.try_stepped_over(all_four(COMMAND), -TICK).err(),
        Some(PlantError::NonPositiveTimestep)
    );
}

#[test]
fn the_checked_step_rejects_a_non_finite_timestep() {
    let mut rotors = rotors();
    assert_eq!(
        rotors.try_stepped_over(all_four(COMMAND), f64::NAN).err(),
        Some(PlantError::NonFinite)
    );
    assert_eq!(
        rotors
            .try_stepped_over(all_four(COMMAND), f64::INFINITY)
            .err(),
        Some(PlantError::NonFinite)
    );
}

#[test]
fn the_checked_step_leaves_the_rotors_untouched_on_rejection() {
    let mut rotors = rotors();
    let _ = rotors.stepped(all_four(COMMAND));
    let before = rotors.thrusts();

    assert!(rotors.try_stepped_over(all_four(COMMAND), -TICK).is_err());
    assert!(
        rotors
            .try_stepped_over(all_four(COMMAND), f64::NAN)
            .is_err()
    );

    assert_eq!(rotors.thrusts(), before);
}

#[test]
fn the_rate_is_the_gap_divided_by_the_lag_time() {
    let mut rotors = rotors();

    // From a standstill the whole command is still to be made up.
    let from_rest = rotors.rate(all_four(COMMAND));
    for rotor in 0..4 {
        assert!((from_rest[rotor] - COMMAND / LAG_TIME).abs() < 1e-12);
    }

    // One tick in, the gap is smaller and so is the rate.
    let _ = rotors.stepped(all_four(COMMAND));
    let after_a_tick = rotors.rate(all_four(COMMAND));
    let gap_left = COMMAND - rotors.thrusts()[0];
    for rotor in 0..4 {
        assert!((after_a_tick[rotor] - gap_left / LAG_TIME).abs() < 1e-12);
    }
}

#[test]
fn spooling_down_mirrors_spooling_up() {
    let mut rotors = rotors().with_thrusts(all_four(COMMAND));

    let checkpoints = [20, 100];
    let mut ticks_taken = 0;
    for checkpoint in checkpoints {
        while ticks_taken < checkpoint {
            let _ = rotors.stepped(all_four(0.0));
            ticks_taken += 1;
        }
        let elapsed = ticks_taken as f64 * TICK;
        let still_giving = COMMAND * (-elapsed / LAG_TIME).exp();
        assert!((rotors.thrusts()[0] - still_giving).abs() < 1e-12);
    }
}

#[test]
fn each_rotor_follows_its_own_command() {
    let mut rotors = rotors();

    let asked_for = Vector::new([1.0, 2.0, 3.0, 4.0]);
    for _ in 0..ticks_in_one_lag_time() {
        let _ = rotors.stepped(asked_for);
    }

    for rotor in 0..4 {
        let on_its_own = closed_form(asked_for[rotor], LAG_TIME);
        assert!((rotors.thrusts()[rotor] - on_its_own).abs() < 1e-12);
    }
}

#[test]
fn resetting_puts_every_rotor_back_to_nothing() {
    let mut rotors = rotors();

    let enough_to_move = 50;
    for _ in 0..enough_to_move {
        let _ = rotors.stepped(all_four(COMMAND));
    }
    assert!(rotors.thrusts()[0] > 0.0);

    rotors.reset();
    for rotor in 0..4 {
        assert_eq!(rotors.thrusts()[rotor], 0.0);
    }
}

#[test]
fn a_command_that_is_not_finite_comes_back_not_finite() {
    let mut rotors = rotors();

    let landed = rotors.stepped(Vector::new([f64::NAN, 0.0, 0.0, 0.0]));

    assert!(landed[0].is_nan());
    assert!(rotors.thrusts()[0].is_nan());
}

#[test]
fn values_that_are_refused() {
    assert_eq!(
        RotorLag::<4, f64>::new(f64::NAN, TICK),
        Err(PlantError::NonFinite)
    );
    assert_eq!(
        RotorLag::<4, f64>::new(LAG_TIME, f64::INFINITY),
        Err(PlantError::NonFinite)
    );

    let no_lag_at_all = 0.0;
    assert_eq!(
        RotorLag::<4, f64>::new(no_lag_at_all, TICK),
        Err(PlantError::NonPositiveTimeConstant)
    );
    assert_eq!(
        RotorLag::<4, f64>::new(-LAG_TIME, TICK),
        Err(PlantError::NonPositiveTimeConstant)
    );

    let no_tick_at_all = 0.0;
    assert_eq!(
        RotorLag::<4, f64>::new(LAG_TIME, no_tick_at_all),
        Err(PlantError::NonPositiveTimestep)
    );
    assert_eq!(
        RotorLag::<4, f64>::new(LAG_TIME, -TICK),
        Err(PlantError::NonPositiveTimestep)
    );
}

#[test]
fn single_precision_follows_the_same_curve() {
    let lag_time = LAG_TIME as f32;
    let tick = TICK as f32;
    let command = COMMAND as f32;
    let mut rotors = RotorLag::<4, f32>::new(lag_time, tick).unwrap();

    for _ in 0..ticks_in_one_lag_time() {
        let _ = rotors.stepped(Vector::new([command, command, command, command]));
    }

    let closed_fraction = rotors.thrusts()[0] / command;
    let closed_in_one_lag_time = 1.0 - (-1.0_f32).exp();
    assert!((closed_fraction - closed_in_one_lag_time).abs() < 1e-6);
}

#[test]
fn the_derivative_of_one_tick_with_respect_to_the_command_is_exact() {
    let mut rotors =
        RotorLag::<4, Dual<f64>>::new(Dual::constant(LAG_TIME), Dual::constant(TICK)).unwrap();

    // Only the first rotor's command is the variable being differentiated against.
    let asked_for = Vector::new([
        Dual::variable(COMMAND),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let landed = rotors.stepped(asked_for);

    // One tick closes a fixed share of the gap, so that share is exactly how much of the command
    // comes through.
    let share_a_tick_closes = 1.0 - (-TICK / LAG_TIME).exp();
    assert!((landed[0].deriv - share_a_tick_closes).abs() < 1e-12);
    assert!((landed[0].value - closed_form(COMMAND, TICK)).abs() < 1e-12);
}

#[test]
fn a_new_model_is_unbounded() {
    let rotors = rotors();

    assert_eq!(rotors.minimum_thrust(), f64::NEG_INFINITY);
    assert_eq!(rotors.maximum_thrust(), f64::INFINITY);
}

#[test]
fn a_command_above_the_maximum_settles_on_the_maximum() {
    let mut rotors = rotors()
        .try_with_thrust_limits(MINIMUM_THRUST, MAXIMUM_THRUST)
        .unwrap();

    let beyond_reach = 4.0 * COMMAND;
    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let landed = rotors.stepped(all_four(beyond_reach));
        for rotor in 0..4 {
            assert!(
                landed[rotor] <= MAXIMUM_THRUST,
                "a rotor must never give more than its maximum"
            );
        }
    }

    for rotor in 0..4 {
        assert_eq!(rotors.thrusts()[rotor], MAXIMUM_THRUST);
    }

    // The rate is a derivative rather than the state, so it is not held inside the limits.
    let rate = rotors.rate(all_four(beyond_reach));
    let gap_to_the_command = (beyond_reach - MAXIMUM_THRUST) / LAG_TIME;
    assert!((rate[0] - gap_to_the_command).abs() < 1e-12);
}

#[test]
fn a_command_below_the_minimum_settles_on_the_minimum() {
    let minimum_thrust = 1.0;
    let mut rotors = rotors()
        .try_with_thrust_limits(minimum_thrust, MAXIMUM_THRUST)
        .unwrap();

    let below_reach = -COMMAND;
    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let landed = rotors.stepped(all_four(below_reach));
        for rotor in 0..4 {
            assert!(
                landed[rotor] >= minimum_thrust,
                "a rotor must never give less than its minimum"
            );
        }
    }

    for rotor in 0..4 {
        assert_eq!(rotors.thrusts()[rotor], minimum_thrust);
    }
}

#[test]
fn the_variable_tick_step_is_held_inside_the_limits_too() {
    let minimum_thrust = 1.0;
    let mut rotors = rotors()
        .try_with_thrust_limits(minimum_thrust, MAXIMUM_THRUST)
        .unwrap();

    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let landed = rotors.stepped_over(all_four(4.0 * COMMAND), TICK);
        for rotor in 0..4 {
            assert!(landed[rotor] <= MAXIMUM_THRUST);
        }
    }
    assert_eq!(rotors.thrusts()[0], MAXIMUM_THRUST);

    for _ in 0..long_enough_to_settle {
        let landed = rotors.stepped_over(all_four(-COMMAND), TICK);
        for rotor in 0..4 {
            assert!(landed[rotor] >= minimum_thrust);
        }
    }
    assert_eq!(rotors.thrusts()[0], minimum_thrust);
}

#[test]
fn limits_that_are_refused() {
    assert_eq!(
        rotors().try_with_thrust_limits(f64::NAN, MAXIMUM_THRUST),
        Err(PlantError::NonFinite)
    );
    assert_eq!(
        rotors().try_with_thrust_limits(MINIMUM_THRUST, f64::INFINITY),
        Err(PlantError::NonFinite)
    );

    let no_room_at_all = 5.0;
    assert_eq!(
        rotors().try_with_thrust_limits(no_room_at_all, no_room_at_all),
        Err(PlantError::InvalidThrustLimits)
    );
    assert_eq!(
        rotors().try_with_thrust_limits(MAXIMUM_THRUST, MINIMUM_THRUST),
        Err(PlantError::InvalidThrustLimits)
    );
}

#[test]
fn limits_far_outside_the_command_leave_the_lag_untouched() {
    let far_below = -100.0;
    let far_above = 100.0;
    let mut bounded = rotors()
        .try_with_thrust_limits(far_below, far_above)
        .unwrap();
    let mut unbounded = rotors();

    let checkpoints = [1, 20, 100, 400];
    let mut ticks_taken = 0;
    for checkpoint in checkpoints {
        while ticks_taken < checkpoint {
            let _ = bounded.stepped(all_four(COMMAND));
            let _ = unbounded.stepped(all_four(COMMAND));
            ticks_taken += 1;
        }
        let elapsed = ticks_taken as f64 * TICK;
        assert!((bounded.thrusts()[0] - closed_form(COMMAND, elapsed)).abs() < 1e-12);
        assert_eq!(bounded.thrusts(), unbounded.thrusts());
    }

    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let _ = bounded.stepped(all_four(COMMAND));
    }
    for rotor in 0..4 {
        assert!((bounded.thrusts()[rotor] - COMMAND).abs() < 1e-12);
    }
}

#[test]
fn thrusts_a_rotor_could_not_be_giving_are_held_inside_the_limits() {
    let minimum_thrust = 1.0;
    let already_spinning = Vector::new([4.0 * COMMAND, -COMMAND, 3.0, COMMAND]);
    let mut rotors = rotors()
        .try_with_thrust_limits(minimum_thrust, MAXIMUM_THRUST)
        .unwrap()
        .with_thrusts(already_spinning);

    assert_eq!(
        rotors.thrusts(),
        Vector::new([MAXIMUM_THRUST, minimum_thrust, 3.0, COMMAND])
    );

    // They stay inside on every later step, even spooling down toward nothing.
    for _ in 0..ticks_in_one_lag_time() {
        let landed = rotors.stepped(all_four(0.0));
        for rotor in 0..4 {
            assert!(landed[rotor] >= minimum_thrust && landed[rotor] <= MAXIMUM_THRUST);
        }
    }
}

#[test]
fn resetting_comes_back_to_zeros_even_under_a_minimum_above_zero() {
    let minimum_thrust = 1.0;
    let mut rotors = rotors()
        .try_with_thrust_limits(minimum_thrust, MAXIMUM_THRUST)
        .unwrap();

    let enough_to_move = 50;
    for _ in 0..enough_to_move {
        let _ = rotors.stepped(all_four(COMMAND));
    }
    assert!(rotors.thrusts()[0] > 0.0);

    rotors.reset();
    for rotor in 0..4 {
        assert_eq!(rotors.thrusts()[rotor], 0.0);
    }

    // The next step climbs back inside the limits.
    let after_a_tick = rotors.stepped(all_four(0.0));
    for rotor in 0..4 {
        assert!(after_a_tick[rotor] >= minimum_thrust);
    }
}

#[test]
fn a_limited_lag_and_the_mixer_agree_on_the_wrench() {
    let mixer = MultirotorMixer::<4, f64>::quadrotor_x(
        ARM_LENGTH,
        TORQUE_PER_THRUST,
        MINIMUM_THRUST,
        MAXIMUM_THRUST,
    )
    .unwrap();

    // Asked for far more push than the rotors have, so the mixer clamps every one of them.
    let beyond_reach = 30.0;
    let no_turn = Vector::new([0.0, 0.0, 0.0]);
    let commands = mixer.rotor_thrusts(beyond_reach, no_turn);
    assert!(commands.saturated());

    // Fed the mixer's clamped thrusts, the lag settles on them and both ways of asking what the
    // body feels give the same push and turn.
    let mut fed_the_clamped_thrusts = rotors()
        .try_with_thrust_limits(mixer.minimum_thrust(), mixer.maximum_thrust())
        .unwrap();
    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let _ = fed_the_clamped_thrusts.stepped(commands.thrusts());
    }
    let from_the_lag = mixer.wrench(fed_the_clamped_thrusts.thrusts());
    let from_the_commands = mixer.wrench(commands.thrusts());
    for axis in 0..3 {
        assert!((from_the_lag.force()[axis] - from_the_commands.force()[axis]).abs() < 1e-12);
        assert!((from_the_lag.torque()[axis] - from_the_commands.torque()[axis]).abs() < 1e-12);
    }

    // Fed the raw command the mixer would have clamped, the lag lands exactly on the limits
    // instead of running past them, so the wrench is the mixer's clamped one exactly.
    let mut fed_the_raw_command = rotors()
        .try_with_thrust_limits(mixer.minimum_thrust(), mixer.maximum_thrust())
        .unwrap();
    for _ in 0..long_enough_to_settle {
        let _ = fed_the_raw_command.stepped(all_four(beyond_reach));
    }
    assert_eq!(fed_the_raw_command.thrusts(), commands.thrusts());
    assert_eq!(
        mixer.wrench(fed_the_raw_command.thrusts()),
        from_the_commands
    );
}
