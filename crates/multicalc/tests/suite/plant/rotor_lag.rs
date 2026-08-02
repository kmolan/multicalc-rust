//! Rotor lag tests: settling on a steady command, matching the closed form tick by tick, the
//! point where two thirds of the gap is closed, a tick far longer than the lag time, the
//! variable-tick step agreeing with the fixed one, rotors not talking to each other, and the
//! values that are refused.

use multicalc::error::PlantError;
use multicalc::linear_algebra::Vector;
use multicalc::plant::RotorLag;
use multicalc::scalar::Dual;

const LAG_TIME: f64 = 0.02;
const TICK: f64 = 0.001;
const COMMAND: f64 = 5.0;

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
