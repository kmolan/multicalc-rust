//! The belief map: that scans move belief the right way, that the clamps let a cell recover, and
//! that unobserved space stays distinguishable from free space.

use multicalc::error::MappingError;
use multicalc::mapping::{CellState, LogOddsGrid, OccupancyMap, ScanGeometry};
use multicalc::scalar::{Numeric, Primal};
use multicalc::{SE2, SO2, Vector2D};

/// A 4 m square at 10 cm cells, and a three-beam scan facing east from the middle of cell (20, 20).
///
/// The pose sits at a cell centre rather than a corner, and the fan is narrow enough that a metre
/// out all three beams are still in row 20 — so the cell arithmetic below is about belief rather
/// than about which cell a beam grazed.
fn room_and_scan<T: Numeric + Primal>() -> (LogOddsGrid<40, 40, T>, ScanGeometry<3, T>, SE2<T>) {
    let belief = LogOddsGrid::try_new(T::from_f64(0.1), [T::ZERO, T::ZERO]).unwrap();
    let scan = ScanGeometry::try_new(T::from_f64(0.02), T::from_f64(4.0)).unwrap();
    let pose = SE2::from_parts(
        SO2::from_angle(T::ZERO),
        Vector2D::new([T::from_f64(2.05), T::from_f64(2.05)]),
    );
    (belief, scan, pose)
}

fn assert_new_grid_is_entirely_unknown<T: Numeric + Primal>() {
    let (belief, _, _) = room_and_scan::<T>();
    for row in 0..40 {
        for column in 0..40 {
            assert_eq!(belief.belief_at(row, column), Some(0));
            assert_eq!(belief.cell_state(row, column), CellState::Unknown);
            assert!(!belief.is_occupied(row, column));
        }
    }
}

fn assert_repeated_hits_cross_the_occupied_threshold<T: Numeric + Primal>() {
    let (mut belief, scan, pose) = room_and_scan::<T>();
    let a_wall_one_metre_off = [T::ONE; 3];

    // Three beams land in the wall cell, so one scan moves it by 3·5 − 3·2 = +9, just short of
    // the +10 threshold.
    belief.integrate_scan(pose, &scan, &a_wall_one_metre_off);
    assert_eq!(belief.belief_at(20, 30), Some(9));
    assert_eq!(belief.cell_state(20, 30), CellState::Unknown);

    belief.integrate_scan(pose, &scan, &a_wall_one_metre_off);
    assert_eq!(belief.cell_state(20, 30), CellState::Occupied);
    assert!(belief.is_occupied(20, 30));
}

fn assert_repeated_misses_cross_the_free_threshold<T: Numeric + Primal>() {
    let (mut belief, scan, pose) = room_and_scan::<T>();
    let a_wall_one_metre_off = [T::ONE; 3];

    // A crossed cell moves by 3·(−2) = −6 a scan, so two scans clear the −10 threshold.
    belief.integrate_scan(pose, &scan, &a_wall_one_metre_off);
    assert_eq!(belief.cell_state(20, 25), CellState::Unknown);

    belief.integrate_scan(pose, &scan, &a_wall_one_metre_off);
    assert_eq!(belief.cell_state(20, 25), CellState::Free);
}

fn assert_belief_saturates_at_the_clamps<T: Numeric + Primal>() {
    let (mut belief, scan, pose) = room_and_scan::<T>();
    let a_wall_one_metre_off = [T::ONE; 3];

    for _ in 0..200 {
        belief.integrate_scan(pose, &scan, &a_wall_one_metre_off);
    }
    assert_eq!(belief.belief_at(20, 30), Some(40));
    assert_eq!(belief.belief_at(20, 25), Some(-40));
}

fn assert_a_transient_obstacle_clears<T: Numeric + Primal>() {
    let (mut belief, scan, pose) = room_and_scan::<T>();
    let with_the_obstacle = [T::ONE; 3];
    let without_it = [T::from_f64(3.0); 3];

    for _ in 0..20 {
        belief.integrate_scan(pose, &scan, &with_the_obstacle);
    }
    assert_eq!(belief.cell_state(20, 30), CellState::Occupied);

    // It walks off, and the beams now pass straight through where it stood.
    for _ in 0..20 {
        belief.integrate_scan(pose, &scan, &without_it);
    }
    assert_eq!(belief.cell_state(20, 30), CellState::Free);
}

#[test]
fn new_grid_is_entirely_unknown_f64() {
    assert_new_grid_is_entirely_unknown::<f64>();
}

#[test]
fn repeated_hits_cross_the_occupied_threshold_f64() {
    assert_repeated_hits_cross_the_occupied_threshold::<f64>();
}

#[test]
fn repeated_hits_cross_the_occupied_threshold_f32() {
    assert_repeated_hits_cross_the_occupied_threshold::<f32>();
}

#[test]
fn repeated_misses_cross_the_free_threshold_f64() {
    assert_repeated_misses_cross_the_free_threshold::<f64>();
}

#[test]
fn repeated_misses_cross_the_free_threshold_f32() {
    assert_repeated_misses_cross_the_free_threshold::<f32>();
}

#[test]
fn belief_saturates_at_the_clamps_f64() {
    assert_belief_saturates_at_the_clamps::<f64>();
}

#[test]
fn a_transient_obstacle_clears_f64() {
    assert_a_transient_obstacle_clears::<f64>();
}

#[test]
fn integrate_scan_marks_free_space_along_the_beam_f64() {
    let (mut belief, scan, pose) = room_and_scan::<f64>();
    belief.integrate_scan(pose, &scan, &[1.0; 3]);

    // Every cell between the robot and the wall moved toward free.
    for column in 20..30 {
        assert!(
            belief.belief_at(20, column).is_some_and(|value| value < 0),
            "column {column}"
        );
    }
    // The wall cell itself moved the other way: the hits outweigh the crossings.
    assert!(belief.belief_at(20, 30).is_some_and(|value| value > 0));
    // Beyond the wall nothing was seen at all.
    assert_eq!(belief.belief_at(20, 35), Some(0));
}

#[test]
fn a_max_range_reading_marks_free_only_f64() {
    let (mut belief, scan, pose) = room_and_scan::<f64>();
    let nothing_out_there = [scan.maximum_range(); 3];
    belief.integrate_scan(pose, &scan, &nothing_out_there);

    for column in 20..40 {
        assert!(
            belief.belief_at(20, column).is_some_and(|value| value <= 0),
            "column {column} was marked blocked by a reading that met nothing"
        );
    }
}

#[test]
fn try_with_thresholds_rejects_unordered_settings() {
    let belief: LogOddsGrid<8, 8> = LogOddsGrid::try_new(0.1, [0.0, 0.0]).unwrap();

    assert_eq!(
        belief.try_with_thresholds(10, -10).err(),
        Some(MappingError::InvalidBeliefSettings)
    );
    // Outside the clamps.
    assert_eq!(
        belief.try_with_thresholds(-50, 10).err(),
        Some(MappingError::InvalidBeliefSettings)
    );
    assert_eq!(
        belief.try_with_thresholds(-10, 50).err(),
        Some(MappingError::InvalidBeliefSettings)
    );
    assert!(belief.try_with_thresholds(-5, 5).is_ok());
}

#[test]
fn try_with_updates_and_clamps_reject_unordered_settings() {
    let belief: LogOddsGrid<8, 8> = LogOddsGrid::try_new(0.1, [0.0, 0.0]).unwrap();

    assert_eq!(
        belief.try_with_updates(2, 5).err(),
        Some(MappingError::InvalidBeliefSettings)
    );
    assert_eq!(
        belief.try_with_updates(-2, 0).err(),
        Some(MappingError::InvalidBeliefSettings)
    );
    assert_eq!(
        belief.try_with_clamps(10, 40).err(),
        Some(MappingError::InvalidBeliefSettings)
    );
    assert!(belief.try_with_updates(-1, 3).is_ok());
    assert!(belief.try_with_clamps(-20, 20).is_ok());
}

#[test]
fn cell_state_reports_unknown_off_grid_f64() {
    let (belief, _, _) = room_and_scan::<f64>();

    // Where a plain occupancy map reads free, a belief map reads unknown — the distinction that
    // stops a planner routing through unmapped space.
    assert_eq!(belief.cell_state(40, 0), CellState::Unknown);
    assert_eq!(belief.cell_state(0, 40), CellState::Unknown);
    assert_eq!(
        belief.cell_state(usize::MAX, usize::MAX),
        CellState::Unknown
    );
}

#[test]
fn reset_returns_every_cell_to_unknown_f64() {
    let (mut belief, scan, pose) = room_and_scan::<f64>();
    for _ in 0..10 {
        belief.integrate_scan(pose, &scan, &[1.0; 3]);
    }
    assert_eq!(belief.cell_state(20, 30), CellState::Occupied);

    belief.reset();
    for row in 0..40 {
        for column in 0..40 {
            assert_eq!(belief.cell_state(row, column), CellState::Unknown);
        }
    }
}
