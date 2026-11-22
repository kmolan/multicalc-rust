//! Finding a robot on a map it already has.

use multicalc::error::EstimationError;
use multicalc::estimation::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};
use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap, ScanGeometry};

const NUM_BEAMS: usize = 16;
const SEED: u64 = 20260804;

/// A 3.5 m by 2.5 m room with a pillar, which is what stops the room reading the same from two
/// different places.
fn room() -> DynamicOccupancyGrid {
    let mut grid = DynamicOccupancyGrid::try_new(80, 60, 0.05_f64, [-0.2, -0.2]).unwrap();
    let walls = [[0.0, 0.0], [3.5, 0.0], [3.5, 2.5], [0.0, 2.5]];
    grid.occupy_polyline(&walls, true);
    grid.occupy_circle([2.6, 1.9], 0.25);
    grid
}

fn scan_geometry() -> ScanGeometry<NUM_BEAMS, f64> {
    ScanGeometry::try_new(2.0 * core::f64::consts::PI / 3.0, 6.0).unwrap()
}

/// The scan a perfect sensor standing at `pose` would report.
fn perfect_scan(
    map: &DynamicOccupancyGrid,
    geometry: &ScanGeometry<NUM_BEAMS, f64>,
    pose: [f64; 3],
) -> [f64; NUM_BEAMS] {
    core::array::from_fn(|beam| {
        let offset = geometry.beam_angle(beam).unwrap();
        map.cast_ray(
            [pose[0], pose[1]],
            pose[2] + offset,
            geometry.maximum_range(),
        )
        .unwrap_or(f64::INFINITY)
    })
}

fn localizer(hint: [f64; 3], particle_count: usize) -> MonteCarloLocalizer<NUM_BEAMS> {
    let cloud = InitialParticleCloud {
        particle_count,
        position_variance: 0.16,
        heading_variance: 0.25,
    };
    let beam_model = BeamModel {
        range_deviation: 0.15,
        ..Default::default()
    };
    MonteCarloLocalizer::new(hint, cloud, beam_model, SEED).unwrap()
}

#[test]
fn a_standing_robot_is_found_from_its_scan() {
    let map = room();
    let geometry = scan_geometry();
    let truth = [1.2, 1.0, 0.3];
    let reading = perfect_scan(&map, &geometry, truth);
    let hint = [1.45, 0.75, 0.35];
    let mut found = localizer(hint, 1500);
    for _ in 0..10 {
        found.update(&reading, &map, &geometry).unwrap();
        found.predict(0.0, 0.0).unwrap();
    }
    let (pose, _) = found.estimate();
    assert!((pose[0] - truth[0]).abs() < 0.25, "x {pose:?}");
    assert!((pose[1] - truth[1]).abs() < 0.25, "y {pose:?}");
    assert!((pose[2] - truth[2]).abs() < 0.25, "heading {pose:?}");
}

#[test]
fn the_cloud_tightens_as_readings_arrive() {
    let map = room();
    let geometry = scan_geometry();
    let truth = [1.2, 1.0, 0.3];
    let reading = perfect_scan(&map, &geometry, truth);
    let mut found = localizer([1.45, 0.75, 0.35], 1500);
    assert!(!found.is_converged(0.02, 0.02), "it starts out spread wide");
    let (_, before) = found.estimate();
    for _ in 0..10 {
        found.update(&reading, &map, &geometry).unwrap();
    }
    let (_, after) = found.estimate();
    assert!(
        after[(0, 0)] < before[(0, 0)],
        "the position spread shrinks"
    );
    assert!(found.is_converged(0.05, 0.2), "{after:?}");
}

#[test]
fn travel_carries_the_estimate_with_it() {
    let map = room();
    let geometry = scan_geometry();
    let truth = [1.2, 1.0, 0.0];
    let reading = perfect_scan(&map, &geometry, truth);
    let mut found = localizer([1.25, 1.05, 0.0], 1500);
    for _ in 0..8 {
        found.update(&reading, &map, &geometry).unwrap();
    }
    let (before, _) = found.estimate();
    let travelled = 0.5;
    found.predict(travelled, 0.0).unwrap();
    let (after, _) = found.estimate();
    // Facing along x, so the whole step lands in x.
    assert!(
        (after[0] - before[0] - travelled).abs() < 0.05,
        "{before:?} {after:?}"
    );
    assert!((after[1] - before[1]).abs() < 0.05);
}

#[test]
fn turning_carries_the_heading_with_it() {
    let mut found = localizer([1.2, 1.0, 0.0], 500);
    let (before, _) = found.estimate();
    let turned = 0.4;
    found.predict(0.0, turned).unwrap();
    let (after, _) = found.estimate();
    assert!((after[2] - before[2] - turned).abs() < 0.05);
}

#[test]
fn the_cloud_keeps_the_size_it_was_asked_for() {
    let found = localizer([1.2, 1.0, 0.0], 250);
    assert_eq!(found.particle_count(), 250);
    assert_eq!(found.particles().len(), 250);
    // A fresh cloud carries its weight evenly, so every guess is still pulling.
    let sample_size = found.effective_sample_size();
    assert!((sample_size - 250.0).abs() < 1e-6, "{sample_size}");
}

#[test]
fn a_cloud_of_no_guesses_is_rejected() {
    let cloud = InitialParticleCloud {
        particle_count: 0,
        ..Default::default()
    };
    assert_eq!(
        MonteCarloLocalizer::<NUM_BEAMS>::new([0.0, 0.0, 0.0], cloud, BeamModel::default(), SEED)
            .unwrap_err(),
        EstimationError::WeightsDegenerate
    );
}

#[test]
fn a_spread_that_is_not_a_spread_is_rejected() {
    let cloud = InitialParticleCloud {
        particle_count: 100,
        position_variance: -1.0,
        heading_variance: 1.0,
    };
    assert!(
        MonteCarloLocalizer::<NUM_BEAMS>::new([0.0, 0.0, 0.0], cloud, BeamModel::default(), SEED)
            .is_err()
    );
}

#[test]
fn motion_noise_can_be_changed_and_a_bad_one_refused() {
    let mut found = localizer([1.2, 1.0, 0.0], 100);
    assert!(found.set_motion_noise([1e-3, 1e-3, 1e-3]).is_ok());
    assert!(found.set_motion_noise([-1.0, 1e-3, 1e-3]).is_err());
}

/// A scan of nothing but no-returns is scored, not panicked on.
#[test]
fn a_scan_that_sees_nothing_is_handled() {
    let map = room();
    let geometry = scan_geometry();
    let mut found = localizer([1.2, 1.0, 0.0], 200);
    let nothing = [f64::INFINITY; NUM_BEAMS];
    let outcome = found.update(&nothing, &map, &geometry);
    assert!(
        outcome.is_ok() || outcome == Err(EstimationError::WeightsDegenerate),
        "{outcome:?}"
    );
}

/// A reading inside the sensor's blind spot is not a distance, so it scores as a no-return.
#[test]
fn a_reading_inside_the_blind_spot_is_not_treated_as_a_distance() {
    let map = room();
    let geometry = scan_geometry().with_minimum_range(0.3).unwrap();
    let mut found = localizer([1.2, 1.0, 0.0], 200);
    let too_close = [0.05; NUM_BEAMS];
    assert!(found.update(&too_close, &map, &geometry).is_ok());
}
