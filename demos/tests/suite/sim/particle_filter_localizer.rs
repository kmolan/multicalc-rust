use multicalc::linear_algebra::Vector3D;
use multicalc_demos::sim::geometry::{box_outline, wrap_angle};
use multicalc_demos::sim::lidar::Lidar2d;
use multicalc_demos::sim::occupancy_grid::OccupancyGrid;
use multicalc_demos::sim::particle_filter_localizer::{GlobalLocalizer, InitialParticleCloud};
use rand::SeedableRng;
use rand_pcg::Pcg32;
use std::f64::consts::TAU;

/// A four-walled room with an off-centre block, so no two poses look alike from the inside.
#[must_use]
fn test_room() -> OccupancyGrid {
    let resolution = 0.05;
    let mut grid = OccupancyGrid::new(88, 68, resolution, [-0.2, -0.2]);
    grid.occupy_polyline(&box_outline([0.0, 0.0], [4.0, 3.0]), true);
    grid.occupy_polyline(&box_outline([2.6, 0.4], [3.2, 1.0]), true);
    grid
}

/// Runs the startup turn for `cycles` steps and returns the final estimate, how many guesses
/// still carry weight, and where the robot truly ended up facing.
fn localize(seed: u64, cycles: usize) -> (Vector3D, f64, f64) {
    const BEAMS: usize = 31;
    let grid = test_room();
    let lidar = Lidar2d::<BEAMS>::new(TAU / 3.0, 4.0, 0.0, 0.0);
    let true_position = [1.0, 0.8];
    let mut localizer = GlobalLocalizer::<BEAMS>::new(
        [true_position[0], true_position[1], 0.0],
        InitialParticleCloud::default(),
        4.0,
        0.1,
        seed,
    )
    .unwrap();
    let mut scan_rng = Pcg32::seed_from_u64(seed ^ 0x9e37_79b9);
    let delta_heading = 0.05;
    let mut true_heading = 0.0;
    for _ in 0..cycles {
        localizer.predict(0.0, delta_heading).unwrap();
        true_heading += delta_heading;
        let scan = lidar.simulate(
            &grid,
            [true_position[0], true_position[1], true_heading],
            &mut scan_rng,
        );
        localizer.update(&scan, &grid, &lidar).unwrap();
    }
    let (pose, _) = localizer.estimate();
    (pose, localizer.effective_sample_size(), true_heading)
}

#[test]
fn it_finds_the_true_pose_while_turning() {
    let (pose, effective_sample_size, true_heading) = localize(7, 60);
    let offset = (pose[0] - 1.0).hypot(pose[1] - 0.8);
    assert!(
        offset < 0.35,
        "position {:?} off the truth by {offset}",
        pose.into_array()
    );
    assert!(
        wrap_angle(pose[2] - true_heading).abs() < 0.3,
        "heading {} off the true {true_heading}",
        pose[2]
    );
    let particle_count = InitialParticleCloud::default().particle_count as f64;
    assert!(
        effective_sample_size > 1.0 && effective_sample_size <= particle_count + 1e-6,
        "effective sample size out of range: {effective_sample_size}"
    );
}

#[test]
fn a_fixed_seed_reproduces_the_estimate() {
    let (first, _, _) = localize(11, 20);
    let (second, _, _) = localize(11, 20);
    assert_eq!(first.into_array(), second.into_array());
}

#[test]
fn the_cloud_size_is_what_was_asked_for() {
    let cloud = InitialParticleCloud {
        particle_count: 120,
        ..InitialParticleCloud::default()
    };
    let localizer = GlobalLocalizer::<31>::new([1.0, 0.8, 0.0], cloud, 4.0, 0.1, 3).unwrap();
    assert_eq!(localizer.particle_count(), 120);
}

#[test]
fn a_scattered_cloud_is_not_yet_converged() {
    // Straight out of the constructor the guesses are spread wide, so no fix is trustworthy yet.
    let localizer = GlobalLocalizer::<31>::new(
        [1.0, 0.8, 0.0],
        InitialParticleCloud::default(),
        4.0,
        0.1,
        3,
    )
    .unwrap();
    assert!(!localizer.is_converged(0.04, 0.01));
}
