use multicalc_demos::sim::lidar::Lidar2d;
use multicalc_demos::sim::occupancy_grid::OccupancyGrid;
use rand::SeedableRng;
use rand_pcg::Pcg32;
use std::f64::consts::PI;

// A grid with one occupied column whose left face sits exactly at x = 2, tall enough that the
// oblique beams meet it away from any row boundary. This keeps the closed-form scan exact.
#[must_use]
fn wall() -> OccupancyGrid {
    let mut grid = OccupancyGrid::new(15, 12, 1.0, [-5.0, -5.25]);
    for row in 0..grid.rows() {
        grid.set_cell(7, row, true);
    }
    grid
}

#[test]
fn noiseless_scan_matches_closed_form() {
    // Three beams at -π/4, 0, +π/4 against a wall at x = 2: the oblique beams travel
    // 2 / cos(π/4) to reach it.
    let lidar = Lidar2d::<3>::new(PI / 2.0, 10.0, 0.0, 0.0);
    let mut rng = Pcg32::seed_from_u64(1);
    let scan = lidar.simulate(&wall(), [0.0, 0.0, 0.0], &mut rng);
    let oblique = 2.0 / (PI / 4.0).cos();
    assert!((scan[0] - oblique).abs() < 1e-9, "{}", scan[0]);
    assert!((scan[1] - 2.0).abs() < 1e-9, "{}", scan[1]);
    assert!((scan[2] - oblique).abs() < 1e-9, "{}", scan[2]);
}

#[test]
fn empty_map_reads_as_no_return() {
    let lidar = Lidar2d::<5>::new(PI / 2.0, 10.0, 0.0, 0.0);
    let mut rng = Pcg32::seed_from_u64(1);
    let empty = OccupancyGrid::new(15, 12, 1.0, [-5.0, -5.25]);
    let scan = lidar.simulate(&empty, [0.0, 0.0, 0.0], &mut rng);
    assert!(scan.iter().all(|range| range.is_infinite()));
}

#[test]
fn certain_dropout_drops_every_beam() {
    let lidar = Lidar2d::<5>::new(PI / 2.0, 10.0, 0.0, 1.0);
    let mut rng = Pcg32::seed_from_u64(1);
    let scan = lidar.simulate(&wall(), [0.0, 0.0, 0.0], &mut rng);
    assert!(scan.iter().all(|range| range.is_infinite()));
}

#[test]
fn a_fixed_seed_reproduces_the_scan() {
    let lidar = Lidar2d::<9>::new(PI / 2.0, 10.0, 0.03, 0.01);
    let map = wall();
    let mut first_rng = Pcg32::seed_from_u64(7);
    let mut second_rng = Pcg32::seed_from_u64(7);
    let first = lidar.simulate(&map, [0.0, 0.0, 0.0], &mut first_rng);
    let second = lidar.simulate(&map, [0.0, 0.0, 0.0], &mut second_rng);
    assert_eq!(first, second);
}

#[test]
fn range_noise_has_about_the_right_spread() {
    // One beam straight at the wall, 10 000 scans from a single seeded generator. Fixed seed,
    // so this is a regression check rather than a flaky statistical test.
    let lidar = Lidar2d::<2>::new(0.0, 10.0, 0.03, 0.0);
    let map = wall();
    let mut rng = Pcg32::seed_from_u64(11);
    let samples = 10_000;
    let mut readings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let scan = lidar.simulate(&map, [0.0, 0.0, 0.0], &mut rng);
        readings.push(scan[0]);
    }
    let mean = readings.iter().sum::<f64>() / samples as f64;
    let variance = readings
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f64>()
        / samples as f64;
    let deviation = variance.sqrt();
    assert!((mean - 2.0).abs() < 0.003, "mean {mean}");
    assert!((deviation - 0.03).abs() < 0.003, "deviation {deviation}");
}
