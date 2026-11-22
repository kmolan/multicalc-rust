use multicalc_demos::sim::sensor_noise::gaussian_noise;
use rand::SeedableRng;
use rand_pcg::Pcg32;

#[test]
fn no_noise_draws_nothing() {
    let mut rng = Pcg32::seed_from_u64(1);
    assert_eq!(gaussian_noise(0.0, &mut rng), 0.0);
    assert_eq!(gaussian_noise(-1.0, &mut rng), 0.0);
}

#[test]
fn a_fixed_seed_reproduces_the_draw() {
    let mut first = Pcg32::seed_from_u64(5);
    let mut second = Pcg32::seed_from_u64(5);
    assert_eq!(
        gaussian_noise(0.3, &mut first),
        gaussian_noise(0.3, &mut second)
    );
}

#[test]
fn the_spread_matches_the_deviation() {
    let deviation = 0.25;
    let mut rng = Pcg32::seed_from_u64(6);
    let count = 4000;
    let (mut sum, mut sum_of_squares) = (0.0, 0.0);
    for _ in 0..count {
        let draw = gaussian_noise(deviation, &mut rng);
        sum += draw;
        sum_of_squares += draw * draw;
    }
    let mean = sum / count as f64;
    let spread = (sum_of_squares / count as f64 - mean * mean).sqrt();
    assert!(mean.abs() < 0.02, "draws should centre on zero: {mean}");
    assert!(
        (spread - deviation).abs() < 0.03,
        "spread off the deviation: {spread}"
    );
}
