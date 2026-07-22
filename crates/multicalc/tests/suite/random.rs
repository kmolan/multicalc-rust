use multicalc::random::{Pcg32, RandomSource};

#[test]
fn known_sequence() {
    // A fixed seed reproduces a fixed sequence of words.
    let mut generator = Pcg32::new(42);
    let sequence = [
        generator.next_u32(),
        generator.next_u32(),
        generator.next_u32(),
    ];
    assert_eq!(sequence, [1898997482, 1014631766, 4096008554]);
}

#[test]
fn same_seed_agrees_different_stream_differs() {
    let mut first = Pcg32::new(7);
    let mut same = Pcg32::new(7);
    for _ in 0..100 {
        assert_eq!(first.next_u32(), same.next_u32());
    }

    let mut other_stream = Pcg32::with_stream(7, 99);
    let mut default_stream = Pcg32::new(7);
    let mut any_difference = false;
    for _ in 0..100 {
        if other_stream.next_u32() != default_stream.next_u32() {
            any_difference = true;
        }
    }
    assert!(any_difference, "different streams should not match");
}

#[test]
fn uniform_stays_in_unit_range() {
    let mut generator = Pcg32::new(123);
    for _ in 0..10_000 {
        let value = generator.next_unit_f64();
        assert!(
            (0.0..1.0).contains(&value),
            "uniform draw out of range: {value}"
        );
    }
}

#[test]
fn standard_normal_batch_has_unit_statistics() {
    let mut generator = Pcg32::new(2024);
    let count = 100_000;
    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;
    for _ in 0..count {
        let value = generator.standard_normal();
        sum += value;
        sum_of_squares += value * value;
    }
    let mean = sum / count as f64;
    let variance = sum_of_squares / count as f64 - mean * mean;
    assert!(mean.abs() < 0.02, "mean too far from zero: {mean}");
    assert!(
        (0.97..1.03).contains(&variance),
        "variance off unit: {variance}"
    );
}
