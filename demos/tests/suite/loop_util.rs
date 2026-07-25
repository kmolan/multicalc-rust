use multicalc_demos::loop_util::{LatencyRing, commas};

#[test]
fn commas_groups_thousands() {
    assert_eq!(commas(0), "0");
    assert_eq!(commas(999), "999");
    assert_eq!(commas(1000), "1,000");
    assert_eq!(commas(61204), "61,204");
    assert_eq!(commas(412000), "412,000");
    assert_eq!(commas(1234567), "1,234,567");
}

#[test]
fn empty_ring_has_no_summary() {
    assert!(LatencyRing::new(8).summary().is_none());
}

#[test]
fn summary_over_a_known_window() {
    // 0..=100 in ascending order: median 50, p99 99, max 100.
    let mut ring = LatencyRing::new(101);
    for v in 0..=100 {
        ring.push(v as f64);
    }
    let p = ring.summary().unwrap();
    assert_eq!(p.median, 50.0);
    assert_eq!(p.p99, 99.0);
    assert_eq!(p.max, 100.0);
}

#[test]
fn ring_evicts_oldest_when_full() {
    // Capacity 3, push 5 values: the window holds the last three (2, 3, 4).
    let mut ring = LatencyRing::new(3);
    for v in 0..5 {
        ring.push(v as f64);
    }
    let p = ring.summary().unwrap();
    assert_eq!(p.max, 4.0);
    assert_eq!(p.median, 3.0);
}
