//! A 2D sensor simulator for the demos: a hardcoded map, a forward-arc lidar, and beacon
//! measurements. Std-only and seeded, so a run reproduces exactly. Never part of the core crate.

pub mod beacons;
pub mod lidar;
pub mod map;

pub use beacons::{Beacon, BeaconMeasurement, nearest_beacon_measurement};
pub use lidar::Lidar2d;
pub use map::{Circle, Map, Obstacle, Segment};

/// Wraps an angle to the interval `(-π, π]`.
pub fn wrap_angle(angle: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let wrapped = angle - TAU * ((angle + PI) / TAU).floor();
    if wrapped <= -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::wrap_angle;
    use std::f64::consts::{PI, TAU};

    #[test]
    fn angles_wrap_to_the_half_open_interval() {
        assert!((wrap_angle(TAU + 0.1) - 0.1).abs() < 1e-12);
        assert!((wrap_angle(-PI - 0.1) - (PI - 0.1)).abs() < 1e-12);
        assert!((wrap_angle(PI) - PI).abs() < 1e-12);
        // The interval is half-open at -π, so -π maps to +π.
        assert!((wrap_angle(-PI) - PI).abs() < 1e-12);
    }
}
