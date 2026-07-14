//! Spatial-math integration tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "spatial/quaternion.rs"]
mod quaternion;

#[path = "spatial/lie.rs"]
mod lie;

#[path = "spatial/twist_wrench.rs"]
mod twist_wrench;
