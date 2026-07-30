#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks model ingestion against MuJoCo's own compile of the same file.
//!
//! MuJoCo defines what an MJCF file means, so it is the reference here rather than a second reader
//! of the same format. It reports a body's inertia as three numbers along the axes the body lines
//! up best with, plus the turn to get there; the generator rebuilds the full tensor in the body's
//! own axes before writing the golden, so what is compared below is nine numbers against nine.

use std::path::Path;

use multicalc::spatial::FreeJointState;
use multicalc_qa::load::*;

/// The vendored models the fixtures name, which sit outside this crate.
fn menagerie() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/menagerie")
}

#[test]
fn ingested_models_match_mujoco() {
    for fx in load_dir("mjcf") {
        let case = fx.case.as_str();
        let path = menagerie().join(fx.inputs["model_file"].as_str());
        let model = multicalc_mjcf::load_path(&path)
            .unwrap_or_else(|e| unreachable!("{case}: loading {path:?}: {e}"));
        let t = fx.tolerances.f64;

        assert_scalar(
            model.inertia().mass(),
            &fx.expected["mass"],
            t,
            &format!("{case} mass"),
        );
        assert_vector::<3>(
            &model.inertia().center_of_mass(),
            &fx.expected["center_of_mass"],
            t,
            &format!("{case} center_of_mass"),
        );
        assert_matrix::<3, 3>(
            &model.inertia().rotational_inertia(),
            &fx.expected["rotational_inertia"],
            t,
            &format!("{case} rotational_inertia"),
        );
        assert_vector::<3>(
            &model.pose().translation(),
            &fx.expected["body_position"],
            t,
            &format!("{case} body_position"),
        );

        // A quaternion and its negative name the same turn, so both sides are compared in their
        // scalar-positive form rather than component by component as written.
        let quaternion = model.pose().rotation().quaternion().as_array();
        let want = fx.expected["body_quaternion"].as_vector();
        let flip = if quaternion[0] < 0.0 { -1.0 } else { 1.0 };
        let want_flip = if want[0] < 0.0 { -1.0 } else { 1.0 };
        for index in 0..4 {
            assert!(
                close(flip * quaternion[index], want_flip * want[index], t),
                "{case} body_quaternion[{index}]: got {}, want {}",
                flip * quaternion[index],
                want_flip * want[index]
            );
        }

        assert_eq!(
            i64::from(model.has_free_joint()),
            fx.expected["free_joint"].as_int(),
            "{case} free_joint"
        );

        // Ties the parsed model to how the core type packs a free joint's numbers. The fixture
        // keys keep MuJoCo's own `nq` / `nv` wording, since they record what MuJoCo reported.
        assert_eq!(
            FreeJointState::<f64>::GENERALIZED_POSITION_DIMENSION as i64,
            fx.expected["configuration_dimension"].as_int(),
            "{case} configuration_dimension"
        );
        assert_eq!(
            FreeJointState::<f64>::GENERALIZED_VELOCITY_DIMENSION as i64,
            fx.expected["velocity_dimension"].as_int(),
            "{case} velocity_dimension"
        );
    }
}
