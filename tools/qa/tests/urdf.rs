#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks URDF ingestion against Pinocchio's own parse of the same file.
//!
//! URDF is Pinocchio's home format, so it is the reference here rather than a second reader of the
//! same format. Pinocchio drops a fixed joint and hangs its child link off the parent joint, so its
//! joint list is short of a link each time; the generator recovers the full link list from the
//! model's BODY frames, and the forward-kinematics fixture compares frame placements, which lean on
//! none of that folding.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc_qa::load::*;
use multicalc_qa::schema::Fixture;

/// The vendored models the fixtures name, which sit outside this crate.
#[must_use]
fn third_party() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party")
}

#[test]
fn ingested_urdf_models_match_pinocchio() {
    for fixture in load_dir("urdf") {
        match fixture.case.as_str() {
            "panda_urdf_tree" => check_panda_tree(&fixture),
            "panda_urdf_forward_kinematics" => check_panda_forward_kinematics(&fixture),
            other => unreachable!("no comparison for fixture {other}"),
        }
    }
}

/// The MoveIt Panda: link list, joint order and kinds, travel limits, and its coupled finger.
fn check_panda_tree(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let path = third_party().join(fixture.inputs["model_file"].as_str());
    let model = multicalc_robot_model::urdf::load_path(&path)
        .unwrap_or_else(|err| unreachable!("{case}: loading {path:?}: {err}"));
    let tolerance = fixture.tolerances.f64;

    let links: Vec<&str> = model.bodies().iter().map(|body| body.name()).collect();
    assert_eq!(
        links.join(" "),
        fixture.expected["link_names"].as_str(),
        "{case} link names"
    );
    assert_eq!(
        model.body_count() as i64,
        fixture.expected["body_count"].as_int(),
        "{case} body count"
    );
    assert_eq!(
        model.movable_joint_count() as i64,
        fixture.expected["movable_joint_count"].as_int(),
        "{case} movable joint count"
    );

    // Pinocchio's joint list holds only the movable joints, in the same order this reader walks
    // the tree, so the two line up entry for entry.
    let joints: Vec<_> = model
        .bodies()
        .iter()
        .filter_map(|body| body.joint())
        .collect();
    let names: Vec<&str> = joints.iter().map(|joint| joint.name()).collect();
    assert_eq!(
        names.join(" "),
        fixture.expected["joint_names"].as_str(),
        "{case} joint names"
    );

    let kinds: String = joints
        .iter()
        .map(|joint| match joint.kind() {
            JointKind::Revolute => 'R',
            JointKind::Prismatic => 'P',
            JointKind::Continuous => 'C',
            other => unreachable!("{case}: unexpected joint kind {other:?}"),
        })
        .collect();
    assert_eq!(
        kinds,
        fixture.expected["joint_kinds"].as_str(),
        "{case} joint kinds"
    );

    let lower = fixture.expected["lower_limits"].as_vector();
    let upper = fixture.expected["upper_limits"].as_vector();
    assert_eq!(lower.len(), joints.len(), "{case} limit count");
    for (index, joint) in joints.iter().enumerate() {
        let ctx = format!("{case} joint {index} ({})", joint.name());
        let (got_lower, got_upper) = joint
            .limits()
            .unwrap_or_else(|| unreachable!("{ctx}: joint states no travel limits"));
        assert!(close(got_lower, lower[index], tolerance), "{ctx} lower");
        assert!(close(got_upper, upper[index], tolerance), "{ctx} upper");
    }

    // The file states no <inertial> anywhere, so no body carries mass properties.
    let with_mass = model
        .bodies()
        .iter()
        .filter(|body| body.inertia().is_some())
        .count();
    assert_eq!(
        with_mass as i64,
        fixture.expected["links_with_mass"].as_int(),
        "{case} links with mass"
    );

    // Read from the file's own text rather than from Pinocchio, which leaves a following joint its
    // own degree of freedom instead of coupling it.
    let follower = model
        .body_named("panda_rightfinger")
        .unwrap_or_else(|| unreachable!("{case}: model has no body called panda_rightfinger"))
        .joint()
        .unwrap_or_else(|| unreachable!("{case}: panda_rightfinger carries no joint"));
    assert_eq!(
        follower.name(),
        fixture.expected["mimicking_joint"].as_str(),
        "{case} mimicking joint"
    );
    let mimic = follower
        .mimic()
        .unwrap_or_else(|| unreachable!("{case}: {} follows nothing", follower.name()));
    assert_eq!(
        mimic.joint(),
        fixture.expected["mimic_joint"].as_str(),
        "{case} mimic joint"
    );
    assert_scalar(
        mimic.multiplier(),
        &fixture.expected["mimic_multiplier"],
        tolerance,
        &format!("{case} mimic multiplier"),
    );
    assert_scalar(
        mimic.offset(),
        &fixture.expected["mimic_offset"],
        tolerance,
        &format!("{case} mimic offset"),
    );
}

/// Where every link on the chain to the hand ends up, across configurations.
fn check_panda_forward_kinematics(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let path = third_party().join(fixture.inputs["model_file"].as_str());
    let model = multicalc_robot_model::urdf::load_path(&path)
        .unwrap_or_else(|err| unreachable!("{case}: loading {path:?}: {err}"));
    let tolerance = fixture.tolerances.f64;

    // One finger mimics the other, which a constraint-free tree cannot hold, so the chain compared
    // here stops at the hand and both fingers sit outside it.
    let frame_count = fixture.expected["frame_count"].as_int() as usize;
    let tree = model
        .kinematic_tree_to::<10, 10>("panda_hand")
        .unwrap_or_else(|err| unreachable!("{case}: building the chain to panda_hand: {err}"));
    assert_eq!(tree.len(), frame_count, "{case} chain length");

    let links: Vec<&str> = model
        .bodies()
        .iter()
        .take(frame_count)
        .map(|body| body.name())
        .collect();
    assert_eq!(
        links.join(" "),
        fixture.expected["link_names"].as_str(),
        "{case} link names"
    );

    let (_, arm_joint_count, configurations) = fixture.inputs["configurations"].as_matrix();
    let (_, _, translations) = fixture.expected["translations"].as_matrix();
    let (_, _, quaternions) = fixture.expected["quaternions"].as_matrix();
    let configuration_count = fixture.expected["configuration_count"].as_int() as usize;

    for run in 0..configuration_count {
        // Slot `k` reads configuration entry `k`, and a welded slot ignores its own, so the seven
        // arm joints sit at entries 1 through 7: slot 0 is the base link, welded to the world.
        let mut configuration = Vector::<10, f64>::zeros();
        for joint in 0..arm_joint_count {
            configuration[joint + 1] = configurations[run * arm_joint_count + joint];
        }

        let state = tree
            .forward_kinematics(&configuration)
            .unwrap_or_else(|err| unreachable!("{case}: forward kinematics for run {run}: {err}"));

        for (slot, link) in links.iter().enumerate() {
            let ctx = format!("{case} run {run} link {link}");
            let pose = state
                .pose(slot)
                .unwrap_or_else(|| unreachable!("{ctx}: no pose for that slot"));

            let base = (run * frame_count + slot) * 3;
            for axis in 0..3 {
                assert!(
                    close(
                        pose.translation()[axis],
                        translations[base + axis],
                        tolerance
                    ),
                    "{ctx} translation[{axis}]"
                );
            }

            // A quaternion and its negative name the same turn, so both sides are compared in
            // scalar-positive form.
            let got = pose.rotation().quaternion();
            let base = (run * frame_count + slot) * 4;
            let want = [
                quaternions[base],
                quaternions[base + 1],
                quaternions[base + 2],
                quaternions[base + 3],
            ];
            let got = [got.w(), got.x(), got.y(), got.z()];
            let flip = if got[0] < 0.0 { -1.0 } else { 1.0 };
            let want_flip = if want[0] < 0.0 { -1.0 } else { 1.0 };
            for component in 0..4 {
                assert!(
                    close(
                        flip * got[component],
                        want_flip * want[component],
                        tolerance
                    ),
                    "{ctx} quaternion[{component}]"
                );
            }
        }
    }
}
