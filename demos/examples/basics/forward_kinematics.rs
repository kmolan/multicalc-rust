//! Forward kinematics on a `KinematicTree`: a planar two-link arm against its closed form, a joint
//! whose reference configuration is shifted away from the model's zero, a branch carrying two links
//! side by side, and a one-`Dual` derivative pushed straight through the whole solve.
//!
//! Run with: `cargo run -p multicalc-demos --example forward_kinematics`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::f64::consts::FRAC_PI_2;

use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector;
use multicalc::scalar::Dual;
use multicalc::spatial::{SE3, SO3};

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<20} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

/// Two revolute joints about z on unit links, plus a welded tool frame past the elbow.
fn planar_arm(shoulder_offset: f64) -> KinematicTree<3, f64> {
    let about_z = Vector::new([0.0, 0.0, 1.0]);
    let along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    KinematicTree::try_from_joints(
        &[
            Joint::revolute(about_z, SE3::identity()).with_zero_offset(shoulder_offset),
            Joint::revolute(about_z, along_x),
            Joint::fixed(along_x),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    )
    .unwrap()
}

fn main() {
    let (shoulder, elbow) = (0.3_f64, -0.7_f64);
    let tool_x = shoulder.cos() + (shoulder + elbow).cos();
    let tool_y = shoulder.sin() + (shoulder + elbow).sin();

    // (1) A planar two-link arm, against the closed form for unit links.
    let arm = planar_arm(0.0);
    let state = arm
        .forward_kinematics(&Vector::new([shoulder, elbow, 0.0]))
        .unwrap();
    let tool = state.pose(2).unwrap().translation();
    println!("Planar two-link arm at q = [0.3, -0.7]");
    report("tool x [m]", tool[0], tool_x);
    report("tool y [m]", tool[1], tool_y);
    report("tool z [m]", tool[2], 0.0);

    // (2) The same geometry with the shoulder's reference configuration moved to 0.3: the link
    // sits at its origin transform when the encoder reads 0.3, so every reading shifts by that
    // much and the arm reaches the same place.
    let shifted = planar_arm(0.3);
    let state = shifted
        .forward_kinematics(&Vector::new([shoulder + 0.3, elbow, 0.0]))
        .unwrap();
    let tool = state.pose(2).unwrap().translation();
    println!("\nShoulder reference configuration at 0.3, read at q = [0.6, -0.7]");
    report("tool x [m]", tool[0], tool_x);
    report("tool y [m]", tool[1], tool_y);

    // (3) A branch: one revolute joint carrying two welded links, neither of which sees the other.
    let about_z = Vector::new([0.0, 0.0, 1.0]);
    let branched = KinematicTree::<3, f64>::try_from_joints(
        &[
            Joint::revolute(about_z, SE3::identity()),
            Joint::fixed(SE3::from_parts(
                SO3::identity(),
                Vector::new([1.0, 0.0, 0.0]),
            )),
            Joint::fixed(SE3::from_parts(
                SO3::identity(),
                Vector::new([0.0, 1.0, 0.0]),
            )),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(0),
        ],
    )
    .unwrap();
    let state = branched
        .forward_kinematics(&Vector::new([FRAC_PI_2, 0.0, 0.0]))
        .unwrap();
    let first = state.pose(1).unwrap().translation();
    let second = state.pose(2).unwrap().translation();
    println!("\nBranch: root at +90 deg carrying two welded links");
    report("first x [m]", first[0], 0.0);
    report("first y [m]", first[1], 1.0);
    report("second x [m]", second[0], -1.0);
    report("second y [m]", second[1], 0.0);

    // (4) The same solve under Dual: seeding the shoulder reading with a derivative part of one
    // gives d(tool)/d(shoulder) exactly, with nothing hand-derived.
    let about_z = Vector::new([
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(1.0),
    ]);
    let along_x = SE3::from_parts(
        SO3::identity(),
        Vector::new([
            Dual::constant(1.0),
            Dual::constant(0.0),
            Dual::constant(0.0),
        ]),
    );
    let differentiable = KinematicTree::<3, Dual<f64>>::try_from_joints(
        &[
            Joint::revolute(about_z, SE3::identity()),
            Joint::revolute(about_z, along_x),
            Joint::fixed(along_x),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    )
    .unwrap();
    let state = differentiable
        .forward_kinematics(&Vector::new([
            Dual::variable(shoulder),
            Dual::constant(elbow),
            Dual::constant(0.0),
        ]))
        .unwrap();
    let tool = state.pose(2).unwrap().translation();
    println!("\nAutodiff through the whole solve");
    report("d(tool x)/d(shoulder)", tool[0].deriv, -tool_y);
    report("d(tool y)/d(shoulder)", tool[1].deriv, tool_x);
    println!("  the same forward kinematics produced the position and its derivative");

    println!("\nAll checks passed.");
}
