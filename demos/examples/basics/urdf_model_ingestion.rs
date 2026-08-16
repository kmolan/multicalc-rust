//! Loads a URDF model file and reports the robot's tree, its joint geometry and travel, and which
//! links carry mass.
//!
//! Run with: `cargo run -p multicalc-demos --example urdf_model_ingestion`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use multicalc::linear_algebra::Vector;

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<22} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    // (1) Read the model. `load_path` picks the reader from the extension: `.urdf` reads URDF.
    // The file ships with this repository under its own upstream licence.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../third_party/moveit_resources_panda/panda.urdf");
    let model = multicalc_robot_model::load_path(&path).unwrap();

    println!("Model: {}  ({})", model.name(), model.format());
    println!("  bodies                 = {}", model.body_count());
    println!("  movable joints         = {}", model.movable_joint_count());
    println!("  floating base          = {}", model.has_floating_base());
    assert_eq!(model.body_count(), 12);
    assert_eq!(model.movable_joint_count(), 9);

    // (2) The body tree. URDF states its links and joints as one flat list, so the reader resolves
    // the tree by name and emits bodies in topological order: every parent precedes its children.
    println!("\nBody tree");
    for (index, body) in model.bodies().iter().enumerate() {
        let parent = body
            .parent()
            .map_or_else(|| "-".to_string(), |slot| slot.to_string());
        let kind = body
            .joint()
            .map_or_else(|| "weld".to_string(), |joint| format!("{:?}", joint.kind()));
        println!(
            "  {index:>2}  {:<20} parent {parent:>2}   {kind}",
            body.name()
        );
        assert!(body.parent().is_none_or(|slot| slot < index));
    }

    // (3) Which links carry mass. URDF states a link's mass outright or not at all — it is never
    // worked out from geometry — so a link with no `<inertial>` is genuinely massless. This file
    // is a kinematics-only description and states none anywhere.
    let with_mass = model
        .bodies()
        .iter()
        .filter(|body| body.inertia().is_some())
        .count();
    println!("\nMass properties");
    println!(
        "  links with mass        = {with_mass} of {}",
        model.body_count()
    );
    println!("  (this file states no <inertial> at all, so nothing is invented)");
    assert_eq!(with_mass, 0);

    // (4) Each movable joint's axis and travel. A URDF joint sits at its child link frame's
    // origin, so it carries no anchor offset, and its axis is stated in that same frame.
    println!("\nMovable joints");
    for body in model.bodies() {
        let Some(joint) = body.joint() else {
            continue;
        };
        let axis = joint.axis();
        let travel = joint.limits().map_or_else(
            || "unlimited".to_string(),
            |(lower, upper)| format!("[{lower:.4}, {upper:.4}]"),
        );
        let driven = joint
            .mimic()
            .map_or_else(String::new, |mimic| format!("   follows {}", mimic.joint()));
        println!(
            "  {:<20} {:<10?} axis ({:>5.2},{:>5.2},{:>5.2})   travel {travel:<20}{driven}",
            joint.name(),
            joint.kind(),
            axis[0],
            axis[1],
            axis[2],
        );
        assert_eq!(joint.anchor().into_array(), [0.0; 3]);
    }

    // (5) The second finger mimics the first. A kinematic tree carries no constraints, so the
    // whole model is refused and the chain to the hand is what builds.
    let whole = model.kinematic_tree::<16, 16>();
    println!("\nWhole model as a tree");
    println!("  refused                = {}", whole.unwrap_err());

    let arm = model.kinematic_tree_to::<10, 10>("panda_hand").unwrap();
    let state = arm.forward_kinematics(&Vector::zeros()).unwrap();
    let hand = state.pose(arm.len() - 1).unwrap().translation();
    println!("\nArm chain to panda_hand, at the zero configuration");
    println!("  slots                  = {}", arm.len());
    report("hand x (m)", hand[0], 0.088);
    report("hand y (m)", hand[1], 0.0);
    report("hand z (m)", hand[2], 0.926);
}
