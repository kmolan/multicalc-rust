//! Geometric-Jacobian tests: closed-form columns, which joints contribute, frame choice, an
//! autodiff cross-check, singularity classification, error cases, and f32 coverage.

use core::f64::consts::FRAC_PI_2;

use multicalc::error::KinematicsError;
use multicalc::kinematics::{
    JacobianFrame, Joint, JointKind, JointParent, KinematicTree, SingularityKind,
};
use multicalc::linear_algebra::{Matrix3D, Vector, Vector3D};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE3, SO3};

const TOL: f64 = 1e-12;

// ---- helpers ----------------------------------------------------------------

fn axis_x<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ONE, T::ZERO, T::ZERO])
}

fn axis_y<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ONE, T::ZERO])
}

fn axis_z<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ZERO, T::ONE])
}

fn translation<T: Numeric>(x: f64, y: f64, z: f64) -> SE3<T> {
    SE3::from_parts(
        SO3::identity(),
        Vector::new([T::from_f64(x), T::from_f64(y), T::from_f64(z)]),
    )
}

/// Two revolute joints about z on unit links, plus a fixed tool one unit past the elbow.
fn planar_arm<T: Numeric>() -> KinematicTree<3, 3, T> {
    let link = translation(1.0, 0.0, 0.0);
    KinematicTree::try_from_joints(
        &[
            Joint::revolute(axis_z(), SE3::identity()),
            Joint::revolute(axis_z(), link),
            Joint::fixed(link),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    )
    .unwrap()
}

/// One hinge about z carrying two sibling hinges about y at different offsets, each with a welded
/// tool at its end. Left branch: joints 1 and 2. Right branch: joints 3 and 4.
fn branching_arm<T: Numeric>() -> KinematicTree<5, 5, T> {
    KinematicTree::try_from_joints(
        &[
            Joint::revolute(axis_z(), SE3::identity()),
            Joint::revolute(axis_y(), translation(0.2, 0.0, 0.3)),
            Joint::fixed(translation(0.5, 0.0, 0.0)),
            Joint::revolute(axis_y(), translation(-0.2, 0.1, 0.3)),
            Joint::fixed(translation(-0.5, 0.0, 0.0)),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
            JointParent::Joint(0),
            JointParent::Joint(3),
        ],
    )
    .unwrap()
}

/// A seven-joint arm of mixed axes and one slide, with a welded tool: eight slots in all.
fn spatial_arm<T: Numeric>() -> KinematicTree<8, 8, T> {
    KinematicTree::try_from_joints(
        &[
            Joint::revolute(axis_z(), SE3::identity()),
            Joint::revolute(axis_y(), translation(0.0, 0.0, 0.33)),
            Joint::revolute(axis_z(), translation(0.0, 0.0, 0.32)),
            Joint::prismatic(axis_x(), translation(0.08, 0.0, 0.0)),
            Joint::revolute(axis_y(), translation(0.25, 0.0, 0.0)),
            Joint::revolute(axis_x(), translation(0.09, 0.0, 0.06)),
            Joint::revolute(axis_z(), translation(0.11, 0.02, 0.0)),
            Joint::fixed(translation(0.0, 0.0, 0.107)),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
            JointParent::Joint(2),
            JointParent::Joint(3),
            JointParent::Joint(4),
            JointParent::Joint(5),
            JointParent::Joint(6),
        ],
    )
    .unwrap()
}

/// Three slides along the world axes plus hinges about x and y through the tool frame: rank 5,
/// degenerate in ω_z.
fn no_spin_about_z() -> KinematicTree<6, 6, f64> {
    KinematicTree::try_from_joints(
        &[
            Joint::prismatic(axis_x(), SE3::identity()),
            Joint::prismatic(axis_y(), SE3::identity()),
            Joint::prismatic(axis_z(), SE3::identity()),
            Joint::revolute(axis_x(), SE3::identity()),
            Joint::revolute(axis_y(), SE3::identity()),
            Joint::fixed(SE3::identity()),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
            JointParent::Joint(2),
            JointParent::Joint(3),
            JointParent::Joint(4),
        ],
    )
    .unwrap()
}

/// Slides along x and y plus hinges about all three axes through the tool frame: rank 5,
/// degenerate in v_z.
fn no_move_along_z() -> KinematicTree<6, 6, f64> {
    KinematicTree::try_from_joints(
        &[
            Joint::prismatic(axis_x(), SE3::identity()),
            Joint::prismatic(axis_y(), SE3::identity()),
            Joint::revolute(axis_x(), SE3::identity()),
            Joint::revolute(axis_y(), SE3::identity()),
            Joint::revolute(axis_z(), SE3::identity()),
            Joint::fixed(SE3::identity()),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
            JointParent::Joint(2),
            JointParent::Joint(3),
            JointParent::Joint(4),
        ],
    )
    .unwrap()
}

/// As [`no_move_along_z`], but the last hinge's axis is (x + z)/sqrt(2) and its anchor sits 1 m
/// along -y of the tool, so its twist carries equal v_z and ω_z: rank 5, degenerate in
/// (v_z - ω_z)/sqrt(2).
fn tilted_offset_arm() -> KinematicTree<6, 6, f64> {
    let leaning = Vector::new([
        core::f64::consts::FRAC_1_SQRT_2,
        0.0,
        core::f64::consts::FRAC_1_SQRT_2,
    ]);
    KinematicTree::try_from_joints(
        &[
            Joint::prismatic(axis_x(), SE3::identity()),
            Joint::prismatic(axis_y(), SE3::identity()),
            Joint::revolute(axis_x(), SE3::identity()),
            Joint::revolute(axis_y(), SE3::identity()),
            Joint::revolute(leaning, translation(0.0, -1.0, 0.0)),
            Joint::fixed(translation(0.0, 1.0, 0.0)),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
            JointParent::Joint(2),
            JointParent::Joint(3),
            JointParent::Joint(4),
        ],
    )
    .unwrap()
}

/// Six hinges alternating about x and y on 0.25 m links, tool welded 0.25 m past the last — the
/// chain `six_hinge_chain_near_singular` fixes. Zero readings leave it stretched along z.
fn stretched_six_hinge_arm() -> KinematicTree<7, 7, f64> {
    let link = translation::<f64>(0.0, 0.0, 0.25);
    let mut tree = KinematicTree::new();
    for index in 0..6 {
        let axis = if index % 2 == 0 { axis_x() } else { axis_y() };
        let origin = if index == 0 { SE3::identity() } else { link };
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(Joint::revolute(axis, origin), parent).unwrap();
    }
    tree.push(Joint::fixed(link), JointParent::Joint(5))
        .unwrap();
    tree
}

/// A configuration that leaves no joint at a special angle.
fn generic_readings() -> Vector<8, f64> {
    Vector::new([0.31, -0.62, 0.44, 0.17, 1.05, -0.83, 0.29, 0.0])
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

fn assert_vector_close(got: Vector3D<f64>, want: [f64; 3], what: &str) {
    let got = *got.as_array();
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(*got, *want, &format!("{what} component {index}"));
    }
}

// ---- closed-form columns ----------------------------------------------------

#[test]
fn planar_two_link_columns_are_closed_form() {
    let tree = planar_arm::<f64>();

    // Stretched along x, the tool sits at (2, 0, 0): the shoulder is two units from it, the elbow
    // one, and turning either sweeps the tool sideways.
    let jacobian = tree
        .geometric_jacobian_at(&Vector::zeros(), 2, JacobianFrame::World)
        .unwrap();
    let shoulder = jacobian.column(0).unwrap();
    let elbow = jacobian.column(1).unwrap();

    assert_vector_close(
        shoulder.linear(),
        [0.0, 2.0, 0.0],
        "shoulder linear at zero",
    );
    assert_vector_close(elbow.linear(), [0.0, 1.0, 0.0], "elbow linear at zero");
    assert_vector_close(shoulder.angular(), [0.0, 0.0, 1.0], "shoulder angular");
    assert_vector_close(elbow.angular(), [0.0, 0.0, 1.0], "elbow angular");

    // Shoulder at a quarter turn, elbow straight: the tool swings round to (0, 2, 0) and the sweep
    // turns with it.
    let jacobian = tree
        .geometric_jacobian_at(&Vector::new([FRAC_PI_2, 0.0, 0.0]), 2, JacobianFrame::World)
        .unwrap();

    assert_vector_close(
        jacobian.column(0).unwrap().linear(),
        [-2.0, 0.0, 0.0],
        "shoulder linear at a quarter turn",
    );
    assert_vector_close(
        jacobian.column(1).unwrap().linear(),
        [-1.0, 0.0, 0.0],
        "elbow linear at a quarter turn",
    );
}

#[test]
fn prismatic_column_is_the_axis_and_no_turn() {
    let mut tree = KinematicTree::<1, 1, f64>::new();
    tree.push(
        Joint::prismatic(axis_x::<f64>(), SE3::identity()),
        JointParent::World,
    )
    .unwrap();

    // A slide carries the tool along its own axis at unit rate whatever the reading is.
    for reading in [0.0, 0.75, -2.5] {
        let jacobian = tree
            .geometric_jacobian_at(&Vector::new([reading]), 0, JacobianFrame::World)
            .unwrap();
        let column = jacobian.column(0).unwrap();

        assert_vector_close(column.linear(), [1.0, 0.0, 0.0], "slide linear");
        assert_vector_close(column.angular(), [0.0, 0.0, 0.0], "slide angular");
    }
}

#[test]
fn fixed_joint_column_is_zero() {
    let jacobian = planar_arm::<f64>()
        .geometric_jacobian_at(&Vector::new([0.3, -0.7, 0.0]), 2, JacobianFrame::World)
        .unwrap();
    let weld = jacobian.column(2).unwrap();

    assert_vector_close(weld.linear(), [0.0, 0.0, 0.0], "weld linear");
    assert_vector_close(weld.angular(), [0.0, 0.0, 0.0], "weld angular");
}

#[test]
fn sibling_branch_columns_are_zero() {
    let tree = branching_arm::<f64>();
    let readings = Vector::new([0.4, -0.6, 0.0, 1.1, 0.0]);

    // The left branch's tool hangs off joints 0 and 1 alone.
    let jacobian = tree
        .geometric_jacobian_at(&readings, 2, JacobianFrame::World)
        .unwrap();

    let base = jacobian.column(0).unwrap();
    assert!(
        base.linear().norm() > 0.1 && base.angular().norm() > 0.5,
        "the shared base should move the left tool"
    );
    assert!(
        jacobian.column(1).unwrap().angular().norm() > 0.5,
        "the left branch's own hinge should move the left tool"
    );

    let sibling = jacobian.column(3).unwrap();
    assert_vector_close(sibling.linear(), [0.0, 0.0, 0.0], "sibling linear");
    assert_vector_close(sibling.angular(), [0.0, 0.0, 0.0], "sibling angular");

    // Both welds sit off the movable chain too.
    for index in [2, 4] {
        let weld = jacobian.column(index).unwrap();
        assert_vector_close(weld.linear(), [0.0, 0.0, 0.0], "weld linear");
        assert_vector_close(weld.angular(), [0.0, 0.0, 0.0], "weld angular");
    }
}

// ---- frames -----------------------------------------------------------------

#[test]
fn body_frame_is_the_world_frame_rotated() {
    let tree = spatial_arm::<f64>();
    let readings = generic_readings();

    let state = tree.forward_kinematics(&readings).unwrap();
    let world = tree
        .geometric_jacobian(&state, 7, JacobianFrame::World)
        .unwrap();
    let body = tree
        .geometric_jacobian(&state, 7, JacobianFrame::Body)
        .unwrap();

    // Both are taken at the same point, so nothing but the tool's own rotation separates them.
    let into_tool_axes = state.pose(7).unwrap().rotation().inverse();
    assert_eq!(world.frame(), JacobianFrame::World);
    assert_eq!(body.frame(), JacobianFrame::Body);

    for index in 0..tree.len() {
        let world_column = world.column(index).unwrap();
        let body_column = body.column(index).unwrap();

        assert_vector_close(
            body_column.linear(),
            *into_tool_axes.act(world_column.linear()).as_array(),
            &format!("column {index} linear"),
        );
        assert_vector_close(
            body_column.angular(),
            *into_tool_axes.act(world_column.angular()).as_array(),
            &format!("column {index} angular"),
        );
    }
}

// ---- autodiff cross-check ---------------------------------------------------

#[test]
fn matches_autodiff_through_forward_kinematics() {
    let tree = spatial_arm::<f64>();
    let dual_tree = spatial_arm::<Dual<f64>>();
    let readings = generic_readings();

    let jacobian = tree
        .geometric_jacobian_at(&readings, 7, JacobianFrame::World)
        .unwrap();

    for moved in 0..tree.len() {
        // Differentiate the whole forward solve with respect to one reading at a time.
        let dual_readings = Vector::<8, Dual<f64>>::from_fn(|index| {
            let reading = *readings.get(index).unwrap();
            if index == moved {
                Dual::variable(reading)
            } else {
                Dual::constant(reading)
            }
        });
        let pose = dual_tree
            .forward_kinematics(&dual_readings)
            .unwrap()
            .pose(7)
            .unwrap();

        // How fast the tool's origin travels is just the derivative of where it is.
        let want_linear = pose.translation().as_array().map(|value| value.deriv);

        // How fast it turns comes from the rotation and its derivative: the turning rate is the
        // part of `derivative · transpose` that survives, read back out as a three-vector.
        let rotation = pose.rotation().to_matrix();
        let value =
            Matrix3D::<f64>::from_fn(|row, column| rotation.get(row, column).unwrap().value);
        let derivative =
            Matrix3D::<f64>::from_fn(|row, column| rotation.get(row, column).unwrap().deriv);
        let want_angular = *SO3::vee(derivative * value.transpose()).as_array();

        let got = jacobian.column(moved).unwrap();
        let want = [want_linear, want_angular].concat();

        // Guard against the comparison passing because both sides are zero. A weld is the one
        // joint that genuinely leaves the tool alone.
        if tree.joint(moved).unwrap().kind() != JointKind::Fixed {
            assert!(
                want.iter().any(|value| value.abs() > 0.01),
                "joint {moved} should move the tool"
            );
        }
        for (row, (got, want)) in got.as_array().iter().zip(want.iter()).enumerate() {
            // Relative, because a stretched-out arm's entries span several orders of magnitude.
            assert!(
                (got - want).abs() <= TOL * want.abs().max(1.0),
                "joint {moved}, row {row}: got {got}, want {want}"
            );
        }
    }
}

// ---- singularity classification ---------------------------------------------

#[test]
fn a_chain_well_clear_of_a_singularity_reports_none() {
    let jacobian = spatial_arm::<f64>()
        .geometric_jacobian_at(&generic_readings(), 7, JacobianFrame::World)
        .unwrap();

    assert_eq!(
        jacobian.classify_singularity(1e-2),
        Ok(SingularityKind::None)
    );
}

#[test]
fn the_threshold_decides_what_counts_as_near_singular() {
    // Same Jacobian either side of its own σ_min.
    let jacobian = spatial_arm::<f64>()
        .geometric_jacobian_at(&generic_readings(), 7, JacobianFrame::World)
        .unwrap();
    let smallest = jacobian.smallest_singular_value().unwrap();

    assert_eq!(
        jacobian.classify_singularity(smallest * 0.5),
        Ok(SingularityKind::None)
    );
    assert_ne!(
        jacobian.classify_singularity(smallest * 2.0),
        Ok(SingularityKind::None)
    );
}

#[test]
fn a_tool_that_cannot_be_spun_about_z_is_rotational() {
    // Rank 5: span is {v_x, v_y, v_z, ω_x, ω_y}, nullity 1 at ω_z.
    let jacobian = no_spin_about_z()
        .geometric_jacobian_at(&Vector::zeros(), 5, JacobianFrame::World)
        .unwrap();

    assert_eq!(
        jacobian.classify_singularity(1e-2),
        Ok(SingularityKind::Rotational)
    );
}

#[test]
fn a_tool_that_cannot_be_moved_along_z_is_positional() {
    // Rank 5: span is {v_x, v_y, ω_x, ω_y, ω_z}, nullity 1 at v_z.
    let jacobian = no_move_along_z()
        .geometric_jacobian_at(&Vector::zeros(), 5, JacobianFrame::World)
        .unwrap();

    assert_eq!(
        jacobian.classify_singularity(1e-2),
        Ok(SingularityKind::Positional)
    );
}

#[test]
fn a_lost_direction_that_both_moves_and_turns_is_mixed() {
    // Rank 5, nullity 1 at (v_z - ω_z)/sqrt(2): the last joint's 1 m offset makes its v_z and ω_z
    // contributions equal, so the halves split 50/50 and neither clears two thirds.
    let jacobian = tilted_offset_arm()
        .geometric_jacobian_at(&Vector::zeros(), 5, JacobianFrame::World)
        .unwrap();

    assert_eq!(
        jacobian.classify_singularity(1e-2),
        Ok(SingularityKind::Mixed)
    );
}

#[test]
fn a_stretched_out_chain_is_degenerate() {
    // Rank 4 at the stretched configuration: nullity 2, spanning v_z and ω_z. Which member the SVD
    // returns at index 5 is arbitrary, so only the degeneracy itself is pinned.
    let jacobian = stretched_six_hinge_arm()
        .geometric_jacobian_at(&Vector::zeros(), 6, JacobianFrame::World)
        .unwrap();

    assert_ne!(
        jacobian.classify_singularity(1e-2),
        Ok(SingularityKind::None)
    );
}

#[test]
fn a_two_link_arm_is_always_degenerate() {
    // Rank <= 2 against a 6-DOF task at every configuration.
    let tree = planar_arm::<f64>();
    for readings in [Vector::zeros(), Vector::new([0.3, -0.7, 0.0])] {
        let jacobian = tree
            .geometric_jacobian_at(&readings, 2, JacobianFrame::World)
            .unwrap();
        assert_ne!(
            jacobian.classify_singularity(1e-2),
            Ok(SingularityKind::None)
        );
    }
}

// ---- errors -----------------------------------------------------------------

#[test]
fn errors_on_a_tool_index_past_the_tree() {
    let tree = planar_arm::<f64>();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    assert_eq!(
        tree.geometric_jacobian(&state, 3, JacobianFrame::World),
        Err(KinematicsError::ToolIndexOutOfRange)
    );
    assert_eq!(
        tree.geometric_jacobian_at(&Vector::zeros(), 3, JacobianFrame::World),
        Err(KinematicsError::ToolIndexOutOfRange)
    );
}

#[test]
fn errors_when_the_state_came_from_another_tree() {
    let two_joints = KinematicTree::<3, 3, f64>::try_from_joints(
        &[
            Joint::revolute(axis_z::<f64>(), SE3::identity()),
            Joint::revolute(axis_z(), translation(1.0, 0.0, 0.0)),
        ],
        &[JointParent::World, JointParent::Joint(0)],
    )
    .unwrap();
    let state = two_joints.forward_kinematics(&Vector::zeros()).unwrap();

    assert_eq!(
        planar_arm::<f64>().geometric_jacobian(&state, 1, JacobianFrame::World),
        Err(KinematicsError::StateShapeMismatch)
    );
}

// ---- scalar coverage --------------------------------------------------------

#[test]
fn runs_in_f32() {
    let jacobian = planar_arm::<f32>()
        .geometric_jacobian_at(&Vector::zeros(), 2, JacobianFrame::World)
        .unwrap();

    let shoulder = jacobian.column(0).unwrap().as_array();
    let elbow = jacobian.column(1).unwrap().as_array();
    let want_shoulder = [0.0_f32, 2.0, 0.0, 0.0, 0.0, 1.0];
    let want_elbow = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0];

    for (row, (got, want)) in shoulder.iter().zip(want_shoulder.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "shoulder row {row}: {got}");
    }
    for (row, (got, want)) in elbow.iter().zip(want_elbow.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "elbow row {row}: {got}");
    }
}
