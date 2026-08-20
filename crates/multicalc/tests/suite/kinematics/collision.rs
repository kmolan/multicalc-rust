//! Collision-query tests: closed-form sphere, capsule and mixed distances against hand-computed
//! cases, degenerate and parallel segment handling, clearance thresholds, self- against
//! environment primitives, pair exclusion, capacity errors, and f32 coverage.

use core::f64::consts::FRAC_PI_2;

use multicalc::error::KinematicsError;
use multicalc::kinematics::{
    CollisionQuery, CollisionSource, Joint, JointParent, KinematicTree, Primitive,
    capsule_capsule_distance, sphere_capsule_distance, sphere_sphere_distance,
};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::scalar::Numeric;
use multicalc::spatial::{SE3, SO3};

const TOL: f64 = 1e-9;

// ---- helpers ----------------------------------------------------------------

fn pose_at<T: Numeric>(x: f64, y: f64, z: f64) -> SE3<T> {
    SE3::from_parts(
        SO3::identity(),
        Vector::new([T::from_f64(x), T::from_f64(y), T::from_f64(z)]),
    )
}

/// A pose turned `angle` about `axis`, then translated.
fn turned(axis: Vector3D<f64>, angle: f64, x: f64, y: f64, z: f64) -> SE3<f64> {
    SE3::from_parts(SO3::exp(axis.scale(angle)), Vector::new([x, y, z]))
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

/// Two welds a metre apart along x, so a primitive on each is a metre from the other.
fn two_frames() -> KinematicTree<2, 2, f64> {
    KinematicTree::try_from_joints(
        &[
            Joint::fixed(SE3::identity()),
            Joint::fixed(pose_at(1.0, 0.0, 0.0)),
        ],
        &[JointParent::World, JointParent::Joint(0)],
    )
    .unwrap()
}

// ---- sphere/sphere ----------------------------------------------------------

#[test]
fn touching_spheres_have_zero_clearance() {
    let distance = sphere_sphere_distance(pose_at(0.0, 0.0, 0.0), 0.5, pose_at(1.0, 0.0, 0.0), 0.5);
    assert_close(distance, 0.0, "touching");
}

#[test]
fn separated_spheres_report_the_gap() {
    let distance = sphere_sphere_distance(pose_at(0.0, 0.0, 0.0), 0.5, pose_at(3.0, 0.0, 0.0), 1.0);
    assert_close(distance, 1.5, "gap");
}

#[test]
fn overlapping_spheres_report_the_penetration() {
    let distance = sphere_sphere_distance(pose_at(0.0, 0.0, 0.0), 0.5, pose_at(0.5, 0.0, 0.0), 0.5);
    assert_close(distance, -0.5, "penetration");
}

// ---- capsule/capsule --------------------------------------------------------

#[test]
fn parallel_capsules_report_the_gap_between_their_axes() {
    // Both along z, axes 1 m apart in x.
    let distance = capsule_capsule_distance(
        pose_at(0.0, 0.0, 0.0),
        0.2,
        1.0,
        pose_at(1.0, 0.0, 0.0),
        0.2,
        1.0,
    );
    assert_close(distance, 0.6, "parallel gap");
}

#[test]
fn parallel_capsules_offset_past_their_ends_measure_end_to_end() {
    // Both along z, collinear, centres 4 m apart: 1 m of half-length each leaves 2 m of segment
    // gap, less the radii.
    let distance = capsule_capsule_distance(
        pose_at(0.0, 0.0, 0.0),
        0.2,
        1.0,
        pose_at(0.0, 0.0, 4.0),
        0.3,
        1.0,
    );
    assert_close(distance, 1.5, "end-to-end gap");
}

#[test]
fn crossed_capsules_measure_between_their_midpoints() {
    // One along x, one along y, 0.5 m apart in z: closest points are both segment midpoints.
    let along_x = turned(Vector::new([0.0, 1.0, 0.0]), FRAC_PI_2, 0.0, 0.0, 0.0);
    let along_y = turned(Vector::new([1.0, 0.0, 0.0]), -FRAC_PI_2, 0.0, 0.0, 0.5);
    let distance = capsule_capsule_distance(along_x, 0.1, 1.0, along_y, 0.1, 1.0);
    assert_close(distance, 0.3, "crossed gap");
}

#[test]
fn overlapping_capsules_report_the_penetration() {
    let distance = capsule_capsule_distance(
        pose_at(0.0, 0.0, 0.0),
        0.5,
        1.0,
        pose_at(0.4, 0.0, 0.0),
        0.5,
        1.0,
    );
    assert_close(distance, -0.6, "penetration");
}

#[test]
fn a_zero_length_capsule_matches_the_sphere_form() {
    // Half-length zero degenerates the segment to a point, which is a sphere.
    let capsules = capsule_capsule_distance(
        pose_at(0.0, 0.0, 0.0),
        0.3,
        0.0,
        pose_at(2.0, 0.0, 0.0),
        0.2,
        0.0,
    );
    let spheres = sphere_sphere_distance(pose_at(0.0, 0.0, 0.0), 0.3, pose_at(2.0, 0.0, 0.0), 0.2);
    assert_close(capsules, spheres, "degenerate capsule");
    assert_close(capsules, 1.5, "degenerate value");
}

// ---- sphere/capsule ---------------------------------------------------------

#[test]
fn a_sphere_past_a_capsule_end_measures_from_the_cap() {
    // Capsule along z, half-length 0.5, so its +z end sits at 0.5; sphere centre 1 m past that.
    let distance = sphere_capsule_distance(
        pose_at(0.0, 0.0, 1.5),
        0.3,
        pose_at(0.0, 0.0, 0.0),
        0.2,
        0.5,
    );
    assert_close(distance, 0.5, "past the cap");
}

#[test]
fn a_sphere_beside_a_capsule_measures_from_the_axis() {
    // Sphere level with the capsule's midpoint, 2 m out in x.
    let distance = sphere_capsule_distance(
        pose_at(2.0, 0.0, 0.0),
        0.3,
        pose_at(0.0, 0.0, 0.0),
        0.2,
        0.5,
    );
    assert_close(distance, 1.5, "beside the barrel");
}

#[test]
fn the_mixed_pair_is_symmetric() {
    let sphere = Primitive::Sphere { radius: 0.3 };
    let capsule = Primitive::Capsule {
        radius: 0.2,
        half_length: 0.5,
    };
    let sphere_pose = pose_at::<f64>(2.0, 0.0, 0.0);
    let capsule_pose = pose_at::<f64>(0.0, 0.0, 0.0);

    let forward = sphere.distance_to(sphere_pose, capsule, capsule_pose);
    let backward = capsule.distance_to(capsule_pose, sphere, sphere_pose);
    assert_close(forward, backward, "symmetry");
    assert_close(forward, 1.5, "value");
}

// ---- CollisionQuery ---------------------------------------------------------

#[test]
fn two_self_primitives_clear_a_wide_gap() {
    let tree = two_frames();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    let mut query = CollisionQuery::<2, 0, 0, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();
    query
        .push_self_primitive(1, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();

    let report = query.check(&state).unwrap();
    assert_close(report.minimum_clearance, 0.8, "clearance");
    assert!(report.is_clear(0.5));
    assert!(!report.is_clear(0.9));
    assert_eq!(
        report.closest_pair,
        Some((
            CollisionSource::SelfPrimitive(0),
            CollisionSource::SelfPrimitive(1)
        ))
    );
}

#[test]
fn a_local_pose_moves_a_primitive_off_its_frame() {
    let tree = two_frames();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    // Second sphere pushed back along x to sit on top of the first.
    let mut query = CollisionQuery::<2, 0, 0, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();
    query
        .push_self_primitive(
            1,
            Primitive::Sphere { radius: 0.1 },
            pose_at(-1.0, 0.0, 0.0),
        )
        .unwrap();

    let report = query.check(&state).unwrap();
    assert_close(report.minimum_clearance, -0.2, "overlap");
    assert!(!report.is_clear(0.0));
}

#[test]
fn an_excluded_pair_is_never_checked() {
    let tree = two_frames();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    let mut query = CollisionQuery::<2, 0, 1, f64>::new();
    let first = query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.8 }, SE3::identity())
        .unwrap();
    let second = query
        .push_self_primitive(1, Primitive::Sphere { radius: 0.8 }, SE3::identity())
        .unwrap();

    // Overlapping by 0.6 until the pair is dropped; excluded, nothing is left to check.
    assert_close(
        query.check(&state).unwrap().minimum_clearance,
        -0.6,
        "before exclusion",
    );
    query.exclude_pair(second, first).unwrap();

    let report = query.check(&state).unwrap();
    assert_eq!(report.closest_pair, None);
    assert_close(report.minimum_clearance, 0.0, "after exclusion");
    assert!(report.is_clear(1.0));
}

#[test]
fn an_environment_primitive_is_checked_against_every_link() {
    let tree = two_frames();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    let mut query = CollisionQuery::<2, 1, 0, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();
    query
        .push_self_primitive(1, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();
    // An obstacle sitting on the second frame.
    query
        .push_environment_primitive(Primitive::Sphere { radius: 0.2 }, pose_at(1.05, 0.0, 0.0))
        .unwrap();

    let report = query.check(&state).unwrap();
    assert_eq!(
        report.closest_pair,
        Some((
            CollisionSource::SelfPrimitive(1),
            CollisionSource::Environment(0)
        ))
    );
    assert_close(report.minimum_clearance, -0.25, "obstacle overlap");
}

#[test]
fn a_query_with_nothing_to_check_is_vacuously_clear() {
    let tree = two_frames();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    let mut query = CollisionQuery::<2, 0, 0, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();

    let report = query.check(&state).unwrap();
    assert_eq!(report.closest_pair, None);
    assert_close(report.minimum_clearance, 0.0, "no pairs");
    assert!(report.is_clear(10.0));
}

#[test]
fn a_moving_joint_carries_its_primitives() {
    // One hinge about z with a sphere a metre out, against a fixed obstacle on the +y axis: the
    // gap closes as the arm turns to face it.
    let tree = KinematicTree::<1, 1, f64>::try_from_joints(
        &[Joint::revolute(
            Vector::new([0.0, 0.0, 1.0]),
            SE3::identity(),
        )],
        &[JointParent::World],
    )
    .unwrap();

    let mut query = CollisionQuery::<1, 1, 0, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, pose_at(1.0, 0.0, 0.0))
        .unwrap();
    query
        .push_environment_primitive(Primitive::Sphere { radius: 0.1 }, pose_at(0.0, 1.0, 0.0))
        .unwrap();

    let stretched = tree.forward_kinematics(&Vector::new([0.0])).unwrap();
    let turned_to_face = tree.forward_kinematics(&Vector::new([FRAC_PI_2])).unwrap();

    // Quarter turn apart: sqrt(2) m between centres. Turned onto it: coincident.
    assert_close(
        query.check(&stretched).unwrap().minimum_clearance,
        2.0_f64.sqrt() - 0.2,
        "stretched",
    );
    assert_close(
        query.check(&turned_to_face).unwrap().minimum_clearance,
        -0.2,
        "turned onto the obstacle",
    );
}

#[test]
fn counts_track_what_was_pushed() {
    let mut query = CollisionQuery::<2, 1, 0, f64>::new();
    assert_eq!(query.self_primitive_count(), 0);
    assert_eq!(query.environment_primitive_count(), 0);

    assert_eq!(
        query
            .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
            .unwrap(),
        0
    );
    assert_eq!(
        query
            .push_environment_primitive(Primitive::Sphere { radius: 0.1 }, SE3::identity())
            .unwrap(),
        0
    );
    assert_eq!(query.self_primitive_count(), 1);
    assert_eq!(query.environment_primitive_count(), 1);
}

// ---- errors -----------------------------------------------------------------

#[test]
fn errors_past_self_capacity() {
    let mut query = CollisionQuery::<1, 1, 1, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();

    assert_eq!(
        query.push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity()),
        Err(KinematicsError::CollisionCapacityExceeded)
    );
}

#[test]
fn errors_past_environment_capacity() {
    let mut query = CollisionQuery::<1, 1, 1, f64>::new();
    query
        .push_environment_primitive(Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();

    assert_eq!(
        query.push_environment_primitive(Primitive::Sphere { radius: 0.1 }, SE3::identity()),
        Err(KinematicsError::CollisionCapacityExceeded)
    );
}

#[test]
fn errors_past_excluded_capacity() {
    let mut query = CollisionQuery::<2, 1, 1, f64>::new();
    query.exclude_pair(0, 1).unwrap();

    assert_eq!(
        query.exclude_pair(0, 1),
        Err(KinematicsError::CollisionCapacityExceeded)
    );
}

#[test]
fn errors_on_a_frame_the_state_does_not_have() {
    let tree = two_frames();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    let mut query = CollisionQuery::<2, 0, 0, f64>::new();
    query
        .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();
    query
        .push_self_primitive(9, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();

    assert_eq!(
        query.check(&state).err(),
        Some(KinematicsError::ToolIndexOutOfRange)
    );
}

// ---- scalar coverage --------------------------------------------------------

#[test]
fn runs_in_f32() {
    let tree = KinematicTree::<2, 2, f32>::try_from_joints(
        &[
            Joint::fixed(SE3::identity()),
            Joint::fixed(pose_at::<f32>(1.0, 0.0, 0.0)),
        ],
        &[JointParent::World, JointParent::Joint(0)],
    )
    .unwrap();
    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    let mut query = CollisionQuery::<2, 0, 0, f32>::new();
    query
        .push_self_primitive(
            0,
            Primitive::Capsule {
                radius: 0.1,
                half_length: 0.2,
            },
            SE3::identity(),
        )
        .unwrap();
    query
        .push_self_primitive(1, Primitive::Sphere { radius: 0.1 }, SE3::identity())
        .unwrap();

    // Capsule along z, sphere a metre out in x: the closest point on the axis is its midpoint.
    let report = query.check(&state).unwrap();
    assert!(
        (report.minimum_clearance - 0.8).abs() < 1e-5,
        "clearance: {}",
        report.minimum_clearance
    );
}
