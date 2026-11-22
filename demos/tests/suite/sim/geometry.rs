use multicalc_demos::sim::geometry::{
    box_outline, circle_outline, rotate_points, rounded_rectangle,
};

#[test]
fn rounded_rectangle_has_four_arcs_of_points_within_the_extent() {
    let center = [3.0, 2.0];
    let half_extent = [3.0, 2.0];
    let points = rounded_rectangle(center, half_extent, 1.2, 8);
    assert_eq!(points.len(), 4 * 8);
    for point in &points {
        assert!(
            (point[0] - center[0]).abs() <= half_extent[0] + 1e-9,
            "point {point:?} outside the x-extent"
        );
        assert!(
            (point[1] - center[1]).abs() <= half_extent[1] + 1e-9,
            "point {point:?} outside the y-extent"
        );
    }
}

#[test]
fn circle_outline_closes_on_itself_at_the_radius() {
    let center = [1.0, -2.0];
    let radius = 0.75;
    let points = circle_outline(center, radius, 24);
    assert_eq!(points.len(), 25);
    assert_eq!(points[0], points[24]);
    for point in &points {
        let distance = (point[0] - center[0]).hypot(point[1] - center[1]);
        assert!(
            (distance - radius).abs() < 1e-12,
            "point {point:?} off the rim"
        );
    }
}

#[test]
fn box_outline_lists_the_corners_counter_clockwise() {
    let corners = box_outline([0.0, 0.0], [2.0, 1.0]);
    assert_eq!(
        corners,
        vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
    );
}

#[test]
fn rotate_points_turns_about_the_centre() {
    // A quarter-turn about the origin sends the point on +x to the point on +y.
    let turned = rotate_points(&[[1.0, 0.0]], [0.0, 0.0], std::f64::consts::FRAC_PI_2);
    assert!(turned[0][0].abs() < 1e-12, "x: {}", turned[0][0]);
    assert!((turned[0][1] - 1.0).abs() < 1e-12, "y: {}", turned[0][1]);
    // The centre itself does not move.
    let fixed = rotate_points(&[[0.5, 0.5]], [0.5, 0.5], 1.0);
    assert_eq!(fixed[0], [0.5, 0.5]);
}
