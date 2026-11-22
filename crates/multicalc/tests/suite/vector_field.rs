#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::scalar::constant;
use multicalc::scalar_fn_vec;

use multicalc::vector_field::{
    curl_2d, curl_3d, divergence_2d, divergence_3d, flux_integral_2d_custom,
    flux_integral_3d_custom, line_integral_2d_custom, line_integral_3d_custom,
    line_integral_partial_2d, line_integral_partial_3d,
};

use multicalc::error::IntegrateError;

#[test]
fn line_integral_of_a_rotational_field_around_the_unit_circle() {
    //vector field is (y, -x)
    //curve is defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let field_components: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| -> f64 { point[1] }),
        &(|point: &[f64; 2]| -> f64 { -point[0] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 2] = [
        &(|parameter: f64| -> f64 { parameter.cos() }),
        &(|parameter: f64| -> f64 { parameter.sin() }),
    ];

    let integration_limit = [0.0, core::f64::consts::TAU];
    let total_iterations = 100;

    //expect an answer of -2.0*pi
    let circulation = line_integral_2d_custom(
        &field_components,
        &curve,
        &integration_limit,
        total_iterations,
    )
    .unwrap();
    assert!(f64::abs(circulation + core::f64::consts::TAU) < 0.01);
}

#[test]
fn line_integral_rejects_a_zero_interval_count() {
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let field_components: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| -> f64 { point[1] }),
        &(|point: &[f64; 2]| -> f64 { -point[0] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 2] = [
        &(|parameter: f64| -> f64 { parameter.cos() }),
        &(|parameter: f64| -> f64 { parameter.sin() }),
    ];

    let integration_limit = [0.0, core::f64::consts::TAU];

    let result = line_integral_2d_custom(&field_components, &curve, &integration_limit, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err() == IntegrateError::IterationsZero);
}

#[test]
fn line_integral_rejects_reversed_limits() {
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))

    let field_components: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| -> f64 { point[1] }),
        &(|point: &[f64; 2]| -> f64 { -point[0] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 2] = [
        &(|parameter: f64| -> f64 { parameter.cos() }),
        &(|parameter: f64| -> f64 { parameter.sin() }),
    ];

    //lower limit higher than upper limit
    let integration_limit = [10.0, 0.0];

    let result = line_integral_2d_custom(&field_components, &curve, &integration_limit, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err() == IntegrateError::LimitsIllDefined);
}

#[test]
fn flux_of_a_rotational_field_through_the_unit_circle_is_zero() {
    //vector field is (y, -x)
    //curve is defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let field_components: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| -> f64 { point[1] }),
        &(|point: &[f64; 2]| -> f64 { -point[0] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 2] = [
        &(|parameter: f64| -> f64 { parameter.cos() }),
        &(|parameter: f64| -> f64 { parameter.sin() }),
    ];

    let integration_limit = [0.0, core::f64::consts::TAU];
    let total_iterations = 100;

    let flux = flux_integral_2d_custom(
        &field_components,
        &curve,
        &integration_limit,
        total_iterations,
    )
    .unwrap();
    assert!(f64::abs(flux + 0.0) < 0.01);
}

#[test]
fn flux_along_a_helix_matches_its_closed_form() {
    // Curve r(t) = (cos(t), sin(t), t)
    // Vector field = (0, 0, z)
    //
    // Flux calculation:
    // partial_x = 0
    // partial_y = 0
    // partial_z = ∫ z dz = ∫ t dt = 2*pi^2
    //
    // Flux = partial_x - partial_y - partial_z = -2*pi^2

    let field_components: [&dyn Fn(&[f64; 3]) -> f64; 3] = [
        &(|_: &[f64; 3]| -> f64 { 0.0 }),
        &(|_: &[f64; 3]| -> f64 { 0.0 }),
        &(|point: &[f64; 3]| -> f64 { point[2] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 3] = [
        &(|parameter: f64| -> f64 { parameter.cos() }),
        &(|parameter: f64| -> f64 { parameter.sin() }),
        &(|parameter: f64| -> f64 { parameter }),
    ];

    let two_pi = 2.0 * core::f64::consts::PI;
    let integration_limit = [0.0, two_pi];
    let total_iterations = 100;

    let flux = flux_integral_3d_custom(
        &field_components,
        &curve,
        &integration_limit,
        total_iterations,
    )
    .unwrap();

    let expected = -2.0 * core::f64::consts::PI * core::f64::consts::PI;

    assert!(f64::abs(flux - expected) < 1e-6);
}

#[test]
fn curl_2d_matches_its_closed_form() {
    //vector field is (2*x*y, 3*cos(y)); curl is known to be -2*x, so -2.0 at x = 1
    let vector_field = scalar_fn_vec!(|point: &[f64; 2]| [
        constant(2.0) * point[0] * point[1],
        constant(3.0) * point[1].cos()
    ]);
    let point = [1.0, core::f64::consts::PI];

    let curl = curl_2d(AutoDiffMulti::default(), &vector_field, &point).unwrap();
    assert!(f64::abs(curl + 2.0) < 1e-12);
}

#[test]
fn curl_3d_matches_its_closed_form() {
    //vector field is (y, -x, 2*z); curl is known to be (0, 0, -2)
    let vector_field =
        scalar_fn_vec!(|point: &[f64; 3]| [point[1], -point[0], constant(2.0) * point[2]]);
    let point = [1.0, 2.0, 3.0];

    let curl = curl_3d(AutoDiffMulti::default(), &vector_field, &point).unwrap();
    assert!(f64::abs(curl[0]) < 1e-12);
    assert!(f64::abs(curl[1]) < 1e-12);
    assert!(f64::abs(curl[2] + 2.0) < 1e-12);
}

#[test]
fn divergence_2d_matches_its_closed_form() {
    //vector field is (2*x*y, 3*cos(y)); divergence is 2*y - 3*sin(y), which is 2*pi at y = pi
    let vector_field = scalar_fn_vec!(|point: &[f64; 2]| [
        constant(2.0) * point[0] * point[1],
        constant(3.0) * point[1].cos()
    ]);
    let point = [1.0, core::f64::consts::PI];

    let divergence = divergence_2d(AutoDiffMulti::default(), &vector_field, &point).unwrap();
    assert!(f64::abs(divergence - core::f64::consts::TAU) < 1e-12);
}

#[test]
fn divergence_3d_matches_its_closed_form() {
    //vector field is (y, -x, 2*z); divergence is known to be 2.0
    let vector_field =
        scalar_fn_vec!(|point: &[f64; 3]| [point[1], -point[0], constant(2.0) * point[2]]);
    let point = [0.0, 1.0, 3.0];

    let divergence = divergence_3d(AutoDiffMulti::default(), &vector_field, &point).unwrap();
    assert!(f64::abs(divergence - 2.0) < 1e-12);
}

#[test]
fn line_integral_along_a_helix_matches_its_closed_form() {
    //helix r(t) = (cos t, sin t, t), with a real z-component (regression for the 3D z typo)
    //vector field is (0, 0, z); the line integral is ∫ z dz = ∫_0^{2π} t dt = 2π²
    let field_components: [&dyn Fn(&[f64; 3]) -> f64; 3] = [
        &(|_: &[f64; 3]| -> f64 { 0.0 }),
        &(|_: &[f64; 3]| -> f64 { 0.0 }),
        &(|point: &[f64; 3]| -> f64 { point[2] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 3] = [
        &(|parameter: f64| -> f64 { parameter.cos() }),
        &(|parameter: f64| -> f64 { parameter.sin() }),
        &(|parameter: f64| -> f64 { parameter }),
    ];

    let two_pi = 2.0 * core::f64::consts::PI;
    let integration_limit = [0.0, two_pi];
    let total_iterations = 100;

    let circulation = line_integral_3d_custom(
        &field_components,
        &curve,
        &integration_limit,
        total_iterations,
    )
    .unwrap();

    let expected = 2.0 * core::f64::consts::PI * core::f64::consts::PI;
    assert!(f64::abs(circulation - expected) < 1e-6);
}

#[test]
fn line_integral_accepts_a_negative_lower_limit() {
    //a [-2.0, 1.0] parameter range must be accepted (regression for the old .abs() check)
    //vector field is (y, x), curve is x = t, y = t. The line integral is
    //∫ y dx + x dy = ∫ t dt + ∫ t dt = 2 * (1/2 - 2) = -3
    let field_components: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| -> f64 { point[1] }),
        &(|point: &[f64; 2]| -> f64 { point[0] }),
    ];

    let curve: [&dyn Fn(f64) -> f64; 2] = [
        &(|parameter: f64| -> f64 { parameter }),
        &(|parameter: f64| -> f64 { parameter }),
    ];

    let integration_limit = [-2.0, 1.0];
    let total_iterations = 100;

    let circulation = line_integral_2d_custom(
        &field_components,
        &curve,
        &integration_limit,
        total_iterations,
    )
    .unwrap();

    let expected = -3.0;
    assert!(f64::abs(circulation - expected) < 1e-9);
}

#[test]
fn line_integral_partial_2d_rejects_invalid_index() {
    use std::f64::consts::TAU;

    //vector field is (y, -x).
    let field_x = |point: &[f64; 2]| point[1]; // P(x, y) = y
    let field_y = |point: &[f64; 2]| -point[0]; // Q(x, y) = -x

    let vector_field: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&field_x, &field_y];

    // r(t) = (cos(t), sin(t))
    let x_of_t = |t: f64| t.cos(); // x(t) = cos(t)
    let y_of_t = |t: f64| t.sin(); // y(t) = sin(t)

    let transformations: [&dyn Fn(f64) -> f64; 2] = [&x_of_t, &y_of_t];

    let integration_limit = [0.0, TAU];
    let total_iterations = 10_000;
    let idx = 2;

    let result = line_integral_partial_2d(
        &vector_field,
        &transformations,
        &integration_limit,
        total_iterations,
        idx,
    );
    assert!(result.unwrap_err() == IntegrateError::IndexOutOfRange);
}

#[test]
fn line_integral_partial_2d_accepts_valid_index() {
    use std::f64::consts::{PI, TAU};

    //vector field is (y, -x).
    let field_x = |point: &[f64; 2]| point[1]; // P(x, y) = y
    let field_y = |point: &[f64; 2]| -point[0]; // Q(x, y) = -x

    let vector_field: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&field_x, &field_y];

    // r(t) = (cos(t), sin(t))
    let x_of_t = |t: f64| t.cos(); // x(t) = cos(t)
    let y_of_t = |t: f64| t.sin(); // y(t) = sin(t)

    let transformations: [&dyn Fn(f64) -> f64; 2] = [&x_of_t, &y_of_t];

    let integration_limit = [0.0, TAU];
    let total_iterations = 10_000;
    let idx = 1;

    let result = line_integral_partial_2d(
        &vector_field,
        &transformations,
        &integration_limit,
        total_iterations,
        idx,
    );

    let expected = -PI;
    assert!(f64::abs(result.unwrap() - expected) < 1e-6); // -PI - (-PI) == -PI + PI == 0
}

#[test]
fn line_integral_partial_3d_accepts_valid_index() {
    use std::f64::consts::TAU;

    //vector field is (y, -x, 2z).
    let field_x = |point: &[f64; 3]| point[1]; // P(x, y) = y
    let field_y = |point: &[f64; 3]| -point[0]; // Q(x, y) = -x
    let field_z = |point: &[f64; 3]| 2.0 * point[0]; // R(x, y) = 2z

    let vector_field: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&field_x, &field_y, &field_z];

    // r(t) = (cos(t), sin(t), 0.0)
    let x_of_t = |t: f64| t.cos(); // x(t) = cos(t)
    let y_of_t = |t: f64| t.sin(); // y(t) = sin(t)
    let z_of_t = |_: f64| 0.0; // z(t) = 0.0

    let transformations: [&dyn Fn(f64) -> f64; 3] = [&x_of_t, &y_of_t, &z_of_t];

    let integration_limit = [0.0, TAU];
    let total_iterations = 10_000;
    let idx = 2;

    let result = line_integral_partial_3d(
        &vector_field,
        &transformations,
        &integration_limit,
        total_iterations,
        idx,
    );

    let expected = 0.0;
    assert!(f64::abs(result.unwrap() - expected) < 1e-9);
}

#[test]
fn line_integral_partial_3d_rejects_invalid_index() {
    use std::f64::consts::TAU;

    //vector field is (y, -x, 2z).
    let field_x = |point: &[f64; 3]| point[1]; // P(x, y) = y
    let field_y = |point: &[f64; 3]| -point[0]; // Q(x, y) = -x
    let field_z = |point: &[f64; 3]| 2.0 * point[0]; // R(x, y) = 2z

    let vector_field: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&field_x, &field_y, &field_z];

    // r(t) = (cos(t), sin(t), 0.0)
    let x_of_t = |t: f64| t.cos(); // x(t) = cos(t)
    let y_of_t = |t: f64| t.sin(); // y(t) = sin(t)
    let z_of_t = |_: f64| 0.0; // z(t) = 0.0

    let transformations: [&dyn Fn(f64) -> f64; 3] = [&x_of_t, &y_of_t, &z_of_t];

    let integration_limit = [0.0, TAU];
    let total_iterations = 10_000;
    let idx = 3;

    let result = line_integral_partial_3d(
        &vector_field,
        &transformations,
        &integration_limit,
        total_iterations,
        idx,
    );
    assert!(result.unwrap_err() == IntegrateError::IndexOutOfRange);
}

#[test]
fn divergence_3d_matches_its_closed_form_at_f32() {
    //field (y, -x, 2z); divergence is 0 + 0 + 2 = 2. The same authored field evaluates at f32.
    let vector_field =
        scalar_fn_vec!(|point: &[f64; 3]| [point[1], -point[0], constant(2.0) * point[2]]);
    let point = [0.0_f32, 1.0, 3.0];

    let divergence = divergence_3d(AutoDiffMulti::<f32>::default(), &vector_field, &point).unwrap();
    assert!(f32::abs(divergence - 2.0) < 1e-6, "got {divergence}");
}
