#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::c;
use multicalc::scalar_fn_vec;

use multicalc::vector_field::curl;
use multicalc::vector_field::divergence;
use multicalc::vector_field::flux_integral;
use multicalc::vector_field::line_integral;

use multicalc::utils::error_codes::*;

#[test]
fn test_line_integral_1() {
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|args: &[f64; 2]| -> f64 { args[1] }),
        &(|args: &[f64; 2]| -> f64 { -args[0] }),
    ];

    let transformation_matrix: [&dyn Fn(f64) -> f64; 2] = [
        &(|t: f64| -> f64 { t.cos() }),
        &(|t: f64| -> f64 { t.sin() }),
    ];

    let integration_limit = [0.0, core::f64::consts::TAU];

    //line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
    let val = line_integral::get_2d_custom(
        &vector_field_matrix,
        &transformation_matrix,
        &integration_limit,
        100,
    )
    .unwrap();
    assert!(f64::abs(val + core::f64::consts::TAU) < 0.01);
}

#[test]
fn test_line_integral_error_1() {
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|args: &[f64; 2]| -> f64 { args[1] }),
        &(|args: &[f64; 2]| -> f64 { -args[0] }),
    ];

    let transformation_matrix: [&dyn Fn(f64) -> f64; 2] = [
        &(|t: f64| -> f64 { t.cos() }),
        &(|t: f64| -> f64 { t.sin() }),
    ];

    let integration_limit = [0.0, core::f64::consts::TAU];

    //expect error because number of steps is zero
    let val = line_integral::get_2d_custom(
        &vector_field_matrix,
        &transformation_matrix,
        &integration_limit,
        0,
    );
    assert!(val.is_err());
    assert!(val.unwrap_err() == CalcError::IterationsZero);
}

#[test]
fn test_line_integral_error_2() {
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|args: &[f64; 2]| -> f64 { args[1] }),
        &(|args: &[f64; 2]| -> f64 { -args[0] }),
    ];

    let transformation_matrix: [&dyn Fn(f64) -> f64; 2] = [
        &(|t: f64| -> f64 { t.cos() }),
        &(|t: f64| -> f64 { t.sin() }),
    ];

    let integration_limit = [10.0, 0.0];

    //expect error because integration limits are ill-defined (lower limit higher than upper limit)
    let val = line_integral::get_2d_custom(
        &vector_field_matrix,
        &transformation_matrix,
        &integration_limit,
        100,
    );
    assert!(val.is_err());
    assert!(val.unwrap_err() == CalcError::IntegrationLimitsIllDefined);
}

#[test]
fn test_flux_integral_1() {
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|args: &[f64; 2]| -> f64 { args[1] }),
        &(|args: &[f64; 2]| -> f64 { -args[0] }),
    ];

    let transformation_matrix: [&dyn Fn(f64) -> f64; 2] = [
        &(|t: f64| -> f64 { t.cos() }),
        &(|t: f64| -> f64 { t.sin() }),
    ];

    let integration_limit = [0.0, core::f64::consts::TAU];

    //flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
    let val = flux_integral::get_2d_custom(
        &vector_field_matrix,
        &transformation_matrix,
        &integration_limit,
        100,
    )
    .unwrap();
    assert!(f64::abs(val + 0.0) < 0.01);
}

#[test]
fn test_curl_2d_1() {
    //vector field is (2*x*y, 3*cos(y)); curl is known to be -2*x, so -2.0 at x = 1
    let vf = scalar_fn_vec!(|v: &[f64; 2]| [c(2.0) * v[0] * v[1], c(3.0) * v[1].cos()]);
    let point = [1.0, core::f64::consts::PI];

    let val = curl::get_2d(AutoDiffMulti::default(), &vf, &point).unwrap();
    assert!(f64::abs(val + 2.0) < 1e-12);
}

#[test]
fn test_curl_3d_1() {
    //vector field is (y, -x, 2*z); curl is known to be (0, 0, -2)
    let vf = scalar_fn_vec!(|v: &[f64; 3]| [v[1], -v[0], c(2.0) * v[2]]);
    let point = [1.0, 2.0, 3.0];

    let val = curl::get_3d(AutoDiffMulti::default(), &vf, &point).unwrap();
    assert!(f64::abs(val[0]) < 1e-12);
    assert!(f64::abs(val[1]) < 1e-12);
    assert!(f64::abs(val[2] + 2.0) < 1e-12);
}

#[test]
fn test_divergence_2d_1() {
    //vector field is (2*x*y, 3*cos(y)); divergence is 2*y - 3*sin(y), which is 2*pi at y = pi
    let vf = scalar_fn_vec!(|v: &[f64; 2]| [c(2.0) * v[0] * v[1], c(3.0) * v[1].cos()]);
    let point = [1.0, core::f64::consts::PI];

    let val = divergence::get_2d(AutoDiffMulti::default(), &vf, &point).unwrap();
    assert!(f64::abs(val - core::f64::consts::TAU) < 1e-12);
}

#[test]
fn test_divergence_3d_1() {
    //vector field is (y, -x, 2*z); divergence is known to be 2.0
    let vf = scalar_fn_vec!(|v: &[f64; 3]| [v[1], -v[0], c(2.0) * v[2]]);
    let point = [0.0, 1.0, 3.0];

    let val = divergence::get_3d(AutoDiffMulti::default(), &vf, &point).unwrap();
    assert!(f64::abs(val - 2.0) < 1e-12);
}

#[test]
fn test_line_integral_3d_helix() {
    //helix r(t) = (cos t, sin t, t), with a real z-component (regression for the 3D z typo)
    //vector field is (0, 0, z); the line integral is ∫ z dz = ∫_0^{2π} t dt = 2π²
    let vector_field_matrix: [&dyn Fn(&[f64; 3]) -> f64; 3] = [
        &(|_: &[f64; 3]| -> f64 { 0.0 }),
        &(|_: &[f64; 3]| -> f64 { 0.0 }),
        &(|args: &[f64; 3]| -> f64 { args[2] }),
    ];

    let transformation_matrix: [&dyn Fn(f64) -> f64; 3] = [
        &(|t: f64| -> f64 { t.cos() }),
        &(|t: f64| -> f64 { t.sin() }),
        &(|t: f64| -> f64 { t }),
    ];

    let two_pi = 2.0 * core::f64::consts::PI;
    let integration_limit = [0.0, two_pi];

    let val = line_integral::get_3d_custom(
        &vector_field_matrix,
        &transformation_matrix,
        &integration_limit,
        100,
    )
    .unwrap();

    let expected = 2.0 * core::f64::consts::PI * core::f64::consts::PI;
    assert!(f64::abs(val - expected) < 1e-6);
}

#[test]
fn test_line_integral_negative_limits() {
    //a [-2.0, 1.0] parameter range must be accepted (regression for the old .abs() check)
    //vector field is (y, x), curve is x = t, y = t. The line integral is
    //∫ y dx + x dy = ∫ t dt + ∫ t dt = 2 * (1/2 - 2) = -3
    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|args: &[f64; 2]| -> f64 { args[1] }),
        &(|args: &[f64; 2]| -> f64 { args[0] }),
    ];

    let transformation_matrix: [&dyn Fn(f64) -> f64; 2] =
        [&(|t: f64| -> f64 { t }), &(|t: f64| -> f64 { t })];

    let integration_limit = [-2.0, 1.0];

    let val = line_integral::get_2d_custom(
        &vector_field_matrix,
        &transformation_matrix,
        &integration_limit,
        100,
    )
    .unwrap();

    assert!(f64::abs(val - (-3.0)) < 1e-9);
}

#[test]
fn test_divergence_3d_f32() {
    //field (y, -x, 2z); divergence is 0 + 0 + 2 = 2. The same authored field evaluates at f32.
    let vf = scalar_fn_vec!(|v: &[f64; 3]| [v[1], -v[0], c(2.0) * v[2]]);
    let point = [0.0_f32, 1.0, 3.0];

    let val = divergence::get_3d(AutoDiffMulti::<f32>::default(), &vf, &point).unwrap();
    assert!(f32::abs(val - 2.0) < 1e-6, "got {val}");
}
