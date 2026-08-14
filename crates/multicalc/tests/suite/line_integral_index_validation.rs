use multicalc::error::IntegrateError;
use multicalc::vector_field::{line_integral_partial_2d, line_integral_partial_3d};

#[test]
fn line_integral_partial_2d_validates_component_index() {
    let vector_field: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| point[0]),
        &(|point: &[f64; 2]| point[1]),
    ];
    let transformations: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t), &(|t: f64| t * t)];
    let limits = [0.0, 1.0];

    for idx in 0..=1 {
        assert!(line_integral_partial_2d(
            &vector_field,
            &transformations,
            &limits,
            10,
            idx,
        )
        .is_ok());
    }

    for idx in [2, 3, usize::MAX] {
        assert_eq!(
            line_integral_partial_2d(
                &vector_field,
                &transformations,
                &limits,
                10,
                idx,
            ),
            Err(IntegrateError::IndexOutOfRange)
        );
    }
}

#[test]
fn line_integral_partial_3d_validates_component_index() {
    let vector_field: [&dyn Fn(&[f64; 3]) -> f64; 3] = [
        &(|point: &[f64; 3]| point[0]),
        &(|point: &[f64; 3]| point[1]),
        &(|point: &[f64; 3]| point[2]),
    ];
    let transformations: [&dyn Fn(f64) -> f64; 3] = [
        &(|t: f64| t),
        &(|t: f64| t * t),
        &(|t: f64| t * t * t),
    ];
    let limits = [0.0, 1.0];

    for idx in 0..=2 {
        assert!(line_integral_partial_3d(
            &vector_field,
            &transformations,
            &limits,
            10,
            idx,
        )
        .is_ok());
    }

    for idx in [3, 4, usize::MAX] {
        assert_eq!(
            line_integral_partial_3d(
                &vector_field,
                &transformations,
                &limits,
                10,
                idx,
            ),
            Err(IntegrateError::IndexOutOfRange)
        );
    }
}
