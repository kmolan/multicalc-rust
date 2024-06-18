use crate::utils::error_codes::ErrorCode;
use crate::core::vector_field::divergence;
use crate::core::vector_field::line_integral;
use crate::core::vector_field::flux_integral;
use crate::core::vector_field::curl;

#[test]
fn test_line_integral_1()
{
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];

    let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];

    let integration_limit = [0.0, 6.28];

    //line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
    let val = line_integral::get_2d_custom(&vector_field_matrix, &transformation_matrix, &integration_limit, 100).unwrap();
    assert!(f64::abs(val + 6.28) < 0.01);
}

#[test]
fn test_line_integral_error_1()
{
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];

    let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];

    let integration_limit = [0.0, 6.28];

    //expect error because number of steps is zero
    let val = line_integral::get_2d_custom(&vector_field_matrix, &transformation_matrix, &integration_limit, 0);
    assert!(val.is_err());
    assert!(val.unwrap_err() == ErrorCode::NumberOfStepsCannotBeZero);
}

#[test]
fn test_line_integral_error_2()
{
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];

    let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];

    let integration_limit = [10.0, 0.0];

    //expect error because integration limits are ill-defined (lower limit higher than upper limit)
    let val = line_integral::get_2d_custom(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
    assert!(val.is_err());
    assert!(val.unwrap_err() == ErrorCode::IntegrationLimitsIllDefined);
}


#[test]
fn test_flux_integral_1()
{
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];

    let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];

    let integration_limit = [0.0, 6.28];

    //flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
    let val = flux_integral::get_2d_custom(&vector_field_matrix, &transformation_matrix, &integration_limit, 100).unwrap();
    assert!(f64::abs(val + 0.0) < 0.01);
}

#[test]
fn test_curl_2d_1()
{
    //vector field is (2*x*y, 3*cos(y))

    //x-component
    let vf_x = | args: &[f64; 2] | -> f64 
    { 
        return 2.0*args[0]*args[1];
    };

    //y-component
    let vf_y = | args: &[f64; 2] | -> f64 
    { 
        return 3.0*args[1].cos()
    };
    
    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&vf_x, &vf_y];

    let point = [1.0, 3.14];

    //curl is known to be -2*x, expect and answer of -2.0
    let val = curl::get_2d(&vector_field_matrix, &point);
    assert!(f64::abs(val + 2.0) < 0.000001); //numerical error less than 1e-6
}

#[test]
fn test_curl_3d_1()
{
    //vector field is (y, -x, 2*z)
    //x-component
    let vf_x = | args: &[f64; 3] | -> f64 
    { 
        return args[1];
    };

    //y-component
    let vf_y = | args: &[f64; 3] | -> f64 
    { 
        return -args[0];
    };

    //z-component
    let vf_z = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[2];
    };

    let vector_field_matrix: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&vf_x, &vf_y, &vf_z];
    let point = [1.0, 2.0, 3.0];

    //curl is known to be (0.0, 0.0, -2.0)
    let val = curl::get_3d(&vector_field_matrix, &point);
    //numerical error less than 1e-6
    assert!(f64::abs(val[0] - 0.0) < 0.000001);
    assert!(f64::abs(val[1] - 0.0) < 0.000001);
    assert!(f64::abs(val[2] + 2.0) < 0.000001);
}

#[test]
fn test_divergence_2d_1()
{
    //vector field is (2*x*y, 3*cos(y))
    //x-component
    let vf_x = | args: &[f64; 2] | -> f64 
    { 
        return 2.0*args[0]*args[1];
    };

    //y-component
    let vf_y = | args: &[f64; 2] | -> f64 
    { 
        return 3.0*args[1].cos()
    };
    
    let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&vf_x, &vf_y];
    let point = [1.0, 3.14];

    //divergence is known to be 2*y - 3*sin(y), expect and answer of 6.27
    let val = divergence::get_2d(&vector_field_matrix, &point);
    assert!(f64::abs(val - 6.27) < 0.01);
}

#[test]
fn test_divergence_3d_1()
{
    //vector field is (y, -x, 2*z)
    //x-component
    let vf_x = | args: &[f64; 3] | -> f64 
    { 
        return args[1];
    };

    //y-component
    let vf_y = | args: &[f64; 3] | -> f64 
    { 
        return -args[0];
    };

    //z-component
    let vf_z = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[2];
    };

    let vector_field_matrix: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&vf_x, &vf_y, &vf_z];
    let point = [0.0, 1.0, 3.0]; //the point of interest

    //diverge known to be 2.0 
    let val = divergence::get_3d(&vector_field_matrix, &point);
    assert!(f64::abs(val - 2.00) < 0.00001);
}