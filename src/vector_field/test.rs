use crate::vector_field::divergence;
use crate::vector_field::line_integral;
use crate::vector_field::flux_integral;
use crate::vector_field::curl;

#[test]
fn test_line_integral_1()
{
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [Box<dyn Fn(&f64, &f64) -> f64>; 2] = [Box::new(|_:&f64, y:&f64|-> f64 { *y }), Box::new(|x:&f64, _:&f64|-> f64 { -x })];

    let transformation_matrix: [Box<dyn Fn(&f64) -> f64>; 2] = [Box::new(|t:&f64|->f64 { t.cos() }), Box::new(|t:&f64|->f64 { t.sin() })];

    let integration_limit = [0.0, 6.28];

    //line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
    let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
    println!("{}", val);
    assert!(f64::abs(val + 6.28) < 0.01);
}


#[test]
fn test_flux_integral_1()
{
    //vector field is (y, -x)
    //curve is a unit circle, defined by (Cos(t), Sin(t))
    //limit t goes from 0->2*pi

    let vector_field_matrix: [Box<dyn Fn(&f64, &f64) -> f64>; 2] = [Box::new(|_:&f64, y:&f64|-> f64 { *y }), Box::new(|x:&f64, _:&f64|-> f64 { -x })];

    let transformation_matrix: [Box<dyn Fn(&f64) -> f64>; 2] = [Box::new(|t:&f64|->f64 { t.cos() }), Box::new(|t:&f64|->f64 { t.sin() })];

    let integration_limit = [0.0, 6.28];

    //flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
    let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
    assert!(f64::abs(val + 0.0) < 0.01);
}

#[test]
fn test_curl_2d_1()
{
    //vector field is (2*x*y, 3*cos(y))
    let vector_field_matrix: [Box<dyn Fn(&[f64; 2]) -> f64>; 2] = [Box::new(|args: &[f64; 2]|-> f64 { 2.0*args[0]*args[1] }), Box::new(|args: &[f64; 2]|-> f64 { 3.0*args[1].cos() })];

    let point = [1.0, 3.14];

    //curl is known to be -2*x, expect and answer of -2.0
    let val = curl::get_2d(&vector_field_matrix, &point);
    assert!(f64::abs(val + 2.0) < 0.000001); //numerical error less than 1e-6
}

#[test]
fn test_curl_3d_1()
{
    //vector field is (y, -x, 2*z)
    let vector_field_matrix: [Box<dyn Fn(&[f64; 3]) -> f64>; 3] = [Box::new(|args: &[f64; 3]|-> f64 { args[1] }), Box::new(|args: &[f64; 3]|-> f64 { -args[0]}), Box::new(|args: &[f64; 3]|-> f64 { 2.0*args[2]})];
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
    let vector_field_matrix: [Box<dyn Fn(&[f64; 2]) -> f64>; 2] = [Box::new(|args: &[f64; 2]|-> f64 { 2.0*args[0]*args[1] }), Box::new(|args: &[f64; 2]|-> f64 { 3.0*args[1].cos() })];

    let point = [1.0, 3.14];

    //divergence is known to be 2*y - 3*sin(y), expect and answer of 6.27
    let val = divergence::get_2d(&vector_field_matrix, &point);
    assert!(f64::abs(val - 6.27) < 0.01);
}

#[test]
fn test_divergence_3d_1()
{
    //vector field is (y, -x, 2*z)
    let vector_field_matrix: [Box<dyn Fn(&[f64; 3]) -> f64>; 3] = [Box::new(|args: &[f64; 3]|-> f64 { args[1] }), Box::new(|args: &[f64; 3]|-> f64 { -args[0]}), Box::new(|args: &[f64; 3]|-> f64 { 2.0*args[2]})];

    let point = [0.0, 1.0, 3.0]; //the point of interest

    //diverge known to be 2.0 
    let val = divergence::get_3d(&vector_field_matrix, &point);
    assert!(f64::abs(val - 2.00) < 0.00001);
}

