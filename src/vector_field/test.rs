use crate::vector_field::line_integral;
use crate::vector_field::flux_integral;

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
    let val = line_integral::get2D(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
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
    let val = flux_integral::get2D(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
    assert!(f64::abs(val + 0.0) < 0.01);
}