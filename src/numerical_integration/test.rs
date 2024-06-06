use crate::numerical_integration::mode::IntegrationMethod;
use crate::numerical_integration::single_integration;
use crate::numerical_integration::double_integration;
use crate::numerical_integration::line_integral;
use crate::numerical_integration::flux_integral;
 
#[test]
fn test_booles_integration_1()
{
    //equation is 2.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Booles, &func, &integration_limit, 100);
    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_booles_integration_2()
{ 
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 0, &integration_limit, &point, 100);
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 1, &integration_limit, &point, 100);
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 2, &integration_limit, &point, 100);
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_booles_integration_3()
{
    //equation is 6.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Booles, &func, &integration_limits, 20);
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_booles_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = vec![1.0, 1.0, 1.0];

    //double partial integration for first x then y, expect a value of ~1.50
    let val = double_integration::get_partial(IntegrationMethod::Booles, &func, [0, 1], &integration_limits, &point, 20);
    assert!(f64::abs(val - 1.50) < 0.00001);
}

#[test] 
fn test_gauss_legendre_quadrature_integration_1()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    let val = single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 2);
    assert!(f64::abs(val - 8.0) < 0.00001);
}

#[test] 
fn test_gauss_legendre_quadrature_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 0, &integration_limit, &point, 2);
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 1, &integration_limit, &point, 2);
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 2, &integration_limit, &point, 2);
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_gauss_legendre_quadrature_integration_3()
{
    //equation is 6.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limits, 2);
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_simpsons_integration_1()
{
    //equation is 2.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_limit, 200);
    assert!(f64::abs(val - 4.0) < 0.05);
}

#[test]
fn test_simpsons_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 0, &integration_limit, &point, 200);
    assert!(f64::abs(val - 7.0) < 0.05);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 1, &integration_limit, &point, 200);
    assert!(f64::abs(val - 10.0) < 0.05);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 2, &integration_limit, &point, 200);
    assert!(f64::abs(val - 15.0) < 0.05);
}

#[test]
fn test_simpsons_integration_3()
{
    //equation is 6.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_limits, 200);
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_simpsons_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = vec![1.0, 1.0, 1.0];

    //double partial integration for first x then y, expect a value of ~1.50
    let val = double_integration::get_partial(IntegrationMethod::Simpsons, &func, [0, 1], &integration_limits, &point, 200);
    assert!(f64::abs(val - 1.50) < 0.05);
}

#[test]
fn test_trapezoidal_integration_1()
{
    //equation is 2.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 10);
    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 0, &integration_limit, &point, 10);
    assert!(f64::abs(val - 7.0) < 0.00001);

    
    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 1, &integration_limit, &point, 10);
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 2, &integration_limit, &point, 10);
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_3()
{
    //equation is 6.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limits, 10);
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = vec![1.0, 2.0, 3.0];

    //double partial integration for first x then y, expect a value of ~2.50
    let val = double_integration::get_partial(IntegrationMethod::Trapezoidal, &func, [0, 1], &integration_limits, &point, 10);
    assert!(f64::abs(val - 2.50) < 0.00001);
}

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


#[test]
fn test_error_checking_1()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [10.0, 1.0];

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 10));
    assert!(result.is_err());
}

#[test]
fn test_error_checking_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];

    //expect failure because number of steps is 0
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 0));
    assert!(result.is_err());
}

#[test]
fn test_error_checking_3()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n < 2
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 1));
    assert!(result.is_err());
}

#[test]
fn test_error_checking_4()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n > 20
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 21));
    assert!(result.is_err());
}