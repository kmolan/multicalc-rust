use crate::numerical_integration::mode::IntegrationMethod;
use crate::numerical_integration::single_integration;
use crate::numerical_integration::double_integration;
 
#[test]
fn test_booles_integration_1()
{
    //equation is 2.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_interval = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Booles, &func, &integration_interval, 100);
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

    let integration_interval = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 0, &integration_interval, &point, 100);
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_interval = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 1, &integration_interval, &point, 100);
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_interval = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 2, &integration_interval, &point, 100);
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

    let integration_intervals = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Booles, &func, &integration_intervals, 20);
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

    let integration_intervals = [[0.0, 1.0], [0.0, 1.0]];
    let point = vec![1.0, 1.0, 1.0];

    //double partial integration for first x then y, expect a value of ~1.50
    let val = double_integration::get_partial(IntegrationMethod::Booles, &func, [0, 1], &integration_intervals, &point, 20);
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

    let integration_interval = [0.0, 2.0];

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    let val = single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_interval, 2);
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

    let integration_interval = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 0, &integration_interval, &point, 2);
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_interval = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 1, &integration_interval, &point, 2);
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_interval = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 2, &integration_interval, &point, 2);
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

    let integration_intervals = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_intervals, 2);
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

    let integration_interval = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_interval, 200);
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

    let integration_interval = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 0, &integration_interval, &point, 200);
    assert!(f64::abs(val - 7.0) < 0.05);


    let integration_interval = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 1, &integration_interval, &point, 200);
    assert!(f64::abs(val - 10.0) < 0.05);


    let integration_interval = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 2, &integration_interval, &point, 200);
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

    let integration_intervals = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_intervals, 200);
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

    let integration_intervals = [[0.0, 1.0], [0.0, 1.0]];
    let point = vec![1.0, 1.0, 1.0];

    //double partial integration for first x then y, expect a value of ~1.50
    let val = double_integration::get_partial(IntegrationMethod::Simpsons, &func, [0, 1], &integration_intervals, &point, 200);
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

    let integration_interval = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_interval, 10);
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

    let integration_interval = [0.0, 1.0];
    let point = vec![1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 0, &integration_interval, &point, 10);
    assert!(f64::abs(val - 7.0) < 0.00001);

    
    let integration_interval = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 1, &integration_interval, &point, 10);
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_interval = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 2, &integration_interval, &point, 10);
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

    let integration_intervals = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_intervals, 10);
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

    let integration_intervals = [[0.0, 1.0], [0.0, 1.0]];
    let point = vec![1.0, 2.0, 3.0];

    //double partial integration for first x then y, expect a value of ~2.50
    let val = double_integration::get_partial(IntegrationMethod::Trapezoidal, &func, [0, 1], &integration_intervals, &point, 10);
    assert!(f64::abs(val - 2.50) < 0.00001);
}

#[test]
fn test_error_checking_1()
{
    //equation is 2.0*x + y*z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_interval = [10.0, 1.0];

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_interval, 10));
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

    let integration_interval = [0.0, 1.0];

    //expect failure because number of steps is 0
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_interval, 0));
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

    let integration_interval = [0.0, 2.0];

    //Gauss Legendre not valid for n < 2
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_interval, 1));
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

    let integration_interval = [0.0, 2.0];

    //Gauss Legendre not valid for n > 20
    let result = std::panic::catch_unwind(||single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_interval, 21));
    assert!(result.is_err());
}