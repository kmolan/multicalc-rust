use crate::numerical_integration::mode;
use crate::numerical_integration::mode::IntegrationMethod;
use crate::numerical_integration::single_integration;
use crate::numerical_integration::double_integration;
use crate::utils::error_codes::ErrorCode;

use crate::numerical_integration::single_integration::SingleIntegrator;
use crate::numerical_integration::single_integration::Iterative;
use crate::numerical_integration::double_integration::DoubleIntegrator;
 
#[test]
fn test_booles_integration_1()
{
    //equation is 2.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    let integrator = single_integration::Iterative::with_parameters(100, mode::IterativeMethod::Booles);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_total(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_booles_integration_2()
{ 
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator = single_integration::Iterative::with_parameters(100, mode::IterativeMethod::Booles);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator.get_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = integrator.get_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = integrator.get_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_booles_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = double_integration::Iterative::with_parameters(20, mode::IterativeMethod::Booles);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_total(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_booles_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 1.0, 1.0];

    let integrator = double_integration::Iterative::with_parameters(20, mode::IterativeMethod::Booles);

    //double partial integration for first x then y, expect a value of ~1.50
    let val = integrator.get_partial(&func, [0, 1], &integration_limits, &point).unwrap();
    assert!(f64::abs(val - 1.50) < 0.00001);
}

#[test]
fn test_gauss_legendre_quadrature_integration_1()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[f64; 2] | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    let integrator = single_integration::Gaussian::with_parameters(2, mode::GaussianMethod::GaussLegendre);

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    //let val = integrator.get_total(&func, &integration_limit).unwrap();
    //assert!(f64::abs(val - 8.0) < 0.00001);
}

#[test] 
fn test_gauss_legendre_quadrature_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator = single_integration::Gaussian::with_parameters(2, mode::GaussianMethod::GaussLegendre);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator.get_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = integrator.get_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = integrator.get_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_gauss_legendre_quadrature_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total_custom(IntegrationMethod::GaussLegendre, &func, &integration_limits, 2).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_simpsons_integration_1()
{
    //equation is 2.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    let integrator = single_integration::Iterative::with_parameters(200, mode::IterativeMethod::Simpsons);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_total(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 0.05);
}

#[test]
fn test_simpsons_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator = single_integration::Iterative::with_parameters(200, mode::IterativeMethod::Simpsons);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator.get_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.05);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = integrator.get_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.05);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = integrator.get_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.05);
}

#[test]
fn test_simpsons_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = double_integration::Iterative::with_parameters(200, mode::IterativeMethod::Simpsons);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_total(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_simpsons_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 1.0, 1.0];

    let integrator = double_integration::Iterative::with_parameters(200, mode::IterativeMethod::Simpsons);

    //double partial integration for first x then y, expect a value of ~1.50
    let val = integrator.get_partial(&func, [0, 1], &integration_limits, &point).unwrap();
    assert!(f64::abs(val - 1.50) < 0.05);
}

#[test]
fn test_trapezoidal_integration_1()
{
    //equation is 2.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    let iterator = Iterative::with_parameters(100, mode::IterativeMethod::Trapezoidal);
    let val = iterator.get_total(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let iterator = Iterative::with_parameters(100, mode::IterativeMethod::Trapezoidal);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = iterator.get_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);

    
    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = iterator.get_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = iterator.get_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = double_integration::Iterative::with_parameters(10, mode::IterativeMethod::Trapezoidal);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_total(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 2.0, 3.0];

    let integrator = double_integration::Iterative::with_parameters(10, mode::IterativeMethod::Trapezoidal);

    //double partial integration for first x then y, expect a value of ~2.50
    let val = integrator.get_partial(&func, [0, 1], &integration_limits, &point).unwrap();
    assert!(f64::abs(val - 2.50) < 0.00001);
}

#[test]
fn test_error_checking_1()
{
    //equation is 2.0*x + y*z

    use crate::utils::error_codes::ErrorCode;
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [10.0, 1.0];

    let integrator = double_integration::Iterative::default();

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    //let result = single_integration::get_total_custom(IntegrationMethod::Trapezoidal, &func, &integration_limit, 10);
    //assert!(result.is_err());
    //assert!(result.unwrap_err() == ErrorCode::IntegrationLimitsIllDefined);
}

#[test]
fn test_error_checking_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];

    //expect failure because number of steps is 0
    //let result = single_integration::get_total_custom(IntegrationMethod::Trapezoidal, &func, &integration_limit, 0);
    //assert!(result.is_err());
    //assert!(result.unwrap_err() == ErrorCode::NumberOfStepsCannotBeZero);
}

#[test]
fn test_error_checking_3()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n < 2
    //let result = single_integration::get_total_custom(IntegrationMethod::GaussLegendre, &func, &integration_limit, 1);
    //assert!(result.is_err());
    //assert!(result.unwrap_err() == ErrorCode::GaussLegendreOrderOutOfRange);
}

#[test]
fn test_error_checking_4()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n > 20
    //let result = single_integration::get_total_custom(IntegrationMethod::GaussLegendre, &func, &integration_limit, 21);
    //assert!(result.is_err());
    //assert!(result.unwrap_err() == ErrorCode::GaussLegendreOrderOutOfRange);
}