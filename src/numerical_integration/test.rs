use crate::numerical_integration::mode::*;
use crate::utils::error_codes::ErrorCode;
use crate::numerical_integration::iterative_integration;
use crate::numerical_integration::gaussian_integration;
use crate::numerical_integration::integrator::*;
 
#[test]
fn test_booles_integration_1()
{
    //equation is 2.0*x
    let func = | args: f64 | -> f64 
    { 
        return 2.0*args;
    };

    let integration_limit = [0.0, 2.0];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(100, IterativeMethod::Booles);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
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

    let integrator = iterative_integration::MultiVariableSolver::from_parameters(100, IterativeMethod::Booles);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator.get_single_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = integrator.get_single_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = integrator.get_single_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_booles_integration_3()
{
    //equation is 6.0*x
    let func = | args: f64 | -> f64 
    { 
        return 6.0*args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(20, IterativeMethod::Booles);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
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

    let integrator = iterative_integration::MultiVariableSolver::from_parameters(20, IterativeMethod::Booles);

    //double partial integration for first x then y, expect a value of ~1.50
    let val = integrator.get_double_partial(&func, [0, 1], &integration_limits, &point).unwrap();
    assert!(f64::abs(val - 1.50) < 0.00001);
}

#[test]
fn test_gauss_legendre_quadrature_integration_1()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: f64 | -> f64 
    { 
        return 4.0*args*args*args - 3.0*args*args;
    };

    let integration_limit = [0.0, 2.0];

    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(4, GaussianQuadratureMethod::GaussLaguerre);

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    std::println!("{}", val);
    assert!(f64::abs(val - 8.0) < 1e-14);
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

    let integrator = gaussian_integration::MultiVariableSolver::from_parameters(2, GaussianQuadratureMethod::GaussLegendre);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator.get_single_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 1e-14);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = integrator.get_single_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 1e-14);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = integrator.get_single_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 1e-14);
}

#[test]
fn test_gauss_legendre_quadrature_integration_3()
{
    //equation is 6.0*x
    let func = | args: f64 | -> f64 
    { 
        return 6.0*args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(2, GaussianQuadratureMethod::GaussLegendre);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 1e-14);
}

#[test]
fn test_simpsons_integration_1()
{
    //equation is 2.0*x
    let func = | args: f64 | -> f64 
    { 
        return 2.0*args;
    };

    let integration_limit = [0.0, 2.0];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(200, IterativeMethod::Simpsons);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
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

    let integrator = iterative_integration::MultiVariableSolver::from_parameters(200, IterativeMethod::Simpsons);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator.get_single_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.05);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = integrator.get_single_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.05);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = integrator.get_single_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.05);
}

#[test]
fn test_simpsons_integration_3()
{
    //equation is 6.0*x
    let func = | args: f64 | -> f64 
    { 
        return 6.0*args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(200, IterativeMethod::Simpsons);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
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

    let integrator = iterative_integration::MultiVariableSolver::from_parameters(200, IterativeMethod::Simpsons);

    //double partial integration for first x then y, expect a value of ~1.50
    let val = integrator.get_double_partial(&func, [0, 1], &integration_limits, &point).unwrap();
    assert!(f64::abs(val - 1.50) < 0.05);
}

#[test]
fn test_trapezoidal_integration_1()
{
    //equation is 2.0*x
    let func = | args: f64 | -> f64 
    { 
        return 2.0*args;
    };

    let integration_limit = [0.0, 2.0];

    let iterator = iterative_integration::SingleVariableSolver::from_parameters(100, IterativeMethod::Trapezoidal);
    let val = iterator.get_single(&func, &integration_limit).unwrap();

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

    let iterator = iterative_integration::MultiVariableSolver::from_parameters(100, IterativeMethod::Trapezoidal);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = iterator.get_single_partial(&func, 0, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);

    
    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = iterator.get_single_partial(&func, 1, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = iterator.get_single_partial(&func, 2, &integration_limit, &point).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_3()
{
    //equation is 6.0*x
    let func = | args: f64 | -> f64 
    { 
        return 6.0*args;
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(10, IterativeMethod::Trapezoidal);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
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

    let integrator = iterative_integration::MultiVariableSolver::from_parameters(10, IterativeMethod::Trapezoidal);

    //double partial integration for first x then y, expect a value of ~2.50
    let val = integrator.get_double_partial(&func, [0, 1], &integration_limits, &point).unwrap();
    assert!(f64::abs(val - 2.50) < 0.00001);
}

#[test]
fn test_error_checking_1()
{
    //equation is 2.0*x
    let func = | args: f64 | -> f64 
    { 
        return 2.0*args;
    };

    let integration_limit = [10.0, 1.0];

    let integrator = iterative_integration::SingleVariableSolver::default();

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::IntegrationLimitsIllDefined);
}

#[test]
fn test_error_checking_2()
{
    //equation is 2.0*x
    let func = | args: f64 | -> f64 
    { 
        return 2.0*args;
    };

    let integration_limit = [0.0, 1.0];

    let integrator = iterative_integration::SingleVariableSolver::from_parameters(0, IterativeMethod::Booles);

    //expect failure because number of steps is 0
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::NumberOfStepsCannotBeZero);
}

//TODO: add more tests

#[test]
fn test_error_checking_3()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: f64 | -> f64 
    { 
        return 4.0*args*args*args - 3.0*args*args;
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n < 1
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(0, GaussianQuadratureMethod::GaussLegendre);
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::GaussianQuadratureOrderOutOfRange);
}

#[test]
fn test_error_checking_4()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: f64 | -> f64 
    { 
        return 4.0*args*args*args - 3.0*args*args;
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n > 30
    let integrator = gaussian_integration::SingleVariableSolver::from_parameters(31, GaussianQuadratureMethod::GaussLegendre);
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::GaussianQuadratureOrderOutOfRange);
}