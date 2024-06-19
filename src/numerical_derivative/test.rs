use crate::numerical_derivative::hessian::Hessian;
use crate::numerical_derivative::jacobian::Jacobian;
use crate::numerical_derivative::mode::*;
use crate::utils::error_codes::ErrorCode;
use crate::numerical_derivative::derivator::Derivator;
use crate::numerical_derivative::fixed_step::FixedStep;

#[cfg(feature = "heap")]
use std::{boxed::Box, vec::Vec};


#[test]
fn test_single_derivative_forward_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Forward);

    let val = derivator.get_single_total(&func, 2.0).unwrap();
    assert!(f64::abs(val - 2.0) < 0.001);
}

#[test]
fn test_single_derivative_backward_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Backward);

    let val = derivator.get_single_total(&func, 2.0).unwrap();
    assert!(f64::abs(val - 2.0) < 0.001);
}

#[test]
fn test_single_derivative_central_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Central);

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = derivator.get_single_total(&func, 2.0).unwrap();
    assert!(f64::abs(val - 2.0) < 0.000001);
}

#[test]
fn test_single_derivative_partial_1() 
{
    //function is 3*x*x + 2*x*y
    let func = | args: &[f64; 2] | -> f64 
    { 
        return 3.0*args[0]*args[0] + 2.0*args[0]*args[1];
    };

    let point = [1.0, 3.0];

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Central);

    //partial derivate for (x, y) = (1.0, 3.0), partial derivative for x is known to be 6*x + 2*y
    let val = derivator.get_single_partial(&func, 0, &point).unwrap();
    assert!(f64::abs(val - 12.0) < 0.000001);

    //partial derivate for (x, y) = (1.0, 3.0), partial derivative for y is known to be 2.0*x
    let val = derivator.get_single_partial(&func, 1, &point).unwrap();
    assert!(f64::abs(val - 2.0) < 0.000001);
}

#[test]
fn test_single_derivative_partial_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Central);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for x is known to be y*cos(x) + cos(y) + y*e^z
    let val = derivator.get_single_partial(&func, 0, &point).unwrap();
    let expected_value = 2.0*f64::cos(1.0) + f64::cos(2.0) + 2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.000001);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for y is known to be sin(x) - x*sin(y) + x*e^z
    let val = derivator.get_single_partial(&func, 1, &point).unwrap();
    let expected_value = f64::sin(1.0) - 1.0*f64::sin(2.0) + 1.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.000001);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for z is known to be x*y*e^z
    let val = derivator.get_single_partial(&func, 2, &point).unwrap();
    let expected_value = 1.0*2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.00001);
}

#[test]
fn test_single_derivative_error_1() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];

    let derivator = FixedStep::from_parameters(0.0, FixedStepMode::Central);
    
    //expect failure because step size is zero
    let result = derivator.get_single_partial(&func, 0, &point);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::NumberOfStepsCannotBeZero);
}

#[test]
fn test_single_derivative_error_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];

    let derivator = FixedStep::default();
    
    //expect failure because idx_to_derivate is greater than the number of points
    let result = derivator.get_single_partial(&func, 3, &point);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::IndexToDerivativeOutOfRange);
}

#[test]
fn test_double_derivative_forward_difference() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Forward);

    //double derivative at x = 1.0
    let val = derivator.get_double_total(&func, 1.0).unwrap();
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.05);
}

#[test]
fn test_double_derivative_backward_difference() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Backward);

    //double derivative at x = 1.0
    let val = derivator.get_double_total(&func, 1.0).unwrap();
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.05);
}

#[test]
fn test_double_derivative_central_difference() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    let mut derivator = FixedStep::default();
    derivator.set_method(FixedStepMode::Central);

    //double derivative at x = 1.0
    let val = derivator.get_double_total(&func, 1.0).unwrap();
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.00001);
}

#[test]
fn test_double_derivative_partial_1() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];

    let derivator = FixedStep::default();

    let idx: [usize; 2] = [0, 0]; 
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for x is known to be -y*sin(x)
    let val = derivator.get_double_partial(&func, &idx, &point).unwrap();
    let expected_value = -2.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_value) < 0.01);

    let idx: [usize; 2] = [1, 1];
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for y is known to be -x*cos(y)
    let val = derivator.get_double_partial(&func, &idx, &point).unwrap();
    let expected_value = -1.0*f64::cos(2.0);
    assert!(f64::abs(val - expected_value) < 0.01);

    let idx: [usize; 2] = [2, 2];
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for z is known to be x*y*e^z
    let val = derivator.get_double_partial(&func, &idx, &point).unwrap();
    let expected_value = 1.0*2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.01);
}

#[test]
fn test_double_derivative_partial_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];

    let derivator = FixedStep::default();

    let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be cos(x) - sin(y) + e^z
    let val = derivator.get_double_partial(&func, &idx, &point).unwrap();
    let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.0001);

    let idx: [usize; 2] = [1, 2]; //mixed partial double derivate d(df/dy)/dz
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be x*e^z
    let val = derivator.get_double_partial(&func, &idx, &point).unwrap();
    let expected_value = 1.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.0001);

    let idx: [usize; 2] = [0, 2]; //mixed partial double derivate d(df/dx)/dz
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be y*e^z
    let val = derivator.get_double_partial(&func, &idx, &point).unwrap();
    let expected_value = 2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.0001);
}

/* 
#[test]
fn test_triple_derivative_forward_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0].powf(4.0);
    };

    //expect a value of 24.00
    let val = triple_derivative::get_total_custom(&func,1.0, 0.001, mode::DiffMode::ForwardFixedStep).unwrap();
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_triple_derivative_backward_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0].powf(4.0);
    };

    //expect a value of 24.00
    let val = triple_derivative::get_total_custom(&func,1.0, 0.001, mode::DiffMode::BackwardFixedStep).unwrap();
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_triple_derivative_central_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0].powf(4.0);
    };

    //expect a value of 24.00
    let val = triple_derivative::get_total_custom(&func,1.0, 0.001, mode::DiffMode::CentralFixedStep).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_triple_derivative_partial_1() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &[f64; 2] | -> f64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let point = [1.0, 3.0];

    let idx = [0, 0, 0];
    //partial derivate for (x, y) = (1.0, 3.0), partial triple derivative for x is known to be -y*cos(x)
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep).unwrap();
    let expected_val = -3.0*f64::cos(1.0);
    assert!(f64::abs(val - expected_val) < 0.0001);

    let idx = [1, 1, 1];
    //partial derivate for (x, y) = (1.0, 3.0), partial triple derivative for y is known to be 2*x*e^y
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep).unwrap();
    let expected_val = 2.0*1.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_val) < 0.0001);
}

#[test]
fn test_triple_derivative_partial_2() 
{
    //function is x^3 * y^3 * z^3
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[0].powf(3.0)*args[1].powf(3.0)*args[2].powf(3.0);
    };

    let point = [1.0, 2.0, 3.0];

    let idx = [0, 1, 2]; //mixed partial double derivate d(d(df/dx)/dy)/dz
    //partial derivate for (x, y) = (1.0, 2.0, 3.0), mixed partial triple derivative is known to be 27.0*x^2*y^2*z^2
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep).unwrap();
    assert!(f64::abs(val - 972.0) < 0.001);

    let idx = [0, 1, 1]; //mixed partial double derivate d(d(df/dx)/dy)/dy
    //partial derivate for (x, y) = (1.0, 2.0, 3.0), mixed partial triple derivative for y is known to be 18*x^2*y*z^3
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep).unwrap();
    assert!(f64::abs(val - 972.0) < 0.001);
}
*/

#[test]
fn test_jacobian_1() 
{
    //function is x*y*z
    let func1 = | args: &[f64; 3] | -> f64 
    { 
        return args[0]*args[1]*args[2];
    };

    //function is x^2 + y^2
    let func2 = | args: &[f64; 3] | -> f64 
    { 
        return args[0].powf(2.0) + args[1].powf(2.0);
    };

    let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&func1, &func2];

    let points = [1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

    let jacobian = Jacobian::default();

    let result = jacobian.get(&function_matrix, &points).unwrap();

    assert!(result.len() == function_matrix.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];

    for i in 0..function_matrix.len()
    {
        for j in 0..points.len()
        {
            assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.000001);
        }
    }
}


#[test]
#[cfg(feature = "heap")]
fn test_jacobian_2() 
{
    //function is x*y*z
    let func1 = | args: &[f64; 3] | -> f64 
    { 
        return args[0]*args[1]*args[2];
    };

    //function is x^2 + y^2
    let func2 = | args: &[f64; 3] | -> f64 
    { 
        return args[0].powf(2.0) + args[1].powf(2.0);
    };

    let function_matrix: Vec<Box<dyn Fn(&[f64; 3]) -> f64>> = std::vec![Box::new(func1), Box::new(func2)];

    let points = [1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

    let jacobian = Jacobian::default();

    let result: Vec<Vec<f64>> = jacobian.get_on_heap(&function_matrix, &points).unwrap();

    assert!(result.len() == function_matrix.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];

    for i in 0..function_matrix.len()
    {
        for j in 0..points.len()
        {
            assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.000001);
        }
    }
}

#[test]
fn test_jacobian_1_error() 
{
    let function_matrix = [];

    //the point around which we want the jacobian matrix
    let points = [1.0, 2.0, 3.0];

    let jacobian = Jacobian::default();

    //expect error because an empty list of function was passed in
    let result = jacobian.get(&function_matrix, &points);

    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::VectorOfFunctionsCannotBeEmpty);
}

#[test]
fn test_hessian_1() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &[f64; 2] | -> f64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let points = [1.0, 2.0]; //the point around which we want the hessian matrix

    let hessian = Hessian::default();

    let result = hessian.get(&func, &points).unwrap();

    assert!(result.len() == points.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = [[-2.0*f64::sin(1.0), f64::cos(1.0) + 2.0*f64::exp(2.0)], 
                                          [f64::cos(1.0) + 2.0*f64::exp(2.0), 2.0*f64::exp(2.0)]];

    for i in 0..points.len()
    {
        for j in 0..points.len()
        {
            assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.01);
        }
    }
}