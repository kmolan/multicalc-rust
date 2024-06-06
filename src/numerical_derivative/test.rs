use crate::numerical_derivative::mode as mode;
use crate::numerical_derivative::single_derivative as single_derivative;
use crate::numerical_derivative::double_derivative as double_derivative;
use crate::numerical_derivative::triple_derivative as triple_derivative;
use crate::numerical_derivative::jacobian as jacobian;
use crate::numerical_derivative::hessian as hessian;

#[test]
fn test_single_derivative_forward_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = single_derivative::get_simple_custom(&func, 2.0, 0.001, &mode::DiffMode::ForwardFixedStep);
    assert!(f64::abs(val - 2.0) < 0.05);
}

#[test]
fn test_single_derivative_backward_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = single_derivative::get_simple_custom(&func, 2.0, 0.001, &mode::DiffMode::BackwardFixedStep);
    assert!(f64::abs(val - 2.0) < 0.05);
}

#[test]
fn test_single_derivative_central_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = single_derivative::get_simple_custom(&func, 2.0, 0.001, &mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 2.0) < 0.01);
}

#[test]
fn test_single_derivative_partial_1() 
{
    //function is 3*x*x + 2*x*y
    let func = | args: &Vec<f64> | -> f64 
    { 
        return 3.0*args[0]*args[0] + 2.0*args[0]*args[1];
    };

    let point = vec![1.0, 3.0];

    //partial derivate for (x, y) = (1.0, 3.0), partial derivative for x is known to be 6*x + 2*y
    let val = single_derivative::get_partial_custom(&func, 0, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 12.0) < 0.01);

    //partial derivate for (x, y) = (1.0, 3.0), partial derivative for y is known to be 2.0*x
    let val2 = single_derivative::get_partial_custom(&func, 1, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val2 - 2.0) < 0.01);
}

#[test]
fn test_single_derivative_partial_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = vec![1.0, 2.0, 3.0];

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for x is known to be y*cos(x) + cos(y) + y*e^z
    let val = single_derivative::get_partial_custom(&func, 0, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value = 2.0*f64::cos(1.0) + f64::cos(2.0) + 2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.01);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for y is known to be sin(x) - x*sin(y) + x*e^z
    let val2 = single_derivative::get_partial_custom(&func, 1, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value_2 = f64::sin(1.0) - 1.0*f64::sin(2.0) + 1.0*f64::exp(3.0);
    assert!(f64::abs(val2 - expected_value_2) < 0.01);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for z is known to be x*y*e^z
    let val2 = single_derivative::get_partial_custom(&func, 2, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value_3 = 1.0*2.0*f64::exp(3.0);
    assert!(f64::abs(val2 - expected_value_3) < 0.01);
}

#[test]
fn test_single_derivative_error_1() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = vec![];

    //expect failure because point vector is empty
    let result = std::panic::catch_unwind(||single_derivative::get_partial_custom(&func, 0, &point, 0.001, &mode::DiffMode::CentralFixedStep));
    assert!(result.is_err());
}

#[test]
fn test_single_derivative_error_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = vec![1.0, 2.0, 3.0];
    
    //expect failure because step size is zero
    let result = std::panic::catch_unwind(||single_derivative::get_partial_custom(&func, 0, &point, 0.0, &mode::DiffMode::CentralFixedStep));
    assert!(result.is_err());
}

#[test]
fn test_single_derivative_error_3() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = vec![1.0, 2.0, 3.0];
    
    //expect failure because idx_to_derivate is greater than the number of points
    let result = std::panic::catch_unwind(||single_derivative::get_partial_custom(&func, 3, &point, 0.001, &mode::DiffMode::CentralFixedStep));
    assert!(result.is_err());
}

#[test]
fn test_double_derivative_forward_difference() 
{
    //function is x*Sin(x)
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    //double derivative at x = 1.0
    let val = double_derivative::get_simple_custom(&func, 1.0, 0.001, &mode::DiffMode::ForwardFixedStep);
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.05);
}

#[test]
fn test_double_derivative_backward_difference() 
{
    //function is x*Sin(x)
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    //double derivative at x = 1.0
    let val = double_derivative::get_simple_custom(&func, 1.0, 0.001, &mode::DiffMode::BackwardFixedStep);
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.05);
}

#[test]
fn test_double_derivative_central_difference() 
{
    //function is x*Sin(x)
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    //double derivative at x = 1.0
    let val = double_derivative::get_simple_custom(&func, 1.0, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.00001);
}


#[test]
fn test_double_derivative_partial_1() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = vec![1.0, 2.0, 3.0];

    let idx: [usize; 2] = [0, 0]; 
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for x is known to be -y*sin(x)
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value = -2.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_value) < 0.01);

    let idx2: [usize; 2] = [1, 1];
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for y is known to be -x*cos(y)
    let val2 = double_derivative::get_partial_custom(&func, &idx2, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value_2 = -1.0*f64::cos(2.0);
    assert!(f64::abs(val2 - expected_value_2) < 0.01);

    let idx3: [usize; 2] = [2, 2];
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for z is known to be x*y*e^z
    let val2 = double_derivative::get_partial_custom(&func, &idx3, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value_3 = 1.0*2.0*f64::exp(3.0);
    assert!(f64::abs(val2 - expected_value_3) < 0.01);
}

#[test]
fn test_double_derivative_partial_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = vec![1.0, 2.0, 3.0];

    let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be cos(x) - sin(y) + e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.00001);

    let idx2: [usize; 2] = [1, 2]; //mixed partial double derivate d(df/dy)/dz
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be x*e^z
    let val2 = double_derivative::get_partial_custom(&func, &idx2, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value_2 = 1.0*f64::exp(3.0);
    assert!(f64::abs(val2 - expected_value_2) < 0.00001);

    let idx3: [usize; 2] = [0, 2]; //mixed partial double derivate d(df/dx)/dz
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be e^z
    let val2 = double_derivative::get_partial_custom(&func, &idx3, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_value_3 = 2.0*f64::exp(3.0);
    assert!(f64::abs(val2 - expected_value_3) < 0.00001);
}


#[test]
fn test_triple_derivative_forward_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0].powf(4.0);
    };

    let point: Vec<f64> = vec![1.0];
    let idx: [usize; 3] = [0,0,0];

    //expect a value of 24.00
    let val = triple_derivative::get(&func, &idx, &point, 0.001, &mode::DiffMode::ForwardFixedStep);
    assert!(f64::abs(val - 24.0) < 0.05);
}


#[test]
fn test_triple_derivative_backward_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0].powf(4.0);
    };

    let point: Vec<f64> = vec![1.0];
    let idx: [usize; 3] = [0,0,0];

    //expect a value of 24.00
    let val = triple_derivative::get(&func, &idx, &point, 0.001, &mode::DiffMode::BackwardFixedStep);
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_triple_derivative_central_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0].powf(4.0);
    };

    let point: Vec<f64> = vec![1.0];
    let idx: [usize; 3] = [0,0,0];

    //expect a value of 24.00
    let val = triple_derivative::get(&func, &idx, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 24.0) < 0.01);
}

#[test]
fn test_triple_derivative_partial_1() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let point = vec![1.0, 3.0];

    let idx1 = [0, 0, 0];
    //partial derivate for (x, y) = (1.0, 3.0), partial triple derivative for x is known to be -y*cos(x)
    let val = triple_derivative::get(&func, &idx1, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_val = -3.0*f64::cos(1.0);
    assert!(f64::abs(val - expected_val) < 0.01);

    let idx2 = [1, 1, 1];
    //partial derivate for (x, y) = (1.0, 3.0), partial triple derivative for y is known to be 2*x*e^y
    let val2 = triple_derivative::get(&func, &idx2, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_val2 = 2.0*1.0*f64::exp(3.0);
    assert!(f64::abs(val2 - expected_val2) < 0.01);
}

#[test]
fn test_triple_derivative_partial_2() 
{
    //function is x^3 * y^3 * z^3
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[0].powf(3.0)*args[1].powf(3.0)*args[2].powf(3.0);
    };

    let point = vec![1.0, 2.0, 3.0];

    let idx1 = [0, 1, 2]; //mixed partial double derivate d(d(df/dx)/dy)/dz
    //partial derivate for (x, y) = (1.0, 2.0, 3.0), mixed partial triple derivative is known to be 27.0*x^2*y^2*z^2
    let val = triple_derivative::get(&func, &idx1, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_val = 27.0*1.0*4.0*9.0;
    assert!(f64::abs(val - expected_val) < 0.01);

    let idx2 = [0, 1, 1]; //mixed partial double derivate d(d(df/dx)/dy)/dy
    //partial derivate for (x, y) = (1.0, 2.0, 3.0), mixed partial triple derivative for y is known to be 18*x^2*y*z^3
    let val2 = triple_derivative::get(&func, &idx2, &point, 0.001, &mode::DiffMode::CentralFixedStep);
    let expected_val2 = 18.0*1.0*2.0*27.0;
    assert!(f64::abs(val2 - expected_val2) < 0.01);
}

#[test]
fn test_jacobian_1() 
{
    //function is x*y*z
    let func1 = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[1]*args[2];
    };

    //function is x^2 + y^2
    let func2 = | args: &Vec<f64> | -> f64 
    { 
        return args[0].powf(2.0) + args[1].powf(2.0);
    };

    let function_matrix: Vec<Box<dyn Fn(&Vec<f64>) -> f64>> = vec![Box::new(func1), Box::new(func2)];

    let points = vec![1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

    let result = jacobian::get(&function_matrix, &points);

    assert!(result.len() == function_matrix.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = vec![vec![6.0, 3.0, 2.0], vec![2.0, 4.0, 0.0]];

    for i in 0..function_matrix.len()
    {
        for j in 0..points.len()
        {
            assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.01);
        }
    }
}


#[test]
fn test_hessian_1() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let points = vec![1.0, 2.0]; //the point around which we want the hessian matrix

    let result = hessian::get(&func, &points);

    assert!(result.len() == points.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = vec![vec![-2.0*f64::sin(1.0), f64::cos(1.0) + 2.0*f64::exp(2.0)], vec![f64::cos(1.0) + 2.0*f64::exp(2.0), 2.0*f64::exp(2.0)]];

    for i in 0..points.len()
    {
        for j in 0..points.len()
        {
            assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.01);
        }
    }
}

#[test]
fn test_hessian_2() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let points = vec![]; //the point around which we want the hessian matrix

    //expect failure because points is an empty vector
    let result = std::panic::catch_unwind(||hessian::get(&func, &points));
    assert!(result.is_err());
}