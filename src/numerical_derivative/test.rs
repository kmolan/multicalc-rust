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
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = single_derivative::get_total_custom(&func, 2.0, 0.001, mode::DiffMode::ForwardFixedStep);
    assert!(f64::abs(val - 2.0) < 0.001);
}

#[test]
fn test_single_derivative_forward_difference_complex() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.0 + 1.0i
    let val = single_derivative::get_total_custom(&func, num_complex::c64(2.0, 1.0), 0.001, mode::DiffMode::ForwardFixedStep);
    assert!(num_complex::ComplexFloat::abs(val.re - 2.0) < 0.001);
    assert!(num_complex::ComplexFloat::abs(val.im - 1.0) < 0.001);
}

#[test]
fn test_single_derivative_backward_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = single_derivative::get_total_custom(&func, 2.0, 0.001, mode::DiffMode::BackwardFixedStep);
    assert!(f64::abs(val - 2.0) < 0.001);
}

#[test]
fn test_single_derivative_backward_difference_complex() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.0 + 1.0i
    let val = single_derivative::get_total_custom(&func, num_complex::c64(2.0, 1.0), 0.001, mode::DiffMode::BackwardFixedStep);
    assert!(num_complex::ComplexFloat::abs(val.re - 2.0) < 0.001);
    assert!(num_complex::ComplexFloat::abs(val.im - 1.0) < 0.001);
}

#[test]
fn test_single_derivative_central_difference() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.00
    let val = single_derivative::get_total_custom(&func, 2.0, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 2.0) < 0.000001);
}

#[test]
fn test_single_derivative_central_difference_complex() 
{
    //function is x*x/2.0, derivative is known to be x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0]*args[0]/2.0;
    };

    //simple derivative around x = 2.0, expect a value of 2.0 + 1.0i
    let val = single_derivative::get_total_custom(&func, num_complex::c64(2.0, 1.0), 0.001, mode::DiffMode::CentralFixedStep);
    assert!(num_complex::ComplexFloat::abs(val.re - 2.0) < 0.000001);
    assert!(num_complex::ComplexFloat::abs(val.im - 1.0) < 0.000001);
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

    //partial derivate for (x, y) = (1.0, 3.0), partial derivative for x is known to be 6*x + 2*y
    let val = single_derivative::get_partial_custom(&func, 0, &point, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 12.0) < 0.000001);

    //partial derivate for (x, y) = (1.0, 3.0), partial derivative for y is known to be 2.0*x
    let val = single_derivative::get_partial_custom(&func, 1, &point, 0.001, mode::DiffMode::CentralFixedStep);
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

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for x is known to be y*cos(x) + cos(y) + y*e^z
    let val = single_derivative::get_partial_custom(&func, 0, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = 2.0*f64::cos(1.0) + f64::cos(2.0) + 2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.000001);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for y is known to be sin(x) - x*sin(y) + x*e^z
    let val = single_derivative::get_partial_custom(&func, 1, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = f64::sin(1.0) - 1.0*f64::sin(2.0) + 1.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.000001);

    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for z is known to be x*y*e^z
    let val = single_derivative::get_partial_custom(&func, 2, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = 1.0*2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.00001);
}

#[test]
fn test_single_derivative_partial_3() 
{
    //function is 3*x*x + 2*x*y
    let func = | args: &[num_complex::Complex64; 2] | -> num_complex::Complex64 
    { 
        return 3.0*args[0]*args[0] + 2.0*args[0]*args[1];
    };

    let point = [num_complex::c64(1.0, 4.0), num_complex::c64(3.0, 6.5)];

    //partial derivate for (x, y) = (1.0 + 4.0i, 3.0 + 6.5i), partial derivative for x is known to be 6*x + 2*y
    let val = single_derivative::get_partial_custom(&func, 0, &point, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(num_complex::ComplexFloat::abs(val.re - 12.0) < 0.000001);
    assert!(num_complex::ComplexFloat::abs(val.im - 37.0) < 0.000001);

    //partial derivate for (x, y) = (1.0 + 4.0i, 3.0 + 6.5i), partial derivative for y is known to be 2.0*x
    let val = single_derivative::get_partial_custom(&func, 1, &point, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(num_complex::ComplexFloat::abs(val.re - 2.0) < 0.000001);
    assert!(num_complex::ComplexFloat::abs(val.im - 8.0) < 0.000001);
}

#[test]
fn test_single_derivative_partial_4() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [num_complex::c64(1.0, 4.0), num_complex::c64(2.0, 6.5), num_complex::c64(3.0, 0.0)];

    //partial derivate for (x, y, z) = (1.0 + 4.0i, 2.0 + 6.5i, 3.0 + 0.0i), partial derivative for x is known to be y*cos(x) + cos(y) + y*e^z
    let val = single_derivative::get_partial_custom(&func, 0, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[1]*point[0].cos() + point[1].cos() + point[1]*point[2].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);

    //partial derivate for (x, y, z) = (1.0 + 4.0i, 2.0 + 6.5i, 3.0 + 0.0i), partial derivative for y is known to be sin(x) - x*sin(y) + x*e^z
    let val = single_derivative::get_partial_custom(&func, 1, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[0].sin() - point[0]*point[1].sin() + point[0]*point[2].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.001);

    //partial derivate for (x, y, z) = (1.0 + 4.0i, 2.0 + 6.5i, 3.0 + 0.0i), partial derivative for z is known to be x*y*e^z
    let val = single_derivative::get_partial_custom(&func, 2, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[0]*point[1]*point[2].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);
}

#[test]
#[cfg(feature = "std")]
fn test_single_derivative_error_2() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];
    
    //expect failure because step size is zero
    let result = std::panic::catch_unwind(||single_derivative::get_partial_custom(&func, 0, &point, 0.0, mode::DiffMode::CentralFixedStep));
    assert!(result.is_err());
}

#[test]
#[cfg(feature = "std")]
fn test_single_derivative_error_3() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [1.0, 2.0, 3.0];
    
    //expect failure because idx_to_derivate is greater than the number of points
    let result = std::panic::catch_unwind(||single_derivative::get_partial_custom(&func, 3, &point, 0.001, mode::DiffMode::CentralFixedStep));
    assert!(result.is_err());
}

#[test]
fn test_double_derivative_forward_difference() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    //double derivative at x = 1.0
    let val = double_derivative::get_total_custom(&func, 1.0, 0.001, mode::DiffMode::ForwardFixedStep);
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.05);
}

#[test]
fn test_double_derivative_forward_difference_complex() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0]*args[0].sin();
    };

    let point = num_complex::c64(1.0, 2.5);

    //double derivative at x = (1.0 + 2.5i)
    let val = double_derivative::get_total_custom(&func, point, 0.001, mode::DiffMode::ForwardFixedStep);
    let expected_val = 2.0*point.cos() - point*point.sin();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.05);
}

#[test]
fn test_double_derivative_backward_difference() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    //double derivative at x = 1.0
    let val = double_derivative::get_total_custom(&func, 1.0, 0.001, mode::DiffMode::BackwardFixedStep);
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.05);
}

#[test]
fn test_double_derivative_backward_difference_complex() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0]*args[0].sin();
    };

    let point = num_complex::c64(1.0, 2.5);

    //double derivative at x = (1.0 + 2.5i)
    let val = double_derivative::get_total_custom(&func, point, 0.001, mode::DiffMode::BackwardFixedStep);
    let expected_val = 2.0*point.cos() - point*point.sin();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.05);
}

#[test]
fn test_double_derivative_central_difference() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0]*args[0].sin();
    };

    //double derivative at x = 1.0
    let val = double_derivative::get_total_custom(&func, 1.0, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_val) < 0.00001);
}

#[test]
fn test_double_derivative_central_difference_complex() 
{
    //function is x*Sin(x), double derivative known to be 2.0*Cos(x) - x*Sin(x)
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0]*args[0].sin();
    };

    let point = num_complex::c64(1.0, 2.5);

    //double derivative at x = (1.0 + 2.5i)
    let val = double_derivative::get_total_custom(&func, point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = 2.0*point.cos() - point*point.sin();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.0001);
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

    let idx: [usize; 2] = [0, 0]; 
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for x is known to be -y*sin(x)
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = -2.0*f64::sin(1.0);
    assert!(f64::abs(val - expected_value) < 0.01);

    let idx: [usize; 2] = [1, 1];
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for y is known to be -x*cos(y)
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = -1.0*f64::cos(2.0);
    assert!(f64::abs(val - expected_value) < 0.01);

    let idx: [usize; 2] = [2, 2];
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for z is known to be x*y*e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
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

    let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be cos(x) - sin(y) + e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.00001);

    let idx: [usize; 2] = [1, 2]; //mixed partial double derivate d(df/dy)/dz
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be x*e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = 1.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.00001);

    let idx: [usize; 2] = [0, 2]; //mixed partial double derivate d(df/dx)/dz
    //partial derivate for (x, y, z) = (1.0, 2.0, 3.0), mixed partial double derivative is known to be y*e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = 2.0*f64::exp(3.0);
    assert!(f64::abs(val - expected_value) < 0.00001);
}

#[test]
fn test_double_derivative_partial_3() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [num_complex::c64(1.0, 3.5), num_complex::c64(2.0, 2.0), num_complex::c64(3.0, 0.0)];

    let idx: [usize; 2] = [0, 0]; 
    //partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), partial double derivative for x is known to be -y*sin(x)
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = -point[1]*point[0].sin();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);

    let idx: [usize; 2] = [1, 1];
    //partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), partial double derivative for y is known to be -x*cos(y)
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = -point[0]*point[1].cos();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);

    let idx: [usize; 2] = [2, 2];
    //partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), partial double derivative for z is known to be x*y*e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[0]*point[1]*point[2].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);
}

#[test]
fn test_double_derivative_partial_4() 
{
    //function is y*sin(x) + x*cos(y) + x*y*e^z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };

    let point = [num_complex::c64(1.0, 3.5), num_complex::c64(2.0, 2.0), num_complex::c64(3.0, 0.0)];

    let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
    //partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), mixed partial double derivative is known to be cos(x) - sin(y) + e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[0].cos() - point[1].sin() + point[2].exp(); 
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);

    let idx: [usize; 2] = [1, 2]; //mixed partial double derivate d(df/dy)/dz
    //partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), mixed partial double derivative is known to be x*e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[0]*point[2].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);

    let idx: [usize; 2] = [0, 2]; //mixed partial double derivate d(df/dx)/dz
    //partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), mixed partial double derivative is known to be y*e^z
    let val = double_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_value = point[1]*point[2].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001);
}


#[test]
fn test_triple_derivative_forward_difference() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return args[0].powf(4.0);
    };

    //expect a value of 24.00
    let val = triple_derivative::get_total_custom(&func,1.0, 0.001, mode::DiffMode::ForwardFixedStep);
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_triple_derivative_forward_difference_complex() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0].powf(4.0);
    };

    let point = num_complex::c64(1.0, 5.0);

    //expect a value of 24.00 + 120i
    let val = triple_derivative::get_total_custom(&func, point, 0.001, mode::DiffMode::ForwardFixedStep);
    let expected_val = 24.0*point;
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.05);
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
    let val = triple_derivative::get_total_custom(&func,1.0, 0.001, mode::DiffMode::BackwardFixedStep);
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_triple_derivative_backward_difference_complex() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0].powf(4.0);
    };

    let point = num_complex::c64(1.0, 5.0);

    //expect a value of 24.00 + 120i
    let val = triple_derivative::get_total_custom(&func, point, 0.001, mode::DiffMode::BackwardFixedStep);
    let expected_val = 24.0*point;
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.05);
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
    let val = triple_derivative::get_total_custom(&func,1.0, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_triple_derivative_central_difference_complex() 
{
    //function is x^4, triple derivative is known to be 24.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return args[0].powf(4.0);
    };

    let point = num_complex::c64(1.0, 5.0);

    //expect a value of 24.00 + 120i
    let val = triple_derivative::get_total_custom(&func, point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = 24.0*point;
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.0001);
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
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = -3.0*f64::cos(1.0);
    assert!(f64::abs(val - expected_val) < 0.0001);

    let idx = [1, 1, 1];
    //partial derivate for (x, y) = (1.0, 3.0), partial triple derivative for y is known to be 2*x*e^y
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
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
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 972.0) < 0.001);

    let idx = [0, 1, 1]; //mixed partial double derivate d(d(df/dx)/dy)/dy
    //partial derivate for (x, y) = (1.0, 2.0, 3.0), mixed partial triple derivative for y is known to be 18*x^2*y*z^3
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    assert!(f64::abs(val - 972.0) < 0.001);
}

#[test]
fn test_triple_derivative_partial_3() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &[num_complex::Complex64; 2] | -> num_complex::Complex64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let point = [num_complex::c64(1.0, 4.0), num_complex::c64(3.0, 1.5)];

    let idx = [0, 0, 0];
    //partial derivate for (x, y) = (1.0 + 4.0i, 3.0 + 1.5i), partial triple derivative for x is known to be -y*cos(x)
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = -point[1]*point[0].cos();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.0001);

    let idx = [1, 1, 1];
    //partial derivate for (x, y) = (1.0 + 4.0i, 3.0 + 1.5i), partial triple derivative for y is known to be 2*x*e^y
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = 2.0*point[0]*point[1].exp();
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.0001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.0001);
}

#[test]
fn test_triple_derivative_partial_4() 
{
    //function is x^3 * y^3 * z^3
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[0].powf(3.0)*args[1].powf(3.0)*args[2].powf(3.0);
    };

    let point = [num_complex::c64(1.0, 4.0), num_complex::c64(2.0, 1.5), num_complex::c64(3.0, 0.0)];

    let idx = [0, 1, 2]; //mixed partial double derivate d(d(df/dx)/dy)/dz
    //partial derivate for (x, y, z) = (1.0 + 4.0i, 2.0 + 1.5i, 3.0 + 0.0i), mixed partial triple derivative is known to be 27.0*x^2*y^2*z^2
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = 27.0*point[0].powf(2.0)*point[1].powf(2.0)*point[2].powf(2.0);
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.01);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.01);

    let idx = [0, 1, 1]; //mixed partial double derivate d(d(df/dx)/dy)/dy
    //partial derivate for (x, y, z) = (1.0 + 4.0i, 2.0 + 1.5i, 3.0 + 0.0i), mixed partial triple derivative for y is known to be 18*x^2*y*z^3
    let val = triple_derivative::get_partial_custom(&func, &idx, &point, 0.001, mode::DiffMode::CentralFixedStep);
    let expected_val = 18.0*point[0].powf(2.0)*point[1]*point[2].powf(3.0);
    assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.001);
    assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.001);
}

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

    let result = jacobian::get(&function_matrix, &points);

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
fn test_jacobian_1_complex() 
{
    //function is x*y*z
    let func1 = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[0]*args[1]*args[2];
    };

    //function is x^2 + y^2
    let func2 = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[0].powf(2.0) + args[1].powf(2.0);
    };

    let function_matrix: [&dyn Fn(&[num_complex::Complex64; 3]) -> num_complex::Complex64; 2] = [&func1, &func2];

    //the point around which we want the jacobian matrix
    let points = [num_complex::c64(1.0, 3.0), num_complex::c64(2.0, 3.5), num_complex::c64(3.0, 0.0)];

    let result = jacobian::get(&function_matrix, &points);

    assert!(result.len() == function_matrix.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = [[points[1]*points[2], points[0]*points[2], points[0]*points[1]], 
                                                   [2.0*points[0], 2.0*points[1], 0.0*points[2]]];


    for i in 0..function_matrix.len()
    {
        for j in 0..points.len()
        {
            assert!(num_complex::ComplexFloat::abs(result[i][j].re - expected_result[i][j].re) < 0.000001);
            assert!(num_complex::ComplexFloat::abs(result[i][j].im - expected_result[i][j].im) < 0.000001);
        }
    }
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

    let result = hessian::get(&func, &points);

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

#[test]
fn test_hessian_1_complex() 
{
    //function is y*sin(x) + 2*x*e^y
    let func = | args: &[num_complex::Complex64; 2] | -> num_complex::Complex64 
    { 
        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    };

    let points = [num_complex::c64(1.0, 2.5), num_complex::c64(2.0, 5.0)]; //the point around which we want the hessian matrix

    let result = hessian::get(&func, &points);

    assert!(result.len() == points.len()); //number of rows
    assert!(result[0].len() == points.len()); //number of columns

    let expected_result = [[-points[1]*points[0].sin(), points[0].cos() + 2.0*points[1].exp()], 
                                                   [points[0].cos() + 2.0*points[1].exp(), 2.0*points[0]*points[1].exp()]];

    for i in 0..points.len()
    {
        for j in 0..points.len()
        {
            assert!(num_complex::ComplexFloat::abs(result[i][j].re - expected_result[i][j].re) < 0.0001);
            assert!(num_complex::ComplexFloat::abs(result[i][j].im - expected_result[i][j].im) < 0.0001);
        }
    }
}