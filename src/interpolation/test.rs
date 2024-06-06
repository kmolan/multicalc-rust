use crate::interpolation::linear as linear;
use crate::interpolation::lagrange as lagrange;


#[test]
fn test_lagrange_interpolation_1()
{
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0];
    let y: Vec<f64> = vec![10.0, 20.0, 30.0, 50.0];

    //linear interpolation, expect a value of 40.0
    let val = lagrange::interpolate(&x, &y, 4.0); 

    assert!(val == 40.0);
}

#[test]
fn test_lagrange_interpolation_2()
{
    let mut x: Vec<f64> = vec![0.0; 10];
    let mut y: Vec<f64> = vec![0.0; 10];

    //create 10 points of sinusoidal curve
    for iter in 0..10
    {
        x[iter] = iter as f64;
        y[iter] = f64::sin(iter as f64);

    }

    //sinusoidal interpolation for 5.5 radians, expect a value of ~ -0.7055
    let val = lagrange::interpolate(&x, &y, 5.5) ;
    assert!(f64::abs(val + 0.7055) < 0.001);
}

#[test]
fn test_lagrange_interpolation_3() 
{
    let x: Vec<f64> = vec![1.0];
    let y: Vec<f64> = vec![2.0];

    //expect failure because at least 2 input points are needed
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());
}

#[test]
fn test_lagrange_interpolation_4() 
{
    let mut x: Vec<f64> = vec![0.0, 1.0, 2.0, 0.5, 7.0];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    //expect error because the input points must either steadily increase or decrease
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());

    x = vec![10.0, 9.0, 8.0, 15.0, 7.0];
    //expect error because the input points must either steadily increase or decrease
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());
}

#[test]
fn test_lagrange_interpolation_5() 
{
    let x: Vec<f64> = vec![1.0, 2.0];
    let y: Vec<f64> = vec![10.0];

    //expect error because x and y are different lengths
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());    
}

#[test]
fn test_lagrange_interpolation_6() 
{
    let x: Vec<f64> = vec![0.0, 1.0, 2.0];
    let y: Vec<f64> = vec![0.0, 10.0, 20.0];

    //expect error because target 3.0 is out of range [0.0, 2.0]
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 3.0));

    assert!(result.is_err());

    //expect error because target -1.0 is out of range [0.0, 2.0]
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, -1.0));

    assert!(result.is_err());
}


#[test]
fn test_linear_interpolation_1() 
{
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0];
    let y: Vec<f64> = vec![10.0, 20.0, 30.0, 50.0];

    //linear interpolation, expect a value of 40.0
    let val = linear::interpolate(&x, &y, 4.0) ;
    
    assert!(val == 40.0);
}

#[test]
fn test_linear_interpolation_2() 
{
    let x: Vec<f64> = vec![1.0];
    let y: Vec<f64> = vec![2.0];

    //expect failure because at least 2 input points are needed
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());
}

#[test]
fn test_linear_interpolation_3() 
{
    let mut x: Vec<f64> = vec![0.0, 1.0, 2.0, 0.5, 7.0];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    //expect error because the input points must either steadily increase or decrease
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());

    x = vec![10.0, 9.0, 8.0, 15.0, 7.0];
    //expect error because the input points must either steadily increase or decrease
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());
}

#[test]
fn test_linear_interpolation_4() 
{
    let x: Vec<f64> = vec![1.0, 2.0];
    let y: Vec<f64> = vec![10.0];

    //expect error because x and y are different lengths
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 5.5));

    assert!(result.is_err());
}

#[test]
fn test_linear_interpolation_5() 
{
    let x: Vec<f64> = vec![0.0, 1.0, 2.0];
    let y: Vec<f64> = vec![0.0, 10.0, 20.0];

    //expect error because target 3.0 is out of range [0.0, 2.0]
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, 3.0));

    assert!(result.is_err());

    //expect error because target -1.0 is out of range [0.0, 2.0]
    let result = std::panic::catch_unwind(||lagrange::interpolate(&x, &y, -1.0));

    assert!(result.is_err());    
}