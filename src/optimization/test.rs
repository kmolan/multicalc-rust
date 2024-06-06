use crate::optimization::gradient_descent as gd;
use crate::numerical_derivative::mode as mode;

#[test]
fn test_gd_1()
{
    //cost_function is x^2 - 2*x - 3
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0] - 2.0*args[0] - 3.0;
    };

    let initial_state = vec![16.37];
    let tolerance = vec![0.0001];
    let learning_rate = 0.01;
    let max_iterations = 1000;

    //the passed cost_function is a known parabola with minima at x = 1.0, expect a result of 1.0 +/- tolerance
    let val = gd::solve(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance);
    
    assert!(val.final_state.len() == initial_state.len());

    let final_gradient = &val.history_of_gradients[val.history_of_gradients.len() - 1];
    
    for iter in 0..val.final_state.len()
    {
        assert!(f64::abs(val.final_state[iter] - 1.0) < tolerance[iter]);

        assert!(final_gradient[iter] < 1.0e-9);
    }    
}

#[test]
fn test_gd_2()
{
    //cost_function is 4.0*x^2 + 2.0*y^2
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0] + 2.0*args[1]*args[1];
    };

    let initial_state = vec![16.37, -5.75];
    let tolerance = vec![0.0001, 0.0001];
    let learning_rate = 0.01;
    let max_iterations = 1000;

    //the passed cost_function has a known minima, expect a result of [0, 0] +/- tolerance
    let val = gd::solve(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance);
    
    assert!(val.final_state.len() == initial_state.len());

    let final_gradient = &val.history_of_gradients[val.history_of_gradients.len() - 1];
    
    for iter in 0..val.final_state.len()
    {
        assert!(f64::abs(val.final_state[iter]) < tolerance[iter]);

        assert!(final_gradient[iter] < 1.0e-9);
    }    
}

#[test]
fn test_gd_3()
{
    //cost_function is 4*x^2 + 2*y^2 + 5*z^2
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0] + 4.0*args[1]*args[1] + 5.0*args[2]*args[2];
    };

    let initial_state = vec![78.9, -52.3, 6.7];
    let tolerance = vec![0.0001, 0.0001, 0.0001];
    let learning_rate = 0.01;
    let max_iterations = 1000;

    //the passed cost_function has a known minima, expect a result of [0, 0, 0] +/- tolerance
    let val = gd::solve(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance);
    
    assert!(val.final_state.len() == initial_state.len());

    let final_gradient = &val.history_of_gradients[val.history_of_gradients.len() - 1];
    
    for iter in 0..val.final_state.len()
    {
        assert!(f64::abs(val.final_state[iter]) < tolerance[iter]);

        assert!(final_gradient[iter] < 1.0e-9);
    }    
}

#[test]
fn test_gd_4()
{
    //cost_function is x*x + y*y
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return args[0]*args[0] + args[1]*args[1];
    };

    let initial_state = vec![0.1, 0.5];
    let tolerance = vec![0.0001, 0.0001];
    let learning_rate = 0.01;
    let max_iterations = 10000;

    //the passed cost_function has a known minima, expect a result of [0, 0, 0] +/- tolerance
    let val = gd::solve_custom(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance, 0.001, mode::DiffMode::CentralFixedStep);
    
    assert!(val.final_state.len() == initial_state.len());

    let final_gradient = &val.history_of_gradients[val.history_of_gradients.len() - 1];

    println!("{:?}", val.final_state);
    
    for iter in 0..val.final_state.len()
    {
        assert!(f64::abs(val.final_state[iter]) < tolerance[iter]);

        assert!(final_gradient[iter] < 1.0e-9);
    }    
}

#[test]
fn test_gd_5()
{
    //cost_function is 4.0*x^2 + 2.0*y^2
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0] + 2.0*args[1]*args[1];
    };

    let initial_state = vec![16.37, -5.75];
    let tolerance = vec![0.0001];
    let learning_rate = 0.01;
    let max_iterations = 1000;

    //expect failure because tolerance has only one element, but initial points is 2
    let result = std::panic::catch_unwind(||gd::solve(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance));

    assert!(result.is_err());
}

#[test]
fn test_gd_6()
{
    //cost_function is 4.0*x^2 + 2.0*y^2
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0] + 2.0*args[1]*args[1];
    };

    let initial_state = vec![16.37, -5.75];
    let tolerance = vec![0.0001, 0.001];
    let learning_rate = 0.01;
    let max_iterations = 0;

    //expect failure because max_iterations is 0
    let result = std::panic::catch_unwind(||gd::solve(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance));
    assert!(result.is_err());
}

#[test]
fn test_gd_7()
{
    //cost_function is 4.0*x^2 + 2.0*y^2
    let cost_function = | args: &Vec<f64> | -> f64 
    { 
        return 4.0*args[0]*args[0] + 2.0*args[1]*args[1];
    };

    let initial_state = vec![16.37, -5.75];
    let tolerance = vec![0.0001, 0.001];
    let step_size = 0.0;
    let learning_rate = 0.01;
    let max_iterations = 0;

    //expect failure because step_size is 0
    let result = std::panic::catch_unwind(||gd::solve_custom(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance, step_size, mode::DiffMode::CentralFixedStep));
    assert!(result.is_err());
}