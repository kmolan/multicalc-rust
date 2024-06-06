use crate::numerical_derivative::single_derivative;
use crate::numerical_derivative::mode as mode;

pub struct GdResults
{
    pub final_state: Vec<f64>, //the final result
    pub history_of_states: Vec<Vec<f64>>, //list of all states as gradient descent runs
    pub history_of_gradients: Vec<Vec<f64>>, //list of all gradients as gradient descent runs
}


/// Runs the gradient descent algorithm on a given cost function
/// Can handle single and multi-variable equations of any complexity 
/// 
/// suppose we want to do gradient descent on the function 4*x^2 + 2*y^2 + 5*z^2
///
/// first create the cost function:
///
/// ```
/// let cost_function = | args: &Vec<f64> | -> f64 
/// { 
///     return 4.0*args[0]*args[0] + 2.0*args[1]*args[1] + 5.0*args[2]*args[2];
/// };
/// 
////then define all the needed parameters:
/// 
/// let initial_state = vec![78.9, -52.3, 6.7];
/// let tolerance = vec![0.0001, 0.0001, 0.0001];
/// let learning_rate = 0.01;
/// let max_iterations = 1000;
/// 
/// use multicalc::optimization::gradient_descent;
/// let val = gradient_descent::solve(&cost_function, &initial_state, learning_rate, max_iterations, &tolerance);
///
/// assert!(val.final_state.len() == initial_state.len());
///
/// let final_gradient = &val.history_of_gradients[val.history_of_gradients.len() - 1];
///
////the passed cost function has a known minima, expect a result of [0, 0, 0] +/- tolerance
/// 
/// for iter in 0..val.final_state.len()
/// {
///     assert!(f64::abs(val.final_state[iter]) < tolerance[iter]);
///
///     assert!(final_gradient[iter] < 1.0e-9);
/// }            
/// ```
///
pub fn solve(cost_function: &dyn Fn(&Vec<f64>) -> f64,
             initial_state: &Vec<f64>, 
             learning_rate: f64, 
             max_iterations: usize, 
             tolerance: &Vec<f64>) -> GdResults
{
    return solve_custom(cost_function, initial_state, learning_rate, max_iterations, tolerance, 0.00001, &mode::DiffMode::CentralFixedStep);
}


///same as [solve()] but with the option to change the differentiation mode used
pub fn solve_custom(cost_function: &dyn Fn(&Vec<f64>) -> f64, 
              initial_state: &Vec<f64>, 
              learning_rate: f64, 
              max_iterations: usize, 
              tolerance: &Vec<f64>,
              step_size: f64,
              diff_mode: &mode::DiffMode) -> GdResults
{
    assert!(step_size != 0.0, "step size cannot be zero");

    assert!(max_iterations != 0, "max_iterations cannot be zero");

    assert!(initial_state.len() == tolerance.len(), "initial state length does not match tolerance length");

    let num_points = initial_state.len();
    let mut cur_state = initial_state.clone();
    let mut previous_state = initial_state.clone(); 
    let mut cur_iteration: usize = 0;

    let mut gradient: Vec<f64>;
    let mut tolerance_reached = false;

    let mut history_points : Vec<Vec<f64>> = vec![];
    history_points.push(cur_state.clone());

    let mut history_gradients : Vec<Vec<f64>> = vec![];

    while (cur_iteration < max_iterations) && (tolerance_reached == false)
    {
        gradient = get_gradient(cost_function, &cur_state, step_size, learning_rate, diff_mode);

        for iter in 0..num_points
        {
            cur_state[iter] = previous_state[iter] - gradient[iter];

            let mut tol = false;
            if f64::abs(previous_state[iter] - cur_state[iter]) < tolerance[iter]
            {
                tol = true;
            }

            tolerance_reached = tolerance_reached && tol; //only break out if desired tolerance has been reached for all points
        }

        previous_state = cur_state.clone();

        history_points.push(cur_state.clone());
        history_gradients.push(gradient.clone());

        cur_iteration += 1;
    }

    let result = GdResults
    {
        final_state: cur_state,
        history_of_states: history_points,
        history_of_gradients: history_gradients
    };

    return result;
}

fn get_gradient(cost_function: &dyn Fn(&Vec<f64>) -> f64, state: &Vec<f64>, step_size: f64, learning_rate: f64, diff_mode: &mode::DiffMode) -> Vec<f64>
{
    let num_points = state.len();
    
    let mut gradient = state.to_vec();

    for iter in 0..num_points
    {
        let val = single_derivative::get_partial_custom(cost_function, iter, state, step_size, diff_mode);
        
        gradient[iter] = learning_rate*val;
    }

    return gradient;
}