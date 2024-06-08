

///solves for the line integral for parametrized curves in a 2D vector field
/// 
/// assume a vector field, V, and a curve, C
/// V is characterized in 2 dimensions, Vx and Vy
/// C is parameterized by a single variable, say, "t".
/// We also need a transformation to go t->x and t->y
/// The line integral limits are also based on this parameter t
/// 
/// [vector_field] is an array of 2 elements. The 0th element has vector field's contribution to the x-axis based on x and y values. The 1st element does the same for y-axis
/// [transformations] is an array of 2 elements. The 0th element contains the transformation from t->x, and 1st element for t->y
/// [integration_limit] is the limit parameter 't' goes to
/// [steps] is the total number of steps that the integration is discretized into. Higher number gives more accuracy, but at the cost of more computation time
/// 
/// Example:
/// Assume we have a vector field (y, -x)
/// The curve is a unit circle, parameterized by (Cos(t), Sin(t)), such that t goes from 0->2*pi
/// ```
/// use multicalc::vector_field::line_integral;
/// let vector_field_matrix: [Box<dyn Fn(&f64, &f64) -> f64>; 2] = [Box::new(|_:&f64, y:&f64|-> f64 { *y }), Box::new(|x:&f64, _:&f64|-> f64 { -x })];
///
/// let transformation_matrix: [Box<dyn Fn(&f64) -> f64>; 2] = [Box::new(|t:&f64|->f64 { t.cos() }), Box::new(|t:&f64|->f64 { t.sin() })];
///
/// let integration_limit = [0.0, 6.28];
///
/// //line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
/// let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
/// assert!(f64::abs(val + 6.28) < 0.01);
/// ```
pub fn get_2d(vector_field: &[Box<dyn Fn(&f64, &f64) -> f64>; 2], transformations: &[Box<dyn Fn(&f64) -> f64>; 2], integration_limit: &[f64; 2], steps: u64) -> f64
{
    return get_partial_2d(vector_field, transformations, integration_limit, steps, 0)
         + get_partial_2d(vector_field, transformations, integration_limit, steps, 1);
}


pub fn get_partial_2d(vector_field: &[Box<dyn Fn(&f64, &f64) -> f64>; 2], transformations: &[Box<dyn Fn(&f64) -> f64>; 2], integration_limit: &[f64; 2], steps: u64, idx: usize) -> f64
{
    let mut ans = 0.0;

    let mut cur_point = integration_limit[0];

    let delta = (integration_limit[1] - integration_limit[0])/(steps as f64);

    //use the trapezoidal rule for line integrals
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    for _ in 0..steps
    {
        let coords = get_transformed_coordinates_2d(transformations, &cur_point, &delta);

        ans += (coords[idx + 2] - coords[idx])*(vector_field[idx](&coords[2], &coords[3]) + vector_field[idx](&coords[0], &coords[1]))/2.0;

        cur_point += delta;
    }

    return ans;
}


///same as [`get_2d`] but for parametrized curves in a 3D vector field
pub fn get_3d(vector_field: &[Box<dyn Fn(&f64, &f64, &f64) -> f64>; 3], transformations: &[Box<dyn Fn(&f64) -> f64>; 3], integration_limit: &[f64; 2], steps: u64) -> f64
{
    return get_partial_3d(vector_field, transformations, integration_limit, steps, 0)
         + get_partial_3d(vector_field, transformations, integration_limit, steps, 1)
         + get_partial_3d(vector_field, transformations, integration_limit, steps, 2);
}


pub fn get_partial_3d(vector_field: &[Box<dyn Fn(&f64, &f64, &f64) -> f64>; 3], transformations: &[Box<dyn Fn(&f64) -> f64>; 3], integration_limit: &[f64; 2], steps: u64, idx: usize) -> f64
{
    let mut ans = 0.0;

    let mut cur_point = integration_limit[0];

    let delta = (integration_limit[1] - integration_limit[0])/(steps as f64);

    //use the trapezoidal rule for line integrals
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    for _ in 0..steps
    {
        let coords = get_transformed_coordinates_3d(transformations, &cur_point, &delta);

        ans += (coords[idx + 3] - coords[idx])*(vector_field[idx](&coords[3], &coords[4], &coords[5]) + vector_field[idx](&coords[0], &coords[1], &coords[2]))/2.0;

        cur_point += delta;
    }

    return ans;
}




fn get_transformed_coordinates_2d(transformations: &[Box<dyn Fn(&f64) -> f64>; 2], cur_point: &f64, time_delta: &f64) -> Vec<f64>
{
    let mut ans = vec![0.0; 4];

    ans[0] = transformations[0](cur_point); //x at t
    ans[1] = transformations[1](cur_point); //y at t

    ans[2] = transformations[0](&(cur_point + time_delta)); //x at t + delta
    ans[3] = transformations[1](&(cur_point + time_delta)); //y at t + delta

    return ans;
}


fn get_transformed_coordinates_3d(transformations: &[Box<dyn Fn(&f64) -> f64>; 3], cur_point: &f64, time_delta: &f64) -> Vec<f64>
{
    let mut ans = vec![0.0; 6];

    ans[0] = transformations[0](cur_point); //x at t
    ans[1] = transformations[1](cur_point); //y at t
    ans[2] = transformations[1](cur_point); //z at t

    ans[3] = transformations[0](&(cur_point + time_delta)); //x at t + delta
    ans[4] = transformations[1](&(cur_point + time_delta)); //y at t + delta
    ans[5] = transformations[1](&(cur_point + time_delta)); //z at t + delta

    return ans;
}