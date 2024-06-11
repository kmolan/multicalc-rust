use crate::vector_field::line_integral;
use num_complex::ComplexFloat;


///solves for the flux integral for parametrized curves in a vector field
/// 
/// assume a vector field, V, and a curve, C
/// V is characterized in 2 dimensions
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
/// use multicalc::vector_field::flux_integral;
/// let vector_field_matrix: [Box<dyn Fn(&f64, &f64) -> f64>; 2] = [Box::new(|_:&f64, y:&f64|-> f64 { *y }), Box::new(|x:&f64, _:&f64|-> f64 { -x })];
///
/// let transformation_matrix: [Box<dyn Fn(&f64) -> f64>; 2] = [Box::new(|t:&f64|->f64 { t.cos() }), Box::new(|t:&f64|->f64 { t.sin() })];
///
/// let integration_limit = [0.0, 6.28];
///
/// //flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
/// let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
/// assert!(f64::abs(val + 0.0) < 0.01);
/// ```
pub fn get_2d<T: ComplexFloat>(vector_field: &[Box<dyn Fn(&T, &T) -> T>; 2], transformations: &[Box<dyn Fn(&T) -> T>; 2], integration_limit: &[T; 2], steps: u64) -> T
{
    return line_integral::get_partial_2d(vector_field, transformations, integration_limit, steps, 0)
         - line_integral::get_partial_2d(vector_field, transformations, integration_limit, steps, 1);
}