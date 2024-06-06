use crate::utils::error_checks as check;

//assumes that points in x are always increasing or decreasing
pub fn interpolate(x : &Vec<f64>, y: &Vec<f64>, xn: f64) -> f64
{
    //check if given target is already in the input points
    for iter in 0..x.len()
    {
        if x[iter] == xn
        {
            return y[iter];
        }
    }

    // need to interpolate, check the inputs first
    check::check_for_interpolation(&x, &y, xn); 
    
    let mut result: f64 = 0.0;
    let num_points = x.len();

    for i in 0..num_points
    {
        let mut term: f64 = y[i];

        for j in 0..num_points
        {
            if i != j
            {
                term = term*(xn - x[j])/(x[i] - x[j]);
            }
        }
        
        result += term;
    }

    return result;
}