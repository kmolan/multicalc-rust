use crate::utils::error_checks as check;

//simple, fast, not as accurate
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
    check::check_for_interpolation(&x, &y, xn) ;

    //find the largest point less than the target
    let mut iter: usize = 0;
    while x[iter] < xn
    {
        iter += 1;
    }

    iter -= 1;

    let result = y[iter] + (xn - x[iter])*(y[iter + 1] - y[iter])/(x[iter + 1] - x[iter]);

    return result;
}