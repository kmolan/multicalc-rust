
//start with x0 and y0, find value of yn at given xn with given step size
pub fn solve(func: fn(f64, f64) -> f64, x0: f64, y0: f64, xn: f64, step_size: f64) -> f64
{
    assert!(x0 != xn, "start and end poitns cannot be the same");

    assert!(step_size > 0.0, "step size must be non-zero and positive");

    assert!(step_size < xn - x0, "step size is too big");

    //note: result will be approximate if the stepsize does not exactly divide (xn - x0)
    let iterations = (xn - x0) / step_size;

    let mut yn: f64 = y0;
    let mut x = x0;

    for _ in 0..iterations as i64
    {
        yn += step_size*func(x, yn);
        x += step_size;
    }

    return yn;
}
