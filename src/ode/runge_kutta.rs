#[derive(PartialEq)]
pub enum RKMode
{
    RK2,
    RK4
}

//start with x0 and y0, find value of yn at given xn with given step size
pub fn solve(mode: RKMode, func: fn(f64, f64) -> f64, x0: f64, y0: f64, xn: f64, step_size: f64) -> f64
{
    assert!(x0 != xn, "start and end poitns cannot be the same");

    assert!(step_size > 0.0, "step size must be non-zero and positive");

    assert!(step_size < xn - x0, "step size is too big");

    if mode == RKMode::RK2
    {
        return solve_rk2(func, x0, y0, xn, step_size);
    }

    return solve_rk4(func, x0, y0, xn, step_size);
}


fn solve_rk2(func: fn(f64, f64) -> f64, x0: f64, y0: f64, xn: f64, step_size: f64) -> f64
{
    //note: result will be approximate if the stepsize does not exactly divide (xn - x0)
    let iterations = (xn - x0) / step_size;

    let mut yn: f64 = y0;
    let mut x = x0;

    for _ in 0..iterations as i64
    {
        let k1 = func(x, yn);
        let k2 = func(x + 0.5*step_size, yn + 0.5*step_size*k1);

        yn += step_size*k2;
        x += step_size;
    }

    return yn;
}


fn solve_rk4(func: fn(f64, f64) -> f64, x0: f64, y0: f64, xn: f64, step_size: f64) -> f64
{
    //note: result will be approximate if the stepsize does not exactly divide (xn - x0)
    let iterations = (xn - x0) / step_size;

    let mut yn: f64 = y0;
    let mut x = x0;

    for _ in 0..iterations as i64
    {
        let k1 = step_size*func(x, yn);
        let k2 = step_size*func(x + 0.5*step_size, yn + 0.5*k1);
        let k3 = step_size*func(x + 0.5*step_size, yn + 0.5*k2);
        let k4 = step_size*func(x + step_size, yn + k3);

        yn += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;

        x += step_size;
    }

    return yn;
}