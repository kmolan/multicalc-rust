pub fn check_for_interpolation(x : &Vec<f64>, y: &Vec<f64>, xn: f64)
{   
    assert!(x.len() > 2, "must pass atleast 2 points");

    //points must steadily increase or decrease
    let direction: bool = x[1] - x[0] > 0.0;
    for iter in 2..x.len()
    {
        let cur_direction: bool = x[iter] - x[iter -1] > 0.0;

        assert!(direction == cur_direction, "points must steadily increase or decrease");
    }

    assert!(x.len() == y.len(), "size of input and output points must be equal");

    //check if target is within the range of input points
    if x[x.len() - 1] - x[0] > 0.0 //increasing order of points
    {
        assert!(xn >= x[0] && xn <= x[x.len() - 1], "interpolation target out of bounds");
    }
    else //decreasing order of points
    {
        assert!(xn <= x[0] && xn >= x[x.len() - 1], "interpolation target out of bounds");
    }
}

pub fn check_for_integration(integration_interval: &[f64; 2], steps: u64)
{
    assert!(integration_interval[0] < integration_interval[1], "lower end of integration interval value must be lower than the higher value");
    assert!(steps != 0, "number of steps cannot be zero");
}