pub fn get(vec: &Vec<f64>) -> f64
{
    assert!(vec.len() > 0, "empty vector not allowed");
    
    let mut result = 0.0;

    for iter in 0..vec.len()
    {
        result += vec[iter];
    }

    result = result/(vec.len() as f64);

    return result;
}

pub fn get2(vec: &Vec<Vec<f64>>) -> Vec<f64>
{
    let mut result = vec![0.0; vec.len()];

    for iter in 0..vec.len()
    {
        result[iter] += get(&vec[iter]);
    }

    return result;
}