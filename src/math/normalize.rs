pub fn norm(vec:&Vec<f64>) -> f64
{
    assert!(vec.len() > 0, "empty vector not allowed");
    let mut result:f64 = 0.0;

    for num in vec.iter()
    {
        result += num*num;
    }

    result = result.sqrt();

    return result;
}