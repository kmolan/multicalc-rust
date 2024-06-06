use crate::math::mean as mean;
use crate::math::factorial as factorial;
use crate::math::permutation as permutation;
use crate::math::binomial as binomial;
use crate::math::normalize as norm;

#[test]
fn test_mean_1()
{
    let mut test_vector = vec![0.0; 10];
    for iter in 0..10
    {
        test_vector[iter] = (iter as f64)*10.0;
    }

    assert!(mean::get(&test_vector) == 45.0);
}

#[test]
fn test_mean_2()
{
    let test_vector = vec![];

    //expect failure because input vector is empty
    let result = std::panic::catch_unwind(||mean::get(&test_vector));
    assert!(result.is_err());
}

#[test]
fn test_mean_3()
{
    let mut test_vector = vec![vec![0.0; 10]; 10];

    for i in 0..10
    {
        for j in 0..10
        {
            test_vector[i][j] = (j as f64)*10.0;
        }
    }

    let test_result = vec![45.0; 10];

    assert!(mean::get2(&test_vector) == test_result);
}

#[test]
fn test_factorial()
{
    assert!(factorial::get(0) == 1);
    assert!(factorial::get(1) == 1);
    assert!(factorial::get(2) == 2);

    assert!(factorial::get(5) == 120);
    assert!(factorial::get(10) == 3_628_800);
}

#[test]
fn test_permutation()
{
    assert!(permutation::get(16, 3) == 3360);
}

#[test]
fn test_combination()
{
    assert!(binomial::get(16, 3) == 560);
}

#[test]
fn test_norm_1()
{
    let input = vec![1.0, 2.0, 3.0, 4.0];
    assert!(norm::norm(&input) == f64::sqrt(30.0));
}

#[test]
fn test_norm_2()
{
    let input = vec![];

    //expect failure because input vector is empty
    let result = std::panic::catch_unwind(||norm::norm(&input));
    assert!(result.is_err());
}