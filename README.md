# multicalc-rust
![maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
![github](https://github.com/kmolan/multicalc-rust/actions/workflows/rust/badge.svg)

Rust scientific computing for single and multi-variable calculus

## Table of Contents

- [Why Peroxide?](#why-peroxide)
  - [1. Customize features](#1-customize-features)
  - [2. Easy to optimize](#2-easy-to-optimize)
  - [3. Friendly syntax](#3-friendly-syntax)
  - [4. Can choose two different coding styles](#4-can-choose-two-different-coding-styles)
  - [5. Batteries included](#5-batteries-included)
  - [6. Compatible with Mathematics](#6-compatible-with-mathematics)
  - [7. Written in Rust](#7-written-in-rust)
- [Latest README version](#latest-readme-version)
- [Pre-requisite](#pre-requisite)
- [Install](#install)
- [Useful tips for features](#useful-tips-for-features)
- [Module Structure](#module-structure)
- [Documentation](#documentation)
- [Examples](#examples)
- [Release Info](#release-info)
- [Contributes Guide](#contributes-guide)
- [LICENSE](#license)

### 1. Single simple derivatives
```rust
//function is x*x/2.0, derivative is known to be x
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[0]*args[0]/2.0;
};

//simple derivative around x = 2.0, expect a value of 2.00
let val = single_derivative::get(&func, 2.0, 0.001);
assert!(f64::abs(val - 2.0) < 0.05);
```

### 2. Single partial derivatives
```rust
//function is y*sin(x) + x*cos(y) + x*y*e^z
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
};

let point = vec![1.0, 2.0, 3.0];

//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for x is known to be y*cos(x) + cos(y) + y*e^z
let val = single_derivative::get_partial(&func, 0, &point, 0.001);
let expected_value = 2.0*f64::cos(1.0) + f64::cos(2.0) + 2.0*f64::exp(3.0);
assert!(f64::abs(val - expected_value) < 0.01);

//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for y is known to be sin(x) - x*sin(y) + x*e^z
let val2 = single_derivative::get_partial(&func, 1, &point, 0.001);
let expected_value_2 = f64::sin(1.0) - 1.0*f64::sin(2.0) + 1.0*f64::exp(3.0);
assert!(f64::abs(val2 - expected_value_2) < 0.01);

//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for z is known to be x*y*e^z
let val2 = single_derivative::get_partial(&func, 2, &point, 0.001);
let expected_value_3 = 1.0*2.0*f64::exp(3.0);
assert!(f64::abs(val2 - expected_value_3) < 0.01);
```

### 3. Double simple derivatives
```rust
//function is x*Sin(x)
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[0]*args[0].sin();
};

//double derivative at x = 1.0
let val = double_derivative::get_simple(&func, 1.0, 0.001);
let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
assert!(f64::abs(val - expected_val) < 0.05);
```

### 4. Double partial derivatives
```rust
//function is y*sin(x) + x*cos(y) + x*y*e^z
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
};

let point = vec![1.0, 2.0, 3.0];

let idx: [usize; 2] = [0, 0]; 
//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for x is known to be -y*sin(x)
let val = double_derivative::get_partial(&func, &idx, &point, 0.001);
let expected_value = -2.0*f64::sin(1.0);
assert!(f64::abs(val - expected_value) < 0.01);

let idx2: [usize; 2] = [1, 1];
//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for y is known to be -x*cos(y)
let val2 = double_derivative::get_partial(&func, &idx2, &point, 0.001);
let expected_value_2 = -1.0*f64::cos(2.0);
assert!(f64::abs(val2 - expected_value_2) < 0.01);

let idx3: [usize; 2] = [2, 2];
//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial double derivative for z is known to be x*y*e^z
let val2 = double_derivative::get_partial(&func, &idx3, &point, 0.001);
let expected_value_3 = 1.0*2.0*f64::exp(3.0);
assert!(f64::abs(val2 - expected_value_3) < 0.01);
```

### 5. Single integrals
```rust
//equation is 2.0*x + y*z
let func = | args: &Vec<f64> | -> f64 
{ 
    return 2.0*args[0] + args[1]*args[2];
};

let integration_interval = [0.0, 1.0];
let point = vec![1.0, 2.0, 3.0];

//partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 0, &integration_interval, &point, 100);
assert!(f64::abs(val - 7.0) < 0.00001);


let integration_interval = [0.0, 2.0];

//partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 1, &integration_interval, &point, 100);
assert!(f64::abs(val - 10.0) < 0.00001);


let integration_interval = [0.0, 3.0];

//partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 2, &integration_interval, &point, 100);
assert!(f64::abs(val - 15.0) < 0.00001);
```

### 6. Double integrals
```rust
//equation is 2.0*x + y*z
let func = | args: &Vec<f64> | -> f64 
{ 
    return 2.0*args[0] + args[1]*args[2];
};

let integration_intervals = [[0.0, 1.0], [0.0, 1.0]];
let point = vec![1.0, 1.0, 1.0];

//double partial integration for first x then y, expect a value of ~1.50
let val = double_integration::get_partial(IntegrationMethod::Booles, &func, [0, 1], &integration_intervals, &point, 20);
assert!(f64::abs(val - 1.50) < 0.00001);
```

### 7. Jacobians
```rust
//function is x*y*z
let func1 = | args: &Vec<f64> | -> f64 
{ 
    return args[0]*args[1]*args[2];
};

//function is x^2 + y^2
let func2 = | args: &Vec<f64> | -> f64 
{ 
    return args[0].powf(2.0) + args[1].powf(2.0);
};

let function_matrix: Vec<Box<dyn Fn(&Vec<f64>) -> f64>> = vec![Box::new(func1), Box::new(func2)];

let points = vec![1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

let result = jacobian::get(&function_matrix, &points);

assert!(result.len() == function_matrix.len()); //number of rows
assert!(result[0].len() == points.len()); //number of columns

let expected_result = vec![vec![6.0, 3.0, 2.0], vec![2.0, 4.0, 0.0]];

for i in 0..function_matrix.len()
{
    for j in 0..points.len()
    {
        assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.01);
    }
}
```

### 8. Hessians
```rust
//function is y*sin(x) + 2*x*e^y
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
};

let points = vec![1.0, 2.0]; //the point around which we want the hessian matrix

let result = hessian::get(&func, &points);

assert!(result.len() == points.len()); //number of rows
assert!(result[0].len() == points.len()); //number of columns

let expected_result = vec![vec![-2.0*f64::sin(1.0), f64::cos(1.0) + 2.0*f64::exp(2.0)], vec![f64::cos(1.0) + 2.0*f64::exp(2.0), 2.0*f64::exp(2.0)]];

for i in 0..points.len()
{
    for j in 0..points.len()
    {
        assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.01);
    }
}
```

### 9. Linear approximation
```rust
//function is x + y^2 + z^3, which we want to linearize
let function_to_approximate = | args: &Vec<f64> | -> f64 
{ 
    return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
};

let point = vec![1.0, 2.0, 3.0]; //the point we want to linearize around

let result = linear_approximation::get(&function_to_approximate, &point);

assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);

//now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
//with random noise between [-0.1, +0.1) 
let mut prediction_points: Vec<Vec<f64>> = vec![vec![]; 1000];
let mut random_generator = rand::thread_rng();

for iter in 0..1000
{
    let noise = random_generator.gen_range(-0.1..0.1);
    prediction_points[iter] = vec![1.0 + noise, 2.0 + noise, 3.0 + noise];
}

let prediction_metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);

assert!(prediction_metrics.root_mean_squared_error < 0.05);
assert!(prediction_metrics.mean_absolute_error < 0.05);
assert!(prediction_metrics.mean_squared_error < 0.05);
assert!(prediction_metrics.r_squared > 0.9999);
assert!(prediction_metrics.adjusted_r_squared > 0.9999);
```

### 10. Quadratic approximation
```rust
//function is e^(x/2) + sin(y) + 2.0*z
let function_to_approximate = | args: &Vec<f64> | -> f64 
{ 
    return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
};

let point = vec![0.0, 3.14/2.0, 10.0]; //the point we want to approximate around

let result = quadratic_approximation::get(&function_to_approximate, &point);

assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);

//now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
//with random noise between [-0.1, +0.1) 
let mut prediction_points: Vec<Vec<f64>> = vec![vec![]; 1000];
let mut random_generator = rand::thread_rng();

for iter in 0..1000
{
    let noise = random_generator.gen_range(-0.1..0.1);
    prediction_points[iter] = vec![0.0 + noise, (3.14/2.0) + noise, 10.0 + noise];
}

let prediction_metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);

assert!(prediction_metrics.root_mean_squared_error < 0.01);
assert!(prediction_metrics.mean_absolute_error < 0.01);
assert!(prediction_metrics.mean_squared_error < 1e-5);
assert!(prediction_metrics.r_squared > 0.99999);
assert!(prediction_metrics.adjusted_r_squared > 0.99999);
```
