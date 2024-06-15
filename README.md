# multicalc
[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
![github](https://github.com/kmolan/multicalc-rust/actions/workflows/build-tests.yml/badge.svg)
![github](https://github.com/kmolan/multicalc-rust/actions/workflows/code-coverage.yml/badge.svg)

Rust scientific computing for single and multi-variable calculus

## Salient Features

- Written in pure, safe rust
- Fully documented with code examples
- Comprehensive suite of tests for full code coverage, including all possible error conditions
- Trait-based generic implementation to support floating point and complex numbers
- Supports linear, polynomial, trigonometric, exponential, and any complex equation you can throw at it, of any number of variables!
  - Single, double and triple differentiation - total and partial
  - Single and double integration - total and partial
  - Jacobians and Hessians
  - Vector Field Calculus: Line and flux integrals, curl and divergence
  - Approximation of any given equation to a linear or quadratic model

## Table of Contents


- [1. Single total derivatives](#1-single-total-derivatives)
- [2. Single partial derivatives](#2-single-partial-derivatives)
- [3. Double total derivatives](#3-double-total-derivatives)
- [4. Double partial derivatives](#4-double-partial-derivatives)
- [5. Single total integrals](#5-single-total-integrals)
- [6. Single partial integrals](#6-single-partial-integrals)
- [7. Double total integrals](#7-double-total-integrals)
- [8. Double partial integrals](#8-double-partial-integrals)
- [9. Jacobians](#9-jacobians)
- [10. Hessians](#10-hessians)
- [11. Linear approximation](#11-linear-approximation)
- [12. Quadratic approximation](#12-quadratic-approximation)
- [13. Line and Flux integrals](#13-line-and-flux-integrals)
- [14. Curl and Divergence](#14-curl-and-divergence)

## 1. Single total derivatives
```rust
//function is x*x/2.0, derivative is known to be x
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[0]*args[0]/2.0;
};

//total derivative around x = 2.0, expect a value of 2.00
let val = single_derivative::get_total(&func, 2.0, 0.001);
assert!(f64::abs(val - 2.0) < 0.000001); //numerical error less than 1e-6
```

## 2. Single partial derivatives
```rust
//function is y*sin(x) + x*cos(y) + x*y*e^z
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
};

let point = vec![1.0, 2.0, 3.0];
let idx_to_derivate = 0;

//partial derivate for (x, y, z) = (1.0, 2.0, 3.0), partial derivative for x is known to be y*cos(x) + cos(y) + y*e^z
let val = single_derivative::get_partial(&func, idx_to_derivate, &point, 0.001);
let expected_value = 2.0*f64::cos(1.0) + f64::cos(2.0) + 2.0*f64::exp(3.0);
assert!(f64::abs(val - expected_value) < 0.000001); //numerical error less than 1e-6
```

## 3. Double total derivatives
```rust
//function is x*Sin(x)
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[0]*args[0].sin();
};

//double derivative at x = 1.0
let val = double_derivative::get_total(&func, 1.0, 0.001);
let expected_val = 2.0*f64::cos(1.0) - 1.0*f64::sin(1.0);
assert!(f64::abs(val - expected_val) < 0.000001); //numerical error less than 1e-6
```

## 4. Double partial derivatives
```rust
//function is y*sin(x) + x*cos(y) + x*y*e^z
let func = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
{ 
    return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
};

let point = vec![num_complex::c64(1.0, 3.5), num_complex::c64(2.0, 2.0), num_complex::c64(3.0, 0.0)];

let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
//partial derivate for (x, y, z) = (1.0 + 3.5i, 2.0 + 2.0i, 3.0 + 0.0i), known to be cos(x) - sin(y) + e^z
let val = double_derivative::get_partial(&func, &idx, &point, 0.001);
let expected_value = point[0].cos() - point[1].sin() + point[2].exp();
assert!(num_complex::ComplexFloat::abs(val.re - expected_value.re) < 0.0001); //numerical error less than 1e-4
assert!(num_complex::ComplexFloat::abs(val.im - expected_value.im) < 0.0001); //numerical error less than 1e-4
```

## 5. Single total integrals
```rust
//equation is 2.0*x
let func = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
{ 
    return 2.0*args[0];
};

//integrate from (0.0 + 0.0i) to (2.0 + 2.0i)
let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];

//simple integration for x, known to be x*x, expect a value of 0.00 + 8.0i
let val = single_integration::get_total(IntegrationMethod::Booles, &func, &integration_limit, 100);
assert!(num_complex::ComplexFloat::abs(val.re - 0.0) < 0.00001);
assert!(num_complex::ComplexFloat::abs(val.im - 8.0) < 0.00001);
```

## 6. Single partial integrals
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
assert!(f64::abs(val - 7.0) < 0.00001); //numerical error less than 1e-5
```

## 7. Double total integrals
```rust
//equation is 6.0*x
let func = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
{ 
    return 6.0*args[0];
};

//integrate over (0.0 + 0.0i) to (2.0 + 1.0i) twice
let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)]];

//simple double integration for 6*x, expect a value of 6.0 + 33.0i
let val = double_integration::get_total(IntegrationMethod::Booles, &func, &integration_limits, 20);
assert!(num_complex::ComplexFloat::abs(val.re - 6.0) < 0.00001);
assert!(num_complex::ComplexFloat::abs(val.im - 33.0) < 0.00001);
```

## 8. Double partial integrals
```rust
//equation is 2.0*x + y*z
let func = | args: &Vec<f64> | -> f64 
{ 
    return 2.0*args[0] + args[1]*args[2];
};

let integration_intervals = [[0.0, 1.0], [0.0, 1.0]];
let point = vec![1.0, 1.0, 1.0];
let idx_to_integrate = [0, 1];

//double partial integration for first x then y, expect a value of ~1.50
let val = double_integration::get_partial(IntegrationMethod::Booles, &func, idx_to_integrate, &integration_intervals, &point, 20);
assert!(f64::abs(val - 1.50) < 0.00001);  //numerical error less than 1e-5
```

## 9. Jacobians
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

let function_vector: Vec<Box<dyn Fn(&Vec<f64>) -> f64>> = vec![Box::new(func1), Box::new(func2)];

let points = vec![1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

let jacobian_matrix = jacobian::get(&function_vector, &points);

let expected_result = vec![vec![6.0, 3.0, 2.0], vec![2.0, 4.0, 0.0]];
for i in 0..function_vector.len()
{
    for j in 0..points.len()
    {
        //numerical error less than 1e-6
        assert!(f64::abs(jacobian_matrix[i][j] - expected_result[i][j]) < 0.000001);
    }
}
```

## 10. Hessians
```rust
//function is y*sin(x) + 2*x*e^y
let func = | args: &Vec<f64> | -> f64 
{ 
    return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
};

let points = vec![1.0, 2.0]; //the point around which we want the hessian matrix

let hessian_matrix = hessian::get(&func, &points);

let expected_result = vec![vec![-2.0*f64::sin(1.0), f64::cos(1.0) + 2.0*f64::exp(2.0)], vec![f64::cos(1.0) + 2.0*f64::exp(2.0), 2.0*f64::exp(2.0)]];

for i in 0..points.len()
{
    for j in 0..points.len()
    {            
        //numerical error less than 1e-4
        assert!(f64::abs(hessian_matrix[i][j] - expected_result[i][j]) < 0.0001);
    }
}
```

## 11. Linear approximation
```rust
//function is x + y^2 + z^3, which we want to linearize
let function_to_approximate = | args: &Vec<f64> | -> f64 
{ 
    return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
};

let point = vec![1.0, 2.0, 3.0]; //the point we want to linearize around

let result = linear_approximation::get(&function_to_approximate, &point);
```

## 12. Quadratic approximation
```rust
//function is e^(x/2) + sin(y) + 2.0*z
let function_to_approximate = | args: &Vec<f64> | -> f64 
{ 
    return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
};

let point = vec![0.0, 3.14/2.0, 10.0]; //the point we want to approximate around

let result = quadratic_approximation::get(&function_to_approximate, &point);
```

## 13. Line and Flux integrals
```rust
//vector field is (y, -x). On a 2D plane this would like a tornado rotating counter-clockwise
//curve is a unit circle, defined by (Cos(t), Sin(t))
//limit t goes from 0->2*pi

let vector_field_matrix: [Box<dyn Fn(&f64, &f64) -> f64>; 2] = [Box::new(|_:&f64, y:&f64|-> f64 { *y }), Box::new(|x:&f64, _:&f64|-> f64 { -x })];

let transformation_matrix: [Box<dyn Fn(&f64) -> f64>; 2] = [Box::new(|t:&f64|->f64 { t.cos() }), Box::new(|t:&f64|->f64 { t.sin() })];

let integration_limit = [0.0, 6.28];

//line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
assert!(f64::abs(val + 6.28) < 0.01);

//flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit, 100);
assert!(f64::abs(val - 0.0) < 0.01);
```

## 14. Curl and Divergence
```rust
//vector field is (2*x*y, 3*cos(y))
let vector_field_matrix: [Box<dyn Fn(&Vec<f64>) -> f64>; 2] = [Box::new(|args: &Vec<f64>|-> f64 { 2.0*args[0]*args[1] }), Box::new(|args: &Vec<f64>|-> f64 { 3.0*args[1].cos() })];

let point = vec![1.0, 3.14]; //the point of interest

//curl is known to be -2*x, expect and answer of -2.0
let val = curl::get_2d(&vector_field_matrix, &point);
assert!(f64::abs(val + 2.0) < 0.000001); //numerical error less than 1e-6

//divergence is known to be 2*y - 3*sin(y), expect and answer of 6.27
let val = divergence::get_2d(&vector_field_matrix, &point);
assert!(f64::abs(val - 6.27) < 0.01);
```

## Contributions Guide
See [CONTRIBUTIONS.md](./CONTRIBUTIONS.md)

## LICENSE
multicalc is licensed under the MIT license.

## Acknowledgements
multicalc uses [num-complex](https://crates.io/crates/num-complex) to provide a generic functionality for all floating type and complex numbers

## Contact
anmolkathail@gmail.com

## TODO
- &vec<T> -> &[T]
- Gauss-Kronrod Quadrature integration
