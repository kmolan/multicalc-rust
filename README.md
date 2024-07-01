# multicalc
[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
![github](https://github.com/kmolan/multicalc-rust/actions/workflows/build-tests.yml/badge.svg)
![github](https://github.com/kmolan/multicalc-rust/actions/workflows/code-coverage.yml/badge.svg)

Rust scientific computing for single and multi-variable calculus

## Salient Features

- Written in pure, safe rust
- no-std with zero heap allocations and no panics
- Trait-based generic implementation to support floating point and complex numbers
- Fully documented with code examples
- Comprehensive suite of tests for full code coverage, including all possible error conditions
- Supports linear, polynomial, trigonometric, exponential, and any complex equation you can throw at it, of any number of variables!
  - Numerical differentiation of any order
      - Finite difference method, for total and partial differentiation
  - Numerical integration of any order
      - Iterative methods: Booles, Simpsons, Trapezoidal
      - Gaussian Quadratures: Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre
  - Jacobians and Hessians
  - Vector Field Calculus: Line and flux integrals, curl and divergence
  - Approximation of any given equation to a linear or quadratic mode


## Table of Contents


- [1. Derivatives for single variable equations](#1-derivatives-for-single-variable-equations)
- [2. Derivatives for multi variable equations](#2-derivatives-for-multi-variable-equations)
- [3. Integrals for single variable equations](#3-integrals-for-single-variable-equations)
- [4. Integrals for multi variable equations](#4-integrals-for-multi-variable-equations)
- [5. Jacobians](#5-jacobians)
- [6. Hessians](#6-hessians)
- [7. Linear approximation](#7-linear-approximation)
- [8. Quadratic approximation](#8-quadratic-approximation)
- [9. Line and Flux integrals](#9-line-and-flux-integrals)
- [10. Curl and Divergence](#10-curl-and-divergence)
- [11. Error Handling](#11-error-handling)
- [12. Experimental](#12-experimental)

## 1. Derivatives for single variable equations
```rust
// assume we want to differentiate f(x) = x^3
let func = | arg: f64 | -> f64 
{ 
    return arg*arg*arg;
};

let point = 2.0; //the point at which we want to differentiate

use multicalc::numerical_derivative::derivator::*;
use multicalc::numerical_derivative::finite_difference::*;
 
let derivator = SingleVariableSolver::default();
//alternatively, you can also create the derivator with custom parameters using SingleVariableSolver::from_parameters()

let val = derivator.get(1, &my_func, point).unwrap(); //single derivative
assert!(f64::abs(val - 12.0) < 1e-7);
let val = derivator.get(2, &my_func, point).unwrap(); //double derivative
assert!(f64::abs(val - 12.0) < 1e-5);
let val = derivator.get(3, &my_func, point).unwrap(); //triple derivative
assert!(f64::abs(val - 6.0) < 1e-3);

//for single and double derivatives, you can also just use the convenience wrappers
let val = derivator.get_single(&my_func, point).unwrap(); //single derivative
assert!(f64::abs(val - 12.0) < 1e-7);
let val = derivator.get_double(&my_func, point).unwrap(); //double derivative
assert!(f64::abs(val - 12.0) < 1e-5);

```

## 2. Derivatives for multi variable equations
```rust
//function is f(x,y,z) = y*sin(x) + x*cos(y) + x*y*e^z
let func = | args: &[f64; 3] | -> f64 
{ 
    return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
};

let point = [1.0, 2.0, 3.0]; //the point of interest

use multicalc::numerical_derivative::derivator::*;
use multicalc::numerical_derivative::finite_difference::*;

let derivator = MultiVariableSolver::default();
//alternatively, you can also create the derivator with custom parameters using MultiVariableSolver::from_parameters()

let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
let val = derivator.get(2, &my_func, &idx, &point).unwrap();
let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
assert!(f64::abs(val - expected_value) < 0.001);
 
let idx: [usize; 2] = [1, 1]; //partial double derivative d(df/dy)/dy 
let val = derivator.get(2, &my_func, &idx, &point).unwrap();
let expected_value = -1.0*f64::cos(2.0);
assert!(f64::abs(val - expected_value) < 0.0001);
```

## 3. Integrals for single variable equations
```rust

//integrate 2*x . the function would be:
let my_func = | arg: f64 | -> f64 
{ 
    return 2.0*arg;
};

use multicalc::numerical_integration::integrator::*;
use multicalc::numerical_integration::iterative_integration;

let integrator = iterative_integration::SingleVariableSolver::default();  
//alternatively, create a custom integrator using SingleVariableSolver::from_parameters()

let integration_limit = [[0.0, 2.0]; 1]; //desired integration limit 
let val = integrator.get(1, &my_func, &integration_limit).unwrap(); //single integration
assert!(f64::abs(val - 4.0) < 1e-6);

let integration_limit = [[0.0, 2.0], [-1.0, 1.0]]; //desired integration limits
let val = integrator.get(2, &my_func, &integration_limit).unwrap(); //double integration
assert!(f64::abs(val - 8.0) < 1e-6);
///
let integration_limit = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]; //desired integration limits
let val = integrator.get(3, &my_func, &integration_limit).unwrap(); //triple integration
assert!(f64::abs(val - 16.0) < 1e-6);


//instead of using iterative algorithms, use gaussian quadratures instead:
use multicalc::numerical_integration::gaussian_integration;
let integrator = gaussian_integration::SingleVariableSolver::default();  
//alternatively, create a custom integrator using SingleVariableSolver::from_parameters()

let integration_limit = [[0.0, 2.0]; 1]; //desired integration limit 
let val = integrator.get(1, &my_func, &integration_limit).unwrap(); //single integration
assert!(f64::abs(val - 4.0) < 1e-6);

//you can also just use convenience wrappers for single and double integration
let integration_limit = [0.0, 2.0];
let val = integrator.get_single(&my_func, &integration_limit).unwrap(); //single integration
assert!(f64::abs(val - 4.0) < 1e-6);

let integration_limit = [[0.0, 2.0], [-1.0, 1.0]];
let val = integrator.get_double(&my_func, &integration_limit).unwrap(); //double integration
assert!(f64::abs(val - 8.0) < 1e-6);
```

## 4. Integrals for multi variable equations
```rust
//equation is 2.0*x + y*z
let func = | args: &[f64; 3] | -> f64 
{ 
    return 2.0*args[0] + args[1]*args[2];
};

//for multivariable integration to work, we must know the final value of all variables
//for variables that are constant, they will have their constant values
//for variables that are being integrated, their values will be the final upper limit of integration
//in this example, we are keeping 'z' constant at 3.0, and integrating x and y to upper limits 1.0 and 2.0 respectively
let point = [1.0, 2.0, 3.0];

let iterator = iterative_integration::MultiVariableSolver::default();

//integration for x, known to be x*x + x*y*z, expect a value of ~7.00
let integration_limit = [0.0, 1.0];
let val = iterator.get_single_partial(&func, 0, &integration_limit, &point).unwrap();
assert!(f64::abs(val - 7.0) < 1e-7);


//integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
let integration_limit = [0.0, 2.0];
let val = iterator.get_single_partial(&func, 1, &integration_limit, &point).unwrap();
assert!(f64::abs(val - 10.0) < 1e-7);


//double integration for first x then y, expect a value of ~8.0
let integration_limit = [[0.0, 1.0], [0.0, 2.0]];
let val = integrator.get_double_partial(&func, [0, 1], &integration_limit, &point).unwrap();
assert!(f64::abs(val - 8.0) < 1e-7);

//triple integration for x, expect a value of ~7.0
let integration_limit = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
let val = integrator.get(3, [0, 0, 0], &func, &integration_limit, &point).unwrap();
assert!(f64::abs(val - 7.0) < 1e-7);
```

## 5. Jacobians
```rust
//function is x*y*z
let func1 = | args: &[f64; 3] | -> f64 
{ 
    return args[0]*args[1]*args[2];
};

//function is x^2 + y^2
let func2 = | args: &[f64; 3] | -> f64 
{ 
    return args[0].powf(2.0) + args[1].powf(2.0);
};

let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&func1, &func2];

let points = [1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;

let jacobian = Jacobian::<MultiVariableSolver>::default();
//instead of using the in-built MultiVariableSolver, you can also supply your own by implementing the base trait

let result = jacobian.get(&function_matrix, &points).unwrap();

let expected_result = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
for i in 0..function_vector.len()
{
    for j in 0..points.len()
    {
        //numerical error less than 1e-6
        assert!(f64::abs(result[i][j] - expected_result[i][j]) < 1e-6);
    }
}
```

## 6. Hessians
```rust

//function is y*sin(x) + 2*x*e^y
let func = | args: &[f64; 2] | -> f64 
{ 
    return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
};

let points = [1.0, 2.0]; //the point around which we want the hessian matrix

use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;

let hessian = Hessian::<MultiVariableSolver>::default();
//instead of using the in-built MultiVariableSolver, you can also supply your own by implementing the base trait

let result = hessian.get(&func, &points).unwrap();

assert!(result.len() == points.len()); //number of rows
assert!(result[0].len() == points.len()); //number of columns

let expected_result = [[-2.0*f64::sin(1.0), f64::cos(1.0) + 2.0*f64::exp(2.0)], 
                                        [f64::cos(1.0) + 2.0*f64::exp(2.0), 2.0*f64::exp(2.0)]];

for i in 0..points.len()
{
    for j in 0..points.len()
    {
        assert!(f64::abs(result[i][j] - expected_result[i][j]) < 1e-5);
    }
}
```

## 7. Linear approximation
```rust
//function is x + y^2 + z^3, which we want to linearize
let function_to_approximate = | args: &[f64; 3] | -> f64 
{ 
    return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
};

let point = [1.0, 2.0, 3.0]; //the point we want to linearize around

use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;

let approximator = LinearApproximator::<MultiVariableSolver>::default();
//instead of using the in-built MultiVariableSolver, you can also supply your own by implementing the base trait

let result = approximator.get(&function_to_approximate, &point).unwrap();
assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);
```

## 8. Quadratic approximation
```rust
//function is e^(x/2) + sin(y) + 2.0*z
let function_to_approximate = | args: &[f64; 3] | -> f64 
{ 
    return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
};

let point = [0.0, 3.14/2.0, 10.0]; //the point we want to approximate around

use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;

let approximator = QuadraticApproximator::<MultiVariableSolver>::default();
//instead of using the in-built MultiVariableSolver, you can also supply your own by implementing the base trait

let result = approximator.get(&function_to_approximate, &point).unwrap();

assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);
```

## 9. Line and Flux integrals
```rust
//vector field is (y, -x). On a 2D plane this would like a tornado rotating counter-clockwise
//curve is a unit circle, defined by (Cos(t), Sin(t))
//limit t goes from 0->2*pi

let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];

let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];

let integration_limit = [0.0, 6.28];

//line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit).unwrap();
assert!(f64::abs(val + 6.28) < 0.01);

//flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit).unwrap();
assert!(f64::abs(val - 0.0) < 0.01);
```

## 10. Curl and Divergence
```rust
//vector field is (2*x*y, 3*cos(y))
//x-component
let vf_x = | args: &[f64; 2] | -> f64 
{ 
    return 2.0*args[0]*args[1];
};

//y-component
let vf_y = | args: &[f64; 2] | -> f64 
{ 
    return 3.0*args[1].cos()
};

let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&vf_x, &vf_y];

let point = [1.0, 3.14]; //the point of interest

use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
let derivator = MultiVariableSolver::default();
//instead of using the in-built MultiVariableSolver, you can also supply your own by implementing the base trait

//curl is known to be -2*x, expect and answer of -2.0
let val = curl::get_2d(derivator, &vector_field_matrix, &point).unwrap();
assert!(f64::abs(val + 2.0) < 0.000001); //numerical error less than 1e-6

//divergence is known to be 2*y - 3*sin(y), expect and answer of 6.27
let val = divergence::get_2d(derivator, &vector_field_matrix, &point).unwrap();
assert!(f64::abs(val - 6.27) < 0.01);
```

## 11. Error Handling
Wherever possible, "safe" versions of functions are provided that fill in the default values and return the required solution directly.
However, that is not always possible either because no default argument can be assumed, or for functions that deliberately give users the freedom to tweak the parameters.
In such cases, a `Result<T, &'static str>` object is returned instead, where all possible `&'static str`s can be viewed at [error_codes](./src/utils/error_codes.rs).

##  12. Experimental
Enable feature "heap" to access `std::Vec` based methods in certain modules. Currently this is only supported for the _Jacobian_ module via `get_on_heap()` and `get_on_heap_custom()` methods. The output is a dynamically allocated `Vec<Vec<T>>`. This is to support large datasets that might otherwise get a stack overflow with static arrays. Future plans might include adding such support for the approximation module.
```rust
//function is x*y*z
let func1 = | args: &[f64; 3] | -> f64 
{ 
    return args[0]*args[1]*args[2];
};

//function is x^2 + y^2
let func2 = | args: &[f64; 3] | -> f64 
{ 
    return args[0].powf(2.0) + args[1].powf(2.0);
};

let function_matrix: Vec<Box<dyn Fn(&[f64; 3]) -> f64>> = std::vec![Box::new(func1), Box::new(func2)];

let points = [1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix

let jacobian = Jacobian::<MultiVariableSolver>::default();

let result: Vec<Vec<f64>> = jacobian.get_on_heap(&function_matrix, &points).unwrap();

assert!(result.len() == function_matrix.len()); //number of rows
assert!(result[0].len() == points.len()); //number of columns

let expected_result = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];

for i in 0..function_matrix.len()
{
    for j in 0..points.len()
    {
        assert!(f64::abs(result[i][j] - expected_result[i][j]) < 0.000001);
    }
}
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
- Gauss-Kronrod Quadrature integration
- infinity outputs