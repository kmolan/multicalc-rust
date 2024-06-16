use crate::numerical_integration::mode::IntegrationMethod;
use crate::numerical_integration::single_integration;
use crate::numerical_integration::double_integration;
use crate::utils::error_codes::ErrorCode;
 
#[test]
fn test_booles_integration_1()
{
    //equation is 2.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Booles, &func, &integration_limit, 100).unwrap();
    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_booles_integration_2()
{ 
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 0, &integration_limit, &point, 100).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 1, &integration_limit, &point, 100).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 2, &integration_limit, &point, 100).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_booles_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Booles, &func, &integration_limits, 20).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_booles_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 1.0, 1.0];

    //double partial integration for first x then y, expect a value of ~1.50
    let val = double_integration::get_partial(IntegrationMethod::Booles, &func, [0, 1], &integration_limits, &point, 20).unwrap();
    assert!(f64::abs(val - 1.50) < 0.00001);
}

#[test]
fn test_booles_integration_5()
{
    //equation is 2.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 2.0*args[0];
    };

    //integrate from (0.0 + 0.0i) to (2.0 + 2.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];

    //simple integration for x, known to be x*x, expect a value of 0.00 + 8.0i
    let val = single_integration::get_total(IntegrationMethod::Booles, &func, &integration_limit, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 0.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 8.0) < 0.00001);
}

#[test]
fn test_booles_integration_6()
{ 
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integrate from (0.0 + 0.0i) to (1.0 + 1.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)];
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    //partial integration for x, known to be x*x + x*y*z, expect a value of 5.0 + 9.0i
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 0, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 5.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 9.0) < 0.00001);

    //integrate from (0.0 + 0.0i) to (2.0 + 0.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 0.0)];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of 10.0 + 5.0i
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 1, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 10.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 5.0) < 0.00001);

    //integrate from (0.0 + 0.0i) to (3.0 + 0.5i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(3.0, 0.5)];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of 13.75 + 10.0i
    let val = single_integration::get_partial(IntegrationMethod::Booles, &func, 2, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 13.75) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 10.0) < 0.00001);
}

#[test]
fn test_booles_integration_7()
{
    //equation is 6.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 6.0*args[0];
    };

    //integrate over (0.0 + 0.0i) to (2.0 + 1.0i) twice
    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)]];

    //simple double integration for 6*x, expect a value of 6.0 + 33.0i
    let val = double_integration::get_total(IntegrationMethod::Booles, &func, &integration_limits, 20).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 6.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 33.0) < 0.00001);
}

#[test]
fn test_booles_integration_8()
{
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integrate over (0.0 + 0.0i) to (1.0 + 1.0i) twice
    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)]];
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    //double partial integration for first x then y, expect a value of -5.5 + 4.5i
    let val = double_integration::get_partial(IntegrationMethod::Booles, &func, [0, 1], &integration_limits, &point, 20).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re + 5.5) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 4.5) < 0.00001);
}

#[test]
#[cfg(feature = "std")] 
fn test_gauss_legendre_quadrature_integration_1()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[f64; 2] | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    let val = single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 2).unwrap();
    assert!(f64::abs(val - 8.0) < 0.00001);
}

#[test] 
#[cfg(feature = "std")]
fn test_gauss_legendre_quadrature_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 0, &integration_limit, &point, 2).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 1, &integration_limit, &point, 2).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 2, &integration_limit, &point, 2).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
#[cfg(feature = "std")]
fn test_gauss_legendre_quadrature_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limits, 2).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test] 
#[cfg(feature = "std")]
fn test_gauss_legendre_quadrature_integration_4()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    //integrate from (0.0 + 0.0i) to (2.0 + 2.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];

    //simple integration for x, known to be known to be x^4 - x^3, expect a value of -48.0 - 16.0i
    let val = single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 2).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re + 48.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im + 16.0) < 0.00001);
}

#[test] 
#[cfg(feature = "std")]
fn test_gauss_legendre_quadrature_integration_5()
{
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integration limit is (0.0 + 0.0i) to (1.0 + 0.5i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 0.5)];
    //the final point
    let point = [num_complex::c64(1.0, 0.5), num_complex::c64(2.0, 2.0), num_complex::c64(3.0, 0.0)];

    //partial integration for x, known to be x*x + x*y*z, expect a value of 3.75 + 10.0i
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 0, &integration_limit, &point, 2).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 3.75) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 10.0) < 0.00001);


    //integration limit is (0.0 + 0.0i) to (2.0 + 2.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of 2.0 + 18.0i
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 1, &integration_limit, &point, 2).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 2.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 18.0) < 0.00001);


    //integration limit is (0.0 + 0.0i) to (3.0 + 0.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(3.0, 0.0)];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of 15.0 + 12.0i
    let val = single_integration::get_partial(IntegrationMethod::GaussLegendre, &func, 2, &integration_limit, &point, 2).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 15.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 12.0) < 0.00001);
}

#[test]
#[cfg(feature = "std")]
fn test_gauss_legendre_quadrature_integration_6()
{
    //equation is 6.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)]];

    //simple double integration for 6*x, expect a value of -48.0 + 48.0i
    let val = double_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limits, 2).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re + 48.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 48.0) < 0.00001);
}

#[test]
fn test_simpsons_integration_1()
{
    //equation is 2.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_limit, 200).unwrap();
    assert!(f64::abs(val - 4.0) < 0.05);
}

#[test]
fn test_simpsons_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 0, &integration_limit, &point, 200).unwrap();
    assert!(f64::abs(val - 7.0) < 0.05);


    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 1, &integration_limit, &point, 200).unwrap();
    assert!(f64::abs(val - 10.0) < 0.05);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 2, &integration_limit, &point, 200).unwrap();
    assert!(f64::abs(val - 15.0) < 0.05);
}

#[test]
fn test_simpsons_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_limits, 200).unwrap();
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_simpsons_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 1.0, 1.0];

    //double partial integration for first x then y, expect a value of ~1.50
    let val = double_integration::get_partial(IntegrationMethod::Simpsons, &func, [0, 1], &integration_limits, &point, 200).unwrap();
    assert!(f64::abs(val - 1.50) < 0.05);
}

#[test]
fn test_simpsons_integration_5()
{
    //equation is 2.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 2.0*args[0];
    };

    //integrate from (0.0 + 0.0i) to (2.0 + 2.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];

    //simple integration for x, known to be x*x, expect a value of 0.00 + 8.0i
    let val = single_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_limit, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 0.0) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - 8.0) < 0.05);
}

#[test]
fn test_simpsons_integration_6()
{ 
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integrate from (0.0 + 0.0i) to (1.0 + 1.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)];
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    //partial integration for x, known to be x*x + x*y*z, expect a value of 5.0 + 9.0i
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 0, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 5.0) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - 9.0) < 0.05);

    //integrate from (0.0 + 0.0i) to (2.0 + 0.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 0.0)];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of 10.0 + 5.0i
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 1, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 10.0) < 0.05);
    assert!(num_complex::ComplexFloat::abs(val.im - 5.0) < 0.05);

    //integrate from (0.0 + 0.0i) to (3.0 + 0.5i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(3.0, 0.5)];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of 13.75 + 10.0i
    let val = single_integration::get_partial(IntegrationMethod::Simpsons, &func, 2, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 13.75) < 0.1);
    assert!(num_complex::ComplexFloat::abs(val.im - 10.0) < 0.1);
}

#[test]
fn test_simpsons_integration_7()
{
    //equation is 6.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 6.0*args[0];
    };

    //integrate over (0.0 + 0.0i) to (2.0 + 1.0i) twice
    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)]];

    //simple double integration for 6*x, expect a value of 6.0 + 33.0i
    let val = double_integration::get_total(IntegrationMethod::Simpsons, &func, &integration_limits, 20).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 6.0) < 0.7);
    assert!(num_complex::ComplexFloat::abs(val.im - 33.0) < 0.7);
}

#[test]
fn test_simpsons_integration_8()
{
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integrate over (0.0 + 0.0i) to (1.0 + 1.0i) twice
    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)]];
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    //double partial integration for first x then y, expect a value of -5.5 + 4.5i
    let val = double_integration::get_partial(IntegrationMethod::Simpsons, &func, [0, 1], &integration_limits, &point, 20).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re + 5.5) < 0.1);
    assert!(num_complex::ComplexFloat::abs(val.im - 4.5) < 0.1);
}

#[test]
fn test_trapezoidal_integration_1()
{
    //equation is 2.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 2.0*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 10).unwrap();
    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 0, &integration_limit, &point, 10).unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);

    
    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00 
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 1, &integration_limit, &point, 10).unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);


    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0 
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 2, &integration_limit, &point, 10).unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_3()
{
    //equation is 6.0*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 6.0*args[0];
    };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    //simple double integration for 6*x, expect a value of ~24.00
    let val = double_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limits, 10).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_4()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 2.0, 3.0];

    //double partial integration for first x then y, expect a value of ~2.50
    let val = double_integration::get_partial(IntegrationMethod::Trapezoidal, &func, [0, 1], &integration_limits, &point, 10).unwrap();
    assert!(f64::abs(val - 2.50) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_5()
{
    //equation is 2.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 2.0*args[0];
    };

    //integrate from (0.0 + 0.0i) to (2.0 + 2.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];

    //simple integration for x, known to be x*x, expect a value of 0.00 + 8.0i
    let val = single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 0.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 8.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_6()
{ 
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integrate from (0.0 + 0.0i) to (1.0 + 1.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)];
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    //partial integration for x, known to be x*x + x*y*z, expect a value of 5.0 + 9.0i
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 0, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 5.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 9.0) < 0.00001);

    //integrate from (0.0 + 0.0i) to (2.0 + 0.0i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 0.0)];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of 10.0 + 5.0i
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 1, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 10.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 5.0) < 0.00001);

    //integrate from (0.0 + 0.0i) to (3.0 + 0.5i)
    let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(3.0, 0.5)];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of 13.75 + 10.0i
    let val = single_integration::get_partial(IntegrationMethod::Trapezoidal, &func, 2, &integration_limit, &point, 100).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 13.75) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 10.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_7()
{
    //equation is 6.0*x
    let func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
    { 
        return 6.0*args[0];
    };

    //integrate over (0.0 + 0.0i) to (2.0 + 1.0i) twice
    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)]];

    //simple double integration for 6*x, expect a value of 6.0 + 33.0i
    let val = double_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limits, 20).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re - 6.0) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 33.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_8()
{
    //equation is 2.0*x + y*z
    let func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    //integrate over (0.0 + 0.0i) to (1.0 + 1.0i) twice
    let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)]];
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    //double partial integration for first x then y, expect a value of -5.5 + 4.5i
    let val = double_integration::get_partial(IntegrationMethod::Trapezoidal, &func, [0, 1], &integration_limits, &point, 20).unwrap();
    assert!(num_complex::ComplexFloat::abs(val.re + 5.5) < 0.00001);
    assert!(num_complex::ComplexFloat::abs(val.im - 4.5) < 0.00001);
}

#[test]
fn test_error_checking_1()
{
    //equation is 2.0*x + y*z

    use crate::utils::error_codes::ErrorCode;
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [10.0, 1.0];

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    let result = single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 10);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::IntegrationLimitsIllDefined);
}

#[test]
fn test_error_checking_2()
{
    //equation is 2.0*x + y*z
    let func = | args: &[f64; 3] | -> f64 
    { 
        return 2.0*args[0] + args[1]*args[2];
    };

    let integration_limit = [0.0, 1.0];

    //expect failure because number of steps is 0
    let result = single_integration::get_total(IntegrationMethod::Trapezoidal, &func, &integration_limit, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::NumberOfStepsCannotBeZero);
}

#[test]
#[cfg(feature = "std")]
fn test_error_checking_3()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n < 2
    let result = single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::GaussLegendreOrderOutOfRange);
}

#[test]
#[cfg(feature = "std")]
fn test_error_checking_4()
{
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = | args: &[f64; 1] | -> f64 
    { 
        return 4.0*args[0]*args[0]*args[0] - 3.0*args[0]*args[0];
    };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n > 20
    let result = single_integration::get_total(IntegrationMethod::GaussLegendre, &func, &integration_limit, 21);
    assert!(result.is_err());
    assert!(result.unwrap_err() == ErrorCode::GaussLegendreOrderOutOfRange);
}