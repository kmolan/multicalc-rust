use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use multicalc::approximation::linear_approximation::LinearApproximator;
use multicalc::approximation::quadratic_approximation::QuadraticApproximator;
use multicalc::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use multicalc::numerical_derivative::finite_difference::{
    FiniteDifferenceMulti, FiniteDifferenceSingle,
};
use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::iterative_integration::IterativeSingle;
use multicalc::numerical_integration::mode::{GaussianQuadratureMethod, IterativeMethod};
use multicalc::vector_field::{curl, divergence, flux_integral, line_integral};

fn differentiation(c: &mut Criterion) {
    let cube = |x: f64| x * x * x;
    let derivator = FiniteDifferenceSingle::default();

    c.bench_function("derivative/single_order_1", |b| {
        b.iter(|| derivator.get(black_box(1), &cube, black_box(2.0)).unwrap())
    });
    c.bench_function("derivative/single_order_2", |b| {
        b.iter(|| derivator.get(black_box(2), &cube, black_box(2.0)).unwrap())
    });
    c.bench_function("derivative/single_order_3", |b| {
        b.iter(|| derivator.get(black_box(3), &cube, black_box(2.0)).unwrap())
    });

    let multi = FiniteDifferenceMulti::default();
    let func = |a: &[f64; 3]| a[1] * a[0].sin() + a[0] * a[1].cos() + a[0] * a[1] * a[2].exp();
    let point = [1.0, 2.0, 3.0];

    c.bench_function("derivative/multi_single_partial", |b| {
        b.iter(|| {
            multi
                .get_single_partial(&func, black_box(0), black_box(&point))
                .unwrap()
        })
    });
    c.bench_function("derivative/multi_mixed_partial", |b| {
        b.iter(|| {
            multi
                .get(&func, black_box(&[0usize, 1]), black_box(&point))
                .unwrap()
        })
    });
}

fn iterative_integration(c: &mut Criterion) {
    let poly = |x: f64| 2.0 * x;
    let decay = |x: f64| (-x * x).exp();

    let boole = IterativeSingle::from_parameters(120, IterativeMethod::Booles);
    let simpson = IterativeSingle::from_parameters(120, IterativeMethod::Simpsons);
    let trapezoidal = IterativeSingle::from_parameters(120, IterativeMethod::Trapezoidal);

    let finite = [0.0, 2.0];
    c.bench_function("iterative/boole_single", |b| {
        b.iter(|| boole.get_single(&poly, black_box(&finite)).unwrap())
    });
    c.bench_function("iterative/simpson_single", |b| {
        b.iter(|| simpson.get_single(&poly, black_box(&finite)).unwrap())
    });
    c.bench_function("iterative/trapezoidal_single", |b| {
        b.iter(|| trapezoidal.get_single(&poly, black_box(&finite)).unwrap())
    });

    // same integrand and method, finite vs infinite limits: the domain transform should
    // be paid only on the infinite domain (the finite fast path)
    let finite_decay = [-5.0, 5.0];
    let infinite = [f64::NEG_INFINITY, f64::INFINITY];
    c.bench_function("iterative/boole_decay_finite", |b| {
        b.iter(|| boole.get_single(&decay, black_box(&finite_decay)).unwrap())
    });
    c.bench_function("iterative/boole_decay_infinite", |b| {
        b.iter(|| boole.get_single(&decay, black_box(&infinite)).unwrap())
    });

    let double = [[0.0, 1.0], [0.0, 1.0]];
    c.bench_function("iterative/boole_double_fold", |b| {
        b.iter(|| boole.get(&poly, black_box(&double)).unwrap())
    });
}

fn gaussian_integration(c: &mut Criterion) {
    let poly = |x: f64| 4.0 * x * x * x - 3.0 * x * x;
    let square = |x: f64| x * x;

    let legendre_4 = GaussianSingle::from_parameters(4, GaussianQuadratureMethod::GaussLegendre);
    let legendre_16 = GaussianSingle::from_parameters(16, GaussianQuadratureMethod::GaussLegendre);
    let finite = [0.0, 2.0];
    c.bench_function("gaussian/legendre_order_4", |b| {
        b.iter(|| legendre_4.get_single(&poly, black_box(&finite)).unwrap())
    });
    c.bench_function("gaussian/legendre_order_16", |b| {
        b.iter(|| legendre_16.get_single(&poly, black_box(&finite)).unwrap())
    });

    let hermite = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussHermite);
    let hermite_limit = [f64::NEG_INFINITY, f64::INFINITY];
    c.bench_function("gaussian/hermite_order_5", |b| {
        b.iter(|| {
            hermite
                .get_single(&square, black_box(&hermite_limit))
                .unwrap()
        })
    });

    let laguerre = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussLaguerre);
    let laguerre_limit = [0.0, f64::INFINITY];
    c.bench_function("gaussian/laguerre_order_5", |b| {
        b.iter(|| {
            laguerre
                .get_single(&square, black_box(&laguerre_limit))
                .unwrap()
        })
    });
}

fn jacobian_hessian(c: &mut Criterion) {
    let f1 = |a: &[f64; 3]| a[0] * a[1] * a[2];
    let f2 = |a: &[f64; 3]| a[0] * a[0] + a[1] * a[1];
    let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&f1, &f2];
    let point = [1.0, 2.0, 3.0];

    let jacobian = Jacobian::<FiniteDifferenceMulti>::default();
    c.bench_function("jacobian/2_funcs_3_vars", |b| {
        b.iter(|| {
            jacobian
                .get(black_box(&function_matrix), black_box(&point))
                .unwrap()
        })
    });

    let func = |a: &[f64; 3]| a[1] * a[0].sin() + 2.0 * a[0] * a[1].exp() + a[2] * a[2];
    let hessian = Hessian::<FiniteDifferenceMulti>::default();
    c.bench_function("hessian/3_vars", |b| {
        b.iter(|| hessian.get(&func, black_box(&point)).unwrap())
    });
}

fn vector_field(c: &mut Criterion) {
    let derivator = FiniteDifferenceMulti::default();

    let vfx = |a: &[f64; 3]| a[1];
    let vfy = |a: &[f64; 3]| -a[0];
    let vfz = |a: &[f64; 3]| 2.0 * a[2];
    let field: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&vfx, &vfy, &vfz];
    let point = [1.0, 2.0, 3.0];

    c.bench_function("vector_field/curl_3d", |b| {
        b.iter(|| curl::get_3d(derivator, black_box(&field), black_box(&point)).unwrap())
    });
    c.bench_function("vector_field/divergence_3d", |b| {
        b.iter(|| divergence::get_3d(derivator, black_box(&field), black_box(&point)).unwrap())
    });

    let line_field: [&dyn Fn(&[f64; 2]) -> f64; 2] =
        [&(|a: &[f64; 2]| a[1]), &(|a: &[f64; 2]| -a[0])];
    let transforms: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
    let limit = [0.0, std::f64::consts::TAU];

    c.bench_function("vector_field/line_integral_2d", |b| {
        b.iter(|| {
            line_integral::get_2d(
                black_box(&line_field),
                black_box(&transforms),
                black_box(&limit),
            )
            .unwrap()
        })
    });
    c.bench_function("vector_field/flux_integral_2d", |b| {
        b.iter(|| {
            flux_integral::get_2d(
                black_box(&line_field),
                black_box(&transforms),
                black_box(&limit),
            )
            .unwrap()
        })
    });
}

fn approximation(c: &mut Criterion) {
    let func = |a: &[f64; 3]| a[0] + a[1] * a[1] + a[2] * a[2] * a[2];
    let point = [1.0, 2.0, 3.0];

    let linear = LinearApproximator::<FiniteDifferenceMulti>::default();
    c.bench_function("approximation/linear_build", |b| {
        b.iter(|| linear.get(&func, black_box(&point)).unwrap())
    });
    let linear_result = linear.get(&func, &point).unwrap();
    c.bench_function("approximation/linear_predict", |b| {
        b.iter(|| linear_result.predict(black_box(&point)))
    });

    let quadratic = QuadraticApproximator::<FiniteDifferenceMulti>::default();
    c.bench_function("approximation/quadratic_build", |b| {
        b.iter(|| quadratic.get(&func, black_box(&point)).unwrap())
    });
    let quadratic_result = quadratic.get(&func, &point).unwrap();
    c.bench_function("approximation/quadratic_predict", |b| {
        b.iter(|| quadratic_result.predict(black_box(&point)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets =
        differentiation,
        iterative_integration,
        gaussian_integration,
        jacobian_hessian,
        vector_field,
        approximation
}
criterion_main!(benches);
