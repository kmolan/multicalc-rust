//! Linear and quadratic (Taylor) approximation of a function about a point, plus
//! goodness-of-fit metrics.
//!
//! Run with: `cargo run --example approximation`

use multicalc::approximation::linear_approximation::LinearApproximator;
use multicalc::approximation::quadratic_approximation::QuadraticApproximator;
use multicalc::scalar::{ScalarFnN, c};
use multicalc::scalar_fn;

fn main() {
    // ---- linearize f(x, y, z) = x + y^2 + z^3 about (1, 2, 3) ----
    let f = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] * v[1] + v[2] * v[2] * v[2]);
    let base = [1.0, 2.0, 3.0];

    let linear: LinearApproximator = LinearApproximator::default();
    let model = linear.get(&f, &base).unwrap();

    println!("Linear model of x + y^2 + z^3 about {base:?}");
    println!(
        "  predict(base)        = {:.6}   (truth {:.6})",
        model.predict(&base),
        f.eval(&base)
    );
    let nearby = [1.1, 2.1, 3.1];
    println!(
        "  predict({nearby:?}) = {:.6}   (truth {:.6})",
        model.predict(&nearby),
        f.eval(&nearby)
    );

    // goodness-of-fit metrics over a spread of sample points
    let samples = [
        [1.0, 2.0, 3.0],
        [1.1, 2.05, 3.1],
        [0.9, 1.95, 2.9],
        [1.2, 2.1, 3.2],
        [0.8, 1.9, 2.8],
    ];
    let metrics = model.get_prediction_metrics(&samples, &f);
    println!(
        "  over {} points: RMSE = {:.4}, R^2 = {:.5}",
        samples.len(),
        metrics.root_mean_squared_error,
        metrics.r_squared
    );

    // ---- quadratic approximation of e^(x/2) + sin(y) + 2z about (0, pi/2, 10) ----
    let g = scalar_fn!(|v: &[f64; 3]| (c(0.5) * v[0]).exp() + v[1].sin() + c(2.0) * v[2]);
    let base = [0.0, std::f64::consts::FRAC_PI_2, 10.0];

    let quadratic: QuadraticApproximator = QuadraticApproximator::default();
    let model = quadratic.get(&g, &base).unwrap();

    println!("\nQuadratic model of e^(x/2) + sin(y) + 2z about (0, pi/2, 10)");
    let nearby = [0.1, std::f64::consts::FRAC_PI_2 + 0.1, 10.1];
    println!(
        "  predict({nearby:?}) = {:.6}   (truth {:.6})",
        model.predict(&nearby),
        g.eval(&nearby)
    );
}
