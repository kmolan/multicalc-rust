//! Estimation: a linear Kalman filter tracking a constant-velocity target — an exact two-step
//! hand check, a noisy track, Joseph vs naive covariance, a control input, innovation gating, and
//! one `Dual` derivative through an update.
//!
//! Run with: `cargo run -p multicalc-demos --example estimation`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::discretization::q_discrete_white_noise;
use multicalc::estimation::{CovarianceUpdate, KalmanFilter};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric};

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<22} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

/// A constant-velocity tracker over a 1 s step: position integrates velocity, position is measured.
fn tracker<T: Numeric>(
    initial_covariance: Matrix<2, 2, T>,
    process_noise: Matrix<2, 2, T>,
    measurement_noise: Matrix<1, 1, T>,
) -> KalmanFilter<2, 1, T> {
    KalmanFilter::new(
        Vector::new([T::ZERO, T::ZERO]),
        initial_covariance,
        Matrix::new([[T::ONE, T::ONE], [T::ZERO, T::ONE]]),
        Matrix::new([[T::ONE, T::ZERO]]),
        process_noise,
        measurement_noise,
    )
}

fn trace(m: Matrix<2, 2>) -> f64 {
    m[(0, 0)] + m[(1, 1)]
}

fn main() {
    // (1) Two steps with no process noise, worked by hand: with P0 = I, R = 1 and z = [1, 2],
    // the posterior is x = [5/3, 2/3] and P = [[2/3, 1/3], [1/3, 1/3]].
    let mut filter = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[1.0]]));
    for z in [1.0, 2.0] {
        filter.predict();
        filter.update(Vector::new([z])).unwrap();
    }
    println!("Exact two-step filter (P0 = I, Q = 0, R = 1, z = [1, 2])");
    report("position", filter.state()[0], 5.0 / 3.0);
    report("velocity", filter.state()[1], 2.0 / 3.0);
    report("P[0,0]", filter.covariance()[(0, 0)], 2.0 / 3.0);
    report("P[0,1] = P[1,0]", filter.covariance()[(0, 1)], 1.0 / 3.0);
    report(
        "innovation covariance",
        filter.innovation_covariance()[(0, 0)],
        3.0,
    );

    // (2) Track a target moving at 1 m/s from a standing start, measuring position only.
    // Truth is position = t; the measurements below are that track plus fixed noise.
    let measurements = [1.15, 1.87, 3.21, 3.94, 5.02, 6.13, 6.89, 8.05];
    let mut filter = tracker::<f64>(
        Matrix::new([[10.0, 0.0], [0.0, 10.0]]),
        q_discrete_white_noise::<2, f64>(1.0, 0.01),
        Matrix::new([[0.25]]),
    );
    let initial_uncertainty = trace(filter.covariance());
    println!("\nTracking a 1 m/s target from x = [0, 0] with a wide prior");
    for (step, z) in measurements.iter().enumerate() {
        filter.predict();
        filter.update(Vector::new([*z])).unwrap();
        println!(
            "  t = {:>2}s  measured {:>5.2}  ->  position {:>6.3}, velocity {:>6.3}  (trace P {:>7.4})",
            step + 1,
            z,
            filter.state()[0],
            filter.state()[1],
            trace(filter.covariance()),
        );
    }
    let (position, velocity) = (filter.state()[0], filter.state()[1]);
    assert!(
        (position - 8.0).abs() < 0.3,
        "position should converge on the truth track"
    );
    assert!(
        (velocity - 1.0).abs() < 0.3,
        "velocity should converge on 1 m/s"
    );
    assert!(
        trace(filter.covariance()) < initial_uncertainty,
        "measurements must reduce uncertainty"
    );
    println!("  velocity recovered from position measurements alone: {velocity:.4} m/s");

    // (3) Joseph vs naive. They agree in exact arithmetic; Joseph keeps symmetry under rounding.
    let mut naive = tracker::<f64>(
        Matrix::new([[10.0, 0.0], [0.0, 10.0]]),
        q_discrete_white_noise::<2, f64>(1.0, 0.01),
        Matrix::new([[0.25]]),
    )
    .with_covariance_update(CovarianceUpdate::Naive);
    for z in measurements.iter() {
        naive.predict();
        naive.update(Vector::new([*z])).unwrap();
    }
    println!("\nJoseph (default) vs naive covariance update");
    report("state agreement", (naive.state()[0] - position).abs(), 0.0);
    report(
        "Joseph symmetry err",
        (filter.covariance()[(0, 1)] - filter.covariance()[(1, 0)]).abs(),
        0.0,
    );
    assert!(
        filter.covariance().cholesky().is_ok(),
        "Joseph stays positive definite"
    );

    // (4) A driven system: constant acceleration enters through the control model.
    let mut driven = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[0.25]]));
    let control_model = Matrix::<2, 1>::new([[0.5], [1.0]]); // [dt²/2; dt] for dt = 1
    for z in [0.4, 2.1, 4.4, 8.2] {
        driven.predict_with_control(control_model, Vector::new([1.0])); // 1 m/s² command
        driven.update(Vector::new([z])).unwrap();
    }
    println!("\nDriven filter (1 m/s² command through the control model)");
    println!(
        "  after 4 s: position {:.3}, velocity {:.3}",
        driven.state()[0],
        driven.state()[1]
    );
    assert!(
        driven.state()[1] > 1.0,
        "a sustained acceleration command must build velocity"
    );

    // (5) Innovation gating: the normalized innovation squared flags an outlier measurement.
    let mut gated = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[0.25]]));
    gated.predict();
    gated.update(Vector::new([1.0])).unwrap();
    let consistent = gated.normalized_innovation_squared().unwrap();
    gated.predict();
    gated.update(Vector::new([50.0])).unwrap();
    let outlier = gated.normalized_innovation_squared().unwrap();
    println!("\nInnovation gating (yᵀ·S⁻¹·y)");
    println!("  consistent measurement: {consistent:.3}");
    println!("  outlier  measurement:   {outlier:.3}");
    assert!(
        outlier > consistent,
        "an outlier must score higher than a consistent measurement"
    );

    // (6) Autodiff: d(posterior position)/d(measurement) is the Kalman gain. Check it against the
    // gain formed independently from the prior covariance and the innovation covariance.
    let mut reference = tracker::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[0.25]]));
    reference.predict();
    let prior_position_variance = reference.covariance()[(0, 0)];
    reference.update(Vector::new([1.0])).unwrap();
    let gain = prior_position_variance / reference.innovation_covariance()[(0, 0)];

    let mut differentiated = tracker::<Dual<f64>>(
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::new([[Dual::constant(0.25)]]),
    );
    differentiated.predict();
    differentiated
        .update(Vector::new([Dual::variable(1.0)]))
        .unwrap();

    println!("\nAutodiff: d(position)/d(measurement) equals the Kalman gain");
    report("d(position)/dz", differentiated.state()[0].deriv, gain);
}
