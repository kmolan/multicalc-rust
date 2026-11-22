//! Discretization: zero-order hold on a double integrator, Van Loan process noise, the discrete
//! white-noise model, and a one-`Dual` derivative through `expm`.
//!
//! Run with: `cargo run -p multicalc-demos --example discretization`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use multicalc::linear_algebra::{Matrix, Matrix2D};
use multicalc::scalar::Dual;

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<22} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    let timestep = 0.1;

    // (1) ZOH of the double integrator: F = [[1, dt], [0, 1]], G = [[dt²/2], [dt]].
    let a = Matrix2D::new([[0.0, 1.0], [0.0, 0.0]]);
    let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
    let (f, b_disc) = zoh::<2, 1, 3, f64>(a, b, timestep).unwrap();
    println!("ZOH double integrator (dt = {timestep})");
    report("F[0,1]", f[(0, 1)], timestep);
    report("G[0,0]", b_disc[(0, 0)], timestep * timestep / 2.0);
    report("G[1,0]", b_disc[(1, 0)], timestep);

    // (2) Van Loan process-noise discretization.
    let process_cov = Matrix2D::new([[0.0, 0.0], [0.0, 1.0]]);
    let (_f, discrete_cov) = van_loan::<2, 4, f64>(a, process_cov, timestep).unwrap();
    println!("\nVan Loan Q_d (continuous white noise on velocity)");
    report("Q_d[1,1]", discrete_cov[(1, 1)], timestep);
    report(
        "symmetry err",
        (discrete_cov[(0, 1)] - discrete_cov[(1, 0)]).abs(),
        0.0,
    );

    // (3) Discrete white-noise model (filterpy-compatible).
    let q_noise = q_discrete_white_noise::<2, f64>(timestep, 2.0);
    println!("\nq_discrete_white_noise(dim = 2, var = 2.0)");
    report("Q[0,0]", q_noise[(0, 0)], 2.0 * timestep.powi(4) / 4.0);
    report("Q[1,1]", q_noise[(1, 1)], 2.0 * timestep * timestep);

    // (4) Autodiff: d/dx expm(x·M)|_{x=0} = M, one Dual through expm.
    let seed = Matrix2D::new([[0.2, 0.5], [-0.1, 0.3]]);
    let a_dual = Matrix2D::<Dual<f64>>::from_fn(|i, j| Dual::new(0.0, seed[(i, j)]))
        .expm()
        .unwrap();
    println!("\nAutodiff: d/dx expm(x·M) at x = 0 equals M");
    report("d/dx [0,1]", a_dual[(0, 1)].deriv, seed[(0, 1)]);
    report("d/dx [1,0]", a_dual[(1, 0)].deriv, seed[(1, 0)]);
}
