//! ODE integrators: fixed-step RK4 and adaptive RK45 (Dormand–Prince) on the harmonic
//! oscillator and three real dynamical systems — a two-link manipulator (acrobot), a
//! torque-free tumbling quadrotor, and an outer-solar-system N-body model. For the harmonic
//! case the exact solution is known; the other three have no closed form, so accuracy is
//! reported as the drift in a conserved quantity (energy, kinetic energy, quaternion norm).
//! These figures reproduce the accuracy table in `benches/ode.md`.
//!
//! Run with: `cargo run -p multicalc-demos --example ode`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::linear_algebra::Vector;
use multicalc::ode::{Rk4, Rk45};

fn main() {
    harmonic_oscillator();
    acrobot();
    quadrotor_attitude();
    solar_system_nbody();
}

// ----- harmonic oscillator y'' = -y, exact solution [cos t, -sin t] -----

fn harmonic_oscillator() {
    // y1' = y2 ; y2' = -y1
    let f = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
    let y0 = Vector::new([1.0, 0.0]);
    let exact = |t: f64| [t.cos(), -t.sin()];

    // RK4: 2000 fixed steps over [0, 2*pi].
    let steps = 2000;
    let dt = core::f64::consts::TAU / steps as f64;
    let mut max_err = 0.0_f64;
    let yf = Rk4::integrate(&f, 0.0, &y0, dt, steps, |t, y| {
        let e = exact(t);
        max_err = max_err.max((y[0] - e[0]).abs()).max((y[1] - e[1]).abs());
    });
    println!("Harmonic oscillator y'' = -y");
    println!("  RK4  {steps} steps over [0, 2*pi]");
    println!(
        "    y(2*pi) = [{:.12}, {:.12}]  max|err| = {max_err:.2e}",
        yf[0], yf[1]
    );
    assert!(
        max_err < 1e-3,
        "RK4 should track the exact harmonic solution"
    );

    // RK45: adaptive solve to t = 2*pi, then dense-output sampling.
    let solver = Rk45::default().with_rtol(1e-9).with_atol(1e-12);
    let yf = solver.solve(&f, 0.0, &y0, core::f64::consts::TAU).unwrap();
    let e = exact(core::f64::consts::TAU);
    println!("  RK45 adaptive solve to t = 2*pi (rtol 1e-9)");
    println!(
        "    y(2*pi) = [{:.12}, {:.12}]  |err| = {:.2e}",
        yf[0],
        yf[1],
        (yf[0] - e[0]).abs().max((yf[1] - e[1]).abs())
    );

    let times = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = [Vector::<2, f64>::zeros(); 7];
    solver
        .solve_on_grid(&f, 0.0, &y0, &times, &mut out)
        .unwrap();
    let grid_err = times
        .iter()
        .zip(out.iter())
        .map(|(&t, y)| {
            let e = exact(t);
            (y[0] - e[0]).abs().max((y[1] - e[1]).abs())
        })
        .fold(0.0_f64, f64::max);
    println!("    dense-output grid max|err| = {grid_err:.2e}");
    assert!(grid_err < 1e-6, "RK45 dense output should be accurate");
}

// Largest drift of the invariant `inv` from its initial value over an RK4 integration.
fn rk4_drift<const N: usize, F, I>(f: &F, y0: &Vector<N, f64>, dt: f64, steps: usize, inv: I) -> f64
where
    F: Fn(f64, &Vector<N, f64>) -> Vector<N, f64>,
    I: Fn(&Vector<N, f64>) -> f64,
{
    let q0 = inv(y0);
    let mut max = 0.0_f64;
    let _ = Rk4::integrate(f, 0.0, y0, dt, steps, |_, y| {
        max = max.max((inv(y) - q0).abs());
    });
    max
}

// Largest drift of the invariant `inv` over the accepted steps of an RK45 solve.
fn rk45_drift<const N: usize, F, I>(
    f: &F,
    y0: &Vector<N, f64>,
    tf: f64,
    rtol: f64,
    atol: f64,
    inv: I,
) -> f64
where
    F: Fn(f64, &Vector<N, f64>) -> Vector<N, f64>,
    I: Fn(&Vector<N, f64>) -> f64,
{
    let q0 = inv(y0);
    let mut max = 0.0_f64;
    let _ = Rk45::default()
        .with_rtol(rtol)
        .with_atol(atol)
        .for_each_step(f, 0.0, y0, tf, |step| {
            max = max.max((inv(&step.y1) - q0).abs());
        })
        .unwrap();
    max
}

// ----- A. Two-link planar manipulator (acrobot), N = 4 -----

const ACRO_G: f64 = 9.81;
const ACRO_M1: f64 = 1.0;
const ACRO_M2: f64 = 1.0;
const ACRO_L1: f64 = 1.0;
const ACRO_LC1: f64 = 0.5;
const ACRO_LC2: f64 = 0.5;
const ACRO_I1: f64 = 1.0 / 12.0;
const ACRO_I2: f64 = 1.0 / 12.0;

fn acrobot_mass(c2: f64) -> (f64, f64, f64) {
    let d11 = ACRO_M1 * ACRO_LC1 * ACRO_LC1
        + ACRO_M2 * (ACRO_L1 * ACRO_L1 + ACRO_LC2 * ACRO_LC2 + 2.0 * ACRO_L1 * ACRO_LC2 * c2)
        + ACRO_I1
        + ACRO_I2;
    let d12 = ACRO_M2 * (ACRO_LC2 * ACRO_LC2 + ACRO_L1 * ACRO_LC2 * c2) + ACRO_I2;
    let d22 = ACRO_M2 * ACRO_LC2 * ACRO_LC2 + ACRO_I2;
    (d11, d12, d22)
}

fn acrobot_rhs(_t: f64, y: &Vector<4, f64>) -> Vector<4, f64> {
    let (th1, th2, w1, w2) = (y[0], y[1], y[2], y[3]);
    let (d11, d12, d22) = acrobot_mass(th2.cos());
    let s2 = th2.sin();
    let h1 = -ACRO_M2 * ACRO_L1 * ACRO_LC2 * s2 * (2.0 * w1 * w2 + w2 * w2);
    let h2 = ACRO_M2 * ACRO_L1 * ACRO_LC2 * s2 * w1 * w1;
    let phi1 = (ACRO_M1 * ACRO_LC1 + ACRO_M2 * ACRO_L1) * ACRO_G * th1.cos()
        + ACRO_M2 * ACRO_LC2 * ACRO_G * (th1 + th2).cos();
    let phi2 = ACRO_M2 * ACRO_LC2 * ACRO_G * (th1 + th2).cos();
    let det = d11 * d22 - d12 * d12;
    let w1d = (-d22 * (h1 + phi1) + d12 * (h2 + phi2)) / det;
    let w2d = (d12 * (h1 + phi1) - d11 * (h2 + phi2)) / det;
    Vector::new([w1, w2, w1d, w2d])
}

fn acrobot_energy(y: &Vector<4, f64>) -> f64 {
    let (th1, th2, w1, w2) = (y[0], y[1], y[2], y[3]);
    let (d11, d12, d22) = acrobot_mass(th2.cos());
    let ke = 0.5 * (d11 * w1 * w1 + 2.0 * d12 * w1 * w2 + d22 * w2 * w2);
    let pe = ACRO_G
        * (ACRO_M1 * ACRO_LC1 * th1.sin()
            + ACRO_M2 * (ACRO_L1 * th1.sin() + ACRO_LC2 * (th1 + th2).sin()));
    ke + pe
}

fn acrobot() {
    let y0 = Vector::new([0.0, 0.0, 0.0, 0.0]);
    let tf = 10.0;
    let steps = 10_000;
    let rk4 = rk4_drift(&acrobot_rhs, &y0, tf / steps as f64, steps, acrobot_energy);
    let rk45 = rk45_drift(&acrobot_rhs, &y0, tf, 1e-8, 1e-10, acrobot_energy);
    println!("\nAcrobot (two-link manipulator, N=4): energy drift over [0, 10]");
    println!("  RK4  (dt = 1e-3)   max|E - E0| = {rk4:.2e}");
    println!("  RK45 (rtol 1e-8)   max|E - E0| = {rk45:.2e}");
}

// ----- B. Quadrotor attitude (torque-free tumble), N = 7 -----

const QUAD_IX: f64 = 0.01;
const QUAD_IY: f64 = 0.02;
const QUAD_IZ: f64 = 0.03;

fn quadrotor_rhs(_t: f64, y: &Vector<7, f64>) -> Vector<7, f64> {
    let (qw, qx, qy, qz) = (y[0], y[1], y[2], y[3]);
    let (wx, wy, wz) = (y[4], y[5], y[6]);
    let qwd = -0.5 * (qx * wx + qy * wy + qz * wz);
    let qxd = 0.5 * (qw * wx + qy * wz - qz * wy);
    let qyd = 0.5 * (qw * wy - qx * wz + qz * wx);
    let qzd = 0.5 * (qw * wz + qx * wy - qy * wx);
    let wxd = (QUAD_IY - QUAD_IZ) * wy * wz / QUAD_IX;
    let wyd = (QUAD_IZ - QUAD_IX) * wz * wx / QUAD_IY;
    let wzd = (QUAD_IX - QUAD_IY) * wx * wy / QUAD_IZ;
    Vector::new([qwd, qxd, qyd, qzd, wxd, wyd, wzd])
}

fn quadrotor_ke(y: &Vector<7, f64>) -> f64 {
    0.5 * (QUAD_IX * y[4] * y[4] + QUAD_IY * y[5] * y[5] + QUAD_IZ * y[6] * y[6])
}

fn quadrotor_qnorm(y: &Vector<7, f64>) -> f64 {
    (y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]).sqrt()
}

fn quadrotor_attitude() {
    let y0 = Vector::new([1.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.1]);
    let tf = 20.0;
    let steps = 20_000;
    let dt = tf / steps as f64;
    let ke_rk4 = rk4_drift(&quadrotor_rhs, &y0, dt, steps, quadrotor_ke);
    let ke_rk45 = rk45_drift(&quadrotor_rhs, &y0, tf, 1e-9, 1e-11, quadrotor_ke);
    let qn_rk4 = rk4_drift(&quadrotor_rhs, &y0, dt, steps, quadrotor_qnorm);
    let qn_rk45 = rk45_drift(&quadrotor_rhs, &y0, tf, 1e-9, 1e-11, quadrotor_qnorm);
    println!("\nQuadrotor attitude (torque-free tumble, N=7): drift over [0, 20]");
    println!("  RK4  (dt = 1e-3)   max|KE - KE0| = {ke_rk4:.2e}   max||q| - 1| = {qn_rk4:.2e}");
    println!("  RK45 (rtol 1e-9)   max|KE - KE0| = {ke_rk45:.2e}   max||q| - 1| = {qn_rk45:.2e}");
}

// ----- C. Solar-system N-body (Sun + 4 outer planets), N = 20 -----

const NB: usize = 5;
const NEWTON_G: f64 = 4.0 * core::f64::consts::PI * core::f64::consts::PI;
const NBODY_MASS: [f64; NB] = [1.0, 9.5e-4, 2.86e-4, 4.37e-5, 5.15e-5];
const NBODY_RADII: [f64; NB] = [0.0, 5.20, 9.58, 19.2, 30.1];

fn nbody_rhs(_t: f64, y: &Vector<20, f64>) -> Vector<20, f64> {
    let pos: [[f64; 2]; NB] = core::array::from_fn(|i| [y[4 * i], y[4 * i + 1]]);
    let vel: [[f64; 2]; NB] = core::array::from_fn(|i| [y[4 * i + 2], y[4 * i + 3]]);
    let mut acc = [[0.0f64; 2]; NB];
    for (i, (ai, pi)) in acc.iter_mut().zip(pos.iter()).enumerate() {
        for (j, (mj, pj)) in NBODY_MASS.iter().zip(pos.iter()).enumerate() {
            if i == j {
                continue;
            }
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let r2 = dx * dx + dy * dy;
            let inv = NEWTON_G * mj / (r2 * r2.sqrt());
            ai[0] += inv * dx;
            ai[1] += inv * dy;
        }
    }
    Vector::from_fn(|k| {
        let i = k / 4;
        match k % 4 {
            0 => vel[i][0],
            1 => vel[i][1],
            2 => acc[i][0],
            _ => acc[i][1],
        }
    })
}

fn nbody_energy(y: &Vector<20, f64>) -> f64 {
    let pos: [[f64; 2]; NB] = core::array::from_fn(|i| [y[4 * i], y[4 * i + 1]]);
    let vel: [[f64; 2]; NB] = core::array::from_fn(|i| [y[4 * i + 2], y[4 * i + 3]]);
    let mut ke = 0.0;
    for (m, v) in NBODY_MASS.iter().zip(vel.iter()) {
        ke += 0.5 * m * (v[0] * v[0] + v[1] * v[1]);
    }
    let mut pe = 0.0;
    for (i, (mi, pi)) in NBODY_MASS.iter().zip(pos.iter()).enumerate() {
        for (mj, pj) in NBODY_MASS.iter().zip(pos.iter()).skip(i + 1) {
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            pe -= NEWTON_G * mi * mj / (dx * dx + dy * dy).sqrt();
        }
    }
    ke + pe
}

fn nbody_y0() -> Vector<20, f64> {
    let bodies: [[f64; 4]; NB] = core::array::from_fn(|i| {
        if i == 0 {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            let r = NBODY_RADII[i];
            let v = (NEWTON_G * NBODY_MASS[0] / r).sqrt();
            [r, 0.0, 0.0, v]
        }
    });
    let mut planet_py = 0.0;
    for (m, b) in NBODY_MASS.iter().zip(bodies.iter()).skip(1) {
        planet_py += m * b[3];
    }
    let sun_vy = -planet_py / NBODY_MASS[0];
    Vector::from_fn(|k| {
        let i = k / 4;
        if i == 0 {
            if k % 4 == 3 { sun_vy } else { 0.0 }
        } else {
            bodies[i][k % 4]
        }
    })
}

fn solar_system_nbody() {
    let y0 = nbody_y0();
    let tf = 100.0;
    let steps = 2_000;
    let e0 = nbody_energy(&y0).abs();
    let rk4 = rk4_drift(&nbody_rhs, &y0, tf / steps as f64, steps, nbody_energy) / e0;
    let rk45 = rk45_drift(&nbody_rhs, &y0, tf, 1e-10, 1e-12, nbody_energy) / e0;
    println!(
        "\nSolar-system N-body (Sun + 4 outer planets, N=20): relative energy drift over 100 yr"
    );
    println!("  RK4  (dt = 0.05)   max|E - E0|/|E0| = {rk4:.2e}");
    println!("  RK45 (rtol 1e-10)  max|E - E0|/|E0| = {rk45:.2e}");
}
