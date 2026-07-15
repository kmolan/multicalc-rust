#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group};

use multicalc::linear_algebra::Vector;
use multicalc::ode::{Rk4, Rk45};

// ----- A. Two-link planar manipulator (acrobot), N = 4: [theta1, theta2, omega1, omega2] -----

const ACRO_G: f64 = 9.81;
const ACRO_M1: f64 = 1.0;
const ACRO_M2: f64 = 1.0;
const ACRO_L1: f64 = 1.0;
const ACRO_LC1: f64 = 0.5;
const ACRO_LC2: f64 = 0.5;
const ACRO_I1: f64 = 1.0 / 12.0;
const ACRO_I2: f64 = 1.0 / 12.0;

// Mass-matrix entries at joint angle theta2 (symmetric, d21 == d12).
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

fn acrobot_y0() -> Vector<4, f64> {
    Vector::new([0.0, 0.0, 0.0, 0.0])
}

// ----- B. Quadrotor attitude (torque-free tumble), N = 7: [qw,qx,qy,qz, wx,wy,wz] -----

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

fn quadrotor_y0() -> Vector<7, f64> {
    Vector::new([1.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.1])
}

// ----- C. Solar-system N-body (Sun + 4 outer planets), N = 20: [x,y,vx,vy] per body -----

const NB: usize = 5;
const NEWTON_G: f64 = 4.0 * core::f64::consts::PI * core::f64::consts::PI; // AU, year, solar mass
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

// Planets on circular orbits; the Sun's velocity cancels total momentum.
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

// ----- benches -----

const ACRO_TF: f64 = 10.0;
const ACRO_STEPS: usize = 10_000;
const QUAD_TF: f64 = 20.0;
const QUAD_STEPS: usize = 20_000;
const NBODY_TF: f64 = 100.0;
const NBODY_STEPS: usize = 2_000;

fn rk4(crit: &mut Criterion) {
    let acro = acrobot_y0();
    let dt = ACRO_TF / ACRO_STEPS as f64;
    crit.bench_function("ode/rk4/acrobot", |b| {
        b.iter(|| {
            Rk4::integrate(
                black_box(&acrobot_rhs),
                0.0,
                black_box(&acro),
                dt,
                ACRO_STEPS,
                |_, _| {},
            )
        })
    });

    let quad = quadrotor_y0();
    let dt = QUAD_TF / QUAD_STEPS as f64;
    crit.bench_function("ode/rk4/quadrotor_attitude", |b| {
        b.iter(|| {
            Rk4::integrate(
                black_box(&quadrotor_rhs),
                0.0,
                black_box(&quad),
                dt,
                QUAD_STEPS,
                |_, _| {},
            )
        })
    });

    let bodies = nbody_y0();
    let dt = NBODY_TF / NBODY_STEPS as f64;
    crit.bench_function("ode/rk4/solar_system_nbody", |b| {
        b.iter(|| {
            Rk4::integrate(
                black_box(&nbody_rhs),
                0.0,
                black_box(&bodies),
                dt,
                NBODY_STEPS,
                |_, _| {},
            )
        })
    });
}

fn rk45(crit: &mut Criterion) {
    let acro = acrobot_y0();
    crit.bench_function("ode/rk45/acrobot", |b| {
        b.iter(|| {
            Rk45::default()
                .with_rtol(1e-8)
                .with_atol(1e-10)
                .solve(black_box(&acrobot_rhs), 0.0, black_box(&acro), ACRO_TF)
                .unwrap()
        })
    });

    let quad = quadrotor_y0();
    crit.bench_function("ode/rk45/quadrotor_attitude", |b| {
        b.iter(|| {
            Rk45::default()
                .with_rtol(1e-9)
                .with_atol(1e-11)
                .solve(black_box(&quadrotor_rhs), 0.0, black_box(&quad), QUAD_TF)
                .unwrap()
        })
    });

    let bodies = nbody_y0();
    crit.bench_function("ode/rk45/solar_system_nbody", |b| {
        b.iter(|| {
            Rk45::default()
                .with_rtol(1e-10)
                .with_atol(1e-12)
                .solve(black_box(&nbody_rhs), 0.0, black_box(&bodies), NBODY_TF)
                .unwrap()
        })
    });
}

criterion_group! {
    name = ode_benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = rk4, rk45
}
