//! ODE integrators: fixed-step RK4 and adaptive RK45 (Dormand–Prince) on the harmonic
//! oscillator and three real dynamical systems — a two-link manipulator (acrobot), a
//! torque-free tumbling quadrotor (stepped three ways, including one that keeps the orientation
//! a true rotation), and an outer-solar-system N-body model. For the harmonic
//! case the exact solution is known; the other three have no closed form, so accuracy is
//! reported as the drift in a conserved quantity (energy, kinetic energy, quaternion norm).
//! These figures reproduce the accuracy table in `benches/ode.md`.
//!
//! Run with: `cargo run -p multicalc-demos --example ode`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::linear_algebra::{Vector, Vector2D, Vector3D};
use multicalc::ode::{ExponentialMap, Rk4, Rk45};
use multicalc::spatial::SO3;

fn main() {
    harmonic_oscillator();
    acrobot();
    quadrotor_attitude();
    solar_system_nbody();
}

// ----- harmonic oscillator y'' = -y, exact solution [cos t, -sin t] -----

fn harmonic_oscillator() {
    // y1' = y2 ; y2' = -y1
    let f = |_t: f64, y: &Vector2D| Vector::new([y[1], -y[0]]);
    let state_start = Vector::new([1.0, 0.0]);
    let exact = |time: f64| [time.cos(), -time.sin()];

    // RK4: 2000 fixed steps over [0, 2*pi].
    let steps = 2000;
    let timestep = core::f64::consts::TAU / steps as f64;
    let mut max_err = 0.0_f64;
    let state_final = Rk4::integrate(&f, 0.0, &state_start, timestep, steps, |time, y| {
        let exact_y = exact(time);
        max_err = max_err
            .max((y[0] - exact_y[0]).abs())
            .max((y[1] - exact_y[1]).abs());
    });
    println!("Harmonic oscillator y'' = -y");
    println!("  RK4  {steps} steps over [0, 2*pi]");
    println!(
        "    y(2*pi) = [{:.12}, {:.12}]  max|err| = {max_err:.2e}",
        state_final[0], state_final[1]
    );
    assert!(
        max_err < 1e-3,
        "RK4 should track the exact harmonic solution"
    );

    // RK45: adaptive solve to t = 2*pi, then dense-output sampling.
    let relative_tolerance = 1e-9;
    let absolute_tolerance = 1e-12;
    let solver = Rk45::default()
        .with_rtol(relative_tolerance)
        .with_atol(absolute_tolerance);

    let start_time = 0.0;
    let one_period = core::f64::consts::TAU;
    let state_final = solver
        .solve(&f, start_time, &state_start, one_period)
        .unwrap();
    let exact_y = exact(core::f64::consts::TAU);
    println!("  RK45 adaptive solve to t = 2*pi (rtol 1e-9)");
    println!(
        "    y(2*pi) = [{:.12}, {:.12}]  |err| = {:.2e}",
        state_final[0],
        state_final[1],
        (state_final[0] - exact_y[0])
            .abs()
            .max((state_final[1] - exact_y[1]).abs())
    );

    let times = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = [Vector2D::zeros(); 7];
    solver
        .solve_on_grid(&f, 0.0, &state_start, &times, &mut out)
        .unwrap();
    let grid_err = times
        .iter()
        .zip(out.iter())
        .map(|(&time, y)| {
            let exact_y = exact(time);
            (y[0] - exact_y[0]).abs().max((y[1] - exact_y[1]).abs())
        })
        .fold(0.0_f64, f64::max);
    println!("    dense-output grid max|err| = {grid_err:.2e}");
    assert!(grid_err < 1e-6, "RK45 dense output should be accurate");
}

// Largest drift of the invariant `inv` from its initial value over an RK4 integration.
#[must_use]
fn rk4_drift<const N: usize, F, I>(
    f: &F,
    state_start: &Vector<N, f64>,
    timestep: f64,
    steps: usize,
    inv: I,
) -> f64
where
    F: Fn(f64, &Vector<N, f64>) -> Vector<N, f64>,
    I: Fn(&Vector<N, f64>) -> f64,
{
    let inv_start = inv(state_start);
    let mut max = 0.0_f64;
    let _ = Rk4::integrate(f, 0.0, state_start, timestep, steps, |_, y| {
        max = max.max((inv(y) - inv_start).abs());
    });
    max
}

// Largest drift of the invariant `inv` over the accepted steps of an RK45 solve.
#[must_use]
fn rk45_drift<const N: usize, F, I>(
    f: &F,
    state_start: &Vector<N, f64>,
    time_final: f64,
    rtol: f64,
    atol: f64,
    inv: I,
) -> f64
where
    F: Fn(f64, &Vector<N, f64>) -> Vector<N, f64>,
    I: Fn(&Vector<N, f64>) -> f64,
{
    let inv_start = inv(state_start);
    let mut max = 0.0_f64;
    let _ = Rk45::default()
        .with_rtol(rtol)
        .with_atol(atol)
        .for_each_step(f, 0.0, state_start, time_final, |step| {
            max = max.max((inv(&step.state_end) - inv_start).abs());
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

#[must_use]
fn acrobot_mass(cos_th2: f64) -> (f64, f64, f64) {
    let d11 = ACRO_M1 * ACRO_LC1 * ACRO_LC1
        + ACRO_M2 * (ACRO_L1 * ACRO_L1 + ACRO_LC2 * ACRO_LC2 + 2.0 * ACRO_L1 * ACRO_LC2 * cos_th2)
        + ACRO_I1
        + ACRO_I2;
    let d12 = ACRO_M2 * (ACRO_LC2 * ACRO_LC2 + ACRO_L1 * ACRO_LC2 * cos_th2) + ACRO_I2;
    let d22 = ACRO_M2 * ACRO_LC2 * ACRO_LC2 + ACRO_I2;
    (d11, d12, d22)
}

fn acrobot_rhs(_t: f64, y: &Vector<4, f64>) -> Vector<4, f64> {
    let [th1, th2, omega1, omega2] = [y[0], y[1], y[2], y[3]];
    let (d11, d12, d22) = acrobot_mass(th2.cos());
    let sin_th2 = th2.sin();
    let coriolis1 =
        -ACRO_M2 * ACRO_L1 * ACRO_LC2 * sin_th2 * (2.0 * omega1 * omega2 + omega2 * omega2);
    let coriolis2 = ACRO_M2 * ACRO_L1 * ACRO_LC2 * sin_th2 * omega1 * omega1;
    let phi1 = (ACRO_M1 * ACRO_LC1 + ACRO_M2 * ACRO_L1) * ACRO_G * th1.cos()
        + ACRO_M2 * ACRO_LC2 * ACRO_G * (th1 + th2).cos();
    let phi2 = ACRO_M2 * ACRO_LC2 * ACRO_G * (th1 + th2).cos();
    let det = d11 * d22 - d12 * d12;
    let w1d = (-d22 * (coriolis1 + phi1) + d12 * (coriolis2 + phi2)) / det;
    let w2d = (d12 * (coriolis1 + phi1) - d11 * (coriolis2 + phi2)) / det;
    Vector::new([omega1, omega2, w1d, w2d])
}

#[must_use]
fn acrobot_energy(y: &Vector<4, f64>) -> f64 {
    let [th1, th2, omega1, omega2] = [y[0], y[1], y[2], y[3]];
    let (d11, d12, d22) = acrobot_mass(th2.cos());
    let kinetic =
        0.5 * (d11 * omega1 * omega1 + 2.0 * d12 * omega1 * omega2 + d22 * omega2 * omega2);
    let potential = ACRO_G
        * (ACRO_M1 * ACRO_LC1 * th1.sin()
            + ACRO_M2 * (ACRO_L1 * th1.sin() + ACRO_LC2 * (th1 + th2).sin()));
    kinetic + potential
}

fn acrobot() {
    let state_start = Vector::new([0.0, 0.0, 0.0, 0.0]);
    let time_final = 10.0;
    let steps = 10_000;
    let rk4 = rk4_drift(
        &acrobot_rhs,
        &state_start,
        time_final / steps as f64,
        steps,
        acrobot_energy,
    );
    let rk45 = rk45_drift(
        &acrobot_rhs,
        &state_start,
        time_final,
        1e-8,
        1e-10,
        acrobot_energy,
    );
    println!("\nAcrobot (two-link manipulator, N=4): energy drift over [0, 10]");
    println!("  RK4  (dt = 1e-3)   max|E - E0| = {rk4:.2e}");
    println!("  RK45 (rtol 1e-8)   max|E - E0| = {rk45:.2e}");
}

// ----- B. Quadrotor attitude (torque-free tumble), N = 7 -----

const QUAD_IX: f64 = 0.01;
const QUAD_IY: f64 = 0.02;
const QUAD_IZ: f64 = 0.03;

fn quadrotor_rhs(_t: f64, y: &Vector<7, f64>) -> Vector<7, f64> {
    let [quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z] =
        [y[0], y[1], y[2], y[3], y[4], y[5], y[6]];
    let qwd = -0.5 * (quat_x * omega_x + quat_y * omega_y + quat_z * omega_z);
    let qxd = 0.5 * (quat_w * omega_x + quat_y * omega_z - quat_z * omega_y);
    let qyd = 0.5 * (quat_w * omega_y - quat_x * omega_z + quat_z * omega_x);
    let qzd = 0.5 * (quat_w * omega_z + quat_x * omega_y - quat_y * omega_x);
    let wxd = (QUAD_IY - QUAD_IZ) * omega_y * omega_z / QUAD_IX;
    let wyd = (QUAD_IZ - QUAD_IX) * omega_z * omega_x / QUAD_IY;
    let wzd = (QUAD_IX - QUAD_IY) * omega_x * omega_y / QUAD_IZ;
    Vector::new([qwd, qxd, qyd, qzd, wxd, wyd, wzd])
}

#[must_use]
fn quadrotor_ke(y: &Vector<7, f64>) -> f64 {
    0.5 * (QUAD_IX * y[4] * y[4] + QUAD_IY * y[5] * y[5] + QUAD_IZ * y[6] * y[6])
}

#[must_use]
fn quadrotor_qnorm(y: &Vector<7, f64>) -> f64 {
    (y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]).sqrt()
}

// How fast the tumble's spin is changing, with nothing pushing on the body.
fn quadrotor_spin_change(rate: Vector3D<f64>) -> Vector3D<f64> {
    Vector::new([
        (QUAD_IY - QUAD_IZ) * rate[1] * rate[2] / QUAD_IX,
        (QUAD_IZ - QUAD_IX) * rate[2] * rate[0] / QUAD_IY,
        (QUAD_IX - QUAD_IY) * rate[0] * rate[1] / QUAD_IZ,
    ])
}

// The same tumble, with the spin stepped by RK4 and the direction the body faces carried
// forward as a turn. Returns the largest drift in spin energy and in orientation length.
#[must_use]
fn quadrotor_exp_map_drift(rate0: Vector3D<f64>, timestep: f64, steps: usize) -> (f64, f64) {
    let spin_rate = |_time: f64, rate: &Vector3D<f64>| quadrotor_spin_change(*rate);
    let kinetic_energy = |rate: &Vector3D<f64>| {
        0.5 * (QUAD_IX * rate[0] * rate[0]
            + QUAD_IY * rate[1] * rate[1]
            + QUAD_IZ * rate[2] * rate[2])
    };

    let energy0 = kinetic_energy(&rate0);
    let mut orientation = SO3::<f64>::identity();
    let mut rate = rate0;
    let mut worst_energy = 0.0_f64;
    let mut worst_length = 0.0_f64;
    for step in 0..steps {
        orientation = ExponentialMap::attitude_step_with_angular_acceleration(
            orientation,
            rate,
            quadrotor_spin_change(rate),
            timestep,
        );
        rate = Rk4::step(&spin_rate, step as f64 * timestep, &rate, timestep);
        worst_energy = worst_energy.max((kinetic_energy(&rate) - energy0).abs());
        worst_length = worst_length.max((orientation.quaternion().norm() - 1.0).abs());
    }
    (worst_energy, worst_length)
}

fn quadrotor_attitude() {
    let state_start = Vector::new([1.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.1]);
    let time_final = 20.0;
    let steps = 20_000;
    let timestep = time_final / steps as f64;
    let ke_rk4 = rk4_drift(&quadrotor_rhs, &state_start, timestep, steps, quadrotor_ke);
    let ke_rk45 = rk45_drift(
        &quadrotor_rhs,
        &state_start,
        time_final,
        1e-9,
        1e-11,
        quadrotor_ke,
    );
    let qn_rk4 = rk4_drift(
        &quadrotor_rhs,
        &state_start,
        timestep,
        steps,
        quadrotor_qnorm,
    );
    let qn_rk45 = rk45_drift(
        &quadrotor_rhs,
        &state_start,
        time_final,
        1e-9,
        1e-11,
        quadrotor_qnorm,
    );
    println!("\nQuadrotor attitude (torque-free tumble, N=7): drift over [0, 20]");
    println!("  RK4  (dt = 1e-3)   max|KE - KE0| = {ke_rk4:.2e}   max||q| - 1| = {qn_rk4:.2e}");
    println!("  RK45 (rtol 1e-9)   max|KE - KE0| = {ke_rk45:.2e}   max||q| - 1| = {qn_rk45:.2e}");

    let rate0 = Vector::new([0.1, 5.0, 0.1]);
    let (ke_exp, qn_exp) = quadrotor_exp_map_drift(rate0, timestep, steps);
    println!("  exp-map (dt = 1e-3)  max|KE - KE0| = {ke_exp:.2e}   max||q| - 1| = {qn_exp:.2e}");
    println!(
        "    the spin is stepped by RK4 either way; the difference is that the direction the\n     body faces is carried forward as a turn, so it never leaves unit length to be scaled back"
    );
    assert!(
        qn_exp < 1e-13,
        "carrying the orientation as a turn should hold unit length to rounding"
    );
}

// ----- C. Solar-system N-body (Sun + 4 outer planets), N = 20 -----

const N_BODIES: usize = 5;
const NEWTON_G: f64 = 4.0 * core::f64::consts::PI * core::f64::consts::PI;
const NBODY_MASS: [f64; N_BODIES] = [1.0, 9.5e-4, 2.86e-4, 4.37e-5, 5.15e-5];
const NBODY_RADII: [f64; N_BODIES] = [0.0, 5.20, 9.58, 19.2, 30.1];

fn nbody_rhs(_t: f64, y: &Vector<20, f64>) -> Vector<20, f64> {
    let pos: [[f64; 2]; N_BODIES] = core::array::from_fn(|i| [y[4 * i], y[4 * i + 1]]);
    let vel: [[f64; 2]; N_BODIES] = core::array::from_fn(|i| [y[4 * i + 2], y[4 * i + 3]]);
    let mut acc = [[0.0f64; 2]; N_BODIES];
    for (i, (acc_i, pos_i)) in acc.iter_mut().zip(pos.iter()).enumerate() {
        for (j, (mass_j, pos_j)) in NBODY_MASS.iter().zip(pos.iter()).enumerate() {
            if i == j {
                continue;
            }
            let dx = pos_j[0] - pos_i[0];
            let delta_y = pos_j[1] - pos_i[1];
            let r_sq = dx * dx + delta_y * delta_y;
            let inv = NEWTON_G * mass_j / (r_sq * r_sq.sqrt());
            acc_i[0] += inv * dx;
            acc_i[1] += inv * delta_y;
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

#[must_use]
fn nbody_energy(y: &Vector<20, f64>) -> f64 {
    let pos: [[f64; 2]; N_BODIES] = core::array::from_fn(|i| [y[4 * i], y[4 * i + 1]]);
    let vel: [[f64; 2]; N_BODIES] = core::array::from_fn(|i| [y[4 * i + 2], y[4 * i + 3]]);
    let mut kinetic = 0.0;
    for (mass, vel) in NBODY_MASS.iter().zip(vel.iter()) {
        kinetic += 0.5 * mass * (vel[0] * vel[0] + vel[1] * vel[1]);
    }
    let mut potential = 0.0;
    for (i, (mass_i, pos_i)) in NBODY_MASS.iter().zip(pos.iter()).enumerate() {
        for (mass_j, pos_j) in NBODY_MASS.iter().zip(pos.iter()).skip(i + 1) {
            let dx = pos_j[0] - pos_i[0];
            let delta_y = pos_j[1] - pos_i[1];
            potential -= NEWTON_G * mass_i * mass_j / (dx * dx + delta_y * delta_y).sqrt();
        }
    }
    kinetic + potential
}

fn nbody_y0() -> Vector<20, f64> {
    let bodies: [[f64; 4]; N_BODIES] = core::array::from_fn(|i| {
        if i == 0 {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            let radius = NBODY_RADII[i];
            let vel = (NEWTON_G * NBODY_MASS[0] / radius).sqrt();
            [radius, 0.0, 0.0, vel]
        }
    });
    let mut planet_py = 0.0;
    for (mass, b) in NBODY_MASS.iter().zip(bodies.iter()).skip(1) {
        planet_py += mass * b[3];
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
    let state_start = nbody_y0();
    let time_final = 100.0;
    let steps = 2_000;
    let energy0 = nbody_energy(&state_start).abs();
    let rk4 = rk4_drift(
        &nbody_rhs,
        &state_start,
        time_final / steps as f64,
        steps,
        nbody_energy,
    ) / energy0;
    let rk45 = rk45_drift(
        &nbody_rhs,
        &state_start,
        time_final,
        1e-10,
        1e-12,
        nbody_energy,
    ) / energy0;
    println!(
        "\nSolar-system N-body (Sun + 4 outer planets, N=20): relative energy drift over 100 yr"
    );
    println!("  RK4  (dt = 0.05)   max|E - E0|/|E0| = {rk4:.2e}");
    println!("  RK45 (rtol 1e-10)  max|E - E0|/|E0| = {rk45:.2e}");
}
