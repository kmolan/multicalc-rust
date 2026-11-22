//! Gradient marbles (autodiff showcase).
//!
//! Two thousand marbles roll across a 3D Himmelblau landscape, each steered every millisecond by
//! an exact autodiff gradient — two thousand exact gradients per tick. The analytic gradient is
//! known in closed form, so the demo shows the autodiff-vs-analytic error pinned at machine zero
//! while the marbles cascade into the four basins and settle.
//!
//! Timing model: the physics advances on logical time (a fixed dt = 1 ms per tick), so cycles are
//! deterministic — the pacer only decides when a tick is displayed, and an OS stall never perturbs
//! the marbles. `plots/grad_batch_us` is multicalc's gradient-batch math cost; host-OS scheduling
//! lateness is measured too but shown only as a hud percentile (not a plot), since it is the OS,
//! not the library. The headline is that math cost and its headroom under the 1 ms budget.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example gradient_marbles

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::numerical_derivative::Jacobian;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc_demos::loop_util::{LatencyRing, Pacer, commas};
use multicalc_demos::{RerunSink, Rgba, VizError, VizSink};
use std::f64::consts::TAU;
use std::time::Instant;

const N_MARBLES: usize = 2000;
const GAIN: f64 = 0.15;
const DRAG: f64 = 1.2;
const V_MAX: f64 = 4.0;
const TIMESTEP: f64 = 1e-3;
const DOMAIN: f64 = 5.0; // domain is [-5, 5]^2
const Z_SCALE: f64 = 80.0; // display height z = f / Z_SCALE
const MARBLE_LIFT: f64 = 0.05; // marbles ride just above the surface
const TERRAIN_GRID: usize = 96;
const RESPAWN_TICKS: i64 = 25_000; // watershed cascade every 25 s
const RESPAWN_RADIUS: f64 = 4.5;
const PROBE_COUNT: usize = 100;
const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz) — mandatory (2000 pts/tick would flood)
const HUD_EVERY: i64 = 1000;
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats

// Sequential ramps  linear-sRGB lerp between endpoints.
const BLUE_LO: [u8; 3] = [0x86, 0xb6, 0xef]; // terrain height low
const BLUE_HI: [u8; 3] = [0x10, 0x42, 0x81]; // terrain height high
const AMBER_LO: [u8; 3] = [0xff, 0xd9, 0xa0]; // |∇f| low
const AMBER_HI: [u8; 3] = [0x7a, 0x51, 0x00]; // |∇f| high

const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // ad_vs_analytic series

/// The Himmelblau function as a `VectorFn<2, 1>`; its gradient is the Jacobian's single row.
struct Himmelblau;

impl VectorFn<2, 1> for Himmelblau {
    fn eval<S: Numeric>(&self, params: &[S; 2]) -> [S; 1] {
        let (x, y) = (params[0], params[1]);
        let a = x * x + y - S::from_f64(11.0);
        let b = x + y * y - S::from_f64(7.0);
        [a * a + b * b]
    }
}

#[must_use]
fn himmelblau_f(x: f64, y: f64) -> f64 {
    let a = x * x + y - 11.0;
    let b = x + y * y - 7.0;
    a * a + b * b
}

#[must_use]
fn himmelblau_grad(x: f64, y: f64) -> [f64; 2] {
    let a = x * x + y - 11.0;
    let b = x + y * y - 7.0;
    [4.0 * x * a + 2.0 * b, 2.0 * a + 4.0 * y * b]
}

/// A small deterministic PRNG for respawn jitter and the accuracy probe points.
struct Lcg(u64);

impl Lcg {
    #[must_use]
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Marble {
    pos: [f64; 2],
    vel: [f64; 2],
}

/// Respawns all marbles on a ring of radius `RESPAWN_RADIUS`, uniform angles jittered ±0.1, v = 0.
fn respawn(marbles: &mut [Marble], rng: &mut Lcg) {
    let n = marbles.len();
    for (i, marble) in marbles.iter_mut().enumerate() {
        let angle = TAU * i as f64 / n as f64 + (rng.unit() - 0.5) * 0.2;
        marble.pos = [RESPAWN_RADIUS * angle.cos(), RESPAWN_RADIUS * angle.sin()];
        marble.vel = [0.0, 0.0];
    }
}

#[must_use]
fn srgb_to_linear(channel: f64) -> f64 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

#[must_use]
fn linear_to_srgb(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// A color on the `lo`→`hi` ramp at `t ∈ [0, 1]`, lerped in linear-sRGB space.
#[must_use]
fn ramp(low: [u8; 3], high: [u8; 3], blend: f64) -> Rgba {
    let blend = blend.clamp(0.0, 1.0);
    let channel = |a: u8, b: u8| {
        let lin_a = srgb_to_linear(a as f64 / 255.0);
        let lin_b = srgb_to_linear(b as f64 / 255.0);
        (linear_to_srgb(lin_a + (lin_b - lin_a) * blend) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    [
        channel(low[0], high[0]),
        channel(low[1], high[1]),
        channel(low[2], high[2]),
        0xff,
    ]
}

/// The `q`-quantile of `vals` (floored at a small positive value to keep it a safe divisor).
#[must_use]
fn quantile(vals: &[f64], frac: f64) -> f64 {
    if vals.is_empty() {
        return 1.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * frac).round() as usize;
    sorted[idx].max(1e-9)
}

/// Max autodiff-vs-analytic gradient error over the fixed probe set.
#[must_use]
fn probe_error(jac: &Jacobian, probes: &[[f64; 2]]) -> f64 {
    let mut max_err = 0.0f64;
    for params in probes {
        let gradient = jac.evaluate(&Himmelblau, params).expect("probe gradient");
        let auto_grad = [gradient[(0, 0)], gradient[(0, 1)]];
        let analytic = himmelblau_grad(params[0], params[1]);
        max_err = max_err
            .max((auto_grad[0] - analytic[0]).abs())
            .max((auto_grad[1] - analytic[1]).abs());
    }
    max_err
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example gradient_marbles"
        );
    }

    let jac = Jacobian::<AutoDiffMulti>::default();
    let mut rng = Lcg(0x9e3779b97f4a7c15);

    // Fixed accuracy-probe points in the domain.
    let probes: Vec<[f64; 2]> = (0..PROBE_COUNT)
        .map(|_| {
            [
                (rng.unit() * 2.0 - 1.0) * DOMAIN,
                (rng.unit() * 2.0 - 1.0) * DOMAIN,
            ]
        })
        .collect();

    let mut marbles: Vec<Marble> = (0..N_MARBLES)
        .map(|_| Marble {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        })
        .collect();
    respawn(&mut marbles, &mut rng);

    let mut sink = RerunSink::live("multicalc-demos/gradient-marbles")?;
    sink.set_sequence("tick", 0);
    sink.series_style(
        "plots/ad_vs_analytic",
        ERROR,
        "autodiff − analytic error",
        2.0,
    )?;
    // Gradient-batch time is summarized in the hud, not plotted.

    // Static terrain: a grid of styled points colored by height.
    {
        let grid = TERRAIN_GRID;
        let mut pts: Vec<[f64; 3]> = Vec::with_capacity(grid * grid);
        let mut heights: Vec<f64> = Vec::with_capacity(grid * grid);
        for row_y in 0..grid {
            for col_x in 0..grid {
                let x = -DOMAIN + 2.0 * DOMAIN * col_x as f64 / (grid - 1) as f64;
                let y = -DOMAIN + 2.0 * DOMAIN * row_y as f64 / (grid - 1) as f64;
                let z = himmelblau_f(x, y) / Z_SCALE;
                pts.push([x, y, z]);
                heights.push(z);
            }
        }
        let zmax = heights.iter().copied().fold(0.0f64, f64::max).max(1e-9);
        let cols: Vec<Rgba> = heights
            .iter()
            .map(|&z| ramp(BLUE_LO, BLUE_HI, z / zmax))
            .collect();
        sink.points3d_styled("world/terrain", &pts, &cols, &[0.03])?;
    }

    let mut pacer = Pacer::new();
    let mut batch_ring = LatencyRing::new(1024);
    let mut grads: Vec<Option<[f64; 2]>> = vec![None; N_MARBLES];
    let mut total_gradients: u64 = 0; // cumulative, for the running hud counter

    let mut n: i64 = 0;
    loop {
        let _ = pacer.wait(); // pace to the next 1 ms boundary
        n += 1;
        sink.set_sequence("tick", n);

        if n % RESPAWN_TICKS == 0 {
            respawn(&mut marbles, &mut rng);
        }

        // The measured batch: 2000 exact autodiff gradients.
        let tick_start = Instant::now();
        for (i, marble) in marbles.iter().enumerate() {
            grads[i] = jac
                .evaluate(&Himmelblau, &marble.pos)
                .ok()
                .map(|j| [j[(0, 0)], j[(0, 1)]]);
        }
        let grad_batch_us = tick_start.elapsed().as_nanos() as f64 / 1000.0;
        total_gradients += N_MARBLES as u64;

        // Dynamics: semi-implicit Euler with drag, speed clamp, and reflecting walls.
        for (i, marble) in marbles.iter_mut().enumerate() {
            let Some(grad) = grads[i] else { continue }; // frozen marble
            marble.vel[0] = marble.vel[0] * (1.0 - DRAG * TIMESTEP) - GAIN * grad[0] * TIMESTEP;
            marble.vel[1] = marble.vel[1] * (1.0 - DRAG * TIMESTEP) - GAIN * grad[1] * TIMESTEP;
            let speed = (marble.vel[0] * marble.vel[0] + marble.vel[1] * marble.vel[1]).sqrt();
            if speed > V_MAX {
                let scale = V_MAX / speed;
                marble.vel[0] *= scale;
                marble.vel[1] *= scale;
            }
            for dim in 0..2 {
                marble.pos[dim] += marble.vel[dim] * TIMESTEP;
                if marble.pos[dim] < -DOMAIN {
                    marble.pos[dim] = -DOMAIN;
                    marble.vel[dim] = -marble.vel[dim];
                } else if marble.pos[dim] > DOMAIN {
                    marble.pos[dim] = DOMAIN;
                    marble.vel[dim] = -marble.vel[dim];
                }
            }
        }

        if n > WARMUP_TICKS {
            batch_ring.push(grad_batch_us);
        }

        // Spatial geometry at ~60 Hz: all marbles, colored by gradient magnitude.
        if n % GEOM_EVERY == 0 {
            let pts: Vec<[f64; 3]> = marbles
                .iter()
                .map(|marble| {
                    let z = himmelblau_f(marble.pos[0], marble.pos[1]) / Z_SCALE + MARBLE_LIFT;
                    [marble.pos[0], marble.pos[1], z]
                })
                .collect();
            let mags: Vec<f64> = grads
                .iter()
                .map(|grad| grad.map_or(0.0, |grad_xy| grad_xy[0].hypot(grad_xy[1])))
                .collect();
            let p95 = quantile(&mags, 0.95);
            let cols: Vec<Rgba> = mags
                .iter()
                .map(|&mag| ramp(AMBER_LO, AMBER_HI, mag / p95))
                .collect();
            sink.points3d_styled("world/marbles", &pts, &cols, &[0.05])?;
        }

        // Accuracy probe and hud at 1 Hz.
        if n % HUD_EVERY == 0 {
            let probe_err = probe_error(&jac, &probes);
            sink.scalar("plots/ad_vs_analytic", probe_err)?;
            if let Some(b) = batch_ring.summary() {
                let grads_per_ms = N_MARBLES as f64 / (b.median / 1000.0);
                let meta = format!(
                    "## gradient_marbles — multicalc live demo\n\
                     ### {} exact autodiff gradients: {:.0} µs/tick — {:.2} % of the 1 ms tick ({}/ms, one core)\n\
                     ### accuracy: max |∇f_AD − ∇f_analytic| = {:.1e}\n\
                     ### gradients computed: {} and counting",
                    commas(N_MARBLES as u64),
                    b.median,
                    b.median / 10.0,
                    commas(grads_per_ms as u64),
                    probe_err,
                    commas(total_gradients),
                );
                sink.text("hud/stats", &meta)?;
            }
        }
    }
}
