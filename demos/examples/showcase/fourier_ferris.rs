//! Fourier epicycles drawing Ferris (integration showcase).
//!
//! Gauss-Legendre quadrature computes the Fourier coefficients of Ferris the crab's outline at
//! startup — a chain of rotating circles then draws the crab, sharpening as harmonics stream in.
//! Every coefficient matches the exact closed-form antiderivative of the piecewise-linear outline
//! to machine precision, which is the integration module's accuracy showcase.
//!
//! Timing model: the drawing advances on logical time (a fixed 1 ms per tick), so it is
//! deterministic — the pacer only decides when a tick is displayed. `plots/tick_us` is multicalc's
//! chain-evaluation math cost; host-OS scheduling lateness is measured too but shown only as a hud
//! percentile (not a plot), since it is the OS, not the library. The headline is that math cost and
//! its headroom under the 1 ms budget.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example fourier_ferris

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;
use multicalc_demos::loop_util::{LatencyRing, Pacer, commas};
use multicalc_demos::{RerunSink, Rgba, VizError, VizSink};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::time::Instant;

include!("../resources/ferris_outline.rs"); // pub const FERRIS: [[f64; 2]; 256]

const N_PTS: usize = 256;
const K_MAX: i32 = 48; // harmonics k = -48..=48
const GL_ORDER: usize = 12; // Gauss-Legendre nodes per segment
const REVOLUTION_TICKS: u64 = 12_000; // one drawing revolution = 12 s
const REVEAL_STEP_TICKS: u64 = 1_500; // +2 harmonics every 1.5 s
const TRACE_MAX: usize = 720; // ~one revolution of tip positions at 60 Hz
const CIRCLE_SEGS: usize = 32;
const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text / coeff-error cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats

// Palette (§2).
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // trace, tip
const SILHOUETTE: Rgba = [0xc9, 0x85, 0x00, 80]; // target outline
const CIRCLES: Rgba = [0x90, 0x85, 0xe9, 120]; // epicycle circles
const SPOKES: Rgba = [0x89, 0x87, 0x81, 0xff]; // chain spokes
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // coeff_error series

/// A minimal complex number for the coefficient math and chain evaluation.
#[derive(Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Cx {
        Cx { re, im }
    }
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Cx) -> Cx {
        Cx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn scale(self, s: f64) -> Cx {
        Cx::new(self.re * s, self.im * s)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    /// e^{iθ}.
    fn expi(theta: f64) -> Cx {
        Cx::new(theta.cos(), theta.sin())
    }
}

/// Chord-length parameterization: `t[j]` is the cumulative chord length up to vertex `j`, over the
/// closed loop, normalized to [0, 1]. Returns `N_PTS + 1` breakpoints with `t[N_PTS] = 1`.
fn chord_params() -> Vec<f64> {
    let mut seg = [0.0; N_PTS];
    let mut total = 0.0;
    for (j, s) in seg.iter_mut().enumerate() {
        let jn = (j + 1) % N_PTS;
        let dx = FERRIS[jn][0] - FERRIS[j][0];
        let dy = FERRIS[jn][1] - FERRIS[j][1];
        *s = dx.hypot(dy);
        total += *s;
    }
    let mut t = vec![0.0; N_PTS + 1];
    let mut acc = 0.0;
    for j in 0..N_PTS {
        acc += seg[j];
        t[j + 1] = acc / total;
    }
    t
}

/// The exact integral of `z(t) e^{-iωt}` over one linear segment `z(t) = a + b·(t − ta)`,
/// `t ∈ [ta, tb]`, by antiderivative — the closed-form reference for the quadrature.
fn closed_segment(a: Cx, b: Cx, ta: f64, tb: f64, omega: f64) -> Cx {
    let dj = tb - ta;
    if omega == 0.0 {
        // ∫ z dt = a·Δ + b·Δ²/2
        return a.scale(dj).add(b.scale(dj * dj * 0.5));
    }
    // antiderivative F(t) = (z(t)·i/ω + b/ω²) e^{-iωt}; d/dt F = z(t) e^{-iωt}
    let term = |t: f64| {
        let zt = a.add(b.scale(t - ta));
        zt.mul(Cx::new(0.0, 1.0 / omega))
            .add(b.scale(1.0 / (omega * omega)))
            .mul(Cx::expi(-omega * t))
    };
    term(tb).sub(term(ta))
}

/// Computes every Fourier coefficient by Gauss-Legendre, the exact coefficient in closed form, the
/// max GL-vs-closed error, and the total quadrature node-evaluation count.
fn compute_coefficients(t: &[f64]) -> (Vec<(i32, Cx)>, f64, u64) {
    let gl = GaussianSingle::from_parameters(GL_ORDER, GaussianQuadratureMethod::GaussLegendre);
    let mut coeffs = Vec::new();
    let mut max_err = 0.0f64;
    let mut node_evals = 0u64;

    for k in -K_MAX..=K_MAX {
        let omega = TAU * k as f64;
        let mut c_gl = Cx::ZERO;
        let mut c_exact = Cx::ZERO;
        for j in 0..N_PTS {
            let jn = (j + 1) % N_PTS;
            let (xj, yj) = (FERRIS[j][0], FERRIS[j][1]);
            let (xj1, yj1) = (FERRIS[jn][0], FERRIS[jn][1]);
            let (ta, tb) = (t[j], t[j + 1]);
            let dj = tb - ta;

            // z(t) e^{-iωt} = (x cos ωt + y sin ωt) + i(y cos ωt − x sin ωt).
            let seg_re = |tt: f64| {
                let s = (tt - ta) / dj;
                let (x, y) = (xj + s * (xj1 - xj), yj + s * (yj1 - yj));
                let th = omega * tt;
                x * th.cos() + y * th.sin()
            };
            let seg_im = |tt: f64| {
                let s = (tt - ta) / dj;
                let (x, y) = (xj + s * (xj1 - xj), yj + s * (yj1 - yj));
                let th = omega * tt;
                y * th.cos() - x * th.sin()
            };
            let re = gl.get_single(&seg_re, &[ta, tb]).expect("gl re");
            let im = gl.get_single(&seg_im, &[ta, tb]).expect("gl im");
            c_gl = c_gl.add(Cx::new(re, im));
            node_evals += 2 * GL_ORDER as u64;

            let a = Cx::new(xj, yj);
            let b = Cx::new(xj1 - xj, yj1 - yj).scale(1.0 / dj);
            c_exact = c_exact.add(closed_segment(a, b, ta, tb, omega));
        }
        max_err = max_err.max(c_gl.sub(c_exact).abs());
        if k != 0 {
            coeffs.push((k, c_gl)); // k = 0 (≈ 0 after centroid normalization) is not drawn
        }
    }
    // Classic epicycle order: biggest circle first.
    coeffs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    (coeffs, max_err, node_evals)
}

/// Active harmonic count for the reveal phase, and whether this tick begins a fresh reveal.
fn reveal_state(reveal_clock: u64, max_h: usize) -> usize {
    let full = ((max_h.max(2) - 2) / 2) as u64 * REVEAL_STEP_TICKS;
    let cycle = full + 2 * REVOLUTION_TICKS; // reveal, then hold two revolutions
    let phase = reveal_clock % cycle;
    if phase >= full {
        max_h
    } else {
        (2 + 2 * (phase / REVEAL_STEP_TICKS) as usize).min(max_h)
    }
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example fourier_ferris"
        );
    }

    let t = chord_params();
    let start = Instant::now();
    let (harmonics, coeff_error, node_evals) = compute_coefficients(&t);
    let startup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let max_h = harmonics.len();
    println!(
        "coefficients: {max_h} harmonics, {} quadrature node evals in {startup_ms:.0} ms, \
         max GL-vs-closed error {coeff_error:.2e}",
        commas(node_evals)
    );

    let mut rr = RerunSink::live("multicalc-demos/fourier-ferris")?;
    rr.set_sequence("tick", 0);
    rr.series_style(
        "plots/coeff_error",
        ERROR,
        "coeff error (GL vs closed)",
        2.0,
    )?;
    // Chain-eval time, active-harmonic count, and host-OS jitter are shown in the hud, not plotted.

    // Static silhouette: the closed target outline.
    let mut silhouette: Vec<[f64; 2]> = FERRIS.to_vec();
    silhouette.push(FERRIS[0]);
    rr.line_strips2d("world/silhouette", &[silhouette], &[SILHOUETTE], &[0.006])?;

    let mut pacer = Pacer::new();
    let mut tick_ring = LatencyRing::new(1024);
    let mut trace: VecDeque<[f64; 2]> = VecDeque::with_capacity(TRACE_MAX);

    let mut reveal_clock: u64 = 0;
    let mut prev_active = 0usize;
    let mut n: i64 = 0;
    loop {
        pacer.wait(); // pace to the next 1 ms boundary
        n += 1;
        rr.set_sequence("tick", n);

        let u = (n as u64 % REVOLUTION_TICKS) as f64 / REVOLUTION_TICKS as f64;
        let active = reveal_state(reveal_clock, max_h);
        if active < prev_active {
            trace.clear(); // reveal reset: drop the old low-harmonic scribble
        }
        prev_active = active;
        reveal_clock += 1;

        // Per-tick chain evaluation (the measured math): the pen tip at parameter u.
        let t0 = Instant::now();
        let mut tip = Cx::ZERO;
        for &(k, c) in &harmonics[..active] {
            tip = tip.add(c.mul(Cx::expi(TAU * k as f64 * u)));
        }
        let tick_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        if n > WARMUP_TICKS {
            tick_ring.push(tick_us);
        }

        // Spatial geometry at ~60 Hz: full chain (circles, spokes), trace, tip.
        if n % GEOM_EVERY == 0 {
            let mut center = Cx::ZERO;
            let mut nodes: Vec<[f64; 2]> = Vec::with_capacity(active + 1);
            let mut circles: Vec<Vec<[f64; 2]>> = Vec::with_capacity(active);
            nodes.push([0.0, 0.0]);
            for &(k, c) in &harmonics[..active] {
                let radius = c.abs();
                let circle: Vec<[f64; 2]> = (0..=CIRCLE_SEGS)
                    .map(|s| {
                        let a = TAU * s as f64 / CIRCLE_SEGS as f64;
                        [center.re + radius * a.cos(), center.im + radius * a.sin()]
                    })
                    .collect();
                circles.push(circle);
                center = center.add(c.mul(Cx::expi(TAU * k as f64 * u)));
                nodes.push([center.re, center.im]);
            }
            let pen = [center.re, center.im];

            if trace.len() == TRACE_MAX {
                trace.pop_front();
            }
            trace.push_back(pen);
            let trace_pts: Vec<[f64; 2]> = trace.iter().copied().collect();

            rr.line_strips2d("world/circles", &circles, &[CIRCLES], &[0.004])?;
            rr.line_strips2d("world/spokes", &[nodes], &[SPOKES], &[0.006])?;
            rr.line_strips2d("world/trace", &[trace_pts], &[HERO], &[0.02])?;
            rr.points2d_styled("world/tip", &[pen], &[HERO], &[0.03])?;
        }

        // Coefficient error (constant) once per second, and the hud headline.
        if n % HUD_EVERY == 0 {
            rr.scalar("plots/coeff_error", coeff_error)?;
            if let Some(tk) = tick_ring.summary() {
                let md = format!(
                    "## fourier_ferris — multicalc live demo\n\
                     ### Fourier epicycle chain evaluation: median {:.1} µs — {:.2} % of the 1 ms tick\n\
                     ### accuracy: max |c_k(GL) − closed| = {:.1e} ({} node evals in {:.0} ms at startup)\n\
                     ### harmonics active: {} / {}",
                    tk.median,
                    tk.median / 10.0,
                    coeff_error,
                    commas(node_evals),
                    startup_ms,
                    active,
                    max_h,
                );
                rr.text("hud/stats", &md)?;
            }
        }
    }
}
