//! Morphing Newton fractal (root-finding showcase).
//!
//! Every pixel is a full `NewtonSystem` solve of a cubic over ℂ, expressed as a 2-real system.
//! Its autodiff Jacobian is automatically the Cauchy-Riemann-consistent one — this is the
//! root-finding module showing its raw single-core throughput: hundreds of thousands of complete
//! Newton solves per second. The basins swirl as the cubic's three roots orbit, and the dark
//! filigree along the basin boundaries is where Newton fails to converge (a singular Jacobian or
//! an exhausted budget) — handled as pixels, never as errors.
//!
//! Timing model: the morph advances on logical time (a fixed `DTAU` per frame), so cycles are
//! deterministic. Throughput is measured single-core compute (the solve loop is timed tightly, no
//! logging inside), so a transient OS stall inflates one frame's wall time and dips the
//! *instantaneous* `plots/solves_per_sec` — the hud headline reports the robust median over recent
//! frames instead, which is the library's real rate.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example newton_fractal

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::root_finding::NewtonSystem;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc_demos::loop_util::{LatencyRing, commas};
use multicalc_demos::{RerunSink, Rgba, VizError, VizSink};
use std::f64::consts::TAU;
use std::time::Instant;

/// Grid resolution (§6.3: may drop to 192 only on measured < 5 fps).
const N: usize = 256;
const DOMAIN: f64 = 2.0; // half-width: the grid covers [-2, 2]^2
const DTAU: f64 = 0.02; // root-orbit advance per completed frame
const WARMUP_FRAMES: i64 = 3; // cold-start frames excluded from the throughput median

// Basin colors: the validated mutually-CVD-safe trio (§2). RGB for the image, RGBA for the roots.
const BASIN_RGB: [[u8; 3]; 3] = [[0x39, 0x87, 0xe5], [0xc9, 0x85, 0x00], [0xe6, 0x67, 0x67]];
const BASIN_RGBA: [Rgba; 3] = [
    [0x39, 0x87, 0xe5, 0xff],
    [0xc9, 0x85, 0x00, 0xff],
    [0xe6, 0x67, 0x67, 0xff],
];
const SURFACE: [u8; 3] = [0x14, 0x16, 0x19]; // fade-to color for iteration shading
const FILIGREE: [u8; 3] = [0x1c, 0x20, 0x23]; // near-surface dark for non-converged pixels

const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff];
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff];

/// The cubic `p(z) = Π (z − r_j)` as a 2-real system; the complex product is inlined generically,
/// so autodiff yields the Cauchy-Riemann-consistent Jacobian.
struct CubicBasins {
    roots: [[f64; 2]; 3],
}

impl VectorFn<2, 2> for CubicBasins {
    fn eval<S: Numeric>(&self, z: &[S; 2]) -> [S; 2] {
        let (mut wr, mut wi) = (S::ONE, S::ZERO); // w = Π (z − r_i), complex MAC
        for r in &self.roots {
            let dr = z[0] - S::from_f64(r[0]);
            let di = z[1] - S::from_f64(r[1]);
            let nr = wr * dr - wi * di;
            let ni = wr * di + wi * dr;
            wr = nr;
            wi = ni;
        }
        [wr, wi]
    }
}

/// The three orbiting roots at choreography parameter `tau`.
fn roots_at(tau: f64) -> [[f64; 2]; 3] {
    core::array::from_fn(|j| {
        let theta = TAU * j as f64 / 3.0 + 0.31 * tau;
        let radius = 1.0 + 0.25 * (0.7 * tau + 2.1 * j as f64).sin();
        [radius * theta.cos(), radius * theta.sin()]
    })
}

/// Per-channel lerp from `a` toward `b` by `t` in [0, 1].
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f64) -> [u8; 3] {
    core::array::from_fn(|k| (a[k] as f64 + (b[k] as f64 - a[k] as f64) * t).round() as u8)
}

/// Shades a converged pixel: fewer iterations are brighter, more fade toward the surface.
fn shade(basin: [u8; 3], iterations: usize) -> [u8; 3] {
    let t = (0.20 * iterations as f64 / 6.0).min(1.0) * 0.85;
    lerp_rgb(basin, SURFACE, t)
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — throughput numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example newton_fractal"
        );
    }

    let mut rr = RerunSink::live("multicalc-demos/newton-fractal")?;
    rr.set_sequence("frame", 0);
    rr.series_style(
        "plots/solves_per_sec",
        HERO,
        "Newton solves/s (one core)",
        2.0,
    )?;
    rr.series_style("plots/mean_iterations", TARGET, "mean iterations", 1.0)?;

    // Built once; solve takes &self. Backtracking stays off (default) — it would smooth away the
    // wild Newton steps that make the filigree.
    let solver = NewtonSystem::<AutoDiffMulti>::default()
        .with_max_iterations(24)
        .with_ftol(1e-13);

    let mut basins = CubicBasins {
        roots: roots_at(0.0),
    };
    let mut buf = vec![0u8; N * N * 3];
    let step = 2.0 * DOMAIN / N as f64; // world units per pixel
    let root_radius_px = 0.05 / step; // spec's 0.05 scene-unit radius, in pixels

    let mut throughput_ring = LatencyRing::new(256); // recent per-frame solves/s, for a stable median
    let mut tau = 0.0;
    let mut frame: i64 = 0;
    loop {
        frame += 1;
        rr.set_sequence("frame", frame);
        basins.roots = roots_at(tau);

        let mut res_sum = 0.0;
        let mut iter_sum = 0u64;
        let mut converged = 0u64;

        let t0 = Instant::now();
        for row in 0..N {
            // Row 0 maps to y = +DOMAIN (top), matching the image's y-down convention.
            let y = DOMAIN - (row as f64 + 0.5) * step;
            for col in 0..N {
                let x = -DOMAIN + (col as f64 + 0.5) * step;
                let rgb = match solver.solve(&basins, &[x, y]) {
                    Ok(rep) => {
                        let (basin, _) = basins
                            .roots
                            .iter()
                            .enumerate()
                            .map(|(j, rj)| {
                                let (dx, dy) = (rep.root[0] - rj[0], rep.root[1] - rj[1]);
                                (j, dx * dx + dy * dy)
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap();
                        res_sum += rep.residual_norm;
                        iter_sum += rep.iterations as u64;
                        converged += 1;
                        shade(BASIN_RGB[basin], rep.iterations)
                    }
                    // SingularMatrix / DidNotConverge / NonFiniteValue: basin-boundary filigree.
                    Err(_) => FILIGREE,
                };
                let i = (row * N + col) * 3;
                buf[i..i + 3].copy_from_slice(&rgb);
            }
        }
        let frame_secs = t0.elapsed().as_secs_f64();
        let solves_per_sec = (N * N) as f64 / frame_secs;

        // Roots in the image's pixel space so they overlay the fractal.
        let roots_px: Vec<[f64; 2]> = basins
            .roots
            .iter()
            .map(|r| [(r[0] + DOMAIN) / step, (DOMAIN - r[1]) / step])
            .collect();

        if frame > WARMUP_FRAMES {
            throughput_ring.push(solves_per_sec);
        }
        let throughput_median = throughput_ring
            .summary()
            .map_or(solves_per_sec, |s| s.median);

        rr.image_rgb8("world/fractal", N as u32, N as u32, &buf)?;
        rr.points2d_styled(
            "world/roots",
            &roots_px,
            &BASIN_RGBA,
            &[root_radius_px as f32],
        )?;
        rr.scalar("plots/solves_per_sec", solves_per_sec)?;

        let conv = converged.max(1) as f64;
        let mean_iterations = iter_sum as f64 / conv;
        let mean_residual = res_sum / conv; // shown in the hud, not plotted
        rr.scalar("plots/mean_iterations", mean_iterations)?;

        let nonconverged_pct = (N * N - converged as usize) as f64 / (N * N) as f64 * 100.0;
        let md = format!(
            "## newton_fractal — multicalc live demo\n\
             ### throughput: {} Newton solves/s (one core, no-std math; median over recent frames)\n\
             ### per frame: {}×{} = {} full solves in {:.0} ms\n\
             ### mean iterations: {:.1}, non-converged: {:.2} %\n\
             ### accuracy: mean residual ‖F‖ = {:.1e}",
            commas(throughput_median as u64),
            N,
            N,
            commas((N * N) as u64),
            frame_secs * 1000.0,
            mean_iterations,
            nonconverged_pct,
            mean_residual,
        );
        rr.text("hud/stats", &md)?;

        tau += DTAU;
    }
}
