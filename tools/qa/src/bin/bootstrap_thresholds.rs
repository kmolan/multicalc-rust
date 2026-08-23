//! Bootstraps the small-angle threshold for each trig ratio in `spatial::lie`.
//!
//! **Theory**
//! Given a trig function `f(θ)` with `θ` is the angle. There are two types of error
//! related to the value of `f`:
//! - Representation error of `θ`: as `f64` only have 52 bits in the mantissa part,
//!   the absolute error magnitue of 1 ULP is around 2^-52 = 2.22e-16;
//! - Calculation error: this accumulates as the representation error of input `θ`
//!   flows through the calculation chain;
//!
//! One solution is to use a Taylor series to approximate that represented value, instead of
//! calculating from `f(θ)`. This does not mitigate the error, but makes the calculation error
//! less severe compared to closed form trig function,
//! as Taylor form is much simpler (just polynormial).
//!
//! **Method**
//! Let's call:
//! - `f(θ)`: the trig function under scrutiny;
//! - `t(θ)`: the Taylor function of `f`;
//! - `d(θ)`: the discrepancy between `f` and `t`;
//! - `E`: the exact floating point value
//! - `B`: a target value that `θ` approaches;
//!
//! We have `d(θ)` the discrepancy between `f` and `t`. This already incorporate errors.
//!
//! ```tex
//! d(θ) = |f(θ) − t(θ)| = |(f − E) − (t − E)| = |Err(f) − Err(t)|
//! ```
//! Given a small enough threshold `T` of `θ` at which the calculation error of `f` jumps.
//! When `θ` decreases towards `T`, the error of both `f` and `g` got smaller, `d(0)` got smaller.
//! At `T`, the error of `f` jumps, `d(0)` jumps up.
//!
//! `Err(f)` is cancellation error, which grows as θ → 0; `Err(t)` is truncation error, which
//! grows as θ → ∞. `d(0)` has a V shape whose vertex is the crossover.
//! That locates the threshold without ever needing a high-precision reference:
//! the two f64 branches are compared against each other.
//!
//! Run with `cargo run -p multicalc-qa --bin bootstrap_thresholds`.

use core::f64;
use std::cmp::Ordering;

use multicalc::Numeric;

trait Scalar: Numeric + std::fmt::LowerExp + std::fmt::Display {}
impl<T: Numeric + std::fmt::LowerExp + std::fmt::Display> Scalar for T {}

/// One sweep point, just for f32 and f64, no need for monomorphism
struct Sample<T: Numeric> {
    theta: T,
    discrepancy: T,
}

/// Sweeps θ over `[lower, upper]` and returns the samples, highest θ first.
fn sweep<T, F, G>(
    closed_form: F,
    taylor: G,
    lower: T,
    upper: T,
    steps_per_octave: u64,
) -> Vec<Sample<T>>
where
    T: Numeric,
    F: Fn(T) -> T,
    G: Fn(T) -> T,
{
    // theta decreases at diminishing step size
    let ratio = T::HALF.powf(T::ONE / T::from_u64(steps_per_octave));
    let mut samples = Vec::new();
    let mut theta = upper;
    while theta >= lower {
        samples.push(Sample {
            theta,
            discrepancy: (closed_form(theta) - taylor(theta)).abs(),
        });
        theta *= ratio;
    }
    samples
}

/// The vertex of the V, i.e. the crossover.
///
/// |d| is not continous at threashold `T`, so cannot find derivative of |d| at `T`.
///
/// |d| jitters by up to a decade between adjacent θ because the cancellation error is rounding
/// luck, so a raw `argmin` lands in whichever dip happened to be lucky. Taking the max over a
/// small window first builds an upper envelope, and the envelope's minimum is stable.
/// Pre-condition: samples are all positive and produced by a convex function
///
/// **Method:**
/// - Scan over +/- window size around index i, search for max discrepancy as upper envelope
/// - Slide i forwards, continue the process, saving i when new upper < previous upper envelope
/// - From a certain threshold of samples, discrepancy stops decreasing
fn crossover<T: Numeric>(samples: &[Sample<T>], window: usize) -> T {
    // Max discrepancy, i, hi, lo
    let mut best = (T::INFINITY, T::NAN);
    for i in 0..samples.len() {
        let lower = i.saturating_sub(window);
        let upper = (i + window + 1).min(samples.len());
        let envelope = samples[lower..upper]
            .iter()
            .fold(T::ZERO, |acc, sample| acc.max(sample.discrepancy));
        if envelope < best.0 {
            best = (envelope, samples[i].theta);
        }
    }
    best.1
}

/// Run bootstrapping and print out result:
/// - `name`: of the run
/// - `close_form`: function closed-form
/// - `samples`: bootstrapping results
/// - `theta_modeled_name`: theta according to closed form of error model
/// - `theta_modeled`: value of modeled theta
fn report<T: Numeric + std::fmt::LowerExp + std::fmt::Display>(
    name: &str,
    closed_form: &str,
    samples: &[Sample<T>],
    theta_modeled_form: &str,
    theta_modeled: T,
) {
    let theta = crossover(samples, 2);
    // Error at the changeover: |d| there is the sum of two errors of similar size, so either
    // branch carries about half of it.
    // `partial_cmp` is `None` only for NaN; treating that as `Equal` just means it never wins.
    let Some(threshold) = samples.iter().min_by(|a, b| {
        (a.theta - theta)
            .abs()
            .partial_cmp(&(b.theta - theta).abs())
            .unwrap_or(Ordering::Equal)
    }) else {
        println!("{name:8}  {closed_form:25}  no samples");
        return;
    };

    let vs_bootstrap = theta_modeled / theta;

    println!(
        "{name:8}  {closed_form:25}  bootstrapped ≈ {theta:.3e}    \
        EPS form = {theta_modeled_form:16}    EPS value ≈ {theta_modeled:.3e}  \
        {vs_bootstrap:.2} x bootstrapped    \
        error there ≈ {:.1e}",
        threshold.discrepancy / T::from_f64(2.0)
    );
}

fn run_all<T: Scalar>(label: &str) {
    let lower = T::from_f64(1.0e-9);
    let upper = T::PI;
    const PER_OCTAVE: u64 = 8;

    println!(
        "==={label}=== crossover of |closed form − Taylor|, swept θ ∈ [{lower:.0e}, {upper:.0e}]\n"
    );

    let thalf = T::from_f64(0.5);
    let one = T::from_f64(1.0);
    let six = T::from_f64(6.0);
    let eight = T::from_f64(8.0);

    // --- left_jacobian_so3 ------------------------------------------------
    report(
        "so3 c1",
        "(1 − cos θ)/θ²",
        &sweep(
            |x| (one - x.cos()) / (x * x),
            |x| thalf - x * x / T::from_f64(24.0),
            lower,
            upper,
            PER_OCTAVE,
        ),
        "(360ϵ)^(1/6)",
        (T::from_f64(360.0) * T::EPSILON).powf(one / six),
    );
    report(
        "so3 c2",
        "(θ − sin θ)/θ³",
        &sweep(
            |x| (x - x.sin()) / (x * x * x),
            |x| one / six - x * x / T::from_f64(120.0),
            lower,
            upper,
            PER_OCTAVE,
        ),
        "(2520ϵ)^(1/6)",
        (T::from_f64(2520.0) * T::EPSILON).powf(one / six),
    );

    // --- inverse_left_jacobian_so3 ----------------------------------------
    report(
        "inv c3",
        "(1 − (θ/2)·cot(θ/2))/θ²",
        &sweep(
            |x| {
                let h = x * thalf;
                (one - h * (h.cos() / h.sin())) / (x * x)
            },
            |x| one / T::from_f64(12.0) + x * x / T::from_f64(720.0),
            lower,
            upper,
            PER_OCTAVE,
        ),
        "(16*945ϵ)^(1/6)",
        (T::from_f64(16.0) * T::from_f64(945.0) * T::EPSILON).powf(one / six),
    );

    // --- q_matrix_se3 -----------------------------------------------------
    report(
        "q c2",
        "(θ − sin θ)/θ³",
        &sweep(
            |x| (x - x.sin()) / (x * x * x),
            |x| one / six - x * x / T::from_f64(120.0),
            lower,
            upper,
            PER_OCTAVE,
        ),
        "(2520ϵ)^(1/6)",
        (T::from_f64(2520.0) * T::EPSILON).powf(one / six),
    );
    // NOTE: the series below is NOT the one in the source. The closed form
    // (1 − θ²/2 − cos θ)/θ⁴ expands to −1/24 + θ²/720, but `q_matrix_se3` codes
    // +1/24 − θ²/720. Sign flipped. Swap the sign back to see the V fail to form.
    report(
        "q c3",
        "(1 − θ²/2 − cos θ)/θ⁴",
        &sweep(
            |x| (one - x * x * thalf - x.cos()) / (x * x * x * x),
            |x| -one / T::from_f64(24.0) + x * x / T::from_f64(720.0),
            lower,
            upper,
            PER_OCTAVE,
        ),
        "(12*1680ϵ)^(1/8)",
        (T::from_f64(12.0) * T::from_f64(1690.0) * T::EPSILON).powf(one / eight),
    );
    report(
        "q c5",
        "(θ − sin θ − θ³/6)/θ⁵",
        &sweep(
            |x| (x - x.sin() - x * x * x / six) / (x * x * x * x * x),
            |x| -one / T::from_f64(120.0) + x * x / T::from_f64(5040.0),
            lower,
            upper,
            PER_OCTAVE,
        ),
        "(0.5*9!ϵ)^(1/8)",
        (T::from_f64(181_440.0) * T::EPSILON).powf(one / eight),
    );
}

fn main() {
    run_all::<f64>("f64");
    println!();
    run_all::<f32>("f32");
}
