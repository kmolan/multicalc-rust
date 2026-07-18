//! Rewrites the accuracy tables in the QA docs from the committed fixtures, so
//! the published tolerances cannot drift from what the QA suite verifies. For each
//! doc it replaces the region between the sentinel comments
//!
//! ```text
//! <!-- BEGIN generated: accuracy -->
//! <!-- END generated -->
//! ```
//!
//! with a table of Operation, Equation, Tolerance, and Tested Against, one row per case.
//! Prose and everything outside the markers are left untouched. A source module
//! whose fixture directory is absent contributes no rows.
//!
//! Run `cargo run -p multicalc-qa --bin gen_accuracy_tables`. The output is
//! deterministic, so CI regenerates it and diffs against the committed docs.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use multicalc_qa::load::load_dir;
use multicalc_qa::schema::{Fixture, Tol};

const BEGIN: &str = "<!-- BEGIN generated: accuracy -->";
const END: &str = "<!-- END generated -->";

/// One fixture rendered as a table row: its case, the operation performed, and the
/// equation it operates on. The tolerance and source are read back from the fixture.
struct Row {
    case: &'static str,
    operation: &'static str,
    equation: &'static str,
}

/// A source fixture module and the rows drawn from it, in display order.
struct Module {
    dir: &'static str,
    rows: &'static [Row],
}

/// A QA doc and the modules whose fixtures populate its accuracy table.
struct Doc {
    file: &'static str,
    modules: &'static [Module],
}

/// The hand-maintained mapping: fixture case to the operation and equation shown.
/// Everything else in each row (tolerance, source) is read back from the fixture.
const DOCS: &[Doc] = &[
    Doc {
        file: "linear_algebra.md",
        modules: &[Module {
            dir: "fixtures/v1/linalg",
            rows: &[
                Row {
                    case: "lu_3x3",
                    operation: "LU decompose + solve, 3×3",
                    equation: "det(A)",
                },
                Row {
                    case: "lu_4x4",
                    operation: "LU decompose + solve, 4×4",
                    equation: "det(A)",
                },
                Row {
                    case: "lu_5x5",
                    operation: "LU decompose + solve, 5×5",
                    equation: "det(A)",
                },
                Row {
                    case: "cholesky_2x2",
                    operation: "Cholesky decompose + solve, 2×2",
                    equation: "det(A)",
                },
                Row {
                    case: "cholesky_3x3",
                    operation: "Cholesky decompose + solve, 3×3",
                    equation: "det(A)",
                },
                Row {
                    case: "cholesky_4x4",
                    operation: "Cholesky decompose + solve, 4×4",
                    equation: "det(A)",
                },
                Row {
                    case: "qr_3x2",
                    operation: "QR least-squares, 3×2",
                    equation: "‖Ax − b‖",
                },
                Row {
                    case: "qr_3x3",
                    operation: "QR least-squares, 3×3",
                    equation: "‖Ax − b‖",
                },
                Row {
                    case: "qr_4x3",
                    operation: "QR least-squares, 4×3",
                    equation: "‖Ax − b‖",
                },
                Row {
                    case: "qr_20x7",
                    operation: "QR least-squares, 20×7",
                    equation: "‖Ax − b‖",
                },
                Row {
                    case: "svd_3x2",
                    operation: "SVD, 3×2",
                    equation: "σ(A)",
                },
                Row {
                    case: "svd_3x3",
                    operation: "SVD, 3×3",
                    equation: "σ(A)",
                },
                Row {
                    case: "svd_4x3",
                    operation: "SVD, 4×3",
                    equation: "σ(A)",
                },
                Row {
                    case: "svd_12x6",
                    operation: "SVD, 12×6",
                    equation: "σ(A)",
                },
                Row {
                    case: "svd_20x6",
                    operation: "SVD, 20×6",
                    equation: "σ(A)",
                },
            ],
        }],
    },
    Doc {
        file: "optimization.md",
        modules: &[Module {
            dir: "fixtures/v1/optimization",
            rows: &[
                Row {
                    case: "rosenbrock",
                    operation: "Rosenbrock least-squares minimizer",
                    equation: "min ‖[10(y − x²), 1 − x]‖²",
                },
                Row {
                    case: "circle_fit",
                    operation: "Geometric circle fit, 40 points",
                    equation: "rᵢ = √((xᵢ − cₓ)² + (yᵢ − cᵧ)²) − r",
                },
                Row {
                    case: "gaussian_peaks",
                    operation: "Two Gaussian peaks fit, 50 samples",
                    equation: "rᵢ = Σₖ aₖ·e^(−((tᵢ − μₖ)/σₖ)²) − yᵢ",
                },
                Row {
                    case: "trigonometric6",
                    operation: "Trigonometric least-squares, 6 vars",
                    equation: "rᵢ = n − Σⱼcos xⱼ + i(1 − cos xᵢ) − sin xᵢ",
                },
            ],
        }],
    },
    Doc {
        file: "calculus.md",
        modules: &[
            Module {
                dir: "fixtures/v1/calculus",
                rows: &[
                    Row {
                        case: "cube_diff_o1",
                        operation: "1st derivative at x=2",
                        equation: "x³",
                    },
                    Row {
                        case: "cube_diff_o2",
                        operation: "2nd derivative at x=2",
                        equation: "x³",
                    },
                    Row {
                        case: "cube_diff_o3",
                        operation: "3rd derivative at x=2",
                        equation: "x³",
                    },
                    Row {
                        case: "g_transcendental_partial_x",
                        operation: "∂/∂x at (1,2,3)",
                        equation: "g = y·sin x + x·cos y + x·y·eᶻ",
                    },
                    Row {
                        case: "g_transcendental_partial_xy",
                        operation: "∂²/∂x∂y at (1,2,3)",
                        equation: "g = y·sin x + x·cos y + x·y·eᶻ",
                    },
                    Row {
                        case: "g_transcendental_partial_xxy",
                        operation: "∂³/∂x²∂y at (1,2,3)",
                        equation: "g = y·sin x + x·cos y + x·y·eᶻ",
                    },
                    Row {
                        case: "jacobian_23",
                        operation: "Jacobian at (1,2,3)",
                        equation: "[x·y·z, x² + y²]",
                    },
                    Row {
                        case: "hessian_3x3",
                        operation: "Hessian at (1,2,3)",
                        equation: "y·sin x + 2x·eʸ + z²",
                    },
                    Row {
                        case: "vfield_curl_div",
                        operation: "Curl at (1,2,3)",
                        equation: "[y, -x, 2z]",
                    },
                    Row {
                        case: "vfield_curl_div",
                        operation: "Divergence at (1,2,3)",
                        equation: "[y, -x, 2z]",
                    },
                    Row {
                        case: "vfield_line_circle",
                        operation: "Line integral on the unit circle",
                        equation: "[y, -x]",
                    },
                    Row {
                        case: "vfield_flux_circle",
                        operation: "Flux integral on the unit circle",
                        equation: "[y, -x]",
                    },
                    Row {
                        case: "approx_taylor",
                        operation: "Linear Taylor predict at (1.1,2.1,2.9)",
                        equation: "x + y² + z³",
                    },
                    Row {
                        case: "approx_taylor",
                        operation: "Quadratic Taylor predict at (1.1,2.1,2.9)",
                        equation: "x + y² + z³",
                    },
                ],
            },
            Module {
                dir: "fixtures/v1/quadrature",
                rows: &[
                    Row {
                        case: "two_x_legendre_o4",
                        operation: "Gauss-Legendre order 4 on [0,2]",
                        equation: "∫ 2x dx",
                    },
                    Row {
                        case: "quartic_legendre_o4",
                        operation: "Gauss-Legendre order 4 on [0,2]",
                        equation: "∫ 4x³ − 3x² dx",
                    },
                    Row {
                        case: "quartic_legendre_o16",
                        operation: "Gauss-Legendre order 16 on [0,2]",
                        equation: "∫ 4x³ − 3x² dx",
                    },
                    Row {
                        case: "cube_booles_120",
                        operation: "Boole's rule, 120 intervals on [0,2]",
                        equation: "∫ x³ dx",
                    },
                    Row {
                        case: "two_x_booles_120",
                        operation: "Boole's rule, 120 intervals on [0,2]",
                        equation: "∫ 2x dx",
                    },
                    Row {
                        case: "two_x_simpsons_120",
                        operation: "Simpson's rule, 120 intervals on [0,2]",
                        equation: "∫ 2x dx",
                    },
                    Row {
                        case: "two_x_trapezoidal_120",
                        operation: "Trapezoidal, 120 intervals on [0,2]",
                        equation: "∫ 2x dx",
                    },
                    Row {
                        case: "exp_neg_sq_booles_120",
                        operation: "Boole's rule, 120 intervals on [-5,5]",
                        equation: "∫ e^{-x²} dx",
                    },
                    Row {
                        case: "x_squared_hermite_o5",
                        operation: "Gauss-Hermite order 5 on ℝ",
                        equation: "∫ x²e^{-x²} dx",
                    },
                    Row {
                        case: "x_squared_laguerre_o5",
                        operation: "Gauss-Laguerre order 5 on [0,∞)",
                        equation: "∫ x²e^{-x} dx",
                    },
                    Row {
                        case: "inv_1px2_trapezoidal_2p20",
                        operation: "Trapezoidal, 2²⁰ intervals on [0,1]",
                        equation: "∫ 1/(1+x²) dx",
                    },
                ],
            },
        ],
    },
    Doc {
        file: "ode.md",
        modules: &[Module {
            dir: "fixtures/v1/ode",
            rows: &[
                Row {
                    case: "exp_decay",
                    operation: "Exponential decay, RK45",
                    equation: "y' = -y",
                },
                Row {
                    case: "harmonic",
                    operation: "Harmonic oscillator, RK45",
                    equation: "y' = [y₁, -y₀]",
                },
                Row {
                    case: "van_der_pol_mild",
                    operation: "Van der Pol (μ=1), RK45",
                    equation: "y' = [y₁, (1 - y₀²)·y₁ - y₀]",
                },
                Row {
                    case: "two_body",
                    operation: "Two-body orbit, RK45",
                    equation: "y' = [vₓ, vᵧ, -x/r³, -y/r³]",
                },
            ],
        }],
    },
    Doc {
        file: "estimation.md",
        modules: &[Module {
            dir: "fixtures/v1/estimation",
            rows: &[
                Row {
                    case: "kalman_filter_constant_velocity_one_dimensional",
                    operation: "Linear Kalman filter, 8 steps, state 2 / measurement 1",
                    equation: "F = [[1, 1], [0, 1]], H = [1, 0]",
                },
                Row {
                    case: "kalman_filter_constant_velocity_two_dimensional",
                    operation: "Linear Kalman filter, 10 steps, state 4 / measurement 2",
                    equation: "F = blkdiag([[1, 0.5], [0, 1]], [[1, 0.5], [0, 1]]), H = [[1, 0, 0, 0], [0, 0, 1, 0]]",
                },
                Row {
                    case: "kalman_filter_with_control_input",
                    operation: "Linear Kalman filter with control, 8 steps, control 1",
                    equation: "x⁻ = [[1, 1], [0, 1]]·x + [[0.5], [1]]·u, H = [1, 0]",
                },
                Row {
                    case: "extended_kalman_filter_landmark_range_and_bearing",
                    operation: "Extended Kalman filter, 8 steps, state 3 / measurement 2",
                    equation: "h = [√((3−x)²+(4−y)²), atan2(4−y, 3−x)−θ], F = I",
                },
            ],
        }],
    },
    Doc {
        file: "root_finding.md",
        modules: &[Module {
            dir: "fixtures/v1/root_finding",
            rows: &[
                Row {
                    case: "wien_bisection",
                    operation: "Bisection on [1,10]",
                    equation: "−5 + x + 5e^{-x} = 0",
                },
                Row {
                    case: "wien_newton",
                    operation: "Newton from x=5",
                    equation: "−5 + x + 5e^{-x} = 0",
                },
                Row {
                    case: "kepler_bisection",
                    operation: "Bisection on [0,π]",
                    equation: "E − e·sin E − M = 0 (e=0.8)",
                },
                Row {
                    case: "kepler_newton",
                    operation: "Newton",
                    equation: "E − e·sin E − M = 0 (e=0.8)",
                },
                Row {
                    case: "colebrook_newton",
                    operation: "Newton",
                    equation: "1/√f + 2·log₁₀(ε/3.7 + 2.51/(Re√f)) = 0",
                },
                Row {
                    case: "sigmoid_damped_newton",
                    operation: "Damped Newton",
                    equation: "x/√(1+x²) = 0",
                },
                Row {
                    case: "two_link_ik",
                    operation: "Newton system, 2×2",
                    equation: "two-link IK: tip at (pₓ, pᵧ)",
                },
                Row {
                    case: "circle_hyperbola",
                    operation: "Newton system, 2×2",
                    equation: "[x²+y²−4, x·y−1] = 0",
                },
                Row {
                    case: "equilibrium_3x3",
                    operation: "Newton system, 3×3",
                    equation: "[x+y+z−1, y−1.25x², z−5x·y] = 0",
                },
            ],
        }],
    },
];

/// A single float, in trimmed scientific notation: `4e0`, `1.5e-15`, `0`.
fn fmt_num(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let s = format!("{x:.6e}");
    let (mantissa, exp) = s
        .split_once('e')
        .unwrap_or_else(|| unreachable!("float format has no exponent: {s}"));
    let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
    format!("{mantissa}e{exp}")
}

/// The relative `f64/host` tolerance.
fn fmt_tol(t: Tol) -> String {
    fmt_num(t.rel)
}

/// The reference source, from the fixture's generator and pinned library
/// versions, so the doc's provenance cannot drift from the fixture.
fn source(fx: &Fixture) -> String {
    let libs = &fx.metadata.libraries;
    let version = |k: &str| libs.get(k).cloned().unwrap_or_default();
    if fx.metadata.generator == "calculus" {
        format!("closed-form analytic (mpmath {})", version("mpmath"))
    } else if fx.metadata.generator == "ode" {
        format!("SciPy solve_ivp {}", version("scipy"))
    } else if libs.contains_key("filterpy") {
        format!("FilterPy {}", version("filterpy"))
    } else if libs.contains_key("scipy") {
        format!("SciPy/MINPACK {}", version("scipy"))
    } else if libs.contains_key("numpy") {
        format!("numpy/LAPACK {}", version("numpy"))
    } else if libs.contains_key("mpmath") {
        format!("mpmath {}", version("mpmath"))
    } else {
        unreachable!("fixture {:?} names no known reference library", fx.case)
    }
}

/// Builds the markdown table for a doc, skipping modules whose fixtures are not
/// on disk. Returns an empty string when no rows are available.
fn build_table(doc: &Doc) -> String {
    let mut body = String::new();
    for module in doc.modules {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(module.dir);
        if !dir.exists() {
            continue;
        }
        let fixtures = load_dir(module.dir);
        for row in module.rows {
            let Some(fx) = fixtures.iter().find(|f| f.case == row.case) else {
                continue;
            };
            let tol = fx
                .tolerances
                .table
                .get("f64/host")
                .copied()
                .unwrap_or_else(|| {
                    unreachable!("fixture {:?} lacks an f64/host tolerance", fx.case)
                });
            let _ = write!(
                body,
                "\n| {} | {} | {} | {} |",
                row.operation,
                row.equation,
                fmt_tol(tol),
                source(fx)
            );
        }
    }
    if body.is_empty() {
        return String::new();
    }
    format!(
        "| Operation | Equation | Tolerance | Tested Against |\n| --- | --- | --- | --- |{body}"
    )
}

/// Replaces the marked region of `path` with `table`, leaving all else intact.
fn rewrite_region(path: &Path, table: &str) {
    let content = fs::read_to_string(path).unwrap_or_else(|e| unreachable!("read {path:?}: {e}"));
    let begin = content
        .find(BEGIN)
        .unwrap_or_else(|| unreachable!("no BEGIN marker in {path:?}"));
    let end = content
        .find(END)
        .unwrap_or_else(|| unreachable!("no END marker in {path:?}"));
    let before = &content[..begin + BEGIN.len()];
    let after = &content[end..];
    let new = format!("{before}\n{table}\n{after}");
    fs::write(path, new).unwrap_or_else(|e| unreachable!("write {path:?}: {e}"));
}

fn main() {
    let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks");
    for doc in DOCS {
        let path = docs_dir.join(doc.file);
        let table = build_table(doc);
        rewrite_region(&path, &table);
        println!("updated {}", path.display());
    }
}
