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
use std::path::Path;

use multicalc_qa::docs::replace_marked_region;
use multicalc_qa::load::load_dir;

const BEGIN: &str = "<!-- BEGIN generated: accuracy -->";
const END: &str = "<!-- END generated -->";

/// A source fixture module and the cases drawn from it, in display order.
struct Module {
    name: &'static str,
    cases: &'static [&'static str],
}

/// A QA doc and the modules whose fixtures populate its accuracy table.
struct Doc {
    file: &'static str,
    modules: &'static [Module],
}

/// Which fixtures each doc shows, in display order. The text of every row
/// comes from the fixture itself.
const DOCS: &[Doc] = &[
    Doc {
        file: "linear_algebra.md",
        modules: &[Module {
            name: "linalg",
            cases: &[
                "lu_3x3",
                "lu_4x4",
                "lu_5x5",
                "cholesky_2x2",
                "cholesky_3x3",
                "cholesky_4x4",
                "qr_3x2",
                "qr_3x3",
                "qr_4x3",
                "qr_20x7",
                "svd_3x2",
                "svd_3x3",
                "svd_4x3",
                "svd_12x6",
                "svd_20x6",
                "symmetric_eigen_3x3",
                "symmetric_eigen_4x4",
                "symmetric_eigen_6x6",
                "symmetric_eigen_indefinite_4x4",
                "symmetric_eigen_indefinite_6x6",
            ],
        }],
    },
    Doc {
        file: "optimization.md",
        modules: &[Module {
            name: "optimization",
            cases: &[
                "rosenbrock",
                "circle_fit",
                "gaussian_peaks",
                "trigonometric6",
            ],
        }],
    },
    Doc {
        file: "calculus.md",
        modules: &[
            Module {
                name: "calculus",
                cases: &[
                    "cube_diff_o1",
                    "cube_diff_o2",
                    "cube_diff_o3",
                    "g_transcendental_partial_x",
                    "g_transcendental_partial_xy",
                    "g_transcendental_partial_xxy",
                    "jacobian_23",
                    "hessian_3x3",
                    "vfield_curl_div",
                    "vfield_line_circle",
                    "vfield_flux_circle",
                    "approx_taylor",
                ],
            },
            Module {
                name: "quadrature",
                cases: &[
                    "two_x_legendre_o4",
                    "quartic_legendre_o4",
                    "quartic_legendre_o16",
                    "cube_booles_120",
                    "two_x_booles_120",
                    "two_x_simpsons_120",
                    "two_x_trapezoidal_120",
                    "exp_neg_sq_booles_120",
                    "x_squared_hermite_o5",
                    "x_squared_laguerre_o5",
                    "inv_1px2_trapezoidal_2p20",
                ],
            },
        ],
    },
    Doc {
        file: "ode.md",
        modules: &[Module {
            name: "ode",
            cases: &["exp_decay", "harmonic", "van_der_pol_mild", "two_body"],
        }],
    },
    Doc {
        file: "estimation.md",
        modules: &[Module {
            name: "estimation",
            cases: &[
                "kalman_filter_constant_velocity_one_dimensional",
                "kalman_filter_constant_velocity_two_dimensional",
                "kalman_filter_with_control_input",
                "extended_kalman_filter_landmark_range_and_bearing",
                "extended_kalman_filter_coordinated_turn_fusion",
                "unscented_kalman_filter_coordinated_turn_fusion",
                "unscented_kalman_filter_landmark_range_and_bearing",
            ],
        }],
    },
    Doc {
        file: "control.md",
        modules: &[Module {
            name: "control",
            // One flying machine end to end — design its position loop, prove that loop
            // settles, hold the attitude it asks for — plus a cart balancing a pole, which is
            // the harder design of the two. The other control fixtures are checked by the QA
            // suite without adding a row here.
            cases: &[
                "riccati_quadrotor_hover",
                "lyapunov_closed_loop_quadrotor_hover",
                "riccati_cart_pole",
                "geometric_attitude_general",
            ],
        }],
    },
    Doc {
        file: "root_finding.md",
        modules: &[Module {
            name: "root_finding",
            cases: &[
                "wien_bisection",
                "wien_newton",
                "kepler_bisection",
                "kepler_newton",
                "colebrook_newton",
                "sigmoid_damped_newton",
                "two_link_ik",
                "circle_hyperbola",
                "equilibrium_3x3",
            ],
        }],
    },
    Doc {
        file: "mjcf.md",
        modules: &[Module {
            name: "mjcf",
            cases: &["skydio_x2_free_joint"],
        }],
    },
    Doc {
        file: "signal_processing.md",
        modules: &[Module {
            name: "signal_processing",
            cases: &[
                "gyro_conditioning_chain",
                "motor_vibration_band_pass",
                "accelerometer_drift_high_pass",
                "encoder_curve_fit",
                "lidar_despiking",
            ],
        }],
    },
    Doc {
        file: "polynomial.md",
        modules: &[Module {
            name: "polynomial",
            cases: &[
                "horner_degree7",
                "derivative_degree5",
                "definite_integral_degree5",
                "product_degree3_by_degree4",
                "composition_degree3_in_degree2",
                "roots_quadratic",
                "roots_quadratic_dominant_linear",
                "roots_cubic_three_real",
                "roots_cubic_one_real",
                "roots_quartic_four_real",
                "roots_quartic_two_real",
                "roots_degree6_sturm",
                "interpolation_five_points",
                "least_squares_cubic_fit",
                "bivariate_evaluation",
                "bivariate_partial_derivatives",
                "trivariate_evaluation",
                "bivariate_product",
                "bivariate_substitution",
            ],
        }],
    },
    Doc {
        file: "motion.md",
        modules: &[Module {
            name: "motion",
            cases: &[
                "minimum_snap_single_segment_3d",
                "minimum_snap_three_segments_3d",
                "minimum_snap_three_segments_moving_ends_3d",
                "minimum_snap_seven_segments_3d",
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

/// Builds the markdown table for a doc, skipping modules whose fixtures are not
/// on disk. Returns an empty string when no rows are available. A fixture that
/// checks several quantities at once contributes one row per operation.
fn build_table(doc: &Doc) -> String {
    let mut body = String::new();
    for module in doc.modules {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(module.name);
        if !dir.exists() {
            continue;
        }
        let fixtures = load_dir(module.name);
        for case in module.cases {
            let Some(fx) = fixtures.iter().find(|f| &f.case == case) else {
                continue;
            };
            for operation in &fx.metadata.operations {
                let _ = write!(
                    body,
                    "\n| {} | {} | {} | {} |",
                    operation,
                    fx.metadata.equation,
                    fmt_num(fx.tolerances.f64.rel),
                    fx.metadata.reference,
                );
            }
        }
    }
    if body.is_empty() {
        return String::new();
    }
    format!(
        "| Operation | Equation | Tolerance | Tested Against |\n| --- | --- | --- | --- |{body}"
    )
}

fn main() {
    let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks");
    for doc in DOCS {
        let path = docs_dir.join(doc.file);
        replace_marked_region(&path, BEGIN, END, &build_table(doc));
        println!("updated {}", path.display());
    }
}
