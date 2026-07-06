//! Loading fixtures from disk and comparing multicalc's output against them.
//!
//! Fixtures are located relative to the crate manifest, not the working
//! directory, so tests find them the same way whether run locally or in CI.
//! The extraction helpers bridge runtime-loaded data into multicalc's
//! const-generic `Matrix`/`Vector` (see D1 in the plan).

use std::io::BufReader;
use std::path::{Path, PathBuf};

use multicalc::linear_algebra::{Matrix, Vector};

use crate::schema::{Fixture, SCHEMA_VERSION, Tol, Value};

/// Reads every `*.json` fixture under `rel` (relative to the crate root),
/// sorted by filename. Panics if the directory is missing, unreadable, holds a
/// file with the wrong schema version, or is empty — a missing fixture set is a
/// bug, never a silent pass.
pub fn load_dir(rel: &str) -> Vec<Fixture> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| unreachable!("read fixture dir {dir:?}: {e}"));
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("json"))
        .collect();
    paths.sort();

    let mut fixtures = Vec::new();
    for path in paths {
        let file =
            std::fs::File::open(&path).unwrap_or_else(|e| unreachable!("open {path:?}: {e}"));
        let fx: Fixture = serde_json::from_reader(BufReader::new(file))
            .unwrap_or_else(|e| unreachable!("parse {path:?}: {e}"));
        assert_eq!(
            fx.schema_version, SCHEMA_VERSION,
            "schema version mismatch in {path:?}"
        );
        fixtures.push(fx);
    }

    assert!(!fixtures.is_empty(), "no fixtures found in {dir:?}");
    fixtures
}

/// Builds an `R×C` `f64` matrix from a `Matrix` value, checking the stored shape.
pub fn to_matrix<const R: usize, const C: usize>(v: &Value) -> Matrix<R, C> {
    let (r, c, d) = v.as_matrix();
    assert_eq!((r, c), (R, C), "matrix shape");
    Matrix::from_fn(|i, j| d[i * C + j])
}

/// Same as [`to_matrix`], narrowing each entry to `f32`.
pub fn to_matrix_f32<const R: usize, const C: usize>(v: &Value) -> Matrix<R, C, f32> {
    let (r, c, d) = v.as_matrix();
    assert_eq!((r, c), (R, C), "matrix shape");
    Matrix::from_fn(|i, j| d[i * C + j] as f32)
}

/// Builds an `N`-element `f64` vector from a `Vector` value, checking its length.
pub fn to_vector<const N: usize>(v: &Value) -> Vector<N> {
    let d = v.as_vector();
    assert_eq!(d.len(), N, "vector length");
    Vector::from_fn(|i| d[i])
}

/// Same as [`to_vector`], narrowing each entry to `f32`.
pub fn to_vector_f32<const N: usize>(v: &Value) -> Vector<N, f32> {
    let d = v.as_vector();
    assert_eq!(d.len(), N, "vector length");
    Vector::from_fn(|i| d[i] as f32)
}

/// True when `got` is within `t` of `want`, using a combined absolute and
/// relative bound: `|got - want| <= abs + rel * max(|got|, |want|)`.
pub fn close(got: f64, want: f64, t: Tol) -> bool {
    (got - want).abs() <= t.abs + t.rel * got.abs().max(want.abs())
}

/// Asserts a scalar matches the expected value within `t`.
pub fn assert_scalar(got: f64, want: &Value, t: Tol, ctx: &str) {
    let w = want.as_scalar();
    assert!(close(got, w, t), "{ctx}: got {got}, want {w}, tol {t:?}");
}

/// Asserts every component of a vector matches within `t`.
pub fn assert_vector<const N: usize>(got: &Vector<N>, want: &Value, t: Tol, ctx: &str) {
    let w = want.as_vector();
    assert_eq!(w.len(), N, "{ctx}: length");
    for i in 0..N {
        assert!(
            close(got[i], w[i], t),
            "{ctx}[{i}]: got {}, want {}, tol {t:?}",
            got[i],
            w[i]
        );
    }
}

/// Asserts every entry of a matrix matches within `t`.
pub fn assert_matrix<const R: usize, const C: usize>(
    got: &Matrix<R, C>,
    want: &Value,
    t: Tol,
    ctx: &str,
) {
    let (r, c, w) = want.as_matrix();
    assert_eq!((r, c), (R, C), "{ctx}: shape");
    for i in 0..R {
        for j in 0..C {
            assert!(
                close(got[(i, j)], w[i * C + j], t),
                "{ctx}({i},{j}): got {}, want {}, tol {t:?}",
                got[(i, j)],
                w[i * C + j]
            );
        }
    }
}
