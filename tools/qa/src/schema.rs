//! Fixture schema: hex-encoded floats, tagged values, tolerances, and metadata.
//!
//! Floating-point numbers are stored as their IEEE-754 bit patterns in hex so a
//! fixture round-trips through JSON with no loss. Serialization is byte-for-byte
//! stable, which lets a regenerated fixture be diffed against the committed one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Current fixture format. A file with a different version is rejected on load.
pub const SCHEMA_VERSION: u32 = 2;

/// An `f64` stored as its 64-bit pattern, written as `"0x{:016x}"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F64(pub u64);

impl F64 {
    pub fn from_f64(x: f64) -> Self {
        F64(x.to_bits())
    }
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl Serialize for F64 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{:016x}", self.0))
    }
}

impl<'de> Deserialize<'de> for F64 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let hex = s.strip_prefix("0x").unwrap_or(&s);
        u64::from_str_radix(hex, 16)
            .map(F64)
            .map_err(serde::de::Error::custom)
    }
}

/// A self-describing fixture value. Floats are hex; ints and strings are plain
/// JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Value {
    Scalar {
        v: F64,
    },
    Vector {
        data: Vec<F64>,
    },
    Matrix {
        rows: usize,
        cols: usize,
        row_major: Vec<F64>,
    },
    Int {
        v: i64,
    },
    Str {
        v: String,
    },
}

impl Value {
    pub fn as_scalar(&self) -> f64 {
        match self {
            Value::Scalar { v } => v.to_f64(),
            other => unreachable!("expected scalar value, got {other:?}"),
        }
    }

    pub fn as_vector(&self) -> Vec<f64> {
        match self {
            Value::Vector { data } => data.iter().map(|b| b.to_f64()).collect(),
            other => unreachable!("expected vector value, got {other:?}"),
        }
    }

    pub fn as_matrix(&self) -> (usize, usize, Vec<f64>) {
        match self {
            Value::Matrix {
                rows,
                cols,
                row_major,
            } => (*rows, *cols, row_major.iter().map(|b| b.to_f64()).collect()),
            other => unreachable!("expected matrix value, got {other:?}"),
        }
    }

    /// The `(rows, cols)` of a matrix value, without decoding its entries.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Value::Matrix { rows, cols, .. } => (*rows, *cols),
            other => unreachable!("expected matrix value, got {other:?}"),
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int { v } => *v,
            other => unreachable!("expected int value, got {other:?}"),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Value::Str { v } => v,
            other => unreachable!("expected string value, got {other:?}"),
        }
    }
}

pub use multicalc_testkit::tol::Tol;

/// How close a computed value has to be to the golden, per scalar type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tolerances {
    /// Bound for the `f64` result, which is the one the golden holds.
    pub f64: Tol,
    /// Bound for the `f32` rerun, on the fixtures that have one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f32: Option<Tol>,
}

/// Provenance for a fixture: the generator, exact library versions, seed, date,
/// a note on how inputs were sampled, and the text the accuracy tables show.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub generator: String,
    pub libraries: BTreeMap<String, String>,
    pub seed: u64,
    pub date: String,
    pub sampling: String,
    /// The problem the fixture is built on, written the way it appears in the
    /// accuracy tables.
    pub equation: String,
    /// What was computed from that problem. Usually one entry; a fixture that
    /// checks several quantities at once lists one per table row.
    pub operations: Vec<String>,
    /// The library and version the golden came from.
    pub reference: String,
}

/// One golden case: its inputs, expected outputs, and the tolerances to compare
/// against.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fixture {
    pub schema_version: u32,
    pub metadata: Metadata,
    pub module: String,
    pub case: String,
    pub tolerances: Tolerances,
    pub inputs: BTreeMap<String, Value>,
    pub expected: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn hex_floats_round_trip() {
        // Exercise ordinary values plus the tricky bit patterns from D6.
        let values = [1.0_f64, -1.0, 0.5, -0.0, f64::INFINITY, f64::NEG_INFINITY];
        for &x in &values {
            let bits = F64::from_f64(x);
            let json = serde_json::to_string(&bits).unwrap();
            let back: F64 = serde_json::from_str(&json).unwrap();
            assert_eq!(bits, back);
            assert_eq!(back.to_f64().to_bits(), x.to_bits());
        }
    }

    #[test]
    fn fixture_round_trips_bit_identically() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "A".to_string(),
            Value::Matrix {
                rows: 2,
                cols: 2,
                row_major: vec![
                    F64::from_f64(1.0),
                    F64::from_f64(-0.0),
                    F64::from_f64(f64::INFINITY),
                    F64::from_f64(2.5),
                ],
            },
        );
        let mut expected = BTreeMap::new();
        expected.insert(
            "det".to_string(),
            Value::Scalar {
                v: F64::from_f64(2.5),
            },
        );

        let mut libraries = BTreeMap::new();
        libraries.insert("numpy".to_string(), "2.1.3".to_string());

        let fx = Fixture {
            schema_version: SCHEMA_VERSION,
            metadata: Metadata {
                generator: "test".to_string(),
                libraries,
                seed: 20260706,
                date: "2026-07-06T00:00:00+00:00".to_string(),
                sampling: "handwritten".to_string(),
                equation: "det(A)".to_string(),
                operations: vec!["LU decompose, 2×2".to_string()],
                reference: "numpy/LAPACK 2.1.3".to_string(),
            },
            module: "linalg".to_string(),
            case: "demo".to_string(),
            tolerances: Tolerances {
                f64: Tol {
                    abs: 1e-11,
                    rel: 1e-10,
                },
                f32: None,
            },
            inputs,
            expected,
        };

        let json = serde_json::to_string(&fx).unwrap();
        let back: Fixture = serde_json::from_str(&json).unwrap();

        let (r, c, data) = back.inputs["A"].as_matrix();
        assert_eq!((r, c), (2, 2));
        assert_eq!(data[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(data[2], f64::INFINITY);
        assert_eq!(back.expected["det"].as_scalar(), 2.5);
        assert_eq!(back.tolerances.f64.rel, 1e-10);
    }
}
