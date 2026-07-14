//! Fixture schema: hex-encoded floats, tagged values, tolerances, and metadata.
//!
//! Floating-point numbers are stored as their IEEE-754 bit patterns in hex so a
//! fixture round-trips through JSON with no loss. Serialization is byte-for-byte
//! stable, which lets a regenerated fixture be diffed against the committed one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Current fixture format. A file with a different version is rejected on load.
pub const SCHEMA_VERSION: u32 = 1;

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

/// An `f32` stored as its 32-bit pattern, written as `"0x{:08x}"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32(pub u32);

impl F32 {
    pub fn from_f32(x: f32) -> Self {
        F32(x.to_bits())
    }
    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl Serialize for F32 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{:08x}", self.0))
    }
}

impl<'de> Deserialize<'de> for F32 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let hex = s.strip_prefix("0x").unwrap_or(&s);
        u32::from_str_radix(hex, 16)
            .map(F32)
            .map_err(serde::de::Error::custom)
    }
}

/// One block of a manifold state (reserved for later phases; unused in v0.7).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifoldBlock {
    pub kind: String,
    pub data: Vec<F64>,
}

/// A self-describing fixture value. Floats are hex; ints, strings, and bools are
/// plain JSON. `Quaternion` and `ManifoldState` are reserved for later phases and
/// carry no data in v0.7.
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
    Quaternion {
        w: F64,
        x: F64,
        y: F64,
        z: F64,
    },
    ManifoldState {
        nq: usize,
        nv: usize,
        blocks: Vec<ManifoldBlock>,
    },
    Int {
        v: i64,
    },
    Str {
        v: String,
    },
    Bool {
        v: bool,
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

/// Absolute and relative thresholds for one comparison.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Tol {
    pub abs: f64,
    pub rel: f64,
}

impl From<Tol> for multicalc_testkit::tol::Tol {
    fn from(t: Tol) -> Self {
        multicalc_testkit::tol::Tol {
            abs: t.abs,
            rel: t.rel,
        }
    }
}

/// Per-`<scalar>/<target>` tolerance table, e.g. `"f64/host"` or `"f32/host"`.
/// Reserved targets: `host`, `aarch64`, `thumbv7em-eabi`, `thumbv7em-eabihf`,
/// `thumbv6m`. v0.7 populates only the `host` entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tolerances {
    pub table: BTreeMap<String, Tol>,
}

impl Tolerances {
    /// Looks up `<scalar>/<target>`, falls back to `<scalar>/default`, then to a
    /// zero tolerance (exact match required).
    pub fn get(&self, scalar: &str, target: &str) -> Tol {
        *self
            .table
            .get(&format!("{scalar}/{target}"))
            .or_else(|| self.table.get(&format!("{scalar}/default")))
            .unwrap_or(&Tol { abs: 0.0, rel: 0.0 })
    }
}

/// Provenance for a fixture: the generator, exact library versions, seed, date,
/// and a note on how inputs were sampled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub generator: String,
    pub libraries: BTreeMap<String, String>,
    pub seed: u64,
    pub date: String,
    pub sampling: String,
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

        let mut table = BTreeMap::new();
        table.insert(
            "f64/host".to_string(),
            Tol {
                abs: 1e-11,
                rel: 1e-10,
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
            },
            module: "linalg".to_string(),
            case: "demo".to_string(),
            tolerances: Tolerances { table },
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
        assert_eq!(back.tolerances.get("f64", "host").rel, 1e-10);
    }
}
