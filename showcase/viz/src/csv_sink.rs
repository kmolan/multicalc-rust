//! CSV backend: the scalar time-series subset, for the `plot.py` fallback.
//!
//! Rows are keyed by the current sequence value; columns are the scalar paths in first-seen
//! order, with blanks where a row is missing one. Point and tensor logs are skipped with a
//! warning — a dense grid is not a natural CSV time-series.

use crate::sink::{VizError, VizSink};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Buffers `(sequence, columns)` rows and writes a CSV on `flush()`.
pub struct CsvSink {
    path: PathBuf,
    seq_name: String,
    seq: i64,
    columns: Vec<String>,
    rows: Vec<(i64, BTreeMap<String, f64>)>,
}

impl CsvSink {
    /// Creates a sink that will write to `path` on flush.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, VizError> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            seq_name: "seq".to_owned(),
            seq: 0,
            columns: Vec::new(),
            rows: Vec::new(),
        })
    }

    /// Returns the row for the current sequence, creating it if needed.
    fn current_row(&mut self) -> &mut BTreeMap<String, f64> {
        let need_new = match self.rows.last() {
            Some((seq, _)) => *seq != self.seq,
            None => true,
        };
        if need_new {
            self.rows.push((self.seq, BTreeMap::new()));
        }
        let last = self.rows.len() - 1;
        &mut self.rows[last].1
    }
}

impl VizSink for CsvSink {
    fn set_sequence(&mut self, timeline: &str, seq: i64) {
        self.seq_name = timeline.to_owned();
        self.seq = seq;
    }

    fn scalar(&mut self, path: &str, value: f64) -> Result<(), VizError> {
        if !self.columns.iter().any(|c| c == path) {
            self.columns.push(path.to_owned());
        }
        self.current_row().insert(path.to_owned(), value);
        Ok(())
    }

    fn points2d(&mut self, path: &str, _xy: &[[f64; 2]]) -> Result<(), VizError> {
        eprintln!("CsvSink: points2d('{path}') skipped (CSV is scalar-series only)");
        Ok(())
    }

    fn points3d(&mut self, path: &str, _xyz: &[[f64; 3]]) -> Result<(), VizError> {
        eprintln!("CsvSink: points3d('{path}') skipped (CSV is scalar-series only)");
        Ok(())
    }

    fn tensor(&mut self, path: &str, _shape: [usize; 2], _data: &[f64]) -> Result<(), VizError> {
        eprintln!("CsvSink: tensor('{path}') skipped (CSV is scalar-series only)");
        Ok(())
    }

    fn flush(&mut self) -> Result<(), VizError> {
        let mut w = BufWriter::new(File::create(&self.path)?);
        write!(w, "{}", self.seq_name)?;
        for c in &self.columns {
            write!(w, ",{c}")?;
        }
        writeln!(w)?;
        for (seq, row) in &self.rows {
            write!(w, "{seq}")?;
            for c in &self.columns {
                match row.get(c) {
                    Some(v) => write!(w, ",{v}")?,
                    None => write!(w, ",")?,
                }
            }
            writeln!(w)?;
        }
        w.flush()?;
        Ok(())
    }
}
