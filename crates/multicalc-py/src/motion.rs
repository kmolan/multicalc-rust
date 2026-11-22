use multicalc::motion::PolylinePath;
use pyo3::prelude::*;

use crate::convert::vector_from_list;
use crate::convert::vector_to_list;
use crate::errors;

/// Planar polyline with at most 8 waypoints.
#[pyclass(name = "PolylinePath8x2")]
pub struct PyPolylinePath8x2 {
    inner: PolylinePath<8, 2>,
}

#[pymethods]
impl PyPolylinePath8x2 {
    /// Build from 2D waypoints.
    #[staticmethod]
    fn try_from_points(points: Vec<Vec<f64>>) -> PyResult<Self> {
        let mut waypoints = Vec::with_capacity(points.len());
        for point in points {
            waypoints.push(vector_from_list::<2>(point)?);
        }
        Ok(Self {
            inner: PolylinePath::<8, 2>::try_from_points(&waypoints)
                .map_err(errors::motion_error)?,
        })
    }

    fn total_arc_length(&self) -> f64 {
        self.inner.total_arc_length()
    }

    /// Point a lookahead distance along the path from `from_arc_length`.
    fn lookahead_point(&self, from_arc_length: f64, lookahead: f64) -> PyResult<[f64; 2]> {
        Ok(self
            .inner
            .lookahead_point(from_arc_length, lookahead)
            .map_err(errors::motion_error)?
            .into_array())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let points: Vec<Vec<f64>> = self
            .inner
            .waypoints()
            .iter()
            .map(|point| vector_to_list(*point))
            .collect();
        format!("PolylinePath8x2({points:?})")
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyPolylinePath8x2>()?;
    Ok(())
}
