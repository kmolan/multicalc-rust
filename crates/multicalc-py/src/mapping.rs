use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap, ScanGeometry};
use pyo3::prelude::*;

use crate::errors;

/// Five-beam planar lidar geometry.
#[pyclass(name = "ScanGeometry5")]
pub struct PyScanGeometry5 {
    inner: ScanGeometry<5>,
}

#[pymethods]
impl PyScanGeometry5 {
    /// Field of view (radians) and maximum range.
    #[new]
    fn new(field_of_view: f64, maximum_range: f64) -> PyResult<Self> {
        Ok(Self {
            inner: ScanGeometry::<5>::try_new(field_of_view, maximum_range)
                .map_err(errors::mapping_error)?,
        })
    }

    /// Bearing of beam `index`. Raises `IndexError` if out of range.
    fn beam_angle(&self, index: usize) -> PyResult<f64> {
        self.inner.beam_angle(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("beam index {index} out of range"))
        })
    }

    fn __len__(&self) -> usize {
        self.inner.num_beams()
    }

    fn __repr__(&self) -> String {
        format!(
            "ScanGeometry5(beams={}, field_of_view={}, maximum_range={})",
            self.inner.num_beams(),
            self.inner.field_of_view(),
            self.inner.maximum_range()
        )
    }
}

/// Occupancy grid with a 2D origin and uniform resolution.
#[pyclass(name = "DynamicOccupancyGrid")]
pub struct PyDynamicOccupancyGrid {
    inner: DynamicOccupancyGrid,
}

#[pymethods]
impl PyDynamicOccupancyGrid {
    /// `columns`, `rows`, cell size, and world origin as a length-2 list.
    #[new]
    fn new(columns: usize, rows: usize, resolution: f64, origin: Vec<f64>) -> PyResult<Self> {
        if origin.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "origin must have 2 values",
            ));
        }
        Ok(Self {
            inner: DynamicOccupancyGrid::try_new(columns, rows, resolution, [origin[0], origin[1]])
                .map_err(errors::mapping_error)?,
        })
    }

    /// Mark the cell containing a 2D world point as occupied.
    fn occupy_point(&mut self, point: Vec<f64>) -> PyResult<()> {
        if point.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "point must have 2 values",
            ));
        }
        MutableOccupancyMap::occupy_point(&mut self.inner, [point[0], point[1]]);
        Ok(())
    }

    fn is_occupied(&self, row: usize, column: usize) -> bool {
        OccupancyMap::is_occupied(&self.inner, row, column)
    }

    /// Range to the first occupied cell along a ray, or `None`.
    fn cast_ray(&self, start: Vec<f64>, bearing: f64, maximum_range: f64) -> PyResult<Option<f64>> {
        if start.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "start must have 2 values",
            ));
        }
        Ok(OccupancyMap::cast_ray(
            &self.inner,
            [start[0], start[1]],
            bearing,
            maximum_range,
        ))
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyScanGeometry5>()?;
    module.add_class::<PyDynamicOccupancyGrid>()?;
    Ok(())
}
