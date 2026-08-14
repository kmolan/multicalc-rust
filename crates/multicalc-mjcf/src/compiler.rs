//! File-wide settings read from the `<compiler>` element.

use roxmltree::Node;

use crate::MjcfError;
use crate::defaults::{bad_attribute, element};

/// File-wide settings that change how the rest of the document is read.
pub(crate) struct CompilerSettings {
    /// Whether angles in the file are in degrees rather than radians.
    pub angle_in_degrees: bool,
    /// Whether a joint stating a range is limited by it without an explicit `limited`.
    pub auto_limits: bool,
    /// When a body's inertia is computed from its geoms rather than an explicit `<inertial>`.
    pub inertia_from_geom: InertiaFromGeom,
}

/// MJCF `inertiafromgeom`: when geom-derived inertia overrides or fills in `<inertial>`.
pub(crate) enum InertiaFromGeom {
    /// `false`: `<inertial>` is required.
    Never,
    /// `auto`: `<inertial>` is used where present, otherwise computed from geoms.
    Auto,
    /// `true`: always computed from geoms, `<inertial>` is ignored.
    Always,
}

impl CompilerSettings {
    /// Reads the first `<compiler>` child of `root`, applying MJCF's defaults
    /// (`angle="degree"`, `autolimits="true"`, `inertiafromgeom="auto"`) where an attribute is
    /// absent.
    pub(crate) fn read(root: Node) -> Result<Self, MjcfError> {
        let Some(compiler) = element(root, "compiler") else {
            return Ok(CompilerSettings {
                angle_in_degrees: true,
                auto_limits: true,
                inertia_from_geom: InertiaFromGeom::Auto,
            });
        };

        let angle_in_degrees = match compiler.attribute("angle").unwrap_or("degree") {
            "degree" => true,
            "radian" => false,
            value => return Err(bad_attribute(compiler, "angle", value)),
        };
        let auto_limits = match compiler.attribute("autolimits").unwrap_or("true") {
            "true" => true,
            "false" => false,
            value => return Err(bad_attribute(compiler, "autolimits", value)),
        };
        let inertia_from_geom = match compiler.attribute("inertiafromgeom").unwrap_or("auto") {
            "false" => InertiaFromGeom::Never,
            "auto" => InertiaFromGeom::Auto,
            "true" => InertiaFromGeom::Always,
            value => return Err(bad_attribute(compiler, "inertiafromgeom", value)),
        };

        Ok(CompilerSettings {
            angle_in_degrees,
            auto_limits,
            inertia_from_geom,
        })
    }

    /// Converts an angle from the file's units to radians.
    pub(crate) fn to_radians(&self, value: f64) -> f64 {
        if self.angle_in_degrees {
            value * std::f64::consts::PI / 180.0
        } else {
            value
        }
    }
}
