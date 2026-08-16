//! `<compiler>` settings.

use roxmltree::Node;

use crate::ModelError;
use crate::xml::{bad_attribute, element};

/// File-wide settings governing how the rest of the document parses.
pub(crate) struct CompilerSettings {
    /// Angles are in degrees rather than radians.
    pub angle_in_degrees: bool,
    /// A stated `range` bounds the joint without an explicit `limited`.
    pub auto_limits: bool,
    /// When inertia is derived from geoms rather than taken from `<inertial>`.
    pub inertia_from_geom: InertiaFromGeom,
}

/// MJCF `inertiafromgeom`.
pub(crate) enum InertiaFromGeom {
    /// `false`: `<inertial>` is required.
    Never,
    /// `auto`: `<inertial>` is used where present, otherwise computed from geoms.
    Auto,
    /// `true`: always computed from geoms, `<inertial>` is ignored.
    Always,
}

impl CompilerSettings {
    /// The first `<compiler>` child of `root`, with MJCF's defaults (`angle="degree"`,
    /// `autolimits="true"`, `inertiafromgeom="auto"`) for absent attributes.
    pub(crate) fn read(root: Node) -> Result<Self, ModelError> {
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

    /// An angle in the file's units, in radians.
    pub(crate) fn to_radians(&self, value: f64) -> f64 {
        if self.angle_in_degrees {
            value * std::f64::consts::PI / 180.0
        } else {
            value
        }
    }
}
