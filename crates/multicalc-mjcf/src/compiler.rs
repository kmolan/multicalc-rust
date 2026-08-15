//! File-wide settings read from the `<compiler>` element.

use multicalc::linear_algebra::{Vector, Vector3D};
use roxmltree::Node;

use crate::MjcfError;
use crate::defaults::{bad_attribute, element};

/// The order MJCF turns a `euler` about when the file names none.
const ASSUMED_EULER_SEQUENCE: &str = "xyz";

/// File-wide settings that change how the rest of the document is read.
pub(crate) struct CompilerSettings {
    /// Whether angles in the file are in degrees rather than radians.
    pub angle_in_degrees: bool,
    /// Whether a joint stating a range is limited by it without an explicit `limited`.
    pub auto_limits: bool,
    /// When a body's inertia is computed from its geoms rather than an explicit `<inertial>`.
    pub inertia_from_geom: InertiaFromGeom,
    /// The three turns a `euler` attribute makes, in the order it makes them.
    pub euler_sequence: [EulerStep; 3],
}

/// One turn in a `euler` sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EulerStep {
    /// Which coordinate axis the turn is about: 0 for x, 1 for y, 2 for z.
    axis: usize,
    /// Whether the axis rides along with the turns already made, rather than standing still in the
    /// frame the element is placed in. The letter's case is what says so: `x` rides along, `X`
    /// stands still.
    carried_along: bool,
}

impl EulerStep {
    /// The axis this turn is about.
    pub(crate) fn direction(self) -> Vector3D {
        let mut axis = [0.0; 3];
        axis[self.axis] = 1.0;
        Vector::new(axis)
    }

    /// Whether the earlier turns in the sequence carried this one's axis with them.
    #[must_use]
    pub(crate) fn carried_along(self) -> bool {
        self.carried_along
    }
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
    /// (`angle="degree"`, `autolimits="true"`, `inertiafromgeom="auto"`, `eulerseq="xyz"`) where an
    /// attribute is absent.
    pub(crate) fn read(root: Node) -> Result<Self, MjcfError> {
        let Some(compiler) = element(root, "compiler") else {
            return Ok(CompilerSettings {
                angle_in_degrees: true,
                auto_limits: true,
                inertia_from_geom: InertiaFromGeom::Auto,
                euler_sequence: assumed_euler_sequence(),
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

        let euler_sequence = euler_sequence(
            compiler,
            compiler
                .attribute("eulerseq")
                .unwrap_or(ASSUMED_EULER_SEQUENCE),
        )?;

        Ok(CompilerSettings {
            angle_in_degrees,
            auto_limits,
            inertia_from_geom,
            euler_sequence,
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

/// The order a file that names none turns a `euler` about.
#[must_use]
fn assumed_euler_sequence() -> [EulerStep; 3] {
    [0, 1, 2].map(|axis| EulerStep {
        axis,
        carried_along: true,
    })
}

/// The three turns an `eulerseq` names, one per letter: which axis each is about, and whether the
/// turns before it carried that axis along. Anything but three letters from `xyzXYZ` names no
/// sequence — repeats are allowed, and a mixture of cases is too.
fn euler_sequence(node: Node, text: &str) -> Result<[EulerStep; 3], MjcfError> {
    let letters: Vec<char> = text.chars().collect();
    let [first, second, third] =
        <[char; 3]>::try_from(letters).map_err(|_| bad_attribute(node, "eulerseq", text))?;

    let mut steps = [EulerStep {
        axis: 0,
        carried_along: true,
    }; 3];
    for (step, letter) in steps.iter_mut().zip([first, second, third]) {
        let axis = match letter.to_ascii_lowercase() {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            _ => return Err(bad_attribute(node, "eulerseq", text)),
        };
        *step = EulerStep {
            axis,
            carried_along: letter.is_ascii_lowercase(),
        };
    }
    Ok(steps)
}
