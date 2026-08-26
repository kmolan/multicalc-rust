//! The five orientation forms MJCF allows, resolved to one quaternion.
//!
//! `quat`, `euler`, `axisangle`, `xyaxes` and `zaxis` state the same three degrees of freedom.
//! MuJoCo holds `quat` in one slot and the other four in a second, and refuses an element filling
//! both. Only a default block can leave both filled, the block supplying one and the element the
//! other, and there the form that is not `quat` wins, so the two are carried separately.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::spatial::Quaternion;
use roxmltree::Node;

use crate::ModelError;
use crate::mjcf::compiler::CompilerSettings;
use crate::xml::{bad_attribute, parse_vector3, parse_vector4, parse_vector6};

/// `sin θ` below which `z × target` no longer names an axis. MuJoCo settles the frame there, and a
/// model read here has to land where MuJoCo puts it.
const ZAXIS_DEGENERATE: f64 = 1e-7;

/// A turn stated as anything but a plain quaternion. Numbers are held as written: what they mean
/// waits on `<compiler>`, which gives the angle units and the euler sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Alternative {
    /// `axisangle`: `[x, y, z, angle]`.
    AxisAngle([f64; 4]),
    /// `euler`: three turns about coordinate axes, in `<compiler eulerseq>` order.
    Euler([f64; 3]),
    /// `xyaxes`: the element's own x and y axes, in parent axes.
    XyAxes([f64; 6]),
    /// `zaxis`: the element's own z axis. The turn about it is free, and MuJoCo spends none of it.
    ZAxis([f64; 3]),
}

/// Which way an element faces, as written. Empty means it faces as its parent does.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Orientation {
    quat: Option<[f64; 4]>,
    alternative: Option<Alternative>,
}

impl Orientation {
    /// Reads whichever of the five forms an element carries.
    ///
    /// Two forms on one element are two answers to one question:
    /// [`MultipleOrientations`](ModelError::MultipleOrientations), as MuJoCo does.
    pub(crate) fn read(node: Node) -> Result<Self, ModelError> {
        let quat = parse_vector4(node, "quat")?;
        let alternatives = [
            parse_vector4(node, "axisangle")?.map(Alternative::AxisAngle),
            parse_vector3(node, "euler")?.map(Alternative::Euler),
            parse_vector6(node, "xyaxes")?.map(Alternative::XyAxes),
            parse_vector3(node, "zaxis")?.map(Alternative::ZAxis),
        ];

        let stated =
            usize::from(quat.is_some()) + alternatives.iter().filter(|form| form.is_some()).count();
        if stated > 1 {
            return Err(ModelError::MultipleOrientations {
                element: node.tag_name().name().to_owned(),
            });
        }

        Ok(Orientation {
            quat,
            alternative: alternatives.into_iter().flatten().next(),
        })
    }

    /// This with everything `other` states over the top. The two slots fill separately, so a block's
    /// `euler` survives an element's own `quat` and then beats it.
    #[must_use]
    pub(crate) fn overridden_by(self, other: Self) -> Self {
        Orientation {
            quat: other.quat.or(self.quat),
            alternative: other.alternative.or(self.alternative),
        }
    }

    /// Whether the element stated any form.
    #[must_use]
    pub(crate) fn is_stated(self) -> bool {
        self.quat.is_some() || self.alternative.is_some()
    }

    /// The turn this states. `node` only locates an error, which for a form a default block supplied
    /// is the element that inherited it.
    pub(crate) fn resolve(
        self,
        node: Node,
        compiler: &CompilerSettings,
    ) -> Result<Quaternion<f64>, ModelError> {
        match self.alternative {
            Some(alternative) => alternative.resolve(node, compiler),
            // MJCF writes the scalar part first, as the crate stores it.
            None => match self.quat {
                Some([w, x, y, z]) => Quaternion::new(w, x, y, z)
                    .try_normalized()
                    .ok_or_else(|| unreadable(node, "quat")),
                None => Ok(Quaternion::identity()),
            },
        }
    }
}

impl Alternative {
    /// The turn this form states, with `<compiler>` supplying angle units and euler order.
    fn resolve(
        self,
        node: Node,
        compiler: &CompilerSettings,
    ) -> Result<Quaternion<f64>, ModelError> {
        match self {
            Alternative::AxisAngle([x, y, z, angle]) => {
                let axis = direction(node, "axisangle", [x, y, z])?;
                Ok(Quaternion::from_axis_angle(
                    axis,
                    compiler.to_radians(angle),
                ))
            }

            // The letter's case names the frame its axis stands in. Lower case rides the turns
            // already made, so it composes on the right; upper case stands still in the parent
            // frame, so it composes on the left.
            Alternative::Euler(angles) => {
                let mut turn = Quaternion::identity();
                for (angle, step) in angles.into_iter().zip(compiler.euler_sequence) {
                    let about =
                        Quaternion::from_axis_angle(step.direction(), compiler.to_radians(angle));
                    turn = if step.carried_along() {
                        turn * about
                    } else {
                        about * turn
                    };
                }
                Ok(turn)
            }

            // Gram-Schmidt: y keeps only the part of the second vector square to x, and z = x × y.
            // Two axes on one line leave nothing of y, the one failure beyond a zero direction.
            Alternative::XyAxes(stated) => {
                let x = direction(node, "xyaxes", [stated[0], stated[1], stated[2]])?;
                let toward_y = Vector::new([stated[3], stated[4], stated[5]]);
                let y = (toward_y - x.scale(toward_y.dot(x)))
                    .try_normalized()
                    .ok_or_else(|| unreadable(node, "xyaxes"))?;
                let z = x.cross(y);

                // The stated axes are the element's own, so they are the columns of the
                // element-to-parent rotation.
                let axes =
                    Matrix::from([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]);
                Quaternion::try_from_rotation_matrix(axes).ok_or_else(|| unreadable(node, "xyaxes"))
            }

            // One axis leaves the turn about it free. MuJoCo spends none of it: turn about
            // `z × target`, through the angle between them.
            Alternative::ZAxis(stated) => {
                let target = direction(node, "zaxis", stated)?;
                let along = Vector::new([0.0, 0.0, 1.0]);
                let square_to_both = along.cross(target);
                let sine = square_to_both.norm();

                // `z × target` names no direction here. Aligned, there is no turn; anti-aligned,
                // every square axis serves and MuJoCo always takes x. The choice is not idle: z
                // lands the same either way, but everything below the element rides the leftover
                // turn about it. `Quaternion::from_two_vectors` takes the principal axis the
                // direction leans on least, y for a flipped z, so a model written that way would
                // load a half turn from where MuJoCo has it.
                if sine < ZAXIS_DEGENERATE {
                    return Ok(if target[2] > 0.0 {
                        Quaternion::identity()
                    } else {
                        Quaternion::from_axis_angle(Vector::new([1.0, 0.0, 0.0]), PI)
                    });
                }

                // `atan2(sine, cosine)`, not `acos(cosine)`: a `zaxis` a whisker off straight up is
                // the common case, and `acos` loses half its digits there.
                Ok(Quaternion::from_axis_angle(
                    square_to_both.scale(1.0 / sine),
                    sine.atan2(target[2]),
                ))
            }
        }
    }
}

/// A stated direction as a unit vector. Three numbers summing to nothing point nowhere.
fn direction(
    node: Node,
    attribute: &'static str,
    stated: [f64; 3],
) -> Result<Vector3D, ModelError> {
    Vector::new(stated)
        .try_normalized()
        .ok_or_else(|| unreadable(node, attribute))
}

/// Error for an orientation attribute whose numbers describe no turn.
#[must_use]
fn unreadable(node: Node, attribute: &'static str) -> ModelError {
    bad_attribute(
        node,
        attribute,
        node.attribute(attribute).unwrap_or_default(),
    )
}
