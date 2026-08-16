//! The five ways a file can say which way an element faces, and the one turn they all come to.
//!
//! A turn is three numbers' worth of freedom however it is written, so `quat`, `euler`,
//! `axisangle`, `xyaxes` and `zaxis` are five spellings of one thing rather than five things. Each
//! is the natural one for a different source — a quaternion for whatever another program wrote,
//! angles for whoever typed the file, an axis and a turn about it for a part mounted on a bolt, two
//! axes for a face taken off a CAD model, one axis for a shape with only one direction worth naming
//! — and each is read here into the quaternion the rest of the loader works in.
//!
//! MuJoCo holds the plain quaternion in one slot and the other four in a second, and refuses an
//! element that fills both at once. Only a default block can leave both filled, the block supplying
//! one and the element the other, and there the one that is not the plain quaternion wins. That is
//! why the two are carried separately rather than as one choice of five.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::spatial::Quaternion;
use roxmltree::Node;

use crate::ModelError;
use crate::mjcf::compiler::CompilerSettings;
use crate::xml::{bad_attribute, parse_vector3, parse_vector4, parse_vector6};

/// How far off the line a `zaxis` has to point before the two directions still name an axis to turn
/// about. Below this MuJoCo stops asking them and answers with a settled frame, and a model read
/// here has to land where MuJoCo puts it.
const ZAXIS_DEGENERATE: f64 = 1e-7;

/// A turn written any way other than as a plain quaternion. The numbers are held as the file wrote
/// them: what they mean waits on the `<compiler>`, which says what units the angles are in and
/// which axes a `euler` turns about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Alternative {
    /// `axisangle`: a turn of the fourth number about the direction the first three point.
    AxisAngle([f64; 4]),
    /// `euler`: three turns about coordinate axes, in the order `<compiler eulerseq>` gives.
    Euler([f64; 3]),
    /// `xyaxes`: where the element's own x and y axes point, in its parent's axes.
    XyAxes([f64; 6]),
    /// `zaxis`: where the element's own z axis points. One axis leaves the turn about it free, and
    /// MuJoCo settles that by spending none of it.
    ZAxis([f64; 3]),
}

/// Which way an element faces, as written. Empty means the element said nothing, and so faces the
/// same way the one it hangs off does.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Orientation {
    quat: Option<[f64; 4]>,
    alternative: Option<Alternative>,
}

impl Orientation {
    /// Reads whichever of the five forms an element carries.
    ///
    /// Two of them on one element are two answers to a single question, and MuJoCo refuses the pair
    /// rather than choosing between them.
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

    /// A copy of this with everything `other` states written over the top. The two slots fill
    /// separately, so a block's `euler` outlives an element's own `quat` — and then beats it, which
    /// is the only way the two ever come to sit together.
    #[must_use]
    pub(crate) fn overridden_by(self, other: Self) -> Self {
        Orientation {
            quat: other.quat.or(self.quat),
            alternative: other.alternative.or(self.alternative),
        }
    }

    /// Whether the element said anything at all about which way it faces.
    #[must_use]
    pub(crate) fn is_stated(self) -> bool {
        self.quat.is_some() || self.alternative.is_some()
    }

    /// The turn this describes. `node` is only where an error points when the numbers describe no
    /// turn, which for one a default block supplied is the element that inherited it.
    pub(crate) fn resolve(
        self,
        node: Node,
        compiler: &CompilerSettings,
    ) -> Result<Quaternion<f64>, ModelError> {
        match self.alternative {
            Some(alternative) => alternative.resolve(node, compiler),
            // MJCF writes the scalar part first, which is also how the crate stores it, so the four
            // numbers carry straight over.
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
    /// The turn this form describes, once the compiler settings have said what its numbers mean.
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

            // Each angle turns about one coordinate axis, and the letter naming that axis also said
            // which frame it belongs to. A lower-case one is an axis the turns before it have
            // already carried along, so it joins on the right; an upper-case one stands still in
            // the parent's frame, so it joins on the left.
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

            // Two axes are three numbers more than a turn needs, so the second is taken only for
            // the part of it standing square to the first, and the third axis follows from the two.
            // Naming the same line twice leaves nothing of the second to take, which is the one way
            // this fails beyond a direction that points nowhere.
            Alternative::XyAxes([x1, y1, z1, x2, y2, z2]) => {
                let x = direction(node, "xyaxes", [x1, y1, z1])?;
                let stated = Vector::new([x2, y2, z2]);
                let y = (stated - x.scale(stated.dot(x)))
                    .try_normalized()
                    .ok_or_else(|| unreadable(node, "xyaxes"))?;
                let z = x.cross(y);

                // The three axes are where the element's own axes point, so they are the columns of
                // the turn that carries a direction from the element's frame into its parent's.
                let axes =
                    Matrix::from([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]);
                Quaternion::try_from_rotation_matrix(axes).ok_or_else(|| unreadable(node, "xyaxes"))
            }

            // One axis is three numbers short, and what is missing is the turn about that axis.
            // MuJoCo spends none of it: the turn is about the line square to both directions,
            // through the angle between them, and nothing else moves.
            Alternative::ZAxis(stated) => {
                let target = direction(node, "zaxis", stated)?;
                let along = Vector::new([0.0, 0.0, 1.0]);
                let square_to_both = along.cross(target);
                let sine = square_to_both.norm();

                // Near enough to the line and the cross product has stopped naming a direction.
                // Pointing the same way there is no turn to make; pointing the other way every axis
                // square to the line serves equally well and MuJoCo always takes x. Which one it
                // takes is not idle: the z axis lands the same either way, but everything hanging
                // off the element rides the leftover turn about it.
                //
                // This is why `Quaternion::from_two_vectors` is not reached for here. It answers
                // the same question everywhere but at the far end of the line, where it takes the
                // principal axis the direction leans on least — y, for a z axis turned right over —
                // and a model written that way would load a half turn away from where MuJoCo has it.
                if sine < ZAXIS_DEGENERATE {
                    return Ok(if target[2] > 0.0 {
                        Quaternion::identity()
                    } else {
                        Quaternion::from_axis_angle(Vector::new([1.0, 0.0, 0.0]), PI)
                    });
                }

                // The angle comes from both the sine and the cosine rather than the cosine alone:
                // an `acos` of a cosine near one is the arc read off a curve that has gone flat, and
                // loses half its digits there, which for a `zaxis` a whisker off straight up is the
                // common case rather than the corner one.
                Ok(Quaternion::from_axis_angle(
                    square_to_both.scale(1.0 / sine),
                    sine.atan2(target[2]),
                ))
            }
        }
    }
}

/// A direction the file gave, as a unit vector. Three numbers that come to nothing point nowhere,
/// and no turn can be read out of them.
fn direction(
    node: Node,
    attribute: &'static str,
    stated: [f64; 3],
) -> Result<Vector3D, ModelError> {
    Vector::new(stated)
        .try_normalized()
        .ok_or_else(|| unreadable(node, attribute))
}

/// The error for an orientation attribute whose numbers describe no turn.
#[must_use]
fn unreadable(node: Node, attribute: &'static str) -> ModelError {
    bad_attribute(
        node,
        attribute,
        node.attribute(attribute).unwrap_or_default(),
    )
}
