use multicalc::error::{KinematicsError, SpatialError};

use crate::ModelFormat;

/// Everything that can stop a model from loading, in either format.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModelError {
    /// Malformed XML.
    Xml(String),
    /// File could not be read.
    FileRead(String),
    /// No `<worldbody>`.
    MissingWorldbody,
    /// No bodies.
    NoBodies,
    /// A body carried several `<joint>`/`<freejoint>` elements; multi-DoF joints are not composed.
    MultipleJoints {
        /// The body's name.
        body: String,
        /// How many joints were found.
        count: usize,
    },
    /// A 6-DoF joint on a non-root body. A tree takes one fixed or floating base, in slot 0.
    FreeJointNotAtRoot {
        /// The body's name.
        body: String,
    },
    /// Unsupported joint type.
    UnsupportedJoint {
        /// The body's name.
        body: String,
        /// The joint type as written in the file.
        joint_type: String,
    },
    /// Orientation stated other than by `quat`.
    UnsupportedOrientation {
        /// The element carrying the attribute.
        element: String,
        /// The attribute name.
        attribute: String,
    },
    /// A joint is limited, explicitly or by default, but states no `range`.
    LimitsNeedRange {
        /// The body's name.
        body: String,
    },
    /// No `<inertial>` and no geoms to derive it from.
    NoInertiaSource {
        /// The body's name.
        body: String,
    },
    /// A mass-bearing geom of a type this reader cannot integrate.
    UnsupportedGeomType {
        /// The body's name.
        body: String,
        /// The shape type as written in the file.
        geom_type: String,
    },
    /// A mass-bearing mesh geom; its inertia is not derivable from the file alone.
    MeshInertiaUnsupported {
        /// The body's name.
        body: String,
    },
    /// `fromto` on a geom type this reader does not read it for.
    UnsupportedFromTo {
        /// The body's name.
        body: String,
        /// The shape type as written in the file.
        geom_type: String,
    },
    /// A geom stated both `fromto` and `pos`, which need not agree.
    ConflictingPlacement {
        /// The body's name.
        body: String,
    },
    /// An attribute did not parse as the numbers it should hold.
    BadAttribute {
        /// The element the attribute was on.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The text that could not be read.
        value: String,
    },
    /// A geom named an undefined default class.
    UndefinedClass {
        /// The class name.
        name: String,
    },
    /// No body by that name.
    UnknownBody {
        /// The name asked for.
        name: String,
    },
    /// More bodies than the target tree's capacity.
    TreeCapacityExceeded {
        /// How many bodies the model has.
        needed: usize,
        /// How many the tree can hold.
        capacity: usize,
    },
    /// An `<include>` was found; resolving one needs a file path, not in-memory text.
    IncludeNeedsFile,
    /// `<include>` nesting exceeded the depth limit, or the includes form a cycle.
    IncludeTooDeep {
        /// How deep the chain of includes had reached.
        depth: usize,
    },
    /// No root link: every link is some joint's child.
    MissingRootLink,
    /// Several root links, which is not one robot.
    MultipleRootLinks {
        /// The parentless link names, sorted.
        names: Vec<String>,
    },
    /// A joint named an undeclared link.
    UnknownLink {
        /// The joint's name.
        joint: String,
        /// The link name given.
        link: String,
    },
    /// A link is the child of several joints.
    LinkHasTwoParents {
        /// The link's name.
        link: String,
        /// The claiming joint names, sorted.
        joints: Vec<String>,
    },
    /// A cycle in the link graph.
    CyclicLinkage {
        /// A link on the cycle.
        link: String,
    },
    /// A bounded joint stated no `<limit>` range.
    JointNeedsLimit {
        /// The joint's name.
        joint: String,
    },
    /// A mimic joint, which a constraint-free tree cannot express.
    MimicJointInTree {
        /// The coupled joint's name.
        joint: String,
        /// The driving joint's name.
        follows: String,
    },
    /// Unrecognised root element.
    UnexpectedRootElement {
        /// The root element found.
        found: String,
    },
    /// The file's format has no reader compiled into this build.
    FormatNotEnabled {
        /// The file's format.
        format: ModelFormat,
    },
    /// The stated mass properties do not describe a usable body.
    Inertia(SpatialError),
    /// The joint parameters do not describe a usable tree.
    Kinematics(KinematicsError),
}

impl From<SpatialError> for ModelError {
    fn from(err: SpatialError) -> Self {
        ModelError::Inertia(err)
    }
}

impl From<KinematicsError> for ModelError {
    fn from(err: KinematicsError) -> Self {
        ModelError::Kinematics(err)
    }
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ModelError::Xml(detail) => write!(f, "file is not well-formed XML: {detail}"),
            ModelError::FileRead(detail) => write!(f, "file could not be read: {detail}"),
            ModelError::MissingWorldbody => f.write_str("model has no worldbody"),
            ModelError::NoBodies => f.write_str("model has no bodies"),
            ModelError::MultipleJoints { body, count } => {
                write!(f, "body {body} carries {count} joints, and one is the limit here")
            }
            ModelError::FreeJointNotAtRoot { body } => write!(
                f,
                "body {body} hangs off the world by a free joint but is not at the top of the model"
            ),
            ModelError::UnsupportedJoint { body, joint_type } => write!(
                f,
                "body {body} has a {joint_type} joint, and only hinge or slide joints are handled"
            ),
            ModelError::UnsupportedOrientation { element, attribute } => write!(
                f,
                "the {attribute} attribute on {element} states a turn in a form this loader does not read; write it as a quaternion"
            ),
            ModelError::LimitsNeedRange { body } => write!(
                f,
                "the joint on body {body} is limited but states no range"
            ),
            ModelError::NoInertiaSource { body } => write!(
                f,
                "body {body} states no mass properties and has no shapes to work them out from"
            ),
            ModelError::UnsupportedGeomType { body, geom_type } => write!(
                f,
                "body {body} carries mass on a {geom_type} shape, which this loader cannot measure"
            ),
            ModelError::MeshInertiaUnsupported { body } => write!(
                f,
                "body {body} carries mass on a mesh, whose mass cannot be worked out from the file alone"
            ),
            ModelError::UnsupportedFromTo { body, geom_type } => write!(
                f,
                "body {body} gives the ends of a {geom_type} shape's axis, which this loader reads only for capsules and cylinders"
            ),
            ModelError::ConflictingPlacement { body } => write!(
                f,
                "a shape on body {body} gives both the ends of its axis and a position, and they need not agree"
            ),
            ModelError::BadAttribute {
                element,
                attribute,
                value,
            } => write!(
                f,
                "the {attribute} attribute on {element} could not be read as numbers: {value}"
            ),
            ModelError::UndefinedClass { name } => {
                write!(
                    f,
                    "a shape names class {name}, which the file never defines"
                )
            }
            ModelError::UnknownBody { name } => write!(f, "the model has no body called {name}"),
            ModelError::TreeCapacityExceeded { needed, capacity } => write!(
                f,
                "the model has {needed} bodies and the model being built holds {capacity}"
            ),
            ModelError::IncludeNeedsFile => f.write_str(
                "the model pulls in another file, which can only be followed when the model is read from a file itself",
            ),
            ModelError::IncludeTooDeep { depth } => {
                write!(f, "files pull in other files more than {depth} deep, or pull in each other")
            }
            ModelError::MissingRootLink => {
                f.write_str("every link in the model hangs off another one, so the model has no top")
            }
            ModelError::MultipleRootLinks { names } => write!(
                f,
                "links {} all sit at the top of the model, and one is the limit here",
                names.join(", ")
            ),
            ModelError::UnknownLink { joint, link } => write!(
                f,
                "joint {joint} names link {link}, which the model does not have"
            ),
            ModelError::LinkHasTwoParents { link, joints } => write!(
                f,
                "link {link} hangs off joints {}, and one is the limit here",
                joints.join(", ")
            ),
            ModelError::CyclicLinkage { link } => {
                write!(f, "link {link} hangs off itself, by a loop of joints")
            }
            ModelError::JointNeedsLimit { joint } => {
                write!(f, "joint {joint} can travel but states no range")
            }
            ModelError::MimicJointInTree { joint, follows } => write!(
                f,
                "joint {joint} follows joint {follows}, which a tree of joints that each move on their own cannot describe"
            ),
            ModelError::UnexpectedRootElement { found } => write!(
                f,
                "the document starts with {found}, and only mujoco or robot are read"
            ),
            ModelError::FormatNotEnabled { format } => write!(
                f,
                "the file is {format} and this build was compiled without that reader"
            ),
            ModelError::Inertia(err) => write!(f, "{err}"),
            ModelError::Kinematics(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Inertia(err) => Some(err),
            ModelError::Kinematics(err) => Some(err),
            _ => None,
        }
    }
}
