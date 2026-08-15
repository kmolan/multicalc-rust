use multicalc::error::{KinematicsError, SpatialError};

/// Everything that can stop a model from loading, whichever format it came from.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModelError {
    /// The file was not well-formed XML.
    Xml(String),
    /// The file could not be read.
    Io(String),
    /// The model had no `<worldbody>`.
    MissingWorldbody,
    /// The model had no bodies.
    NoBodies,
    /// A body carried more than one `<joint>`/`<freejoint>`, which this loader does not compose
    /// into a multi-DOF joint.
    MultipleJoints {
        /// The body's name.
        body: String,
        /// How many joints were found.
        count: usize,
    },
    /// A free joint sat on a body other than the root, which cannot be represented in a tree with
    /// a single fixed or floating base.
    FreeJointNotAtRoot {
        /// The body's name.
        body: String,
    },
    /// The body carried a joint kind other than hinge or slide.
    UnsupportedJoint {
        /// The body's name.
        body: String,
        /// The joint type as written in the file.
        joint_type: String,
    },
    /// An element gave its orientation in a form other than `quat`.
    UnsupportedOrientation {
        /// The element carrying the attribute.
        element: String,
        /// The attribute name.
        attribute: String,
    },
    /// A joint was marked limited, or defaults to limited, but states no `range`.
    LimitsNeedRange {
        /// The body's name.
        body: String,
    },
    /// The body stated no mass properties and had no shapes to work them out from.
    NoInertiaSource {
        /// The body's name.
        body: String,
    },
    /// A shape carrying mass had a form this loader cannot measure.
    UnsupportedGeomType {
        /// The body's name.
        body: String,
        /// The shape type as written in the file.
        geom_type: String,
    },
    /// A shape carrying mass was a mesh, whose mass cannot be worked out from the file alone.
    MeshInertiaUnsupported {
        /// The body's name.
        body: String,
    },
    /// A shape gave the two ends of its axis, on a form this loader does not read them for.
    UnsupportedFromTo {
        /// The body's name.
        body: String,
        /// The shape type as written in the file.
        geom_type: String,
    },
    /// A shape gave both the two ends of its axis and a position, which say different things
    /// about where it sits.
    ConflictingPlacement {
        /// The body's name.
        body: String,
    },
    /// An attribute could not be read as the numbers it should hold.
    BadAttribute {
        /// The element the attribute was on.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The text that could not be read.
        value: String,
    },
    /// A shape named a class the file never defines.
    UndefinedClass {
        /// The class name.
        name: String,
    },
    /// The model has no body by the name asked for.
    UnknownBody {
        /// The name asked for.
        name: String,
    },
    /// The model has more bodies than the tree being built can hold.
    TreeCapacityExceeded {
        /// How many bodies the model has.
        needed: usize,
        /// How many the tree can hold.
        capacity: usize,
    },
    /// The model pulls in another file, which can only be followed when the model is read from a
    /// file rather than text already in memory.
    IncludeNeedsFile,
    /// `<include>` files pull in each other, or nest deeper than this loader follows.
    IncludeTooDeep {
        /// How deep the chain of includes had reached.
        depth: usize,
    },
    /// The mass properties read from the file did not describe a usable body.
    Inertia(SpatialError),
    /// A joint's own numbers did not describe a usable joint once built into a tree.
    Kinematics(KinematicsError),
}

impl From<SpatialError> for ModelError {
    fn from(e: SpatialError) -> Self {
        ModelError::Inertia(e)
    }
}

impl From<KinematicsError> for ModelError {
    fn from(e: KinematicsError) -> Self {
        ModelError::Kinematics(e)
    }
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ModelError::Xml(detail) => write!(f, "file is not well-formed XML: {detail}"),
            ModelError::Io(detail) => write!(f, "file could not be read: {detail}"),
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
            ModelError::Inertia(e) => write!(f, "{e}"),
            ModelError::Kinematics(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Inertia(e) => Some(e),
            ModelError::Kinematics(e) => Some(e),
            _ => None,
        }
    }
}
