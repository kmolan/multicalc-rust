//! Writes a `RobotModel` out as Rust source: one function that builds the same tree, so a program
//! can carry a checked-in model without reading a file at all.

use std::fmt::Write as _;

use multicalc::kinematics::{Joint, JointKind, JointParent, KinematicTree};

use crate::{MjcfError, RobotModel};

/// Large enough for any model this loader can realistically parse. Used only to check every
/// joint's own numbers are sound before anything is written; unrelated to the capacity the caller
/// asks the *generated* tree to have.
const VALIDATION_CAPACITY: usize = 128;

/// Which scalar the generated model is built in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedScalar {
    F32,
    F64,
}

impl GeneratedScalar {
    fn token(self) -> &'static str {
        match self {
            GeneratedScalar::F32 => "f32",
            GeneratedScalar::F64 => "f64",
        }
    }

    /// Narrows a value to this scalar and writes it with `{:?}`, the shortest text that reads
    /// back as the same number.
    fn format(self, value: f64) -> String {
        match self {
            GeneratedScalar::F32 => format!("{:?}", value as f32),
            GeneratedScalar::F64 => format!("{value:?}"),
        }
    }
}

/// How to write a model out as Rust source.
#[derive(Debug, Clone, PartialEq)]
pub struct RustSourceOptions {
    function_name: String,
    scalar: GeneratedScalar,
    capacity: usize,
    tip: Option<String>,
    header: String,
    documentation: String,
}

impl RustSourceOptions {
    /// Defaults: single precision, one slot per body, no tip, no header or documentation line.
    #[must_use]
    pub fn new(function_name: &str) -> Self {
        RustSourceOptions {
            function_name: function_name.to_owned(),
            scalar: GeneratedScalar::F32,
            capacity: 0,
            tip: None,
            header: String::new(),
            documentation: String::new(),
        }
    }

    /// The scalar the generated tree is built in.
    #[must_use]
    pub fn with_scalar(self, scalar: GeneratedScalar) -> Self {
        RustSourceOptions { scalar, ..self }
    }

    /// The generated tree's slot count. `0` (the default) means as many slots as the model needs.
    #[must_use]
    pub fn with_capacity(self, capacity: usize) -> Self {
        RustSourceOptions { capacity, ..self }
    }

    /// Writes the chain down to `tip` rather than the whole model.
    #[must_use]
    pub fn with_tip(self, tip: &str) -> Self {
        RustSourceOptions {
            tip: Some(tip.to_owned()),
            ..self
        }
    }

    /// A line prepended as `// {header}`, ahead of the module attributes.
    #[must_use]
    pub fn with_header(self, header: &str) -> Self {
        RustSourceOptions {
            header: header.to_owned(),
            ..self
        }
    }

    /// A line written as the generated function's doc comment.
    #[must_use]
    pub fn with_documentation(self, documentation: &str) -> Self {
        RustSourceOptions {
            documentation: documentation.to_owned(),
            ..self
        }
    }
}

impl RobotModel {
    /// The model written out as Rust source: one function that builds the tree, ready to be
    /// checked in and compiled into a program that never reads the file.
    ///
    /// Errors: as [`kinematic_tree_to`](RobotModel::kinematic_tree_to), since the model is built
    /// once here to check it is sound before anything is written.
    pub fn to_rust_source(&self, options: &RustSourceOptions) -> Result<String, MjcfError> {
        let slots = match &options.tip {
            Some(tip) => self.path_to(tip)?,
            None => (0..self.bodies().len()).collect::<Vec<usize>>(),
        };

        if self.has_floating_base() {
            return Err(MjcfError::FloatingBaseUnsupported {
                body: self
                    .body(0)
                    .unwrap_or_else(|| unreachable!("a model always has at least one body"))
                    .name()
                    .to_owned(),
            });
        }
        let capacity = if options.capacity == 0 {
            slots.len()
        } else {
            options.capacity
        };
        if slots.len() > capacity {
            return Err(MjcfError::TreeCapacityExceeded {
                needed: slots.len(),
                capacity,
            });
        }

        let (joints, parents) = self.joints_and_parents(&slots);
        KinematicTree::<VALIDATION_CAPACITY, f64>::try_from_joints(&joints, &parents)
            .map_err(MjcfError::Kinematics)?;

        let names: Vec<&str> = slots
            .iter()
            .map(|&index| {
                self.body(index)
                    .unwrap_or_else(|| unreachable!("slots only ever name real bodies"))
                    .name()
            })
            .collect();
        Ok(render(&joints, &parents, &names, capacity, options))
    }
}

fn render(
    joints: &[Joint<f64>],
    parents: &[JointParent],
    names: &[&str],
    capacity: usize,
    options: &RustSourceOptions,
) -> String {
    let scalar = options.scalar;
    let token = scalar.token();
    let mut out = String::new();

    if !options.header.is_empty() {
        let _ = writeln!(out, "// {}", options.header);
    }
    let _ = writeln!(out, "#![cfg_attr(rustfmt, rustfmt::skip)]");
    let _ = writeln!(out, "#![allow(clippy::unreadable_literal)]");
    let _ = writeln!(out, "#![allow(dead_code)]");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "use multicalc::kinematics::{{Joint, JointParent, KinematicTree}};"
    );
    let _ = writeln!(out, "use multicalc::linear_algebra::Vector;");
    let _ = writeln!(out, "use multicalc::spatial::{{Quaternion, SE3, SO3}};");
    let _ = writeln!(out);
    if !options.documentation.is_empty() {
        let _ = writeln!(out, "/// {}", options.documentation);
    }
    let _ = writeln!(
        out,
        "pub fn {}() -> KinematicTree<{capacity}, {token}> {{",
        options.function_name
    );
    let _ = writeln!(
        out,
        "    let mut tree = KinematicTree::<{capacity}, {token}>::new();"
    );

    for ((joint, parent), name) in joints.iter().zip(parents).zip(names) {
        render_push(&mut out, name, *joint, *parent, scalar);
    }

    let _ = writeln!(out, "    tree");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Where a joint sits and how it is turned, from the numbers the model file gave."
    );
    let _ = writeln!(
        out,
        "fn pose(translation: [{token}; 3], quaternion: [{token}; 4]) -> SE3<{token}> {{"
    );
    let _ = writeln!(
        out,
        "    let turn = Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3])"
    );
    let _ = writeln!(out, "        .try_normalized()");
    let _ = writeln!(
        out,
        "        .unwrap_or_else(|| unreachable!(\"the model this was written from was checked\"));"
    );
    let _ = writeln!(
        out,
        "    SE3::from_parts(SO3::from_quaternion(turn), Vector::new(translation))"
    );
    let _ = writeln!(out, "}}");

    out
}

fn render_push(
    out: &mut String,
    name: &str,
    joint: Joint<f64>,
    parent: JointParent,
    scalar: GeneratedScalar,
) {
    let n = |value: f64| scalar.format(value);
    let origin = joint.origin();
    let translation = origin.translation();
    let quaternion = origin.rotation().quaternion().as_array();
    let pose_text = format!(
        "pose([{}, {}, {}], [{}, {}, {}, {}])",
        n(translation[0]),
        n(translation[1]),
        n(translation[2]),
        n(quaternion[0]),
        n(quaternion[1]),
        n(quaternion[2]),
        n(quaternion[3]),
    );
    let parent_text = match parent {
        JointParent::World => "JointParent::World".to_owned(),
        JointParent::Joint(index) => format!("JointParent::Joint({index})"),
    };

    let _ = writeln!(out, "    // {name}");
    match joint.kind() {
        JointKind::Fixed => {
            let _ = writeln!(out, "    tree.push(");
            let _ = writeln!(out, "        Joint::fixed({pose_text}),");
            let _ = writeln!(out, "        {parent_text},");
            let _ = writeln!(out, "    )");
        }
        kind @ (JointKind::Revolute | JointKind::Prismatic) => {
            let axis = joint.axis();
            let axis_text = format!("[{}, {}, {}]", n(axis[0]), n(axis[1]), n(axis[2]));
            let constructor = if kind == JointKind::Revolute {
                "revolute"
            } else {
                "prismatic"
            };
            let _ = writeln!(out, "    tree.push(");
            let _ = writeln!(
                out,
                "        Joint::{constructor}(Vector::new({axis_text}), {pose_text})"
            );
            if kind == JointKind::Revolute {
                let anchor = joint.anchor();
                let anchor_text = format!("[{}, {}, {}]", n(anchor[0]), n(anchor[1]), n(anchor[2]));
                let _ = writeln!(out, "            .with_anchor(Vector::new({anchor_text}))");
            }
            if let Some((lower, upper)) = joint.limits() {
                let _ = writeln!(out, "            .with_limits({}, {})", n(lower), n(upper));
            }
            let _ = writeln!(
                out,
                "            .with_zero_offset({})",
                n(joint.zero_offset())
            );
            let _ = writeln!(out, "            .with_armature({})", n(joint.armature()));
            let _ = writeln!(out, "            .with_damping({})", n(joint.damping()));
            let _ = writeln!(
                out,
                "            .with_friction_loss({}),",
                n(joint.friction_loss())
            );
            let _ = writeln!(out, "        {parent_text},");
            let _ = writeln!(out, "    )");
        }
    }
    let _ = writeln!(
        out,
        "    .unwrap_or_else(|_| unreachable!(\"the model this was written from was checked\"));"
    );
}
