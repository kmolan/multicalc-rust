"""Forward-kinematics and Jacobian goldens from MuJoCo's own solve of the same model.

The model travels in the fixture: one entry per joint saying what it does, what it hangs off, where
it sits, which way its axis points, where it turns about, and what reading counts as not having
moved. Those entries are read out of MuJoCo's compiled model, so the tree the Rust side builds is
the model MuJoCo was given rather than a transcription of it.

Alongside every body's world pose, each configuration also carries MuJoCo's own Jacobian for that
body: how each joint's rate moves it, taken at the body's own origin and read in world axes.
"""

import os

import mujoco
import numpy as np

import schema

MENAGERIE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "third_party", "menagerie"
)
FRANKA = "franka_emika_panda/panda.xml"
GO1 = "unitree_go1/go1.xml"

# One letter per joint, in the order the fixture lists them.
KIND_LETTER = {mujoco.mjtJoint.mjJNT_HINGE: "R", mujoco.mjtJoint.mjJNT_SLIDE: "P"}

FIXED_AXIS = np.array([1.0, 0.0, 0.0])


def _tol():
    # Forward kinematics is host-side f64 work; the goldens are MuJoCo's own solve of the same
    # model, so the bar sits just above f64 composition noise over seven joints.
    return {"f64": schema.tol(1e-12, 1e-11)}


def _model_inputs(model):
    """One fixture joint per body, in the order MuJoCo compiled them.

    Returns the fixture's model entries and the joint each body carries, which the golden pass
    needs to know where a reading goes in `qpos`.
    """
    kinds = []
    parents = []
    origin_positions = []
    origin_quaternions = []
    axes = []
    anchors = []
    zero_offsets = []
    joint_of_body = {}

    for body in range(1, model.nbody):  # index 0 is always the world body
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        count = int(model.body_jntnum[body])
        assert count <= 1, f"body {name!r} carries {count} joints; one per body is the limit here"

        parent = int(model.body_parentid[body])
        parents.append(-1 if parent == 0 else parent - 1)
        origin_positions.append(np.array(model.body_pos[body], dtype=float))
        origin_quaternions.append(np.array(model.body_quat[body], dtype=float))

        if count == 0:
            kinds.append("F")
            axes.append(FIXED_AXIS)
            anchors.append(np.zeros(3))
            zero_offsets.append(0.0)
            continue

        joint = int(model.body_jntadr[body])
        kind = model.jnt_type[joint]
        assert kind in KIND_LETTER, f"body {name!r} carries an unsupported joint type"
        joint_of_body[body] = joint
        kinds.append(KIND_LETTER[kind])
        axes.append(np.array(model.jnt_axis[joint], dtype=float))
        anchors.append(np.array(model.jnt_pos[joint], dtype=float))
        # MuJoCo's resting reading for the joint, set by `ref` in the file.
        zero_offsets.append(float(model.qpos0[model.jnt_qposadr[joint]]))

    inputs = {
        "joint_kinds": schema.string("".join(kinds)),  # one letter per joint: R, P, or F
        "parents": schema.vector(parents),  # -1 means the world
        "origin_positions": schema.matrix(origin_positions),  # N x 3
        "origin_quaternions": schema.matrix(origin_quaternions),  # N x 4, scalar first
        "axes": schema.matrix(axes),  # N x 3
        "anchors": schema.matrix(anchors),  # N x 3
        "zero_offsets": schema.vector(zero_offsets),  # N
    }
    return inputs, joint_of_body


def _expected(model, configurations, joint_of_body):
    """MuJoCo's own solve: every body's world pose and Jacobian, one block of bodies per
    configuration."""
    data = mujoco.MjData(model)
    slot_count = model.nbody - 1
    # `mj_jac` gives one column per degree of freedom, while the fixture gives every body a slot of
    # its own, so a body carrying no joint has no column. This says which degree of freedom belongs
    # in which fixture slot; a slot with no joint stays zero, which is what the Rust side expects
    # for a weld. Written out here so the test can compare column i against fixture joint i with no
    # index arithmetic of its own.
    dof_of_slot = {
        body - 1: int(model.jnt_dofadr[joint]) for body, joint in joint_of_body.items()
    }

    positions = []
    quaternions = []
    jacobians = []
    for row in configurations:
        data.qpos[:] = model.qpos0
        for body, joint in joint_of_body.items():
            data.qpos[model.jnt_qposadr[joint]] = row[body - 1]
        mujoco.mj_kinematics(model, data)
        # `mj_jac` reads the per-joint motion axes that this fills in, so it has to run first.
        mujoco.mj_comPos(model, data)
        for body in range(1, model.nbody):
            positions.append(np.array(data.xpos[body], dtype=float))
            quaternions.append(np.array(data.xquat[body], dtype=float))

            linear = np.zeros((3, model.nv))
            angular = np.zeros((3, model.nv))
            # Measured at the body's own origin, so the golden lines up with a Jacobian asked for
            # at that body's frame.
            mujoco.mj_jac(model, data, linear, angular, data.xpos[body], body)
            # Linear rows first, matching the crate's [v; w] ordering.
            by_dof = np.vstack((linear, angular))
            by_slot = np.zeros((6, slot_count))
            for slot, dof in dof_of_slot.items():
                by_slot[:, slot] = by_dof[:, dof]
            jacobians.extend(by_slot)
    return {
        "world_positions": schema.matrix(positions),  # (K*N) x 3, configuration-major
        "world_quaternions": schema.matrix(quaternions),  # (K*N) x 4, scalar first
        # (K*N*6) x N, configuration-major then body-major, six rows per body
        "world_jacobians": schema.matrix(jacobians),
    }


def _random_configurations(model, joint_of_body, rng, count):
    """Readings drawn uniformly inside each joint's range, or [-pi, pi] where it has none."""
    rows = np.zeros((count, model.nbody - 1))
    for body, joint in joint_of_body.items():
        if model.jnt_limited[joint]:
            low, high = (float(v) for v in model.jnt_range[joint])
        else:
            low, high = -np.pi, np.pi
        rows[:, body - 1] = rng.uniform(low, high, size=count)
    return rows


def _write(out, case, meta, model, configurations, *, equation, operations, extra=None):
    inputs, joint_of_body = _model_inputs(model)
    inputs["joint_positions"] = schema.matrix(configurations)  # K x N, one row per configuration
    inputs.update(extra or {})
    schema.write_fixture(
        out, "kinematics", case,
        meta, _tol(), inputs, _expected(model, configurations, joint_of_body),
        equation=equation,
        operations=operations,
    )


def _planar_two_link_revolute(out, meta):
    """Two hinges about z on unit links, with a tool frame welded past the second."""
    model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="upper_arm">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.04"/>
      <body name="forearm" pos="1 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.04"/>
        <body name="tool" pos="1 0 0">
          <geom type="sphere" size="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    # The tool carries no joint, so its slot is padded with a zero.
    configurations = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, -0.7, 0.0],
            [np.pi / 2, np.pi / 2, 0.0],
            [-1.2, 2.1, 0.0],
        ]
    )
    _write(
        out, "planar_two_link_revolute", meta, model, configurations,
        equation="2 z-hinges, unit links, fixed tool",
        operations=[
            "planar 2-link FK: q -> pose per link",
            "planar 2-link Jacobian: v -> twist per link",
        ],
    )


def _mixed_revolute_prismatic_fixed(out, meta, rng):
    """Every joint kind at once, with shifted zeros and an off-origin turning point."""
    model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="turret">
      <joint name="yaw" type="hinge" axis="0 0 1" ref="0.25"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.04"/>
      <body name="ram" pos="0 0 0.2">
        <joint name="reach" type="slide" axis="1 0 0" ref="-0.1" range="-0.5 0.5"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
        <body name="bracket" pos="0.4 0 0">
          <geom type="box" size="0.05 0.05 0.05"/>
          <body name="wrist" pos="0 0 0.1">
            <joint name="pitch" type="hinge" axis="0 1 0" pos="0.3 0 0"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    _, joint_of_body = _model_inputs(model)
    configurations = _random_configurations(model, joint_of_body, rng, 4)
    _write(
        out, "mixed_revolute_prismatic_fixed", meta, model, configurations,
        equation="hinge+slide (shifted zeros), fixed link, off-origin hinge",
        operations=[
            "mixed-joint FK: q -> pose per link",
            "mixed-joint Jacobian: v -> twist per link",
        ],
    )


def _branching_three_joint(out, meta, rng):
    """One hinge carrying two siblings, so neither branch may see the other."""
    model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="base">
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.05"/>
      <body name="left_arm" pos="0.2 0 0.3">
        <joint name="left_pitch" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03"/>
      </body>
      <body name="right_arm" pos="-0.2 0.1 0.3">
        <joint name="right_pitch" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 -0.5 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    _, joint_of_body = _model_inputs(model)
    configurations = _random_configurations(model, joint_of_body, rng, 4)
    _write(
        out, "branching_three_joint", meta, model, configurations,
        equation="1 hinge + 2 sibling hinges",
        operations=[
            "branching FK: q -> pose per link",
            "branching Jacobian: v -> twist per link",
        ],
    )


def _franka_panda_seven_joint(out, meta, rng):
    """The committed Menagerie Franka: seven hinges, two finger slides, and fixed links."""
    model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, FRANKA))
    _, joint_of_body = _model_inputs(model)
    configurations = _random_configurations(model, joint_of_body, rng, 8)
    _write(
        out, "franka_panda_seven_joint", meta, model, configurations,
        equation="Franka Panda: 7 hinges + fixed links",
        operations=[
            "Franka Panda FK, 8 configs: q -> pose per link",
            "Franka Panda Jacobian, 8 configs: v -> twist per link",
        ],
        extra={"model_file": schema.string(FRANKA)},
    )


def _unitree_go1_floating_base(out, meta, rng):
    """The committed Menagerie Go1: a floating base (the trunk, on a free joint) carrying twelve
    hinge joints across four legs.

    Bespoke rather than routed through `_model_inputs`/`_write` above: those assume exactly one
    scalar reading per joint slot everywhere, which a free joint's seven-wide configuration and
    six-wide velocity break. Configurations here are MuJoCo's own `qpos` layout directly (free
    joint first, sized 7, then one scalar per hinge) rather than a per-slot reading, since that is
    also exactly the layout `KinematicTree::config_offset` produces for a tree whose first joint is
    floating — so the Rust side reads a fixture row straight into its own configuration vector with
    no reindexing. The Rust side builds its tree by parsing the vendored file itself, not by
    reconstructing it from fixture fields, so this fixture carries no per-joint axis/anchor data.
    """
    model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, GO1))
    assert model.njnt == 13 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nq == 19 and model.nv == 18

    hinge_ranges = np.array(
        [model.jnt_range[joint] for joint in range(1, model.njnt)], dtype=float
    )

    count = 4
    translations = rng.uniform(-1.0, 1.0, size=(count, 3))
    raw_quaternions = rng.normal(size=(count, 4))
    quaternions = raw_quaternions / np.linalg.norm(raw_quaternions, axis=1, keepdims=True)
    hinge_readings = rng.uniform(hinge_ranges[:, 0], hinge_ranges[:, 1], size=(count, 12))
    configurations = np.vstack(
        [
            model.key_qpos[0],  # the file's own "home" pose, a sanity-checkable baseline
            np.hstack([translations, quaternions, hinge_readings]),
        ]
    )

    data = mujoco.MjData(model)
    positions = []
    quats = []
    jacobians = []
    for row in configurations:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for body in range(1, model.nbody):  # index 0 is always the world body
            positions.append(np.array(data.xpos[body], dtype=float))
            quats.append(np.array(data.xquat[body], dtype=float))
            linear = np.zeros((3, model.nv))
            angular = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, linear, angular, data.xpos[body], body)
            jacobians.extend(np.vstack((linear, angular)))

    inputs = {
        "model_file": schema.string(GO1),
        "configurations": schema.matrix(configurations),  # K x 19: MuJoCo's own qpos layout
    }
    expected = {
        "world_positions": schema.matrix(positions),  # (13*K) x 3
        "world_quaternions": schema.matrix(quats),  # (13*K) x 4, scalar first
        "world_jacobians": schema.matrix(jacobians),  # (13*K*6) x 18
    }
    schema.write_fixture(
        out, "kinematics", "unitree_go1_floating_base",
        meta, _tol(), inputs, expected,
        equation="Go1: floating base (free joint) + 12 hinges",
        operations=[
            "floating-base FK: q -> pose per body",
            "floating-base Jacobian: v -> twist per body",
        ],
    )


def run(out, seed):
    meta = schema.metadata(
        "kinematics", seed,
        "three hand-written models plus two committed Menagerie models; joint readings drawn "
        "uniformly inside each joint's range; goldens are MuJoCo's own solve of the same model",
        libraries=("mujoco",),
        reference="MuJoCo {mujoco}",
    )
    # A generator of its own, so the streams the other modules draw from stay untouched.
    rng = np.random.default_rng(seed)
    _planar_two_link_revolute(out, meta)
    _mixed_revolute_prismatic_fixed(out, meta, rng)
    _branching_three_joint(out, meta, rng)
    _franka_panda_seven_joint(out, meta, rng)
    _unitree_go1_floating_base(out, meta, rng)
