"""Forward-kinematics goldens from MuJoCo's own solve of the same model.

The model travels in the fixture: one entry per joint saying what it does, what it hangs off, where
it sits, which way its axis points, where it turns about, and what reading counts as not having
moved. Those entries are read out of MuJoCo's compiled model, so the tree the Rust side builds is
the model MuJoCo was given rather than a transcription of it.
"""

import os

import mujoco
import numpy as np

import schema

MENAGERIE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "third_party", "menagerie"
)
FRANKA = "franka_emika_panda/panda.xml"

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
    """MuJoCo's own solve: every body's world pose, one block of bodies per configuration."""
    data = mujoco.MjData(model)
    positions = []
    quaternions = []
    for row in configurations:
        data.qpos[:] = model.qpos0
        for body, joint in joint_of_body.items():
            data.qpos[model.jnt_qposadr[joint]] = row[body - 1]
        mujoco.mj_kinematics(model, data)
        for body in range(1, model.nbody):
            positions.append(np.array(data.xpos[body], dtype=float))
            quaternions.append(np.array(data.xquat[body], dtype=float))
    return {
        "world_positions": schema.matrix(positions),  # (K*N) x 3, configuration-major
        "world_quaternions": schema.matrix(quaternions),  # (K*N) x 4, scalar first
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
        equation="two hinges about z, unit links, plus a fixed tool frame",
        operations=[
            "planar two-link forward kinematics: world position and orientation per link"
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
        equation=(
            "hinge with a shifted zero, slide with a shifted zero, fixed link, hinge turning "
            "about an off-origin point"
        ),
        operations=[
            "mixed-joint forward kinematics: world position and orientation per link"
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
        equation="one hinge carrying two sibling hinges",
        operations=[
            "branching forward kinematics: world position and orientation per link"
        ],
    )


def _franka_panda_seven_joint(out, meta, rng):
    """The committed Menagerie Franka: seven hinges, two finger slides, and fixed links."""
    model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, FRANKA))
    _, joint_of_body = _model_inputs(model)
    configurations = _random_configurations(model, joint_of_body, rng, 8)
    _write(
        out, "franka_panda_seven_joint", meta, model, configurations,
        equation="Menagerie franka_emika_panda, 7 hinges plus fixed links",
        operations=[
            "Franka Panda forward kinematics over 8 configurations: world position and "
            "orientation per link"
        ],
        extra={"model_file": schema.string(FRANKA)},
    )


def run(out, seed):
    meta = schema.metadata(
        "kinematics", seed,
        "three hand-written models plus one committed Menagerie model; joint readings drawn "
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
