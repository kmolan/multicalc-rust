"""Articulated dynamics goldens from Pinocchio: joint torques, the joint-space inertia matrix,
and joint accelerations.

Pinocchio is the oracle for the rigid-body terms; MuJoCo's `mj_inverse` is a second opinion
asserted at generation time on a model with damping, friction loss and joint springs zeroed, so
neither library's passive-force convention enters the comparison. Neither oracle produces the
joint-friction term — Pinocchio's `rnea` has none and MuJoCo solves `frictionloss` as a
constraint — so armature, viscous damping and Coulomb friction are added in numpy on top of
Pinocchio's rigid-body torque, and Pinocchio's own armature handling is cross-checked against
`rnea(armature=0) + armature*qddot` before anything is written.
"""

import os
import tempfile

import mujoco
import numpy as np
import pinocchio as pin

import schema

MENAGERIE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "third_party", "menagerie"
)
FRANKA = "franka_emika_panda/panda.xml"

GRAVITY = np.array([0.0, 0.0, -9.81])
STATE_COUNT = 128

# One letter per joint, in the order the fixture lists them.
KIND_LETTER = {mujoco.mjtJoint.mjJNT_HINGE: "R", mujoco.mjtJoint.mjJNT_SLIDE: "P"}

FIXED_AXIS = np.array([1.0, 0.0, 0.0])

# The Franka states no frictionloss, so the flagship model would never exercise the Coulomb path.
# One fixed value per joint, in joint order, injected before the model is compiled.
FRANKA_FRICTION_LOSS = [0.30, 0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.05, 0.05]


def _quaternion_to_matrix(q):
    """A rotation matrix from a scalar-first `[w, x, y, z]` quaternion, MuJoCo's storage order."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _assert_zero_reference(model):
    """Refuses a model whose joints rest anywhere but zero.

    The crate applies MJCF `ref` as `Joint::zero_offset`; Pinocchio's own handling of `ref` is not
    something these goldens should rest on.
    """
    for joint in range(model.njnt):
        reference = float(model.qpos0[model.jnt_qposadr[joint]])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert reference == 0.0, f"joint {name!r} rests at {reference}, not zero"


def _model_inputs(model):
    """One fixture slot per body, in the order MuJoCo compiled them.

    Returns the fixture's model entries and the joint each body carries.
    """
    kinds = []
    parents = []
    origin_positions = []
    origin_quaternions = []
    axes = []
    anchors = []
    zero_offsets = []
    masses = []
    centers_of_mass = []
    rotational_inertias = []
    armatures = []
    dampings = []
    friction_losses = []
    joint_of_body = {}

    for body in range(1, model.nbody):  # index 0 is always the world body
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        count = int(model.body_jntnum[body])
        assert count <= 1, f"body {name!r} carries {count} joints; one per body is the limit here"

        parent = int(model.body_parentid[body])
        parents.append(-1 if parent == 0 else parent - 1)
        origin_positions.append(np.array(model.body_pos[body], dtype=float))
        origin_quaternions.append(np.array(model.body_quat[body], dtype=float))

        # MuJoCo stores the principal inertia and the frame it is principal in; `SpatialInertia`
        # takes the full 3x3 about the centre of mass in body axes.
        rotation = _quaternion_to_matrix(model.body_iquat[body])
        full = rotation @ np.diag(np.array(model.body_inertia[body], dtype=float)) @ rotation.T
        masses.append(float(model.body_mass[body]))
        centers_of_mass.append(np.array(model.body_ipos[body], dtype=float))
        rotational_inertias.append(full.reshape(9))

        if count == 0:
            kinds.append("F")
            axes.append(FIXED_AXIS)
            anchors.append(np.zeros(3))
            zero_offsets.append(0.0)
            armatures.append(0.0)
            dampings.append(0.0)
            friction_losses.append(0.0)
            continue

        joint = int(model.body_jntadr[body])
        kind = model.jnt_type[joint]
        assert kind in KIND_LETTER, f"body {name!r} carries an unsupported joint type"
        joint_of_body[body] = joint
        degree_of_freedom = int(model.jnt_dofadr[joint])
        kinds.append(KIND_LETTER[kind])
        axes.append(np.array(model.jnt_axis[joint], dtype=float))
        anchors.append(np.array(model.jnt_pos[joint], dtype=float))
        zero_offsets.append(float(model.qpos0[model.jnt_qposadr[joint]]))
        armatures.append(float(model.dof_armature[degree_of_freedom]))
        dampings.append(float(model.dof_damping[degree_of_freedom]))
        friction_losses.append(float(model.dof_frictionloss[degree_of_freedom]))

    inputs = {
        "joint_kinds": schema.string("".join(kinds)),  # one letter per joint: R, P, or F
        "parents": schema.vector(parents),  # -1 means the world
        "origin_positions": schema.matrix(origin_positions),  # N x 3
        "origin_quaternions": schema.matrix(origin_quaternions),  # N x 4, scalar first
        "axes": schema.matrix(axes),  # N x 3
        "anchors": schema.matrix(anchors),  # N x 3
        "zero_offsets": schema.vector(zero_offsets),  # N
        "masses": schema.vector(masses),  # N
        "centers_of_mass": schema.matrix(centers_of_mass),  # N x 3, in the body frame
        "rotational_inertias": schema.matrix(rotational_inertias),  # N x 9, row-major
        "armatures": schema.vector(armatures),  # N, zero where the slot has no joint
        "dampings": schema.vector(dampings),  # N
        "friction_losses": schema.vector(friction_losses),  # N
    }
    return inputs, joint_of_body


def _pinocchio_models(path):
    """Two copies of the model, one keeping the file's armature and one with it zeroed.

    The pair is what the armature cross-check in `_expected` compares.
    """
    with_armature = pin.buildModelFromMJCF(path)
    without_armature = pin.buildModelFromMJCF(path)
    with_armature.gravity.linear = GRAVITY.copy()
    without_armature.gravity.linear = GRAVITY.copy()
    without_armature.armature[:] = 0.0
    return with_armature, without_armature


def _slot_to_velocity_index(mujoco_model, pinocchio_model):
    """Each MuJoCo body slot mapped to Pinocchio's velocity index, by joint name.

    Never by position: the two libraries order their joints independently.
    """
    mapping = {}
    for body in range(1, mujoco_model.nbody):
        if int(mujoco_model.body_jntnum[body]) == 0:
            continue
        joint = int(mujoco_model.body_jntadr[body])
        name = mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert pinocchio_model.existJointName(name), f"Pinocchio has no joint {name!r}"
        mapping[body - 1] = int(
            pinocchio_model.idx_vs[pinocchio_model.getJointId(name)]
        )
    assert len(mapping) == pinocchio_model.nv, "joint counts disagree between the two libraries"
    return mapping


def _to_pinocchio(row, slot_to_velocity, size):
    """A slot-ordered row reordered into Pinocchio's own layout."""
    out = np.zeros(size)
    for slot, index in slot_to_velocity.items():
        out[index] = row[slot]
    return out


def _to_slots(values, slot_to_velocity, slot_count):
    """A Pinocchio-ordered vector reordered back into slot order, welds left at zero."""
    out = np.zeros(slot_count)
    for slot, index in slot_to_velocity.items():
        out[slot] = values[index]
    return out


def _sample_states(model, joint_of_body, rng, position_range, velocity_range, acceleration_range):
    """Readings per slot: positions inside each joint's range, rates and accelerations uniform.

    The first row is forced to zero velocity exactly, so the fixture pins the zero-rate Coulomb
    behaviour against the oracle rather than only in unit tests.
    """
    slot_count = model.nbody - 1
    positions = np.zeros((STATE_COUNT, slot_count))
    velocities = np.zeros((STATE_COUNT, slot_count))
    accelerations = np.zeros((STATE_COUNT, slot_count))

    for body, joint in joint_of_body.items():
        slot = body - 1
        if model.jnt_limited[joint]:
            low, high = (float(v) for v in model.jnt_range[joint])
        else:
            low, high = position_range
        positions[:, slot] = rng.uniform(low, high, size=STATE_COUNT)
        velocities[:, slot] = rng.uniform(*velocity_range, size=STATE_COUNT)
        accelerations[:, slot] = rng.uniform(*acceleration_range, size=STATE_COUNT)

    velocities[0, :] = 0.0
    return positions, velocities, accelerations


def _quiet_model(path):
    """The same model with every passive force zeroed and contacts disabled.

    Damping, friction loss and joint stiffness go to zero so `qfrc_passive` is zero under any
    convention, and contacts are disabled so no constraint force enters `qfrc_inverse`.
    """
    quiet = mujoco.MjModel.from_xml_path(path)
    quiet.dof_damping[:] = 0.0
    quiet.dof_frictionloss[:] = 0.0
    quiet.jnt_stiffness[:] = 0.0
    quiet.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    return quiet


def _mujoco_second_opinion(
    quiet, data, joint_of_body, position, velocity, acceleration, torque
):
    """MuJoCo's `mj_inverse` against Pinocchio's rigid-body-plus-armature torque."""
    data.qpos[:] = quiet.qpos0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    for body, joint in joint_of_body.items():
        data.qpos[quiet.jnt_qposadr[joint]] = position[body - 1]
        data.qvel[quiet.jnt_dofadr[joint]] = velocity[body - 1]
        data.qacc[quiet.jnt_dofadr[joint]] = acceleration[body - 1]
    mujoco.mj_inverse(quiet, data)

    theirs = np.array(
        [data.qfrc_inverse[quiet.jnt_dofadr[joint]] for _, joint in sorted(joint_of_body.items())]
    )
    ours = np.array([torque[body - 1] for body, _ in sorted(joint_of_body.items())])
    # MuJoCo's solver tolerances make 1e-8 the right bar for a second opinion; the golden itself
    # is Pinocchio's and is checked at 1e-10.
    np.testing.assert_allclose(theirs, ours, rtol=1e-8, atol=1e-8)


def _expected(mujoco_model, path, joint_of_body, positions, velocities, accelerations):
    """Pinocchio's solve of every sampled state, reordered into slot order."""
    with_armature, without_armature = _pinocchio_models(path)
    data = with_armature.createData()
    plain_data = without_armature.createData()
    quiet = _quiet_model(path)
    quiet_data = mujoco.MjData(quiet)
    slot_to_velocity = _slot_to_velocity_index(mujoco_model, with_armature)
    slot_count = mujoco_model.nbody - 1
    size = with_armature.nv

    armature = np.array(with_armature.armature, dtype=float)
    damping = np.array(with_armature.damping, dtype=float)
    friction_loss = np.array(with_armature.friction, dtype=float)

    rigid_body_torques = []
    torques = []
    joint_space_inertias = []
    forward_accelerations = []

    for state in range(len(positions)):
        position = _to_pinocchio(positions[state], slot_to_velocity, size)
        velocity = _to_pinocchio(velocities[state], slot_to_velocity, size)
        acceleration = _to_pinocchio(accelerations[state], slot_to_velocity, size)

        rigid = pin.rnea(without_armature, plain_data, position, velocity, acceleration).copy()
        with_rotor = pin.rnea(with_armature, data, position, velocity, acceleration).copy()
        # Decides whether this Pinocchio applies `model.armature` in `rnea`, rather than assuming
        # it. A failure here is visible to the maintainer, not baked into a golden.
        np.testing.assert_allclose(
            with_rotor, rigid + armature * acceleration, rtol=1e-12, atol=1e-12
        )

        # `np.sign(0.0)` is `0.0`, matching the crate's `coulomb_direction`.
        total = with_rotor + damping * velocity + friction_loss * np.sign(velocity)

        inertia = pin.crba(with_armature, data, position).copy()
        # Pinocchio fills only the upper triangle.
        inertia = np.triu(inertia) + np.triu(inertia, 1).T
        np.testing.assert_allclose(inertia, inertia.T, rtol=1e-12, atol=1e-12)
        np.linalg.cholesky(inertia)

        # The accelerations that were sampled, put through RNEA and back through ABA, must return.
        recovered = pin.aba(with_armature, data, position, velocity, with_rotor).copy()
        np.testing.assert_allclose(recovered, acceleration, rtol=1e-10, atol=1e-10)

        rigid_body_torques.append(_to_slots(with_rotor, slot_to_velocity, slot_count))
        torques.append(_to_slots(total, slot_to_velocity, slot_count))
        forward_accelerations.append(_to_slots(recovered, slot_to_velocity, slot_count))

        by_slot = np.zeros((slot_count, slot_count))
        for row_slot, row_index in slot_to_velocity.items():
            for column_slot, column_index in slot_to_velocity.items():
                by_slot[row_slot, column_slot] = inertia[row_index, column_index]
        joint_space_inertias.extend(by_slot)

        _mujoco_second_opinion(
            quiet,
            quiet_data,
            joint_of_body,
            positions[state],
            velocities[state],
            accelerations[state],
            rigid_body_torques[-1],
        )

    return {
        # K x N, Pinocchio's `rnea` with armature and no friction.
        "rigid_body_torques": schema.matrix(rigid_body_torques),
        # K x N, the full torque including damping and Coulomb.
        "torques": schema.matrix(torques),
        # (K*N) x N, configuration-major, the full symmetric H per state.
        "joint_space_inertias": schema.matrix(joint_space_inertias),
        # K x N, Pinocchio's `aba` fed `rigid_body_torques`.
        "forward_accelerations": schema.matrix(forward_accelerations),
    }


def _write(out, case, meta, model, path, label, extra=None):
    _assert_zero_reference(model)
    inputs, joint_of_body = _model_inputs(model)

    rng = np.random.default_rng(meta["seed"])
    position_range = (-np.pi, np.pi)
    velocity_range = (-3.0, 3.0) if case.endswith("double_pendulum") else (-2.0, 2.0)
    acceleration_range = (-8.0, 8.0) if case.endswith("double_pendulum") else (-5.0, 5.0)
    positions, velocities, accelerations = _sample_states(
        model, joint_of_body, rng, position_range, velocity_range, acceleration_range
    )

    inputs["joint_positions"] = schema.matrix(positions)  # K x N
    inputs["joint_velocities"] = schema.matrix(velocities)  # K x N
    inputs["joint_accelerations"] = schema.matrix(accelerations)  # K x N
    inputs.update(extra or {})

    expected = _expected(model, path, joint_of_body, positions, velocities, accelerations)
    schema.write_fixture(
        out, "dynamics", case, meta,
        {"f64": schema.tol(1e-10, 1e-10)},
        inputs, expected,
        equation=(
            "τ = H(q)·q̈ + C(q,q̇)·q̇ + G(q) + armature⊙q̈ + damping⊙q̇ + frictionloss⊙sign(q̇)"
        ),
        operations=[
            f"{label} inverse dynamics, {STATE_COUNT} states: (q, q̇, q̈) -> τ",
            f"{label} joint-space inertia, {STATE_COUNT} states: q -> H(q)",
            f"{label} forward dynamics, {STATE_COUNT} states: (q, q̇, τ) -> q̈",
        ],
    )


DOUBLE_PENDULUM_XML = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="upper" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" armature="0.05" damping="0.7"
             frictionloss="0.2"/>
      <inertial pos="0.5 0 0" mass="1.5" diaginertia="0.02 0.02 0.02"/>
      <body name="lower" pos="1 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" armature="0.03" damping="0.4"
               frictionloss="0.1"/>
        <inertial pos="0.5 0 0" mass="1.5" diaginertia="0.02 0.02 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _double_pendulum(out, meta, directory):
    """Two hinges about y on unit links, every joint-dynamics field stated in the file."""
    path = os.path.join(directory, "articulated_double_pendulum.xml")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(DOUBLE_PENDULUM_XML)
    model = mujoco.MjModel.from_xml_path(path)
    _write(out, "articulated_double_pendulum", meta, model, path, "double pendulum")


def _franka_panda(out, meta, directory):
    """The vendored Menagerie Franka, with the elements neither library represents removed.

    `<tendon>` and `<equality>` are deleted: neither multicalc nor Pinocchio carries a tendon, and
    leaving the equality in would put constraint forces into MuJoCo's `qfrc_inverse`. The
    `<actuator>` block goes with them — one of its entries drives the tendon, and no actuator
    enters inverse dynamics — and the `<key>` block with those, since a keyframe states a `ctrl`
    the actuator-free model no longer has. The file states no `frictionloss`, so one is injected
    per joint to exercise the Coulomb path.
    """
    source = os.path.join(MENAGERIE, FRANKA)
    spec = mujoco.MjSpec.from_file(source)

    for keyframe in list(spec.keys):
        spec.delete(keyframe)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for tendon in list(spec.tendons):
        spec.delete(tendon)
    for equality in list(spec.equalities):
        spec.delete(equality)
    for joint, loss in zip(spec.joints, FRANKA_FRICTION_LOSS):
        joint.frictionloss = loss

    # The edited model is written to a scratch directory, so the mesh paths — relative to the
    # vendored file — have to be made absolute before they move.
    spec.meshdir = os.path.abspath(os.path.join(MENAGERIE, os.path.dirname(FRANKA), spec.meshdir))

    model = spec.compile()
    assert model.ntendon == 0 and model.neq == 0, "tendons or equalities survived the edit"

    # Pinocchio reads a file, so the edited model is written back out and both libraries load the
    # same bytes.
    path = os.path.join(directory, "articulated_franka_panda.xml")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(path)

    _write(
        out, "articulated_franka_panda", meta, model, path, "Franka Panda",
        extra={"model_file": schema.string(FRANKA)},
    )


def run(out, seed):
    meta = schema.metadata(
        "articulated_dynamics",
        seed,
        f"{STATE_COUNT} states per case; q uniform in each joint's range or [-pi, pi], "
        "qdot uniform on [-3, 3] (double pendulum) or [-2, 2] (Franka), qddot uniform on "
        "[-8, 8] or [-5, 5]; the first state is forced to qdot = 0 exactly. The Franka has its "
        "<tendon>, <equality>, <actuator> and <key> elements deleted and joint frictionloss set "
        f"to {FRANKA_FRICTION_LOSS}, since the file states none.",
        libraries=("pin", "mujoco", "numpy"),
        reference="Pinocchio {pin}",
    )
    with tempfile.TemporaryDirectory() as directory:
        _double_pendulum(out, meta, directory)
        _franka_panda(out, meta, directory)
