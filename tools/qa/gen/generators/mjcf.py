"""Model-ingestion goldens from MuJoCo's own compile of the file."""

import os

import mujoco
import numpy as np

import schema

MODEL = "skydio_x2/x2.xml"
FRANKA = "franka_emika_panda/panda.xml"
MENAGERIE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "third_party", "menagerie"
)


def _tol():
    # Ingestion is host-side f64 work and never reruns in single precision, so there is no f32
    # entry. The goldens are exact ratios of whole numbers, so the bar is close to f64 noise.
    return {"f64": schema.tol(1e-12, 1e-11)}


def _quaternion_to_matrix(q):
    """A rotation matrix from a scalar-first `[w, x, y, z]` quaternion.

    Written out rather than borrowed so the storage order is visible in the diff: a helper from
    elsewhere might read the four numbers the other way round and the mistake would be silent.
    """
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _body_inertia(model, index):
    """One body's full inertia tensor, in the body's own axes.

    MuJoCo does not store it that way. It stores three numbers along the axes the body happens to
    line up best with, plus the turn from the body's axes to those, so the two have to be put back
    together before anything can be compared against them.
    """
    rotation = _quaternion_to_matrix(model.body_iquat[index])
    return rotation @ np.diag(np.array(model.body_inertia[index], dtype=float)) @ rotation.T


def _skydio_x2(out, meta):
    """The Skydio X2: one body on a free joint, its mass coming from the shapes."""
    model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, MODEL))

    body = 1  # index 0 is always the world body
    mass = float(model.body_mass[body])
    center_of_mass = np.array(model.body_ipos[body], dtype=float)
    inertia = _body_inertia(model, body)

    # Cross-check against numbers worked out by hand from the file before writing anything. Every
    # number in the file is an exact decimal and every mass-carrying shape states its mass
    # outright, so the truth here is a ratio of whole numbers and the bar can be tight. If
    # `body_iquat` turns the other way the corner terms change sign and this fires in the
    # generator, rather than pinning a wrong golden that every later comparison agrees with.
    assert model.nbody == 2, "expected the world body and exactly one more"
    assert model.njnt == 1 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nq == 7 and model.nv == 6
    np.testing.assert_allclose(mass, 1.325, atol=1e-12)
    np.testing.assert_allclose(
        center_of_mass, [0.0, 0.0, 0.053962264150943406], atol=1e-12
    )
    np.testing.assert_allclose(
        inertia,
        [
            [0.036651698113207544, 0.0, -0.0021],
            [0.0, 0.025411698113207547, 0.0],
            [-0.0021, 0.0, 0.060528],
        ],
        atol=1e-12,
    )

    inputs = {"model_file": schema.string(MODEL)}
    expected = {
        "mass": schema.scalar(mass),
        "center_of_mass": schema.vector(center_of_mass),
        "rotational_inertia": schema.matrix(inertia),
        "body_position": schema.vector(np.array(model.body_pos[body], dtype=float)),
        "body_quaternion": schema.vector(np.array(model.body_quat[body], dtype=float)),
        "free_joint": schema.integer(1),
        "configuration_dimension": schema.integer(model.nq),
        "velocity_dimension": schema.integer(model.nv),
    }
    schema.write_fixture(
        out, "mjcf", "skydio_x2_free_joint",
        meta, _tol(), inputs, expected,
        equation="body mass from 4 ellipsoid rotors (0.25 kg) + 1 ellipsoid hull (0.325 kg)",
        operations=["Menagerie skydio_x2 ingest: mass, centre of mass, inertia about the centre"],
    )


def _franka_panda_tree(out, meta):
    """The Franka Panda: eleven bodies, seven hinges, two sliding fingers — every joint's kind,
    geometry and travel, plus the dynamics figures forward kinematics never reads, taken straight
    off MuJoCo's compiled model."""
    model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, FRANKA))

    names = []
    kinds = []
    parents = []
    origin_positions = []
    origin_quaternions = []
    axes = []
    anchors = []
    zero_offsets = []
    limited = []
    ranges = []
    armatures = []
    dampings = []
    friction_losses = []
    spring_references = []
    spring_stiffnesses = []
    masses = []
    centers_of_mass = []
    rotational_inertias = []

    for body in range(1, model.nbody):  # index 0 is always the world body
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        names.append(name)

        parent = int(model.body_parentid[body])
        parents.append(-1 if parent == 0 else parent - 1)

        origin_positions.append(np.array(model.body_pos[body], dtype=float))
        origin_quaternions.append(np.array(model.body_quat[body], dtype=float))
        masses.append(float(model.body_mass[body]))
        centers_of_mass.append(np.array(model.body_ipos[body], dtype=float))
        rotational_inertias.append(_body_inertia(model, body))

        count = int(model.body_jntnum[body])
        assert count <= 1, f"body {name!r} carries {count} joints; one per body is the limit here"

        if count == 0:
            kinds.append("F")
            axes.append(np.array([1.0, 0.0, 0.0]))
            anchors.append(np.zeros(3))
            zero_offsets.append(0.0)
            limited.append(0.0)
            ranges.append(np.zeros(2))
            armatures.append(0.0)
            dampings.append(0.0)
            friction_losses.append(0.0)
            spring_references.append(0.0)
            spring_stiffnesses.append(0.0)
            continue

        joint = int(model.body_jntadr[body])
        kind = model.jnt_type[joint]
        assert kind in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE), (
            f"body {name!r} carries an unsupported joint type"
        )
        kinds.append("R" if kind == mujoco.mjtJoint.mjJNT_HINGE else "P")

        dof = int(model.jnt_dofadr[joint])
        qpos = int(model.jnt_qposadr[joint])
        axes.append(np.array(model.jnt_axis[joint], dtype=float))
        anchors.append(np.array(model.jnt_pos[joint], dtype=float))
        zero_offsets.append(float(model.qpos0[qpos]))
        limited.append(float(model.jnt_limited[joint]))
        ranges.append(np.array(model.jnt_range[joint], dtype=float))
        armatures.append(float(model.dof_armature[dof]))
        dampings.append(float(model.dof_damping[dof]))
        friction_losses.append(float(model.dof_frictionloss[dof]))
        spring_references.append(float(model.qpos_spring[qpos]))
        spring_stiffnesses.append(float(model.jnt_stiffness[joint]))

    # Cross-check the hand-derived truths behind the field-for-field comparison before writing
    # anything, so a wrong golden fails here rather than being pinned and passed on downstream.
    assert model.nbody == 12, "expected the world body and eleven more"
    assert model.njnt == 9 and int(model.body_jntnum[1]) == 0  # link0 is welded to the world
    assert "".join(kinds) == "FRRRRRRRFPP"
    np.testing.assert_allclose(model.dof_armature[0], 0.1, atol=1e-12)
    np.testing.assert_allclose(model.dof_damping[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(model.body_mass[2], 4.970684, atol=1e-12)
    np.testing.assert_allclose(model.jnt_range[0], [-2.8973, 2.8973], atol=1e-12)

    inputs = {"model_file": schema.string(FRANKA)}
    expected = {
        "body_names": schema.string(" ".join(names)),
        "joint_kinds": schema.string("".join(kinds)),
        "parents": schema.vector(parents),
        "origin_positions": schema.matrix(origin_positions),
        "origin_quaternions": schema.matrix(origin_quaternions),
        "axes": schema.matrix(axes),
        "anchors": schema.matrix(anchors),
        "zero_offsets": schema.vector(zero_offsets),
        "limited": schema.vector(limited),
        "ranges": schema.matrix(ranges),
        "armatures": schema.vector(armatures),
        "dampings": schema.vector(dampings),
        "friction_losses": schema.vector(friction_losses),
        "spring_references": schema.vector(spring_references),
        "spring_stiffnesses": schema.vector(spring_stiffnesses),
        "masses": schema.vector(masses),
        "centers_of_mass": schema.matrix(centers_of_mass),
        "rotational_inertias": schema.matrix(np.vstack(rotational_inertias)),
        "body_count": schema.integer(model.nbody - 1),
        "movable_joint_count": schema.integer(model.njnt),
        "free_joint": schema.integer(0),
    }
    schema.write_fixture(
        out, "mjcf", "franka_panda_tree",
        meta, _tol(), inputs, expected,
        equation="Menagerie franka_emika_panda: eleven bodies, seven hinges and two sliding fingers",
        operations=[
            "Menagerie franka_emika_panda ingest: body tree, joint axes and travel limits",
            "Menagerie franka_emika_panda ingest: per-body mass, balance point and inertia",
            "Menagerie franka_emika_panda ingest: armature, damping, friction and spring settings",
        ],
    )


def run(out, seed):
    meta = schema.metadata(
        "mjcf", seed,
        "two committed Menagerie models; goldens are MuJoCo's own compile of the same files",
        libraries=("mujoco",),
        reference="MuJoCo {mujoco}",
    )
    _skydio_x2(out, meta)
    _franka_panda_tree(out, meta)
