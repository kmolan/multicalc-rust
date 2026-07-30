"""Model-ingestion goldens from MuJoCo's own compile of the file."""

import os

import mujoco
import numpy as np

import schema

MODEL = "skydio_x2/x2.xml"
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


def run(out, seed):
    meta = schema.metadata(
        "mjcf", seed,
        "one committed Menagerie model; goldens are MuJoCo's own compile of the same file",
        libraries=("mujoco",),
        reference="MuJoCo {mujoco}",
    )
    _skydio_x2(out, meta)
