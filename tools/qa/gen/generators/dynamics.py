"""Rigid-body goldens from MuJoCo: one body on a free joint, told what is pushing on it.

MuJoCo is the external check. Every case is also worked out again in numpy straight from the
equations, and the two have to agree before anything is written — so a frame read the wrong way
round fails here, on the machine that generates the fixtures, rather than pinning a wrong golden
that every later comparison agrees with.
"""

import mujoco
import numpy as np

import schema

BODY = 1  # index 0 is always the world body


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


def _model_xml(mass, center_of_mass, inertia, gravity):
    """One body on a free joint, with its mass properties stated outright so nothing is derived
    from shapes."""
    full = [
        inertia[0][0],
        inertia[1][1],
        inertia[2][2],
        inertia[0][1],
        inertia[0][2],
        inertia[1][2],
    ]
    return f"""
<mujoco>
  <option gravity="{gravity[0]} {gravity[1]} {gravity[2]}"/>
  <worldbody>
    <body name="body">
      <freejoint/>
      <inertial pos="{center_of_mass[0]} {center_of_mass[1]} {center_of_mass[2]}"
                mass="{mass}"
                fullinertia="{full[0]} {full[1]} {full[2]} {full[3]} {full[4]} {full[5]}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _by_hand(
    mass, center_of_mass, inertia, gravity, quaternion, angular_rate, force, torque
):
    """The same four equations the crate implements, written out again in numpy."""
    rotation = _quaternion_to_matrix(quaternion)
    turn_about_balance_point = torque - np.cross(center_of_mass, force)
    spin_resistance = np.cross(angular_rate, inertia @ angular_rate)
    angular = np.linalg.solve(inertia, turn_about_balance_point - spin_resistance)
    balance_point_acceleration = rotation @ force / mass + gravity
    swing = np.cross(angular, center_of_mass) + np.cross(
        angular_rate, np.cross(angular_rate, center_of_mass)
    )
    return balance_point_acceleration - rotation @ swing, angular


def _case(
    out,
    meta,
    name,
    description,
    mass,
    center_of_mass,
    inertia,
    gravity,
    position,
    quaternion,
    angular_rate,
    force,
    torque,
):
    model = mujoco.MjModel.from_xml_string(
        _model_xml(mass, center_of_mass, inertia, gravity)
    )
    data = mujoco.MjData(model)
    assert model.nbody == 2 and model.nq == 7 and model.nv == 6

    data.qpos[:3] = position
    data.qpos[3:7] = quaternion  # MuJoCo writes the leading number first, as we do
    data.qvel[:3] = 0.0  # a straight-line speed changes nothing without drag
    data.qvel[3:6] = angular_rate

    # MuJoCo takes an applied push and turn in world axes, acting at the balance point.
    rotation = _quaternion_to_matrix(quaternion)
    data.xfrc_applied[BODY, :3] = rotation @ force
    data.xfrc_applied[BODY, 3:] = rotation @ (torque - np.cross(center_of_mass, force))

    mujoco.mj_forward(model, data)
    linear = np.array(data.qacc[:3], dtype=float)
    angular = np.array(data.qacc[3:6], dtype=float)

    hand_linear, hand_angular = _by_hand(
        mass, center_of_mass, inertia, gravity, quaternion, angular_rate, force, torque
    )
    np.testing.assert_allclose(linear, hand_linear, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(angular, hand_angular, atol=1e-9, rtol=1e-9)

    inputs = {
        "mass": schema.scalar(mass),
        "center_of_mass": schema.vector(center_of_mass),
        "rotational_inertia": schema.matrix(inertia),
        "gravity": schema.vector(gravity),
        "orientation": schema.vector(quaternion),
        "angular_rate": schema.vector(angular_rate),
        "force": schema.vector(force),
        "torque": schema.vector(torque),
    }
    expected = {
        "linear_acceleration": schema.vector(linear),
        "angular_acceleration": schema.vector(angular),
    }
    schema.write_fixture(
        out,
        "dynamics",
        name,
        meta,
        {"f64": schema.tol(1e-10, 1e-10)},
        inputs,
        expected,
        equation="m·a = R·f + m·g ; I·α + ω×(I·ω) = τ − c×f",
        operations=[f"single rigid body accelerations, {description}"],
    )


def run(out, seed):
    meta = schema.metadata(
        "dynamics",
        seed,
        "two hand-chosen bodies and loadings; goldens are MuJoCo's own solve of the same model",
        libraries=("mujoco", "numpy"),
        reference="MuJoCo {mujoco}",
    )
    level = np.array([1.0, 0.0, 0.0, 0.0])
    # A quarter turn about x.
    tilted = np.array([0.9238795325112867, 0.3826834323650898, 0.0, 0.0])
    gravity = np.array([0.0, 0.0, -9.81])
    quadrotor = np.diag([0.005, 0.007, 0.009])

    # Balanced on its own origin with nothing pulling or pushing on it, so every term drops out
    # but the one where a spinning body resists having its axis moved.
    _case(
        out, meta, "free_body_spinning_no_torque", "a spinning body with nothing applied",
        0.8, np.zeros(3), quadrotor, np.zeros(3),
        np.zeros(3), level, np.array([7.0, 3.0, 5.0]), np.zeros(3), np.zeros(3),
    )

    # The opposite: every term carrying at once — tilted, off-centre, spinning, and under both a
    # push and a turn as well as gravity.
    _case(
        out, meta, "free_body_tilted_with_wrench",
        "tilted and spinning, under a push and a turn",
        1.4, np.array([0.03, -0.02, 0.05]), np.diag([0.02, 0.03, 0.04]), gravity,
        np.array([1.0, -2.0, 3.0]), tilted, np.array([1.5, -0.8, 2.2]),
        np.array([0.4, -0.3, 16.0]), np.array([0.05, -0.02, 0.03]),
    )
