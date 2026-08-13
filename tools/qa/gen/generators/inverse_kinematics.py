"""Inverse-kinematics goldens from mink's differential IK on the same MuJoCo models.

The model travels in the fixture the same way the forward-kinematics cases carry it, so the tree
the Rust side builds is the model mink was given rather than a transcription of it.

What is pinned, and what deliberately is not. mink and the crate's solver are both local
differential methods: they linearize about the current configuration and step. On a redundant arm
the two land on different valid configurations, so a golden joint vector would pin one arbitrary
branch of a multi-valued answer and break the day either solver's damping is retuned. Every case
therefore pins the *pose reached* and whether the solve converged. Joint angles are pinned only for
the six-hinge case, which is not redundant and is seeded next to its solution, so the branch is
locally unique and the two solvers have to agree on it.

mink calls MuJoCo's own `mj_jac`, so these goldens do not independently re-check the Jacobian —
the kinematics fixtures already do that. What they check is the solve: limit handling, behaviour
near a singular pose, and where the iteration lands.
"""

import os

import mink
import mujoco
import numpy as np

import schema
from generators.kinematics import _model_inputs

MENAGERIE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "third_party", "menagerie"
)
FRANKA = "franka_emika_panda/panda.xml"

# mink's per-step solve settings, held here so the fixture's own note can quote them.
SOLVE_DAMPING = 1e-3
SOLVE_DT = 1e-2
MAX_STEPS = 500
# Convergence is on the task residual, matching the tolerances the crate's solver defaults to.
POSITION_TOLERANCE = 1e-6
ORIENTATION_TOLERANCE = 1e-6


def _tol():
    # The goldens are another solver's answer, not a closed form, so the bar is the pose both
    # solvers have to agree on rather than f64 composition noise. A converged differential IK is
    # only as tight as the residual it stopped at.
    return {"f64": schema.tol(1e-6, 1e-6)}


def _solve(model, task_body, target, seed, limits):
    """Runs mink to convergence and returns the configuration, the pose reached, and whether it
    converged inside both tolerances."""
    configuration = mink.Configuration(model)
    configuration.update(seed.copy())

    frame_task = mink.FrameTask(
        frame_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, task_body),
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    frame_task.set_target(mink.SE3.from_matrix(target))
    tasks = [frame_task]

    converged = False
    for _ in range(MAX_STEPS):
        velocity = mink.solve_ik(
            configuration, tasks, SOLVE_DT, "daqp", SOLVE_DAMPING, limits=limits
        )
        configuration.integrate_inplace(velocity, SOLVE_DT)
        error = frame_task.compute_error(configuration)
        if (
            np.linalg.norm(error[:3]) <= POSITION_TOLERANCE
            and np.linalg.norm(error[3:]) <= ORIENTATION_TOLERANCE
        ):
            converged = True
            break

    reached = configuration.get_transform_frame_to_world(
        frame_task.frame_name, frame_task.frame_type
    )
    return configuration.q.copy(), reached, converged


def _pose_matrix(model, data, body):
    """The body's world pose as a 4x4, which is what mink's SE3 takes."""
    pose = np.eye(4)
    pose[:3, :3] = np.array(data.xmat[body], dtype=float).reshape(3, 3)
    pose[:3, 3] = np.array(data.xpos[body], dtype=float)
    return pose


def _target_from_configuration(model, readings, joint_of_body, body):
    """A target pose taken by running MuJoCo's own forward kinematics on `readings`, so the target
    is reachable by construction and the fixture never pins an unreachable ask."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for mapped_body, joint in joint_of_body.items():
        data.qpos[model.jnt_qposadr[joint]] = readings[mapped_body - 1]
    mujoco.mj_kinematics(model, data)
    return _pose_matrix(model, data, body)


def _seed_qpos(model, readings, joint_of_body):
    """A full `qpos` vector from one reading per fixture joint slot."""
    seed = model.qpos0.copy()
    for body, joint in joint_of_body.items():
        seed[model.jnt_qposadr[joint]] = readings[body - 1]
    return seed


def _slot_readings(model, qpos, joint_of_body, slot_count):
    """Readings back out of a `qpos` vector, one per fixture joint slot; a slot with no joint
    reads zero."""
    readings = np.zeros(slot_count)
    for body, joint in joint_of_body.items():
        readings[body - 1] = qpos[model.jnt_qposadr[joint]]
    return readings


def _write(
    out,
    case,
    meta,
    model,
    *,
    task_body,
    solution,
    seed_offset,
    limits,
    pin_configuration,
    equation,
    operations,
    extra=None,
):
    inputs, joint_of_body = _model_inputs(model)
    slot_count = model.nbody - 1

    target = _target_from_configuration(model, solution, joint_of_body, task_body)
    seed_readings = np.array(
        [
            reading + (seed_offset if index in {b - 1 for b in joint_of_body} else 0.0)
            for index, reading in enumerate(solution)
        ]
    )
    reached_q, reached, converged = _solve(
        model,
        task_body,
        target,
        _seed_qpos(model, seed_readings, joint_of_body),
        limits,
    )

    assert converged, f"{case}: mink did not converge, so there is no golden to pin"

    inputs.update(
        {
            # The frame the task pins, as a fixture joint slot.
            "tool_index": schema.integer(task_body - 1),
            "target_position": schema.vector(target[:3, 3]),
            "target_quaternion": schema.vector(_quaternion(target[:3, :3])),
            "seed": schema.vector(seed_readings),
        }
    )
    inputs.update(extra or {})

    expected = {
        "reached_position": schema.vector(reached.translation()),
        "reached_quaternion": schema.vector(reached.rotation().wxyz),
        # 1 when mink reached the target inside both tolerances, 0 otherwise. Always 1 here, since
        # a case that does not converge is rejected above.
        "converged": schema.integer(1 if converged else 0),
    }
    if pin_configuration:
        # Only where the arm is not redundant and the seed sits next to the solution, so the two
        # solvers have no room to pick different branches.
        expected["joint_positions"] = schema.vector(
            _slot_readings(model, reached_q, joint_of_body, slot_count)
        )

    schema.write_fixture(
        out, "inverse_kinematics", case,
        meta, _tol(), inputs, expected,
        equation=equation,
        operations=operations,
    )


def _quaternion(rotation):
    """A rotation matrix as a scalar-first quaternion, matching the rest of the fixtures."""
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    return quaternion


def _six_hinge_chain(out, meta):
    """Six hinges alternating about x and y on 0.25 links, with a welded tool.

    Six actuated joints against a six-dimensional task, so the solution next to the seed is
    locally unique — this is the one case whose joint angles are pinned.
    """
    model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
      <body name="link2" pos="0 0 0.25">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
        <body name="link3" pos="0 0 0.25">
          <joint name="j3" type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
          <body name="link4" pos="0 0 0.25">
            <joint name="j4" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
            <body name="link5" pos="0 0 0.25">
              <joint name="j5" type="hinge" axis="1 0 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
              <body name="link6" pos="0 0 0.25">
                <joint name="j6" type="hinge" axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
                <body name="tool" pos="0 0 0.25">
                  <geom type="sphere" size="0.03"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    _write(
        out, "six_hinge_chain", meta, model,
        task_body=7,  # the welded tool
        solution=np.array([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0]),
        seed_offset=0.15,
        limits=None,
        pin_configuration=True,
        equation="six hinges alternating about x and y on 0.25 links, plus a welded tool frame",
        operations=[
            "six-hinge inverse kinematics: joint readings and the pose they reach",
        ],
    )


def _six_hinge_chain_near_singular(out, meta):
    """The same chain seeded straight, where every axis lies in one plane and the tool cannot be
    turned about the axis the arm is stretched along.

    Redundancy is not the issue here — the seed is on a singularity, so the branch taken depends
    on how each solver damps its way off it. Only the pose is pinned.
    """
    model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
      <body name="link2" pos="0 0 0.25">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
        <body name="link3" pos="0 0 0.25">
          <joint name="j3" type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
          <body name="link4" pos="0 0 0.25">
            <joint name="j4" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
            <body name="link5" pos="0 0 0.25">
              <joint name="j5" type="hinge" axis="1 0 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
              <body name="link6" pos="0 0 0.25">
                <joint name="j6" type="hinge" axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02"/>
                <body name="tool" pos="0 0 0.25">
                  <geom type="sphere" size="0.03"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    _write(
        out, "six_hinge_chain_near_singular", meta, model,
        task_body=7,
        solution=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]),
        seed_offset=-0.05,  # lands the seed exactly on the stretched-out pose
        limits=None,
        pin_configuration=False,
        equation="the same six-hinge chain seeded fully stretched, on a singular pose",
        operations=[
            "near-singular inverse kinematics: the pose reached off a singular seed",
        ],
    )


def _franka_panda_redundant(out, meta):
    """The committed Menagerie Franka: seven hinges against a six-dimensional task, so one degree
    of freedom is spare and the two solvers resolve it differently. Only the pose is pinned."""
    model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, FRANKA))
    _, joint_of_body = _model_inputs(model)
    solution = np.zeros(model.nbody - 1)
    for body, joint in joint_of_body.items():
        if model.jnt_limited[joint]:
            low, high = (float(v) for v in model.jnt_range[joint])
            solution[body - 1] = 0.5 * (low + high)
        else:
            solution[body - 1] = 0.3

    # The hand frame, which is what a Franka task actually pins.
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    _write(
        out, "franka_panda_redundant", meta, model,
        task_body=hand,
        solution=solution,
        seed_offset=0.1,
        limits=[mink.ConfigurationLimit(model)],
        pin_configuration=False,
        equation="Menagerie franka_emika_panda, 7 hinges against a 6-DoF task",
        operations=[
            "Franka Panda inverse kinematics: the pose reached with one degree of freedom spare",
        ],
        extra={"model_file": schema.string(FRANKA)},
    )


def run(out, seed):
    meta = schema.metadata(
        "inverse_kinematics", seed,
        "targets taken by running forward kinematics on a chosen configuration, then seeding the "
        "solve away from it; goldens are mink's differential IK on the same model",
        libraries=("mujoco", "mink"),
        reference="mink {mink} on MuJoCo {mujoco}",
    )
    _six_hinge_chain(out, meta)
    _six_hinge_chain_near_singular(out, meta)
    _franka_panda_redundant(out, meta)
