"""Model-based control-law goldens on the same two models the articulated dynamics rows use.

Pinocchio is the oracle for the model terms — `rnea` for the torque, `computeJointJacobian` for
the tool Jacobian — and the control laws are then re-evaluated in numpy on top of them. So what
these pin is the conventions and the crate's composition of the terms, not the formulas
themselves, the same status as the geometric-attitude rows in `control.py`. The model each
fixture carries is the model the oracle was given.
"""

import tempfile

import mujoco
import numpy as np
import pinocchio as pin

import schema
from generators.articulated_dynamics import (
    FRANKA,
    FRANKA_FRICTION_LOSS,
    assert_zero_reference,
    double_pendulum_model,
    franka_model,
    model_inputs,
    pinocchio_models,
    slot_to_velocity_index,
    to_pinocchio,
    to_slots,
)

# The model terms are already pinned at 128 states by the articulated dynamics rows; these add the
# law on top of them.
STATE_COUNT = 32


def _sample(model, joint_of_body, rng):
    """Measured and desired readings per slot, positions inside each joint's range.

    Desired rates are drawn independently of measured rates, so the Coulomb feedforward is
    exercised with the two signs disagreeing rather than only where they happen to match.
    """
    slot_count = model.nbody - 1
    columns = (
        "joint_positions",
        "joint_velocities",
        "desired_positions",
        "desired_velocities",
        "desired_accelerations",
    )
    sampled = {name: np.zeros((STATE_COUNT, slot_count)) for name in columns}

    for body, joint in joint_of_body.items():
        slot = body - 1
        if model.jnt_limited[joint]:
            low, high = (float(v) for v in model.jnt_range[joint])
        else:
            low, high = (-np.pi, np.pi)
        sampled["joint_positions"][:, slot] = rng.uniform(low, high, size=STATE_COUNT)
        sampled["desired_positions"][:, slot] = rng.uniform(low, high, size=STATE_COUNT)
        sampled["joint_velocities"][:, slot] = rng.uniform(-1.0, 1.0, size=STATE_COUNT)
        sampled["desired_velocities"][:, slot] = rng.uniform(-1.0, 1.0, size=STATE_COUNT)
        sampled["desired_accelerations"][:, slot] = rng.uniform(-1.0, 1.0, size=STATE_COUNT)

    return sampled


def _gains(rng, slot_count, joint_of_body, low, high):
    """One non-negative gain per movable slot, zero on a weld."""
    gains = np.zeros(slot_count)
    for body in joint_of_body:
        gains[body - 1] = rng.uniform(low, high)
    return gains


class _Oracle:
    """Pinocchio's model terms for one MuJoCo model, in slot order."""

    def __init__(self, mujoco_model, path, joint_of_body):
        self.mujoco_model = mujoco_model
        self.joint_of_body = joint_of_body
        self.model, _ = pinocchio_models(path)
        self.data = self.model.createData()
        self.slot_to_velocity = slot_to_velocity_index(mujoco_model, self.model)
        self.slot_count = mujoco_model.nbody - 1
        self.size = self.model.nv
        self.armature = np.array(self.model.armature, dtype=float)
        self.damping = to_slots(
            np.array(self.model.damping, dtype=float), self.slot_to_velocity, self.slot_count
        )
        self.friction_loss = to_slots(
            np.array(self.model.friction, dtype=float), self.slot_to_velocity, self.slot_count
        )

    def to_pinocchio(self, row):
        return to_pinocchio(row, self.slot_to_velocity, self.size)

    def to_slots(self, values):
        return to_slots(values, self.slot_to_velocity, self.slot_count)

    def rnea(self, position, velocity, acceleration):
        """The rigid-body torque with armature, in slot order."""
        return self.to_slots(
            pin.rnea(
                self.model,
                self.data,
                self.to_pinocchio(position),
                self.to_pinocchio(velocity),
                self.to_pinocchio(acceleration),
            ).copy()
        )

    def bias(self, position, velocity):
        """`C(q,q̇)·q̇ + G(q) + damping⊙q̇ + friction_loss⊙sign(q̇)`, the crate's `bias_torque`."""
        rigid = self.rnea(position, velocity, np.zeros(self.slot_count))
        # `np.sign(0.0)` is `0.0`, matching the crate's `coulomb_direction`.
        return rigid + self.damping * velocity + self.friction_loss * np.sign(velocity)

    def gravity(self, position):
        """`G(q)`, the crate's `gravity_torque`: RNEA at rest, where friction contributes nothing."""
        zeros = np.zeros(self.slot_count)
        return self.rnea(position, zeros, zeros)

    def tool_joint(self, tool_slot):
        name = mujoco.mj_id2name(
            self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_of_body[tool_slot + 1]
        )
        return self.model.getJointId(name)

    def tool_pose(self, position, tool_slot):
        pin.forwardKinematics(self.model, self.data, self.to_pinocchio(position))
        return self.data.oMi[self.tool_joint(tool_slot)].copy()

    def body_jacobian(self, position, tool_slot):
        """The tool-frame Jacobian, `[v; ω]` rows, one column per slot and zero on a weld."""
        joint = self.tool_joint(tool_slot)
        local = pin.computeJointJacobian(
            self.model, self.data, self.to_pinocchio(position), joint
        ).copy()
        by_slot = np.zeros((6, self.slot_count))
        for slot, index in self.slot_to_velocity.items():
            by_slot[:, slot] = local[:, index]
        return by_slot


def _assert_jacobian_convention(oracle, position, tool_slot):
    """Pinocchio's `LOCAL` Jacobian against a finite difference of the tool pose.

    Decides that `computeJointJacobian` returns the body-frame Jacobian in `[v; ω]` order, rather
    than assuming it: column `i` must be `log(X(q)⁻¹·X(q + h·eᵢ))/h`.
    """
    analytic = oracle.body_jacobian(position, tool_slot)
    here = oracle.tool_pose(position, tool_slot)
    step = 1e-6
    for slot, _ in oracle.slot_to_velocity.items():
        nudged = position.copy()
        nudged[slot] += step
        ahead = oracle.tool_pose(nudged, tool_slot)
        nudged[slot] -= 2.0 * step
        behind = oracle.tool_pose(nudged, tool_slot)
        difference = (pin.log6(here.actInv(ahead)).vector - pin.log6(here.actInv(behind)).vector) / (
            2.0 * step
        )
        np.testing.assert_allclose(analytic[:, slot], difference, rtol=1e-7, atol=1e-7)

    # The crate's frames are MuJoCo's body frames; the oracle's are Pinocchio's joint frames. They
    # have to be the same frame or the Jacobian above belongs to a different tool.
    data = mujoco.MjData(oracle.mujoco_model)
    data.qpos[:] = oracle.mujoco_model.qpos0
    for body, joint in oracle.joint_of_body.items():
        data.qpos[oracle.mujoco_model.jnt_qposadr[joint]] = position[body - 1]
    mujoco.mj_forward(oracle.mujoco_model, data)
    body = tool_slot + 1
    np.testing.assert_allclose(here.translation, data.xpos[body], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        here.rotation, data.xmat[body].reshape(3, 3), rtol=1e-9, atol=1e-9
    )


def _write(out, case, meta, inputs, expected, label, equation, operations):
    schema.write_fixture(
        out,
        "control",
        case,
        meta,
        {"f64": schema.tol(1e-11, 1e-11)},
        inputs,
        expected,
        equation=equation,
        operations=[f"{label} {operation}" for operation in operations],
    )


def _joint_space_case(law, oracle, sampled, extra_inputs):
    """One joint-space law over every sampled state, with the model block it is stated on."""
    inputs, _ = model_inputs(oracle.mujoco_model)
    for name, values in sampled.items():
        inputs[name] = schema.matrix(values)  # K x N
    inputs.update(extra_inputs)

    torques = [
        law(
            sampled["joint_positions"][state],
            sampled["joint_velocities"][state],
            sampled["desired_positions"][state],
            sampled["desired_velocities"][state],
            sampled["desired_accelerations"][state],
        )
        for state in range(STATE_COUNT)
    ]
    return inputs, {"torques": schema.matrix(torques)}  # K x N


def _computed_torque(out, meta, case, label, oracle, rng):
    """`τ = H·a_ref + C·q̇ + G + armature⊙a_ref + damping⊙q̇ + friction⊙sign(q̇_d)`."""
    sampled = _sample(oracle.mujoco_model, oracle.joint_of_body, rng)
    position_gains = _gains(rng, oracle.slot_count, oracle.joint_of_body, 25.0, 400.0)
    velocity_gains = _gains(rng, oracle.slot_count, oracle.joint_of_body, 5.0, 40.0)

    def law(position, velocity, desired_position, desired_velocity, desired_acceleration):
        reference = (
            desired_acceleration
            + velocity_gains * (desired_velocity - velocity)
            + position_gains * (desired_position - position)
        )
        return (
            oracle.rnea(position, velocity, reference)
            + oracle.damping * velocity
            + oracle.friction_loss * np.sign(desired_velocity)
        )

    inputs, expected = _joint_space_case(
        law, oracle, sampled,
        {
            "position_gains": schema.vector(position_gains),  # N
            "velocity_gains": schema.vector(velocity_gains),  # N
        },
    )
    _write(
        out, case, meta, inputs, expected, label,
        "τ = H(q)·a_ref + C(q,q̇)·q̇ + G(q) + armature⊙a_ref + damping⊙q̇ + frictionloss⊙sign(q̇_d), "
        "a_ref = q̈_d + kd⊙(q̇_d − q̇) + kp⊙(q_d − q)",
        [f"computed torque, {STATE_COUNT} states: (q, q̇, reference) -> τ"],
    )


def _joint_impedance(out, meta, case, label, oracle, rng):
    """`τ = k⊙e + d⊙ė + C·q̇ + G + damping⊙q̇ + friction⊙sign(q̇_d)`."""
    sampled = _sample(oracle.mujoco_model, oracle.joint_of_body, rng)
    stiffness = _gains(rng, oracle.slot_count, oracle.joint_of_body, 0.0, 120.0)
    damping = _gains(rng, oracle.slot_count, oracle.joint_of_body, 0.0, 30.0)

    def law(position, velocity, desired_position, desired_velocity, _desired_acceleration):
        feedback = stiffness * (desired_position - position) + damping * (
            desired_velocity - velocity
        )
        correction = oracle.friction_loss * (
            np.sign(desired_velocity) - np.sign(velocity)
        )
        return feedback + oracle.bias(position, velocity) + correction

    inputs, expected = _joint_space_case(
        law, oracle, sampled,
        {
            "stiffness": schema.vector(stiffness),  # N
            "damping": schema.vector(damping),  # N
        },
    )
    _write(
        out, case, meta, inputs, expected, label,
        "τ = k⊙(q_d − q) + d⊙(q̇_d − q̇) + C(q,q̇)·q̇ + G(q) + damping⊙q̇ + frictionloss⊙sign(q̇_d)",
        [f"joint impedance, {STATE_COUNT} states: (q, q̇, reference) -> τ"],
    )


def _joint_pd(out, meta, case, label, oracle, rng):
    """`τ = kp⊙e + kd⊙ė + G(q)`, gravity compensation on."""
    sampled = _sample(oracle.mujoco_model, oracle.joint_of_body, rng)
    position_gains = _gains(rng, oracle.slot_count, oracle.joint_of_body, 50.0, 600.0)
    velocity_gains = _gains(rng, oracle.slot_count, oracle.joint_of_body, 5.0, 50.0)

    def law(position, velocity, desired_position, desired_velocity, _desired_acceleration):
        return (
            position_gains * (desired_position - position)
            + velocity_gains * (desired_velocity - velocity)
            + oracle.gravity(position)
        )

    inputs, expected = _joint_space_case(
        law, oracle, sampled,
        {
            "position_gains": schema.vector(position_gains),  # N
            "velocity_gains": schema.vector(velocity_gains),  # N
        },
    )
    _write(
        out, case, meta, inputs, expected, label,
        "τ = kp⊙(q_d − q) + kd⊙(q̇_d − q̇) + G(q)",
        [f"gravity-compensated joint PD, {STATE_COUNT} states: (q, q̇, reference) -> τ"],
    )


def _cartesian_impedance(out, meta, case, label, oracle, rng, tool_slot):
    """`τ = Jᵀ·(k⊙e + d⊙e_twist) + C·q̇ + G + damping⊙q̇ + friction⊙sign(q̇)`."""
    sampled = _sample(oracle.mujoco_model, oracle.joint_of_body, rng)
    positions = sampled["joint_positions"]
    velocities = sampled["joint_velocities"]

    _assert_jacobian_convention(oracle, positions[0], tool_slot)

    stiffness = np.concatenate(
        [rng.uniform(200.0, 1200.0, size=3), rng.uniform(10.0, 80.0, size=3)]
    )
    damping = 2.0 * np.sqrt(stiffness)

    # A target near the tool's own pose, so the law is exercised on a small error rather than on a
    # pose the arm could never reach.
    desired_poses = []
    desired_twists = []
    pose_errors = []
    jacobians = []
    torques = []
    for state in range(STATE_COUNT):
        position = positions[state]
        velocity = velocities[state]
        here = oracle.tool_pose(position, tool_slot)
        offset = np.concatenate(
            [rng.uniform(-0.02, 0.02, size=3), rng.uniform(-0.05, 0.05, size=3)]
        )
        # Renormalized through a quaternion: composing down the Franka's chain leaves the
        # rotation orthonormal only to about 2e-14, and the fixture has to carry a proper one.
        displaced = here * pin.exp6(pin.Motion(offset))
        target = pin.SE3(
            pin.Quaternion(displaced.rotation).normalized().toRotationMatrix(),
            displaced.translation,
        )
        twist = np.concatenate(
            [rng.uniform(-0.1, 0.1, size=3), rng.uniform(-0.2, 0.2, size=3)]
        )

        error = pin.log6(here.actInv(target)).vector
        jacobian = oracle.body_jacobian(position, tool_slot)
        twist_error = twist - jacobian @ velocity
        wrench = stiffness * error + damping * twist_error
        torque = jacobian.T @ wrench + oracle.bias(position, velocity)

        desired_poses.append(target.homogeneous.reshape(16))
        desired_twists.append(twist)
        pose_errors.append(error)
        jacobians.append(jacobian.reshape(6 * oracle.slot_count))
        torques.append(torque)

    inputs, _ = model_inputs(oracle.mujoco_model)
    inputs["joint_positions"] = schema.matrix(positions)  # K x N
    inputs["joint_velocities"] = schema.matrix(velocities)  # K x N
    inputs["tool_index"] = schema.integer(tool_slot)
    inputs["desired_poses"] = schema.matrix(desired_poses)  # K x 16, row-major 4x4
    inputs["desired_twists"] = schema.matrix(desired_twists)  # K x 6, [v; w]
    inputs["stiffness"] = schema.vector(stiffness)  # 6
    inputs["damping"] = schema.vector(damping)  # 6
    inputs["frame"] = schema.string("body")

    expected = {
        "torques": schema.matrix(torques),  # K x N
        "pose_errors": schema.matrix(pose_errors),  # K x 6, tool frame
        "jacobians": schema.matrix(jacobians),  # K x (6*N), row-major 6 x N
    }
    _write(
        out, case, meta, inputs, expected, label,
        "τ = Jᵀ·(k⊙(X⁻¹·X_d).log() + d⊙(twist_d − J·q̇)) + C(q,q̇)·q̇ + G(q) + damping⊙q̇ "
        "+ frictionloss⊙sign(q̇)",
        [
            f"Cartesian impedance, {STATE_COUNT} states: (q, q̇, reference) -> τ",
            f"tool-frame Jacobian, {STATE_COUNT} states: q -> J(q)",
        ],
    )


def run(out, seed):
    meta = schema.metadata(
        "articulated_control",
        seed,
        f"{STATE_COUNT} states per case; measured and desired q uniform in each joint's range or "
        "[-pi, pi], drawn independently so the position error is never small; measured and desired "
        "qdot uniform on [-1, 1], drawn independently so the Coulomb feedforward is exercised with "
        "the two signs disagreeing; qddot_d uniform on [-1, 1]. Gains are uniform per joint. The "
        "Cartesian target is the tool's own pose displaced by a twist uniform on [-0.02, 0.02] m "
        "and [-0.05, 0.05] rad. The Franka is edited as the articulated dynamics cases edit it, "
        f"with joint frictionloss set to {FRANKA_FRICTION_LOSS}.",
        libraries=("pin", "mujoco", "numpy"),
        reference="Pinocchio {pin}",
    )
    with tempfile.TemporaryDirectory() as directory:
        pendulum_model, pendulum_path = double_pendulum_model(directory, "control_double_pendulum")
        assert_zero_reference(pendulum_model)
        _, pendulum_joints = model_inputs(pendulum_model)
        pendulum = _Oracle(pendulum_model, pendulum_path, pendulum_joints)

        franka, franka_path = franka_model(directory, "control_franka_panda")
        assert_zero_reference(franka)
        _, franka_joints = model_inputs(franka)
        panda = _Oracle(franka, franka_path, franka_joints)

        # The seventh arm joint's body: the wrist both libraries carry, and the last slot before
        # the welded hand.
        franka_tool = 7

        _computed_torque(
            out, meta, "computed_torque_double_pendulum", "double pendulum",
            pendulum, np.random.default_rng(seed),
        )
        _computed_torque(
            out, meta, "computed_torque_franka_panda", "Franka Panda",
            panda, np.random.default_rng(seed + 1),
        )
        _joint_impedance(
            out, meta, "joint_impedance_franka_panda", "Franka Panda",
            panda, np.random.default_rng(seed + 2),
        )
        _joint_pd(
            out, meta, "joint_pd_franka_panda", "Franka Panda",
            panda, np.random.default_rng(seed + 3),
        )
        _cartesian_impedance(
            out, meta, "cartesian_impedance_franka_panda", "Franka Panda",
            panda, np.random.default_rng(seed + 4), franka_tool,
        )
