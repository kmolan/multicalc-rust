"""Spatial-algebra goldens from Pinocchio: carrying motions and forces between frames, the two
cross products, and what a body's mass distribution does to a motion.

Pinocchio is the reference because it stores spatial vectors linear-first, the same way this crate
does, so the numbers line up one to one with nothing reordered on the way in. Every golden is also
worked out again in numpy straight from the definitions, and the two have to agree before anything
is written — so an ordering read the wrong way round fails here, on the machine that generates the
fixtures, rather than pinning a wrong golden that every later comparison agrees with.
"""

import numpy as np
import pinocchio as pin

import schema

SAMPLE_COUNT = 8


def _tol():
    # Host-side f64 compositions of small products; single-precision coverage lives in the
    # in-crate tests rather than here.
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


def _skew(v):
    """The 3x3 matrix that turns a cross product into a matrix product."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _random_quaternion(rng):
    """A uniformly random rotation, stored scalar-first and always the short way round."""
    q = rng.standard_normal(4)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def _random_pose(rng):
    """A random pose as `(translation, quaternion, pin.SE3)`.

    The rotation matrix is built from a quaternion this generator sampled itself, so what order
    Pinocchio stores its own quaternion coefficients in never comes up.
    """
    quaternion = _random_quaternion(rng)
    rotation = _quaternion_to_matrix(quaternion)
    translation = rng.uniform(-1.0, 1.0, 3)
    return translation, quaternion, pin.SE3(rotation, translation)


def _random_inertia(rng):
    """A random mass distribution as `(mass, center_of_mass, rotational_inertia, pin.Inertia)`."""
    # Masses stay inside one order of magnitude: the weighted balance point in `combined` loses
    # precision when two masses differ wildly, and keeping the range narrow means a golden failure
    # is a formula error rather than conditioning.
    mass = rng.uniform(0.2, 5.0)
    center_of_mass = rng.uniform(-0.3, 0.3, 3)
    square = rng.standard_normal((3, 3))
    scattered = 0.05 * (square @ square.T)
    rotational_inertia = 0.5 * (scattered + scattered.T) + 0.02 * np.eye(3)
    return (
        mass,
        center_of_mass,
        rotational_inertia,
        pin.Inertia(mass, center_of_mass, rotational_inertia),
    )


def _inertia_row(mass, center_of_mass, rotational_inertia):
    """The ten numbers a fixture row stores for one mass distribution."""
    return [
        mass,
        center_of_mass[0],
        center_of_mass[1],
        center_of_mass[2],
        rotational_inertia[0][0],
        rotational_inertia[1][1],
        rotational_inertia[2][2],
        rotational_inertia[0][1],
        rotational_inertia[0][2],
        rotational_inertia[1][2],
    ]


def _pose_row(translation, quaternion):
    """The seven numbers a fixture row stores for one pose."""
    return [
        translation[0],
        translation[1],
        translation[2],
        quaternion[0],
        quaternion[1],
        quaternion[2],
        quaternion[3],
    ]


def _inertia_matrix(mass, center_of_mass, rotational_inertia):
    """The 6x6 mass distribution about the frame origin, in `[v; ω]` blocks."""
    offset = _skew(center_of_mass)
    about_origin = rotational_inertia + mass * (
        center_of_mass @ center_of_mass * np.eye(3)
        - np.outer(center_of_mass, center_of_mass)
    )
    block = np.zeros((6, 6))
    block[0:3, 0:3] = mass * np.eye(3)
    block[0:3, 3:6] = -mass * offset
    block[3:6, 0:3] = mass * offset
    block[3:6, 3:6] = about_origin
    return block


def _agree(first, second, context):
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-13, err_msg=context)


def _plucker_transforms(out, meta, seed):
    """Carrying twists and wrenches between frames, and the two 6x6 transforms."""
    rng = np.random.default_rng(seed)
    poses, twists, wrenches = [], [], []
    acted_twists, acted_wrenches = [], []
    inverse_twists, inverse_wrenches = [], []
    motion_adjoints, force_adjoints = [], []

    for _ in range(SAMPLE_COUNT):
        translation, quaternion, pose = _random_pose(rng)
        rotation = _quaternion_to_matrix(quaternion)
        linear, angular = rng.uniform(-2.0, 2.0, 3), rng.uniform(-2.0, 2.0, 3)
        force, torque = rng.uniform(-2.0, 2.0, 3), rng.uniform(-2.0, 2.0, 3)

        acted = pose.act(pin.Motion(linear, angular))
        _agree(acted.angular, rotation @ angular, "acted twist angular")
        _agree(
            acted.linear,
            rotation @ linear + np.cross(translation, rotation @ angular),
            "acted twist linear",
        )

        acted_force = pose.act(pin.Force(force, torque))
        _agree(acted_force.linear, rotation @ force, "acted wrench force")
        _agree(
            acted_force.angular,
            rotation @ torque + np.cross(translation, rotation @ force),
            "acted wrench torque",
        )

        undone = pose.actInv(pin.Motion(linear, angular))
        _agree(undone.angular, rotation.T @ angular, "inverse twist angular")
        _agree(
            undone.linear,
            rotation.T @ (linear - np.cross(translation, angular)),
            "inverse twist linear",
        )

        undone_force = pose.actInv(pin.Force(force, torque))
        _agree(undone_force.linear, rotation.T @ force, "inverse wrench force")
        _agree(
            undone_force.angular,
            rotation.T @ (torque - np.cross(translation, force)),
            "inverse wrench torque",
        )

        motion_adjoint = np.asarray(pose.action)
        by_hand = np.zeros((6, 6))
        by_hand[0:3, 0:3] = rotation
        by_hand[0:3, 3:6] = _skew(translation) @ rotation
        by_hand[3:6, 3:6] = rotation
        _agree(motion_adjoint, by_hand, "motion adjoint")

        force_adjoint = np.asarray(pose.dualAction)
        by_hand = np.zeros((6, 6))
        by_hand[0:3, 0:3] = rotation
        by_hand[3:6, 0:3] = _skew(translation) @ rotation
        by_hand[3:6, 3:6] = rotation
        _agree(force_adjoint, by_hand, "force adjoint")

        poses.append(_pose_row(translation, quaternion))
        twists.append(list(linear) + list(angular))
        wrenches.append(list(force) + list(torque))
        acted_twists.append(list(acted.linear) + list(acted.angular))
        acted_wrenches.append(list(acted_force.linear) + list(acted_force.angular))
        inverse_twists.append(list(undone.linear) + list(undone.angular))
        inverse_wrenches.append(list(undone_force.linear) + list(undone_force.angular))
        motion_adjoints.extend(motion_adjoint.tolist())
        force_adjoints.extend(force_adjoint.tolist())

    schema.write_fixture(
        out,
        "spatial",
        "plucker_transforms",
        meta,
        _tol(),
        {
            "poses": schema.matrix(poses),
            "twists": schema.matrix(twists),
            "wrenches": schema.matrix(wrenches),
        },
        {
            "transformed_twists": schema.matrix(acted_twists),
            "transformed_wrenches": schema.matrix(acted_wrenches),
            "inverse_transformed_twists": schema.matrix(inverse_twists),
            "inverse_transformed_wrenches": schema.matrix(inverse_wrenches),
            "motion_adjoints": schema.matrix(motion_adjoints),
            "force_adjoints": schema.matrix(force_adjoints),
        },
        equation=(
            "X = (R, p) on [v; ω] and [f; τ]: v' = R·v + p×(R·ω), ω' = R·ω ; "
            "f' = R·f, τ' = R·τ + p×(R·f)"
        ),
        operations=[
            "carrying a twist between frames, 8 random poses",
            "carrying a wrench between frames, 8 random poses",
            "carrying a twist back, 8 random poses",
            "carrying a wrench back, 8 random poses",
            "the 6x6 motion and force transform matrices, 8 random poses",
        ],
    )


def _spatial_cross_products(out, meta, seed):
    """The motion and force cross products, and the power product."""
    rng = np.random.default_rng(seed + 1)
    first_twists, second_twists, wrenches = [], [], []
    motion_crosses, force_crosses, powers = [], [], []

    for _ in range(SAMPLE_COUNT):
        first_linear, first_angular = (
            rng.uniform(-2.0, 2.0, 3),
            rng.uniform(-2.0, 2.0, 3),
        )
        second_linear, second_angular = (
            rng.uniform(-2.0, 2.0, 3),
            rng.uniform(-2.0, 2.0, 3),
        )
        force, torque = rng.uniform(-2.0, 2.0, 3), rng.uniform(-2.0, 2.0, 3)

        first = pin.Motion(first_linear, first_angular)
        second = pin.Motion(second_linear, second_angular)
        wrench = pin.Force(force, torque)

        crossed = first.cross(second)
        _agree(
            crossed.linear,
            np.cross(first_angular, second_linear)
            + np.cross(first_linear, second_angular),
            "motion cross linear",
        )
        _agree(
            crossed.angular,
            np.cross(first_angular, second_angular),
            "motion cross angular",
        )

        crossed_force = first.cross(wrench)
        _agree(crossed_force.linear, np.cross(first_angular, force), "force cross force")
        _agree(
            crossed_force.angular,
            np.cross(first_angular, torque) + np.cross(first_linear, force),
            "force cross torque",
        )

        power = float(first_linear @ force + first_angular @ torque)

        first_twists.append(list(first_linear) + list(first_angular))
        second_twists.append(list(second_linear) + list(second_angular))
        wrenches.append(list(force) + list(torque))
        motion_crosses.append(list(crossed.linear) + list(crossed.angular))
        force_crosses.append(list(crossed_force.linear) + list(crossed_force.angular))
        powers.append(power)

    schema.write_fixture(
        out,
        "spatial",
        "spatial_cross_products",
        meta,
        _tol(),
        {
            "first_twists": schema.matrix(first_twists),
            "second_twists": schema.matrix(second_twists),
            "wrenches": schema.matrix(wrenches),
        },
        {
            "motion_crosses": schema.matrix(motion_crosses),
            "force_crosses": schema.matrix(force_crosses),
            "powers": schema.vector(powers),
        },
        equation=(
            "a × b = [ω_a×v_b + v_a×ω_b ; ω_a×ω_b] ; "
            "a ×* w = [ω_a×f ; ω_a×τ + v_a×f] ; a·w = v_a·f + ω_a·τ"
        ),
        operations=[
            "twist crossed with twist, 8 random pairs",
            "twist crossed with wrench, 8 random pairs",
            "the rate a wrench does work, 8 random pairs",
        ],
    )


def _spatial_inertia_algebra(out, meta, seed):
    """A body's 6x6 form, and the momentum, bias wrench, and energy a motion gives it."""
    rng = np.random.default_rng(seed + 2)
    inertia_parameters, twists = [], []
    inertia_matrices, momenta, bias_wrenches, kinetic_energies = [], [], [], []

    for _ in range(SAMPLE_COUNT):
        mass, center_of_mass, rotational_inertia, inertia = _random_inertia(rng)
        linear, angular = rng.uniform(-2.0, 2.0, 3), rng.uniform(-2.0, 2.0, 3)
        flat = np.concatenate([linear, angular])
        motion = pin.Motion(linear, angular)

        block = np.asarray(inertia.matrix())
        _agree(
            block,
            _inertia_matrix(mass, center_of_mass, rotational_inertia),
            "inertia matrix",
        )

        momentum = inertia * motion
        balance_point_velocity = linear + np.cross(angular, center_of_mass)
        by_hand_linear = mass * balance_point_velocity
        _agree(momentum.linear, by_hand_linear, "momentum linear")
        _agree(
            momentum.angular,
            rotational_inertia @ angular + np.cross(center_of_mass, by_hand_linear),
            "momentum angular",
        )

        bias = motion.cross(momentum)
        _agree(bias.linear, np.cross(angular, by_hand_linear), "bias force")
        _agree(
            bias.angular,
            np.cross(angular, momentum.angular) + np.cross(linear, by_hand_linear),
            "bias torque",
        )

        energy = 0.5 * flat @ block @ flat
        _agree(
            energy,
            0.5 * mass * balance_point_velocity @ balance_point_velocity
            + 0.5 * angular @ rotational_inertia @ angular,
            "kinetic energy",
        )

        inertia_parameters.append(
            _inertia_row(mass, center_of_mass, rotational_inertia)
        )
        twists.append(list(linear) + list(angular))
        inertia_matrices.extend(block.tolist())
        momenta.append(list(momentum.linear) + list(momentum.angular))
        bias_wrenches.append(list(bias.linear) + list(bias.angular))
        kinetic_energies.append(float(energy))

    schema.write_fixture(
        out,
        "spatial",
        "spatial_inertia_algebra",
        meta,
        _tol(),
        {
            "inertia_parameters": schema.matrix(inertia_parameters),
            "twists": schema.matrix(twists),
        },
        {
            "inertia_matrices": schema.matrix(inertia_matrices),
            "momenta": schema.matrix(momenta),
            "bias_wrenches": schema.matrix(bias_wrenches),
            "kinetic_energies": schema.vector(kinetic_energies),
        },
        equation=(
            "I·v = [m(v + ω×c) ; I_c·ω + c×(m(v + ω×c))] ; v ×* (I·v) ; ½·vᵀ·I·v"
        ),
        operations=[
            "a body's 6x6 mass distribution, 8 random bodies",
            "the motion a body carries, 8 random bodies",
            "what a body needs to hold its motion, 8 random bodies",
            "the energy a body's motion carries, 8 random bodies",
        ],
    )


def _spatial_inertia_transform_and_composite(out, meta, seed):
    """Reading a body in another frame, and sticking two bodies together."""
    rng = np.random.default_rng(seed + 3)
    first_parameters, second_parameters, poses = [], [], []
    transformed_parameters, combined_parameters = [], []

    for _ in range(SAMPLE_COUNT):
        first_mass, first_center, first_inertia, first = _random_inertia(rng)
        second_mass, second_center, second_inertia, second = _random_inertia(rng)
        translation, quaternion, pose = _random_pose(rng)
        rotation = _quaternion_to_matrix(quaternion)

        moved = pose.act(first)
        _agree(moved.mass, first_mass, "transformed mass")
        _agree(moved.lever, rotation @ first_center + translation, "transformed lever")
        _agree(
            np.asarray(moved.inertia),
            rotation @ first_inertia @ rotation.T,
            "transformed inertia",
        )

        whole = first + second
        total_mass = first_mass + second_mass
        shared_center = (
            first_mass * first_center + second_mass * second_center
        ) / total_mass
        _agree(whole.mass, total_mass, "combined mass")
        _agree(whole.lever, shared_center, "combined lever")
        _agree(
            np.asarray(whole.inertia),
            _shifted(first_mass, first_center, first_inertia, shared_center)
            + _shifted(second_mass, second_center, second_inertia, shared_center),
            "combined inertia",
        )

        first_parameters.append(_inertia_row(first_mass, first_center, first_inertia))
        second_parameters.append(
            _inertia_row(second_mass, second_center, second_inertia)
        )
        poses.append(_pose_row(translation, quaternion))
        transformed_parameters.append(
            _inertia_row(moved.mass, moved.lever, np.asarray(moved.inertia))
        )
        combined_parameters.append(
            _inertia_row(whole.mass, whole.lever, np.asarray(whole.inertia))
        )

    schema.write_fixture(
        out,
        "spatial",
        "spatial_inertia_transform_and_composite",
        meta,
        _tol(),
        {
            "first_inertia_parameters": schema.matrix(first_parameters),
            "second_inertia_parameters": schema.matrix(second_parameters),
            "poses": schema.matrix(poses),
        },
        {
            "transformed_inertia_parameters": schema.matrix(transformed_parameters),
            "combined_inertia_parameters": schema.matrix(combined_parameters),
        },
        equation=(
            "X·I = (m, R·c + p, R·I_c·Rᵀ) ; I₁+I₂ = (m₁+m₂, (m₁c₁+m₂c₂)/(m₁+m₂), "
            "both inertias shifted to the shared balance point)"
        ),
        operations=[
            "a body's mass distribution read in another frame, 8 random bodies",
            "two bodies stuck together, 8 random pairs",
        ],
    )


def _shifted(mass, center_of_mass, rotational_inertia, point):
    """A body's rotational inertia restated about some other point."""
    offset = point - center_of_mass
    return rotational_inertia + mass * (
        offset @ offset * np.eye(3) - np.outer(offset, offset)
    )


def run(out, seed):
    meta = schema.metadata(
        "spatial",
        seed,
        "8 random poses, twists, wrenches and mass distributions per case; "
        "translations on [-1, 1], twist and wrench components on [-2, 2], masses on [0.2, 5], "
        "balance points on [-0.3, 0.3]; rotational inertias are symmetric positive definite by "
        "construction and are not required to be physically realizable, which the algebra does "
        "not depend on",
        libraries=("pin",),
        reference="Pinocchio {pin}",
    )
    _plucker_transforms(out, meta, seed)
    _spatial_cross_products(out, meta, seed)
    _spatial_inertia_algebra(out, meta, seed)
    _spatial_inertia_transform_and_composite(out, meta, seed)
