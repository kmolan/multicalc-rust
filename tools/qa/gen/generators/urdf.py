"""Model-ingestion goldens from Pinocchio's own parse of the file.

URDF is Pinocchio's home format, so it is the reference here rather than a second reader of the
same file. Pinocchio folds a link joined by a fixed joint into its parent, so the link list is
recovered from the model's frames rather than its joints.
"""

import os
import xml.etree.ElementTree as ElementTree

import numpy as np
import pinocchio as pin

import schema

PANDA = "moveit_resources_panda/panda.urdf"
THIRD_PARTY = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "third_party"
)

# The arm joints, in model order. The two finger joints are held at zero: one of them mimics the
# other, which the Rust side cannot put in a tree, so the chain it compares stops at the hand.
ARM_JOINT_COUNT = 7
# link0 through link8 and the hand — the frames on the chain the Rust side builds.
CHAIN_FRAME_COUNT = 10
CONFIGURATION_COUNT = 9


def _tol():
    # Ingestion is host-side f64 work and never reruns in single precision, so there is no f32
    # entry. Both readers compose the same numbers in the same order, so the bar is f64 noise.
    return {"f64": schema.tol(1e-12, 1e-11)}


def _joint_kind(joint):
    """The single letter this suite uses for a joint kind, from Pinocchio's own name for it."""
    name = joint.shortname()
    if name.startswith("JointModelR"):
        return "R"
    if name.startswith("JointModelP"):
        return "P"
    raise AssertionError(f"no letter for joint kind {name!r}")


def _body_frames(model):
    """Every link, as `(frame_id, name)`, in the order the file declares them.

    Pinocchio drops a fixed joint and attaches its child link to the parent joint, so `model.joints`
    is short of a link each time. The BODY frames keep all of them.
    """
    return [
        (index, frame.name)
        for index, frame in enumerate(model.frames)
        if frame.type == pin.FrameType.BODY
    ]


def _mimic_from_file(path):
    """The `<mimic>` coupling, read from the file rather than from Pinocchio.

    Pinocchio's handling of `<mimic>` has moved across releases, and 4.1.0 leaves both
    `model.mimicking_joints` and `model.mimicked_joints` empty for this file: the following finger
    keeps its own degree of freedom. So the one field it will not report is taken from the text.
    """
    root = ElementTree.parse(path).getroot()
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            return (
                joint.get("name"),
                mimic.get("joint"),
                float(mimic.get("multiplier", 1.0)),
                float(mimic.get("offset", 0.0)),
            )
    raise AssertionError("expected one joint to follow another")


def _panda_tree(out, meta, path, model):
    """The MoveIt Panda's joints, travel and link list, plus its coupled finger."""
    joint_names = list(model.names)[1:]  # index 0 is always "universe"
    joint_kinds = "".join(_joint_kind(model.joints[i]) for i in range(1, model.njoints))
    lower = np.array(model.lowerPositionLimit, dtype=float)
    upper = np.array(model.upperPositionLimit, dtype=float)
    link_names = [name for _, name in _body_frames(model)]
    follower, driver, multiplier, offset = _mimic_from_file(path)

    # Cross-check the hand-derived truths before writing anything, so a wrong golden fails here
    # rather than being pinned and passed on downstream.
    assert model.nq == 9 and model.nv == 9, "seven arm joints and two fingers, one number each"
    assert model.njoints == 10, "nine joints and the universe"
    assert joint_kinds == "RRRRRRRPP"
    assert len(link_names) == 12, "eleven links plus the base"
    assert all(model.inertias[i].mass == 0.0 for i in range(model.njoints)), (
        "the file as published states no <inertial> anywhere"
    )
    assert (follower, driver) == ("panda_finger_joint2", "panda_finger_joint1")
    # Two values read straight off the file's own text.
    np.testing.assert_allclose(lower[0], -2.9671, atol=1e-12)
    np.testing.assert_allclose([lower[3], upper[3]], [-3.1416, 0.0873], atol=1e-12)

    inputs = {"model_file": schema.string(PANDA)}
    expected = {
        "joint_names": schema.string(" ".join(joint_names)),
        "joint_kinds": schema.string(joint_kinds),
        "lower_limits": schema.vector(lower),
        "upper_limits": schema.vector(upper),
        "link_names": schema.string(" ".join(link_names)),
        "body_count": schema.integer(len(link_names)),
        "movable_joint_count": schema.integer(model.njoints - 1),
        # Stated plainly rather than left to be inferred: this file carries no mass at all.
        "links_with_mass": schema.integer(0),
        "mimicking_joint": schema.string(follower),
        "mimic_joint": schema.string(driver),
        "mimic_multiplier": schema.scalar(multiplier),
        "mimic_offset": schema.scalar(offset),
    }
    schema.write_fixture(
        out, "urdf", "panda_urdf_tree",
        meta, _tol(), inputs, expected,
        equation="MoveIt Panda: 12 links, 7 revolute + 2 prismatic, one coupled finger",
        operations=[
            "MoveIt Panda read: link list, joint order and kinds",
            "MoveIt Panda read: travel limits",
            "MoveIt Panda read: the coupled finger joint",
        ],
    )


def _panda_forward_kinematics(out, meta, model, seed):
    """Where every link on the arm ends up, across configurations.

    The strong check, and the one that leans on no assumption about how Pinocchio folds fixed
    joints: it compares world placements of the link frames themselves.
    """
    rng = np.random.default_rng(seed)
    lower = np.array(model.lowerPositionLimit, dtype=float)[:ARM_JOINT_COUNT]
    upper = np.array(model.upperPositionLimit, dtype=float)[:ARM_JOINT_COUNT]

    configurations = [np.zeros(ARM_JOINT_COUNT)]
    while len(configurations) < CONFIGURATION_COUNT:
        configurations.append(rng.uniform(lower, upper))

    data = model.createData()
    frames = _body_frames(model)[:CHAIN_FRAME_COUNT]
    translations = []
    quaternions = []
    for configuration in configurations:
        # The fingers stay at zero; the chain being compared stops at the hand.
        q = np.concatenate([configuration, np.zeros(model.nq - ARM_JOINT_COUNT)])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for frame_id, _ in frames:
            placement = data.oMf[frame_id]
            translations.append(np.array(placement.translation, dtype=float))
            rotation = pin.Quaternion(placement.rotation)
            # Scalar first, matching how the Rust side stores a quaternion.
            quaternions.append(
                np.array([rotation.w, rotation.x, rotation.y, rotation.z], dtype=float)
            )

    assert len(translations) == CONFIGURATION_COUNT * CHAIN_FRAME_COUNT
    assert all(np.all(c >= lower - 1e-12) and np.all(c <= upper + 1e-12) for c in configurations)
    # The zero configuration is the one value in here that can be checked against the file by hand:
    # the hand sits 0.926 m up and 0.088 m forward of the base.
    np.testing.assert_allclose(
        translations[CHAIN_FRAME_COUNT - 1], [0.088, 0.0, 0.926], atol=1e-9
    )

    inputs = {
        "model_file": schema.string(PANDA),
        "configurations": schema.matrix(configurations),
    }
    expected = {
        "link_names": schema.string(" ".join(name for _, name in frames)),
        "translations": schema.matrix(translations),
        "quaternions": schema.matrix(quaternions),
        "frame_count": schema.integer(CHAIN_FRAME_COUNT),
        "configuration_count": schema.integer(CONFIGURATION_COUNT),
    }
    schema.write_fixture(
        out, "urdf", "panda_urdf_forward_kinematics",
        meta, _tol(), inputs, expected,
        equation="MoveIt Panda: world placement of every link on the chain to the hand",
        operations=["MoveIt Panda forward kinematics: 9 configurations x 10 links"],
    )


def run(out, seed):
    meta = schema.metadata(
        "urdf", seed,
        "one committed URDF model; goldens are Pinocchio's own parse of the same file",
        libraries=("pin",),
        reference="Pinocchio {pin}",
    )
    path = os.path.join(THIRD_PARTY, PANDA)
    model = pin.buildModelFromUrdf(path)
    _panda_tree(out, meta, path, model)
    _panda_forward_kinematics(out, meta, model, seed)
