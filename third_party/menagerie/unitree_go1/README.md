# Unitree Go1 model

`go1.xml` and everything in `assets/` are copied unmodified from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), directory `unitree_go1`.
They are licensed under a BSD 3-Clause License, © 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
("Unitree Robotics"); the full licence text sits beside them in `LICENSE`.

It is used here as a test input: a floating base (the trunk, on a free joint) carrying twelve
articulated hinge joints across four legs — the combined case a single free-jointed body (Skydio X2)
or a fixed-base arm (Franka Panda) does not exercise on its own. This repository checks its parsing,
forward kinematics, and inverse kinematics against MuJoCo's own solve of the same file. The meshes
are vendored even though the checks read only joint geometry, because MuJoCo's compiler loads every
asset a file names before it will report anything.
