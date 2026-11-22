# Franka Emika Panda model

`panda.xml` and everything in `assets/` are copied unmodified from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), directory
`franka_emika_panda`. They are licensed under the Apache License 2.0, © 2022 Franka Emika GmbH; the
full licence text sits beside them in `LICENSE`.

It is used here as a test input: a seven-joint arm whose forward kinematics this repository checks
against MuJoCo's own solve of the same file. The meshes are vendored even though the check reads
only joint geometry, because MuJoCo's compiler loads every asset a file names before it will report
anything.
