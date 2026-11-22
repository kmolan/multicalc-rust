# MoveIt Panda model

`panda.urdf` is copied unmodified from [moveit_resources](https://github.com/moveit/moveit_resources),
path `panda_description/urdf/panda.urdf`, at commit
[`a035b68`](https://github.com/moveit/moveit_resources/commit/a035b68b49cbd811e9f9595a657a47ba73dfd3b4).
It is licensed under the Apache License 2.0; the full licence text sits beside it in `LICENSE`,
copied from `panda_description/LICENSE` at the same commit. The package's own `package.xml` records
the licence as `BSD`, which disagrees with both the licence file it ships and its README; the
Apache-2.0 terms in `LICENSE` are the ones taken to apply. Upstream describes the model as copied
from Franka Emika's `franka_ros` and adapted, all of it released under Apache-2.0.

It is used here as a test input for the URDF reader: the same robot as the vendored MJCF Franka, so
the two formats can be read side by side. It also covers two things no other model here does — a
joint that follows another joint, on the second gripper finger, and a description that states no
mass anywhere at all.

No meshes are vendored. The file names them by `package://` URI inside its `<visual>` and
`<collision>` blocks, and this repository's reader skips both, so no mesh file is ever looked for.
