# Skydio X2 model

`x2.xml` and everything in `assets/` are copied unmodified from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), directory `skydio_x2`. They
are licensed under the Apache License 2.0, © 2022 Shadow Robot Company Ltd; the full licence text
sits beside them in `LICENSE`.

It is used here as a test input. The mesh and texture are vendored even though this repository's own
parser never reads a mesh: MuJoCo is the oracle the parse is checked against, and its compiler loads
every asset a file names before it will report anything, whether or not that shape carries mass.
