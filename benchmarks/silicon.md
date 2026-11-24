# Silicon benchmarks

Time of each check on three microcontrollers. Each run checks each answer against a golden or an
identity.

<!-- BEGIN generated: silicon -->
Commit `936cbf6`, 2026-11-24, 3 runs. Rust 1.92, `opt-level = 3`. `f32` runs on the FPU
of the Nucleo-F411RE and the Raspberry Pi Pico 2; `f64` runs in software on each board, and the
XIAO ESP32-C3 has no FPU. Each cell is the best and the worst time of seven repeats.

| Board | Core | Clock |
|---|---|---:|
| Nucleo-F411RE | Cortex-M4F | 100 MHz |
| Raspberry Pi Pico 2 | Cortex-M33 | 150 MHz |
| XIAO ESP32-C3 | RV32IMC | 160 MHz |

| What it runs | Scalar | Nucleo-F411RE | Raspberry Pi Pico 2 | XIAO ESP32-C3 |
|---|---|---:|---:|---:|
| Dot product, 4 elements | `f64` | 16.96 µs / 17.66 µs | 14.76 µs / 25.41 µs | 6.62 µs / 13.83 µs |
| SVD, 3×3 | `f64` | 1.14 ms / 1.14 ms | 787.31 µs / 870.05 µs | 715.47 µs / 726.90 µs |
| Forward-mode autodiff of `x³` | `f64` | 1.79 µs / 2.68 µs | 7.74 µs / 22.99 µs | 3.45 µs / 9.98 µs |
| 101 PID updates with a first-order plant | `f64` | 3.03 ms / 3.03 ms | 1.68 ms / 1.70 ms | 1.61 ms / 1.61 ms |
| Singular LU and indefinite Cholesky return `Err` | `f64` | 26.19 µs / 26.77 µs | 48.63 µs / 68.14 µs | 13.44 µs / 13.95 µs |
| Franka Panda inverse dynamics (RNEA): `(q, q̇, q̈) → τ` | `f64` | 12.18 ms / 12.21 ms | 10.24 ms / 10.37 ms | 7.42 ms / 7.43 ms |
| Franka Panda mass matrix (CRBA): `q → H(q)` | `f64` | 11.48 ms / 11.51 ms | 8.49 ms / 8.56 ms | 6.75 ms / 6.77 ms |
| Franka Panda forward dynamics (ABA): `(q, q̇, τ) → q̈` | `f64` | 15.99 ms / 16.02 ms | 12.69 ms / 13.10 ms | 10.41 ms / 10.43 ms |
| Franka Panda control tick: computed torque and momentum observer | `f32` | 700.98 µs / 701.39 µs | 918.52 µs / 946.49 µs | 6.91 ms / 6.92 ms |
| Franka Panda exact `∂τ/∂q̈` column, one `Dual` RNEA pass | `Dual<f64>` | 30.79 ms / 30.81 ms | 22.63 ms / 23.19 ms | 19.08 ms / 19.42 ms |
| Skydio X2 control tick: ESKF, LQR, thrust, attitude, mixer | `f32` | 1.86 ms / 1.86 ms | 2.25 ms / 2.26 ms | 16.53 ms / 16.69 ms |
| Skydio X2 attitude control and mixer | `f32` | 16.09 µs / 16.66 µs | 75.76 µs / 104.78 µs | 267.93 µs / 278.65 µs |
| 15-state ESKF over 200 IMU steps and 30 updates | `f64` | 4.52 s / 4.52 s | 2.74 s / 2.74 s | 2.24 s / 2.24 s |
| Mobile-robot control tick: 5-state EKF and pure pursuit | `f32` | 895.44 µs / 896.52 µs | 752.34 µs / 781.10 µs | 1.93 ms / 1.94 ms |
| MLP policy, 22→64→64→4 | `f64` | 16.46 ms / 16.46 ms | 15.29 ms / 15.58 ms | 10.64 ms / 10.65 ms |
| MLP policy, 22→64→64→4 | `f32` | 531.75 µs / 532.57 µs | 1.68 ms / 1.73 ms | 6.11 ms / 6.12 ms |
<!-- END generated -->
