# Silicon benchmarks

Time of each check on three microcontrollers. Each run checks each answer against a golden or an
identity.

<!-- BEGIN generated: silicon -->
Commit `5daaca7`, 2026-11-24, 3 runs. Rust 1.92, `opt-level = 3`. `f32` runs on the FPU
of the Nucleo-F411RE and the Raspberry Pi Pico 2; `f64` runs in software on each board, and the
XIAO ESP32-C3 has no FPU. Each cell is the best and the worst time of seven repeats.

| Board | Core | Clock |
|---|---|---:|
| Nucleo-F411RE | Cortex-M4F | 100 MHz |
| Raspberry Pi Pico 2 | Cortex-M33 | 150 MHz |
| XIAO ESP32-C3 | RV32IMC | 160 MHz |

| What it runs | Scalar | Nucleo-F411RE | Raspberry Pi Pico 2 | XIAO ESP32-C3 |
|---|---|---:|---:|---:|
| Dot product, 4 elements | `f64` | 16.96 µs / 17.66 µs | 22.73 µs / 38.99 µs | 6.58 µs / 12.26 µs |
| SVD, 3×3 | `f64` | 1.14 ms / 1.14 ms | 854.26 µs / 968.09 µs | 726.35 µs / 739.71 µs |
| Forward-mode autodiff of `x³` | `f64` | 1.79 µs / 2.68 µs | 11.21 µs / 20.72 µs | 3.43 µs / 11.19 µs |
| 101 PID updates with a first-order plant | `f64` | 3.03 ms / 3.03 ms | 1.68 ms / 1.70 ms | 1.61 ms / 1.61 ms |
| Singular LU and indefinite Cholesky return `Err` | `f64` | 26.19 µs / 26.77 µs | 33.07 µs / 62.26 µs | 13.40 µs / 15.46 µs |
| Franka Panda inverse dynamics (RNEA): `(q, q̇, q̈) → τ` | `f64` | 12.18 ms / 12.21 ms | 9.16 ms / 9.43 ms | 7.32 ms / 7.34 ms |
| Franka Panda mass matrix (CRBA): `q → H(q)` | `f64` | 11.48 ms / 11.51 ms | 8.54 ms / 8.80 ms | 6.77 ms / 6.78 ms |
| Franka Panda forward dynamics (ABA): `(q, q̇, τ) → q̈` | `f64` | 15.99 ms / 16.02 ms | 12.88 ms / 13.19 ms | 10.46 ms / 10.48 ms |
| Franka Panda control tick: computed torque and momentum observer | `f32` | 700.98 µs / 701.39 µs | 874.30 µs / 1.01 ms | 7.10 ms / 7.11 ms |
| Franka Panda exact `∂τ/∂q̈` column, one `Dual` RNEA pass | `Dual<f64>` | 30.79 ms / 30.81 ms | 22.61 ms / 23.12 ms | 19.17 ms / 19.92 ms |
| Skydio X2 control tick: ESKF, LQR, thrust, attitude, mixer | `f32` | 1.86 ms / 1.86 ms | 2.23 ms / 2.25 ms | 16.40 ms / 16.57 ms |
| Skydio X2 attitude control and mixer | `f32` | 16.06 µs / 16.63 µs | 73.81 µs / 118.34 µs | 296.95 µs / 303.91 µs |
| Mobile-robot control tick: 5-state EKF and pure pursuit | `f32` | 895.44 µs / 896.52 µs | 761.30 µs / 784.65 µs | 1.93 ms / 1.94 ms |
| MLP policy, 22→64→64→4 | `f64` | 16.47 ms / 16.48 ms | 15.29 ms / 15.54 ms | 10.64 ms / 10.65 ms |
| MLP policy, 22→64→64→4 | `f32` | 531.81 µs / 532.63 µs | 1.67 ms / 1.73 ms | 6.12 ms / 6.12 ms |
<!-- END generated -->
