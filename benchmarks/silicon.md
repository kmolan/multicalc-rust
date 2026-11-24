# Silicon benchmarks

Warm time of each check of the HILShare apps ([`tools/hilshare`](../tools/hilshare)) on three
microcontrollers. Each run checks each answer against a golden or an identity. **The numbers are
board-dependent.** The boards are not in CI, so CI does not guard this table.

Regenerate from the logs of Board runs:

```sh
gh run download <run-id> --pattern 'hilshare-multicalc-hilshare-*' --dir <run directory>
cargo run -p multicalc-qa --bin gen_silicon_table -- --commit <sha> --date <YYYY-MM-DD> <run directory>...
```

<!-- BEGIN generated: silicon -->
Commit `936cbf6`, 3 Board runs, 2026-11-24. Toolchain 1.92, `opt-level = 3`. The
Nucleo-F411RE and the Raspberry Pi Pico 2 run `f32` on the FPU; `f64` is software on each
board, and the XIAO ESP32-C3 has no FPU. Each cell is the best and the worst warm time, net of
the round trip of the cycle counter.

| Board | Core | Clock | Kernel |
|---|---|---:|---|
| Nucleo-F411RE | Cortex-M4F | 100 MHz | `0.3.1` |
| Raspberry Pi Pico 2 | Cortex-M33 | 150 MHz | `0.3.0` |
| XIAO ESP32-C3 | RV32IMC | 160 MHz | `0.2.0` |

| App | Check | What it runs | Scalar | Nucleo-F411RE | Raspberry Pi Pico 2 | XIAO ESP32-C3 |
|---|---|---|---|---:|---:|---:|
| `arm` | `vector_dot` | Dot product, 4 elements | `f64` | 16.96 µs / 17.66 µs | 14.76 µs / 25.41 µs | 6.62 µs / 13.83 µs |
| `arm` | `svd_singular_values` | SVD, 3×3 | `f64` | 1.14 ms / 1.14 ms | 787.31 µs / 870.05 µs | 715.47 µs / 726.90 µs |
| `arm` | `autodiff_derivative` | Forward-mode derivative of `x³` | `f64` | 1.79 µs / 2.68 µs | 7.74 µs / 22.99 µs | 3.45 µs / 9.98 µs |
| `arm` | `pid_step` | 101 PID updates with a first-order plant | `f64` | 3.03 ms / 3.03 ms | 1.68 ms / 1.70 ms | 1.61 ms / 1.61 ms |
| `arm` | `error_path_returns_err` | Singular LU and indefinite Cholesky return `Err` | `f64` | 26.19 µs / 26.77 µs | 48.63 µs / 68.14 µs | 13.44 µs / 13.95 µs |
| `arm` | `franka_inverse_dynamics` | Franka Panda RNEA, 11 slots: `(q, q̇, q̈) → τ` | `f64` | 12.18 ms / 12.21 ms | 10.24 ms / 10.37 ms | 7.42 ms / 7.43 ms |
| `arm` | `franka_mass_matrix` | Franka Panda CRBA: `q → H(q)` | `f64` | 11.48 ms / 11.51 ms | 8.49 ms / 8.56 ms | 6.75 ms / 6.77 ms |
| `arm` | `franka_forward_dynamics` | Franka Panda ABA: `(q, q̇, τ) → q̈` | `f64` | 15.99 ms / 16.02 ms | 12.69 ms / 13.10 ms | 10.41 ms / 10.43 ms |
| `arm` | `arm_control_tick_f32` | Franka Panda tick: computed torque and momentum observer | `f32` | 700.98 µs / 701.39 µs | 918.52 µs / 946.49 µs | 6.91 ms / 6.92 ms |
| `arm` | `franka_inverse_dynamics_derivative` | Exact `∂τ/∂q̈ · eᵢ` by one RNEA pass | `Dual<f64>` | 30.79 ms / 30.81 ms | 22.63 ms / 23.19 ms | 19.08 ms / 19.42 ms |
| `drone` | `drone_control_tick_f32` | Skydio X2 tick: ESKF predict and update, LQR, thrust, attitude, mixer | `f32` | 1.86 ms / 1.86 ms | 2.25 ms / 2.26 ms | 16.53 ms / 16.69 ms |
| `drone` | `attitude_and_mixer_f32` | Skydio X2 attitude controller and mixer | `f32` | 16.09 µs / 16.66 µs | 75.76 µs / 104.78 µs | 267.93 µs / 278.65 µs |
| `drone` | `eskf_trajectory` | 15-state ESKF: 200 IMU steps and 30 updates | `f64` | 4.52 s / 4.52 s | 2.74 s / 2.74 s | 2.24 s / 2.24 s |
| `mobile_and_policy` | `mobile_robot_tick_f32` | Mobile robot tick: 5-state EKF and pure pursuit | `f32` | 895.44 µs / 896.52 µs | 752.34 µs / 781.10 µs | 1.93 ms / 1.94 ms |
| `mobile_and_policy` | `mlp_policy_forward` | MLP policy, 22→64→64→4 | `f64` | 16.46 ms / 16.46 ms | 15.29 ms / 15.58 ms | 10.64 ms / 10.65 ms |
| `mobile_and_policy` | `mlp_policy_forward_f32` | MLP policy, 22→64→64→4 | `f32` | 531.75 µs / 532.57 µs | 1.68 ms / 1.73 ms | 6.11 ms / 6.12 ms |
<!-- END generated -->
