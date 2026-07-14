# ODE integrator benchmarks

Results for the [`ode`](ode.rs) suite (`cargo bench -- ode`). The suite runs the fixed-step
[`Rk4`] and adaptive [`Rk45`] (Dormand–Prince 5(4)) integrators on three real dynamical systems,
each solved by both methods so the two can be compared directly:

- **Acrobot** — a passive two-link planar manipulator (`N = 4`), released horizontally into a
  chaotic gravity swing.
- **Quadrotor attitude** — a torque-free asymmetric rigid body (`N = 7`: quaternion + body rates),
  spun near its intermediate axis so it tumbles (the Dzhanibekov effect).
- **Solar-system N-body** — the Sun plus the four outer planets (`N = 20`, planar), Newtonian
  gravity in AU / year / solar-mass units, integrated for 100 years.

The accuracy table reports how well each method conserves a known invariant; the latency table
reports wall-clock medians on the machine noted in [README.md](README.md).

## Accuracy

None of these systems has a closed-form solution, so accuracy is the drift of a conserved
quantity: `max |Q(t) − Q(0)|` over the trajectory (relative, for the N-body energy). The same
systems run in the [`ode`](../../../demos) demo (`cargo run -p multicalc-demos --example ode`).

| System                         | Invariant             | RK4 drift | RK45 drift | Notes                                                       |
| ------------------------------ | --------------------- | --------- | ---------- | ----------------------------------------------------------- |
| Acrobot (N=4)                  | total energy          | 1.5e-7    | 4.5e-7     | passive chaotic swing; RK4 `dt=1e-3`, RK45 `rtol=1e-8`      |
| Quadrotor attitude (N=7)       | rotational KE         | 9.8e-14   | 1.3e-10    | torque-free tumble; RK4 `dt=1e-3`, RK45 `rtol=1e-9`         |
| Quadrotor attitude (N=7)       | quaternion norm       | 6.0e-13   | 3.7e-9     | drift of \|q\| from 1 (integrated, not renormalized)       |
| Solar-system N-body (N=20)     | total energy (rel.)   | 1.6e-8    | 2.2e-9     | Sun + 4 outer planets over 100 yr; RK45 `rtol=1e-10`        |

The invariants stay put to between 7 and 14 digits. RK4's drift is set by its fixed step; RK45's by
the requested tolerance — so on the tightly-toleranced N-body run RK45 conserves energy an order of
magnitude better than the fixed step, while on the loosely-toleranced acrobot the fixed 1 ms step is
slightly tighter. The quadrotor's rotational energy is conserved to near machine precision because
Euler's equations keep the kinetic energy on a smooth manifold; the quaternion norm drifts a little
more, as the integrator does not renormalize it.

## Latency

Median of criterion's estimate; wall-clock and therefore machine- and build-specific (see the
[environment note](README.md#environment)). RK4 runs a fixed step count; RK45 chooses its own steps
to hit the tolerance, so the two columns are not a like-for-like step comparison.

Operation                                                      | Median time |
-------------------------------------------------------------- | ----------- |
RK4, acrobot (N=4, 10k steps)                                  | 937 µs      |
RK4, quadrotor attitude (N=7, 20k steps)                       | 892 µs      |
RK4, solar-system N-body (N=20, 2k steps)                      | 390 µs      |
RK45, acrobot (N=4, rtol 1e-8)                                 | 302 µs      |
RK45, quadrotor attitude (N=7, rtol 1e-9)                      | 208 µs      |
RK45, solar-system N-body (N=20, rtol 1e-10)                   | 708 µs      |

[`Rk4`]: ../src/ode/rk4.rs
[`Rk45`]: ../src/ode/rk45.rs
