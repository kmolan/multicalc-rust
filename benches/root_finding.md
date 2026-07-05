# Root-finding benchmarks

Results for the [`root_finding`](root_finding.rs) suite (`cargo bench --bench root_finding`).
The accuracy tables report how close each solver lands to the known root; the latency table
reports wall-clock medians on the machine noted in [README.md](README.md).

## Accuracy

Reported below are the **residual** magnitude at the returned root ($$|f(x)|$$ for the scalar
solvers, $$\lVert F\rVert$$ for the systems) and the **root error**, the distance from the
returned root to the known analytic root. A "0.0" is exact to the last bit; a dash marks a
problem with no closed-form root, where only the residual is meaningful. The iteration count
is the number of steps the solver actually took.

Scalar solvers:

| Problem                                            | Residual | Root error | Iterations | Notes                                                        |
| -------------------------------------------------- | -------- | ---------- | ---------- | ------------------------------------------------------------ |
| Bisection, Wien's displacement root on [1, 10]     | 3e-16    | 0.0        | 50         | Halves to the bracket-width floor; root exact to the last bit |
| Bisection, Kepler $$E - e\sin E = M$$ (e = 0.8)    | 2e-16    | 4e-16      | 47         | Guaranteed [0, π] bracket; eccentric anomaly within 1 ulp    |
| Newton, Wien's displacement root (x0 = 5)          | 3e-16    | 0.0        | 4          | Exact `Dual` derivative; quadratic convergence               |
| Newton, Kepler (e = 0.8, x0 = M)                   | 0.0      | 1e-16      | 7          | Exact derivative; angle to machine precision                 |
| Newton, Colebrook-White friction factor            | 2e-15    | —          | 5          | No closed form; f ≈ 0.018514, residual to machine precision  |
| Newton (finite-difference), Wien's root            | 3e-16    | 0.0        | 4          | Central-difference derivative; matches the autodiff run here |
| Damped Newton, $$x/\sqrt{1+x^2}$$ from x0 = 2      | 0.0      | 0.0        | 6          | Backtracking rescues a start where plain Newton diverges     |

System solver (`NewtonSystem`, exact Jacobian):

| Problem                                            | Residual | Root error | Iterations | Notes                                                        |
| -------------------------------------------------- | -------- | ---------- | ---------- | ------------------------------------------------------------ |
| Two-link inverse kinematics (N = 2)                | 0.0      | 3e-16      | 5          | Joint angles (0.5, 0.8) recovered to ~1 ulp                  |
| Circle ∩ hyperbola (N = 2)                         | 4e-15    | 1e-15      | 6          | Root $$(\sqrt{2+\sqrt3},\ \sqrt{2-\sqrt3})$$                  |
| Chemical equilibrium mass balance (N = 3)          | 6e-17    | 3e-17      | 5          | Mass balance + two equilibria; (0.4, 0.2, 0.4) exact         |

Bisection spends its full halving budget (47-50 steps) to drive the bracket to the width floor,
while Newton and the systems reach machine precision in a handful of steps thanks to quadratic
convergence. On the smooth Wien problem the finite-difference derivative lands on the same root
as autodiff to the last bit. Colebrook-White has no closed-form root, so only its residual is
reported; every other case is known by construction and recovered to within a few ulp. Accuracy
here is bounded by floating-point rounding, not by the algorithm.

## Latency

Median of criterion's estimate; wall-clock and therefore machine- and build-specific (see the
[environment note](README.md#environment)).

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Damped Newton, $$x/\sqrt{1+x^2}$$ (scalar, backtracking)         | 71 ns       |
Newton, Wien's displacement (scalar, exact `Dual`)              | 109 ns      |
Newton, Wien's displacement (scalar, finite difference)          | 130 ns      |
Newton system, circle ∩ hyperbola (2×2)                         | 139 ns      |
Newton, Kepler's equation (scalar, exact `Dual`)               | 143 ns      |
Newton, Colebrook-White friction factor (scalar)                | 183 ns      |
Newton system, chemical equilibrium (3×3)                        | 290 ns      |
Bisection, Kepler's equation on [0, π]                          | 322 ns      |
Bisection, Wien's displacement on [1, 10]                        | 497 ns      |
Newton system, two-link inverse kinematics (2×2)                | 706 ns      |
