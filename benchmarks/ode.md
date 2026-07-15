# ODE integrators accuracy

The ODE integrators are tested against scipy (`solve_ivp`). Each row integrates `y' = f(t, y)` with
the adaptive `Rk45` solver and checks the trajectory against scipy within the listed tolerance.

<!-- BEGIN generated: accuracy -->
| Operation | Equation | Tolerance | Tested Against |
| --- | --- | --- | --- |
| Exponential decay, RK45 | y' = -y | 1e-8 | SciPy solve_ivp 1.14.1 |
| Harmonic oscillator, RK45 | y' = [y₁, -y₀] | 1e-8 | SciPy solve_ivp 1.14.1 |
| Van der Pol (μ=1), RK45 | y' = [y₁, (1 - y₀²)·y₁ - y₀] | 1e-7 | SciPy solve_ivp 1.14.1 |
| Two-body orbit, RK45 | y' = [vₓ, vᵧ, -x/r³, -y/r³] | 1e-7 | SciPy solve_ivp 1.14.1 |
<!-- END generated -->
