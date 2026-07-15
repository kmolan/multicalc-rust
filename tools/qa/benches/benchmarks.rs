use criterion::criterion_main;

mod calculus;
mod linear_algebra;
mod ode;
mod optimization;
mod root_finding;

criterion_main!(
    calculus::calculus_benches,
    linear_algebra::linear_algebra_benches,
    ode::ode_benches,
    optimization::optimization_benches,
    root_finding::root_finding_benches,
);
