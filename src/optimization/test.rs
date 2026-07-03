use crate::optimization::{MinimizationReport, TerminationReason};

#[test]
fn report_holds_its_fields() {
    let report = MinimizationReport {
        solution: [1.0, 2.0, 3.0],
        objective_function: 0.5,
        evaluations: 7,
        termination: TerminationReason::Ftol,
    };
    assert_eq!(report.solution, [1.0, 2.0, 3.0]);
    assert_eq!(report.objective_function, 0.5);
    assert_eq!(report.evaluations, 7);
    assert_eq!(report.termination, TerminationReason::Ftol);
}
