use crate::linear_algebra::{Matrix, PivotedQr, Vector};
use crate::optimization::trust_region::determine_lambda_and_parameter_update;

// A full-rank 4x3 Jacobian and residual for the trust-region tests.
fn sample_jacobian() -> (Matrix<4, 3>, Vector<4>) {
    let j = Matrix::<4, 3>::new([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 1.0, 1.0],
        [1.0, 0.0, 2.0],
    ]);
    let residual = Vector::new([1.0, 2.0, 3.0, 4.0]);
    (j, residual)
}

#[test]
fn lmpar_accepts_gauss_newton_inside_region() {
    let (j, b) = sample_jacobian();
    let diag = [1.0, 1.0, 1.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);

    // A trust region larger than the Gauss-Newton step keeps the step undamped.
    let result = determine_lambda_and_parameter_update(&dls, &diag, 100.0, 0.0);
    assert_eq!(result.lambda, 0.0);

    let (gn, _) = dls.solve_with_zero_diagonal();
    for i in 0..3 {
        assert!((result.step[i] - gn[i]).abs() < 1e-12);
    }
}

#[test]
fn lmpar_hits_trust_region_boundary() {
    let (j, b) = sample_jacobian();
    let diag = [1.0, 1.0, 1.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);

    // Shrink the region below the Gauss-Newton length so damping is required.
    let (gn, _) = dls.solve_with_zero_diagonal();
    let delta = 0.5 * gn.norm();
    let result = determine_lambda_and_parameter_update(&dls, &diag, delta, 0.0);

    // Damping is positive and the step lands within 10% of the boundary.
    assert!(result.lambda > 0.0);
    let step_norm = result.step.norm();
    assert!((step_norm - delta).abs() <= 0.1 * delta);

    // The step solves the damped normal equations (JᵀJ + λI) p = Jᵀb (D = I here).
    let jtj = j.transpose() * j;
    let jtb = j.transpose() * b;
    let lhs =
        Matrix::<3, 3>::from_fn(|r, c| jtj[(r, c)] + if r == c { result.lambda } else { 0.0 })
            * result.step;
    for i in 0..3 {
        assert!((lhs[i] - jtb[i]).abs() < 1e-10);
    }
}

#[test]
fn lmpar_stronger_damping_shortens_step() {
    let (j, b) = sample_jacobian();
    let diag = [1.0, 1.0, 1.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let (gn, _) = dls.solve_with_zero_diagonal();

    // A tighter trust region yields a larger λ and a shorter step.
    let loose = determine_lambda_and_parameter_update(&dls, &diag, 0.6 * gn.norm(), 0.0);
    let tight = determine_lambda_and_parameter_update(&dls, &diag, 0.3 * gn.norm(), 0.0);
    assert!(tight.lambda > loose.lambda);
    assert!(tight.step.norm() < loose.step.norm());
}
