//! The Levenberg-Marquardt trust-region parameter search (MINPACK `lmpar`).

use crate::linear_algebra::Vector;
use crate::linear_algebra::qr::{DampedLeastSquares, enorm, max, min};
use crate::scalar::Numeric;

/// The damping parameter and step produced by the trust-region search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LmParameter<const N: usize, T = f64> {
    /// The Levenberg damping parameter λ.
    pub lambda: T,
    /// The step `p` solving `(JᵀJ + λD²) p = Jᵀb`.
    pub step: Vector<N, T>,
}

/// Finds the damping `λ` and step `p` with `‖D·p‖ ≈ delta` (MINPACK `lmpar`).
///
/// Runs a bounded Newton iteration on `λ` (at most 10 steps), reusing the one factorization in
/// `dls` for every trial. `delta` must be positive and every entry of `diag` must be positive.
pub(crate) fn determine_lambda_and_parameter_update<const N: usize, T: Numeric>(
    dls: &DampedLeastSquares<N, T>,
    diag: &[T; N],
    delta: T,
    initial_lambda: T,
) -> LmParameter<N, T> {
    let dwarf = T::MIN_POSITIVE;
    let p1 = T::from_f64(0.1);
    let p001 = T::from_f64(0.001);

    let scale = |v: &Vector<N, T>| -> [T; N] {
        core::array::from_fn(|j| diag[j] * v.get(j).copied().unwrap_or(T::ZERO))
    };

    // Gauss-Newton direction and its scaled length.
    let (gauss_newton, _) = dls.solve_with_zero_diagonal();
    let full_rank = dls.is_non_singular();
    let mut dxnorm = enorm(&scale(&gauss_newton));
    let mut fp = dxnorm - delta;

    // If the Gauss-Newton step already sits inside the trust region, take it undamped.
    if fp <= p1 * delta {
        return LmParameter {
            lambda: T::ZERO,
            step: gauss_newton,
        };
    }

    // Lower bound on λ, available only when the Jacobian has full rank.
    let mut parl = T::ZERO;
    if full_rank {
        let scaled = scale(&gauss_newton);
        let mut w: [T; N] = core::array::from_fn(|j| {
            diag[dls.permutation[j]] * (scaled[dls.permutation[j]] / dxnorm)
        });
        // Forward-solve Rᵀ w = w.
        for j in 0..N {
            let mut sum = T::ZERO;
            for (i, &wi) in w.iter().enumerate().take(j) {
                sum += dls.r.get(i, j).copied().unwrap_or(T::ZERO) * wi;
            }
            let diag_r = dls.r.get(j, j).copied().unwrap_or(T::ONE);
            if let Some(slot) = w.get_mut(j) {
                *slot = (*slot - sum) / diag_r;
            }
        }
        let temp = enorm(&w);
        parl = ((fp / delta) / temp) / temp;
    }

    // Upper bound on λ, from the scaled gradient.
    let w: [T; N] = core::array::from_fn(|j| {
        let mut sum = T::ZERO;
        for i in 0..=j {
            sum += dls.r.get(i, j).copied().unwrap_or(T::ZERO) * dls.qt_b[i];
        }
        sum / diag[dls.permutation[j]]
    });
    let gnorm = enorm(&w);
    let mut paru = gnorm / delta;
    if paru == T::ZERO {
        paru = dwarf / min(delta, p1);
    }

    // Clamp the starting λ into the bracket.
    let mut par = min(max(initial_lambda, parl), paru);
    if par == T::ZERO {
        par = gnorm / dxnorm;
    }

    let mut step = gauss_newton;
    for iter in 1..=10 {
        if par == T::ZERO {
            par = max(dwarf, p001 * paru);
        }

        // Solve the damped system with √par·diag.
        let sqrt_par = par.sqrt();
        let scaled_diag: [T; N] = core::array::from_fn(|j| sqrt_par * diag[j]);
        let (x, cholesky) = dls.solve_with_diagonal(&scaled_diag);
        step = x;

        let scaled_step = scale(&step);
        dxnorm = enorm(&scaled_step);
        let previous_fp = fp;
        fp = dxnorm - delta;

        // Accept when close to the boundary, on the rank-deficient plateau, or after 10 tries.
        if fp.abs() <= p1 * delta
            || (parl == T::ZERO && fp <= previous_fp && previous_fp < T::ZERO)
            || iter == 10
        {
            break;
        }

        // Newton correction on λ, using the Cholesky factor of the damped system.
        let rhs: [T; N] = core::array::from_fn(|j| {
            diag[dls.permutation[j]] * (scaled_step[dls.permutation[j]] / dxnorm)
        });
        let temp = enorm(&cholesky.solve(rhs));
        if temp == T::ZERO {
            break;
        }
        let parc = ((fp / delta) / temp) / temp;

        // Tighten the bracket by the sign of the residual, then improve λ.
        if fp > T::ZERO {
            parl = max(parl, par);
        } else {
            paru = min(paru, par);
        }
        par = max(parl, par + parc);
    }

    LmParameter { lambda: par, step }
}
