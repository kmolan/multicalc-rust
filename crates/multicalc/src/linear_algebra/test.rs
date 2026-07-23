use crate::linear_algebra::qr::{PivotedQr, enorm, max, min};
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::sync::atomic::{AtomicUsize, Ordering};

/// Asserts every entry of `m` is within tolerance of the identity matrix.
fn approx_identity<const N: usize>(m: Matrix<N, N>) {
    let id: Matrix<N, N> = Matrix::identity();
    for r in 0..N {
        for c in 0..N {
            assert!((m.get(r, c).copied().unwrap() - id.get(r, c).copied().unwrap()).abs() < 1e-12);
        }
    }
}

// A full-rank 4x3 problem reused across the damped-solve tests.
fn sample_problem() -> (Matrix<4, 3>, Vector<4>) {
    let j = Matrix::<4, 3>::new([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 1.0, 1.0],
        [1.0, 0.0, 2.0],
    ]);
    let b = Vector::new([1.0, 2.0, 3.0, 4.0]);
    (j, b)
}

// ----- overflow-safe norm (enorm) and comparison helpers -----

#[test]
fn enorm_matches_naive_norm() {
    // On ordinary values, enorm agrees with the plain sqrt-of-dot norm.
    assert!((enorm(&[3.0_f64, 4.0]) - 5.0).abs() < 1e-12);

    let v = Vector::new([1.0_f64, 2.0, 2.0]);
    assert!((enorm(v.as_array()) - v.norm()).abs() < 1e-12);
}

#[test]
fn enorm_survives_huge_components() {
    // A naive norm would overflow to infinity here; enorm stays finite.
    let result = enorm(&[3.0e200_f64, 4.0e200]);
    assert!(result.is_finite());
    assert!((result / 5.0e200 - 1.0).abs() < 1e-12);
}

#[test]
fn enorm_survives_tiny_components() {
    // A naive norm would underflow to zero here; enorm keeps the magnitude.
    let result = enorm(&[3.0e-200_f64, 4.0e-200]);
    assert!(result > 0.0);
    assert!((result / 5.0e-200 - 1.0).abs() < 1e-12);
}

#[test]
fn enorm_f32_extremes_stay_finite() {
    let big = enorm(&[3.0e30_f32, 4.0e30]);
    assert!(big.is_finite());
    assert!((big / 5.0e30 - 1.0).abs() < 1e-5);

    let small = enorm(&[3.0e-30_f32, 4.0e-30]);
    assert!(small > 0.0);
    assert!((small / 5.0e-30 - 1.0).abs() < 1e-5);
}

#[test]
fn min_max_pick_an_argument() {
    assert_eq!(max(2.0_f64, 3.0), 3.0);
    assert_eq!(max(3.0_f64, 2.0), 3.0);
    assert_eq!(min(2.0_f64, 3.0), 2.0);
    assert_eq!(min(3.0_f64, 2.0), 2.0);

    // An incomparable pair returns the first argument unchanged.
    assert_eq!(max(1.0_f64, f64::NAN), 1.0);
    assert_eq!(min(1.0_f64, f64::NAN), 1.0);
}

// ----- column-pivoted QR internals (pivot order, column norms, diagonal) -----

#[test]
fn qr_reconstructs_pivoted_matrix() {
    let a = Matrix::<4, 3>::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [2.0, -1.0, 1.0],
    ]);
    let f = PivotedQr::decompose(a).unwrap();
    let perm = f.permutation();

    // The most-normed column (index 2) pivots to the front, so |r_diag[0]| is its norm.
    assert_eq!(perm[0], 2);
    assert!((f.r_diag[0].abs() - 146.0_f64.sqrt()).abs() < 1e-12);

    // Column norms are the original ones, in original order.
    assert!((f.column_norms[0] - 70.0_f64.sqrt()).abs() < 1e-12);
    assert!((f.column_norms[1] - 94.0_f64.sqrt()).abs() < 1e-12);
    assert!((f.column_norms[2] - 146.0_f64.sqrt()).abs() < 1e-12);

    let q = f.q();
    let r = f.r();

    // R is upper-triangular.
    for row in 0..3 {
        for col in 0..row {
            assert_eq!(r.get(row, col).copied().unwrap(), 0.0);
        }
    }

    // Q has orthonormal columns.
    approx_identity(q.transpose() * q);

    // A * P == Q * R.
    let ap = Matrix::<4, 3>::from_fn(|i, c| a.get(i, perm[c]).copied().unwrap());
    let product = q * r;
    for i in 0..4 {
        for c in 0..3 {
            assert!(
                (ap.get(i, c).copied().unwrap() - product.get(i, c).copied().unwrap()).abs()
                    < 1e-12
            );
        }
    }
}

#[test]
fn qr_handles_zero_column() {
    // Column 1 is entirely zero: decompose must not divide by its zero norm.
    let a = Matrix::<4, 3>::new([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0],
        [7.0, 0.0, 8.0],
    ]);
    let f = PivotedQr::decompose(a).unwrap();
    let perm = f.permutation();

    // The zero column sorts last and carries a zero diagonal.
    assert_eq!(perm[2], 1);
    assert!(f.r_diag[2].abs() < 1e-12);

    // The factorization still reproduces A * P.
    let ap = Matrix::<4, 3>::from_fn(|i, c| a.get(i, perm[c]).copied().unwrap());
    let product = f.q() * f.r();
    for i in 0..4 {
        for c in 0..3 {
            assert!(
                (ap.get(i, c).copied().unwrap() - product.get(i, c).copied().unwrap()).abs()
                    < 1e-12
            );
        }
    }
}

// ----- damped least squares: the Cholesky factor of the normal matrix -----

#[test]
fn damped_cholesky_factor_matches_normal_matrix() {
    let (j, b) = sample_problem();
    let diag = [1.0, 0.5, 2.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let (_, cf) = dls.solve_with_diagonal(&diag);

    // Reconstruct S (upper triangular) from the factor.
    let s = Matrix::<3, 3>::from_fn(|row, col| {
        if row == col {
            cf.s_diag[row]
        } else if col > row {
            cf.s.get(col, row).copied().unwrap()
        } else {
            0.0
        }
    });

    // SᵀS must equal RᵀR + D², with D permuted the way qrsolv applies it.
    let sts = s.transpose() * s;
    let rtr = dls.r.transpose() * dls.r;
    for row in 0..3 {
        for col in 0..3 {
            let mut expected = rtr.get(row, col).copied().unwrap();
            if row == col {
                let d = diag[dls.permutation[row]];
                expected += d * d;
            }
            assert!((sts.get(row, col).copied().unwrap() - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn enorm_handles_extreme_dynamic_range() {
    // Twelve large components: a naive sum of squares would overflow to infinity.
    let many_large = [1.0e160_f64; 12];
    let result = enorm(&many_large);
    assert!(result.is_finite());
    assert!((result / (12.0_f64.sqrt() * 1.0e160) - 1.0).abs() < 1e-12);

    // A vector mixing all three magnitude bands; the large band sets the norm.
    let mut mixed = [0.0_f64; 16];
    mixed[0] = 3.0e160;
    mixed[1] = 4.0e160;
    mixed[2] = 1.0;
    mixed[3] = 1.0;
    mixed[4] = 3.0e-160;
    mixed[5] = 4.0e-160;
    let norm = enorm(&mixed);
    assert!(norm.is_finite());
    assert!((norm / 5.0e160 - 1.0).abs() < 1e-12);
}

// ----- work-count regression guard -----

// A scalar that tallies every multiply and divide it performs, so a test can pin the arithmetic
// work of a factorization to a fixed count. That count is a deterministic function of the matrix
// size, independent of wall-clock timing, so it fails if an algorithm starts doing more work than
// the counts asserted in `factorization_work_counts`. Only that test touches the counter, and it
// runs single-threaded, so the shared tally needs no synchronization beyond atomicity.
static MUL_DIV_OPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct Counted(f64);

impl Counted {
    fn tick() {
        MUL_DIV_OPS.fetch_add(1, Ordering::Relaxed);
    }
}

impl Add for Counted {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Counted(self.0 + rhs.0)
    }
}
impl Sub for Counted {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Counted(self.0 - rhs.0)
    }
}
impl Mul for Counted {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::tick();
        Counted(self.0 * rhs.0)
    }
}
impl Div for Counted {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::tick();
        Counted(self.0 / rhs.0)
    }
}
impl Neg for Counted {
    type Output = Self;
    fn neg(self) -> Self {
        Counted(-self.0)
    }
}
impl AddAssign for Counted {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for Counted {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl MulAssign for Counted {
    fn mul_assign(&mut self, rhs: Self) {
        Self::tick();
        self.0 *= rhs.0;
    }
}
impl DivAssign for Counted {
    fn div_assign(&mut self, rhs: Self) {
        Self::tick();
        self.0 /= rhs.0;
    }
}

impl Numeric for Counted {
    const ZERO: Self = Counted(0.0);
    const ONE: Self = Counted(1.0);
    const TWO: Self = Counted(2.0);
    const HALF: Self = Counted(0.5);
    const HUNDRED: Self = Counted(f64::HUNDRED);
    const PI: Self = Counted(core::f64::consts::PI);
    const TWO_PI: Self = Counted(core::f64::consts::TAU);
    const EPSILON: Self = Counted(f64::EPSILON);
    const EPSILON_X4: Self = Counted(f64::EPSILON_X4);
    const EPSILON_X30: Self = Counted(f64::EPSILON_X30);
    const NAN: Self = Counted(f64::NAN);
    const INFINITY: Self = Counted(f64::INFINITY);
    const NEG_INFINITY: Self = Counted(f64::NEG_INFINITY);
    const MAX: Self = Counted(f64::MAX);
    const MIN_POSITIVE: Self = Counted(f64::MIN_POSITIVE);

    fn from_f64(value: f64) -> Self {
        Counted(value)
    }
    fn from_u64(value: u64) -> Self {
        Counted(value as f64)
    }
    fn from_usize(value: usize) -> Self {
        Counted(value as f64)
    }

    fn abs(self) -> Self {
        Counted(libm::fabs(self.0))
    }
    fn sqrt(self) -> Self {
        Counted(libm::sqrt(self.0))
    }
    fn sin(self) -> Self {
        Counted(libm::sin(self.0))
    }
    fn cos(self) -> Self {
        Counted(libm::cos(self.0))
    }
    fn tan(self) -> Self {
        Counted(libm::tan(self.0))
    }
    fn exp(self) -> Self {
        Counted(libm::exp(self.0))
    }
    fn ln(self) -> Self {
        Counted(libm::log(self.0))
    }
    fn atan2(self, other: Self) -> Self {
        Counted(libm::atan2(self.0, other.0))
    }
    fn copysign(self, sign: Self) -> Self {
        Counted(libm::copysign(self.0, sign.0))
    }
    fn floor(self) -> Self {
        Counted(libm::floor(self.0))
    }
    fn round(self) -> Self {
        Counted(libm::round(self.0))
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }
    fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

#[test]
fn factorization_work_counts() {
    // Symmetric positive-definite and invertible, so LU, Cholesky, and the direct 4x4 inverse all
    // apply. The multiply/divide counts below are fixed functions of the size — for a 4x4:
    //   LU:       divisions N(N-1)/2 = 6, multiplications sum_{p<N} p^2 = 14  -> 20
    //   Cholesky: multiplications 10, divisions 6                            -> 16
    //   Inverse:  cofactor expansion (95) plus EPSILON * n * scale^n (5)     -> 100
    // If any of these change, update the expected counts below.
    let a =
        Matrix::<4, 4, Counted>::from_fn(|i, j| if i == j { Counted(4.0) } else { Counted(1.0) });

    let measure = |f: &dyn Fn()| {
        MUL_DIV_OPS.store(0, Ordering::Relaxed);
        f();
        MUL_DIV_OPS.load(Ordering::Relaxed)
    };

    let lu = measure(&|| {
        let _ = a.lu().unwrap();
    });
    let cholesky = measure(&|| {
        let _ = a.cholesky().unwrap();
    });
    let inverse = measure(&|| {
        let _ = a.inverse().unwrap();
    });

    // One-sided Jacobi SVD on a fixed 3x3 (a deterministic sweep sequence).
    let a3 = Matrix::<3, 3, Counted>::from_fn(|i, j| {
        Counted([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]][i][j])
    });
    let svd = measure(&|| {
        let _ = a3.svd().unwrap();
    });

    assert_eq!((lu, cholesky, inverse), (20, 16, 100));
    // One-sided Jacobi converges in a fixed number of sweeps for this input.
    assert_eq!(svd, 441);
}
