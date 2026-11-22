# Generate high-precision reference values for the `spatial::lie` small-angle branches.
#
# The Rust code switches each trig ratio to a Taylor series below its own threshold. Neither
# branch is exact, so the tests need a reference computed well outside f64 to say which one is
# closer. mpmath at 80 digits provides that.
#
# Two tables come out of this, and they test different things:
#
# - The *coefficient* tables are the threshold test. A coefficient's error is undiluted there,
#   so a wrong cutoff or a wrong series term shows up directly.
# - The *matrix* tables are the assembly test: a coefficient plugged into the wrong term, a
#   sign slip in c4, a transposed product. They cannot see coefficient precision, because a
#   1e-13 relative error in c3 moves Q by only 1.5e-19 relative at c3's own threshold -- three
#   orders below an f64 ulp. Do not try to test thresholds through the matrices.
#
# Exactness discipline: every number handed to Rust must round-trip, and both sides must
# evaluate at *identical bits*. Grid points are chosen as f64, and rotation vectors are built
# at high precision then rounded to f64 once before being fed back in. The comparison then
# measures the branch, not a mismatched input.
#
# Run with `python3 tools/qa/src/bin/spatial_small_angle.py`.
import sys

from mpmath import eye, mp, mpf, sin, cos, matrix, fdot, sqrt

# 80 digits. c5's numerator `theta - sin(theta) - theta^3/6` cancels to a relative size of
# theta^4/120, so at theta = 1e-9 it loses 38 digits outright. 50 would leave too little.
mp.dps = 80

HALF = mpf(1) / 2
EPS_F64 = mpf(2) ** -52  # f64 machine epsilon, as an exact power of two
EPS_F32 = mpf(2) ** -32  # f32 machine epsilon


# --- coefficients -----------------------------------------------------------------------
#
# Each takes theta^2, matching the Rust functions, which branch on `theta_sq` to avoid a sqrt
# they may not need. Every one is even in theta, hence expressible in theta^2 alone.

def c1_so3(theta_sq):
    """`(1 - cos t)/t^2`, the [phi]x coefficient of the SO(3) left Jacobian. Limit 1/2."""
    if theta_sq == 0:
        return HALF
    return (1 - cos(sqrt(theta_sq))) / theta_sq


def c2_so3(theta_sq):
    """`(t - sin t)/t^3`, the [phi]x^2 coefficient of the SO(3) left Jacobian. Limit 1/6.

    Also the c2 of the Barfoot Q block -- same ratio, same threshold.
    """
    if theta_sq == 0:
        return mpf(1) / 6
    theta = sqrt(theta_sq)
    return (theta - sin(theta)) / (theta_sq * theta)


def c3_inverse_so3(theta_sq):
    """`(1 - (t/2)*cot(t/2))/t^2`, the [phi]x^2 coefficient of the inverse. Limit 1/12.

    Finite on (0, pi], so only theta = 0 needs the limit.
    """
    if theta_sq == 0:
        return mpf(1) / 12
    half = sqrt(theta_sq) * HALF
    return (1 - half * (cos(half) / sin(half))) / theta_sq


def c3_q(theta_sq):
    """`(1 - t^2/2 - cos t)/t^4`, from the Barfoot Q block. Limit -1/24.

    The numerator cancels to a relative size of t^4/24, which is why this one needs a much
    later cutoff than the c1/c2 family.
    """
    if theta_sq == 0:
        return -mpf(1) / 24
    return (1 - theta_sq * HALF - cos(sqrt(theta_sq))) / (theta_sq * theta_sq)


def c5_q(theta_sq):
    """`(t - sin t - t^3/6)/t^5`, from the Barfoot Q block. Limit -1/120.

    Worst cancellation of the five: the numerator survives at a relative t^4/120.
    """
    if theta_sq == 0:
        return -mpf(1) / 120
    theta = sqrt(theta_sq)
    theta3 = theta_sq * theta
    return (theta - sin(theta) - theta3 / 6) / (theta_sq * theta3)


# --- matrices ---------------------------------------------------------------------------

def skew3(v):
    """The 3x3 skew-symmetric matrix S(v), so that S(v) * p = cross(v, p).

    v: mpmath column vector, e.g. `matrix([1, 2, 3])`.
    """
    return matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
        ])


def left_jacobian_so3(v):
    """The SO(3) left Jacobian `J = I + c1*S + c2*S^2`.

    v: rotation vector phi = theta * axis, as an mpmath column vector.
    """
    theta_sq = fdot(v, v)
    s = skew3(v)
    return eye(3) + s * c1_so3(theta_sq) + (s * s) * c2_so3(theta_sq)


def inverse_left_jacobian_so3(v):
    """The inverse SO(3) left Jacobian `J^-1 = I - S/2 + c3*S^2`.

    v: rotation vector phi = theta * axis, as an mpmath column vector.
    """
    theta_sq = fdot(v, v)
    s = skew3(v)
    return eye(3) - s * HALF + (s * s) * c3_inverse_so3(theta_sq)


def q_matrix_se3(rho, phi):
    """The Barfoot SE(3) `Q(rho, phi)` block (Eq. 7.86), the top-right of the 6x6 Jacobian.

    rho: translation part of the twist, as an mpmath column vector.
    phi: rotation vector theta * axis, as an mpmath column vector.
    """
    theta_sq = fdot(phi, phi)
    p = skew3(rho)
    ph = skew3(phi)

    c2 = c2_so3(theta_sq)
    c3 = c3_q(theta_sq)
    c5 = c5_q(theta_sq)
    c4 = (c3 - 3 * c5) * HALF

    phph = ph * p * ph                                  # Phi P Phi, reused in two terms
    t2 = ph * p + p * ph + phph                         # PhiP + PPhi + PhiPPhi
    t3 = ph * ph * p + p * ph * ph - phph * 3           # Phi^2 P + P Phi^2 - 3 PhiPPhi
    t4 = ph * p * ph * ph + ph * ph * p * ph            # PhiPPhi^2 + Phi^2 PPhi
    return p * HALF + t2 * c2 - t3 * c3 - t4 * c4


# --- inputs -----------------------------------------------------------------------------
#
# AXIS is [2, -3, 6]/7. That triple is a Pythagorean quadruple (4 + 9 + 36 = 49), so the axis
# is a *rational* unit vector: |AXIS| is exactly 1 in exact arithmetic, which makes
# |AXIS * theta| exactly theta. No component is zero and no two share a magnitude, so every
# entry of S and S^2 is populated and no term drops out by symmetry.
AXIS = matrix([mpf(2) / 7, mpf(-3) / 7, mpf(6) / 7])

# RHO is [2, 1, -3]/4. Dyadic, so all three components are exact in f64 as well as in mpmath.
# It is neither parallel nor perpendicular to AXIS (cross = [3, 18, 8], dot = -17/28), so the
# Q terms stay generic. Q is linear in rho, so only its scale and orientation matter.
RHO = matrix([mpf(1) / 2, mpf(1) / 4, mpf(-3) / 4])


def phi_at(theta):
    """Rotation vector at angle `theta`, rounded to f64 componentwise.

    Rounding here is the whole point: the returned values are what the Rust literals will
    hold, so the reference is evaluated at the same bits Rust sees.
    """
    t = mpf(theta)
    return matrix([mpf(float(AXIS[i] * t)) for i in range(3)])

def thresholds_sq():
    """The five f64 branch cutoffs in theta^2, mirroring `spatial::small_angle_*_sq`."""
    e = EPS_F64
    return {
        "so3 c1": ("360e^1/3", (360 * e) ** (mpf(1) / 3)),
        "so3 c2": ("2,025e^1/3", (2520 * e) ** (mpf(1) / 3)),
        "inv c3": ("15,120e^1/3", (15120 * e) ** (mpf(1) / 3)),
        "q c2": ("2,025e^1/3", (2520 * e) ** (mpf(1) / 3)),
        "q c3": ("20,160e^1/4", (20160 * e) ** (mpf(1) / 4)),
        "q c5": ("181,440e^1/4", (181440 * e) ** (mpf(1) / 4)),
    }

# --- emitters ---------------------------------------------------------------------------

def lit(x):
    """Shortest f64 literal that round-trips. Rust's parser is correctly rounded, so it
    recovers the identical bits."""
    return repr(float(x))

def emit_matrix_table(name, threshold, dst, fn, rho = None):
    """Emit `(phi, matrix)` pairs.
    """
    (form, threshold) = threshold
    theta = sqrt(threshold)
    print("=======================================================================")
    print("==============Hard-coded high-precision value for testing==============")
    print(f"/// {name}")
    print(f"/// Threshold for theta_sq {form}")
    print(f"/// {theta} +/- {dst}")
    print(f"/// Computed using mpmath at 80 digits.")

    suffixes = ["hi", "lo"]
    signs = [1, -1]

    for sign, suffix in zip(signs, suffixes):
        phi = phi_at(theta + sign*dst)
        if rho is None:
            m = fn(phi)
        else:
            m = fn(rho, phi)

        print(f"let t_{suffix}: Vector<3, f64> = Vector::new([{', '.join([lit(phi[i]) for i in range(3)])}]);")
        print(f"let matrix_{suffix} = Matrix::new([")
        for r in range(3):
            entries = ", ".join(lit(m[r, c]) for c in range(3))
            print(f"    [{entries}],")
        print("]);")
        # key = ", ".join(lit(phi[i]) for i in range(3))
        # entries = ", ".join(lit(m[r, c]) for r in range(3) for c in range(3))
        # print(f"const theta_{} = {lit(theta)}")
        # print(f"    ([{key}], [{entries}]),")
        # print("];\n")

    print("")


BANNER = """// @generated by tools/qa/src/bin/spatial_small_angle.py -- do not edit by hand.
//
// High-precision reference values for the small-angle branches in `spatial::lie`, computed
// with mpmath at 80 decimal digits and rounded to f64 exactly once. Regenerate with:
//
//     python3 tools/qa/src/bin/spatial_small_angle.py matrices > <this file>
"""

def emit_matrices():
    # The assembly test. Blind to coefficient precision, see the header.
    print(f"// axis = {[lit(AXIS[i]) for i in range(3)]}")
    print(f"// rho  = {[lit(RHO[i]) for i in range(3)]}\n")
    # print(f"pub const RHO_REF: [f64; 3] = [{', '.join(lit(RHO[i]) for i in range(3))}];\n")

    thresholds = thresholds_sq()

    emit_matrix_table(
        "Left Jacobian SO3",
        thresholds["so3 c1"],
        1e-9,
        left_jacobian_so3)

    emit_matrix_table(
        "Left Jacobian SO3",
        thresholds["so3 c2"],
        1e-9,
        left_jacobian_so3)

    emit_matrix_table(
        "Inverse Left Jacobian SO3",
        thresholds["inv c3"],
        1e-9,
        inverse_left_jacobian_so3)

    emit_matrix_table(
        "Matrix Q SE3",
        thresholds["q c2"],
        1e-9,
        q_matrix_se3,
        RHO)

    emit_matrix_table(
        "Matrix Q SE3",
        thresholds["q c3"],
        1e-9,
        q_matrix_se3,
        RHO)

    emit_matrix_table(
        "Matrix Q SE3",
        thresholds["q c5"],
        1e-9,
        q_matrix_se3,
        RHO)

def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(BANNER)
    if section in ("all", "matrices"):
        emit_matrices()


if __name__ == "__main__":
    main()
