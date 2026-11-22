//! Dormand–Prince 5(4) Butcher tableau (the RK45 coefficients).

// Stage nodes (c1 = 0, c6 = c7 = 1).
pub(super) const NODE2: f64 = 1.0 / 5.0;
pub(super) const NODE3: f64 = 3.0 / 10.0;
pub(super) const NODE4: f64 = 4.0 / 5.0;
pub(super) const NODE5: f64 = 8.0 / 9.0;

// Nonzero a[i][j] stage coefficients.
pub(super) const STAGE_A21: f64 = 1.0 / 5.0;
pub(super) const STAGE_A31: f64 = 3.0 / 40.0;
pub(super) const STAGE_A32: f64 = 9.0 / 40.0;
pub(super) const STAGE_A41: f64 = 44.0 / 45.0;
pub(super) const STAGE_A42: f64 = -56.0 / 15.0;
pub(super) const STAGE_A43: f64 = 32.0 / 9.0;
pub(super) const STAGE_A51: f64 = 19372.0 / 6561.0;
pub(super) const STAGE_A52: f64 = -25360.0 / 2187.0;
pub(super) const STAGE_A53: f64 = 64448.0 / 6561.0;
pub(super) const STAGE_A54: f64 = -212.0 / 729.0;
pub(super) const STAGE_A61: f64 = 9017.0 / 3168.0;
pub(super) const STAGE_A62: f64 = -355.0 / 33.0;
pub(super) const STAGE_A63: f64 = 46732.0 / 5247.0;
pub(super) const STAGE_A64: f64 = 49.0 / 176.0;
pub(super) const STAGE_A65: f64 = -5103.0 / 18656.0;

// 5th-order solution weights (b7 = 0; b == stage-7 row, so k7 = f(t+h, y5) → FSAL).
pub(super) const WEIGHT1: f64 = 35.0 / 384.0;
pub(super) const WEIGHT3: f64 = 500.0 / 1113.0;
pub(super) const WEIGHT4: f64 = 125.0 / 192.0;
pub(super) const WEIGHT5: f64 = -2187.0 / 6784.0;
pub(super) const WEIGHT6: f64 = 11.0 / 84.0;

// Error weights e = b(5th) − b*(4th), applied to k1..k7 (e2 = 0).
pub(super) const ERROR1: f64 = 71.0 / 57600.0;
pub(super) const ERROR3: f64 = -71.0 / 16695.0;
pub(super) const ERROR4: f64 = 71.0 / 1920.0;
pub(super) const ERROR5: f64 = -17253.0 / 339200.0;
pub(super) const ERROR6: f64 = 22.0 / 525.0;
pub(super) const ERROR7: f64 = -1.0 / 40.0;
