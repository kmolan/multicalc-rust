//! Dormand–Prince 5(4) Butcher tableau (the RK45 coefficients).

// Stage nodes (c1 = 0, c6 = c7 = 1).
pub(super) const C2: f64 = 1.0 / 5.0;
pub(super) const C3: f64 = 3.0 / 10.0;
pub(super) const C4: f64 = 4.0 / 5.0;
pub(super) const C5: f64 = 8.0 / 9.0;

// Nonzero a[i][j] stage coefficients.
pub(super) const A21: f64 = 1.0 / 5.0;
pub(super) const A31: f64 = 3.0 / 40.0;
pub(super) const A32: f64 = 9.0 / 40.0;
pub(super) const A41: f64 = 44.0 / 45.0;
pub(super) const A42: f64 = -56.0 / 15.0;
pub(super) const A43: f64 = 32.0 / 9.0;
pub(super) const A51: f64 = 19372.0 / 6561.0;
pub(super) const A52: f64 = -25360.0 / 2187.0;
pub(super) const A53: f64 = 64448.0 / 6561.0;
pub(super) const A54: f64 = -212.0 / 729.0;
pub(super) const A61: f64 = 9017.0 / 3168.0;
pub(super) const A62: f64 = -355.0 / 33.0;
pub(super) const A63: f64 = 46732.0 / 5247.0;
pub(super) const A64: f64 = 49.0 / 176.0;
pub(super) const A65: f64 = -5103.0 / 18656.0;

// 5th-order solution weights (b7 = 0; b == stage-7 row, so k7 = f(t+h, y5) → FSAL).
pub(super) const B1: f64 = 35.0 / 384.0;
pub(super) const B3: f64 = 500.0 / 1113.0;
pub(super) const B4: f64 = 125.0 / 192.0;
pub(super) const B5: f64 = -2187.0 / 6784.0;
pub(super) const B6: f64 = 11.0 / 84.0;

// Error weights e = b(5th) − b*(4th), applied to k1..k7 (e2 = 0).
pub(super) const E1: f64 = 71.0 / 57600.0;
pub(super) const E3: f64 = -71.0 / 16695.0;
pub(super) const E4: f64 = 71.0 / 1920.0;
pub(super) const E5: f64 = -17253.0 / 339200.0;
pub(super) const E6: f64 = 22.0 / 525.0;
pub(super) const E7: f64 = -1.0 / 40.0;
