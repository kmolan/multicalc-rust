//! Linear algebra integration tests, split by topic. Shared helpers live in `helpers`; the
//! topic modules are kept in the `linear_algebra/` subdirectory so they form one test binary
//! rather than one per file.

#[path = "linear_algebra/helpers.rs"]
mod helpers;

#[path = "linear_algebra/cholesky.rs"]
mod cholesky;
#[path = "linear_algebra/lu.rs"]
mod lu;
#[path = "linear_algebra/matrix.rs"]
mod matrix;
#[path = "linear_algebra/qr.rs"]
mod qr;
#[path = "linear_algebra/svd.rs"]
mod svd;
#[path = "linear_algebra/vector.rs"]
mod vector;
