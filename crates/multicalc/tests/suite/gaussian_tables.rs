#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::gaussian_tables::{MAX_ORDER, nodes};
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;

use proptest::prelude::*;

#[test]
fn lookup() {
    assert_eq!(
        nodes(GaussianQuadratureMethod::GaussLegendre, 4)
            .unwrap()
            .len(),
        4
    );
    assert!(nodes(GaussianQuadratureMethod::GaussLegendre, 0).is_err());
    assert!(nodes(GaussianQuadratureMethod::GaussHermite, MAX_ORDER + 1).is_err());
}

proptest! {
    #[test]
    fn proptest_lookup(order in 1..=MAX_ORDER) {
        prop_assert_eq!(
            nodes(GaussianQuadratureMethod::GaussLegendre, order)
                .unwrap()
                .len(),
            order
        );
        prop_assert_eq!(
            nodes(GaussianQuadratureMethod::GaussHermite, order)
                .unwrap()
                .len(),
            order
        );
        prop_assert_eq!(
            nodes(GaussianQuadratureMethod::GaussLaguerre, order)
                .unwrap()
                .len(),
            order
        );
    }
}
