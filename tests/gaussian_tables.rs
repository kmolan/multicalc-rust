use multicalc::gaussian_tables::{MAX_ORDER, nodes};
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;

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
