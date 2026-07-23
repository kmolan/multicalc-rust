use multicalc::Dual;
use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk4;

// y' = -y, generic over the scalar so the same RHS runs at f64 and through Dual.
fn decay<T: multicalc::Numeric>(_t: T, y: &Vector<1, T>) -> Vector<1, T> {
    -*y
}

#[test]
fn exp_decay_matches_closed_form() {
    // y(1) = e^{-1} from 1000 fixed steps.
    let steps = 1000;
    let yf = Rk4::integrate(
        &decay::<f64>,
        0.0,
        &Vector::new([1.0]),
        1.0 / steps as f64,
        steps,
        |_, _| {},
    );
    assert!((yf.as_array()[0] - (-1.0_f64).exp()).abs() < 1e-9);
}

#[test]
fn harmonic_matches_closed_form() {
    // y'' = -y as [y2, -y1]; over one period the state returns to [cos, -sin].
    let f = |_t: f64, y: &Vector<2, f64>| {
        let [y0, y1] = *y.as_array();
        Vector::new([y1, -y0])
    };
    let tf = core::f64::consts::TAU;
    let steps = 2000;
    let yf = Rk4::integrate(
        &f,
        0.0,
        &Vector::new([1.0, 0.0]),
        tf / steps as f64,
        steps,
        |_, _| {},
    );
    assert!((yf.as_array()[0] - tf.cos()).abs() < 1e-7);
    assert!((yf.as_array()[1] - (-tf.sin())).abs() < 1e-7);
}

#[test]
fn fourth_order_convergence() {
    // Halving the step should cut the global endpoint error by ~2^4 = 16.
    let exact = (-1.0_f64).exp();
    let err = |steps: usize| {
        let yf = Rk4::integrate(
            &decay::<f64>,
            0.0,
            &Vector::new([1.0]),
            1.0 / steps as f64,
            steps,
            |_, _| {},
        );
        (yf.as_array()[0] - exact).abs()
    };
    let ratio = err(50) / err(100);
    assert!((12.0..=20.0).contains(&ratio), "convergence ratio {ratio}");
}

#[test]
fn ad_through_rk4_matches_fd() {
    // Differentiate the final state w.r.t. the initial condition. For y' = -y the
    // exact sensitivity is d y_f / d a = e^{-t_f}; check it against a central FD.
    let a = 1.3_f64;
    let tf = 0.7;
    let steps = 100;

    let y0 = Vector::new([Dual::variable(a)]);
    let dt = Dual::constant(tf / steps as f64);
    let yf = Rk4::integrate(
        &decay::<Dual<f64>>,
        Dual::constant(0.0),
        &y0,
        dt,
        steps,
        |_, _| {},
    );
    let ad = yf.as_array()[0].deriv;
    assert!((ad - (-tf).exp()).abs() < 1e-6);

    let run = |a0: f64| {
        Rk4::integrate(
            &decay::<f64>,
            0.0,
            &Vector::new([a0]),
            tf / steps as f64,
            steps,
            |_, _| {},
        )
        .as_array()[0]
    };
    let h = 1e-6;
    let fd = (run(a + h) - run(a - h)) / (2.0 * h);
    assert!((ad - fd).abs() < 1e-6);
}

#[test]
fn f32_energy_round_trip() {
    // Harmonic oscillator at f32 over one period conserves y0^2 + y1^2 = 1.
    let f = |_t: f32, y: &Vector<2, f32>| {
        let [y0, y1] = *y.as_array();
        Vector::new([y1, -y0])
    };
    let tf = core::f32::consts::TAU;
    let steps = 2000;
    let yf = Rk4::integrate(
        &f,
        0.0_f32,
        &Vector::new([1.0, 0.0]),
        tf / steps as f32,
        steps,
        |_, _| {},
    );
    let yf = *yf.as_array();
    let energy = yf[0] * yf[0] + yf[1] * yf[1];
    assert!((energy - 1.0).abs() < 1e-3);
}
