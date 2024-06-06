use crate::ode::euler as euler;
use crate::ode::heun as heun;
use crate::ode::runge_kutta as runge_kutta;

#[test]
fn test_euler_solve_1() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect a value of ~1.11
    let val = euler::solve(func, 0.0, 1.0, 0.1, 0.025);
    assert!(val > 1.11 && val < 1.12); 
}

#[test]
fn test_euler_solve_2() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because x0 == xn
    let result = std::panic::catch_unwind(||euler::solve(func, 0.0, 1.0, 0.0, 0.025));
    assert!(result.is_err());
}

#[test]
fn test_euler_solve_3() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is not non-zero
    let result = std::panic::catch_unwind(||euler::solve(func, 0.0, 1.0, 0.1, -1.0));
    assert!(result.is_err());
}

#[test]
fn test_euler_solve_4() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is too big
    let result = std::panic::catch_unwind(||euler::solve(func, 0.0, 1.0, 0.1, 5.0));
    assert!(result.is_err());
}

#[test]
fn test_heun_solve_1() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect a value of ~1.11
    let val = heun::solve(func, 0.0, 1.0, 0.1, 0.025);
    assert!(val > 1.11 && val < 1.12);
}

#[test]
fn test_heun_solve_2() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because x0 == xn
    let result = std::panic::catch_unwind(||heun::solve(func, 0.0, 1.0, 0.0, 0.025));
    assert!(result.is_err());    
}

#[test]
fn test_heun_solve_3() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is not non-zero
    let result = std::panic::catch_unwind(||heun::solve(func, 0.0, 1.0, 0.1, -1.0));
    assert!(result.is_err());
}

#[test]
fn test_heun_solve_4() 
{
    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is too big
    let result = std::panic::catch_unwind(||heun::solve(func, 0.0, 1.0, 0.1, 5.0));
    assert!(result.is_err());
}

#[test]
fn test_rk2_1() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect a value of ~1.11
    let val = runge_kutta::solve(RKMode::RK2, func, 0.0, 1.0, 0.1, 0.025);
    assert!(val > 1.11 && val < 1.12);
}

#[test]
fn test_rk2_2() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because x0 == xn
    let result = std::panic::catch_unwind(||runge_kutta::solve(RKMode::RK2, func, 0.0, 1.0, 0.0, 0.025));
    assert!(result.is_err());
}

#[test]
fn test_rk2_3() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is not non-zero
    let result = std::panic::catch_unwind(||runge_kutta::solve(RKMode::RK2, func, 0.0, 1.0, 0.1, -1.0));
    assert!(result.is_err());
}

#[test]
fn test_rk2_4() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is too big

    let result = std::panic::catch_unwind(||runge_kutta::solve(RKMode::RK2, func, 0.0, 1.0, 0.1, 5.0));
    assert!(result.is_err());
}

#[test]
fn test_rk4_1() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect a value of ~1.11
    let val = runge_kutta::solve(RKMode::RK4, func, 0.0, 1.0, 0.1, 0.025);
    assert!(val > 1.11 && val < 1.12);
}

#[test]
fn test_rk4_2() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because x0 == xn
    let result = std::panic::catch_unwind(||runge_kutta::solve(RKMode::RK4, func, 0.0, 1.0, 0.0, 0.025));
    assert!(result.is_err());    
}

#[test]
fn test_rk4_3() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is not non-zero
    let result = std::panic::catch_unwind(||runge_kutta::solve(RKMode::RK4, func, 0.0, 1.0, 0.1, -1.0));
    assert!(result.is_err());    
}

#[test]
fn test_rk4_4() 
{
    use runge_kutta::RKMode as RKMode;

    let func = | x:f64, y:f64 | -> f64 { return x + y + x*y; };

    //expect an error because step size is too big
    let result = std::panic::catch_unwind(||runge_kutta::solve(RKMode::RK4, func, 0.0, 1.0, 0.1, 5.0));
    assert!(result.is_err());
}