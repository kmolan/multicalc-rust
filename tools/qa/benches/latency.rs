//! Runs criterion benchmarks and writes the results to benchmarks/latency.md.
//! Run with `cargo bench -p multicalc-qa`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{BatchSize, Criterion};

use multicalc::control::Lqr;
use multicalc::dynamics::{ArticulatedBody, DynamicsWorkspace, RigidBody};
use multicalc::estimation::{ErrorStateKalmanFilter, ImuNoise, NominalState};
use multicalc::kinematics::{InverseKinematics, Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Matrix4D, Vector, Vector2D};
use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::numerical_derivative::Jacobian;
use multicalc::ode::Rk45;
use multicalc::root_finding::NewtonSystem;
use multicalc::scalar::{Numeric, VectorFn, constant};
use multicalc::scalar_fn_vec;
use multicalc::spatial::{FreeJointState, SE3, SO3, SpatialInertia, Twist, Wrench};
use multicalc_qa::docs::replace_marked_region;
use multicalc_qa::problems::{ConstantTurnAndSpeed, GlobalPosition, StationaryPose};

/// Diagonally dominant, mildly non-symmetric — well-conditioned and invertible.
fn general<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| {
        if i == j {
            (N + 2) as f64
        } else {
            1.0 / (1.0 + i as f64 + 2.0 * j as f64)
        }
    })
}

/// Symmetric and diagonally dominant, so its eigenvalues are well separated.
fn symmetric<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| {
        if i == j {
            (N + 2) as f64
        } else {
            1.0 / (1.0 + (i + j) as f64)
        }
    })
}

fn bench_jacobian_large(crit: &mut Criterion) {
    // A coupled 6-in / 6-out map: a 6x6 Jacobian
    let f = scalar_fn_vec!(|point: &[f64; 6]| [
        point[0] * point[1] + point[2].sin(),
        point[1] * point[2] + point[3].cos(),
        point[2] * point[3] + constant(0.1) * point[4].exp(),
        point[3] * point[4] + point[0] * point[5],
        point[4] * point[5] + point[1].sin(),
        point[5] * point[0] + point[2] * point[3],
    ]);
    let j: Jacobian = Jacobian::default();
    let point = [0.5, 1.0, 1.5, 0.3, 0.8, 1.2];
    crit.bench_function("jacobian_large", |b| {
        b.iter(|| j.evaluate(&f, black_box(&point)).unwrap())
    });
}

fn bench_lu_solve(criterion: &mut Criterion) {
    // Well-conditioned 10x10; decompose + solve A x = b.
    let a = general::<10>();
    let x_true = Vector::<10>::from_fn(|i| 1.0 + i as f64);
    let b_rhs = a * x_true;
    criterion.bench_function("lu_solve", |b| {
        b.iter(|| black_box(a).lu_decompose().unwrap().solve(black_box(b_rhs)))
    });
}

fn bench_svd_solve(criterion: &mut Criterion) {
    // Overdetermined 30x3 least-squares.
    let sample = |i: usize| ((i as f64 * 0.37).sin(), (i as f64 * 0.53).cos());
    let design = Matrix::<30, 3>::from_fn(|i, col| {
        let (x, y) = sample(i);
        match col {
            0 => x,
            1 => y,
            _ => 1.0,
        }
    });
    let rhs = Vector::<30>::from_fn(|i| {
        let (x, y) = sample(i);
        0.5 * x - 1.2 * y + 2.0 + 1e-3 * (i as f64 * 1.7).sin()
    });
    criterion.bench_function("svd_solve", |b| {
        b.iter(|| black_box(design).svd().unwrap().solve(black_box(rhs)))
    });
}

fn bench_symmetric_eigen(criterion: &mut Criterion) {
    // Eigenvalues and directions of a fixed symmetric 6x6.
    let matrix = symmetric::<6>();
    criterion.bench_function("symmetric_eigen", |b| {
        b.iter(|| black_box(matrix).symmetric_eigendecomposition().unwrap())
    });
}

fn bench_rk45_solve(criterion: &mut Criterion) {
    // Harmonic oscillator y'' = -y, adaptive solve to t = 2*pi.
    let f = |_t: f64, y: &Vector2D| Vector::new([y[1], -y[0]]);
    let state_start = Vector::new([1.0, 0.0]);
    let solver = Rk45::default().with_rtol(1e-9).with_atol(1e-12);
    criterion.bench_function("rk45_solve", |b| {
        b.iter(|| {
            solver
                .solve(&f, 0.0, black_box(&state_start), core::f64::consts::TAU)
                .unwrap()
        })
    });
}

fn bench_rigid_body_step(criterion: &mut Criterion) {
    // One 1 ms tick of a small flying machine under gravity and a steady push.
    let inertia = SpatialInertia::from_diagonal_inertia(
        0.8,
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([0.005, 0.007, 0.009]),
    )
    .unwrap();
    let body = RigidBody::new(inertia, Vector::new([0.0, 0.0, -9.81])).unwrap();
    let state = FreeJointState::new(
        SE3::identity(),
        Twist::new(Vector::new([1.5, 0.0, 2.0]), Vector::new([7.0, 3.0, 5.0])),
    );
    let wrench = Wrench::new(
        Vector::new([0.0, 0.0, 8.0]),
        Vector::new([0.02, -0.01, 0.005]),
    );
    criterion.bench_function("rigid_body_step", |b| {
        b.iter(|| body.stepped(black_box(state), black_box(wrench), 0.001))
    });
}

/// Slots in the benchmark arm: six hinges plus the welded tool.
const ARM_FRAMES: usize = 7;

/// Six hinges alternating about x and y on 0.25 links, with the tool welded 0.25 further.
fn bench_arm() -> KinematicTree<ARM_FRAMES, ARM_FRAMES, f64> {
    let link = SE3::from_parts(SO3::identity(), Vector::new([0.0, 0.0, 0.25]));
    let mut tree = KinematicTree::new();
    for index in 0..6 {
        let axis = if index % 2 == 0 {
            Vector::new([1.0, 0.0, 0.0])
        } else {
            Vector::new([0.0, 1.0, 0.0])
        };
        let origin = if index == 0 { SE3::identity() } else { link };
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(Joint::revolute(axis, origin).with_limits(-2.6, 2.6), parent)
            .unwrap();
    }
    tree.push(Joint::fixed(link), JointParent::Joint(5))
        .unwrap();
    tree
}

/// The benchmark arm with mass: a 1 kg link 0.15 m out per slot, with rotor inertia and joint
/// friction as a real arm states them.
fn bench_articulated_arm() -> ArticulatedBody<ARM_FRAMES, ARM_FRAMES, f64> {
    let mut tree = KinematicTree::<ARM_FRAMES, ARM_FRAMES, f64>::new();
    let plain = bench_arm();
    for index in 0..plain.len() {
        let joint = plain
            .joint(index)
            .unwrap()
            .with_armature(0.1)
            .with_damping(1.0)
            .with_friction_loss(0.2);
        tree.push(joint, plain.parent(index).unwrap()).unwrap();
    }

    let link = SpatialInertia::new(
        1.0,
        Vector::new([0.15, 0.0, 0.0]),
        Matrix::from_diagonal([0.01, 0.01, 0.01]),
    )
    .unwrap();
    let inertias = [Some(link); ARM_FRAMES];
    ArticulatedBody::new(tree, &inertias, Vector::new([0.0, 0.0, -9.81])).unwrap()
}

/// The two-link pendulum the goldens use, as an `ArticulatedBody`.
fn bench_double_pendulum() -> ArticulatedBody<2, 2, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    let mut tree = KinematicTree::<2, 2, f64>::new();
    tree.push(
        Joint::revolute(axis, SE3::identity())
            .with_armature(0.05)
            .with_damping(0.7)
            .with_friction_loss(0.2),
        JointParent::World,
    )
    .unwrap();
    tree.push(
        Joint::revolute(axis, link)
            .with_armature(0.03)
            .with_damping(0.4)
            .with_friction_loss(0.1),
        JointParent::Joint(0),
    )
    .unwrap();

    let inertia = SpatialInertia::new(
        1.5,
        Vector::new([0.5, 0.0, 0.0]),
        Matrix::from_diagonal([0.02, 0.02, 0.02]),
    )
    .unwrap();
    ArticulatedBody::new(
        tree,
        &[Some(inertia), Some(inertia)],
        Vector::new([0.0, 0.0, -9.81]),
    )
    .unwrap()
}

fn bench_inverse_dynamics_double_pendulum(criterion: &mut Criterion) {
    // Forward kinematics runs once outside the timed closure, so the number is the recursion's
    // cost rather than a forward sweep measured twice.
    let body = bench_double_pendulum();
    let configuration = Vector::new([0.4, -0.9]);
    let state = body.tree().forward_kinematics(&configuration).unwrap();
    let velocity = Vector::new([1.1, -0.7]);
    let acceleration = Vector::new([0.6, 1.3]);
    criterion.bench_function("inverse_dynamics_double_pendulum", |b| {
        b.iter(|| {
            body.inverse_dynamics(
                black_box(&state),
                black_box(&velocity),
                black_box(&acceleration),
            )
            .unwrap()
        })
    });
}

fn bench_inverse_dynamics_arm(criterion: &mut Criterion) {
    let body = bench_articulated_arm();
    let configuration = Vector::new([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0]);
    let state = body.tree().forward_kinematics(&configuration).unwrap();
    let velocity = Vector::new([0.5, -0.3, 0.4, 0.2, -0.6, 0.1, 0.0]);
    let acceleration = Vector::new([0.9, 0.4, -0.7, 0.3, 0.2, -0.5, 0.0]);
    criterion.bench_function("inverse_dynamics_arm", |b| {
        b.iter(|| {
            body.inverse_dynamics(
                black_box(&state),
                black_box(&velocity),
                black_box(&acceleration),
            )
            .unwrap()
        })
    });
}

fn bench_joint_space_inertia_double_pendulum(criterion: &mut Criterion) {
    let body = bench_double_pendulum();
    let state = body
        .tree()
        .forward_kinematics(&Vector::new([0.4, -0.9]))
        .unwrap();
    criterion.bench_function("joint_space_inertia_double_pendulum", |b| {
        b.iter(|| body.joint_space_inertia(black_box(&state)).unwrap())
    });
}

fn bench_joint_space_inertia_arm(criterion: &mut Criterion) {
    let body = bench_articulated_arm();
    let state = body
        .tree()
        .forward_kinematics(&Vector::new([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0]))
        .unwrap();
    criterion.bench_function("joint_space_inertia_arm", |b| {
        b.iter(|| body.joint_space_inertia(black_box(&state)).unwrap())
    });
}

fn bench_forward_dynamics_arm(criterion: &mut Criterion) {
    let body = bench_articulated_arm();
    let state = body
        .tree()
        .forward_kinematics(&Vector::new([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0]))
        .unwrap();
    let velocity = Vector::new([0.5, -0.3, 0.4, 0.2, -0.6, 0.1, 0.0]);
    let torque = Vector::new([2.0, -1.4, 0.8, 0.5, -0.9, 0.3, 0.0]);
    criterion.bench_function("forward_dynamics_arm", |b| {
        b.iter(|| {
            body.forward_dynamics(black_box(&state), black_box(&velocity), black_box(&torque))
                .unwrap()
        })
    });
}

fn bench_forward_dynamics_arm_with_workspace(criterion: &mut Criterion) {
    // The pair with `forward_dynamics_arm` is what says whether the caller-owned workspace earns
    // its keep: same work, the model-sized frame kept off the stack.
    let body = bench_articulated_arm();
    let state = body
        .tree()
        .forward_kinematics(&Vector::new([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0]))
        .unwrap();
    let velocity = Vector::new([0.5, -0.3, 0.4, 0.2, -0.6, 0.1, 0.0]);
    let torque = Vector::new([2.0, -1.4, 0.8, 0.5, -0.9, 0.3, 0.0]);
    let mut workspace = DynamicsWorkspace::<ARM_FRAMES, ARM_FRAMES, f64>::new();
    criterion.bench_function("forward_dynamics_arm_with_workspace", |b| {
        b.iter(|| {
            body.forward_dynamics_with_workspace(
                black_box(&state),
                black_box(&velocity),
                black_box(&torque),
                &mut workspace,
            )
            .unwrap()
        })
    });
}

fn bench_inverse_kinematics_solve(criterion: &mut Criterion) {
    let tree = bench_arm();
    let solver = InverseKinematics::<ARM_FRAMES, f64>::new();

    // The target is the pose 0.05 rad per joint away from the seed, which is the warm-started solve
    // a control loop actually runs: five iterations. A cold solve from across the workspace times a
    // different thing entirely — 0.3 rad per joint takes 21 iterations and four times as long.
    //
    // 0.05 sits on a plateau: anything from 0.02 to 0.05 converges in the same five iterations, so
    // the count does not sit on a knife edge and drift with the next change to the damping.
    let seed = Vector::new([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0]);
    let displaced = Vector::from_fn(|index| {
        let reading = seed.get(index).copied().unwrap_or(0.0);
        if index < 6 { reading + 0.05 } else { reading }
    });
    let target = tree
        .forward_kinematics(&displaced)
        .unwrap()
        .pose(6)
        .unwrap();

    criterion.bench_function("inverse_kinematics_solve", |b| {
        b.iter(|| solver.solve(black_box(&tree), 6, black_box(target), black_box(&seed)))
    });
}

/// Beacons at the corners of a 10 m square room, ranged against for robot localization.
/// Example taken from https://github.com/destenson/kalman-filter-rs/blob/master/examples/particle_filter_robot.rs
const BEACONS: [(f64, f64); 4] = [(2.0, 8.0), (8.0, 8.0), (8.0, 2.0), (2.0, 2.0)];

/// The straight-line distance from the robot's position to each beacon.
struct BeaconRanges;
impl VectorFn<3, 4> for BeaconRanges {
    fn eval<S: Numeric>(&self, pose: &[S; 3]) -> [S; 4] {
        core::array::from_fn(|i| {
            let to_beacon_x = pose[0] - S::from_f64(BEACONS[i].0);
            let to_beacon_y = pose[1] - S::from_f64(BEACONS[i].1);
            (to_beacon_x * to_beacon_x + to_beacon_y * to_beacon_y).sqrt()
        })
    }
}

fn bench_particle_filter(criterion: &mut Criterion) {
    use multicalc::estimation::{GaussianLikelihood, ParticleFilter};

    // One predict→update cycle of a 1000-particle range-only robot localizer: a 3-DOF pose scored
    // against 4 beacon ranges. The robot is held stopped so the reused filter stays near the fixed
    // measurement across criterion's iterations; resampling fires when the cloud concentrates.
    let mut filter = ParticleFilter::<3, 4>::new(
        1000,
        Vector::new([5.0, 5.0, 0.0]),
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]),
        Matrix::new([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
        7,
    )
    .unwrap();
    let sensor = GaussianLikelihood::new(Matrix4D::identity().scale(0.09)).unwrap();
    let motion = StationaryPose;
    let ranges = BeaconRanges;
    // The measurement is the true ranges from the robot's held pose, formed once outside the loop.
    let measurement = Vector::new(ranges.eval(&[5.0, 5.0, 0.0]));

    criterion.bench_function("particle_filter", |b| {
        b.iter(|| {
            filter.predict(black_box(&motion)).unwrap();
            filter
                .update(
                    black_box(&ranges),
                    black_box(&sensor),
                    black_box(measurement),
                )
                .unwrap();
        })
    });
}

fn bench_kalman_filter_step(criterion: &mut Criterion) {
    // One predict + update of a constant-velocity tracker: position measured, velocity inferred.
    use multicalc::estimation::{KalmanFilter, KalmanModel};
    let mut filter = KalmanFilter::<2, 1>::new(
        Vector::new([0.0, 1.0]),
        Matrix::identity(),
        KalmanModel {
            state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
            measurement_model: Matrix::new([[1.0, 0.0]]),
            process_noise: Matrix::new([[0.05, 0.0], [0.0, 0.05]]),
            measurement_noise: Matrix::new([[0.5]]),
        },
    );
    let measurement = Vector::new([1.0]);
    criterion.bench_function("kalman_filter_step", |b| {
        b.iter(|| {
            filter.predict();
            filter.update(black_box(measurement)).unwrap();
        })
    });
}

fn bench_extended_kalman_filter_step(criterion: &mut Criterion) {
    // One predict + update of the showcase's 5-state coordinated-turn filter against a
    // position fix: the per-tick cost the 1 kHz loop is built around.
    use multicalc::estimation::ExtendedKalmanFilter;
    let motion = ConstantTurnAndSpeed { timestep: 0.001 };
    let mut filter = ExtendedKalmanFilter::<5, 2>::new(
        Vector::new([0.0, 0.0, 0.0, 1.0, 0.3]),
        Matrix::from_fn(|i, j| if i == j { 0.5 } else { 0.0 }),
        Matrix::from_fn(|i, j| {
            if i != j {
                0.0
            } else if i < 3 {
                1e-7
            } else {
                4e-4
            }
        }),
        Matrix::from_fn(|i, j| if i == j { 0.09 } else { 0.0 }),
    );
    let measurement = Vector::new([0.001, 0.0]);
    criterion.bench_function("extended_kalman_filter_step", |b| {
        b.iter(|| {
            filter.predict(black_box(&motion)).unwrap();
            filter
                .update(black_box(&GlobalPosition), black_box(measurement))
                .unwrap();
        })
    });
}

fn bench_unscented_kalman_filter_step(criterion: &mut Criterion) {
    // One predict + update of the same 5-state coordinated-turn filter, sampled at a spread of
    // points instead of linearized: the other side of the accuracy-for-evaluations trade.
    // Each iteration starts from a fresh copy of the same belief, which the untimed setup hands
    // over. Reusing one filter for millions of ticks against a single repeated measurement lets the
    // heading uncertainty — nothing in a position fix observes it — grow until the sampled points
    // straddle ±π, where this motion model wraps its own heading and averaging them stops meaning
    // anything. That is a property of the run, not of one step, and it is not what this row times.
    use multicalc::estimation::UnscentedKalmanFilter;
    let motion = ConstantTurnAndSpeed { timestep: 0.001 };
    let initial = UnscentedKalmanFilter::<5, 2>::new(
        Vector::new([0.0, 0.0, 0.0, 1.0, 0.3]),
        Matrix::from_fn(|i, j| if i == j { 0.5 } else { 0.0 }),
        Matrix::from_fn(|i, j| {
            if i != j {
                0.0
            } else if i < 3 {
                1e-7
            } else {
                4e-4
            }
        }),
        Matrix::from_fn(|i, j| if i == j { 0.09 } else { 0.0 }),
    );
    let measurement = Vector::new([0.001, 0.0]);
    criterion.bench_function("unscented_kalman_filter_step", |b| {
        b.iter_batched(
            || initial,
            |mut filter| {
                filter.predict(black_box(&motion)).unwrap();
                filter
                    .update(black_box(&GlobalPosition), black_box(measurement))
                    .unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}

/// The filter both error-state rows are timed against: tilted a little off level, with the noise
/// figures a hobby-grade IMU quotes.
fn error_state_kalman_filter() -> ErrorStateKalmanFilter<3, f64> {
    let facing = SO3::exp(Vector::new([0.3, -0.2, 0.5]));
    let imu_noise = ImuNoise {
        gyroscope_noise_density: 0.02,
        accelerometer_noise_density: 0.05,
        gyroscope_bias_random_walk: 1e-4,
        accelerometer_bias_random_walk: 1e-3,
    };
    let tracker_spread = 0.03;
    ErrorStateKalmanFilter::new(
        NominalState::at_rest(facing),
        Matrix::from_diagonal([0.1; 15]),
        imu_noise,
        Matrix::from_diagonal([tracker_spread * tracker_spread; 3]),
    )
}

fn bench_error_state_kalman_filter_predict(criterion: &mut Criterion) {
    // One IMU step of the 15-state error filter: the closed-form transition plus two 15-square
    // products. This is the side that runs at 1 kHz, so it is timed on its own.
    let mut filter = error_state_kalman_filter();
    let gyroscope_reading = Vector::new([0.4, -0.3, 0.9]);
    let accelerometer_reading = Vector::new([0.5, -1.2, 9.4]);
    let timestep = 0.001;
    criterion.bench_function("error_state_kalman_filter_predict", |b| {
        b.iter(|| {
            filter
                .predict(
                    black_box(gyroscope_reading),
                    black_box(accelerometer_reading),
                    black_box(timestep),
                )
                .unwrap();
        })
    });
}

fn bench_polynomial_root_solve(criterion: &mut Criterion) {
    // The degree-6 polynomial from the fixtures, past where any formula reaches.
    use multicalc::Polynomial;
    let polynomial = Polynomial::<7, f64>::new([84.0, -131.0, -126.5, 104.5, 5.5, -9.5, 1.0]);
    let bound = polynomial.cauchy_root_bound().unwrap();
    criterion.bench_function("polynomial_root_solve", |b| {
        b.iter(|| {
            polynomial
                .real_roots_in(black_box(-bound), black_box(bound), 1e-10, 400)
                .unwrap()
        })
    });
}

/// The cart-and-pole system the LQR bench uses, discretized at a 0.02 s timestep.
fn cart_pole_design() -> (Matrix<4, 4>, Matrix<4, 1>, Matrix<4, 4>, Matrix<1, 1>) {
    let state_transition = Matrix::<4, 4>::new([
        [1.0, 0.02, 0.0, 0.0],
        [0.0, 1.0, 0.0196, 0.0],
        [0.0, 0.0, 1.0, 0.02],
        [0.0, 0.0, 0.4316, 1.0],
    ]);
    let input_model = Matrix::<4, 1>::new([[0.0002], [0.02], [-0.0004], [-0.04]]);
    let state_cost = Matrix::<4, 4>::from_diagonal([10.0, 1.0, 10.0, 1.0]);
    let input_cost = Matrix::<1, 1>::new([[0.1]]);
    (state_transition, input_model, state_cost, input_cost)
}

fn bench_infinite_horizon_lqr_solve(criterion: &mut Criterion) {
    // Same system as the control bench, timed for the once-per-startup solve rather than the tick.
    let (state_transition, input_model, state_cost, input_cost) = cart_pole_design();
    criterion.bench_function("infinite_horizon_lqr_solve", |b| {
        b.iter(|| {
            black_box(
                Lqr::new(
                    black_box(state_transition),
                    black_box(input_model),
                    black_box(state_cost),
                    black_box(input_cost),
                )
                .unwrap(),
            )
        });
    });
}

fn bench_minimum_snap_plan(criterion: &mut Criterion) {
    // The seven-segment loop from the motion fixtures, planned from scratch each time.
    use multicalc::motion::{MinimumSnapPlanner, durations_from_average_speed};
    let waypoints = [
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([7.0, 0.0, 1.0]),
        Vector::new([7.0, 7.0, 2.5]),
        Vector::new([0.0, 7.0, 1.5]),
        Vector::new([0.0, 0.0, 2.0]),
        Vector::new([3.5, 3.5, 2.5]),
        Vector::new([7.0, 3.5, 1.0]),
        Vector::new([3.5, 0.0, 0.5]),
    ];
    let mut durations = [0.0_f64; 7];
    durations_from_average_speed(&waypoints, 3.0, &mut durations).unwrap();

    let planner = MinimumSnapPlanner::<8, 21, 3, f64>::new();
    criterion.bench_function("minimum_snap_plan", |b| {
        b.iter(|| {
            planner
                .plan(black_box(&waypoints), black_box(&durations))
                .unwrap()
        })
    });
}

fn bench_newton_system(crit: &mut Criterion) {
    // x^2 + y^2 = 4 and x*y = 1 (circle ∩ hyperbola).
    let system = scalar_fn_vec!(|point: &[f64; 2]| [
        constant(-4.0) + point[0] * point[0] + point[1] * point[1],
        constant(-1.0) + point[0] * point[1],
    ]);
    crit.bench_function("newton_system", |b| {
        b.iter(|| {
            NewtonSystem::<AutoDiffMulti>::default()
                .solve(&system, black_box(&[1.5, 0.8]))
                .unwrap()
        })
    });
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    for (_, run, _) in BENCHES {
        run(&mut criterion);
    }

    criterion.final_summary();

    render_latency_md();
}

// =================== latency.md rendering ===================

/// Every benchmark: its criterion id, the function that runs it, and the equation
/// shown in the table. One list, so a bench cannot be run without a row or given
/// a row it never fills.
type BenchFn = fn(&mut Criterion);
const BENCHES: &[(&str, BenchFn, &str)] = &[
    (
        "jacobian_large",
        bench_jacobian_large,
        "Jacobian of a 6-in/6-out map",
    ),
    ("lu_solve", bench_lu_solve, "solve A·x = b (10×10)"),
    ("svd_solve", bench_svd_solve, "least-squares fit (30×3)"),
    (
        "symmetric_eigen",
        bench_symmetric_eigen,
        "eigenvalues and directions (6×6)",
    ),
    ("rk45_solve", bench_rk45_solve, "y″ = −y, adaptive to 2π"),
    (
        "rigid_body_step",
        bench_rigid_body_step,
        "one 1 ms tick of a free body: gravity + steady push, orientation on the manifold",
    ),
    (
        "inverse_dynamics_double_pendulum",
        bench_inverse_dynamics_double_pendulum,
        "2-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ",
    ),
    (
        "inverse_dynamics_arm",
        bench_inverse_dynamics_arm,
        "7-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ",
    ),
    (
        "joint_space_inertia_double_pendulum",
        bench_joint_space_inertia_double_pendulum,
        "2-joint arm, composite-rigid-body: q -> H(q)",
    ),
    (
        "joint_space_inertia_arm",
        bench_joint_space_inertia_arm,
        "7-joint arm, composite-rigid-body: q -> H(q)",
    ),
    (
        "forward_dynamics_arm",
        bench_forward_dynamics_arm,
        "7-joint arm, articulated-body: (q, q̇, τ) -> q̈, workspace on the stack",
    ),
    (
        "forward_dynamics_arm_with_workspace",
        bench_forward_dynamics_arm_with_workspace,
        "7-joint arm, articulated-body: (q, q̇, τ) -> q̈, caller-owned workspace",
    ),
    (
        "inverse_kinematics_solve",
        bench_inverse_kinematics_solve,
        "7-joint arm, warm-started SE(3) pose solve with joint limits",
    ),
    ("newton_system", bench_newton_system, "x²+y² = 4, x·y = 1"),
    (
        "particle_filter",
        bench_particle_filter,
        "1000 particles, diff-drive motion + process noise + systematic resample",
    ),
    (
        "kalman_filter_step",
        bench_kalman_filter_step,
        "2-state constant velocity, predict + update",
    ),
    (
        "extended_kalman_filter_step",
        bench_extended_kalman_filter_step,
        "5-state coordinated turn + position fix, autodiff Jacobians, predict + update",
    ),
    (
        "error_state_kalman_filter_predict",
        bench_error_state_kalman_filter_predict,
        "15-state error filter, one IMU step: closed-form transition + covariance",
    ),
    (
        "unscented_kalman_filter_step",
        bench_unscented_kalman_filter_step,
        "5-state coordinated turn + position fix, 11 sampled points, predict + update",
    ),
    (
        "polynomial_root_solve",
        bench_polynomial_root_solve,
        "six real roots of a sixth-power polynomial",
    ),
    (
        "minimum_snap_plan",
        bench_minimum_snap_plan,
        "smoothest path through 8 waypoints in 3D",
    ),
    (
        "infinite_horizon_lqr_solve",
        bench_infinite_horizon_lqr_solve,
        "solve the Riccati equation and form K, 4 states / 1 input",
    ),
];

const BEGIN: &str = "<!-- BEGIN generated: latency -->";
const END: &str = "<!-- END generated -->";

/// Locate <target>/criterion: honor $CARGO_TARGET_DIR, else workspace target at repo root.
#[must_use]
fn criterion_dir() -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(target_dir) => PathBuf::from(target_dir).join("criterion"),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion"),
    }
}

/// Read median & mean point estimates (ns) for one bench id from its estimates.json.
#[must_use]
fn read_estimate(dir: &std::path::Path, bench_id: &str) -> Option<(f64, f64)> {
    let path = dir.join(bench_id).join("new").join("estimates.json");
    let text = std::fs::read_to_string(path).ok()?;
    let j: serde_json::Value = serde_json::from_str(&text).ok()?;
    let median = j["median"]["point_estimate"].as_f64()?;
    let mean = j["mean"]["point_estimate"].as_f64()?;
    Some((median, mean))
}

/// Human-readable ns → "123.4 ns" / "12.3 µs" / "1.23 ms".
#[must_use]
fn fmt_ns(nanos: f64) -> String {
    if nanos < 1_000.0 {
        format!("{nanos:.1} ns")
    } else if nanos < 1_000_000.0 {
        format!("{:.2} µs", nanos / 1_000.0)
    } else {
        format!("{:.2} ms", nanos / 1_000_000.0)
    }
}

fn render_latency_md() {
    use std::fmt::Write as _;

    let dir = criterion_dir();
    let mut table = String::from("| Operation | Equation | Median | Mean |\n");
    table.push_str("|-----------|----------|-------:|-----:|\n");
    let mut missing = Vec::new();
    for (bench_id, _, equation) in BENCHES {
        let cell = match read_estimate(&dir, bench_id) {
            Some((median, mean)) => format!("{} | {}", fmt_ns(median), fmt_ns(mean)),
            None => {
                missing.push(*bench_id);
                "n/a | n/a".to_string()
            }
        };
        let _ = writeln!(table, "| `{bench_id}` | {equation} | {cell} |");
    }
    assert!(
        missing.is_empty(),
        "no criterion results for {missing:?} — the table would print n/a for them"
    );

    let doc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/latency.md");
    replace_marked_region(&doc, BEGIN, END, &table);
    println!("updated {}", doc.display());
}
