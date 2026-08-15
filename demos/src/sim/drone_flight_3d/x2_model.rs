//! The machine's own numbers: its mass read out of the model file, its rotor layout copied from the
//! same file by hand. Anything that cannot be read off that file is not a fact about this airframe
//! and belongs in `flight_world` instead.

use std::error::Error;
use std::path::Path;

use multicalc::dynamics::RigidBody;
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::plant::{MultirotorMixer, RotorSpin};
use multicalc::spatial::{Quaternion, SO3, SpatialInertia};

/// How many rotors the machine has.
pub const ROTOR_COUNT: usize = 4;

/// The pull of gravity, in metres per second squared.
pub const GRAVITY_STRENGTH: f64 = 9.81;

/// How many times a second a rotor comes round while the machine is flying.
///
/// The shake the machine puts into anything bolted to it sits at this rate, and the notches that
/// take that shake out are placed on it. Both read the number from here, because a notch placed
/// anywhere other than where the shake is does nothing at all.
pub const ROTOR_TONE_HERTZ: f64 = 180.0;

/// Where each rotor sits in the body's own axes, from `x2.xml`.
///
/// The four sit at the corners of a rectangle 28 cm along the body by 36 cm across it, and the
/// front pair sits three centimetres higher than the back pair.
const ROTOR_POSITIONS: [[f64; 3]; ROTOR_COUNT] = [
    [-0.14, -0.18, 0.05],
    [-0.14, 0.18, 0.05],
    [0.14, 0.18, 0.08],
    [0.14, -0.18, 0.08],
];

/// Which way each rotor turns, read from the sign of its motor's twist about z in `x2.xml`.
///
/// The two on each diagonal turn the same way, so their twists cancel while the body holds still.
const ROTOR_SPINS: [RotorSpin; ROTOR_COUNT] = [
    RotorSpin::CounterClockwise,
    RotorSpin::Clockwise,
    RotorSpin::CounterClockwise,
    RotorSpin::Clockwise,
];

/// How much a rotor twists the body about z for each newton it pushes with, from the motor gearing
/// in `x2.xml`.
const TORQUE_PER_THRUST: f64 = 0.0201;

/// The most one rotor can push with, from the motors' control range in `x2.xml`.
const MAXIMUM_ROTOR_THRUST: f64 = 13.0;

/// The least a rotor is asked for while it is running, as a share of its hovering push.
///
/// The file lets a motor stop dead. A stopped rotor cannot be pushed any lower, so there is no
/// spread left between the rotors to twist the body with, and the body loses the ability to hold
/// itself upright at the worst possible moment. Keeping a quarter of the hovering push as a floor
/// costs nothing and keeps that spread available.
const IDLE_THRUST_FRACTION: f64 = 0.25;

/// Where the model file sits, relative to this crate, and where the shape it is drawn as sits
/// beside it.
const MODEL_FILE: &str = "../third_party/menagerie/skydio_x2/x2.xml";
const MESH_FILE: &str = "../third_party/menagerie/skydio_x2/assets/X2_lowpoly.obj";

/// How much the drawn shape has to shrink, from `<mesh scale="0.01 0.01 0.01"/>` in `x2.xml`.
///
/// The shape file is drawn in centimetres and says so nowhere. Left as it comes it is about 55
/// metres across instead of 0.55, nothing complains, and a viewer that frames itself on a 55 metre
/// object puts its camera a hundred metres out — at which point the whole course is smaller than a
/// pixel and it reads exactly like nothing was ever drawn.
const MESH_SCALE: f64 = 0.01;

/// How the drawn shape is turned before it is drawn, from `quat="0 0 1 1"` on the geom in `x2.xml`,
/// written here in the same order the file uses.
///
/// It is a half turn about the line between the shape's own second and third axes, which is what
/// takes a shape drawn with its second axis pointing up and stands it up the way this world does.
const MESH_ROTATION: [f64; 4] = [
    0.0,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];

/// The machine's mass, its rotor layout, and the relation between the two.
///
/// The mass, the point the body balances on, and how hard it is to spin come from the model file.
/// Everything about the rotors is transcribed by hand. Both are settled once, at startup, so
/// nothing here has to be worked out again during a flight.
pub struct X2Model {
    body: RigidBody<f64>,
    mixer: MultirotorMixer<ROTOR_COUNT, f64>,
    distribution: Matrix<ROTOR_COUNT, 4, f64>,
    rotor_positions: [Vector3D<f64>; ROTOR_COUNT],
    hover_thrust_per_rotor: f64,
    mesh_size: Vector3D<f64>,
}

impl X2Model {
    /// Reads the model file and builds the body and the rotor layout from it.
    ///
    /// Returns whatever the reader, the body, or the rotor layout refuses on.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_FILE);
        let model = multicalc_robot_model::load_path(&path)?;
        let inertia = model
            .body_named("x2")
            .ok_or("the model file has no body called x2")?
            .inertia();

        let gravity = Vector::new([0.0, 0.0, -GRAVITY_STRENGTH]);
        let body = RigidBody::new(inertia, gravity)?;

        let hover_thrust_per_rotor = inertia.mass() * GRAVITY_STRENGTH / ROTOR_COUNT as f64;
        let rotor_positions = ROTOR_POSITIONS.map(Vector::new);
        let mixer = MultirotorMixer::<ROTOR_COUNT, f64>::new(
            rotor_positions,
            ROTOR_SPINS,
            TORQUE_PER_THRUST,
            IDLE_THRUST_FRACTION * hover_thrust_per_rotor,
            MAXIMUM_ROTOR_THRUST,
        )?;
        let distribution = mixer.allocation().pseudo_inverse()?;

        Ok(X2Model {
            body,
            mixer,
            distribution,
            rotor_positions,
            hover_thrust_per_rotor,
            mesh_size: measured_mesh_size(&mesh_path())?,
        })
    }

    /// Where the shape the machine is drawn as sits on disk.
    #[must_use]
    pub fn mesh_path(&self) -> std::path::PathBuf {
        mesh_path()
    }

    /// How much that shape has to shrink to be the size of the real machine.
    #[inline]
    #[must_use]
    pub fn mesh_scale(&self) -> f64 {
        MESH_SCALE
    }

    /// How it has to be turned before it stands up the way this world does.
    #[inline]
    #[must_use]
    pub fn mesh_rotation(&self) -> [f64; 4] {
        MESH_ROTATION
    }

    /// How big the drawn shape comes out, once shrunk and turned, in metres along each world axis.
    ///
    /// Measured from the file rather than written down, because a shape that comes out a hundred
    /// times too big does not fail — it just quietly puts the camera so far away that everything
    /// else disappears, and this is the number that says so before anything is drawn.
    #[inline]
    pub fn mesh_size(&self) -> Vector3D<f64> {
        self.mesh_size
    }

    /// The body itself: its mass, where it balances, and what forces do to it.
    #[inline]
    #[must_use]
    pub fn body(&self) -> RigidBody<f64> {
        self.body
    }

    /// How the body's mass is spread out.
    #[inline]
    #[must_use]
    pub fn inertia(&self) -> SpatialInertia<f64> {
        self.body.inertia()
    }

    /// The relation between the rotor thrusts and the push and turn the body feels.
    #[inline]
    #[must_use]
    pub fn mixer(&self) -> MultirotorMixer<ROTOR_COUNT, f64> {
        self.mixer
    }

    /// How a wanted push and turn is shared out across the rotors, before any limit is applied.
    ///
    /// This is the way back through the layout, worked out once here rather than every tick.
    #[inline]
    pub fn distribution(&self) -> Matrix<ROTOR_COUNT, 4, f64> {
        self.distribution
    }

    /// Where each rotor sits in the body's own axes.
    #[inline]
    pub fn rotor_positions(&self) -> [Vector3D<f64>; ROTOR_COUNT] {
        self.rotor_positions
    }

    /// What one rotor has to push with to carry a quarter of the machine's weight.
    #[inline]
    #[must_use]
    pub fn hover_thrust_per_rotor(&self) -> f64 {
        self.hover_thrust_per_rotor
    }

    /// The machine's mass, in kilograms.
    #[inline]
    #[must_use]
    pub fn mass(&self) -> f64 {
        self.body.inertia().mass()
    }

    /// Every number the model was built from, ready to print.
    #[must_use]
    pub fn report(&self) -> String {
        let inertia = self.body.inertia();
        let balance_point = inertia.center_of_mass();
        let spin = inertia.rotational_inertia();
        let mut lines = String::new();
        lines.push_str(&format!("mass                  {:.4} kg\n", inertia.mass()));
        lines.push_str(&format!(
            "balances at           ({:.4}, {:.4}, {:.4}) m above its own origin\n",
            balance_point[0], balance_point[1], balance_point[2]
        ));
        lines.push_str(&format!(
            "resists spinning      [{:.6} {:.6} {:.6} / {:.6} {:.6} {:.6} / {:.6} {:.6} {:.6}] kg m^2\n",
            spin[(0, 0)],
            spin[(0, 1)],
            spin[(0, 2)],
            spin[(1, 0)],
            spin[(1, 1)],
            spin[(1, 2)],
            spin[(2, 0)],
            spin[(2, 1)],
            spin[(2, 2)],
        ));
        for (rotor, (position, spin)) in self.rotor_positions.iter().zip(ROTOR_SPINS).enumerate() {
            let spin_direction = match spin {
                RotorSpin::Clockwise => "clockwise",
                RotorSpin::CounterClockwise => "counter-clockwise",
            };
            lines.push_str(&format!(
                "rotor {}               ({:+.2}, {:+.2}, {:+.2}) m, turns {}\n",
                rotor + 1,
                position[0],
                position[1],
                position[2],
                spin_direction,
            ));
        }
        lines.push_str(&format!(
            "twist per push        {TORQUE_PER_THRUST:.4} N m per N\n"
        ));
        lines.push_str(&format!(
            "one rotor pushes      {:.4} N to {:.4} N (hovering share {:.4} N)\n",
            self.mixer.minimum_thrust(),
            self.mixer.maximum_thrust(),
            self.hover_thrust_per_rotor,
        ));
        lines.push_str(&format!(
            "drawn as a shape      {:.3} x {:.3} x {:.3} m, once shrunk {:.2}x and stood up\n",
            self.mesh_size[0], self.mesh_size[1], self.mesh_size[2], MESH_SCALE,
        ));
        lines
    }
}

/// Where the drawn shape sits, relative to this crate.
fn mesh_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(MESH_FILE)
}

/// How big the drawn shape is along each world axis, once shrunk and turned.
///
/// Only the corner points of the shape are read — every line of the file that starts with a `v` —
/// which is enough to say how big it is and cheap enough to do once at startup.
fn measured_mesh_size(path: &Path) -> Result<Vector3D<f64>, Box<dyn Error>> {
    let turn = SO3::from_quaternion(Quaternion::new(
        MESH_ROTATION[0],
        MESH_ROTATION[1],
        MESH_ROTATION[2],
        MESH_ROTATION[3],
    ));
    let mut lowest: Vector3D<f64> = Vector::new([f64::INFINITY; 3]);
    let mut highest: Vector3D<f64> = Vector::new([f64::NEG_INFINITY; 3]);
    for line in std::fs::read_to_string(path)?.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some("v") {
            continue;
        }
        let mut corner = [0.0; 3];
        for slot in &mut corner {
            let Some(Ok(value)) = parts.next().map(str::parse::<f64>) else {
                continue;
            };
            *slot = value;
        }
        let placed: Vector3D<f64> = turn.act(Vector::new(corner)).scale(MESH_SCALE);
        for axis in 0..3 {
            lowest[axis] = lowest[axis].min(placed[axis]);
            highest[axis] = highest[axis].max(placed[axis]);
        }
    }
    if !lowest.is_finite() || !highest.is_finite() {
        return Err("the drawn shape has no corners in it".into());
    }
    Ok(highest - lowest)
}
