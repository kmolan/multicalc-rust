//! Working out where in the room the machine is before it flies anywhere: it hovers, turns on the
//! spot, and a cloud of guesses is scored against the floor plan until they agree on one place.
//! None of it runs once the machine is flying.

use std::error::Error;

use multicalc::estimation::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::scalar::Numeric;
use multicalc::spatial::{FreeJointState, SO3};
use rand_pcg::Pcg32;

use crate::sim::lidar::Lidar2d;

use super::flight_estimator::EstimatedState;
use super::flight_hangar::FlightHangar;
use super::flight_plant::{angle_from_upright, level_heading};

/// How many beams the sideways-looking scanner has, how far it reaches, and how much its answer
/// jitters.
///
/// The beams are spread evenly the whole way round rather than across an arc in front: a machine
/// working out where it is in a room wants the wall behind it as much as the one ahead. The field
/// of view stops one beam short of a full turn so the last beam does not land on top of the first.
pub const LIDAR_BEAMS: usize = 24;
const LIDAR_RANGE: f64 = 9.0;
const LIDAR_NOISE: f64 = 0.03;
const LIDAR_DROPOUT: f64 = 0.02;

/// How many guesses the search carries, how far they are scattered to begin with, and how far apart
/// they may end up before the answer counts as settled.
///
/// The machine knows roughly where it was set down, to within a metre or two, and nothing at all
/// about which way it is pointing — so the guesses start scattered about the pad and facing every
/// which way, and the beams have to do the rest. The two numbers are spreads, so the heading one
/// being four means a guess may be pointing anywhere.
const LOCALIZER_GUESSES: usize = 600;
const LOCALIZER_POSITION_SCATTER: f64 = 2.25;
const LOCALIZER_HEADING_SCATTER: f64 = 4.0;
const LOCALIZER_SETTLED_POSITION: f64 = 0.25;
const LOCALIZER_SETTLED_HEADING: f64 = 0.10;

/// How gently a single scan is allowed to judge the guesses.
///
/// A scan of two dozen beams, each judged harshly, decides the whole thing on its own: one guess
/// takes all the weight, every other is thrown away, and the cloud has settled on an answer after
/// one look — which is a cloud that has collapsed, not one that has found anything. Judging each
/// beam gently leaves the guesses spread out long enough for several looks from several headings to
/// agree with each other, and it is the agreement that is worth having.
const BEAM_FORGIVENESS: f64 = 0.30;
const BEAM_MISMATCH_PENALTY: f64 = -1.0;

/// How much the guesses are shaken up on every move, as a spread on where and which way.
///
/// Without this they only ever narrow, so an early wrong answer can never be talked out of itself.
const LOCALIZER_SHAKE_UP: [f64; 3] = [0.01, 0.01, 0.01];

/// How many scans have to agree before the answer is taken as found.
///
/// At ten scans a second and half a radian a second of turning, this is a few seconds and most of a
/// half turn — enough that the walls have been seen from properly different directions.
const LOCALIZER_LEAST_SCANS: u64 = 30;

/// How often the scanner is read while the machine is working out where it is, in ticks.
pub const SCAN_PERIOD_TICKS: u64 = 100;

/// How high the machine sits while it works out where it is, in metres.
///
/// Low, because that is where a machine that has just been set down is: it looks around from just
/// above the pad and only climbs onto the path once it knows where it is standing. Leaving the
/// climb until afterwards also means the first thing the path asks for is a real manoeuvre rather
/// than a body already sitting on the answer.
pub const PAD_HEIGHT: f64 = 0.35;

/// How fast the machine turns on the spot while it looks around, in radians per second, and how
/// long it is given before it flies anyway.
///
/// Turning is what makes the search work: a machine holding still sees the same walls for ever, and
/// every guess pointing the wrong way stays as good as the right one.
const LOCALIZER_TURN_RATE: f64 = 0.5;
const LOCALIZER_SECONDS_ALLOWED: f64 = 30.0;

/// How far the machine may be leaning before a scan is thrown away rather than matched.
///
/// The beams are bolted to the body, so a leaning machine sweeps them through the floor and the
/// ceiling instead of round the walls. Levelling the reading covers a small lean; past this, what
/// comes back is not a slice of the room at all and scoring the guesses on it would be scoring them
/// on a lie.
const LOCALIZER_WORST_LEAN: f64 = 0.25;

/// The search for where in the room the machine is: a scanner, a cloud of guesses, and the slow turn
/// that makes the two worth having.
pub struct StartupLocalization {
    lidar: Lidar2d<LIDAR_BEAMS>,
    guesses: MonteCarloLocalizer<LIDAR_BEAMS>,
    wanted_heading: f64,
    last_guess: Vector3D<f64>,
    scans_matched: u64,
    scans_refused: u64,
}

impl StartupLocalization {
    /// A search that starts from where the machine was set down, facing whichever way it was left
    /// facing.
    ///
    /// `pad` is where it is taken to have been put and `starting_heading` which way its nose points;
    /// the guesses are scattered about the first and spread over every direction, because the pad is
    /// surveyed and the facing is not known at all. `seed` settles every draw the search makes.
    ///
    /// Returns whatever the cloud of guesses refuses to be scattered as.
    pub fn new(
        pad: Vector3D<f64>,
        starting_heading: f64,
        seed: u64,
    ) -> Result<Self, Box<dyn Error>> {
        let lidar = Lidar2d::<LIDAR_BEAMS>::new(
            std::f64::consts::TAU * (LIDAR_BEAMS - 1) as f64 / LIDAR_BEAMS as f64,
            LIDAR_RANGE,
            LIDAR_NOISE,
            LIDAR_DROPOUT,
        )?;
        let mut guesses = MonteCarloLocalizer::<LIDAR_BEAMS>::new(
            [pad[0], pad[1], 0.0],
            InitialParticleCloud {
                particle_count: LOCALIZER_GUESSES,
                position_variance: LOCALIZER_POSITION_SCATTER,
                heading_variance: LOCALIZER_HEADING_SCATTER,
            },
            BeamModel {
                range_deviation: BEAM_FORGIVENESS,
                mismatch_penalty: BEAM_MISMATCH_PENALTY,
                ..Default::default()
            },
            seed,
        )?;
        guesses.set_motion_noise(LOCALIZER_SHAKE_UP)?;

        Ok(StartupLocalization {
            lidar,
            guesses,
            wanted_heading: starting_heading,
            last_guess: Vector::new([pad[0], pad[1], starting_heading]),
            scans_matched: 0,
            scans_refused: 0,
        })
    }

    /// The cloud of guesses about where in the room the machine is.
    #[inline]
    #[must_use]
    pub fn guesses(&self) -> &MonteCarloLocalizer<LIDAR_BEAMS> {
        &self.guesses
    }

    /// How many scans were matched against the floor plan.
    #[inline]
    #[must_use]
    pub fn scans_matched(&self) -> u64 {
        self.scans_matched
    }

    /// How many scans were thrown away, for being taken at too much of a lean or for being ones the
    /// guesses could not be scored against.
    #[inline]
    #[must_use]
    pub fn scans_refused(&self) -> u64 {
        self.scans_refused
    }

    /// Turns the machine a tick's worth further round, and says which way its nose should now point.
    pub fn turn_on_the_spot(&mut self, timestep: f64) -> f64 {
        self.wanted_heading += LOCALIZER_TURN_RATE * timestep;
        self.wanted_heading
    }

    /// Whether the search is over, either because the guesses have agreed or because it has had
    /// `seconds_elapsed` and clearly is not going to.
    ///
    /// Agreed means several scans from several headings said the same thing, not that one scan
    /// happened to leave the guesses close together: a cloud that has collapsed onto one guess looks
    /// exactly like a cloud that has agreed on one.
    #[must_use]
    pub fn is_finished(&self, seconds_elapsed: f64) -> bool {
        let settled = self.scans_matched >= LOCALIZER_LEAST_SCANS
            && self
                .guesses
                .is_converged(LOCALIZER_SETTLED_POSITION, LOCALIZER_SETTLED_HEADING);
        settled || seconds_elapsed > LOCALIZER_SECONDS_ALLOWED
    }

    /// Takes one scan of the room and lets the guesses answer for it.
    ///
    /// The beams are bolted to the body, so a lean sends them off toward the floor and the ceiling
    /// instead of round the walls. Each reading is brought back to the flat by how much of its beam
    /// was pointing along the floor at all; past a quarter of a radian of lean there is not enough
    /// of it left to be worth correcting, and the scan is thrown away instead of matched.
    ///
    /// `truth` is only ever handed to the beams, which is what a real room would do to them. Where
    /// the guesses are moved to comes from `believed` and never from the truth.
    pub fn match_one_scan(
        &mut self,
        hangar: &FlightHangar,
        truth: FreeJointState<f64>,
        believed: EstimatedState,
        rng: &mut Pcg32,
    ) {
        if angle_from_upright(truth) > LOCALIZER_WORST_LEAN {
            self.scans_refused += 1;
            return;
        }

        let pose = truth.pose();
        let scan = self.lidar.simulate(
            hangar.grid(),
            [
                pose.translation()[0],
                pose.translation()[1],
                level_heading(pose.rotation()),
            ],
            rng,
        );
        let levelled = self.levelled_scan(&scan, believed.orientation);
        let geometry = self.lidar.geometry();

        // How far the machine believes it has moved and turned since the last scan.
        let guess = Vector::new([
            believed.position[0],
            believed.position[1],
            level_heading(believed.orientation),
        ]);
        let moved = (guess[0] - self.last_guess[0]).hypot(guess[1] - self.last_guess[1]);
        let turned = (guess[2] - self.last_guess[2]).wrap_to_pi();
        self.last_guess = guess;

        let matched = self.guesses.predict(moved, turned).is_ok()
            && self
                .guesses
                .update(&levelled, hangar.grid(), &geometry)
                .is_ok();
        if matched {
            self.scans_matched += 1;
        } else {
            self.scans_refused += 1;
        }
    }

    /// A scan brought back to the flat, as it would read from a body sitting level.
    ///
    /// A beam that has been tipped away from the floor covers less ground than it travels; how much
    /// less is how much of its direction is still pointing along the floor.
    fn levelled_scan(
        &self,
        scan: &[f64; LIDAR_BEAMS],
        orientation: SO3<f64>,
    ) -> [f64; LIDAR_BEAMS] {
        std::array::from_fn(|beam| {
            let range = scan[beam];
            let Some(angle) = self.lidar.beam_angle(beam) else {
                return f64::INFINITY;
            };
            if !range.is_finite() {
                return range;
            }
            let pointing: Vector3D<f64> =
                orientation.act(Vector::new([angle.cos(), angle.sin(), 0.0]));
            range * pointing[0].hypot(pointing[1])
        })
    }
}
