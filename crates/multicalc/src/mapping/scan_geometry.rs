#![deny(clippy::indexing_slicing)]

//! The directions the beams of a forward-facing range scan point.

use crate::error::MappingError;
use crate::scalar::{Numeric, Primal};

/// A forward-facing scan of `NUM_BEAMS` beams spread evenly across its field of view.
///
/// Beam directions are measured from straight ahead and grow to the left, so beam `0` points at the
/// right edge of the arc and the last beam at the left edge, with the middle beam straight ahead
/// when there is an odd number of them.
///
/// This is the shape of a scan, not a sensor: it says where each beam points and how far it can
/// see, and nothing about noise or how often it reads.
///
/// ```
/// use multicalc::mapping::ScanGeometry;
///
/// // A five-beam scan across a quarter turn, reaching four metres.
/// const NUM_BEAMS: usize = 5;
/// let quarter_turn = core::f64::consts::FRAC_PI_2;
/// let maximum_range = 4.0;
/// let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
///
/// // Beam 0 sits at the right edge of the arc, the middle beam points straight ahead, and the
/// // last sits at the left edge. Past the last beam there is nothing to ask about.
/// let half_the_arc = quarter_turn / 2.0;
/// assert!(scan.beam_angle(0).is_some_and(|angle| (angle + half_the_arc).abs() < 1e-12));
/// assert!(scan.beam_angle(2).is_some_and(|angle| angle.abs() < 1e-12));
/// assert!(scan.beam_angle(4).is_some_and(|angle| (angle - half_the_arc).abs() < 1e-12));
/// assert_eq!(scan.beam_angle(NUM_BEAMS), None);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanGeometry<const NUM_BEAMS: usize, T: Numeric = f64> {
    /// How wide an angle the scan covers.
    field_of_view: T,
    /// How close a reading can be and still be believed.
    minimum_range: T,
    /// How far the scan reaches.
    maximum_range: T,
}

impl<const NUM_BEAMS: usize, T: Numeric> ScanGeometry<NUM_BEAMS, T> {
    /// A scan across `field_of_view` radians reaching `maximum_range`.
    ///
    /// The closest range it can see to starts at zero, which is a sensor with no blind spot in
    /// front of it; [`with_minimum_range`](Self::with_minimum_range) sets one.
    ///
    /// Returns [`MappingError::TooFewBeams`] if `NUM_BEAMS` is below two,
    /// [`MappingError::NonFinite`] if either value is infinite or NaN,
    /// [`MappingError::InvalidFieldOfView`] if `field_of_view` is outside `(0, 2π]`, and
    /// [`MappingError::NonPositiveRange`] if `maximum_range` is zero or negative.
    pub fn try_new(field_of_view: T, maximum_range: T) -> Result<Self, MappingError> {
        if NUM_BEAMS < 2 {
            return Err(MappingError::TooFewBeams);
        }
        if !field_of_view.is_finite() || !maximum_range.is_finite() {
            return Err(MappingError::NonFinite);
        }
        if field_of_view <= T::ZERO || field_of_view > T::TWO_PI {
            return Err(MappingError::InvalidFieldOfView);
        }
        if maximum_range <= T::ZERO {
            return Err(MappingError::NonPositiveRange);
        }
        Ok(ScanGeometry {
            field_of_view,
            minimum_range: T::ZERO,
            maximum_range,
        })
    }

    /// Sets how close a reading can be and still be believed — the blind spot a sensor has right in
    /// front of it, inside which it reports nothing usable.
    ///
    /// Returns [`MappingError::NonFinite`] if `minimum_range` is infinite or NaN, and
    /// [`MappingError::InvalidRangeLimits`] if it is negative or reaches
    /// [`maximum_range`](Self::maximum_range).
    ///
    /// ```
    /// use multicalc::mapping::ScanGeometry;
    ///
    /// // A scanner blind inside 12 cm, seeing out to four metres.
    /// let quarter_turn = core::f64::consts::FRAC_PI_2;
    /// let blind_spot = 0.12;
    /// let maximum_range = 4.0;
    /// let scan: ScanGeometry<3> =
    ///     ScanGeometry::try_new(quarter_turn, maximum_range)?.with_minimum_range(blind_spot)?;
    /// assert_eq!(scan.minimum_range(), blind_spot);
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    pub fn with_minimum_range(mut self, minimum_range: T) -> Result<Self, MappingError> {
        if !minimum_range.is_finite() {
            return Err(MappingError::NonFinite);
        }
        if minimum_range < T::ZERO || minimum_range >= self.maximum_range {
            return Err(MappingError::InvalidRangeLimits);
        }
        self.minimum_range = minimum_range;
        Ok(self)
    }

    /// How wide an angle the scan covers, in radians.
    #[inline]
    #[must_use]
    pub fn field_of_view(&self) -> T {
        self.field_of_view
    }

    /// How close a reading can be and still be believed. Zero when the sensor has no blind spot.
    #[inline]
    #[must_use]
    pub fn minimum_range(&self) -> T {
        self.minimum_range
    }

    /// How far the scan reaches.
    #[inline]
    #[must_use]
    pub fn maximum_range(&self) -> T {
        self.maximum_range
    }

    /// How many beams the scan carries.
    #[inline]
    #[must_use]
    pub fn num_beams(&self) -> usize {
        NUM_BEAMS
    }

    /// The angle between one beam and the next.
    ///
    /// ```
    /// use multicalc::mapping::ScanGeometry;
    ///
    /// // Five beams across a quarter turn leave four gaps between them.
    /// let quarter_turn = core::f64::consts::FRAC_PI_2;
    /// let maximum_range = 4.0;
    /// let scan: ScanGeometry<5> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
    ///
    /// let expected = quarter_turn / 4.0;
    /// assert!((scan.angle_increment() - expected).abs() < 1e-12);
    /// assert_eq!(scan.num_beams(), 5);
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn angle_increment(&self) -> T {
        // `try_new` rejects fewer than two beams, so this never divides by zero.
        self.field_of_view / T::from_usize(NUM_BEAMS - 1)
    }

    /// Whether a reading is one the sensor could actually have taken: a real distance, no closer
    /// than [`minimum_range`](Self::minimum_range) and no further than
    /// [`maximum_range`](Self::maximum_range).
    ///
    /// A beam that met nothing reads as infinity, which is not a distance and so is not valid here.
    ///
    /// ```
    /// use multicalc::mapping::ScanGeometry;
    ///
    /// // A scanner blind inside 12 cm, seeing out to four metres.
    /// let quarter_turn = core::f64::consts::FRAC_PI_2;
    /// let blind_spot = 0.12;
    /// let maximum_range = 4.0;
    /// let scan: ScanGeometry<3> =
    ///     ScanGeometry::try_new(quarter_turn, maximum_range)?.with_minimum_range(blind_spot)?;
    ///
    /// let a_wall_two_metres_off = 2.0;
    /// assert!(scan.range_is_valid(a_wall_two_metres_off));
    ///
    /// // Inside the blind spot, past the range, and no return at all: none of them are readings.
    /// let inside_the_blind_spot = 0.05;
    /// let past_the_range = 5.0;
    /// assert!(!scan.range_is_valid(inside_the_blind_spot));
    /// assert!(!scan.range_is_valid(past_the_range));
    /// assert!(!scan.range_is_valid(f64::INFINITY));
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn range_is_valid(&self, range: T) -> bool {
        range.is_finite() && range >= self.minimum_range && range <= self.maximum_range
    }

    /// The direction beam `index` points, measured from straight ahead, or `None` if there is no
    /// such beam.
    ///
    /// ```
    /// use multicalc::mapping::ScanGeometry;
    ///
    /// // Three beams across a quarter turn: one at each edge, one straight ahead.
    /// let quarter_turn = core::f64::consts::FRAC_PI_2;
    /// let maximum_range = 4.0;
    /// let scan: ScanGeometry<3> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
    ///
    /// let eighth_turn = quarter_turn / 2.0;
    /// assert!(scan.beam_angle(0).is_some_and(|angle| (angle + eighth_turn).abs() < 1e-12));
    /// assert!(scan.beam_angle(1).is_some_and(|angle| angle.abs() < 1e-12));
    /// assert!(scan.beam_angle(2).is_some_and(|angle| (angle - eighth_turn).abs() < 1e-12));
    ///
    /// // There is no fourth beam to ask about.
    /// assert_eq!(scan.beam_angle(3), None);
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn beam_angle(&self, index: usize) -> Option<T> {
        (index < NUM_BEAMS).then(|| beam_angle_across(self.field_of_view, NUM_BEAMS, index))
    }

    /// Each beam's angle from straight ahead, in order.
    pub fn beams(&self) -> impl Iterator<Item = T> + '_ {
        (0..NUM_BEAMS).filter_map(|index| self.beam_angle(index))
    }
}

/// Turning a direction back into a beam needs a whole number, which only a plain scalar can give,
/// so this one method asks for a little more of its scalar than the rest of the type does.
impl<const NUM_BEAMS: usize, T: Numeric + Primal> ScanGeometry<NUM_BEAMS, T> {
    /// The beam pointing closest to `angle`, measured from straight ahead, or `None` when the
    /// direction falls outside the arc the scan covers.
    ///
    /// The reverse of [`beam_angle`](Self::beam_angle): it answers which beam is looking at
    /// something you know the direction of. A direction between two beams goes to the nearer one.
    ///
    /// ```
    /// use multicalc::mapping::ScanGeometry;
    ///
    /// // Three beams across a quarter turn, so they sit at -45°, 0° and +45°.
    /// let quarter_turn = core::f64::consts::FRAC_PI_2;
    /// let maximum_range = 4.0;
    /// let scan: ScanGeometry<3> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
    ///
    /// // Straight ahead is the middle beam, and something just off it still is.
    /// let straight_ahead = 0.0;
    /// let a_little_to_the_left = 0.1;
    /// assert_eq!(scan.nearest_beam_index(straight_ahead), Some(1));
    /// assert_eq!(scan.nearest_beam_index(a_little_to_the_left), Some(1));
    ///
    /// // Far enough left and the last beam is nearer.
    /// let well_to_the_left = 0.7;
    /// assert_eq!(scan.nearest_beam_index(well_to_the_left), Some(2));
    ///
    /// // Behind the sensor there is no beam to answer with.
    /// let over_the_shoulder = 2.0;
    /// assert_eq!(scan.nearest_beam_index(over_the_shoulder), None);
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    #[must_use]
    pub fn nearest_beam_index(&self, angle: T) -> Option<usize> {
        let half_the_arc = self.field_of_view * T::HALF;
        if !angle.is_finite() || angle < -half_the_arc || angle > half_the_arc {
            return None;
        }
        // How many whole beam steps in from the right edge the direction sits, rounded to the
        // nearest one. The bounds above put this inside the scan, and the clamp keeps rounding at
        // an edge from reaching past it.
        let steps = ((angle + half_the_arc) / self.angle_increment())
            .round()
            .to_f64();
        if steps < 0.0 {
            return Some(0);
        }
        let steps = steps as usize;
        Some(steps.min(NUM_BEAMS - 1))
    }
}

/// The direction one beam of an evenly spread scan points, measured from straight ahead and growing
/// to the left.
///
/// Shared so that everything numbering beams agrees on which way each one faces. `num_beams` must be
/// at least two, which every caller checks when it is built.
///
/// ```
/// use multicalc::control::FollowTheGap;
/// use multicalc::mapping::ScanGeometry;
///
/// // Both types number their beams through this one formula, so a scan and the steering worked
/// // out from it agree beam for beam.
/// let quarter_turn = core::f64::consts::FRAC_PI_2;
/// let maximum_range = 4.0;
/// let scan: ScanGeometry<3> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
/// let chassis_width = 0.5;
/// let free_range_threshold = 0.6;
/// let cruise_speed = 0.4;
/// let follower: FollowTheGap<3> = FollowTheGap::try_new(
///     quarter_turn,
///     maximum_range,
///     chassis_width,
///     free_range_threshold,
///     cruise_speed,
/// )?;
///
/// for beam in 0..4 {
///     assert_eq!(scan.beam_angle(beam), follower.beam_angle(beam));
/// }
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[inline]
#[must_use]
pub(crate) fn beam_angle_across<T: Numeric>(field_of_view: T, num_beams: usize, index: usize) -> T {
    let span = T::from_usize(num_beams - 1);
    -field_of_view * T::HALF + field_of_view * T::from_usize(index) / span
}
