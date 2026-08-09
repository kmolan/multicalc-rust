//! The room the machine flies in, drawn as a flat floor plan it can match itself against. Its shape
//! is deliberately lopsided, so there is only one place the machine can be standing.

use multicalc::error::MappingError;
use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap};

use crate::sim::geometry::box_outline;

/// How far the room reaches from the middle, along and across, in metres.
///
/// The flight loop swings out to 4.3 m, so this leaves better than a metre of clearance everywhere,
/// and the two lengths differ so the room cannot be mistaken for its own quarter turn.
const HANGAR_HALF_LENGTH: f64 = 6.5;
const HANGAR_HALF_WIDTH: f64 = 5.5;

/// How big one cell of the floor plan is, in metres.
const CELL_SIZE: f64 = 0.05;

/// How much clear space is left around the room in the plan, in metres, so the wall itself has
/// somewhere to be drawn.
const PLAN_MARGIN: f64 = 0.3;

/// Where the two pillars stand and how thick they are, in metres.
///
/// Both sit well outside the flight loop and nowhere near each other, which is what turns a room
/// that still looks like itself upside down into one that does not.
const PILLARS: [([f64; 2], f64); 2] = [([5.2, 3.6], 0.25), ([-5.4, -2.0], 0.35)];

/// How far the step in the long wall juts into the room, and how wide it is, in metres.
///
/// It stands well clear of the flight loop, which reaches 4.3 m from the middle.
const STEP_DEPTH: f64 = 0.9;
const STEP_HALF_WIDTH: f64 = 0.8;

/// The room, and the rough idea the machine starts with about where in it it is.
#[derive(Debug, Clone, PartialEq)]
pub struct FlightHangar {
    /// The walls: the plan the machine matches against, and the surface its beams come back off.
    grid: DynamicOccupancyGrid,
    /// The first guess the search starts from, as `[x, y, heading]`.
    ///
    /// It is a rough one on purpose: somewhere near the middle of the room, facing nowhere in
    /// particular. Handing over the answer would not be a search.
    starting_guess: [f64; 3],
}

impl FlightHangar {
    /// The walls, for casting beams against and for drawing.
    #[inline]
    #[must_use]
    pub fn grid(&self) -> &DynamicOccupancyGrid {
        &self.grid
    }

    /// The rough idea the search starts from.
    #[inline]
    #[must_use]
    pub fn starting_guess(&self) -> [f64; 3] {
        self.starting_guess
    }
}

/// Builds the room: four walls, a step jutting in from the long one, and two pillars.
///
/// Returns whatever the floor plan refuses to be built at that size.
pub fn flight_hangar() -> Result<FlightHangar, MappingError> {
    let columns = (2.0 * (HANGAR_HALF_LENGTH + PLAN_MARGIN) / CELL_SIZE).ceil() as usize;
    let rows = (2.0 * (HANGAR_HALF_WIDTH + PLAN_MARGIN) / CELL_SIZE).ceil() as usize;
    let corner = [
        -HANGAR_HALF_LENGTH - PLAN_MARGIN,
        -HANGAR_HALF_WIDTH - PLAN_MARGIN,
    ];
    let mut grid = DynamicOccupancyGrid::try_new(columns, rows, CELL_SIZE, corner)?;

    grid.occupy_polyline(
        &box_outline(
            [-HANGAR_HALF_LENGTH, -HANGAR_HALF_WIDTH],
            [HANGAR_HALF_LENGTH, HANGAR_HALF_WIDTH],
        ),
        true,
    );

    // A step jutting in from the far wall: three sides of a box whose fourth side is the wall
    // itself. It changes what the room looks like from one end without narrowing the flying room.
    let step_face = HANGAR_HALF_LENGTH - STEP_DEPTH;
    grid.occupy_polyline(
        &[
            [HANGAR_HALF_LENGTH, -STEP_HALF_WIDTH],
            [step_face, -STEP_HALF_WIDTH],
            [step_face, STEP_HALF_WIDTH],
            [HANGAR_HALF_LENGTH, STEP_HALF_WIDTH],
        ],
        false,
    );

    for (middle, half_width) in PILLARS {
        grid.occupy_polyline(
            &box_outline(
                [middle[0] - half_width, middle[1] - half_width],
                [middle[0] + half_width, middle[1] + half_width],
            ),
            true,
        );
    }

    Ok(FlightHangar {
        grid,
        starting_guess: [0.0, 0.0, 0.0],
    })
}
