#![deny(clippy::indexing_slicing)]

//! The pieces RRT, RRT\* and PRM share: nearest neighbour, steering, edge checking, and pulling a
//! path out of a tree.

use crate::error::PlanningError;
use crate::linear_algebra::Vector;
use crate::motion::PolylinePath;
use crate::planning::state_space::{StateSpace, StateValidity};
use crate::scalar::Numeric;

/// A parent slot meaning the node is a root.
pub(crate) const NO_PARENT: u32 = u32::MAX;

/// The node nearest `target`, by the space's own metric.
///
/// A linear scan, so O(n) a call and O(n²) over a whole build. At `MAX_NODES = 2000` that is about
/// 4 M distance evaluations — milliseconds, off-loop. A k-d tree is *not* the upgrade: its
/// axis-split rule assumes a Euclidean product space, which is false for the wrapped and
/// manifold-valued joints [`StateSpace`] exists to support, so it would make the answer wrong on
/// exactly those states.
pub(crate) fn nearest_index<const DIMENSION: usize, T: Numeric, S: StateSpace<DIMENSION, T>>(
    space: &S,
    states: &[Vector<DIMENSION, T>],
    target: &Vector<DIMENSION, T>,
) -> Option<usize> {
    let mut best: Option<(usize, T)> = None;
    for (index, state) in states.iter().enumerate() {
        let separation = space.distance(state, target);
        let closer = match best {
            None => true,
            Some((_, shortest)) => separation < shortest,
        };
        if closer {
            best = Some((index, separation));
        }
    }
    best.map(|(index, _)| index)
}

/// Hands every node within `radius` of `centre` to `visit`, with its distance.
///
/// The same single pass as [`nearest_index`], reporting through a callback so nothing is collected.
pub(crate) fn neighbours_within<const DIMENSION: usize, T: Numeric, S, F>(
    space: &S,
    states: &[Vector<DIMENSION, T>],
    centre: &Vector<DIMENSION, T>,
    radius: T,
    mut visit: F,
) where
    S: StateSpace<DIMENSION, T>,
    F: FnMut(usize, T),
{
    for (index, state) in states.iter().enumerate() {
        let separation = space.distance(state, centre);
        if separation <= radius {
            visit(index, separation);
        }
    }
}

/// `to` if it is within `step`, else the state `step` along the way to it.
pub(crate) fn steer_towards<const DIMENSION: usize, T: Numeric, S: StateSpace<DIMENSION, T>>(
    space: &S,
    from: &Vector<DIMENSION, T>,
    into: &Vector<DIMENSION, T>,
    step: T,
) -> Vector<DIMENSION, T> {
    let separation = space.distance(from, into);
    if separation <= step || separation == T::ZERO {
        return *into;
    }
    space.interpolate(from, into, step / separation)
}

/// Whether the segment between two states is free, tested at `checks` interior stations plus the
/// far end.
///
/// This is a **discrete** check, not continuous collision detection: an obstacle thinner than the
/// spacing between two stations is missed. Raise `checks` where that matters.
pub(crate) fn edge_is_valid<const DIMENSION: usize, T: Numeric, S, V>(
    space: &S,
    validity: &V,
    from: &Vector<DIMENSION, T>,
    into: &Vector<DIMENSION, T>,
    checks: usize,
) -> bool
where
    S: StateSpace<DIMENSION, T>,
    V: StateValidity<DIMENSION, T>,
{
    let stations = T::from_usize(checks + 1);
    for station in 1..=checks {
        let amount = T::from_usize(station) / stations;
        if !validity.is_state_valid(&space.interpolate(from, into, amount)) {
            return false;
        }
    }
    validity.is_state_valid(into)
}

/// The path from the tree's root down to `leaf`, root first.
///
/// Returns [`PlanningError::PathCapacityExceeded`] with the length it would need, so a caller can
/// resize and retry in one round trip.
pub(crate) fn extract_tree_path<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric>(
    states: &[Vector<DIMENSION, T>],
    parents: &[u32],
    leaf: usize,
) -> Result<PolylinePath<MAX_POINTS, DIMENSION, T>, PlanningError> {
    let budget = states.len();

    // Count first, so an overlong chain reports the size it needs rather than failing part-written.
    let mut needed = 0usize;
    let mut index = leaf;
    for _ in 0..=budget {
        if states.get(index).is_none() {
            break;
        }
        needed += 1;
        match parents.get(index).copied() {
            Some(NO_PARENT) | None => break,
            Some(parent) => index = parent as usize,
        }
    }
    if needed > MAX_POINTS {
        return Err(PlanningError::PathCapacityExceeded { needed });
    }

    // Stage leaf-to-root, then push in reverse so the path runs root first.
    let mut staged = [Vector::<DIMENSION, T>::zeros(); MAX_POINTS];
    let mut filled = 0usize;
    let mut index = leaf;
    for _ in 0..=budget {
        let Some(&state) = states.get(index) else {
            break;
        };
        if let Some(slot) = staged.get_mut(filled) {
            *slot = state;
            filled += 1;
        }
        match parents.get(index).copied() {
            Some(NO_PARENT) | None => break,
            Some(parent) => index = parent as usize,
        }
    }

    let mut path = PolylinePath::<MAX_POINTS, DIMENSION, T>::new();
    for slot in (0..filled).rev() {
        let Some(state) = staged.get(slot).copied() else {
            continue;
        };
        path.push(state)?;
    }
    Ok(path)
}

#[cfg(test)]
mod test;
