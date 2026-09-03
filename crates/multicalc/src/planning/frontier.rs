#![deny(clippy::indexing_slicing)]

//! A fixed-capacity binary min-heap with an index, so each item is queued at most once.

use crate::scalar::Numeric;

/// An item was pushed that the heap has no slot for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrontierFull;

/// A slot number meaning the item is not queued.
const NOT_IN_HEAP: u32 = u32::MAX;

/// A binary min-heap over item ids below `CAPACITY`, keyed by a scalar, with a decrease-key.
///
/// `slot_of` maps an item to where it sits, so re-pushing an item lowers its key in place rather
/// than queueing it twice. That is what bounds the heap at `CAPACITY` rather than at the number of
/// relaxations, so there is no second capacity parameter and no user-facing "frontier full".
///
/// Ordering is [`precedes`]: a strict `<` on the key with the item id breaking ties, never
/// `partial_cmp().unwrap()` — `T` is not `Ord`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Frontier<const CAPACITY: usize, T: Numeric = f64> {
    keys: [T; CAPACITY],
    items: [u32; CAPACITY],
    slot_of: [u32; CAPACITY],
    length: usize,
}

impl<const CAPACITY: usize, T: Numeric> Frontier<CAPACITY, T> {
    /// An empty heap.
    #[must_use]
    pub(crate) fn new() -> Self {
        Frontier {
            keys: [T::ZERO; CAPACITY],
            items: [0; CAPACITY],
            slot_of: [NOT_IN_HEAP; CAPACITY],
            length: 0,
        }
    }

    /// Empties the heap, forgetting every queued item.
    pub(crate) fn clear(&mut self) {
        for slot in self.slot_of.iter_mut() {
            *slot = NOT_IN_HEAP;
        }
        self.length = 0;
    }

    /// Empties the heap, forgetting only items below `items`.
    ///
    /// A search over a map smaller than the workspace pays for the map rather than the capacity.
    pub(crate) fn clear_prefix(&mut self, items: usize) {
        for slot in self.slot_of.iter_mut().take(items.min(CAPACITY)) {
            *slot = NOT_IN_HEAP;
        }
        self.length = 0;
    }

    /// How many items are queued.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.length
    }

    /// Whether the item is queued.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn contains(&self, item: usize) -> bool {
        self.slot_of
            .get(item)
            .is_some_and(|&slot| slot != NOT_IN_HEAP)
    }

    /// Queues `item` at `key`, or lowers its key if it is already queued and `key` is below it.
    ///
    /// A key that would raise a queued item's key is ignored. Returns [`FrontierFull`] for an item
    /// id the heap has no slot for.
    pub(crate) fn push_or_lower(&mut self, item: usize, key: T) -> Result<(), FrontierFull> {
        if item >= CAPACITY {
            return Err(FrontierFull);
        }
        let item_id = item as u32;

        if let Some(&slot) = self.slot_of.get(item)
            && slot != NOT_IN_HEAP
        {
            let slot = slot as usize;
            let current = self.keys.get(slot).copied().unwrap_or(T::INFINITY);
            // `!(key < current)` rather than `key >= current`: the two differ on a NaN key, and a
            // `partial_cmp().unwrap()` would panic on one. `T` is not `Ord`.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(key < current) {
                return Ok(());
            }
            self.write(slot, key, item_id);
            self.sift_up(slot);
            return Ok(());
        }

        if self.length >= CAPACITY {
            return Err(FrontierFull);
        }
        let slot = self.length;
        self.write(slot, key, item_id);
        self.length += 1;
        self.sift_up(slot);
        Ok(())
    }

    /// Removes and returns the lowest-keyed item.
    pub(crate) fn pop_minimum(&mut self) -> Option<(usize, T)> {
        if self.length == 0 {
            return None;
        }
        let item = self.items.first().copied()?;
        let key = self.keys.first().copied()?;
        if let Some(slot) = self.slot_of.get_mut(item as usize) {
            *slot = NOT_IN_HEAP;
        }

        self.length -= 1;
        if self.length > 0 {
            let (last_key, last_item) = self.read(self.length)?;
            self.write(0, last_key, last_item);
            self.sift_down(0);
        }
        Some((item as usize, key))
    }

    /// The key and item at a slot.
    fn read(&self, slot: usize) -> Option<(T, u32)> {
        Some((
            self.keys.get(slot).copied()?,
            self.items.get(slot).copied()?,
        ))
    }

    /// Places an entry at a slot and records where its item now sits.
    fn write(&mut self, slot: usize, key: T, item: u32) {
        if let Some(stored) = self.keys.get_mut(slot) {
            *stored = key;
        }
        if let Some(stored) = self.items.get_mut(slot) {
            *stored = item;
        }
        if let Some(stored) = self.slot_of.get_mut(item as usize) {
            *stored = slot as u32;
        }
    }

    fn sift_up(&mut self, from: usize) {
        let mut slot = from;
        let Some(entry) = self.read(slot) else {
            return;
        };
        while slot > 0 {
            let parent = (slot - 1) / 2;
            let Some(above) = self.read(parent) else {
                break;
            };
            if !precedes(entry, above) {
                break;
            }
            self.write(slot, above.0, above.1);
            slot = parent;
        }
        self.write(slot, entry.0, entry.1);
    }

    fn sift_down(&mut self, from: usize) {
        let mut slot = from;
        let Some(entry) = self.read(slot) else {
            return;
        };
        loop {
            let left = slot * 2 + 1;
            if left >= self.length {
                break;
            }
            let right = left + 1;
            let mut lowest = left;
            if right < self.length
                && let (Some(right_entry), Some(left_entry)) = (self.read(right), self.read(left))
                && precedes(right_entry, left_entry)
            {
                lowest = right;
            }
            let Some(below) = self.read(lowest) else {
                break;
            };
            if !precedes(below, entry) {
                break;
            }
            self.write(slot, below.0, below.1);
            slot = lowest;
        }
        self.write(slot, entry.0, entry.1);
    }
}

impl<const CAPACITY: usize, T: Numeric> Default for Frontier<CAPACITY, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether `left` comes out of the heap before `right`.
///
/// A strict `<` on the key, then on the item id. The tie-break makes a search's output reproducible
/// across platforms and across `f32` and `f64`, and it never asks `T` for a total order it does not
/// have.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn precedes<T: Numeric>(left: (T, u32), right: (T, u32)) -> bool {
    // `!(right < left)` rather than `left <= right`: the two differ on a NaN key, and asking for a
    // total order `T` does not have would mean a `partial_cmp().unwrap()`.
    left.0 < right.0 || (!(right.0 < left.0) && left.1 < right.1)
}

#[cfg(test)]
mod test;
