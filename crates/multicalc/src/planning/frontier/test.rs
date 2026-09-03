//! White-box tests for the indexed heap: pop order, the decrease-key, and that `slot_of` stays in
//! step with every move.

use super::{Frontier, FrontierFull};

/// Drains a heap into a fixed array, returning what came out and how much.
fn drain<const CAPACITY: usize, const LIMIT: usize>(
    frontier: &mut Frontier<CAPACITY>,
) -> ([(usize, f64); LIMIT], usize) {
    let mut popped = [(0usize, 0.0f64); LIMIT];
    let mut count = 0;
    while let Some(entry) = frontier.pop_minimum() {
        if let Some(slot) = popped.get_mut(count) {
            *slot = entry;
            count += 1;
        }
    }
    (popped, count)
}

#[test]
fn pop_order_is_ascending_by_key() {
    let mut frontier: Frontier<16> = Frontier::new();
    for (item, key) in [(3, 5.0), (1, 2.0), (7, 9.0), (0, 1.0), (4, 7.0)] {
        frontier.push_or_lower(item, key).unwrap();
    }
    assert_eq!(frontier.len(), 5);

    let (popped, count) = drain::<16, 8>(&mut frontier);
    assert_eq!(count, 5);
    assert_eq!(
        popped.get(..5),
        Some(&[(0, 1.0), (1, 2.0), (3, 5.0), (4, 7.0), (7, 9.0)][..])
    );
    assert_eq!(frontier.len(), 0);
}

#[test]
fn lowering_a_key_moves_the_item_forward() {
    let mut frontier: Frontier<16> = Frontier::new();
    frontier.push_or_lower(1, 10.0).unwrap();
    frontier.push_or_lower(2, 2.0).unwrap();
    frontier.push_or_lower(3, 4.0).unwrap();

    frontier.push_or_lower(1, 1.0).unwrap();
    assert_eq!(frontier.len(), 3);
    assert_eq!(frontier.pop_minimum(), Some((1, 1.0)));
}

#[test]
fn raising_a_key_is_ignored() {
    let mut frontier: Frontier<16> = Frontier::new();
    frontier.push_or_lower(1, 2.0).unwrap();
    frontier.push_or_lower(1, 9.0).unwrap();

    assert_eq!(frontier.len(), 1);
    assert_eq!(frontier.pop_minimum(), Some((1, 2.0)));
}

#[test]
fn each_item_occupies_at_most_one_slot() {
    let mut frontier: Frontier<64> = Frontier::new();
    for round in 0..2 {
        for item in 0..64 {
            frontier
                .push_or_lower(item, (64 - item) as f64 - round as f64)
                .unwrap();
        }
    }
    assert_eq!(frontier.len(), 64);

    let mut seen = [false; 64];
    while let Some((item, _)) = frontier.pop_minimum() {
        let slot = seen.get_mut(item).unwrap();
        assert!(!*slot, "item {item} came out twice");
        *slot = true;
    }
    assert!(seen.into_iter().all(|hit| hit));
}

#[test]
fn slot_of_stays_consistent_through_interleaved_pushes_and_pops() {
    const CAPACITY: usize = 32;
    let mut frontier: Frontier<CAPACITY> = Frontier::new();
    // The model: the lowest key each queued item currently carries.
    let mut model = [None::<f64>; CAPACITY];
    let mut state = 0x2026_0830_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };

    for _ in 0..1000 {
        let draw = next();
        if draw % 3 == 0 {
            // The model's own minimum, by the same strict-`<` then item-id rule the heap uses.
            let mut expected: Option<(usize, f64)> = None;
            for (item, held) in model.iter().enumerate() {
                let Some(key) = *held else { continue };
                let better = match expected {
                    None => true,
                    Some((_, lowest)) => key < lowest,
                };
                if better {
                    expected = Some((item, key));
                }
            }
            match (frontier.pop_minimum(), expected) {
                (Some((item, key)), Some((expected_item, expected_key))) => {
                    assert_eq!((item, key), (expected_item, expected_key));
                    *model.get_mut(item).unwrap() = None;
                }
                (None, None) => {}
                (popped, expected) => panic!("popped {popped:?}, expected {expected:?}"),
            }
        } else {
            let item = draw % CAPACITY;
            let key = (next() % 1000) as f64;
            frontier.push_or_lower(item, key).unwrap();
            let held = model.get_mut(item).unwrap();
            *held = Some(match *held {
                Some(current) => current.min(key),
                None => key,
            });
        }
        assert_eq!(
            frontier.len(),
            model.iter().filter(|key| key.is_some()).count()
        );
        for (item, key) in model.iter().enumerate() {
            assert_eq!(frontier.contains(item), key.is_some(), "item {item}");
        }
    }
}

#[test]
fn push_beyond_capacity_reports_full() {
    let mut frontier: Frontier<4> = Frontier::new();
    for item in 0..4 {
        frontier.push_or_lower(item, item as f64).unwrap();
    }
    assert_eq!(frontier.push_or_lower(4, 0.0), Err(FrontierFull));
    assert_eq!(frontier.push_or_lower(usize::MAX, 0.0), Err(FrontierFull));
}

#[test]
fn pop_on_empty_is_none() {
    let mut frontier: Frontier<4> = Frontier::new();
    assert_eq!(frontier.pop_minimum(), None);

    frontier.push_or_lower(2, 1.0).unwrap();
    assert_eq!(frontier.pop_minimum(), Some((2, 1.0)));
    assert_eq!(frontier.pop_minimum(), None);
}

#[test]
fn equal_keys_break_ties_by_item_id() {
    let mut frontier: Frontier<16> = Frontier::new();
    for item in [5, 2, 9, 0, 7] {
        frontier.push_or_lower(item, 3.0).unwrap();
    }

    let (popped, count) = drain::<16, 8>(&mut frontier);
    assert_eq!(count, 5);
    let items: [usize; 5] =
        core::array::from_fn(|index| popped.get(index).map(|entry| entry.0).unwrap());
    assert_eq!(items, [0, 2, 5, 7, 9]);
}

#[test]
fn clear_and_clear_prefix_forget_what_they_should() {
    let mut frontier: Frontier<16> = Frontier::new();
    for item in 0..16 {
        frontier.push_or_lower(item, item as f64).unwrap();
    }

    frontier.clear();
    assert_eq!(frontier.len(), 0);
    for item in 0..16 {
        assert!(!frontier.contains(item));
    }

    for item in 0..8 {
        frontier.push_or_lower(item, item as f64).unwrap();
    }
    frontier.clear_prefix(8);
    assert_eq!(frontier.len(), 0);
    for item in 0..8 {
        assert!(!frontier.contains(item));
    }
}
