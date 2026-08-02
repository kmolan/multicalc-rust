# Random

A seedable generator that works without an operating system, so the stochastic parts of the library
run on bare metal. The particle filter uses it internally; it is public because process noise,
sensor models, and Monte-Carlo checks need the same thing.

- `RandomSource`: the trait a generator implements. `next_u32` is the only required method; the
  trait supplies `next_u64`, `next_unit_f64` (uniform in `[0, 1)` with 53 bits of precision), and
  `standard_normal` (mean 0, standard deviation 1) on top of it. Implement it to plug in a hardware
  generator or your own algorithm.
- `Pcg32`: the built-in generator (PCG-XSH-RR, 32-bit output). `new(seed)` uses the default stream;
  `with_stream(seed, stream)` picks another, so independent filters draw independent sequences from
  the same seed. Deterministic — the same seed reproduces the same run exactly, which is what makes
  a seeded simulation repeatable. Not for cryptography.

```rust
use multicalc::{Pcg32, RandomSource};

let seed = 20260722;
let mut generator = Pcg32::new(seed);

let uniform = generator.next_unit_f64();     // in [0, 1)
let noise = generator.standard_normal();     // mean 0, standard deviation 1

// The same seed replays the same sequence.
let mut replay = Pcg32::new(seed);
assert_eq!(replay.next_unit_f64(), uniform);

// A second stream from the same seed draws an independent sequence.
let stream = 1;
let mut other = Pcg32::with_stream(seed, stream);
let independent = other.standard_normal();
```

Demo: `2d_localization_obstacle_avoidance` seeds every noise source from one number, so the whole
run repeats exactly.


---

[Back to the tutorial index](README.md)
