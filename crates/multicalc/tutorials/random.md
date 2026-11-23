# Random

A seedable pseudo-random generator with no operating-system dependency, so the stochastic parts
of the library run on bare metal. The particle filter uses it. It is public for process noise,
sensor models and Monte Carlo checks.

- `RandomSource<T>`: the generator trait. An implementation supplies `next_u32` and the
  `get_cache` / `set_cache` pair that holds the second normal draw. The trait then supplies:
  - `next_u64`: two 32-bit words, high word first.
  - `next_unit`: uniform in `[0, 1)`, with 53 bits of precision for `f64` and 24 bits for `f32`.
  - `standard_normal`: `N(0, 1)` by the Marsaglia polar method.

  Implement `RandomSource` to use a hardware generator or a different algorithm.
- `Pcg32<T>`: the built-in generator (PCG-XSH-RR, 32-bit output). `new(seed)` uses the default
  stream. `with_stream(seed, stream)` selects a different stream, so independent filters draw
  independent sequences from the same seed. The same seed and stream give the same sequence.
  `Pcg32` is not for cryptography.

```rust
use multicalc::{Pcg32, RandomSource};

let seed = 20260722;
let mut generator = Pcg32::<f64>::new(seed);

let uniform = generator.next_unit();         // in [0, 1)
let noise = generator.standard_normal();     // mean 0, standard deviation 1

// The same seed gives the same sequence.
let mut replay = Pcg32::<f64>::new(seed);
assert_eq!(replay.next_unit(), uniform);

// A second stream from the same seed gives an independent sequence.
let stream = 1;
let mut other = Pcg32::<f64>::with_stream(seed, stream);
let independent = other.standard_normal();
```

Credits: PCG by Melissa O'Neill, with the constants from her reference implementation
(<https://www.pcg-random.org/>).

Demos that use `Pcg32`:

- [signal_filters.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/signal_filters.rs): a seeded noisy signal.
- [estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/estimation.rs): the particle filter.
- [path_planning.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/path_planning.rs): a seeded RRT that gives the same tree each run.

---

[Back to the tutorial index](README.md)
