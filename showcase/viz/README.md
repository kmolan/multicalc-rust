# multicalc-viz

A thin, std-only [Rerun](https://rerun.io) visualization adapter for `multicalc`. It maps core
types to Rerun archetypes behind a small `VizSink` trait, with a Rerun backend (`live()` or
`record(path)`) and a CSV backend for a matplotlib fallback.

This is a satellite crate: it is never a dependency of the core library, is excluded from
bare-metal builds and the default `cargo test`, and its dependency tree is excluded from the
workspace supply-chain audit.

## Versions

Rerun SDK `=0.33.1` ⇄ viewer `0.33.1`. The SDK is exact-pinned; the viewer must match.

## Viewer install (for the live example)

`live()` spawns the external Rerun viewer found on PATH, so install it version-matched to the SDK:

```
cargo install rerun-cli --locked --version 0.33.1
# or: pip install rerun-sdk==0.33.1
# or: cargo binstall rerun-cli --version 0.33.1
```

## Running the examples

Recorded (no viewer needed; writes a `.rrd` and a `.csv` to the temp dir):

```
cargo run -p multicalc-viz --example curve_fit_record
```

Open the printed `.rrd` in the viewer, or render the CSV fallback:

```
python showcase/viz/plot.py <printed-csv-path> --x t
```

Live (on a normal host this spawns a local viewer; under WSL see the section below):

```
cargo run -p multicalc-viz --example curve_fit_live
```

## Showcases

Four live demos, one per core module, each an attention-grabbing animated scene that markets the
library's raw speed and accuracy. **Every number on screen is measured live** with
`std::time::Instant` inside the demo — nothing is hardcoded. Run each with `--release` (mandatory
for the timing readouts) and the viewer already up.

Each demo advances its simulation on logical time (a fixed 1 ms per tick / one step per frame),
so the numbers are deterministic and reproducible. An OS scheduling spike can make a tick display late or jitter but never changes
what the demo computes.

The figures below are representative of a modern desktop core (`x86_64`, `--release`).

- **`ik_servo`** (optimization) — a 3-link arm runs a complete Levenberg-Marquardt IK solve, with
  exact autodiff Jacobians, every single millisecond. **Median solve ≈ 6 µs — under 1 % of the
  1 ms budget — with zero missed ticks over 120,000 solves.**

  ```
  cargo run --release -p multicalc-viz --example ik_servo
  ```

  ![ik_servo — a 3-link arm running a full LM IK solve every millisecond](examples/support/ik_servo_showcase.gif)

- **`newton_fractal`** (root finding) — every pixel is a full Newton-system solve with an exact
  autodiff Jacobian, and the cubic's basins swirl as its roots orbit. **≈ 4 million Newton
  solves/sec on one core** (a 256×256 grid re-solved at ~60 fps), each converged root accurate to
  **≈ 5e-15**.

  ```
  cargo run --release -p multicalc-viz --example newton_fractal
  ```

  ![newton_fractal — cubic basins swirling, every pixel a full Newton solve](examples/support/newton_fractal_showcase.gif)

- **`fourier_ferris`** (integration) — Gauss-Legendre quadrature computes the Fourier coefficients
  of Ferris's outline; a chain of epicycles then draws the crab. **≈ 600,000 quadrature node
  evaluations in ≈ 8 ms** at startup, with every coefficient matching the exact closed form to
  **≈ 1e-15**.

  ```
  cargo run --release -p multicalc-viz --example fourier_ferris
  ```

  ![fourier_ferris — an epicycle chain drawing Ferris from Fourier coefficients](examples/support/fourier_ferris_showcase.gif)

- **`gradient_marbles`** (autodiff) — 2,000 marbles across a 3D Himmelblau landscape, each steered
  by an exact autodiff gradient every millisecond. **2,000 exact gradients in under 3 µs per tick
  (~750,000 gradients/ms), and the autodiff-vs-analytic error is pinned at exactly 0.0** on screen.

  ```
  cargo run --release -p multicalc-viz --example gradient_marbles
  ```

  ![gradient_marbles — 2,000 marbles steered by exact autodiff gradients down a 3D landscape](examples/support/gradient_marbles_showcase.gif)

## WSL usage (viewer on Windows)

The live viewer is a GPU application; under WSL its virtualized GPU often cannot start it. Run
the viewer on Windows instead (real GPU) and stream to it from WSL over gRPC.

1. Enable mirrored networking so WSL and Windows share `localhost`. In `C:\Users\<you>\.wslconfig`:

   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

   Then from Windows PowerShell run `wsl --shutdown`, reopen WSL, and confirm:

   ```
   wslinfo --networking-mode      # -> mirrored
   ```

2. Install the viewer on Windows if needed, version-matched to the SDK (0.33.1):

   ```
   pip install rerun-sdk==0.33.1      # provides the `rerun` command
   # or download the prebuilt rerun.exe for 0.33.1
   ```

3. Start the viewer on Windows (it listens on port 9876):

   ```
   rerun
   ```

4. From WSL, run the live example. Under WSL it auto-detects the environment and streams to the
   Windows viewer over the shared localhost instead of spawning a local one:

   ```
   cargo run -p multicalc-viz --example curve_fit_live
   ```

   The Windows viewer from step 3 MUST already be running — under WSL the example connects to it
   and does not spawn one.

On NAT networking (the WSL default) instead of mirrored, set `RERUN_VIZ_URL` to the Windows host,
launch the viewer bound to `0.0.0.0`, and allow inbound TCP 9876 in Windows Firewall:

```
export RERUN_VIZ_URL="rerun+http://$(ip route show default | awk '{print $3}'):9876/proxy"
cargo run -p multicalc-viz --example curve_fit_live
```
