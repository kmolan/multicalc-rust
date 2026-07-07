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

Live (spawns the viewer and streams to it):

```
cargo run -p multicalc-viz --example curve_fit_live
```
