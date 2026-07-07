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
