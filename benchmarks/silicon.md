# Silicon benchmarks

Warm time of each check of the HILShare apps ([`tools/hilshare`](../tools/hilshare)) on three
microcontrollers. Each run checks each answer against a golden or an identity. **The numbers are
board-dependent.** The boards are not in CI, so CI does not guard this table.

Regenerate from the logs of Board runs:

```sh
gh run download <run-id> --pattern 'hilshare-multicalc-hilshare-*' --dir <run directory>
cargo run -p multicalc-qa --bin gen_silicon_table -- --commit <sha> --date <YYYY-MM-DD> <run directory>...
```

<!-- BEGIN generated: silicon -->
<!-- END generated -->
