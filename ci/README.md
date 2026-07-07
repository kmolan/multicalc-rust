# CI budget gate

`ci/budgets.toml` holds `.text` and per-scalar stack budgets, keyed by `<binary>.<target>`.
`ci/check_budgets.sh <binary> <target> <smoke-out>` measures the release smoke binary and fails
if any gated row exceeds `budget × (1 + tolerance_pct/100)`.

## Budget-raise protocol

A red gate means a change grew a binary's `.text` or stack past its tolerance. Resolve it one of
two ways, **in the same PR**:

1. **Regression** — investigate and shrink the change back under budget. Prefer this.
2. **Legitimate growth** — raise the specific `text_bytes` / `stack_bytes_<scalar>` value in
   `ci/budgets.toml`, with a one-line justification in the PR description. The reviewer approves the
   new baseline. This is the line between a **reviewed re-baseline** and **silent bloat**: the number
   moves in a diff a human signed off on.

Do **not** raise `tolerance_pct` to get under a gate — that loosens every row at once and hides real
regressions. Budgets are per-binary and per-scalar precisely so one domain's growth never spends
another's headroom; keep them tight and per-row.

## Adding a binary or a scalar row

- New smoke crate (v0.9+): add a `[<binary>.<target>]` section per target and add the package name
  to the loop in `.github/workflows/matrix.yml`.
- New scalar row: add `stack_bytes_<scalar>` to the section and make the smoke binary print
  `STACK_HWM_BYTES_<SCALAR>=<n>`. An absent or `0` row is report-only.
