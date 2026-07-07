#!/usr/bin/env bash
# Compare a release smoke binary's .text size and peak stack use against the committed
# budgets in ci/budgets.toml, for one (binary, target) pair. Stack is gated per scalar
# type: each `stack_bytes_<scalar>` key in the section is matched to a
# STACK_HWM_BYTES_<SCALAR>=<n> line in the run output (the bare STACK_HWM_BYTES=<n> line
# counts as f64). On any problem it prints ">>> ERROR" / ">>> FIX" and exits non-zero.
#
# Usage:
#   bash ci/check_budgets.sh <binary> <target-triple> <smoke-output-file>
#
#   <binary>            cargo package name of the smoke crate (also the -p arg to
#                       `cargo size`), e.g. embedded-smoke. Must match the "<binary>."
#                       prefix of a [<binary>.<target>] section in ci/budgets.toml.
#   <target-triple>     thumbv7em-none-eabi | thumbv7em-none-eabihf | thumbv6m-none-eabi
#   <smoke-output-file> captured stdout of the run; contains a STACK_HWM_BYTES[...]=<n>
#                       line. Produce it with:
#                         cargo run -p <binary> --release --target <triple> | tee <file>
#
# Run from the repository root: it reads ci/budgets.toml by that relative path.
set -euo pipefail

budgets="ci/budgets.toml"
valid_targets="thumbv7em-none-eabi thumbv7em-none-eabihf thumbv6m-none-eabi"

die() {
  echo ">>> ERROR: $1" >&2
  shift
  for hint in "$@"; do echo ">>> FIX:   $hint" >&2; done
  exit 1
}

# ---- Argument checks -------------------------------------------------------
if [ "$#" -ne 3 ]; then
  die "expected exactly 3 arguments, got $#." \
      "Invoke as: bash ci/check_budgets.sh <binary> <target-triple> <smoke-output-file>" \
      "Valid targets: $valid_targets"
fi

binary="$1"
target="$2"
smoke_out="$3"
section="[$binary.$target]"
echo "== budget check for $binary on $target =="

# ---- Preconditions ---------------------------------------------------------
[ -f "$budgets" ] || die "budget file '$budgets' not found (looked under $(pwd))." \
  "Run this script from the repository root." \
  "Confirm ci/budgets.toml exists and is committed."

[ -f "$smoke_out" ] || die "smoke output file '$smoke_out' not found." \
  "Capture the run first: cargo run -p $binary --release --target $target | tee $smoke_out"

if ! grep -Fxq "$section" "$budgets"; then
  echo ">>> WARNING: no '$section' section in $budgets; nothing will be gated." >&2
  echo ">>> FIX:     add a $section section to $budgets, or fix the binary/target spelling." >&2
  echo ">>> FIX:     valid targets: $valid_targets" >&2
fi

# Read one key's value from this binary+target section.
read_budget() {
  awk -v sect="$section" -v key="$1" '
    $0 == sect { inside=1; next }
    /^\[/      { inside=0 }
    inside && $1 == key { print $3 }
  ' "$budgets"
}

# List the stack_bytes_<scalar> keys present in this section.
scalar_keys() {
  awk -v sect="$section" '
    $0 == sect { inside=1; next }
    /^\[/      { inside=0 }
    inside && $1 ~ /^stack_bytes_/ { print $1 }
  ' "$budgets"
}

# ---- Tolerance -------------------------------------------------------------
tol=$(awk -F'= *' '/^tolerance_pct/ { print $2 }' "$budgets")
printf '%s' "$tol" | grep -qE '^[0-9]+$' || die \
  "could not read a numeric 'tolerance_pct' from $budgets (got '$tol')." \
  "Add a top-level line such as: tolerance_pct = 10"

# ---- Compare one measurement against its budget ----------------------------
fail=0
check() {
  local name="$1" now="$2" budget="$3"
  printf '%s' "$now" | grep -qE '^[0-9]+$' || die \
    "measured $name for $binary/$target is not a number (got '$now')." \
    "This points at a parsing bug above; do not trust the run."
  if [ -z "$budget" ] || [ "$budget" = "0" ]; then
    echo "note: no $name budget for $binary/$target (measured $now) — reporting only, not gating."
    return
  fi
  printf '%s' "$budget" | grep -qE '^[0-9]+$' || die \
    "$name budget in $section of $budgets is not a number (got '$budget')." \
    "Fix the value to a plain integer, e.g. $name = $now"
  local limit=$(( budget + budget * tol / 100 ))
  if [ "$now" -gt "$limit" ]; then
    echo ">>> FAIL: $binary/$target $name = $now exceeds budget $budget + ${tol}% = $limit (over by $(( now - limit )) bytes)." >&2
    echo ">>> FIX:  if this growth is expected, raise $name in $section of $budgets in this same PR (see ci/README.md)." >&2
    echo ">>> FIX:  otherwise investigate the change that grew $name before merging." >&2
    fail=1
  else
    echo "ok: $binary/$target $name = $now (budget $budget, limit $limit)"
  fi
}

# ---- Measure .text ---------------------------------------------------------
if ! size_out=$(cargo size --release -p "$binary" --target "$target" -- -A 2>&1); then
  echo "$size_out" >&2
  die "'cargo size' failed for $binary/$target (its output is above)." \
      "Install the tool:       cargo install cargo-binutils" \
      "Add the LLVM tools:     rustup component add llvm-tools-preview" \
      "Add the target:         rustup target add $target" \
      "Build it at least once: cargo build -p $binary --release --target $target"
fi
text_now=$(printf '%s\n' "$size_out" | awk '/^\.text/ { print $2 }')
printf '%s' "$text_now" | grep -qE '^[0-9]+$' || { echo "$size_out" >&2; die \
  "could not parse a numeric .text size for $binary/$target (got '$text_now')." \
  "Look for a '.text' row in the 'cargo size ... -- -A' output above."; }
check "text_bytes" "$text_now" "$(read_budget text_bytes)"

# ---- Per-scalar stack ------------------------------------------------------
keys=$(scalar_keys)
if [ -z "$keys" ]; then
  echo "note: no stack_bytes_<scalar> keys in $section — stack not gated for $binary/$target."
else
  for key in $keys; do
    scalar=${key#stack_bytes_}
    upper=$(printf '%s' "$scalar" | tr '[:lower:]' '[:upper:]')
    meas=$(grep -oE "STACK_HWM_BYTES_${upper}=[0-9]+" "$smoke_out" | head -n1 | cut -d= -f2 || true)
    if [ -z "$meas" ] && [ "$scalar" = "f64" ]; then
      # Back-compat: the bare STACK_HWM_BYTES= line is the f64 measurement.
      meas=$(grep -oE 'STACK_HWM_BYTES=[0-9]+' "$smoke_out" | head -n1 | cut -d= -f2 || true)
    fi
    budget=$(read_budget "$key")
    if [ -n "$meas" ]; then
      check "$key" "$meas" "$budget"
    else
      hint=""
      [ "$scalar" = "f64" ] && hint=" (or the bare STACK_HWM_BYTES= line)"
      echo ">>> WARNING: no STACK_HWM_BYTES_${upper}=${hint} measurement in $smoke_out for $key." >&2
      if [ -n "$budget" ] && [ "$budget" != "0" ]; then
        die "$key budget ($budget) is set for $binary/$target but no matching stack measurement was found." \
            "Make the smoke run print STACK_HWM_BYTES_${upper}= (or STACK_HWM_BYTES= for f64), or set $key = 0 for report-only."
      fi
    fi
  done
fi

if [ "$fail" -ne 0 ]; then
  echo ">>> Budget check FAILED for $binary/$target — see the FAIL lines above." >&2
else
  echo "== budget check passed for $binary/$target =="
fi
exit "$fail"
