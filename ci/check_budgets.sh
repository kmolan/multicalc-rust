#!/usr/bin/env bash
# Compare the release smoke binary's .text size and peak stack use against the
# committed budgets in ci/budgets.toml for one bare-metal target. On any problem
# it prints a clear ">>> ERROR" / ">>> FIX" message and exits non-zero, so a
# failure is never silent.
#
# Usage:
#   bash ci/check_budgets.sh <target-triple> <smoke-output-file>
#
# Arguments (both required, in this order):
#   <target-triple>       The bare-metal target to check. One of:
#                           thumbv7em-none-eabi     (Cortex-M4, soft-float)
#                           thumbv7em-none-eabihf   (Cortex-M4, hardware FPU)
#                           thumbv6m-none-eabi      (Cortex-M0 build)
#                         Must match a [section] header in ci/budgets.toml.
#   <smoke-output-file>   File holding the captured stdout of the smoke run.
#                         It must contain a line "STACK_HWM_BYTES=<n>". Produce it with:
#                           cargo run -p embedded-smoke --release --target <triple> | tee <file>
#
# Example:
#   cargo run -p embedded-smoke --release --target thumbv6m-none-eabi | tee smoke-m0.txt
#   bash ci/check_budgets.sh thumbv6m-none-eabi smoke-m0.txt
#
# Run from the repository root: it reads ci/budgets.toml by that relative path.
set -euo pipefail

budgets="ci/budgets.toml"
# Kept in sync with the [sections] in ci/budgets.toml and .cargo/config.toml.
valid_targets="thumbv7em-none-eabi thumbv7em-none-eabihf thumbv6m-none-eabi"

# Print an error and one FIX line per corrective hint, then stop.
die() {
  echo ">>> ERROR: $1" >&2
  shift
  for hint in "$@"; do
    echo ">>> FIX:   $hint" >&2
  done
  exit 1
}

# ---- Argument checks -------------------------------------------------------
if [ "$#" -ne 2 ]; then
  die "expected exactly 2 arguments, got $#." \
      "Invoke as: bash ci/check_budgets.sh <target-triple> <smoke-output-file>" \
      "Valid targets: $valid_targets"
fi

target="$1"
smoke_out="$2"
echo "== budget check for $target =="

# ---- Preconditions ---------------------------------------------------------
if [ ! -f "$budgets" ]; then
  die "budget file '$budgets' not found (looked under $(pwd))." \
      "Run this script from the repository root." \
      "Confirm ci/budgets.toml exists and is committed."
fi

if [ ! -f "$smoke_out" ]; then
  die "smoke output file '$smoke_out' not found." \
      "Capture the run first: cargo run -p embedded-smoke --release --target $target | tee $smoke_out"
fi

# A missing [section] means the target is unconfigured (usually a typo). Warn
# loudly; both checks below will then fall through to report-only.
if ! grep -q "^\[$target\]$" "$budgets"; then
  echo ">>> WARNING: '$target' has no [section] in $budgets; nothing will be gated." >&2
  echo ">>> FIX:     add a [$target] section to $budgets, or fix the target spelling." >&2
  echo ">>> FIX:     valid targets: $valid_targets" >&2
fi

# Read one key from a target's section of the simple TOML file.
read_budget() {
  awk -v sect="[$target]" -v key="$1" '
    $0 == sect { inside=1; next }
    /^\[/      { inside=0 }
    inside && $1 == key { print $3 }
  ' "$budgets"
}

# ---- Tolerance -------------------------------------------------------------
tol=$(awk -F'= *' '/^tolerance_pct/ { print $2 }' "$budgets")
if ! printf '%s' "$tol" | grep -qE '^[0-9]+$'; then
  die "could not read a numeric 'tolerance_pct' from $budgets (got '$tol')." \
      "Add a top-level line such as: tolerance_pct = 10"
fi

# ---- Measure .text ---------------------------------------------------------
# Capture cargo-size separately so a tool failure gives a clear message instead
# of a bare non-zero exit from a pipeline.
if ! size_out=$(cargo size --release -p embedded-smoke --target "$target" -- -A 2>&1); then
  echo "$size_out" >&2
  die "'cargo size' failed for $target (its output is above)." \
      "Install the tool:       cargo install cargo-binutils" \
      "Add the LLVM tools:     rustup component add llvm-tools-preview" \
      "Add the target:         rustup target add $target" \
      "Build it at least once: cargo build -p embedded-smoke --release --target $target"
fi

text_now=$(printf '%s\n' "$size_out" | awk '/^\.text/ { print $2 }')
if ! printf '%s' "$text_now" | grep -qE '^[0-9]+$'; then
  echo "$size_out" >&2
  die "could not parse a numeric .text size for $target (got '$text_now')." \
      "Look for a '.text' row in the 'cargo size ... -- -A' output above."
fi

# ---- Read the peak stack from the smoke output -----------------------------
stack_now=$(grep -oE 'STACK_HWM_BYTES=[0-9]+' "$smoke_out" | head -n1 | cut -d= -f2 || true)

# ---- Compare one measurement against its budget ----------------------------
fail=0
check() {
  local name="$1" now="$2" budget="$3"
  if ! printf '%s' "$now" | grep -qE '^[0-9]+$'; then
    die "measured $name for $target is not a number (got '$now')." \
        "This points at a parsing bug above; do not trust the run."
  fi
  # An empty or zero budget means "report only, do not gate" (see ci/budgets.toml).
  if [ -z "$budget" ] || [ "$budget" = "0" ]; then
    echo "note: no $name budget set for $target (measured $now) — reporting only, not gating."
    return
  fi
  if ! printf '%s' "$budget" | grep -qE '^[0-9]+$'; then
    die "$name budget in [$target] of $budgets is not a number (got '$budget')." \
        "Fix the value to a plain integer, e.g. $name = $now"
  fi
  local limit=$(( budget + budget * tol / 100 ))
  if [ "$now" -gt "$limit" ]; then
    echo ">>> FAIL: $target $name = $now exceeds budget $budget + ${tol}% = $limit (over by $(( now - limit )) bytes)." >&2
    echo ">>> FIX:  if this growth is expected, raise $name in [$target] of $budgets in this same PR." >&2
    echo ">>> FIX:  otherwise investigate the change that grew $name for $target before merging." >&2
    fail=1
  else
    echo "ok: $target $name = $now (budget $budget, limit $limit)"
  fi
}

check "text_bytes" "$text_now" "$(read_budget text_bytes)"

if [ -n "$stack_now" ]; then
  check "stack_bytes" "$stack_now" "$(read_budget stack_bytes)"
else
  # No stack line: the run may have panicked, or this ABI deliberately drops it.
  echo ">>> WARNING: no 'STACK_HWM_BYTES=' line in $smoke_out for $target." >&2
  echo ">>> FIX:     if the smoke run panicked, fix the failing check (the run must exit non-zero)." >&2
  echo ">>> FIX:     if this ABI intentionally drops the stack line, keep its stack_bytes budget at 0." >&2
  echo ">>> FIX:     otherwise confirm main.rs still prints STACK_HWM_BYTES and the run finished." >&2
  # A stack budget set with no measurement means the gate cannot run — fail hard
  # rather than pass silently.
  sb="$(read_budget stack_bytes)"
  if [ -n "$sb" ] && [ "$sb" != "0" ]; then
    die "stack_bytes budget ($sb) is set for $target but no stack measurement was found." \
        "Make the smoke run print STACK_HWM_BYTES, or set stack_bytes = 0 for report-only."
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo ">>> Budget check FAILED for $target — see the FAIL lines above." >&2
else
  echo "== budget check passed for $target =="
fi
exit "$fail"
