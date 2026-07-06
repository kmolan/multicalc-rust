#!/usr/bin/env bash
# Compare .text size and peak stack against ci/budgets.toml for one target.
# Usage: bash ci/check_budgets.sh <target-triple> <smoke-output-file>
set -euo pipefail

target="$1"
smoke_out="$2"
budgets="ci/budgets.toml"

# Read one key from a target's section of the simple TOML file.
read_budget() {
  awk -v sect="[$target]" -v key="$1" '
    $0 == sect { inside=1; next }
    /^\[/      { inside=0 }
    inside && $1 == key { print $3 }
  ' "$budgets"
}

tol=$(awk -F'= *' '/^tolerance_pct/ { print $2 }' "$budgets")

text_now=$(cargo size --release -p embedded-smoke --target "$target" -- -A \
           | awk '/^\.text/ { print $2 }')
if [ -z "$text_now" ]; then
  echo "FAIL: could not read .text size for $target"
  exit 1
fi

stack_now=$(grep -oE 'STACK_HWM_BYTES=[0-9]+' "$smoke_out" | head -n1 | cut -d= -f2 || true)

fail=0
check() {
  local name="$1" now="$2" budget="$3"
  if [ -z "$budget" ] || [ "$budget" = "0" ]; then
    echo "note: no $name budget set for $target (measured $now)"
    return
  fi
  local limit=$(( budget + budget * tol / 100 ))
  if [ "$now" -gt "$limit" ]; then
    echo "FAIL: $target $name = $now > budget $budget + ${tol}% = $limit"
    fail=1
  else
    echo "ok: $target $name = $now (budget $budget, limit $limit)"
  fi
}

check "text_bytes" "$text_now" "$(read_budget text_bytes)"
if [ -n "$stack_now" ]; then
  check "stack_bytes" "$stack_now" "$(read_budget stack_bytes)"
else
  echo "note: no stack line in smoke output for $target"
fi

exit "$fail"
