#!/usr/bin/env bash
# Compare two on-target smoke runs — soft-float (eabi) vs hardware-FPU (eabihf) — and
# fail if any shared headline scalar differs by more than the relative tolerance. This
# is the soft-float-vs-FPU / libm divergence guard: both ABIs run the same math on the
# same inputs, so their SMOKE_VAL_* outputs must agree closely. A gap beyond tolerance
# means the libm software path and the hardware FPU disagree (or one ABI regressed).
#
# Each run prints lines of the form:
#   SMOKE_VAL_<name>=<value>
# where <value> is an f64 in Rust `{:e}` format (shortest round-trip decimal, e.g.
# 1.3281e0). Only keys present in both files are compared, and the key sets must match.
#
# Usage:
#   bash ci/check_cross_abi.sh <eabi-output-file> <eabihf-output-file>
#
# Run from the repository root.
set -euo pipefail

# Relative-agreement bound between the two ABIs. Soft-float libm and the hardware FPU
# round intermediate steps differently, so exact equality is not expected; 1e-9 is far
# tighter than the fixture goldens yet loose enough for benign last-bit divergence.
# Widening this is a reviewed change (see ci/README.md), never a silent one.
rel_tol="1e-9"

die() {
  echo ">>> ERROR: $1" >&2
  shift
  for hint in "$@"; do echo ">>> FIX:   $hint" >&2; done
  exit 1
}

if [ "$#" -ne 2 ]; then
  die "expected exactly 2 arguments, got $#." \
      "Invoke as: bash ci/check_cross_abi.sh <eabi-output-file> <eabihf-output-file>"
fi

eabi_out="$1"
eabihf_out="$2"
echo "== cross-ABI divergence check (soft-float vs hardware-FPU), rel tol $rel_tol =="

[ -f "$eabi_out" ] || die "soft-float output '$eabi_out' not found." \
  "Capture it: cargo run -p embedded-smoke --release --target thumbv7em-none-eabi | tee $eabi_out"
[ -f "$eabihf_out" ] || die "hardware-FPU output '$eabihf_out' not found." \
  "Capture it: cargo run -p embedded-smoke --release --target thumbv7em-none-eabihf | tee $eabihf_out"

# Pull "name value" pairs (SMOKE_VAL_ prefix stripped) from a run's output.
extract() {
  grep -oE '^SMOKE_VAL_[A-Za-z0-9_]+=[-+0-9.eE]+' "$1" | sed -e 's/^SMOKE_VAL_//' -e 's/=/ /' || true
}

status=0
report=$(
  awk -v tol="$rel_tol" '
    FNR == NR { a[$1] = $2; next }   # first file: soft-float (eabi)
    { b[$1] = $2 }                   # second file: hardware-FPU (eabihf)
    END {
      tol += 0
      for (k in a) if (!(k in b)) { print "MISSING in eabihf: " k; miss = 1 }
      for (k in b) if (!(k in a)) { print "MISSING in eabi:   " k; miss = 1 }
      if (miss) { exit 3 }
      n = 0
      for (k in a) {
        x = a[k] + 0; y = b[k] + 0
        d = x - y; if (d < 0) d = -d
        ax = x; if (ax < 0) ax = -ax
        ay = y; if (ay < 0) ay = -ay
        m = (ax > ay) ? ax : ay
        rel = (m > 0) ? d / m : d
        printf "%-12s eabi=%s eabihf=%s rel=%g\n", k, a[k], b[k], rel
        if (rel > tol) bad = 1
        n++
      }
      if (n == 0) { print "no SMOKE_VAL_* keys found"; exit 4 }
      if (bad) exit 2
      exit 0
    }
  ' <(extract "$eabi_out") <(extract "$eabihf_out")
) || status=$?

[ -n "$report" ] && echo "$report"

case "$status" in
  0) echo "== cross-ABI check passed: all shared scalars agree within rel $rel_tol ==" ;;
  2) die "soft-float (eabi) and hardware-FPU (eabihf) disagree beyond rel $rel_tol (see rel= above)." \
        "This is a soft-float-vs-FPU / libm divergence: the same math gives different answers per ABI." \
        "Investigate the libm software path against the hardware FPU for the diverging scalar before merging." \
        "If the divergence is understood and acceptable, widen rel_tol in ci/check_cross_abi.sh (documented in ci/README.md), reviewed in this PR." ;;
  3) die "the two runs do not expose the same SMOKE_VAL_* keys (see MISSING above)." \
        "Both ABIs must print the same headline scalars; check the smoke binary's SMOKE_VAL_ output." ;;
  4) die "no SMOKE_VAL_* lines found in the runs." \
        "The smoke binary must print SMOKE_VAL_<name>=<f64> lines; confirm both runs completed." ;;
  *) die "cross-ABI comparison failed unexpectedly (awk status $status)." \
        "Inspect $eabi_out and $eabihf_out by hand." ;;
esac
