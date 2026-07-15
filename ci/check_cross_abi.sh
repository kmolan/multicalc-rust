#!/usr/bin/env bash
# Compare on-target smoke runs across ABIs and architectures, failing if any shared
# headline scalar differs by more than the relative tolerance. This is the
# soft-float-vs-FPU / libm divergence guard: every leg runs the same math on the same
# inputs, so their SMOKE_VAL_* outputs must agree closely. A gap beyond tolerance means
# a software libm path and the hardware FPU disagree, or one leg's codegen regressed.
#
# Legs come in two classes:
#   - full-keyset legs (eabi, eabihf, riscv): passed positionally, must all expose an
#     identical SMOKE_VAL_* key set.
#   - subset legs (m0): passed with --subset, contribute only the keys they carry.
# Each key is compared across whichever legs contain it.
#
# Each run prints lines of the form:
#   SMOKE_VAL_<name>=<value>
# where <value> is an f64 in Rust `{:e}` format (shortest round-trip decimal, e.g.
# 1.3281e0).
#
# Usage:
#   bash ci/check_cross_abi.sh <eabi> <eabihf> [<riscv> ...] [--subset <m0>]
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

full=()          # full-keyset legs (eabi, eabihf, riscv): must share one key set
subset=()        # subset legs (m0): contribute only the keys they carry
while [ "$#" -gt 0 ]; do
  case "$1" in
    --subset) shift; [ -n "${1:-}" ] || die "--subset needs a file argument."; subset+=("$1") ;;
    *)        full+=("$1") ;;
  esac
  shift
done
if [ "${#full[@]}" -lt 2 ]; then
  die "expected at least the two thumbv7em outputs as full-keyset legs, got ${#full[@]}." \
      "Invoke as: bash ci/check_cross_abi.sh <eabi> <eabihf> [<riscv> ...] [--subset <m0>]"
fi

echo "== cross-ABI divergence check (soft-float vs hardware-FPU), rel tol $rel_tol =="
for f in "${full[@]}" "${subset[@]}"; do
  [ -f "$f" ] || die "smoke output '$f' not found." \
    "Capture it: cargo run -p embedded-smoke --release --target <triple> | tee $f"
done

# Pull "name value" pairs (SMOKE_VAL_ prefix stripped) from a run's output.
extract() {
  grep -oE '^SMOKE_VAL_[A-Za-z0-9_]+=[-+0-9.eE]+' "$1" | sed -e 's/^SMOKE_VAL_//' -e 's/=/ /' || true
}

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
files=(); i=0
for f in "${full[@]}" "${subset[@]}"; do
  i=$((i + 1)); extract "$f" > "$tmp/$i"; files+=("$tmp/$i")
done

status=0
report=$(
  awk -v tol="$rel_tol" -v nfull="${#full[@]}" '
    FNR == 1 { file++ }              # legs 1..nfull = full-keyset, rest = subset
    { v[file, $1] = $2; if (!seen[$1]++) keys[++nk] = $1; present[file, $1] = 1;
      if (file <= nfull) fullkey[file, $1] = 1 }
    END {
      tol += 0
      # Every full-keyset leg must expose the same key set as leg 1.
      for (i = 1; i <= nk; i++) {
        k = keys[i]
        for (f = 1; f <= nfull; f++) {
          if (((1 SUBSEP k) in fullkey) && !((f SUBSEP k) in fullkey)) { print "MISSING in full leg " f ": " k; miss = 1 }
          if (((f SUBSEP k) in fullkey) && !((1 SUBSEP k) in fullkey)) { print "MISSING in full leg 1: " k; miss = 1 }
        }
      }
      if (miss) exit 3
      n = 0
      for (i = 1; i <= nk; i++) {
        k = keys[i]
        lo = ""; hi = ""
        for (f = 1; f <= file; f++) if ((f SUBSEP k) in present) {
          x = v[f, k] + 0
          if (lo == "" || x < lo) lo = x
          if (hi == "" || x > hi) hi = x
        }
        d = hi - lo; if (d < 0) d = -d
        m = (hi < 0 ? -hi : hi); if ((lo < 0 ? -lo : lo) > m) m = (lo < 0 ? -lo : lo)
        rel = (m > 0) ? d / m : d
        printf "%-14s span=[%g, %g] rel=%g\n", k, lo, hi, rel
        if (rel > tol) bad = 1
        n++
      }
      if (n == 0) { print "no SMOKE_VAL_* keys found"; exit 4 }
      if (bad) exit 2
      exit 0
    }
  ' "${files[@]}"
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
        "Inspect these by hand: ${full[*]} ${subset[*]}" ;;
esac
