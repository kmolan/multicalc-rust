#!/usr/bin/env bash
# Build and run every headless basic demo, failing on the first nonzero exit. Basics
# take no features and self-check with asserts; showcases are skipped (they need the
# Rerun viewer and loop forever). Run from the repository root.
set -euo pipefail

basics=(
  approximation
  autodiff_scalars
  curve_fit
  differentiation
  discretization
  gaussian_integration
  iterative_integration
  jacobian_hessian
  lie_groups
  linear_algebra
  ode
  optimization_solvers
  root_finding
  svd
  vector_field
)

for name in "${basics[@]}"; do
  echo "== $name =="
  cargo run -p multicalc-demos --example "$name" --no-default-features
done

echo "all ${#basics[@]} basics passed"
