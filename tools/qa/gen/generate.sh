#!/usr/bin/env bash
# Regenerates the QA golden fixtures in a local virtualenv.
#
# Usage: tools/qa/gen/generate.sh [output-dir]
#
# Writes to tools/qa/fixtures unless another directory is given. Set PYTHON= to
# pick a different interpreter.
set -euo pipefail

gen_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$gen_dir/.venv"
out="${1:-$gen_dir/../fixtures}"
python_bin="${PYTHON:-python3.12}"

if ! command -v "$python_bin" >/dev/null; then
  echo "need $python_bin on PATH (set PYTHON=... to pick another interpreter)" >&2
  exit 1
fi

if [ ! -x "$venv/bin/python" ]; then
  "$python_bin" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --quiet --require-hashes -r "$gen_dir/requirements.lock"

# One thread and a fixed hash seed. Several threads let a BLAS call add its
# partial sums in a different order each run, which moves the last bits of a
# golden.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
# The instant stamped into every fixture's metadata date, so a rerun produces
# the same bytes rather than a fresh timestamp.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1783296000}"

# The generators import each other by bare name, so run from this directory.
cd "$gen_dir"
exec "$venv/bin/python" generate.py --out "$out"
