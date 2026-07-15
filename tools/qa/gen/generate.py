"""Entry point: seed the RNG and run each module's generator in a fixed order.

Writes fixtures under `<out>/v1/**`. Run inside the pinned container so the
library versions recorded in each fixture match the committed ones.
"""

import argparse

import numpy as np

from generators import (
    calculus,
    discretization,
    linalg,
    ode,
    optimization,
    quadrature,
    root_finding,
)

SEED = 20260706


def main():
    parser = argparse.ArgumentParser(description="Generate QA fixtures.")
    parser.add_argument("--out", required=True, help="output directory (holds v1/)")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    linalg.run(args.out, rng, SEED)
    discretization.run(args.out, rng, SEED)
    optimization.run(args.out, SEED)
    quadrature.run(args.out, SEED)
    ode.run(args.out, SEED)
    # These take only SEED (never the shared rng) and register last, so they
    # cannot perturb the linalg/discretization streams above.
    calculus.run(args.out, SEED)
    root_finding.run(args.out, SEED)


if __name__ == "__main__":
    main()
