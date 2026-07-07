#!/usr/bin/env python3
"""Plot a multicalc-viz CSV. Usage: python plot.py <file.csv> [--x COLUMN] [--save out.png]

The first column is the x axis unless --x names another column. Every other numeric column is
drawn as a line series. Depends only on the standard library plus matplotlib.
"""
import argparse
import csv
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--x", default=None, help="column to use as the x axis")
    ap.add_argument("--save", default=None, help="write a PNG instead of showing a window")
    args = ap.parse_args()

    with open(args.csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        print("empty CSV", file=sys.stderr)
        return 1
    header, data = rows[0], rows[1:]

    import matplotlib
    if args.save:
        matplotlib.use("Agg")  # headless backend when only saving
    import matplotlib.pyplot as plt

    x_col = args.x if args.x in header else header[0]
    xi = header.index(x_col)

    def col(j):
        out = []
        for r in data:
            try:
                out.append(float(r[j]))
            except (ValueError, IndexError):
                out.append(float("nan"))
        return out

    xs = col(xi)
    plt.figure()
    for j, name in enumerate(header):
        if j == xi:
            continue
        plt.plot(xs, col(j), marker=".", label=name)
    plt.xlabel(x_col)
    plt.legend()
    plt.title(args.csv_path)
    if args.save:
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"wrote {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
