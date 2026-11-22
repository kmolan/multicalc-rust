"""Fixture writer mirroring the Rust schema in ../src/schema.rs.

Floats are stored as hex IEEE-754 bit patterns so a fixture round-trips exactly.
The endianness is forced to little-endian on both ends, so the hex string is
host-independent. NaN is rejected before writing. Output is written with sorted
keys and a trailing newline, so a regenerated fixture is byte-for-byte stable.
"""

import json
import math
import os
import struct
from datetime import datetime, timezone
from importlib.metadata import version

SCHEMA_VERSION = 2


def hex_f64(x):
    """A Python float as the hex of its 64-bit pattern, e.g. `"0x3ff0000000000000"`."""
    x = float(x)
    assert not math.isnan(x), "NaN is not allowed in fixtures"
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


# --- value builders ---


def scalar(x):
    return {"kind": "scalar", "v": hex_f64(x)}


def vector(xs):
    return {"kind": "vector", "data": [hex_f64(x) for x in xs]}


def matrix(rows_data):
    """Builds a matrix value from a 2-D array-like, stored row-major."""
    rows = [list(row) for row in rows_data]
    r = len(rows)
    c = len(rows[0]) if r else 0
    flat = []
    for row in rows:
        assert len(row) == c, "ragged matrix"
        flat.extend(hex_f64(x) for x in row)
    return {"kind": "matrix", "rows": r, "cols": c, "row_major": flat}


def integer(n):
    return {"kind": "int", "v": int(n)}


def string(s):
    return {"kind": "str", "v": str(s)}


# --- tolerances and metadata ---


def tol(abs_, rel):
    return {"abs": abs_, "rel": rel}


def _generation_date():
    # Honour SOURCE_DATE_EPOCH so a regeneration can reproduce the metadata date
    # (and thus the whole file) byte-for-byte; otherwise stamp the current time.
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def metadata(generator, seed, sampling, libraries, reference):
    """Provenance shared by every fixture one generator writes: its name, the live
    library versions, the seed, the date, a note on how inputs were sampled, and
    where the goldens came from.

    `reference` is a label with the library name in braces, e.g.
    `"numpy/LAPACK {numpy}"`; the version is filled in from the same lookup that
    records `libraries`, so the two cannot disagree."""
    libs = {name: version(name) for name in libraries}
    return {
        "generator": generator,
        "libraries": libs,
        "seed": int(seed),
        "date": _generation_date(),
        "sampling": sampling,
        "reference": reference.format(**libs),
    }


def write_fixture(
    out, module, case, meta, tolerances, inputs, expected, *, equation, operations
):
    """Writes `{out}/{module}/{case}.json`, creating directories as needed.

    `equation` is the problem this fixture is built on and `operations` names what
    was computed from it, one entry per row it contributes to an accuracy table."""
    assert "f64" in tolerances, f"fixture {module}/{case} has no f64 tolerance"
    assert operations, f"fixture {module}/{case} names no operation"
    obj = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {**meta, "equation": equation, "operations": list(operations)},
        "module": module,
        "case": case,
        "tolerances": tolerances,
        "inputs": inputs,
        "expected": expected,
    }
    path = os.path.join(out, module, case + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=True)
        f.write("\n")
