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

SCHEMA_VERSION = 1


def hex_f64(x):
    """A Python float as the hex of its 64-bit pattern, e.g. `"0x3ff0000000000000"`."""
    x = float(x)
    assert not math.isnan(x), "NaN is not allowed in fixtures"
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def hex_f32(x):
    """A Python float as the hex of its 32-bit pattern, e.g. `"0x3f800000"`."""
    x = float(x)
    assert not math.isnan(x), "NaN is not allowed in fixtures"
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


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


def boolean(b):
    return {"kind": "bool", "v": bool(b)}


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


def metadata(generator, seed, sampling, libraries=("numpy", "scipy", "mpmath")):
    """Provenance header: generator name, live library versions, seed, date, and a
    note on how inputs were sampled."""
    return {
        "generator": generator,
        "libraries": {name: version(name) for name in libraries},
        "seed": int(seed),
        "date": _generation_date(),
        "sampling": sampling,
    }


def write_fixture(out, module, case, meta, tolerances, inputs, expected):
    """Writes `{out}/v1/{module}/{case}.json`, creating directories as needed."""
    # A missing f64/host key would make the loader fall back to an exact-match
    # (zero) tolerance, silently demanding bit-exact equality. Fail loudly instead.
    assert "f64/host" in tolerances, (
        f"fixture {module}/{case} has no f64/host tolerance"
    )
    obj = {
        "schema_version": SCHEMA_VERSION,
        "metadata": meta,
        "module": module,
        "case": case,
        "tolerances": {"table": tolerances},
        "inputs": inputs,
        "expected": expected,
    }
    path = os.path.join(out, "v1", module, case + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=True)
        f.write("\n")
