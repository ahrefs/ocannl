#!/usr/bin/env python3
"""Compare the headerless UTF-8 TSVs from --dump-bounded-outcomes.

Read exactly three tab-separated fields per row: a nonblank opaque store
label, a comma-separated permutation of zero-based indices or "batch",
and either "refused" or the printed v, u, k assignments. An assignment has
a decimal size with an opaque basis token, or is "unsized". Headers, blank
rows, empty dumps, and repeated (store, order) keys are invalid. The input
order is immaterial. Comparing keys checks coverage relative to the other
dump, without reconstructing the sweep or interpreting constraints or bases.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys


DIMENSION = r"(?:unsized|-?(?:0|[1-9][0-9]*) of [^\s=]+)"
ANSWER = re.compile(rf"v={DIMENSION} u={DIMENSION} k={DIMENSION}")
REFUSED = "refused"
TRANSITIONS = ("refused-to-accepted", "accepted-to-refused", "changed-answer")


def valid_order(order):
    if order == "batch":
        return True
    indices = order.split(",")
    return set(indices) == {str(index) for index in range(len(indices))}


def read_dump(path):
    rows = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{number}: expected three tab-separated fields")
            store, order, outcome = fields
            if not store.strip() or not valid_order(order):
                raise ValueError(f"{path}:{number}: invalid store or order")
            if outcome != REFUSED and not ANSWER.fullmatch(outcome):
                raise ValueError(f"{path}:{number}: invalid outcome")
            key = (store, order)
            if key in rows:
                raise ValueError(f"{path}:{number}: duplicate key {store!r}, {order}")
            rows[key] = outcome
    if not rows:
        raise ValueError(f"{path}: empty dump")
    return rows


def print_acceptance(label, rows):
    stores = set()
    runs = Counter()
    for (store, order), outcome in rows.items():
        if outcome != REFUSED:
            stores.add(store)
            runs[order == "batch"] += 1
    print(f"{label}: {len(stores)} accepted stores, "
          f"{runs[False]} accepted ordered runs, {runs[True]} accepted batches")


def compare(base, head, expected):
    print_acceptance("base", base)
    print_acceptance("head", head)
    differences = Counter()
    for key in sorted(base.keys() | head.keys()):
        before, after = base.get(key), head.get(key)
        if before == after:
            continue
        store, order = key
        if before is None:
            kind = "missing-from-base"
        elif after is None:
            kind = "missing-from-head"
        elif before == REFUSED:
            kind = "refused-to-accepted"
        elif after == REFUSED:
            kind = "accepted-to-refused"
        else:
            kind = "changed-answer"
        differences[kind, order == "batch"] += 1
        print(kind, store, order, before or "-", after or "-", sep="\t")

    mismatch = False
    for kind, wanted in expected.items():
        ordered, batched = differences[kind, False], differences[kind, True]
        total = ordered + batched
        print(f"{kind}: {total} ({ordered} ordered, {batched} batched), expected {wanted}")
        mismatch |= total != wanted
    return int(mismatch)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Exit 1 for unexpected differences, 2 for invalid input or arguments.",
        allow_abbrev=False,
    )
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    for kind in TRANSITIONS:
        parser.add_argument(f"--expect-{kind}", type=int, default=0, metavar="COUNT",
                            help=f"expect exactly COUNT {kind} rows (default: 0)")
    args = parser.parse_args()
    expected = {kind: getattr(args, f"expect_{kind.replace('-', '_')}") for kind in TRANSITIONS}
    if any(count < 0 for count in expected.values()):
        parser.error("expected counts must be nonnegative")
    expected.update({"missing-from-base": 0, "missing-from-head": 0})
    try:
        base, head = read_dump(args.base), read_dump(args.head)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    return compare(base, head, expected)


if __name__ == "__main__":
    sys.exit(main())
