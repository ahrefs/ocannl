"""Shared cell parsing, validation and tables for the gh-ocannl-1051 drivers.

Used by gh1051_bf16_ab.sh (mode "ab") and gh1051_boundary_split.sh (mode "split"), so that the
two cannot drift apart on what a valid cell is.

A timing is accepted only from a cell whose OUTPUT is validated, not merely whose exit status is.
schedule_bench's own verdict cannot do that for bf16: its k-term partial sums round, so it reports
cell differences as rounding (since 78d664c1; before that it exited 1 on them), and the NARROW
tensorized arm's error is gross by design -- it is the finding (a 2048-term cell reads 93 where the
exact answer is 536) -- so no single error bound separates it from a corrupt kernel. The checks
are therefore read off the bench's position-weighted whole-output checksums (`chk a/b`), which see
every cell:

- DETERMINISM, every cell: each (size, arm, precision, variant) prints the identical checksum in
  every round. A racy or uninitialized kernel does not.
- STRUCTURE, not accuracy. The f32 control computes the same operands exactly (they are exact in
  bf16 too), so its checksum is the exact answer's. Wide bf16 cells, and the narrow arm's SERIAL
  legs, must lie within REL_BOUND of it -- a bound on corruption (a zeroed, duplicated or misplaced
  block moves the checksum by its own share of the output), NOT an accuracy claim: per-step bf16
  narrowing over 2048 terms is legitimately ~5% off exact, and so is schedule_bench's regtile
  schedule even under Bf16_wide on HIP, whose privatized register tile keeps bf16 residency (its
  2048 checksum equals the narrow serial legs' bit for bit). Each cell's error is REPORTED in the
  table instead. The narrow arm's two tensorized pipelinings (mma_pd1, mma_pd2), whose error is
  gross by design, run the same arithmetic and must agree with each other BITWISE.
- SPLIT mode: the two revisions differ only in how the wide d boundary is addressed, with identical
  conversions, so every variant's checksum must be bitwise identical across them; the mma cells
  must also lie within REL_BOUND of the same run's parallel (serial, wide) checksum.

Exit statuses: 0, or 1 only for a bf16 cell whose log carries the pre-78d664c1 bench's rounding
verdict ("WRONG RESULT") and no SIZE mismatch -- a size mismatch is never rounding. Any FAILED
variant or missing timing line fails the run.
"""
import glob
import os
import re
import statistics
import sys

VARIANTS = ["parallel", "smem", "regtile", "mma_pd1", "mma_pd2"]
# The structural bound (see above): five times the ~5% a legitimately per-step-narrowed 2048-term
# bf16 cell shows on minix and tuf, a quarter of what a zeroed output shows.
REL_BOUND = 0.25
LINE = re.compile(r"^(\w+)\s+([0-9.]+) ms\s+([0-9.]+) GFLOP/s\s+\(.*chk ([^,]+),.*\[(.*)\]\s*$")


def parse_log(path):
    """{variant: (ms, (chk_a, chk_b), census)} for one cell log."""
    out = {}
    for line in open(path):
        m = LINE.match(line)
        if m and m.group(1) in VARIANTS:
            chk = tuple(float(x) for x in m.group(4).split("/"))
            out[m.group(1)] = (float(m.group(2)), chk, m.group(5))
    return out


def status_ok(path, prec):
    """The cell's exit status (recorded by the driver in <log>.status) is an accepted one."""
    status = int(open(path + ".status").read().strip())
    text = open(path).read()
    if "FAILED" in text or "SIZE " in text:
        return False, "FAILED variant or SIZE mismatch"
    if status == 0:
        return True, ""
    if status == 1 and prec == "bfloat16" and "WRONG RESULT" in text:
        return True, ""
    return False, f"unexpected exit {status}"


def rel(a, b):
    return max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(a, b))


def load(out, pattern, key_of):
    cells, problems = {}, []
    for f in sorted(glob.glob(os.path.join(out, pattern))):
        key = key_of(os.path.basename(f))
        if key is None:
            continue
        ok, why = status_ok(f, key["prec"])
        if not ok:
            problems.append(f"{os.path.basename(f)}: {why}")
        parsed = parse_log(f)
        for v in VARIANTS:
            if v not in parsed:
                problems.append(f"{os.path.basename(f)}: MISSING timing for {v}")
                continue
            cells.setdefault(tuple(key[k] for k in ("size", "arm", "prec")) + (v,), []).append(
                parsed[v]
            )
    return cells, problems


def checksum_of(cells, key, problems):
    """The checksum every round of this cell printed, or None (recording a problem) if they differ."""
    chks = {c for _, c, _ in cells.get(key, [])}
    if len(chks) != 1:
        problems.append(f"{key}: checksum not deterministic across rounds ({len(chks)} distinct)")
        return None
    return next(iter(chks))


def fmt(xs):
    ms = [x for x, _, _ in xs]
    return f"{statistics.median(ms):.3f} ({min(ms):.3f}-{max(ms):.3f})"


def med(xs):
    return statistics.median(x for x, _, _ in xs)


def ab(out, label):
    def key_of(name):
        m = re.match(r"n(\d+)_r(\d+)_(\w+?)_(bfloat16|single)\.log$", name)
        return m and {"size": int(m.group(1)), "arm": m.group(3), "prec": m.group(4)}

    cells, problems = load(out, "n*_r*_*_*.log", key_of)
    sizes = sorted({k[0] for k in cells})
    for size in sizes:
        chk = {}
        for arm in ("true", "false"):
            for prec in ("bfloat16", "single"):
                for v in VARIANTS:
                    chk[(arm, prec, v)] = checksum_of(cells, (size, arm, prec, v), problems)
        exact = chk[("false", "single", "parallel")]
        if exact is None:
            continue
        for arm in ("true", "false"):
            for v in VARIANTS:
                c = chk[(arm, "single", v)]
                if c is not None and c != exact:
                    problems.append(f"n={size} {arm} f32 control {v}: checksum differs from the exact one")
        for v in VARIANTS:
            c = chk[("false", "bfloat16", v)]
            if c is not None and rel(c, exact) > REL_BOUND:
                problems.append(f"n={size} wide {v}: checksum off the exact one by {rel(c, exact):.3g}")
        for v in ("parallel", "smem", "regtile"):
            c = chk[("true", "bfloat16", v)]
            if c is not None and rel(c, exact) > REL_BOUND:
                problems.append(f"n={size} narrow {v}: checksum off the exact one by {rel(c, exact):.3g}")
        p1, p2 = chk[("true", "bfloat16", "mma_pd1")], chk[("true", "bfloat16", "mma_pd2")]
        if p1 is not None and p2 is not None and p1 != p2:
            problems.append(f"n={size} narrow mma_pd1/mma_pd2: checksums differ ({p1} vs {p2})")
        if p1 is not None:
            print(f"n={size}: narrow tensorized checksum off the exact one by {rel(p1, exact):.3g} "
                  f"(the arm's accuracy, reported not bounded)")
    print("\n| memory | n | variant | narrow (true) ms | wide (false) ms | wide/narrow "
          "| f32 control wide/narrow | chk err narrow / wide | census (wide) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for size in sizes:
        for v in VARIANTS:
            try:
                a, w = cells[(size, "true", "bfloat16", v)], cells[(size, "false", "bfloat16", v)]
                ca, cw = cells[(size, "true", "single", v)], cells[(size, "false", "single", v)]
            except KeyError:
                print(f"| {label} | {size} | {v} | missing | missing | | | | |")
                continue
            exact = cw[0][1]
            err = f"{rel(a[0][1], exact):.2g} / {rel(w[0][1], exact):.2g}"
            print(f"| {label} | {size} | {v} | {fmt(a)} | {fmt(w)} | {med(w) / med(a):.3f} | "
                  f"{med(cw) / med(ca):.3f} | {err} | {w[0][2]} |")
    return problems


def split(out, label):
    def key_of(name):
        m = re.match(r"n(\d+)_r(\d+)_(before|after)\.log$", name)
        return m and {"size": int(m.group(1)), "arm": m.group(3), "prec": "bfloat16"}

    cells, problems = load(out, "n*_r*_*.log", key_of)
    sizes = sorted({k[0] for k in cells})
    for size in sizes:
        for v in VARIANTS:
            b = checksum_of(cells, (size, "before", "bfloat16", v), problems)
            a = checksum_of(cells, (size, "after", "bfloat16", v), problems)
            if a is not None and b is not None and a != b:
                problems.append(f"n={size} {v}: checksum changed across the revisions ({b} -> {a})")
        par = checksum_of(cells, (size, "after", "bfloat16", "parallel"), [])
        for v in ("mma_pd1", "mma_pd2"):
            c = checksum_of(cells, (size, "after", "bfloat16", v), [])
            if par is not None and c is not None and rel(c, par) > REL_BOUND:
                problems.append(f"n={size} {v}: checksum off the serial wide one by {rel(c, par):.3g}")
    print("\n| memory | n | variant | role | before ms | after ms | after/before | census (after) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for size in sizes:
        for v in VARIANTS:
            b, a = cells[(size, "before", "bfloat16", v)], cells[(size, "after", "bfloat16", v)]
            role = "treatment" if v.startswith("mma") else "control"
            print(f"| {label} | {size} | {v} | {role} | {fmt(b)} | {fmt(a)} | "
                  f"{med(a) / med(b):.3f} | {a[0][2]} |")
    return problems


if __name__ == "__main__":
    mode, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
    problems = {"ab": ab, "split": split}[mode](out, label)
    for p in problems:
        print("INVALID:", p)
    sys.exit(1 if problems else 0)
