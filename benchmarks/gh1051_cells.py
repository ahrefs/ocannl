"""Shared cell parsing, validation and tables for the gh-ocannl-1051 drivers.

Used by gh1051_bf16_ab.sh (mode "ab") and gh1051_boundary_split.sh (mode "split"), so that the
two cannot drift apart on what a valid cell is. The ab driver's cells are bin_cells.py runs, one
per size (`load_runs`: a cell counts only when its run found it clean -- exit 0, its resolved
configuration its command line, no failed output check); the split driver keeps its own cell logs
(`load`), since the two revisions it measures predate the configuration stream.

A timing is accepted only from a cell whose OUTPUT is validated, not merely whose exit status is.
schedule_bench's own verdict cannot do that for bf16: its k-term partial sums round, so it reports
cell differences as rounding (since 78d664c1; before that it exited 1 on them), and the NARROW
tensorized arm's error is gross by design -- it is the finding (a 2048-term cell reads 93 where the
exact answer is 536) -- so no single error bound separates it from a corrupt kernel. The checks
are therefore read off the bench's position-weighted whole-output checksums (`chk a/b`), which see
every cell. Equality checks compare the checksums as printed: losslessly since schedule_bench
renders them with %.17g (the gh-ocannl-1051 change to Bench_checksum.render), to ten significant
digits at older revisions -- the split mode's two revisions among them -- and the report says
which it could establish. Equal checksums are strong evidence of equal outputs, not
proof: a weighted double sum can in principle collide.

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
  gross by design, run the same arithmetic and must print IDENTICAL checksums, and each
  checksum stream must be NON-DEGENERATE: the exact answer's sign and at least a tenth of its
  magnitude, which a zeroed or sign-flipped output fails. RESIDUAL, stated rather than hidden: a
  deterministic defect shared by both pipelinings that still yields plausible magnitudes is not
  detectable here -- gfx11's bf16-accumulate is not exactly rounded, and this bench's operands are
  exact in bf16 at no k, so no cheap independent oracle exists at the driver. The narrow arm's
  values are pinned where one does: schedule_mma_matmul's Bf16_narrow twins (a known-answer k-split
  envelope and the two-sided narrowing-bound claim).
- TENSORIZED means tensorized: every bf16 mma_pd1/mma_pd2 cell's census must report
  Mma_intrinsics and no scalar fallback -- schedule_bench only warns when a Tile_mma declines (no
  tile MMA on the device, no rocWMMA headers), and a scalar timing must not stand in for a WMMA one.
- EVERY log is accounted for: a cell file the parser cannot name (a mistyped size, say) is a
  problem, not silently skipped, and so is a bin_cells run whose cells are not all there.
- SPLIT mode: the two revisions differ only in how the wide d boundary is addressed, with identical
  conversions, so every variant's checksum must be identical across them; the mma cells
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

import bench_output
import cells as cell_runner
from display_safe import display_value, log_line, table_lines

VARIANTS = ["parallel", "smem", "regtile", "mma_pd1", "mma_pd2"]
DIGITS = set()
# The structural bound (see above): five times the ~5% a legitimately per-step-narrowed 2048-term
# bf16 cell shows on minix and tuf, a quarter of what a zeroed output shows.
REL_BOUND = 0.25


def digits(chk):
    """The most significant digits any stream of a printed checksum carries."""
    return max(len(re.sub(r"[^0-9]", "", x.split("e")[0]).lstrip("0")) for x in chk.split("/"))


def parse_text(text):
    """{variant: (ms, (chk_a, chk_b), mma census)} for one cell's output, its timing lines read by
    bench_output's one reader of them (gh-ocannl-1767); ValueError for a timing line that does not
    read as one."""
    out = {}
    for t in bench_output.timing_lines(text):
        if t["variant"] in VARIANTS:
            if t["chk"] is None:
                raise ValueError(f"{t['variant']}: a timing line without its checksum")
            chk = tuple(float(x) for x in t["chk"].split("/"))
            DIGITS.add(digits(t["chk"]))
            out[t["variant"]] = (t["ms"], chk, t["mma"])
    return out


def parse_log(path):
    """`parse_text` of one cell log."""
    with open(path) as f:
        return parse_text(f.read())


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


def non_degenerate(c, exact):
    return all(x * y > 0 and abs(x) >= 0.1 * abs(y) for x, y in zip(c, exact))


def rel(a, b):
    return max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(a, b))


def add_cell(cells, problems, name, key, parsed):
    """One cell's timings into `cells`, keyed (size, arm, prec, variant); a missing variant or a
    tensorized one that did not tensorize is a problem of `name`."""
    for v in VARIANTS:
        if v not in parsed:
            problems.append(f"{name}: MISSING timing for {v}")
            continue
        census = parsed[v][2]
        if key["prec"] == "bfloat16" and v.startswith("mma") and (
            "Mma_intrinsics" not in census or "fallback" in census
        ):
            problems.append(f"{name}: {v} did not tensorize [{census}]")
        cells.setdefault(tuple(key[k] for k in ("size", "arm", "prec")) + (v,), []).append(
            parsed[v]
        )


def load(out, pattern, key_of):
    cells, problems = {}, []
    for f in sorted(glob.glob(os.path.join(out, pattern))):
        key = key_of(os.path.basename(f))
        if key is None:
            problems.append(f"{display_value(os.path.basename(f))}: not a cell this driver can name")
            continue
        ok, why = status_ok(f, key["prec"])
        if not ok:
            problems.append(f"{os.path.basename(f)}: {why}")
        try:
            parsed = parse_log(f)
        except ValueError as e:
            problems.append(f"{display_value(os.path.basename(f))}: {display_value(str(e))}")
            continue
        add_cell(cells, problems, os.path.basename(f), key, parsed)
    return cells, problems


ARM_RE = re.compile(r"(true|false)-(bfloat16|single)")


def load_runs(out):
    """The ab driver's cells: one bin_cells run per size under `out` (`n<size>/`), each cell named
    by its arm `<bf16_arithmetic>-<prec>`. A cell its run did not find clean, a run missing a
    planned cell or failing a run-level check, an unnamed arm and a stray run directory are
    problems."""
    cells, problems = {}, []
    for run in sorted(glob.glob(os.path.join(out, "n*"))):
        name = os.path.basename(run)
        m = re.fullmatch(r"n(\d+)", name)
        if not m:
            problems.append(f"{display_value(name)}: not a run this driver can name")
            continue
        try:
            manifest, records = cell_runner.read_run(run)
        except (OSError, ValueError) as e:
            problems.append(f"{name}: no complete bin_cells run ({display_value(str(e))})")
            continue
        size = int(m.group(1))
        if manifest.get("common_args", [None])[:1] != [str(size)]:
            problems.append(f"{name}: the run measured {display_value(str(manifest.get('common_args')))}")
        planned = sum(len(order) for order in manifest["orders"])
        if len(records) != planned:
            problems.append(f"{name}: {len(records)} of {planned} planned cells ran")
        problems += [f"{name}: {display_value(p)}"
                     for p in cell_runner.run_level_problems(manifest, records)]
        for r in records:
            cell = f"{name} cell {r['cell']} ({display_value(r['arm'])})"
            arm = ARM_RE.fullmatch(r["arm"])
            if not arm:
                problems.append(f"{cell}: not an arm this driver can name")
                continue
            if not r["ok"]:
                problems.append(f"{cell}: {display_value('; '.join(r['problems']))}")
                continue
            text = cell_runner.cell_stream(run, r, "stdout")
            if "FAILED" in text or "SIZE " in text:
                problems.append(f"{cell}: FAILED variant or SIZE mismatch")
            try:
                parsed = parse_text(text)
            except ValueError as e:
                problems.append(f"{cell}: {display_value(str(e))}")
                continue
            add_cell(cells, problems, cell, {"size": size, "arm": arm.group(1),
                                             "prec": arm.group(2)}, parsed)
    if not cells and not problems:
        problems.append("no bin_cells run under the output directory")
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
    cells, problems = load_runs(out)
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
        for v, c in (("mma_pd1", p1), ("mma_pd2", p2)):
            if c is not None and not non_degenerate(c, exact):
                problems.append(f"n={size} narrow {v}: checksum {c} degenerate against exact {exact}")
        if p1 is not None:
            print(f"n={size}: narrow tensorized checksum off the exact one by {rel(p1, exact):.3g} "
                  f"(the arm's accuracy, reported not bounded)")
    shown, rows = display_value(label), []
    for size in sizes:
        for v in VARIANTS:
            try:
                a, w = cells[(size, "true", "bfloat16", v)], cells[(size, "false", "bfloat16", v)]
                ca, cw = cells[(size, "true", "single", v)], cells[(size, "false", "single", v)]
            except KeyError:
                rows.append((shown, str(size), v, "missing", "missing", "", "", "", ""))
                continue
            exact = cw[0][1]
            err = f"{rel(a[0][1], exact):.2g} / {rel(w[0][1], exact):.2g}"
            rows.append((shown, str(size), v, fmt(a), fmt(w), f"{med(w) / med(a):.3f}",
                         f"{med(cw) / med(ca):.3f}", err, w[0][2]))
    print()
    print("\n".join(table_lines(
        ["memory", "n", "variant", "narrow (true) ms", "wide (false) ms", "wide/narrow",
         "f32 control wide/narrow", "chk err narrow / wide", "census (wide)"], rows)))
    return problems


def split(out, label):
    def key_of(name):
        m = re.match(r"n(\d+)_r(\d+)_(before|after)\.log$", name)
        return m and {"size": int(m.group(1)), "arm": m.group(3), "prec": "bfloat16"}

    cells, problems = load(out, "n*.log", key_of)
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
    shown, rows = display_value(label), []
    for size in sizes:
        for v in VARIANTS:
            b, a = cells[(size, "before", "bfloat16", v)], cells[(size, "after", "bfloat16", v)]
            role = "treatment" if v.startswith("mma") else "control"
            rows.append((shown, str(size), v, role, fmt(b), fmt(a), f"{med(a) / med(b):.3f}",
                         a[0][2]))
    print()
    print("\n".join(table_lines(
        ["memory", "n", "variant", "role", "before ms", "after ms", "after/before",
         "census (after)"], rows)))
    return problems


if __name__ == "__main__":
    mode, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
    problems = {"ab": ab, "split": split}[mode](out, label)
    # %.10g never prints more than ten significant digits and %.17g is lossless whatever it prints
    # (it drops trailing zeros), so a longer checksum anywhere proves the lossless renderer.
    if max(DIGITS, default=0) > 10:
        print("checksum equality compared losslessly (the bench renders %.17g)")
    else:
        print("checksum equality compared as printed: at most ten significant digits, lossless only "
              "if this bench renders %.17g (older revisions' %.10g may round)")
    for p in problems:
        print(log_line(f"INVALID: {p}"))
    sys.exit(1 if problems else 0)
