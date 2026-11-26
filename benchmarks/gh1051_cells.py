"""Shared cell parsing, validation and tables for the gh-ocannl-1051 drivers.

Used by gh1051_bf16_ab.sh (GEMM and training modes, on HIP and Metal),
gh2195_f16_ab.sh (the same two modes at `half`) and gh1051_boundary_split.sh (mode "split"), so
that they cannot drift apart on what a valid cell is. The ab driver's cells are bin_cells.py runs, one
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

HALF (mode "ab" with PREC `half`, gh-ocannl-2195's gh2195_f16_ab.sh: the fp16_arithmetic arms
`false`, wide, and `true`, narrow, on CUDA and HIP). The cells, arms and checks above are the
same, at `half` in place of `bfloat16`, except where bf16's reasons do not carry over:
- The f32 control is still the exact answer: the operands are exact in f16 too (ma's 1/16
  multiples below 3 need six significand bits, f16 has eleven).
- STRUCTURE holds as for bf16: the wide cells and the narrow SERIAL legs lie within REL_BOUND, and
  the narrow tensorized pair is bounded only below (non-degenerate), its error REPORTED. Bounding it
  structurally was this driver's first reading, and the gfx1151 window refuted it (gh-ocannl-2195,
  minix run-44): the narrow f16 WMMA's checksum missed the exact one by 0.072 at n = 1024 and by
  0.822 at n = 2048, where the narrow serial legs missed by 0.0011 and 0.0019 -- the per-pair
  narrowing of gfx11's f16-accumulate WMMA grows grossly with k, as its bf16 twin does.
- The two tensorized pipelinings print IDENTICAL checksums in BOTH arms: the pipelining depth is
  bitwise neutral by the transform's invariant (bin/schedule_bench.ml), whatever the accumulator.
- TENSORIZED applies to both arms' f16 mma cells, as to bf16's.
- Reported, not checked: whether the wide arm's five variants share one checksum. Every partial
  sum is exact in f32 here, so an f32-accumulating variant stores round_f16(exact) whatever its
  order, and a variant off the shared value narrowed somewhere (or, on HIP, met gfx11's mixed-sign
  WMMA ulps, gh-ocannl-1420) -- evidence about the arm, not a corrupt cell.
- f16 holds schedule_bench's exact verdict only to k = 5, below every size the mma variants
  accept, so the bench reports its cell differences as rounding (a NOTE, exit 0); a cell that
  exits nonzero is a problem here exactly as in bf16's ab mode (bin_cells' clean-cell rule), and
  the WRONG RESULT exemption above belongs to the split mode's pre-78d664c1 logs alone.

PAIRED RATIOS (mode "ab"). The table's wide/narrow columns, treatment and f32 control alike, are
paired round by round -- each round's wide and narrow cells of one size and precision ran in one
pass of the mirrored rotation -- as the median and range of the per-round ratios; the control's
range is the session's drift a treatment ratio is read against. Before gh-ocannl-2150 the columns
divided the arms' medians, which hid that range. Reported beside the table, per size and precision:
each arm's fastest variant by median and the paired ratio of those two fixed schedules' round
timings -- what a tuner crowning each arm's fastest of these five variants pays for the wide arm,
which differs from any one variant's ratio when the arms' fastest variants differ (on Metal the
narrow arm's is a tile and the wide arm's the serial regtile, gh-ocannl-2150).

METAL (gh-ocannl-2150, the bf16_arithmetic arms on Apple GPUs). The checks above carry over, with
the narrow arm's reasons changed but not its checks: Metal's narrow tensorized arm (the
`simdgroup_bfloat8x8` accumulator) sums each 8-deep k block in f32 and rounds to bf16 once per block
(docs/agent-notes/backend-precision-and-simd.md, gh-ocannl-2029), so it is MORE accurate than the
narrow serial legs' per-step bfloat FMA rather than gross as gfx11's is; it is still bounded only
below (non-degenerate) and its error reported, since no structural bound separates a k-blocked
narrowing from a corrupt kernel any better here. TENSORIZED reads Metal's census the same way
(`Mma_intrinsics`, no scalar fallback).

UNTUNED (mode "untuned RUN LABEL FIXTURE PREC"): the drivers' training row, one bin_cells
`--reader record` run of bench_gpt's untuned step (no search) on a recorded fixture, four arms
`<flag>-<PREC>` and `<flag>-single`, metric step_ms.p50, its wide/narrow ratio paired by round.
Moved here from gh2195_f16_ab.sh's inline summary so the bf16 driver reads the row the same way.
Its checks: every planned cell ran and is clean, each cell's result line reports the storage
precision its arm asked for, and each arm's seeding census (the `seeded_default:` lines
autotune_log prints) is the same in every round. Whether the two treatment arms' censuses differ is
REPORTED, as is each arm's shipped tensorization (the result line's `shipped_mma`), which says
whether the row reaches a tensor-unit arm at all. Identical censuses mean the arms seed the same
choices, not that the treatment misses the row's codegen: an accumulator width can change inside a
seeded choice. Reach is the generated sources' question, which the bf16 driver's source proof
answers (gh2195_f16_ab.sh runs none for this row).

TUNED: the same four-arm design with the bf16 cells replaying independent cold crowns and both
f32 drift controls untuned. The closed provenance check reads the bench_gpt JSON result records
for the cold searches, the source-proof replays and the timed cells. It requires placement
replay of a measured crown and reports MMA from the shipped artifact. Candidate diagnostics
and generated kernel text stay with the driver.
"""
import glob
import os
import re
import statistics
import sys
from pathlib import Path

import bench_output
import cells as cell_runner
from display_safe import display_value, log_line, table_lines

VARIANTS = ["parallel", "smem", "regtile", "mma_pd1", "mma_pd2"]
DIGITS = set()
# The structural bound (see above): five times the ~5% a legitimately per-step-narrowed 2048-term
# bf16 cell shows on minix and tuf, a quarter of what a zeroed output shows.
REL_BOUND = 0.25
# The treatment precisions mode "ab" validates (the f32 control is `single` for both): bf16 for the
# gh-ocannl-1051 drivers, half for gh2195_f16_ab.sh (see HALF above).
NARROW_PRECS = ("bfloat16", "half")


def digits(chk):
    """The most significant digits any stream of a printed checksum carries."""
    return max(len(re.sub(r"[^0-9]", "", x.split("e")[0]).lstrip("0")) for x in chk.split("/"))


def parse_text(text, layouts=(bench_output.TIMING_CENSUSES,)):
    """{variant: (ms, (chk_a, chk_b), mma census)} for one cell's output, its timing lines read by
    bench_output's one reader of them (gh-ocannl-1767) in `layouts`; ValueError for a timing line
    that does not read as one."""
    out = {}
    for t in bench_output.timing_lines(text, layouts):
        if t["variant"] in VARIANTS:
            if t["chk"] is None:
                raise ValueError(f"{t['variant']}: a timing line without its checksum")
            chk = tuple(float(x) for x in t["chk"].split("/"))
            DIGITS.add(digits(t["chk"]))
            out[t["variant"]] = (t["ms"], chk, t["mma"])
    return out


def parse_log(path, layouts=(bench_output.TIMING_CENSUSES,)):
    """`parse_text` of one cell log."""
    with open(path) as f:
        return parse_text(f.read(), layouts)


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


def add_cell(cells, problems, name, key, parsed, rounds=None, round_=None):
    """One cell's timings into `cells`, keyed (size, arm, prec, variant); a missing variant or a
    tensorized one that did not tensorize is a problem of `name`. With `rounds` (a dict) and the
    cell's `round_`, each timing also goes to `rounds[key][round_]`, which the ab table pairs."""
    for v in VARIANTS:
        if v not in parsed:
            problems.append(f"{name}: MISSING timing for {v}")
            continue
        census = parsed[v][2]
        if key["prec"] in NARROW_PRECS and v.startswith("mma") and (
            "Mma_intrinsics" not in census or "fallback" in census
        ):
            problems.append(f"{name}: {v} did not tensorize [{census}]")
        k = tuple(key[k] for k in ("size", "arm", "prec")) + (v,)
        cells.setdefault(k, []).append(parsed[v])
        if rounds is not None:
            rounds.setdefault(k, {})[round_] = parsed[v][0]


def load(out, pattern, key_of, layouts=(bench_output.TIMING_CENSUSES,)):
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
            parsed = parse_log(f, layouts)
        except ValueError as e:
            problems.append(f"{display_value(os.path.basename(f))}: {display_value(str(e))}")
            continue
        add_cell(cells, problems, os.path.basename(f), key, parsed)
    return cells, problems


def arm_re(prec):
    """The ab driver's arm names at treatment precision `prec`: `<flag>-<prec>` or `<flag>-single`."""
    return re.compile(rf"(true|false)-({prec}|single)")


def load_runs(out, prec="bfloat16"):
    """The ab driver's cells: one bin_cells run per size under `out` (`n<size>/`), each cell named
    by its arm `<flag>-<prec>` (the bf16_arithmetic or fp16_arithmetic value). A cell its run did
    not find clean, a run missing a planned cell or failing a run-level check, an unnamed arm (one
    at another treatment precision included) and a stray run directory are problems. Returns the
    cells, the problems, and each cell's timing per round (`add_cell`'s `rounds`)."""
    arm_names = arm_re(prec)
    cells, problems, rounds = {}, [], {}
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
            arm = arm_names.fullmatch(r["arm"])
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
                                             "prec": arm.group(2)}, parsed, rounds, r["round"])
    if not cells and not problems:
        problems.append("no bin_cells run under the output directory")
    return cells, problems, rounds


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


def paired(wide, narrow):
    """The wide/narrow ratio paired round by round -- each round's two cells of one size and
    precision ran in the same pass of the mirrored rotation -- as `median (min-max)`, over the rounds
    both arms have; "missing" when they share none. Rounds are paired rather than medians divided
    because a round's drift moves both of its cells (the A/B protocol)."""
    common = sorted(set(wide) & set(narrow))
    if not common:
        return "missing"
    ratios = [wide[r] / narrow[r] for r in common]
    return f"{statistics.median(ratios):.3f} ({min(ratios):.3f}-{max(ratios):.3f})"


def ab(out, label, prec="bfloat16"):
    """Mode "ab" at treatment precision `prec` (`bfloat16`, or `half`: see HALF above)."""
    half = prec == "half"
    cells, problems, rounds = load_runs(out, prec)
    sizes = sorted({k[0] for k in cells})
    for size in sizes:
        chk = {}
        for arm in ("true", "false"):
            for p in (prec, "single"):
                for v in VARIANTS:
                    chk[(arm, p, v)] = checksum_of(cells, (size, arm, p, v), problems)
        exact = chk[("false", "single", "parallel")]
        if exact is None:
            continue
        for arm in ("true", "false"):
            for v in VARIANTS:
                c = chk[(arm, "single", v)]
                if c is not None and c != exact:
                    problems.append(f"n={size} {arm} f32 control {v}: checksum differs from the exact one")
        for v in VARIANTS:
            c = chk[("false", prec, v)]
            if c is not None and rel(c, exact) > REL_BOUND:
                problems.append(f"n={size} wide {v}: checksum off the exact one by {rel(c, exact):.3g}")
        # The narrow tensorized pair is gross by design on gfx11 (bf16 and f16 alike) and bounded
        # only below (non-degenerate); its error is reported. The serial legs take the bound.
        for v in ("parallel", "smem", "regtile"):
            c = chk[("true", prec, v)]
            if c is not None and rel(c, exact) > REL_BOUND:
                problems.append(f"n={size} narrow {v}: checksum off the exact one by {rel(c, exact):.3g}")
        if half:
            w1, w2 = chk[("false", prec, "mma_pd1")], chk[("false", prec, "mma_pd2")]
            if w1 is not None and w2 is not None and w1 != w2:
                problems.append(f"n={size} wide mma_pd1/mma_pd2: checksums differ ({w1} vs {w2})")
        p1, p2 = chk[("true", prec, "mma_pd1")], chk[("true", prec, "mma_pd2")]
        if p1 is not None and p2 is not None and p1 != p2:
            problems.append(f"n={size} narrow mma_pd1/mma_pd2: checksums differ ({p1} vs {p2})")
        for v, c in (("mma_pd1", p1), ("mma_pd2", p2)):
            if c is not None and not non_degenerate(c, exact):
                problems.append(f"n={size} narrow {v}: checksum {c} degenerate against exact {exact}")
        if p1 is not None:
            print(f"n={size}: narrow tensorized checksum off the exact one by {rel(p1, exact):.3g} "
                  + "(the arm's accuracy, reported not bounded)")
        if half:
            wide = {v: chk[("false", prec, v)] for v in VARIANTS}
            if None not in wide.values():
                groups = {}
                for v, c in wide.items():
                    groups.setdefault(c, []).append(v)
                shared = max(groups.values(), key=len)
                print(f"n={size}: wide arm: {len(shared)} of {len(VARIANTS)} variants share one "
                      f"checksum ({', '.join(shared)}), off the exact one by "
                      f"{rel(wide[shared[0]], exact):.3g}"
                      + ("" if len(groups) == 1 else
                         "; others: " + ", ".join(f"{', '.join(vs)} at {rel(c, exact):.3g}"
                                                  for c, vs in groups.items() if vs is not shared))
                      + " (reported, not checked)")
    for size in sizes:
        for p, what in ((prec, "treatment"), ("single", "f32 control")):
            def fastest(arm):
                return min(VARIANTS, key=lambda v: statistics.median(
                    rounds.get((size, arm, p, v), {}).values() or [float("inf")]))

            nv, wv = fastest("true"), fastest("false")
            print(f"n={size} {what}: fastest variant by median narrow {nv}, wide {wv}; those two "
                  "schedules' paired wide/narrow "
                  f"{paired(rounds.get((size, 'false', p, wv), {}), rounds.get((size, 'true', p, nv), {}))}"
                  " (what a tuner crowning each arm's fastest of these variants pays)")
    shown, rows = display_value(label), []
    for size in sizes:
        for v in VARIANTS:
            try:
                a, w = cells[(size, "true", prec, v)], cells[(size, "false", prec, v)]
                ca, cw = cells[(size, "true", "single", v)], cells[(size, "false", "single", v)]
            except KeyError:
                rows.append((shown, str(size), v, "missing", "missing", "", "", "", ""))
                continue
            exact = cw[0][1]
            err = f"{rel(a[0][1], exact):.2g} / {rel(w[0][1], exact):.2g}"
            def per_round(arm, p):
                return rounds.get((size, arm, p, v), {})

            rows.append((shown, str(size), v, fmt(a), fmt(w),
                         paired(per_round("false", prec), per_round("true", prec)),
                         paired(per_round("false", "single"), per_round("true", "single")),
                         err, w[0][2]))
    print()
    print("\n".join(table_lines(
        ["memory", "n", "variant", "narrow (true) ms", "wide (false) ms",
         "paired wide/narrow (range)", "f32 control paired wide/narrow (range)",
         "chk err narrow / wide", "census (wide)"], rows)))
    return problems


# The storage-precision labels bench_gpt's result line prints (Bench_harness.precision_label) for
# each treatment precision and the f32 control.
STORAGE_LABEL = {"bfloat16": "bf16", "half": "f16", "single": "f32"}


def training_crown(rec, *, replay, prec="bfloat16"):
    """The shipped placement, problem digest, actual MMA census and reported schedule label.

    Check the seed-only bench_gpt result schema. The cached form's label can differ from the
    search label, so only replay cells are compared on the label.
    """
    want_storage = STORAGE_LABEL[prec]
    if rec["variant"] != "tuned" or rec["storage_precision"] != want_storage:
        raise ValueError(f"expected a tuned {want_storage} training result")
    tune = rec["tune"]
    arms = tune["arms"]
    placement = tune["shipped"]
    expected = (0, 1, 0) if replay else (2, 0, 0)
    counts = tuple(tune[k] for k in ("searches", "replays", "no_searches"))
    searches_expected = not replay
    state = "cache-replay" if replay else "searched"
    names = {placement} if replay else {"A", "B"}
    if (counts != expected or rec["searched"] is not searches_expected
            or tune["placement_replay"] is not replay or tune["ship_substituted"] is not False
            or placement not in ("A", "B")
            or len(arms) != len(names) or {a["arm"] for a in arms} != names
            or any(a["state"] != state or a["terminal_failure"] is not None
                   or a["budget_cut"] or a["timings_contended"] for a in arms)):
        raise ValueError(f"expected {'one clean placement replay' if replay else 'two clean cold searches'}"
                         f", observed counts (search, replay, skipped) {counts}")
    winner = next(a for a in arms if a["arm"] == placement)
    census = rec["shipped_mma"]
    mma = (census["tensorization"], census["statements"], census["scalar_fallbacks"])
    digest, schedule = tune["shipped_digest"], winner["best_label"]
    if not digest or not schedule or not mma[0] or winner["best_ms"] is None:
        raise ValueError("missing shipped crown or MMA census")
    return placement, digest, mma, schedule


def read_training_crown(path, *, replay, prec="bfloat16"):
    try:
        return training_crown(bench_output.only_result_record(path.read_text()),
                              replay=replay, prec=prec)
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path.parent.name}: {e}") from e


def tuned_references(out, label=None, prec="bfloat16"):
    """Validate the cold searches and their source-proof replays before timing."""
    storage = STORAGE_LABEL[prec]
    references = {}
    for flag, policy in (("false", "wide"), ("true", "narrow")):
        cold = out / f"search_{flag}" / "stdout"
        proof = out / f"identity_run_train_{prec}_{flag}" / "stdout"
        searched = read_training_crown(cold, replay=False, prec=prec)
        replayed = read_training_crown(proof, replay=True, prec=prec)
        if searched[:3] != replayed[:3]:
            raise ValueError(f"{policy} {storage} source-proof replay differs from its cold crown")
        references[flag] = replayed
        if label is not None:
            placement, _, (tensorization, statements, fallbacks), schedule = searched
            print(log_line(
                f"{display_value(label)} tuned {policy} {storage}: placement {placement} shipped "
                f"{display_value(schedule)} (replay {display_value(replayed[3])}); "
                f"shipped MMA {display_value(tensorization)}, {statements} statements, "
                f"{fallbacks} scalar fallbacks"))
    return references


def tuned_result(reference, arm, stdout, prec="bfloat16"):
    """Admit a timed record only under the provenance for its planned arm."""
    flag, _, arm_prec = arm.partition("-")
    if flag not in reference or arm_prec not in ("single", prec):
        raise ValueError("unknown tuned training arm")
    rec = bench_output.only_result_record(stdout)
    try:
        if arm_prec == prec:
            if training_crown(rec, replay=True, prec=prec) != reference[flag]:
                raise ValueError("timed replay differs from the source-proof replay")
        elif (rec["storage_precision"] != "f32" or rec["variant"] != "default"
              or rec["searched"] is not False or rec.get("tune") is not None):
            raise ValueError("expected the untuned f32 drift control")
    except (KeyError, TypeError) as e:
        raise ValueError(f"unreadable training provenance ({e})") from e
    return rec


def training(run, label, fixture, prec, *, tuned=False):
    """Print the paired training table for one bin_cells record run and return its problems."""
    want = {prec: STORAGE_LABEL[prec], "single": "f32"}
    problems = []
    mode = "tuned" if tuned else "untuned"
    try:
        manifest, records = cell_runner.read_run(run)
        reference = tuned_references(Path(run).parent, prec=prec) if tuned else {}
    except (OSError, ValueError) as e:
        return [f"{mode}: invalid training run ({display_value(str(e))})"]
    planned = sum(len(order) for order in manifest["orders"])
    if len(records) != planned:
        problems.append(f"{len(records)} of {planned} planned cells ran")
    problems += [display_value(p) for p in cell_runner.run_level_problems(manifest, records)]
    # (arm, prec) -> {round: step p50 ms}; -> {census tuple}; -> {shipped tensorization}
    ms, census, shipped = {}, {}, {}
    for r in records:
        cell = f"cell {r['cell']} ({display_value(r['arm'])})"
        flag, _, p = r["arm"].partition("-")
        if flag not in ("false", "true") or p not in want:
            problems.append(f"{cell}: not an arm this driver can name")
            continue
        if not r["ok"]:
            problems.append(f"{cell}: {display_value('; '.join(r['problems']))}")
            continue
        try:
            stdout = cell_runner.cell_stream(run, r, "stdout")
            rec = (tuned_result(reference, r["arm"], stdout, prec=prec) if tuned
                   else bench_output.result_record(stdout))
            got = bench_output.field(rec, "storage_precision")
            step = bench_output.field(rec, "step_ms.p50")
        except (OSError, KeyError, ValueError, TypeError) as e:
            problems.append(f"{cell}: unreadable result line ({display_value(str(e))})")
            continue
        if got != want[p]:
            problems.append(f"{cell}: ran at storage precision {display_value(str(got))}, "
                            f"not {want[p]}")
        ms.setdefault((flag, p), {})[r["round"]] = step
        try:
            shipped.setdefault((flag, p), set()).add(
                str(bench_output.field(rec, "shipped_mma.tensorization")))
        except (KeyError, ValueError, TypeError):
            shipped.setdefault((flag, p), set()).add("not reported")
        if not tuned or p == "single":
            _, lines = bench_output.seeded_default(cell_runner.cell_stream(run, r, "stderr"))
            census.setdefault((flag, p), set()).add(tuple(lines))
    for key, seen in sorted(census.items()):
        if len(seen) != 1:
            problems.append(f"{'-'.join(key)}: the seeding census differs across rounds "
                            f"({len(seen)} distinct)")
    for flag in ("false", "true"):
        for lines in sorted(census.get((flag, prec), set())):
            print(log_line(f"untuned {flag}-{prec} census: "
                           f"{display_value(' | '.join(lines)) if lines else 'none logged'}"))
    for key, seen in sorted(shipped.items()):
        print(log_line(f"{mode} {'-'.join(key)} shipped tensorization: "
                       f"{display_value(', '.join(sorted(seen)))}"))
    wide, narrow = census.get(("false", prec)), census.get(("true", prec))
    if wide and narrow and len(wide) == len(narrow) == 1:
        if wide == narrow == {()}:
            print("untuned seeding: no seeding census logged in either arm (the seeded untuned "
                  "plan does not run here); the driver's source proof, where it ran one, says "
                  "whether the treatment reaches this row's codegen")
        else:
            print(f"untuned seeding: the two {STORAGE_LABEL[prec]} arms' seeding censuses "
                  + ("are IDENTICAL -- the arms seed the same choices; the census alone does not "
                     "show whether the accumulator width reaches this row's codegen"
                     if wide == narrow else "differ -- the arms seed different choices"))
    rows, shown = [], display_value(label)
    for p in (prec, "single"):
        w, n = ms.get(("false", p), {}), ms.get(("true", p), {})
        common = sorted(set(w) & set(n))
        if not common:
            rows.append((shown, display_value(fixture), want[p], "missing", "missing", "", ""))
            continue
        slower = sum(w[k] > n[k] for k in common)
        rows.append((shown, display_value(fixture), want[p],
                     f"{statistics.median(n.values()):.3f}", f"{statistics.median(w.values()):.3f}",
                     paired(w, n), f"{slower} of {len(common)}"))
    print()
    print("\n".join(table_lines(
        ["memory", f"tuned {STORAGE_LABEL[prec]} / untuned f32 (bench_gpt)" if tuned else "untuned default (bench_gpt)",
         "prec", "narrow (true) step p50 ms",
         "wide (false) step p50 ms", "paired wide/narrow (range)", "rounds wide slower"], rows)))
    print(f"(paired sign tests and parity losses: {display_value(run)}/report.md)")
    return [f"{mode}: {p}" for p in problems]


def split(out, label):
    def key_of(name):
        m = re.match(r"n(\d+)_r(\d+)_(before|after)\.log$", name)
        return m and {"size": int(m.group(1)), "arm": m.group(3), "prec": "bfloat16"}

    # The two revisions around gh-ocannl-1064 predate gh-ocannl-1668's grid census: their timing
    # lines carry the mma census alone.
    cells, problems = load(out, "n*.log", key_of,
                           (bench_output.TIMING_CENSUSES, bench_output.TIMING_CENSUSES_BEFORE_1668))
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
    # ab OUT LABEL [PREC]: PREC is the treatment precision, bfloat16 (the default) or half.
    mode, out = sys.argv[1:3]
    if mode in ("tuned-search", "tuned-cells"):
        try:
            if mode == "tuned-search":
                prec = sys.argv[3] if len(sys.argv) > 3 else "bfloat16"
                if prec not in NARROW_PRECS or len(sys.argv) > 4:
                    sys.exit(f"usage: gh1051_cells.py tuned-search DIR [{'|'.join(NARROW_PRECS)}]")
                read_training_crown(Path(out) / "stdout", replay=False, prec=prec)
                sys.exit(0)
            if len(sys.argv) >= 5 and sys.argv[3] == "--prec":
                prec = sys.argv[4]
                if prec not in NARROW_PRECS:
                    sys.exit(f"usage: gh1051_cells.py tuned-cells OUT [--prec {'|'.join(NARROW_PRECS)}] ...")
                rest = sys.argv[5:]
            else:
                prec = "bfloat16"
                rest = sys.argv[3:]
            references = tuned_references(Path(out), prec=prec)
        except ValueError as e:
            sys.exit(log_line(f"INVALID: {display_value(str(e))}"))
        import bin_cells

        sys.exit(bin_cells.main(rest,
                                validate_cell=lambda arm, stdout: tuned_result(references, arm, stdout, prec=prec)))
    label = sys.argv[3]
    if mode == "tuned-proof":
        prec = sys.argv[4] if len(sys.argv) > 4 else "bfloat16"
        if prec not in NARROW_PRECS or len(sys.argv) > 5:
            sys.exit(f"usage: gh1051_cells.py tuned-proof OUT LABEL [{'|'.join(NARROW_PRECS)}]")
        try:
            tuned_references(Path(out), label, prec=prec)
        except ValueError as e:
            sys.exit(log_line(f"INVALID: {display_value(str(e))}"))
        sys.exit(0)
    if mode in ("untuned", "tuned"):
        # One bin_cells run of bench_gpt's step metric.
        if len(sys.argv) != 6 or sys.argv[5] not in NARROW_PRECS:
            sys.exit(f"usage: gh1051_cells.py {mode} RUN LABEL FIXTURE {'|'.join(NARROW_PRECS)}")
        problems = training(out, label, sys.argv[4], sys.argv[5], tuned=mode == "tuned")
        for p in problems:
            print(log_line(f"INVALID: {p}"))
        sys.exit(1 if problems else 0)
    if mode == "ab":
        prec = sys.argv[4] if len(sys.argv) > 4 else "bfloat16"
        if prec not in NARROW_PRECS or len(sys.argv) > 5:
            sys.exit(f"usage: gh1051_cells.py ab OUT LABEL [{'|'.join(NARROW_PRECS)}]")
        problems = ab(out, label, prec)
    else:
        problems = {"split": split}[mode](out, label)
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
