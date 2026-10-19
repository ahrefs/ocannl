#!/usr/bin/env python3
"""Bucket a profiled gpt2_mini step by kernel class, following the gh-ocannl-531 method.

Joins two artifacts:

  1. the emitted batch source for the fissioned step -- one translation unit holding every
     ``<routine>__seg<N>`` kernel (``build_files/bench_gpt/*_seg.{cu,hip,metal}``), which is
     where a segment's *identity* lives: its kernel parameters carry the tensor nodes' debug
     names, and the named model weights (``w_q_l0``, ``l0_ffn_w1``, ``wte``, ``gamma_ln1_l0``,
     ...) say what the segment computes;
  2. a per-kernel time table from the platform profiler -- ``rocprofv3 --kernel-trace --stats``
     on HIP, ``nsys stats --report cuda_gpu_kern_sum`` on CUDA -- which says what it cost.

Segment numbering is compile-specific, so the mapping must be derived from the *same* compile
that was profiled; never carry a seg->bucket table between runs.

Buckets are gh-531's four, so the shares are directly comparable to
``benchmarks/report-gh531-profile.md``:

  ffn          FFN GEMMs (up-projection + gelu epilogue, down-projection, their zero-inits)
  attention    q/k/v projections, QK^T, the softmax chain, scores.V, the output projection
  emb_logits   token/positional embedding, the tied lm_head, and the softmax-CE head
  layernorm    LayerNorm chains, residual adds and the remaining elementwise passes

Classification is two-stage and deliberately conservative. A segment naming a model weight is
*seeded* directly from it; every other segment (the pure-intermediate ones -- QK^T, the softmax
chain, zero-inits) inherits, to a fixpoint, from the segments it shares tensor nodes with. A
segment that never resolves is reported as ``other`` rather than guessed at, and the summary
prints the unresolved list so an unclassified tail can never hide inside a bucket share.

Usage:
    gpt2_bucket.py --source build_files/bench_gpt/cross_entropy_loss_fwd__seg.hip \\
                   --stats  prof/out_kernel_stats.csv [--steps 53] [--dump]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_safe import code_span, display_value, log_line, table_lines  # noqa: E402

# --- what a named tensor node says about its segment --------------------------------------
# Order matters: these are tried in sequence and the first match wins, because a segment
# legitimately names weights from more than one class.
#
# LayerNorm goes FIRST, which is not the obvious order and is worth the paragraph. A gain/bias
# pair is DEFINITIONAL: only a LayerNorm computes with gamma/beta, and no GEMM names them --
# the FFN up-projection and the lm_head read the layer-norm OUTPUT node (`n309_layer_norm`,
# `n794_layer_norm`), never the gains. The reverse is not true. Because the residual stream is
# Virtual, every consumer re-derives it by re-summing all prior contributions (gh-531 bin 2), so
# each LayerNorm's signature also carries `wpe` and the FFN output biases of every LAYER BELOW
# it -- l0 for the layer-1 norms, l0+l1 for layer 2, all four for the final norm. That triangle
# is the re-summation, not an FFN GEMM, and seeding on `l*_ffn_b2` first would file all nine
# LayerNorm sites under FFN and empty the elementwise bucket entirely.
SEED_RULES = [
    ("layernorm", re.compile(r"^(gamma|beta)_")),
    ("ffn", re.compile(r"^l\d+_ffn_(w|b)[12]$")),
    ("attention", re.compile(r"^w_[qkvo]_l\d+$|^mask$")),
    (
        "emb_logits",
        re.compile(
            r"^(wte|wpe|ids|embedded|logits|max_logits|log_probs|neg_nll|shifted|tgt"
            r"|stop_grad|cross_entropy|cross_entropy_loss)$"
        ),
    ),
]

BUCKET_ORDER = ["ffn", "attention", "emb_logits", "layernorm", "other"]
BUCKET_LABEL = {
    "ffn": "FFN GEMMs",
    "attention": "attention",
    "emb_logits": "embedding / logits",
    "layernorm": "layernorm / elementwise",
    "other": "other",
}
# Inheritance precedence for the propagation stage. Deliberately NOT the seeding order: seeding
# reads a definitional weight, while propagation is a guess about a kernel with no weight at all
# (a zero-init, a softmax step). For those the GEMM classes are the informative answer, so
# layernorm sinks to last -- an unnamed kernel adjacent to both an FFN GEMM and a norm belongs to
# the FFN chain, not to the elementwise tail.
INHERIT_PRIORITY = {"ffn": 0, "attention": 1, "emb_logits": 2, "layernorm": 3}

KERNEL_RE = re.compile(r'^\s*(?:extern\s+"C"\s+)?__(?:global|kernel)__\s+\w+\s+(\w+__seg(\d+))\s*\(')
PARAM_RE = re.compile(r"(?:__restrict__|\*)\s*([A-Za-z_]\w*)\s*(?:,|\))")


class Seg:
    __slots__ = ("name", "params", "gist")

    def __init__(self, name: str, params: set[str], gist: str):
        self.name = name
        self.params = params
        self.gist = gist


def parse_source(path: Path) -> dict[str, Seg]:
    """seg kernel name -> its tensor-node parameters, plus a one-line gist of what it does.

    Scalar bound-symbol parameters (``const int i1``) carry no node identity and are skipped:
    PARAM_RE only matches a name preceded by a pointer star or ``__restrict__``. The gist is the
    kernel's first store or its ``tile_mma`` marker -- enough to check a row against the source by
    eye, which is how the dominant kernels' identity is actually established here (the classifier
    settles the long cheap tail, not the claim-bearing kernels).
    """
    lines = path.read_text().splitlines()
    out: dict[str, Seg] = {}
    starts = [(i, m.group(1)) for i, ln in enumerate(lines) if (m := KERNEL_RE.match(ln))]
    for k, (idx, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        sig_end = idx
        for j in range(idx, min(idx + 64, end)):
            sig_end = j
            if re.search(r"\)\s*\{\s*$", lines[j]):
                break
        params = set(PARAM_RE.findall("\n".join(lines[idx : sig_end + 1])))
        gist = ""
        for ln in lines[sig_end + 1 : end]:
            s = ln.strip()
            if s.startswith("{ /* tile_mma"):
                gist = s.lstrip("{ ").rstrip()
                break
            m = re.match(r"(\w+)\s*\[", s)
            if m and m.group(1) in params and not gist:
                gist = " ".join(s.split())
        out[name] = Seg(name, params, gist)
    if not out:
        sys.exit(f"{path}: no __global__ ...__segN kernels found -- wrong file?")
    return out


def classify(segs: dict[str, Seg]) -> tuple[dict[str, str], set[str], list[str]]:
    """Seed from named model weights, then propagate through shared tensor nodes to a fixpoint.

    Returns the assignment, the set of DIRECTLY SEEDED kernels, and the unresolved ones. The seeded
    set is the point: every GEMM in this model reads a named weight, so it is seeded outright, and
    only the cheap pure-intermediate tail (softmax chain, zero-inits, residual adds) reaches the
    propagation step. Reporting the time split between the two says exactly how much of a bucket
    share rests on the heuristic rather than on a name in a kernel signature.
    """
    bucket: dict[str, str] = {}
    for name, s in segs.items():
        for b, rule in SEED_RULES:
            if any(rule.match(p) for p in s.params):
                bucket[name] = b
                break
    seeded = set(bucket)

    node_segs: dict[str, list[str]] = defaultdict(list)
    for name, s in segs.items():
        for p in s.params:
            node_segs[p].append(name)

    changed = True
    while changed:
        changed = False
        for name in sorted(segs, key=seg_index):   # deterministic, independent of source order
            if name in bucket:
                continue
            votes = {
                bucket[o] for p in segs[name].params for o in node_segs[p]
                if o != name and o in bucket
            }
            if votes:
                bucket[name] = min(votes, key=lambda b: INHERIT_PRIORITY[b])
                changed = True

    unresolved = sorted(set(segs) - set(bucket), key=seg_index)
    for name in unresolved:
        bucket[name] = "other"
    return bucket, seeded, unresolved


def seg_index(name: str) -> int:
    m = re.search(r"__seg(\d+)$", name)
    return int(m.group(1)) if m else -1


def parse_stats(path: Path) -> dict[str, tuple[int, float]]:
    """kernel name -> (dispatch count, total ns). Accepts rocprofv3 and nsys CSV headers."""
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path}: empty stats file")
    cols = {c.lower().strip(): c for c in rows[0]}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    name_c = pick("name", "kernel_name", '"name"')
    calls_c = pick("calls", "count", "instances", "n")
    total_c = pick("totaldurationns", "total_duration", "total time (ns)", "totaltime", "duration")
    if not (name_c and total_c):
        sys.exit(f"{path}: cannot find name/total-duration columns in {list(rows[0])}")
    out: dict[str, tuple[int, float]] = {}
    for r in rows:
        nm = r[name_c].strip().strip('"')
        cnt = int(float(r[calls_c])) if calls_c and r.get(calls_c) else 0
        out[nm] = (cnt, float(r[total_c]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="emitted batch source (.hip/.cu)")
    ap.add_argument("--stats", required=True, type=Path, help="profiler per-kernel CSV")
    ap.add_argument("--steps", type=int, default=None,
                    help="dispatches per kernel per step (default: the modal dispatch count), "
                         "used to turn totals into per-step numbers")
    ap.add_argument("--dump", action="store_true", help="print the per-segment assignment")
    args = ap.parse_args()

    segs = parse_source(args.source)
    bucket, seeded, unresolved = classify(segs)
    stats = parse_stats(args.stats)

    matched = {k: v for k, v in stats.items() if k in bucket}
    if not matched:
        sys.exit(
            "no profiled kernel name matches a segment in the source. Profiled names:\n  "
            + "\n  ".join(sorted(stats)[:10])
        )
    unprofiled = sorted(set(bucket) - set(matched), key=seg_index)
    unsourced = sorted(set(stats) - set(bucket))

    counts = [c for c, _ in matched.values() if c]
    steps = args.steps or (max(set(counts), key=counts.count) if counts else 1)

    agg_ns: dict[str, float] = defaultdict(float)
    agg_n: dict[str, int] = defaultdict(int)
    for k, (_, ns) in matched.items():
        agg_ns[bucket[k]] += ns
        agg_n[bucket[k]] += 1
    total = sum(agg_ns.values())

    print(log_line(f"source : {display_value(str(args.source))}"))
    print(log_line(f"stats  : {display_value(str(args.stats))}"))
    print(f"kernels: {len(matched)} profiled / {len(bucket)} emitted;  dispatches per kernel: {steps}")
    print(f"total GPU kernel time: {total / 1e6:.3f} ms over {steps} steps "
          f"= {total / 1e6 / steps:.3f} ms/step\n")
    rows = [(BUCKET_LABEL[b], str(agg_n[b]), f"{agg_ns[b] / 1e6 / steps:.3f}",
             f"{100 * agg_ns[b] / total:.1f}%") for b in BUCKET_ORDER if agg_n[b]]
    rows.append(("**total**", str(len(matched)), f"{total / 1e6 / steps:.3f}", "100.0%"))
    print("\n".join(table_lines(["bucket", "kernels", "ms/step", "share"], rows, align="lrrr")))

    # How much of the above rests on a name in a kernel signature rather than on the propagation
    # heuristic. If the seeded share is high the bucket table is essentially read off the source.
    seeded_ns = sum(ns for k, (_, ns) in matched.items() if k in seeded)
    print(f"\ndirectly seeded by a named model weight: "
          f"{len(seeded & set(matched))}/{len(matched)} kernels, "
          f"{100 * seeded_ns / total:.1f}% of kernel time; "
          f"the remaining {100 * (total - seeded_ns) / total:.1f}% is assigned by propagation.")

    if unresolved:
        print(f"\nUNRESOLVED (reported as 'other', not folded into a bucket): {len(unresolved)}")
        for s in unresolved:
            print(log_line(f"  {display_value(s)}: {sorted(segs[s].params)}"))
    if unprofiled:
        print("\n" + log_line(f"emitted but not profiled ({len(unprofiled)}): "
                             f"{', '.join(map(display_value, unprofiled))}"))
    if unsourced:
        print("\n" + log_line(f"profiled but not in source ({len(unsourced)}): "
                             f"{', '.join(map(display_value, unsourced[:12]))}"))

    if args.dump:
        # Params and the kernel's gist, so every row can be checked against the source by eye --
        # which is how the dominant kernels' identity is established, not by the classifier.
        # The gist is a line of kernel source (`||` included), so it is shown, not pasted.
        rows = [(display_value(k), bucket[k], "yes" if k in seeded else "prop",
                 f"{matched[k][1] / 1e6 / steps:.4f}",
                 " ".join(map(display_value, sorted(segs[k].params))), code_span(segs[k].gist[:110]))
                for k in sorted(matched, key=seg_index)]
        print()
        print("\n".join(table_lines(["seg", "bucket", "seeded", "ms/step", "params", "gist"], rows,
                                    align="lllrll")))


if __name__ == "__main__":
    main()
