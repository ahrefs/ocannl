"""Shared helpers for the Python benchmark runners."""

import json
import math
import struct


def read_st_metadata(path):
    """Read the __metadata__ string map from a safetensors file header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header.get("__metadata__", {})


def percentiles(xs):
    s = sorted(xs)

    def p(q):
        return s[round(q / 100 * (len(s) - 1))]

    return {"p10": p(10), "p50": p(50), "p90": p(90)}


def json_safe(obj):
    """`obj` with every non-finite float replaced by None, recursively.

    A diverged training run is exactly the run whose loss trajectory the report needs, and
    `json.dumps` writes it as `NaN` / `Infinity` -- tokens JSON does not have. Python's own loader
    accepts them, so the sweep survives, but `results.jsonl` then holds a line no other JSON reader
    will take, and the OCANNL runner emitting the same fact as `nan` had its whole cell dropped
    (gh-ocannl-676). `null` is what all three runners emit for a number they have and JSON cannot
    express; `orchestrate.py` reads it as "ran, and this number is not a number".
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def emit(result):
    # allow_nan=False so that a non-finite value json_safe did not reach -- one behind a type it
    # does not walk -- raises here rather than being written as an unparseable token.
    print(json.dumps(json_safe(result), allow_nan=False), flush=True)


# --- Search provenance (gh-ocannl-644) ---------------------------------------------------
#
# Whether THIS process ran its own kernel search / codegen or replayed a cache. OCANNL's tuned
# cell splits the two into separate processes, because a searching process is measurably slower
# per launch on small kernels; tinygrad's BEAM cell and torch.compile search in the process that
# then times steps. What that costs THEM is measured and is per box (gh-ocannl-675): beam
# +6.4% on CUDA and +14.9% on ROCm, torch.compile -12.0% and +7.2% -- so they stay single pass.
# Every runner SAYS which it did, which is what made that measurable at all.
#
# Both probes read framework internals, so they answer None ("cannot tell", reported as UNKNOWN)
# rather than guess. A wrong False is exactly the silent claim the field exists to prevent, so
# a probe that saw nothing at all when it should have seen something says so.


def instrument_tinygrad_beam():
    """Start counting tinygrad's beam searches and their disk-cache reads and writes.

    Returns the counts dict (live, read it after the steps) or None if this tinygrad cannot be
    instrumented. Call it before the first step: the search happens on the first kernel launch,
    and `tinygrad.engine.search` binds the cache helpers by from-import, so the wrappers have to
    go on *that* module's attributes rather than on `tinygrad.helpers`.

    Calls are counted as well as cache traffic because the cache is not the only regime: under
    CACHELEVEL=0 or IGNORE_BEAM_CACHE a search runs and neither reads nor writes, which cache
    counting alone reports as "cannot tell".
    """
    # tinygrad moved the beam search out of `tinygrad.engine.search` into
    # `tinygrad.codegen.opt.search` (0.13); try both, newest first, so the probe answers on
    # either. A layout it does not know still answers None rather than guessing.
    search = None
    for mod in ("tinygrad.codegen.opt.search", "tinygrad.engine.search"):
        try:
            search = __import__(mod, fromlist=["_"])
        except Exception:
            continue
        if hasattr(search, "diskcache_get") and hasattr(search, "diskcache_put"):
            break
        search = None
    if search is None:
        return None
    counts = {"call": 0, "hit": 0, "put": 0}
    get, put = search.diskcache_get, search.diskcache_put

    def get_counting(table, key, *args, **kwargs):
        val = get(table, key, *args, **kwargs)
        if table == "beam_search" and val is not None:
            counts["hit"] += 1
        return val

    def put_counting(table, key, val, *args, **kwargs):
        if table == "beam_search":
            counts["put"] += 1
        return put(table, key, val, *args, **kwargs)

    search.diskcache_get, search.diskcache_put = get_counting, put_counting

    beam = getattr(search, "beam_search", None)
    if callable(beam):

        def beam_counting(*args, **kwargs):
            counts["call"] += 1
            return beam(*args, **kwargs)

        search.beam_search = beam_counting
    return counts


def tinygrad_searched(counts, beam):
    """The `searched` field for a tinygrad cell: did this process run a beam search?

    Three outcomes, not two. A cache write means it searched, and so does a search call that read
    no cached entry (the uncached regimes above). Calls that all came back from
    `~/.cache/tinygrad`, or reads with nothing else observed, mean every beam result was replayed.
    Beam requested and nothing at all observed means the internals moved under the probe: that is
    UNKNOWN, never False.
    """
    if not beam:
        return False
    if not counts:
        return None
    if counts["put"] or counts["call"] > counts["hit"]:
        return True
    if counts["hit"] or counts["call"]:
        return False
    return None


# Inductor's FX-graph cache outcomes, by what they say about THIS process. A bypass is a graph
# the cache refused to serve, so it was generated here exactly as a miss was; reading only
# hit-vs-miss calls a run with one hit and one bypass a replay, which is a compiled graph this
# process paid for and the report would not show.
TORCH_CODEGEN_COUNTERS = ("fxgraph_cache_miss", "fxgraph_cache_bypass")
TORCH_REPLAY_COUNTERS = ("fxgraph_cache_hit",)


def torch_searched(torch, compiled):
    """The `searched` field for a pytorch cell: did this process run inductor's codegen?

    An eager cell compiles nothing. For a compiled one the FX-graph cache counters say whether any
    graph was generated here or all of them came from `~/.cache/torch/inductor`; a torch that
    reports none of them cannot answer. A run is mixed as soon as it has more than one graph, so
    the question is whether ANY graph was generated here, not whether the last one was.
    """
    if not compiled:
        return False
    try:
        counters = torch._dynamo.utils.counters["inductor"]
        codegen = sum(counters.get(name, 0) for name in TORCH_CODEGEN_COUNTERS)
        replayed = sum(counters.get(name, 0) for name in TORCH_REPLAY_COUNTERS)
    except Exception:
        return None
    if codegen:
        return True
    return False if replayed else None


# --- Peak device memory (gh-ocannl-1006) --------------------------------------------------
#
# The report's memory column is what a footprint-for-time trade (gh-ocannl-616) is read against,
# so a cell reports the peak over its TIMED STEPS and not over the process: a tuned cell's search
# allocates a candidate buffer per arm, and a counter read at exit reports the search's high water
# rather than the workload's.
#
# The counters the three frameworks expose are not one quantity, and the difference does not
# follow the framework column, so each cell NAMES the counter it read. Two kinds:
#
#   high-water -- the allocator maintains the maximum itself (`torch.cuda.max_memory_allocated`,
#     OCANNL's own allocator seam). Exact over the window, and the two are the same quantity
#     (requested bytes off an allocator), so those rows compare honestly with each other.
#   sampled -- the framework offers only a CURRENT gauge (`torch.mps.driver_allocated_memory`,
#     tinygrad's `GlobalCounters.mem_used`), so the runner takes the maximum over the step
#     boundaries. That is a lower bound on the window: an allocation made and given back inside one
#     step is invisible to it.
#
# A framework with neither gets None, which reaches the report as a dash. There is no host-RSS
# substitute: RSS is not the device footprint and a column mixing the two would be read as if it
# were one number.


class PeakMemoryProbe:
    """One framework's peak-device-memory counter over a bracketed window.

    `start()` opens the window, `sample()` is called at every step boundary (a no-op for a
    high-water counter, which needs no help), and `read()` closes it with the peak in bytes.
    """

    def __init__(self, source, read, reset=None, sampled=False):
        self.source = source
        self._read = read
        self._reset = reset
        self._sampled = sampled
        self._max = 0

    def start(self):
        self._max = 0
        if self._reset is not None:
            self._reset()
        # The gauge's reading at the window's start is part of the window: the workload's weights
        # are already resident, and a peak that began at zero would report only what the timed
        # steps went on to allocate on top of them.
        self.sample()

    def sample(self):
        if self._sampled:
            self._max = max(self._max, self._read())

    def read(self):
        return self._max if self._sampled else self._read()


def torch_peak_memory(torch, device):
    """The peak-device-memory probe for a pytorch cell, or None where torch has no counter.

    CPU is the None case on purpose: torch allocates host memory through the system allocator and
    exposes no per-process figure for it, and `psutil`-style RSS is a different quantity.
    """
    if device == "cuda":
        # The same quantity as OCANNL's: bytes requested off the caching allocator, maximum over
        # the window. Not `max_memory_reserved`, which is the pool torch grew and would flatter a
        # workload whose footprint shrank.
        return PeakMemoryProbe(
            source="torch.cuda.max_memory_allocated (requested bytes, high-water)",
            read=torch.cuda.max_memory_allocated,
            reset=torch.cuda.reset_peak_memory_stats,
        )
    if device == "mps":
        read = getattr(getattr(torch, "mps", None), "driver_allocated_memory", None)
        if read is None:
            return None
        return PeakMemoryProbe(
            source="torch.mps.driver_allocated_memory (driver bytes, sampled at step boundaries)",
            read=read,
            sampled=True,
        )
    return None


def tinygrad_peak_memory():
    """The peak-device-memory probe for a tinygrad cell, or None if this tinygrad has no counter."""
    try:
        from tinygrad.helpers import GlobalCounters
    except Exception:
        return None
    if not hasattr(GlobalCounters, "mem_used"):
        return None
    return PeakMemoryProbe(
        source="tinygrad GlobalCounters.mem_used (requested bytes, sampled at step boundaries)",
        read=lambda: int(GlobalCounters.mem_used),
        sampled=True,
    )


def peak_memory_fields(probe):
    """The two result-line keys for `probe`, both None where there was no counter to read.

    Together, never one without the other: a byte count whose counter is not named would be
    compared with a different quantity in the next row, and a named counter with no bytes says
    nothing. A cell that measured none reports null for both and the report prints a dash -- never
    a zero, which would read as a workload with no footprint.
    """
    if probe is None:
        return {"peak_memory_bytes": None, "peak_memory_source": None}
    return {"peak_memory_bytes": probe.read(), "peak_memory_source": probe.source}
