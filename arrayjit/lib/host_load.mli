(** Host CPU accounting for telling a co-tenant apart from a routine's own variance
    (gh-ocannl-1558). The autotuner reads it around a CPU backend's timing window: a window whose
    samples are dispersed is refused as host contention only when this signal shows CPU work that is
    not this process's own. Facts and one pure classification; the policy that consumes them lives
    in [Autotune]. Every probe degrades to [None] rather than raising.

    What it cannot see, by construction: load from this process's own other domains or threads
    (counted as ours), a child process still running at a reading (only reaped children are in
    [own_s]), a co-tenant on a device rather than the host CPUs, and a co-tenant using less than
    {!cotenant_cores} on average over the span. *)

type reading = {
  wall_s : float;  (** A monotonic clock, in seconds since an arbitrary origin. *)
  busy_s : float;
      (** Cumulative CPU-seconds every logical CPU of the machine has spent busy (not idle, not
          waiting on I/O), summed over the CPUs. Tick-sampled on Linux and macOS. *)
  own_s : float;
      (** Cumulative CPU-seconds of this whole process -- every thread, so the cc backend's worker
          pool -- plus its reaped children (a finished compiler run), from [Unix.times]. *)
}
[@@deriving sexp_of]

(** A span's verdict: the foreign load it measured, in core-equivalents, and whether that is a
    co-tenant; or no usable measurement (no counter on this platform, a non-finite reading, an empty
    span, or a counter that went backwards -- macOS's 32-bit tick counters wrap). *)
type verdict = Quiet of float | Cotenant of float | Unavailable [@@deriving sexp_of, equal]

val read : unit -> reading option
(** The current reading, or [None] where the platform gives no machine-wide busy counter (Windows)
    or it could not be read. *)

val cotenant_cores : float
(** The foreign load, in core-equivalents averaged over the span, at and above which a span shows a
    co-tenant: one whole core. Ordinary background activity on a quiet box is a small fraction of
    one. *)

val counter_allowance_s : float
(** The busy counter's resolution allowance, in CPU-seconds: two ticks of the 100 Hz counters both
    providers expose, for the tick each end of a span can be off by. Charged on top of
    {!cotenant_cores} as [counter_allowance_s /. span], so a short span needs more evidence. *)

val min_span_s : float
(** The shortest span a verdict should rest on: below it the tick-sampled busy counter's error is
    comparable to a core. A caller with a shorter window waits out the remainder before its second
    reading. *)

val classify : ?window_s:float -> before:reading -> after:reading -> unit -> verdict
(** The foreign load over [before .. after], attributed to a window of [window_s] seconds (default
    the whole span): [(busy - own) / window_s] over the deltas. [Cotenant] when it reaches
    [cotenant_cores +. counter_allowance_s /. window_s], [Quiet] below. Negative foreign load (the
    tick-sampled busy counter trailing the precise own time) is [Quiet]. Pure.

    [window_s] shorter than the span is the WORST-CASE attribution a padded span takes: a caller
    that extended a short window's span to {!min_span_s} charges every foreign CPU-second the span
    saw to the window it was judging, so the padding can add evidence but never dilute a co-tenant
    that ran only during the window. A [window_s] that is not finite, not positive, or longer than
    the span is [Unavailable]. *)
