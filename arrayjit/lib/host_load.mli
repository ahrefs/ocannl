(** Host CPU accounting for telling a co-tenant apart from a routine's own variance (gh-ocannl-1558,
    gh-ocannl-1640). The autotuner reads it around a CPU backend's timing window: a window whose
    samples are dispersed is refused as host contention only when this signal shows CPU work that is
    not this process's own. Facts and one pure classification; the policy that consumes them lives
    in [Autotune]. Every probe degrades to [None] rather than raising.

    What it cannot see, by construction: load from this process's own other domains or threads
    (counted as ours), a child process still running at a reading (only reaped children are in
    [own_s] on POSIX; Windows [GetProcessTimes] counts this process's threads only), a co-tenant on
    a device rather than the host CPUs, and a co-tenant using less than {!cotenant_cores} on average
    over the span. *)

type reading = {
  wall_s : float;  (** A monotonic clock, in seconds since an arbitrary origin. *)
  busy_s : float;
      (** Cumulative CPU-seconds every logical CPU of the machine has spent busy (not idle, not
          waiting on I/O), summed over the CPUs. Tick-sampled on Linux, macOS, and Windows. *)
  own_s : float;
      (** Cumulative CPU-seconds of this whole process -- every thread, so the cc backend's worker
          pool -- plus its reaped children (a finished compiler run) on POSIX from [Unix.times], or
          this process's kernel + user time from [GetProcessTimes] on Windows. *)
}
[@@deriving sexp_of]

(** A span's verdict: the foreign load it measured, in core-equivalents, and whether that is a
    co-tenant; or no usable measurement (no counter on this platform, a non-finite reading, an empty
    span, or a counter that went backwards -- macOS's 32-bit tick counters wrap). *)
type verdict = Quiet of float | Cotenant of float | Unavailable [@@deriving sexp_of, equal]

type detail = {
  verdict : verdict;
  span_s : float option;
      (** Wall span between [before] and [after], including any padding wait. [None] when a reading
          is missing or non-finite. *)
  window_s : float option;
      (** Attribution window in seconds: the explicit [~window_s] when padded, otherwise [span_s].
          [None] when a reading is missing or [window_s] is invalid. *)
  busy_s : float option;  (** Machine busy CPU-seconds delta over the span. *)
  own_s : float option;  (** Process CPU-seconds delta over the span. *)
  foreign_cores : float option;
      (** [(busy_s -. own_s) /. window_s] when the inputs were valid, else [None]. *)
  allowance_cores : float option;
      (** [counter_allowance_s /. window_s] when [window_s] is finite and positive, else [None]. *)
  bar_cores : float option;
      (** [cotenant_cores +. allowance_cores], the co-tenant threshold for this window. *)
}
[@@deriving sexp_of]
(** A span's full classification detail, shared by {!classify} and the per-window diagnostic stream
    (config [autotune_log_host_load], gh-ocannl-1640). *)

val supported : bool
(** Whether this host provides a scope-matched machine-wide busy counter (Linux, macOS, or
    single-group Windows where [GetActiveProcessorGroupCount() = 1]; multi-group Windows hosts with
    over 64 logical processors degrade to [false] / [None] rather than subtracting whole-process
    [GetProcessTimes] from single-group [GetSystemTimes]), published so tests gate on platform
    support rather than on {!read}'s own report of success. *)

val read : unit -> reading option
(** The current reading, or [None] where the platform gives no scope-matched machine-wide busy
    counter or it could not be read. *)

val cotenant_cores : float
(** The foreign load, in core-equivalents averaged over the span, at and above which a span shows a
    co-tenant: one whole core. Ordinary background activity on a quiet box is a small fraction of
    one. *)

val counter_allowance_s : float
(** The busy counter's resolution allowance, in CPU-seconds: two ticks of the host's busy-counter
    clock ([2 / sysconf(_SC_CLK_TCK)] on POSIX, [0.02] s at 100 Hz; [2 * increment] from
    [GetSystemTimeAdjustment] on Windows, [0.03125] s at 64 Hz; fallback [0.02] s), for the tick
    each end of a span can be off by. Charged on top of {!cotenant_cores} as
    [counter_allowance_s /. span], so a short span needs more evidence. *)

val min_span_s : float
(** The shortest span a verdict should rest on: below it the tick-sampled busy counter's error is
    comparable to a core. A caller with a shorter window waits out the remainder before its second
    reading. *)

val unavailable_detail : detail
(** The detail record for a window missing an opening or closing reading. *)

val classify_detail : ?window_s:float -> before:reading -> after:reading -> unit -> detail
(** Like {!classify}, returning the intermediate span, attribution window, foreign load, tick
    allowance and co-tenant bar beside the verdict. {!classify} is [(classify_detail ...).verdict].
    Pure. *)

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

val verdict_name : verdict -> string
(** Stable lowercase spelling for diagnostic lines: ["quiet"], ["cotenant"], or ["unavailable"]. *)

val format_detail : detail -> string
(** One space-separated [key=value] record of a {!detail} for diagnostic output: [verdict],
    [foreign], [bar], [allowance], [span_s], [window_s], [busy_s], and [own_s], printing [none] for
    absent or non-finite floats. *)
