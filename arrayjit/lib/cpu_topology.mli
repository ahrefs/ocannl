(** CPU topology facts for the worker-pool uniformity policy (gh-ocannl-530,
    docs/design/gh-ocannl-530-pool-uniformity.md). Facts only — the policy that consumes them lives
    in [Cc_backend]. Every probe degrades to an "unknown" answer ([[]], [`Unknown], a fallback
    count) rather than raising; the consumers treat unknown conservatively. *)

type core_class = {
  perf_rank : int;
      (** Relative performance of this class among the machine's classes: higher = faster cores.
          Absolute values are platform-native (Windows [EfficiencyClass], Darwin perflevel order
          reversed, list position for Linux) — only the ordering is meaningful. *)
  count : int;  (** Logical CPUs in the class (SMT siblings included). *)
  mask : int64;
      (** Affinity mask over logical CPUs (bit i = CPU i); [0L] when the platform has no affinity
          masks (Darwin) — such classes are informational and cannot be restricted to. *)
}
[@@deriving sexp, compare, equal]

(** {2 Pure parsing, exposed for tests} *)

val parse_cpu_list : string -> int list option
(** Parses a sysfs-style CPU list, e.g. ["0-1,10-13,22-23"]. [None] on malformed input. *)

val mask_of_cpu_list : int list -> int64 option
(** Affinity mask of a CPU list; [None] when empty or any CPU is outside the 64-bit mask (v1 does
    not cross Windows processor groups, and the policy no-ops on such machines). *)

val parse_classes_str : string -> core_class list
(** Parses the stub's class string ["rank:count:maskhex;..."], sorted fastest first. [[]] on any
    malformed item. *)

val classes_of_ranked_cpu_lists : (int * int list) list -> core_class list
(** Builds perf-ranked (fastest-first) classes from (rank, cpu list) pairs; [[]] if any list fails
    to fit a 64-bit mask. *)

(** {2 Platform probes} *)

val core_classes : unit -> core_class list
(** The machine's core classes, perf-ranked fastest first; [[]] when uniform or unknown. Probed once
    and cached. *)

val effective_cpu_count : unit -> int
(** Logical CPUs available to this process — affinity-respecting, unlike
    [Domain.recommended_domain_count] (affinity-blind on Windows, [_SC_NPROCESSORS_ONLN]-based
    elsewhere). Queried fresh on each call: the answer changes when the process restricts itself. *)

val current_affinity_mask : unit -> int64 option
(** The process's current affinity mask over logical CPUs; [None] when the platform has no such
    notion (Darwin), the mask involves CPUs beyond the 64-bit range, or the query fails. Like
    {!effective_cpu_count}, queried fresh — the answer changes when the process restricts itself. *)

val hypervisor_present : unit -> [ `Yes | `No | `Unknown ]
(** Whether the process runs as a hypervisor {e guest} (WSL2, Hyper-V VMs, KVM, ...), where
    guest-visible topology is fabricated and must not drive geometry decisions (gh-ocannl-530). A
    native Windows host with the Hyper-V role enabled reads as [`No] (the hypervisor is the host
    platform itself); [`Unknown] on platforms without a detection path (non-x86 Linux). *)

(** {2 The pool-uniformity decision} *)

type pool_decision = {
  pool_restrict : core_class option;
      (** The class to confine the process to; [None] = leave the pool as found. *)
  pool_width : int;  (** Worker-pool width after the decision, feeding the auto chunk count. *)
  pool_tag : string;
      (** Compact pool signature ([w8P], [w24], ...) for the autotune disk-cache key: crowned
          schedules do not transfer across pools (the gh-530 cross-arm replay), so a policy flip or
          a different external pinning must re-tune, not replay. *)
}
[@@deriving sexp_of]

val unrestricted_decision : effective:int -> affinity_mask:int64 option -> pool_decision
(** The undecided/degraded outcome: leave the pool as found. The tag carries the current affinity
    mask when one is known and narrower than trivial — two same-width external pinnings over
    different cores are different pools (crowns do not transfer between them any more than between
    the policy's classes), so they must not share autotune cache keys. *)

val decide_pool_restriction :
  openmp:bool ->
  setting:[ `All | `Auto | `Performance | `Efficiency ] ->
  classes:core_class list ->
  hypervisor:[ `Yes | `No | `Unknown ] ->
  effective:int ->
  affinity_mask:int64 option ->
  pool_decision
(** The pure pool-uniformity decision (config [cc_pool_core_class]; the applying wrapper lives in
    [Cc_backend], the rationale in docs/design/gh-ocannl-530-pool-uniformity.md). [classes] is
    perf-ranked fastest-first; [effective] the process's affinity-respecting logical CPU count;
    [affinity_mask] its current mask when known (enters the unrestricted tag, see
    {!unrestricted_decision}).

    Explicit [`Performance]/[`Efficiency] override an external pinning (the user asked for the
    class) and skip the hypervisor gate (so e.g. big.LITTLE ARM Linux, where detection is
    [`Unknown], can still be forced); [`Auto] respects the pinning as the user's own pool decision,
    and treats [`Unknown] like [`Yes] — "cannot rule virtualization out" must not restrict by
    default. [`Efficiency] picks the class {e just below} the performance class, not the slowest: on
    three-class parts (P / E / LP-E) the low-power island is a near-serial pool nobody means by
    "efficiency cores". *)

val restrict_process_to_mask : int64 -> (unit, string) result
(** Restricts execution to the CPUs of [mask]. Scope differs by OS, and the difference is
    load-bearing: Windows ([SetProcessAffinityMask]) restricts the whole process including
    already-running threads; Linux ([sched_setaffinity(0)]) restricts the {e calling thread} and
    whatever it spawns afterwards — threads and domains that already exist keep their mask. So the
    call must happen before the threads that will run OpenMP teams are created (the Multidev worker
    domains force it before spawning; the Sync scheduler runs kernels on the calling thread).
    Children of the calling thread inherit on both platforms. *)
