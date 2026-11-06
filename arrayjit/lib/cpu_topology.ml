(** CPU topology facts for the worker-pool uniformity policy (gh-ocannl-530,
    docs/design/gh-ocannl-530-pool-uniformity.md). Facts only — the policy that consumes them lives
    in [Cc_backend]. Every probe degrades to an "unknown" answer ([[]], [`Unknown], a fallback
    count) rather than raising; the consumers treat unknown conservatively. *)

open Base

external native_core_classes_str : unit -> string = "ocannl_core_classes_str"
external effective_cpu_count_stub : unit -> int = "ocannl_effective_cpu_count"
external affinity_mask_stub : unit -> int64 = "ocannl_affinity_mask"
external hypervisor_present_stub : unit -> int = "ocannl_hypervisor_present"
external set_process_affinity_stub : int64 -> int = "ocannl_set_process_affinity"

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

(* --- Pure parsing, exposed for tests. --- *)

(** Parses a sysfs-style CPU list, e.g. ["0-1,10-13,22-23"]. [None] on malformed input. *)
let parse_cpu_list s =
  let parse_range part =
    match String.lsplit2 part ~on:'-' with
    | None -> Option.map (Int.of_string_opt (String.strip part)) ~f:(fun c -> [ c ])
    | Some (lo, hi) -> (
        match (Int.of_string_opt (String.strip lo), Int.of_string_opt (String.strip hi)) with
        | Some lo, Some hi when lo <= hi -> Some (List.range lo (hi + 1))
        | _ -> None)
  in
  let s = String.strip s in
  if String.is_empty s then None
  else String.split s ~on:',' |> List.map ~f:parse_range |> Option.all |> Option.map ~f:List.concat

(** Affinity mask of a CPU list; [None] when empty or any CPU is outside the 64-bit mask (v1 does
    not cross Windows processor groups, and the policy no-ops on such machines). *)
let mask_of_cpu_list cpus =
  if List.is_empty cpus then None
  else if List.exists cpus ~f:(fun c -> c < 0 || c >= 64) then None
  else Some (List.fold cpus ~init:0L ~f:(fun acc c -> Int64.(acc lor shift_left 1L c)))

(* [decide_pool_restriction] picks classes positionally (fastest first), so the one ordering
   convention lives here and both parsers share it. *)
let sort_fastest_first classes =
  List.sort classes ~compare:(fun a b -> Int.compare b.perf_rank a.perf_rank)

(** Parses the stub's class string ["rank:count:maskhex;..."]. [[]] on any malformed item. *)
let parse_classes_str s =
  let parse_item item =
    match String.split item ~on:':' with
    | [ rank; count; mask ] -> (
        match
          (Int.of_string_opt rank, Int.of_string_opt count, Int64.of_string_opt ("0x" ^ mask))
        with
        | Some perf_rank, Some count, Some mask when count > 0 -> Some { perf_rank; count; mask }
        | _ -> None)
    | _ -> None
  in
  if String.is_empty s then []
  else
    match String.split s ~on:';' |> List.map ~f:parse_item |> Option.all with
    | None -> []
    | Some classes -> sort_fastest_first classes

(** Builds perf-ranked classes from (rank, cpu list) pairs; [[]] if any list fails to fit a 64-bit
    mask. *)
let classes_of_ranked_cpu_lists ranked =
  ranked
  |> List.map ~f:(fun (perf_rank, cpus) ->
      Option.map (mask_of_cpu_list cpus) ~f:(fun mask ->
          { perf_rank; count = List.length cpus; mask }))
  |> Option.all
  |> Option.value_map ~default:[] ~f:sort_fastest_first

(* --- Platform probes. --- *)

let read_sys_file path = try Some (String.strip (Stdio.In_channel.read_all path)) with _ -> None

(* Intel hybrid exposes the two classes directly; ARM big.LITTLE (and DynamIQ) exposes per-CPU
   relative capacity instead, so group CPUs by capacity and rank the groups by it. Any missing or
   unparsable piece yields [[]] (read as: uniform or unknown). *)
let linux_classes () =
  match
    (read_sys_file "/sys/devices/cpu_core/cpus", read_sys_file "/sys/devices/cpu_atom/cpus")
  with
  | Some core, Some atom -> (
      match (parse_cpu_list core, parse_cpu_list atom) with
      | Some core, Some atom -> classes_of_ranked_cpu_lists [ (1, core); (0, atom) ]
      | _ -> [])
  | _ when not (Stdlib.Sys.file_exists "/sys/devices/system/cpu/cpu0/cpu_capacity") ->
      (* The common uniform-x86 case: skip the per-CPU walk below (one failed open per CPU on a big
         server) when the first capacity file is already absent. *)
      []
  | _ -> (
      match Option.bind (read_sys_file "/sys/devices/system/cpu/present") ~f:parse_cpu_list with
      | None -> []
      | Some cpus -> (
          let capacities =
            List.map cpus ~f:(fun c ->
                Option.bind
                  (read_sys_file (Printf.sprintf "/sys/devices/system/cpu/cpu%d/cpu_capacity" c))
                  ~f:Int.of_string_opt
                |> Option.map ~f:(fun cap -> (cap, c)))
            |> Option.all
          in
          match capacities with
          | None -> []
          | Some pairs ->
              let groups =
                Hashtbl.of_alist_multi (module Int) pairs
                |> Hashtbl.to_alist
                |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
              in
              if List.length groups < 2 then []
              else classes_of_ranked_cpu_lists (List.mapi groups ~f:(fun i (_, cpus) -> (i, cpus))))
      )

let core_classes =
  let classes =
    lazy
      (let s = native_core_classes_str () in
       if not (String.is_empty s) then parse_classes_str s
       else if Stdlib.Sys.file_exists "/sys/devices/system/cpu" then linux_classes ()
       else [])
  in
  fun () -> Lazy.force classes

let total_logical_count classes = List.sum (module Int) classes ~f:(fun c -> c.count)

(** Logical CPUs available to this process — affinity-respecting, unlike
    [Domain.recommended_domain_count] (affinity-blind on Windows, [_SC_NPROCESSORS_ONLN]-based
    elsewhere). Queried fresh on each call: the answer changes when the process restricts itself. *)
let effective_cpu_count () =
  let n = effective_cpu_count_stub () in
  if n > 0 then n else max 1 (Stdlib.Domain.recommended_domain_count ())

(** The process's current affinity mask over logical CPUs; [None] when the platform has no such
    notion (Darwin), the mask involves CPUs beyond the 64-bit range, or the query fails. Like
    {!effective_cpu_count}, queried fresh — the answer changes when the process restricts itself. *)
let current_affinity_mask () = match affinity_mask_stub () with 0L -> None | m -> Some m

(** Whether the process runs as a hypervisor {e guest} (WSL2, Hyper-V VMs, KVM, ...), where
    guest-visible topology is fabricated and must not drive geometry decisions (gh-ocannl-530). A
    native Windows host with the Hyper-V role (or WSL2/VBS) enabled sets the CPUID hypervisor bit
    yet sees real topology, and reads as [`No] here: on Windows a "Microsoft Hv" hypervisor is the
    host platform itself, while Hyper-V guests see fabricated {e uniform} topology, making the
    core-classes check the operative gate for them (see the stub for why the root-partition
    privilege check does not work). [`Unknown] on platforms without a detection path (non-x86
    Linux). *)
let hypervisor_present () =
  match hypervisor_present_stub () with 1 -> `Yes | 0 -> `No | _ -> `Unknown

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

(** The undecided/degraded outcome: leave the pool as found. The tag carries the current affinity
    mask when one is known and narrower than trivial — two same-width external pinnings over
    different cores are different pools (crowns do not transfer between them any more than between
    the policy's classes), so they must not share autotune cache keys. *)
let unrestricted_decision ~effective ~affinity_mask =
  let pool_tag =
    match affinity_mask with
    | Some m when Int64.popcount m = effective && effective < 64 ->
        Printf.sprintf "w%dx%Lx" effective m
    | _ -> "w" ^ Int.to_string effective
  in
  { pool_restrict = None; pool_width = effective; pool_tag }

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
let decide_pool_restriction ~openmp ~setting ~classes ~hypervisor ~effective ~affinity_mask =
  let keep = unrestricted_decision ~effective ~affinity_mask in
  let restrict_to (cls : core_class) suffix =
    {
      pool_restrict = Some cls;
      pool_width = cls.count;
      pool_tag = "w" ^ Int.to_string cls.count ^ suffix;
    }
  in
  let hybrid_restrictable =
    List.length classes >= 2
    && List.for_all classes ~f:(fun (c : core_class) -> Int64.(c.mask <> 0L))
  in
  let native = match hypervisor with `No -> true | `Yes | `Unknown -> false in
  let externally_pinned = effective < total_logical_count classes in
  if not (openmp && hybrid_restrictable) then keep
  else
    match setting with
    | `All -> keep
    | `Auto when (not native) || externally_pinned -> keep
    | `Auto | `Performance -> restrict_to (List.hd_exn classes) "P"
    | `Efficiency -> restrict_to (List.nth_exn classes 1) "E"

(** Restricts execution to the CPUs of [mask]. Scope differs by OS, and the difference is
    load-bearing: Windows ([SetProcessAffinityMask]) restricts the whole process including
    already-running threads; Linux ([sched_setaffinity(0)]) restricts the {e calling thread} and
    whatever it spawns afterwards — threads and domains that already exist keep their mask. So the
    call must happen before the threads that will run OpenMP teams are created (the Multidev worker
    domains force it before spawning; the Sync scheduler runs kernels on the calling thread).
    Children of the calling thread inherit on both platforms. *)
let restrict_process_to_mask mask =
  match set_process_affinity_stub mask with
  | 0 -> Ok ()
  | -1 -> Error "process affinity is not supported on this platform"
  | _ -> Error (Printf.sprintf "setting the process affinity mask 0x%Lx failed" mask)
