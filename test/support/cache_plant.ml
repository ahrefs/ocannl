(** Planting schedule-cache entries by hand, and waiving the claims a cache operation rests on where
    the filesystem refused it.

    A test that needs an entry the tuner no longer writes (a corrupt or stale winner, a fissioned
    winner built from a chosen segmentation, a synthetic entry for a pure-filesystem exercise)
    builds it here rather than spelling out the {!Ir.Schedule_cache.entry} record, so a field added
    to the entry reaches every planted one through these defaults. Every entry built here records no
    default-pipeline measurement, tensorized best, default fingerprint or search trajectory: the
    shape of an entry written before those fields existed, which replays as "nothing is known".

    The backend name is an explicit argument wherever a key or a codegen tag is derived (as
    {!Ir.Schedule_cache.codegen_tag} takes it), never read off the limits record. *)

open Base
module SC = Ir.Schedule_cache

(** [entry ~backend ~source_digest ()] is an entry of the current {!SC.entry_version} under the
    current numerics policy, with no codegen or objective self-description (what a synthetic entry
    that no key derivation checks needs), an empty schedule, no segments, and zero times, each
    overridable. *)
let entry ?(numerics = SC.numerics_tag ()) ?codegen ?objective ?(saved = []) ?segments
    ?(best_ms = 0.) ?(baseline_ms = 0.) ?mma_best_ms ~backend ~source_digest () : SC.entry =
  {
    version = SC.entry_version;
    backend;
    numerics;
    codegen;
    objective;
    source_digest;
    saved;
    segments;
    best_ms;
    baseline_ms;
    default_ms = None;
    mma_best_ms;
    default_fingerprint = None;
    best_steps = None;
  }

(** [key ~backend ~timing_identity ~limits ~capabilities canon] is the key [Autotune.tune] looks
    [canon]'s routine up under, from the compiling context's timing identity, limits and codegen
    capabilities; [None] without a concrete timing identity. *)
let key ~backend ~timing_identity ~limits ~capabilities canon =
  SC.cache_key ~timing_identity ~limits ~capabilities canon ~backend

(** [plant ~cache_dir ~backend ~timing_identity ~limits ~capabilities canon] stores, under {!key},
    an entry for [canon]'s routine self-described the way [Autotune.tune] describes one it stores
    (the codegen tag and timing objective of this process, [canon]'s digest), carrying [saved] and
    [segments]. Returns the store's cache I/O ({!SC.recording_cache_io}): what a claim resting on
    the planted entry is waived by ({!unless_refused}). Without a concrete timing identity the key
    is [None], nothing is stored, and the record is empty. *)
let plant ?saved ?segments ?best_ms ?baseline_ms ~cache_dir ~backend ~timing_identity ~limits
    ~capabilities canon =
  let stored =
    entry ?saved ?segments ?best_ms ?baseline_ms ~backend
      ~codegen:(SC.codegen_tag ~backend ~limits ~capabilities ())
      ~objective:(SC.objective_tag ()) ~source_digest:(SC.digest canon) ()
  in
  let (), io =
    SC.recording_cache_io (fun () ->
        SC.store ~dir:cache_dir
          ~key:(key ~backend ~timing_identity ~limits ~capabilities canon)
          stored)
  in
  io

(** {2 The cache-I/O refusal waiver}

    The cache absorbs a filesystem refusal by design (gh-ocannl-1040): a refused store or lookup
    leaves a claim about a store or a replay false for a reason none of its own predicates observes.
    Such a claim is waived, as an environment skip and never as a pass, where an operation it rests
    on was refused. A deterministic refusal (a rejected filename, a commit that always fails) would
    waive it on every run, so a test using the waiver also keeps one claim no refusal waives: not
    every store it attempted was refused. *)

(** Whether the filesystem refused the operation. *)
let refused (io : SC.cache_io) = Option.is_some io.refusal

(** The waiver's skip reason. *)
let refused_on = "refused cache I/O"

(** [refused_in ios label]: whether any of [ios] was refused, naming on stderr, for the claim
    [label] it waives, what refused and why. *)
let refused_in ios label =
  let refusals = List.filter ios ~f:refused in
  List.iter refusals ~f:(fun (io : SC.cache_io) ->
      Stdio.eprintf "%s (not part of the golden): cache I/O under %s refused: %s\n%!" label io.key
        (Option.value io.refusal ~default:""));
  not (List.is_empty refusals)

(** [unless_refused ios label b] claims [b] where none of [ios] was refused, and skips [label] as an
    environment skip ({!refused_on}) where one was. *)
let unless_refused ios label b =
  Verdict.gated ~aggregation:`Environment ~when_:(not (refused_in ios label)) ~on:refused_on label b
