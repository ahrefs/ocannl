(** Test-only digest-free replay fixtures extracted from saved autotune winners (gh-ocannl-2394).

    Saved autotune cache entries carry regime- and lowering-specific digests ([source_digest],
    per-segment [seg_digest], cache keys, versions, and timing diagnostics). When lowering or
    rendering moves those digests, replaying the saved winner through [Schedule_cache] fails even
    though the transforms themselves still apply.

    This module strips all cache metadata and digests to keep only the transforms — whole-routine or
    split-reduce prelude ops and, for fissioned winners, each segment's kind, unit count, and saved
    ops with canonical axis and tensor-node references — and validates every anchor against a fresh
    lowering before replaying, refusing by name whenever an anchor no longer exists.

    {2 Validation guarantee and boundary}

    {!Ir.Schedule_cache.entry} persists only canonical preorder ordinals ([Base i], [Static k],
    tensor-node index [n]) and never loop extents or tensor labels, so an extracted fixture carries
    no external names for them. {!validate} therefore gives a positional-plus-structural guarantee:
    segmentation coverage, [`Zeros] segment kind coherence, unambiguous resolution of every
    referenced axis and tensor-node ordinal, and live op preconditions on a hermetic copy. It cannot
    detect a lowering change that swaps two structurally compatible loops or tensor nodes while
    leaving every op legal; regression tests pinning a winner should pair replay with structural
    and/or value assertions (placement, MMA census, output parity).

    {2 Accepted input boundary (fail-closed)}

    {!extract_file} reads only {!Ir.Schedule_cache}'s on-disk stored-entry format
    [((key <k>) (payload <entry>))] via {!Ir.Schedule_cache.read_entry_file} and
    {!Ir.Schedule_cache.entry_of_sexp}. Entries in the ["placements-"] or ["abandonment-"] key
    namespaces, unreadable or unparseable files, non-entry payloads, empty segmentations
    [(segments ())], and segments with non-positive unit counts are refused with a named error.
    Older [entry_version] values whose payload schema decodes with
    {!Ir.Schedule_cache.entry_of_sexp} are accepted without checking [entry_version] or
    [cache_regime_version]. *)

open Base

type segment = {
  kind : Ir.Schedule.segment_kind;
  units : int;
  ops : Ir.Schedule_cache.saved_schedule;
}
[@@deriving sexp, equal]

type t = {
  prelude : Ir.Schedule_cache.saved_schedule;
  segments : segment list option; [@sexp.option]
}
[@@deriving sexp, equal]

type anchor_location = Prelude | Segment of int [@@deriving sexp, equal]

type anchor_error = {
  location : anchor_location;
  op_index : int option;
  op_name : string option;
  anchor : string;
  detail : string;
}
[@@deriving sexp, equal]

val error_to_string : anchor_error -> string
(** Human-readable diagnostic naming the location, op index/name (when applicable), anchor, and
    reason for refusal. *)

val extract_entry : Ir.Schedule_cache.entry -> (t, string) Result.t
(** Strips all digests, cache keys, versions, and timing diagnostics from a decoded schedule-cache
    entry, keeping only [prelude] and [segments]. Refuses empty segment lists and non-positive
    segment unit counts. *)

val extract_file : string -> (t, string) Result.t
(** Reads an on-disk saved winner [.sexp] file (fail-closed on {!Ir.Schedule_cache}'s stored-entry
    format; refuses ["placements-"] and ["abandonment-"] entries) and extracts a digest-free
    fixture. *)

val of_sexp_string : string -> (t, string) Result.t
(** Parses a digest-free fixture from its s-expression string and checks structural well-formedness.
*)

val load_file : string -> (t, string) Result.t
(** Loads a digest-free fixture [.sexp] written by {!save_file} or {!to_sexp_string}. *)

val load_file_exn : string -> t

val to_sexp_string : t -> string
(** Canonical human-readable s-expression representation, LF-terminated. *)

val save_file : path:string -> t -> unit
(** Writes {!to_sexp_string} to [path] in binary mode (LF line endings on every platform). *)

val validate :
  is_gpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  ?static_indices:Ir.Indexing.static_symbol list ->
  t ->
  Ir.Low_level.optimized ->
  (unit, anchor_error) Result.t
(** Validates [fixture] against a hermetic copy of [opt] without mutating [opt]:
    - checks fixture structure and segmentation coverage against [opt]'s lowered units (including
      [`Zeros] segment kind coherence and scope-local containment),
    - checks every axis anchor ([Base], [Static], [Minted] including op-index ordering and role
      validity against the fresh lowering's node ranks) and tensor-node anchor ([tn], [source],
      [target]) in [prelude] and every segment before applying transforms,
    - applies the validated transforms hermetically so stale structural preconditions (a loop
      consumed by an earlier op, a missing [Zero_out], non-nested [Swap] loops, a missing epilogue)
      are refused by name before any real compile runs. *)

val replay :
  is_gpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  ?static_indices:Ir.Indexing.static_symbol list ->
  t ->
  Ir.Low_level.optimized ->
  (Ir.Low_level.optimized list, anchor_error) Result.t
(** Runs {!validate} on a hermetic copy of [opt] first, then replays [fixture] onto [opt] through
    the production prelude / {!Autotune.tuner_fission} pipeline, returning the scheduled kernel
    list. Leaves [opt] untouched when validation fails. *)

val replay_exn :
  is_gpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  ?static_indices:Ir.Indexing.static_symbol list ->
  t ->
  Ir.Low_level.optimized ->
  Ir.Low_level.optimized list
(** {!replay}, raising [Invalid_argument] with {!error_to_string} on refusal. Useful directly as a
    [Context.compile ~lowered_transform] callback. *)
