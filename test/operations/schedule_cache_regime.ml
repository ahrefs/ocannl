(* gh-ocannl-835: cache-open prunes one whole generation when the filename-key regime advances.

   These are synthetic cache directories: the old-stamp case proves that the transition removes a
   non-empty population exactly once and leaves a current entry alone; the unstamped case exercises
   the first upgrade from legacy caches; and the future-stamp case proves that an older
   participating binary refuses both reads and writes without touching what the newer regime owns.

   gh-ocannl-1040: every store and lookup notes what it came to to
   [Schedule_cache.recording_cache_io], the record a test reads to tell a refusal the cache absorbed
   from a store or replay that never happened. A committed store and an admitted lookup (hit or
   miss, a missing directory and an undecodable entry included) are not refusals; a regime refusal,
   a refused lock, a refused read of an existing entry and a filesystem refusal in the
   staged-but-uncommitted window -- where a Windows commit that outlives its bounded retry fails --
   are, each with its reason. *)

open Base
module SC = Ir.Schedule_cache
open Verdict.Claims

let entry backend : SC.entry =
  {
    version = SC.entry_version;
    backend;
    numerics = SC.numerics_tag ();
    codegen = None;
    objective = None;
    source_digest = "gh835-source";
    saved = [];
    segments = None;
    finer_fission = None;
    best_ms = 1.;
    baseline_ms = 2.;
    default_ms = None;
    mma_best_ms = None;
    default_fingerprint = None;
    best_steps = None;
  }

let clean_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then (
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir name));
    Stdlib.Sys.rmdir dir)

let make_dir dir =
  clean_dir dir;
  Stdlib.Sys.mkdir dir 0o755

let stamp_file dir = Stdlib.Filename.concat dir SC.regime_stamp_filename
let entry_file dir key = Stdlib.Filename.concat dir (key ^ ".sexp")

let write_stamp dir version =
  Stdio.Out_channel.write_all (stamp_file dir) ~data:(Int.to_string version ^ "\n")

let write_entry dir key value =
  Stdio.Out_channel.write_all (entry_file dir key)
    ~data:(Sexp.to_string_hum (SC.sexp_of_entry value))

let read path = Stdio.In_channel.read_all path

module FI = Ir.Resource_fault_injection

(* The record of [f]'s cache I/O, as (op, key, refused) triples. *)
let recorded f =
  let (), io = SC.recording_cache_io f in
  List.map io ~f:(fun (r : SC.cache_io) -> (r.SC.op, r.SC.key, Option.is_some r.SC.refusal))

let equal_op a b = Poly.equal (a : SC.cache_op) b

let equal_record =
  List.equal (fun (o, k, r) (o', k', r') -> equal_op o o' && String.equal k k' && Bool.equal r r')

let () =
  let old_cache_dir = "autotune_cache_regime" in
  make_dir old_cache_dir;
  let old_keys = [ "old-a"; "old-b" ] in
  List.iter old_keys ~f:(fun key -> write_entry old_cache_dir key (entry key));
  write_stamp old_cache_dir (SC.cache_regime_version - 1);
  p "an old stamped generation opens as a cache miss"
    (Option.is_none (SC.lookup ~dir:old_cache_dir ~key:(Some "old-a")));
  Verdict.p_none ~min:2 "every old-regime entry is swept" old_keys ~f:(fun key ->
      Stdlib.Sys.file_exists (entry_file old_cache_dir key));
  p "the completed sweep atomically advances the regime stamp"
    (String.equal
       (String.strip (read (stamp_file old_cache_dir)))
       (Int.to_string SC.cache_regime_version));
  p "a committed store and an admitted lookup are recorded, neither as a refusal"
    (equal_record
       (recorded (fun () ->
            SC.store ~dir:old_cache_dir ~key:(Some "current") (entry "current");
            ignore (SC.lookup ~dir:old_cache_dir ~key:(Some "absent") : SC.entry option)))
       [ (SC.Store, "current", false); (SC.Lookup, "absent", false) ]);
  p "a current entry remains readable across later cache opens"
    (match SC.lookup ~dir:old_cache_dir ~key:(Some "current") with
    | Some value -> String.equal value.SC.backend "current"
    | None -> false);

  (* Existing caches predate the stamp itself. Absence is the initial superseded generation, not a
     reason to preserve the entries that motivated this transition. *)
  let legacy_cache_dir = "autotune_cache_regime_legacy" in
  make_dir legacy_cache_dir;
  write_entry legacy_cache_dir "legacy" (entry "legacy");
  p "an unstamped legacy generation is swept and stamped current"
    (Option.is_none (SC.lookup ~dir:legacy_cache_dir ~key:(Some "legacy"))
    && (not (Stdlib.Sys.file_exists (entry_file legacy_cache_dir "legacy")))
    && String.equal
         (String.strip (read (stamp_file legacy_cache_dir)))
         (Int.to_string SC.cache_regime_version));

  let future_cache_dir = "autotune_cache_regime_refusal" in
  make_dir future_cache_dir;
  let kept = entry "future-owned" in
  write_entry future_cache_dir "kept" kept;
  let kept_before = read (entry_file future_cache_dir "kept") in
  let future_version = SC.cache_regime_version + 1 in
  write_stamp future_cache_dir future_version;
  let refused_io =
    recorded (fun () ->
        p "a future regime refuses an otherwise readable entry"
          (Option.is_none (SC.lookup ~dir:future_cache_dir ~key:(Some "kept")));
        SC.store ~dir:future_cache_dir ~key:(Some "refused-write") (entry "older-writer"))
  in
  p "a regime-refused lookup and store are each recorded as a refusal"
    (equal_record refused_io [ (SC.Lookup, "kept", true); (SC.Store, "refused-write", true) ]);
  p "a refused future-regime open changes no entries"
    (String.equal kept_before (read (entry_file future_cache_dir "kept"))
    && not (Stdlib.Sys.file_exists (entry_file future_cache_dir "refused-write")));
  p "a refused future-regime open does not rewrite its stamp"
    (String.equal
       (String.strip (read (stamp_file future_cache_dir)))
       (Int.to_string future_version));
  (* The record's remaining cases, in a fresh current directory. *)
  let io_cache_dir = "autotune_cache_regime_io" in
  clean_dir io_cache_dir;
  p "a lookup in a directory not yet created is a miss, not a refusal"
    (equal_record
       (recorded (fun () ->
            ignore (SC.lookup ~dir:io_cache_dir ~key:(Some "early") : SC.entry option)))
       [ (SC.Lookup, "early", false) ]);
  p "a call without a key does no I/O and records nothing"
    (equal_record
       (recorded (fun () ->
            SC.store ~dir:io_cache_dir ~key:None (entry "keyless");
            ignore (SC.lookup ~dir:io_cache_dir ~key:None : SC.entry option)))
       []);
  SC.store ~dir:io_cache_dir ~key:(Some "held") (entry "held");
  let held_before = read (entry_file io_cache_dir "held") in
  (* What a Windows commit raises once its bounded retry runs out ([Atomic_file.publish_staged]): a
     [Sys_error] in the staged-but-uncommitted window. *)
  let commit_refusal = "gh1040 injected sharing violation" in
  let (), commit_io =
    SC.recording_cache_io (fun () ->
        FI.with_callback
          (fun point ->
            if FI.equal_point point FI.Schedule_cache_before_commit then
              raise (Stdlib.Sys_error commit_refusal))
          ~f:(fun () -> SC.store ~dir:io_cache_dir ~key:(Some "held") (entry "replacement")))
  in
  p "a store whose commit the filesystem refused is recorded as a refusal, with its reason"
    (match commit_io with
    | [ { SC.op; key; refusal = Some reason; _ } ] ->
        equal_op op SC.Store && String.equal key "held"
        && String.is_substring reason ~substring:commit_refusal
    | _ -> false);
  p "the refused commit leaves the earlier entry in place"
    (String.equal held_before (read (entry_file io_cache_dir "held")));
  let replay_with exn =
    recorded (fun () ->
        FI.with_callback
          (fun point -> if FI.equal_point point FI.Schedule_cache_before_replay then raise exn)
          ~f:(fun () -> ignore (SC.lookup ~dir:io_cache_dir ~key:(Some "held") : SC.entry option)))
  in
  p "a read of an existing entry the filesystem refused is recorded as a lookup refusal"
    (equal_record
       (replay_with (Stdlib.Sys_error "gh1040 injected read refusal"))
       [ (SC.Lookup, "held", true) ]);
  p "an entry that fails to decode is a miss the lookup decided, not a refusal"
    (equal_record
       (replay_with (Failure "gh1040 injected decode failure"))
       [ (SC.Lookup, "held", false) ]);
  let lock_refused =
    recorded (fun () ->
        FI.with_callback
          (fun point ->
            if FI.equal_point point FI.Schedule_cache_before_lock then
              raise (Unix.Unix_error (Unix.EACCES, "lockf", "gh1040")))
          ~f:(fun () ->
            SC.store ~dir:io_cache_dir ~key:(Some "locked-out") (entry "locked-out");
            ignore (SC.lookup ~dir:io_cache_dir ~key:(Some "held") : SC.entry option)))
  in
  p "a refused lock is recorded as a refusal of both a store and a lookup"
    (equal_record lock_refused [ (SC.Store, "locked-out", true); (SC.Lookup, "held", true) ]);
  let (), outer =
    SC.recording_cache_io (fun () ->
        p "recordings nest: the inner one sees its own I/O"
          (equal_record
             (recorded (fun () ->
                  ignore (SC.lookup ~dir:io_cache_dir ~key:(Some "inner") : SC.entry option)))
             [ (SC.Lookup, "inner", false) ]);
        ignore (SC.lookup ~dir:io_cache_dir ~key:(Some "outer") : SC.entry option))
  in
  p "recordings nest: the outer one sees the inner one's I/O too"
    (List.equal String.equal
       (List.map outer ~f:(fun (r : SC.cache_io) -> r.SC.key))
       [ "inner"; "outer" ]);
  clean_dir old_cache_dir;
  clean_dir legacy_cache_dir;
  clean_dir future_cache_dir;
  clean_dir io_cache_dir
