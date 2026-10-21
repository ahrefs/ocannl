(* gh-ocannl-835: a store's cache-open prunes one whole generation when the filename-key regime
   advances. gh-ocannl-1439: a lookup never does -- it reads a superseded generation as misses and
   leaves every file in place, so reading a cache another binary wrote cannot destroy it.

   These are synthetic cache directories: the old-stamp case proves that a lookup leaves a non-empty
   superseded population whole, that the next store then removes it exactly once, and that the store
   leaves a current entry alone; the unstamped case exercises the first upgrade from legacy caches
   the same way; and the future-stamp case proves that an older participating binary refuses both
   reads and writes without touching what the newer regime owns.

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
module Cache_plant = Test_utils.Cache_plant

let entry backend =
  Cache_plant.entry ~backend ~source_digest:"gh835-source" ~best_ms:1. ~baseline_ms:2. ()

let clean_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then (
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir name));
    Stdlib.Sys.rmdir dir)

let make_dir dir =
  clean_dir dir;
  Stdlib.Sys.mkdir dir 0o755

let stamp_file dir = Stdlib.Filename.concat dir SC.regime_stamp_filename
let entry_file ~cache_dir key = SC.cache_file ~dir:cache_dir ~key

let write_stamp dir version =
  Stdio.Out_channel.write_all (stamp_file dir) ~data:(Int.to_string version ^ "\n")

let write_entry ~cache_dir key value =
  Stdio.Out_channel.write_all (entry_file ~cache_dir key)
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

(* Every entry and the stamp, by name and contents: what a read must leave as it found it. The lock
   file is left out -- a lookup takes the record lock, creating the permanent lock file, but never
   writes to it. *)
let snapshot dir =
  Stdlib.Sys.readdir dir |> Array.to_list
  |> List.filter ~f:(fun name -> not (String.equal name SC.regime_lock_filename))
  |> List.sort ~compare:String.compare
  |> List.map ~f:(fun name -> (name, read (Stdlib.Filename.concat dir name)))

let equal_snapshot = List.equal (fun (n, c) (n', c') -> String.equal n n' && String.equal c c')

let () =
  let old_cache_dir = "autotune_cache_regime" in
  make_dir old_cache_dir;
  let old_keys = [ "old-a"; "old-b" ] in
  List.iter old_keys ~f:(fun key -> write_entry ~cache_dir:old_cache_dir key (entry key));
  write_stamp old_cache_dir (SC.cache_regime_version - 1);
  let old_before = snapshot old_cache_dir in
  p "an old stamped generation looks up as a miss the lookup decided, not a refusal"
    (equal_record
       (recorded (fun () ->
            p "an old stamped generation opens as a cache miss"
              (Option.is_none (SC.lookup ~dir:old_cache_dir ~key:(Some "old-a")))))
       [ (SC.Lookup, "old-a", false) ]);
  Verdict.p_all ~min:2 "a lookup leaves every old-regime entry in place" old_keys ~f:(fun key ->
      Stdlib.Sys.file_exists (entry_file ~cache_dir:old_cache_dir key));
  p "a lookup changes no entry and leaves the old stamp as it was"
    (equal_snapshot old_before (snapshot old_cache_dir));
  p "a committed store and an admitted lookup are recorded, neither as a refusal"
    (equal_record
       (recorded (fun () ->
            SC.store ~dir:old_cache_dir ~key:(Some "current") (entry "current");
            ignore (SC.lookup ~dir:old_cache_dir ~key:(Some "absent") : SC.entry option)))
       [ (SC.Store, "current", false); (SC.Lookup, "absent", false) ]);
  Verdict.p_none ~min:2 "the store sweeps every old-regime entry" old_keys ~f:(fun key ->
      Stdlib.Sys.file_exists (entry_file ~cache_dir:old_cache_dir key));
  p "the store's completed sweep atomically advances the regime stamp"
    (String.equal
       (String.strip (read (stamp_file old_cache_dir)))
       (Int.to_string SC.cache_regime_version));
  p "a current entry remains readable across later cache opens"
    (match SC.lookup ~dir:old_cache_dir ~key:(Some "current") with
    | Some value -> String.equal value.SC.backend "current"
    | None -> false);

  (* Existing caches predate the stamp itself. Absence is the initial superseded generation, not a
     reason to preserve the entries that motivated this transition -- once a store comes. *)
  let legacy_cache_dir = "autotune_cache_regime_legacy" in
  make_dir legacy_cache_dir;
  write_entry ~cache_dir:legacy_cache_dir "legacy" (entry "legacy");
  let legacy_before = snapshot legacy_cache_dir in
  p "an unstamped legacy generation looks up as a miss, left whole and unstamped"
    (Option.is_none (SC.lookup ~dir:legacy_cache_dir ~key:(Some "legacy"))
    && equal_snapshot legacy_before (snapshot legacy_cache_dir)
    && not (Stdlib.Sys.file_exists (stamp_file legacy_cache_dir)));
  SC.store ~dir:legacy_cache_dir ~key:(Some "fresh") (entry "fresh");
  p "a store sweeps an unstamped legacy generation and stamps it current"
    ((not (Stdlib.Sys.file_exists (entry_file ~cache_dir:legacy_cache_dir "legacy")))
    && Stdlib.Sys.file_exists (entry_file ~cache_dir:legacy_cache_dir "fresh")
    && String.equal
         (String.strip (read (stamp_file legacy_cache_dir)))
         (Int.to_string SC.cache_regime_version));

  let future_cache_dir = "autotune_cache_regime_refusal" in
  make_dir future_cache_dir;
  let kept = entry "future-owned" in
  write_entry ~cache_dir:future_cache_dir "kept" kept;
  let kept_before = read (entry_file ~cache_dir:future_cache_dir "kept") in
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
    (String.equal kept_before (read (entry_file ~cache_dir:future_cache_dir "kept"))
    && not (Stdlib.Sys.file_exists (entry_file ~cache_dir:future_cache_dir "refused-write")));
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
  let held_before = read (entry_file ~cache_dir:io_cache_dir "held") in
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
    (String.equal held_before (read (entry_file ~cache_dir:io_cache_dir "held")));
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
