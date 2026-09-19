(* gh-ocannl-835: cache-open prunes one whole generation when the filename-key regime advances.

   These are synthetic cache directories: the old-stamp case proves that the transition removes a
   non-empty population exactly once and leaves a current entry alone; the unstamped case exercises
   the first upgrade from legacy caches; and the future-stamp case proves that an older
   participating binary refuses both reads and writes without touching what the newer regime
   owns. *)

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
  SC.store ~dir:old_cache_dir ~key:(Some "current") (entry "current");
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
  p "a future regime refuses an otherwise readable entry"
    (Option.is_none (SC.lookup ~dir:future_cache_dir ~key:(Some "kept")));
  SC.store ~dir:future_cache_dir ~key:(Some "refused-write") (entry "older-writer");
  p "a refused future-regime open changes no entries"
    (String.equal kept_before (read (entry_file future_cache_dir "kept"))
    && not (Stdlib.Sys.file_exists (entry_file future_cache_dir "refused-write")));
  p "a refused future-regime open does not rewrite its stamp"
    (String.equal
       (String.strip (read (stamp_file future_cache_dir)))
       (Int.to_string future_version));
  clean_dir old_cache_dir;
  clean_dir legacy_cache_dir;
  clean_dir future_cache_dir
