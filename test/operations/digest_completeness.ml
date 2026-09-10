(* gh-ocannl-572: every codegen-affecting knob is in the cache identity, or explicitly exempted.

   gh-ocannl-568 found the schedule disk cache keying on the canonical digest and the backend name
   while the numerics policy — consulted at codegen, invisible to the lowered code — was absent, so
   a default-flags run replayed a tf32-tuned winner at 5.9x slower than not tuning at all. That is
   one instance of a general hazard: any knob that changes emitted code, schedule semantics or
   placement decisions and does NOT reach the identity a cache keys on produces silent cross-regime
   replay. This test turns the class into a standing invariant, the way [test_config_consistency]
   does for documentation and registration:

   - every known config key is classified in [Utils.config_key_classification] (a new key with no
   classification fails here, which is the forcing function), - a classification that claims a
   cache-key component names one that exists — the components are enumerated by
   [Schedule_cache.key_components], which drives [cache_key] itself, so the enumeration cannot go
   stale, - every component other than the digest is claimed by at least one key, so a component
   cannot quietly become vestigial, - and no key READ AT CODEGEN is classified as code-borne: those
   reads happen after the lowered code the digest names, so they cannot reach it — which is exactly
   how gh-ocannl-568 happened.

   The exemptions (search-shaping and execution-neutral) are deliberately not machine-checkable:
   they are the reviewable text the issue asks for, printed below so that a diff shows when one
   changes. *)

open Base
open Stdio

(* Failures go through [Verdict]: reported on both channels, and the run exits nonzero from its
   teardown. A golden-diff test that prints its failures and exits 0 can be `dune promote`d into
   passing, blessing the FAIL text as the expected output (Codex P2, round 10 of PR #343); since a
   nonzero exit means dune never writes the redirected stdout, the same lines go to stderr, where
   they survive to be read (gh-ocannl-601). *)
open Verdict.Claims

let printf = Test_utils.Refusal_control_manifest.printf

module SC = Ir.Schedule_cache
module Config_key_scan = Test_utils.Config_key_scan

(* Modules that run AFTER lowering: rendering a kernel, compiling it, dispatching it, and building
   the cache identity itself. A key whose only reads are here is invisible to the canonical digest
   by construction. [schedule_cache.ml] belongs in the list for the same reason as the backends
   (Codex P2 on PR #337): it reads its keys while assembling the codegen component, downstream of
   the code the digest names, so a [Code_borne] classification of one of them would be wrong in
   exactly the way check 3 exists to catch. *)
let codegen_stage_modules =
  [
    "c_syntax.ml";
    "cc_backend.ml";
    "cuda_backend.ml";
    "hip_backend.ml";
    "metal_backend.ml";
    "schedule_cache.ml";
  ]

let class_name : Utils.config_key_class -> string = function
  | Utils.Aggregate -> "aggregate"
  | Utils.Code_borne -> "code-borne (digest)"
  | Utils.Keyed component -> "keyed: " ^ component
  | Utils.Search_shaping -> "search-shaping"
  | Utils.Execution_neutral -> "execution-neutral"

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <dependency...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* The arguments are the rule's whole dependency set, globbed over the library directories rather
     than enumerated by hand (gh-ocannl-592). *)
  let source_files =
    Config_key_scan.sources_among (Array.to_list (Array.subo Stdlib.Sys.argv ~pos:1))
  in
  if List.is_empty source_files then (
    eprintf "%s: no sources among the arguments -- the rule's globs match nothing\n"
      Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let by_file = Config_key_scan.keys_by_file source_files in
  let listing keys = String.concat ~sep:", " (Set.to_list keys) in
  (* [codegen_stage_modules] selects by basename, so two scanned files sharing a name would make the
     selection ambiguous -- possible only since the globs replaced the enumerated list. *)
  let duplicate_basenames = Config_key_scan.duplicate_basenames source_files in
  if not (List.is_empty duplicate_basenames) then
    fail
      (Printf.sprintf
         "scanned files share a basename, which the codegen-stage list is keyed by -- rename one \
          or key the list by path: %s"
         (String.concat ~sep:", " duplicate_basenames));
  (* 1. The classification covers exactly the known keys, once each. *)
  let classified = List.concat_map Utils.config_key_classification ~f:(fun (_, _, keys) -> keys) in
  let duplicates =
    List.find_all_dups classified ~compare:String.compare |> Set.of_list (module String)
  in
  if not (Set.is_empty duplicates) then
    fail (Printf.sprintf "keys classified more than once: %s" (listing duplicates));
  let classified = Set.of_list (module String) classified in
  let unclassified = Set.diff Utils.known_config_keys classified in
  if not (Set.is_empty unclassified) then
    fail
      (Printf.sprintf
         "config keys with no cache-identity classification: %s -- classify them in \
          Utils.config_key_classification (see gh-ocannl-572)"
         (listing unclassified));
  let unknown = Set.diff classified Utils.known_config_keys in
  if not (Set.is_empty unknown) then
    fail
      (Printf.sprintf "classified keys that are not in Utils.known_config_keys: %s"
         (listing unknown));
  (* 2. Claimed components exist, and every component other than the digest is claimed. *)
  let components = Set.of_list (module String) SC.key_components in
  let claimed =
    List.filter_map Utils.config_key_classification ~f:(function
      | Utils.Keyed component, _, _ -> Some component
      | _ -> None)
    |> Set.of_list (module String)
  in
  let missing_components = Set.diff claimed components in
  if not (Set.is_empty missing_components) then
    fail
      (Printf.sprintf "classifications claim cache-key components that do not exist: %s"
         (listing missing_components));
  let unclaimed = Set.diff (Set.remove components "digest") claimed in
  if not (Set.is_empty unclaimed) then
    fail
      (Printf.sprintf
         "cache-key components no key is classified against: %s -- either a key belongs there or \
          the component is dead weight"
         (listing unclaimed));
  (* 3. A key read at codegen cannot be code-borne. *)
  let codegen_read =
    List.filter_map by_file ~f:(fun (basename, keys) ->
        if List.mem codegen_stage_modules basename ~equal:String.equal then Some keys else None)
    |> Set.union_list (module String)
  in
  let miscl =
    Set.filter codegen_read ~f:(fun key ->
        match Utils.classify_config_key key with Some (Utils.Code_borne, _) -> true | _ -> false)
  in
  if not (Set.is_empty miscl) then
    fail
      (Printf.sprintf
         "keys read at codegen but classified code-borne: %s -- a codegen read happens after the \
          lowered code the canonical digest names, so it cannot reach the digest"
         (listing miscl));
  (* Which directories the census came from is part of the golden, so that a glob which stops
     matching -- a renamed directory, a source layout change -- reads as a diff rather than as a
     quietly smaller census (gh-ocannl-592). By NAME: the file count was a tally of the repository,
     so every correct addition under these directories owed a promote round for it, and two branches
     adding a file to the same directory merged cleanly to a wrong total (gh-ocannl-701). What the
     count was for is kept by a floor per scan root, asserted below; the exact numbers go to stderr,
     which a `(test)` stanza's golden does not see. *)
  Config_key_scan.report_counts source_files;
  let floor_violations = Config_key_scan.floor_violations source_files in
  List.iter floor_violations ~f:fail;
  printf "Sources scanned: %s\n" (String.concat ~sep:", " (Config_key_scan.scan_roots source_files));
  (* The reviewable part: the classification itself, and which keys the scan found at codegen. The
     census counts both spellings of a read, [get_global_arg] call sites and [Utils.settings]
     fields, so a settings-borne key cannot slip past check 3 (Codex P2 on PR #337). *)
  printf "Cache-key components (Schedule_cache.key_components): %s\n"
    (String.concat ~sep:", " SC.key_components);
  printf "Keys read at codegen (%s): %d\n"
    (String.concat ~sep:", " codegen_stage_modules)
    (Set.length codegen_read);
  List.iter Utils.config_key_classification ~f:(fun (cls, why, keys) ->
      printf "\n%s -- %s\n" (class_name cls) why;
      List.iter (List.sort keys ~compare:String.compare) ~f:(fun key ->
          printf "  %s%s\n" key (if Set.mem codegen_read key then " [read at codegen]" else "")));
  printf "\n";
  Verdict.p_empty "every scanned root meets its source-count floor" ~over:source_files
    floor_violations;
  if not (Verdict.any_failed ()) then
    printf "OK: %d config keys classified against %d cache-key components.\n"
      (Set.length classified) (List.length SC.key_components);
  Test_utils.Refusal_control_manifest.print "digest_completeness.ml"
