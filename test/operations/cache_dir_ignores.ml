(* Every schedule cache directory a source names carries the prefix the ignore list globs on.

   An autotune search saves its schedules into a directory named relative to the working directory.
   Under `dune runtest` that is inside `_build/`, so a test naming one leaves nothing behind in the
   repository -- until someone runs the test executable from the repository root by hand, which is
   how the tuning tests are actually developed, and the directory appears there untracked. The root
   `.gitignore` used to carry an entry per directory, and nothing but a person noticing kept the two
   in step: two entries had drifted out and were added only when a hand run on one machine surfaced
   them, one of them months after the test landed.

   A list that has to be maintained is the wrong shape for that. The ignore list is one glob over a
   name prefix instead, and what this check holds is the premise the glob rests on: every argument
   naming a cache directory names one the glob covers, so a new tuning test is ignored the day it is
   written and neither the ignore list nor this test's golden has anything to say about it.

   Two requirements, over every OCaml source in the repository:

   - every argument that names a cache directory -- an `Autotune.tune ~cache_dir`, and the `~dir` of
   a direct `Schedule_cache` operation, which creates the directory just as surely -- resolves to a
   string literal the glob covers, to the empty string, which turns the cache off, or to a parameter
   forwarded from a call site scanned in its own right. A spelling that resolves to none of those is
   reported rather than assumed harmless; - and the root `.gitignore` still carries the glob, so
   that the first requirement is not being enforced in support of a rule that has been edited away.

   "Covered by the glob" is a single directory name carrying the prefix, not merely a string
   starting with it: `Schedule_cache.ensure_dir` walks the path it is handed, so
   `autotune_cache/../leaked_cache` carries the prefix and creates `leaked_cache` beside it, which a
   glob segment -- stopping at the separator -- does not match.

   What is deliberately NOT covered: the `autotune_cache_dir` configuration key, which can point the
   cache anywhere from a config file, the environment or the commandline. That is the user-facing
   knob, and its value is legitimately an absolute path outside the repository -- benchmarks already
   set it to a temp directory -- so a repository naming convention is the wrong rule to hold it to.
   No config file in the tree sets it today; if one ever does with a bare relative name, that is a
   reviewed edit to a tracked config rather than a literal buried in a new test, which is the case
   this check exists for.

   The built-in default is a third requirement, not a free ride on the first two. Nothing in a
   source names the directory a search falls back to when no `~cache_dir` is passed, so the prefix
   holding over every explicit argument says nothing about it: change that default to a name without
   the prefix and every default search creates an untracked directory with this check green. It is
   read out of the library that defines it and held to the same rule, and its absence -- a default
   this scan can no longer locate -- is itself a failure.

   How a directory name is recovered from a source -- and why only a parse can do it -- is
   {!Cache_dir_scan}. *)

open Base
open Stdio

(* Failures go through [Verdict]: reported on both channels, and the run exits nonzero from its
   teardown -- so the exit status, not the promotable golden diff, carries the verdict
   (gh-ocannl-601). *)
open Verdict.Claims

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv

module Scan = Test_utils.Cache_dir_scan

let printf = Test_utils.Refusal_control_manifest.printf
let base_dir = Test_utils.Dune_stanza_scan.base_dir
let repo_relative = Test_utils.Dune_stanza_scan.repo_relative

let require_ignore_file ~fail = function
  | Some on_disk -> Some on_disk
  | None ->
      fail
        "the repository-root .gitignore is not among the arguments -- the rule's dependency on it \
         is missing";
      None

let require_sources ~fail sources =
  if List.is_empty sources then (
    fail "no OCaml sources among the arguments -- the rule's globs match nothing";
    false)
  else true

let require_glob ~fail ignore_content =
  if not (Scan.declares_required_glob ignore_content) then
    fail
      (Printf.sprintf
         "the root .gitignore no longer carries `%s` -- the prefix this check enforces buys \
          nothing without it, and covering the current names with bespoke entries instead is the \
          name-by-name list this replaced"
         Scan.required_glob)

let refuse_unreadable_patterns ~fail patterns =
  List.iter (Scan.unreadable_patterns patterns) ~f:(fun pattern ->
      fail
        (Printf.sprintf
           "the root .gitignore pattern `%s` could match a root-level directory and uses a glob \
            form Cache_dir_scan.glob_matches does not implement -- teach it that form rather than \
            letting the pattern count as non-matching"
           pattern))

let read_source_or_refusal ~fail ~source content =
  match Or_error.try_with (fun () -> Scan.read content) with
  | Ok reading -> Some reading
  | Error error ->
      fail
        (Printf.sprintf
           "%s is among the sources this check scans and does not parse as OCaml (%s) -- if it is \
            a build artifact rather than a source, exclude it beside the `.pp.ml` expansions"
           source
           (String.concat ~sep:" " (String.split_lines (Error.to_string_hum error))));
      None

let checked_name ~fail ~source ~line ~spelling = function
  | Scan.Names name when Scan.covered_by_glob name -> Some name
  | Scan.Names name ->
      fail
        (Printf.sprintf
           "%s:%d: `%s` names the cache directory %S, which `%s` does not cover -- it has to be a \
            single directory name starting with `%s`, since a glob segment stops at a separator \
            and `ensure_dir` follows one wherever it leads"
           source line spelling name Scan.required_glob Scan.required_prefix);
      None
  | Scan.Disabled | Scan.Forwarded _ -> None
  | Scan.Unresolved _ as resolution ->
      fail
        (Printf.sprintf
           "%s:%d: this `%s` argument %s, which this scan cannot resolve to a literal -- pass the \
            directory as a literal here, or bind it to a name mentioning `%s` whose right-hand \
            side is one, so that the prefix can be checked"
           source line spelling (Scan.describe resolution) Scan.tune_label);
      None

let check_defaults ~fail defaults =
  match defaults with
  | [] ->
      fail
        (Printf.sprintf
           "no source reads `%s` with a non-empty default -- the directory a search uses when no \
            `~%s` is passed can no longer be recovered, so nothing checks that the glob still \
            covers it. Point Cache_dir_scan.default_config_key at the key that names it now"
           Scan.default_config_key Scan.tune_label)
  | defaults ->
      List.iter defaults ~f:(fun name ->
          if not (Scan.covered_by_glob name) then
            fail
              (Printf.sprintf
                 "the built-in default of `%s` is %S, which `%s` does not cover -- a search that \
                  passes no `~%s` would create it in the working directory untracked"
                 Scan.default_config_key name Scan.required_glob Scan.tune_label))

let check_effectively_ignored ~fail patterns name =
  if not (Scan.effectively_ignored patterns name) then
    fail
      (Printf.sprintf
         "the sources name the cache directory %s, and the root .gitignore does not ignore it -- \
          some pattern after `%s` un-ignores it, so the directory would be left untracked in the \
          repository root"
         name Scan.required_glob)

let refusal_control () =
  let source = "test/operations/cache_dir_ignores.ml" in
  let case label ~format run =
    let refused = ref false in
    let fail _message =
      refused := true;
      Test_utils.Refusal_control_manifest.observe_failure ~source ~format
    in
    run fail;
    Verdict.p label !refused
  in
  case "a missing repository-root ignore file reaches its refusal"
    ~format:
      "the repository-root .gitignore is not among the arguments -- the rule's dependency on it is \
       missing" (fun fail -> ignore (require_ignore_file ~fail None : string option));
  case "an empty OCaml source corpus reaches its refusal"
    ~format:"no OCaml sources among the arguments -- the rule's globs match nothing" (fun fail ->
      ignore (require_sources ~fail [] : bool));
  case "a missing required ignore glob reaches its refusal"
    ~format:
      "the root .gitignore no longer carries `%s` -- the prefix this check enforces buys nothing \
       without it, and covering the current names with bespoke entries instead is the name-by-name \
       list this replaced" (fun fail -> require_glob ~fail "");
  case "an unsupported root ignore pattern reaches its refusal"
    ~format:
      "the root .gitignore pattern `%s` could match a root-level directory and uses a glob form \
       Cache_dir_scan.glob_matches does not implement -- teach it that form rather than letting \
       the pattern count as non-matching" (fun fail ->
      Scan.ignore_patterns "/auto**tune_cache*/" |> refuse_unreadable_patterns ~fail);
  case "an invalid OCaml source reaches the parse refusal"
    ~format:
      "%s is among the sources this check scans and does not parse as OCaml (%s) -- if it is a \
       build artifact rather than a source, exclude it beside the `.pp.ml` expansions" (fun fail ->
      ignore (read_source_or_refusal ~fail ~source:"bad.ml" "let ="));
  case "a cache name outside the required prefix reaches its refusal"
    ~format:
      "%s:%d: `%s` names the cache directory %S, which `%s` does not cover -- it has to be a \
       single directory name starting with `%s`, since a glob segment stops at a separator and \
       `ensure_dir` follows one wherever it leads" (fun fail ->
      ignore
        (checked_name ~fail ~source:"bad.ml" ~line:1 ~spelling:"~cache_dir"
           (Scan.Names "leaked_cache")));
  case "an unresolved cache argument reaches its refusal"
    ~format:
      "%s:%d: this `%s` argument %s, which this scan cannot resolve to a literal -- pass the \
       directory as a literal here, or bind it to a name mentioning `%s` whose right-hand side is \
       one, so that the prefix can be checked" (fun fail ->
      ignore
        (checked_name ~fail ~source:"bad.ml" ~line:1 ~spelling:"~cache_dir"
           (Scan.Unresolved "an expression")));
  case "an absent built-in default reaches its refusal"
    ~format:
      "no source reads `%s` with a non-empty default -- the directory a search uses when no `~%s` \
       is passed can no longer be recovered, so nothing checks that the glob still covers it. \
       Point Cache_dir_scan.default_config_key at the key that names it now" (fun fail ->
      check_defaults ~fail []);
  case "a built-in default outside the required prefix reaches its refusal"
    ~format:
      "the built-in default of `%s` is %S, which `%s` does not cover -- a search that passes no \
       `~%s` would create it in the working directory untracked" (fun fail ->
      check_defaults ~fail [ "leaked_cache" ]);
  case "a later ignore negation reaches the effective-ignore refusal"
    ~format:
      "the sources name the cache directory %s, and the root .gitignore does not ignore it -- some \
       pattern after `%s` un-ignores it, so the directory would be left untracked in the \
       repository root" (fun fail ->
      check_effectively_ignored ~fail
        (Scan.ignore_patterns "/autotune_cache*/\n!/autotune_cache_leak/")
        "autotune_cache_leak");
  Test_utils.Refusal_control_manifest.print source

let () =
  if Array.length argv = 2 && String.equal argv.(1) "--refusal-control" then (
    refusal_control ();
    Stdlib.exit 0);
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <.gitignore and source files...>\n" argv.(0);
    Stdlib.exit 1);
  let base = base_dir argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let paths =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  (* The ROOT ignore file, by its repository-relative path rather than its basename: the glob is
     root-anchored, and a `.gitignore` in a subdirectory anchors to that subdirectory instead. *)
  let ignore_file = List.Assoc.find paths ".gitignore" ~equal:String.equal in
  (* The globs reach into the build tree, which holds derived copies beside the sources dune staged:
     a `.pp.ml` is a source after preprocessing, and a `*_actual.ml` under test/ppx is a rule's
     captured output -- sometimes a compiler error message rather than OCaml at all. Neither is a
     place a directory can be named that its own source does not already name, and whether either
     exists depends on what has been built. Anything else that fails to parse is reported by name
     below rather than taking the run down.

     This check's own source is NOT excluded: a check that skipped it would carve out the one place
     a directory could be named without the prefix rule being applied to it. *)
  let derived path =
    (* `.pp.ml` is dune's own name for a source after preprocessing, so whatever one names its
       source in the same directory names too. `_actual.ml` is narrowed to the directory that
       generates them: as a bare suffix it was a repository-wide escape hatch, quietly excusing any
       real source someone happened to name that way from a check that claims to read every source
       (Codex P2, round 3). *)
    String.is_suffix path ~suffix:".pp.ml"
    || (String.is_prefix path ~prefix:"test/ppx/" && String.is_suffix path ~suffix:"_actual.ml")
  in
  let sources =
    List.filter paths ~f:(fun (path, _) ->
        String.is_suffix path ~suffix:".ml" && not (derived path))
  in
  let ignore_file =
    match require_ignore_file ~fail ignore_file with
    | Some on_disk -> on_disk
    | None -> Stdlib.exit 1
  in
  if not (require_sources ~fail sources) then Stdlib.exit 1;
  let ignore_content = In_channel.read_all ignore_file in
  let patterns = Scan.ignore_patterns ignore_content in
  require_glob ~fail ignore_content;
  (* A pattern that bears on a root-level directory and that the matcher cannot read: reported,
     because "not ignored" and "not understood" are different answers and only one of them is this
     check's to report. *)
  refuse_unreadable_patterns ~fail patterns;
  let naming = ref [] and names = ref [] and defaults = ref [] in
  List.iter sources ~f:(fun (source, on_disk) ->
      let content = In_channel.read_all on_disk in
      match read_source_or_refusal ~fail ~source content with
      | None -> ()
      | Some { Scan.uses; builtin_defaults } ->
          defaults := builtin_defaults @ !defaults;
          List.iter uses ~f:(fun { Scan.resolution; line; spelling } ->
              match checked_name ~fail ~source ~line ~spelling resolution with
              | Some name ->
                  naming := source :: !naming;
                  names := name :: !names
              | None -> ()));
  (* The directory a search uses when no `~cache_dir` is passed is named in no source, so the prefix
     rule over explicit arguments says nothing about it. Read out of the library that defines the
     default and held to the same rule -- and required to be found at all, since a default this scan
     can no longer locate is one the glob has stopped being checked against (Codex P2, round 3). *)
  let defaults = List.dedup_and_sort !defaults ~compare:String.compare in
  check_defaults ~fail defaults;
  (* The prefix rule says the glob would cover these names; whether git ACTUALLY ignores them is a
     separate question, and the one that matters. gitignore is last-match-wins, so a negation after
     the glob takes coverage away while the glob line sits there looking intact (Codex P2, round 2).
     Asked per name, against the file's patterns in order. *)
  List.iter
    (List.dedup_and_sort (defaults @ !names) ~compare:String.compare)
    ~f:(check_effectively_ignored ~fail patterns);
  (* The sources that name one, not the names themselves: what the check establishes is a property
     every name has, so listing them would only make the golden churn on each new tuning test -- the
     maintenance burden this check exists to remove. *)
  let naming = List.dedup_and_sort !naming ~compare:String.compare in
  printf "Sources naming a schedule cache directory, all with the `%s` prefix that `%s` covers:\n"
    Scan.required_prefix Scan.required_glob;
  List.iter naming ~f:(fun source -> printf "  %s\n" source);
  (* No count of sources SCANNED: the globs reach into the build tree, so how many `.ml` files they
     match depends on which `(select …)` copies have been built by the time this rule runs, and a
     golden pinning that would fail on build order rather than on anything true. *)
  (* The list above is what the census establishes; its length is a tally that every new tuning test
     moved, and two branches each adding one merged cleanly to a wrong total (gh-ocannl-1046). *)
  eprintf "Sources naming a schedule cache directory: %d (not part of the golden).\n"
    (List.length naming);
  if not (Verdict.any_failed ()) then
    printf "\nOK: every source listed names one; the ignore list needs no entry per directory.\n";
  Test_utils.Refusal_control_manifest.print "cache_dir_ignores.ml"
