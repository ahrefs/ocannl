(* gh-ocannl-597: every stanza that runs a test executable declares the shared ocannl_config.

   A test resolves its configuration by walking UP from its working directory, which under dune is
   [_build/default/<test dir>]; `(copy_files ../config/ocannl_config)` puts the shared config file
   there, but only for the stanzas that DEPEND on it. Nothing is sandboxed, so a stanza that omits
   the dep does not fail -- it reads whatever is in the directory when it happens to run. The test
   then gets default `print_decimals_precision`/`fixed_state_for_init` and a config-sourcing trace
   on stderr, or not, depending on run order.

   gh-ocannl-586 fixed seven stanzas that had drifted that way and left a sentence in AGENTS.md
   standing between the project and the eighth -- which is the same guarantee the seven had, none.
   Seventeen candidates had accumulated unnoticed, which is the measure of how quiet the failure is.
   This test is the forcing function instead.

   What it checks, over every dune file in the repository:

   - every `(test)`/`(tests)` stanza, and every `(library)` with `(inline_tests)`, lists
   `ocannl_config` in the deps dune uses when it runs them; - every `(rule ...)` that runs an
   executable does too -- the predicate that carries the content, because an `(executable)` stanza
   has no `deps` field at all, so the dep belongs on its companion rule; - and a directory with any
   such stanza has an `ocannl_config` to depend on in the first place; - and a rule running
   something whose reads this scan cannot place -- a PATH interpreter handed a workspace script, a
   shell line -- runs it in a sandbox that copies (`(sandbox always preserve_file_kind)`), so that
   dune enforces the dependency list this scan cannot (gh-ocannl-1359).

   What the golden holds, and what it deliberately does not: the KINDS of test-running stanza each
   dune file has, not how many of each. A tally there moved on every test added anywhere in the
   repository, so every contributor had to promote this file over a change that never touched it --
   a promote indistinguishable from blessing a real regression here -- and one hot line collected a
   textual conflict from every parallel branch (gh-ocannl-665). The tallies were the "the scan did
   not go blind" signal, which survives in two churn-free forms: a directory that stops being read
   loses its kinds from its line below, and within a directory `Scan.raw_stanzas` -- a second reader
   of the raw text, sharing none of the sexp walk's machinery -- puts a floor under it: the stanzas
   it finds, and what each of them runs, are what the walk must place at least as many of. The exact
   numbers go to stderr, which a `(test)` stanza does not diff.

   The two floors compare differently, and the asymmetry is forced by how the walk groups what it
   finds. Tests compare as COUNTS: one site per name, so more stanzas than sites is a hole. Rules
   compare as a MULTISET OVER (working directory, executable), which is the pair the walk makes one
   site of -- coarser fails to detect, finer fails a correct scan. A flat set would let five of the
   six rules that each run `profile_precedence.exe` be answered for by the sixth; dropping the
   directory would do the same to one rule running an executable under two `chdir`s, whose two sites
   resolve two different configs; counting raw occurrences would fail a `progn` that runs one
   executable twice, which is one site (Codex P2, rounds 1 to 3). Those comparisons read a site's
   structured `executables`, never its display `name`, which joins several with ", " and so cannot
   be taken apart again where a path contains a comma.

   An `(executable)` no rule runs is structurally not a site and needs no exemption: that is how the
   diagnostic and tutorial executables (`bench_circles_step`, `gpt2_generate`, the `@slow` runners'
   companions) stay off this list without anyone maintaining one. What does need an exemption is a
   rule that runs something this scan cannot vouch for, and those are named below with their
   reasons. *)

open Base
open Stdio

(* Failures go through [Verdict]: reported on both channels, and the run exits nonzero from its
   teardown -- so the exit status, not the promotable golden diff, carries the verdict. A
   golden-diff test that prints its failures and exits 0 can be `dune promote`d into passing,
   blessing the FAIL text as the expected output (Codex P2, round 10); since a nonzero exit means
   dune never writes the redirected stdout, the same lines go to stderr, where they survive to be
   read (gh-ocannl-601). *)
open Verdict.Claims
module Scan = Test_utils.Dune_stanza_scan

(* Rules that run something which reads no configuration. Keyed by "<dir>:<what the scan reports>",
   and each entry has to earn its place on every run (see the staleness check): an exemption is a
   claim about what a program links, and a claim that stops being true is not a free pass. *)
let exempt_sites =
  [
    ( "test/operations:python3, handed %{dep:gh1133_summary.py}, \
       %{dep:../../benchmarks/gh1133_summary.py}",
      "the training-table report controls import only the stdlib Python summary module and launch \
       no OCANNL executable, so no configuration reader is in reach" );
    ( "test/operations:python3, handed %{dep:gh1002_kernels.py}, \
       %{dep:../../benchmarks/gh1002_cells.py}",
      "the kernel-table controls import only Python modules and parse synthetic stderr; they \
       launch no OCANNL executable and read no OCANNL configuration" );
    ( "test/operations:python3, handed %{dep:kernel_attribution.py}, \
       %{dep:../../benchmarks/kernel_attribution.py}",
      "the kprof attribution controls (gh-ocannl-1347) import only Python modules and join \
       synthetic traces; they launch no OCANNL executable and read no OCANNL configuration" );
    ( "arrayjit/lib:gen_builtins/generate.exe",
      "the builtin header generator links only the OCaml stdlib and a pure C definition table; it \
       reads no OCANNL configuration" );
    ( "test/operations:gen_codec_header.exe",
      "the codec header generator for half_codec_exhaustive and bf16_codec_exhaustive \
       (gh-ocannl-985, gh-ocannl-1069) links only the OCaml stdlib and a copy of the same pure C \
       definition table; it reads no OCANNL configuration" );
    ( "test/ppx:pp.exe",
      "the ppx driver, run to expand a source file and diff the expansion: `ppx_ocannl` links \
       base, ppxlib, str and einsum_parser -- no configuration reader -- so no `ocannl_config` can \
       reach its output" );
    ( "benchmarks/runners/ocannl:metal_queue_probe.exe",
      "the standalone Metal queue probe, run on `@bin-smoke` as the bindings' runtime canary \
       (gh-ocannl-905): it links `metal`, `ctypes` and `unix` and no arrayjit at all -- no \
       configuration reader, so no `ocannl_config` can reach what it does" );
    ( "benchmarks:python3, handed %{dep:test_orchestrate.py}",
      "the benchmark orchestrator's own unit tests, in Python: the interpreter runs a script that \
       imports orchestrate.py and runners/bench_common.py and calls no OCANNL executable, so there \
       is no configuration in reach -- and a Python process would not read one anyway" );
  ]

(* Paths arrive relative to the rule's directory deep in the build tree; [Scan.base_dir] and
   [Scan.repo_relative] turn them into repository-relative ones (see their comment there). *)
let base_dir = Scan.base_dir
let repo_relative = Scan.repo_relative

(* Where the ambient record and the configuration carrying it are produced (gh-ocannl-1149). *)
let ambient_home = "test/config"
let ambient_subdir = "ambient"

(* gh-ocannl-1038: the host half of the ambient record -- what the backends' own probes found,
   written under `(universe)` by tools/host_caps.exe and appended to `ocannl_ambient`. *)
let host_record_file = "ocannl_host_capabilities"

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune file or ocannl_config...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let paths =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let of_basename name =
    List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) name)
  in
  let dune_files = of_basename "dune" in
  (* Where an `ocannl_config` resolves: a file checked in next to the dune file, or one a
     `(copy_files ...)` puts there -- the glob that supplies these paths matches dune's targets as
     well as its sources, so both arrive the same way. The stanza is read too, so that a directory
     whose config has simply never been built still counts as having one. *)
  let config_dirs =
    of_basename Scan.config_file
    (* One spelling of the repository root: it is "" everywhere else here, while `Filename.dirname`
       calls it ".". The ancestor walk below normalizes the same way, so the two agree wherever a
       directory from one meets a directory from the other (Codex P2, round 16). *)
    |> List.map ~f:(fun (path, _) ->
        match Stdlib.Filename.dirname path with "." -> "" | directory -> directory)
    |> Set.of_list (module String)
  in
  if List.is_empty dune_files then (
    Verdict.fail "no dune files among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  (* Every directory any dune file copies a config into, gathered before anything is checked: a
     stanza in one directory may run something in another, whose config a THIRD file materializes
     (Codex P2, round 14). Read from the stanzas rather than from the glob, which matches sources
     only, so a copy that has never been built still counts. *)
  let copied_config_dirs =
    List.concat_map dune_files ~f:(fun (dune_file, on_disk) ->
        let dir = Stdlib.Filename.dirname dune_file in
        Scan.config_copy_dirs (In_channel.read_all on_disk)
        |> List.map ~f:(fun subdir -> repo_relative [ dir ] subdir))
    |> Set.of_list (module String)
  in
  (* gh-ocannl-1149: the environment, the one configuration source dune cannot see unless told.
     `test/config/dune` records the ambient `OCANNL_*` environment in `ocannl_ambient`, under
     `(universe)`, and appends that record to the shared configuration as
     `test/config/ambient/ocannl_config`. A copy of THAT file changes whenever the environment does,
     so a rule depending on it reruns; a configuration checked in beside its tests carries no
     record, and a rule reading one depends on `ocannl_ambient` itself. Both producers are checked
     below rather than trusted, since the whole guarantee rests on them. *)
  let copies name =
    List.concat_map dune_files ~f:(fun (dune_file, on_disk) ->
        let dir = Stdlib.Filename.dirname dune_file in
        Scan.copy_sources (In_channel.read_all on_disk) ~name
        |> List.map ~f:(fun (subdir, source) ->
            let into = repo_relative [ dir ] subdir in
            (into, repo_relative (Scan.split_path into) source)))
  in
  let ambient_config_source =
    String.concat ~sep:"/" [ ambient_home; ambient_subdir; Scan.config_file ]
  in
  let ambient_record_source = ambient_home ^ "/" ^ Scan.ambient_file in
  let copied_from name source =
    List.filter_map (copies name) ~f:(fun (into, from) ->
        Option.some_if (String.equal from source) into)
    |> Set.of_list (module String)
  in
  let ambient_config_dirs = copied_from Scan.config_file ambient_config_source in
  let ambient_record_dirs =
    Set.add (copied_from Scan.ambient_file ambient_record_source) ambient_home
  in
  (let producer =
     List.find dune_files ~f:(fun (path, _) -> String.equal path (ambient_home ^ "/dune"))
   in
   let rules =
     match producer with
     | None -> []
     | Some (_, on_disk) ->
         Scan.walk ""
           (Scan.stanzas (In_channel.read_all on_disk))
           ~f:(fun subdir stanza ->
             match Scan.head stanza with Some "rule" -> [ (subdir, stanza) ] | _ -> [])
   in
   let target_of stanza =
     match Scan.field stanza "target" with Some [ Sexp.Atom target ] -> Some target | _ -> None
   in
   let produces ~subdir ~target ~deps_ok =
     List.exists rules ~f:(fun (at, stanza) ->
         String.equal at subdir
         && Option.equal String.equal (target_of stanza) (Some target)
         && deps_ok (Scan.field stanza "deps"))
   in
   let record_ok =
     produces ~subdir:"" ~target:Scan.ambient_file ~deps_ok:Scan.depends_on_universe
   in
   let config_ok =
     produces ~subdir:ambient_subdir ~target:Scan.config_file ~deps_ok:(fun deps ->
         List.mem
           (Scan.declared_config_paths ~name:Scan.ambient_file deps)
           ("../" ^ Scan.ambient_file) ~equal:String.equal)
   in
   (* The host half (gh-ocannl-1038): its own `(universe)` rule, and the record's producer naming it
      IN ITS ACTION. Asked only of a producer that passed the check above, so a tree lacking both
      reports the environment's refusal alone. A dependency alone is not enough: it reruns the
      producer when the host moves, but an action that never reads the host record writes the same
      bytes, and dune's early cutoff then serves every test's cached verdict. *)
   let rec mentions sexp =
     match sexp with
     | Sexp.Atom atom -> String.is_substring atom ~substring:host_record_file
     | Sexp.List items -> List.exists items ~f:mentions
   in
   let host_ok =
     produces ~subdir:"" ~target:host_record_file ~deps_ok:Scan.depends_on_universe
     && List.exists rules ~f:(fun (at, stanza) ->
         String.is_empty at
         && Option.equal String.equal (target_of stanza) (Some Scan.ambient_file)
         && Option.exists (Scan.field stanza "action") ~f:(List.exists ~f:mentions))
   in
   if not (record_ok && config_ok) then
     fail
       (Printf.sprintf
          "%s/dune no longer produces the ambient record its tests depend on: it needs a rule \
           targeting `%s` that depends on `(universe)`, and one in `(subdir %s …)` targeting `%s` \
           that depends on `../%s` (gh-ocannl-1149)"
          ambient_home Scan.ambient_file ambient_subdir Scan.config_file Scan.ambient_file)
   else if not host_ok then
     fail
       (Printf.sprintf
          "%s/dune records the ambient environment but not the host: it needs a rule targeting \
           `%s` that depends on `(universe)`, and the rule producing `%s` must read it, or a \
           capability a backend probes on the host can move without rerunning a test \
           (gh-ocannl-1038)"
          ambient_home host_record_file Scan.ambient_file));
  let exemptions = Map.of_alist_exn (module String) exempt_sites in
  let exemptions_used = ref (Set.empty (module String)) in
  let counts = Hashtbl.create (module String) in
  let bump kind = Hashtbl.update counts kind ~f:(fun n -> 1 + Option.value n ~default:0) in
  (* The per-directory tallies, reported on stderr rather than in the golden, and how many of them
     the raw-text floor below had anything to say about -- a floor that applies nowhere would be a
     silent way for this check to become decoration. *)
  let inventory = ref [] in
  let floors_checked = ref 0 in
  let floor_holes = ref 0 in
  printf
    "Which kinds of test-running stanza each dune file has: presence, not tallies. A tally here\n\
     moved whenever a test was added anywhere in the repository, so the numbers go to stderr\n\
     instead (gh-ocannl-665). What they guarded is guarded still: a line below changes when a\n\
     directory gains its first stanza of a kind or loses its last, and within a directory the\n\
     raw-text floor reported by the claim at the end holds the count up.\n\n";
  List.iter dune_files ~f:(fun (dune_file, on_disk) ->
      let dir = Stdlib.Filename.dirname dune_file in
      let content = In_channel.read_all on_disk in
      let sites = Scan.sites content in
      (* A stanza kind the scan cannot place might carry an action that runs a test executable, so
         it is reported rather than counted as nothing (Codex P2, round 1). *)
      List.iter (Scan.unclassified_heads content) ~f:(fun head ->
          fail
            (Printf.sprintf
               "%s has a `(%s ...)` stanza, which this check has no classification for -- add it \
                to Dune_stanza_scan.action_heads if it can run an executable, or to inert_heads if \
                it cannot"
               dune_file head));
      List.iter (Scan.path_rewriting_stanzas content) ~f:(fun head ->
          fail
            (Printf.sprintf
               "%s has a `(%s ...)` stanza that changes command resolution through PATH or a \
                binaries mapping -- here and in subdirectories. This check does not model \
                environment stanzas: run programs by path, or teach Dune_stanza_scan to carry the \
                override"
               dune_file head));
      (* Both displacements matter and they compose: a `(subdir …)` stanza applies elsewhere, and a
         `chdir` action runs elsewhere again. The config an executable finds is the one in the
         directory the PROCESS runs in. *)
      let stanza_dir site = repo_relative [ dir ] site.Scan.subdir in
      let directory_of site =
        repo_relative [ dir ] (Scan.in_subdir site.Scan.subdir site.Scan.cwd)
      in
      (* Where the executable's own config search leads, nearest first: its directory and every
         ancestor below the repository root, as `Utils.config_file_args` walks them. What it reads
         is the FIRST of those that has a config -- so that is both the file that must exist and the
         one the dependency must name, whether the process runs in a descendant of the stanza's
         directory, a sibling, or the stanza's own (Codex P2, rounds 9 to 11). *)
      let has_config directory =
        Set.mem config_dirs directory || Set.mem copied_config_dirs directory
      in
      (* The repository root itself is NOT one of those stops, even though the executable's own
         search does end there. A root `ocannl_config` is gitignored personal dev settings, so
         letting one satisfy a site would make this check pass on the developer's machine and fail
         in CI, for the very directory whose missing `(copy_files ../config/ocannl_config)` it
         exists to catch. Only a config a checked-in stanza puts in reach counts. *)
      let nearest_config site =
        let rec ancestors directory =
          if String.is_empty directory then []
          else
            (* A root is its own dirname ("/", or "C:/" on Windows, where a fixture's absolute site
               reaches it): the walk ends there instead of recursing on it forever. *)
            directory
            ::
            (match Stdlib.Filename.dirname directory with
            | "." -> []
            | parent when String.equal parent directory -> []
            | parent -> ancestors parent)
        in
        List.find (ancestors (directory_of site)) ~f:has_config
      in
      let described =
        List.map sites
          ~f:(fun
              ({
                 Scan.kind;
                 name;
                 declared_config_paths;
                 declared_ambient_paths;
                 depends_on_universe;
                 isolating_sandbox;
                 declares_config = _;
                 (* gh-ocannl-659's business, checked by `env_var_deps`; this test is about the
                    config dep. *)
                 declares_backend = _;
                 path_rewritten = _;
                 executables = _;
                 subdir = _;
                 cwd = _;
               } as site)
            ->
            let key = directory_of site ^ ":" ^ name in
            (* Declaring the config the process will actually read: the nearest one on its search
               path, not merely some ancestor's (Codex P2, round 10). Dependency paths are written
               relative to the stanza, so they are resolved from there before comparing. *)
            let declares_config =
              match nearest_config site with
              | None -> false
              | Some nearest ->
                  List.exists declared_config_paths ~f:(fun declared ->
                      String.equal
                        (Stdlib.Filename.dirname (repo_relative [ stanza_dir site ] declared))
                        (if String.is_empty nearest then "." else nearest))
            in
            (* Whether a change of the environment reaches this run (gh-ocannl-1149): through the
               configuration it depends on, when that is a copy carrying the ambient record; through
               the record itself, depended on beside a configuration checked in; or through
               `(universe)`, which reruns the rule on every invocation anyway. *)
            let ambient_covered =
              depends_on_universe
              || Option.exists (nearest_config site) ~f:(Set.mem ambient_config_dirs)
              || List.exists declared_ambient_paths ~f:(fun declared ->
                  Set.mem ambient_record_dirs
                    (Stdlib.Filename.dirname
                       (repo_relative (Scan.split_path (stanza_dir site)) declared)))
            in
            (* gh-ocannl-1359: what such a site runs, and so what it reads, is past this scan's
               reach -- a PATH interpreter handed a workspace script, a shell line -- and so is
               whether its hand-listed dependencies are all of them. Unsandboxed, an unlisted file
               is present whenever another rule happened to copy it, so the omission passes
               everywhere but a fresh build. Dune can enforce the list where this scan cannot, in a
               sandbox that copies: required whether or not the site is exempt from the config dep,
               which is a different question. *)
            (match kind with
            | Scan.Unreadable_directory when not isolating_sandbox ->
                fail
                  (Printf.sprintf
                     "%s runs `%s`, whose reads this check cannot place, outside an isolating \
                      sandbox -- an unlisted file it reads passes whenever another rule copied it. \
                      Add `(sandbox always preserve_file_kind)` to its deps: `always` alone may \
                      pick the symlink sandbox, which an interpreter resolves straight back into \
                      _build (build-and-test.md#dune-mechanics)"
                     dune_file name)
            | _ -> ());
            (* An exemption is spent only where the dep is actually absent, so declaring it anyway
               makes the entry stale and the list gets pruned rather than growing quietly. *)
            let exempt =
              match kind with
              (* A shell action needs an exemption even where the dep is declared, since the dep
                 cannot be shown to be the right one; the others only where it is missing. *)
              | Scan.Unreadable_directory when Map.mem exemptions key ->
                  exemptions_used := Set.add !exemptions_used key;
                  true
              | Scan.Runs_executable when (not declares_config) && Map.mem exemptions key ->
                  exemptions_used := Set.add !exemptions_used key;
                  true
              | _ -> false
            in
            (* What the check requires is the dependency; an extra one harms nothing. So a stanza
               this scan cannot read through matters only where the dependency is ABSENT — there,
               and only there, does "what does this run?" decide anything.
               Except when what is unknown is WHERE it runs: no dependency of this stanza's can be
               shown to build the config that process will find, so declaring one settles nothing
               (Codex P2, round 8). *)
            (* An exempt site needs no config where it runs -- that is what the exemption says --
               so the availability check comes after it, not over every site (Codex P2, round
               11). *)
            (match (declares_config, exempt, kind) with
            | _, true, _ -> bump "exempt"
            | _, false, _ when Option.is_none (nearest_config site) ->
                fail
                  (Printf.sprintf
                     "%s runs %s in %s, which has no %s at or above it to depend on -- add \
                      `(copy_files ../config/%s)`"
                     dune_file name (directory_of site) Scan.config_file Scan.config_file)
            | _, false, Scan.Unreadable_directory ->
                fail
                  (Printf.sprintf
                     "%s runs `%s`, and a shell may change directory without dune knowing -- so \
                      which %s the process finds cannot be established here. Write the action as a \
                      `run` (with a `chdir` if it needs one), or exempt it by name with the reason"
                     dune_file name Scan.config_file)
            | true, _, _ when not ambient_covered ->
                fail
                  (Printf.sprintf
                     "%s: the %s %s reads %s/%s, which carries no record of the ambient \
                      environment, and depends on neither `%s` nor `(universe)` -- so a changed \
                      OCANNL_* variable serves its previous result. Copy the configuration from \
                      %s, or add `%s` (copied from %s) to its deps (gh-ocannl-1149)"
                     dune_file (Scan.kind_name kind) name
                     (Option.value_map (nearest_config site) ~default:"." ~f:(fun d ->
                          if String.is_empty d then "." else d))
                     Scan.config_file Scan.ambient_file ambient_config_source Scan.ambient_file
                     ambient_home)
            | true, _, _ -> bump (Scan.kind_name kind)
            | false, false, Scan.Unreadable_command ->
                fail
                  (Printf.sprintf
                     "%s runs `%s`, which this check cannot read, and declares no %s -- declare it \
                      anyway, or name the executable in a way Dune_stanza_scan.classify_command \
                      places (a path, `%%{bin:…}`, or a named dependency)"
                     dune_file name Scan.config_file)
            | false, false, Scan.Unclassified_action ->
                fail
                  (Printf.sprintf
                     "%s has a `(%s ...)` action this check has no classification for, and \
                      declares no %s -- declare it anyway, or add the head to \
                      Dune_stanza_scan.program_actions / inert_actions"
                     dune_file name Scan.config_file)
            | false, false, _ ->
                fail
                  (Printf.sprintf "%s: the %s %s does not declare %s in its deps" dune_file
                     (Scan.kind_name kind) name Scan.config_file));
            (site, kind, name, exempt))
      in
      let tally kind =
        List.count described ~f:(fun (_, k, _, exempt) -> Poly.equal k kind && not exempt)
      in
      let exempted = List.filter described ~f:(fun (_, _, _, exempt) -> exempt) in
      let exempt_names =
        List.map exempted ~f:(fun (_, _, name, _) -> name)
        |> List.dedup_and_sort ~compare:String.compare
        |> String.concat ~sep:", "
      in
      (* The floor the raw text puts under the sexp walk, read by [Scan.raw_stanzas]: a second
         reader that shares none of the walk's machinery, so a walk that stops seeing stanzas is
         caught here instead of quietly reporting a smaller number. Sites are counted including
         exempt ones -- an exemption is about the config dependency, not about whether the stanza
         was seen. *)
      let raw = Scan.raw_stanzas content in
      let inline_floor =
        List.filter_map raw ~f:(fun r ->
            if String.equal r.Scan.raw_head "library" && r.Scan.raw_inline_tests then
              Some r.Scan.raw_subdir
            else None)
      in
      (* Test stanzas are compared per WORKING DIRECTORY, not merely counted: a test action can run
         `%{test}` in several `chdir` branches, and [sites] emits one Test site per directory
         because each resolves a different config -- so counting the stanza once would let one of
         those sites be dropped unnoticed (Codex P2, round 5). A stanza with no custom action runs
         where it is, which is the "" both sides fall back to. *)
      let test_floor = List.concat_map raw ~f:(fun r -> r.Scan.raw_test_cwds) in
      (* The directory a process actually runs in is the stanza's `(subdir …)` composed with its
         `chdir`s -- which is what `directory_of` above resolves the config against, so it is what
         both sides must compare (Codex P2, round 7). *)
      let effective site = Scan.in_subdir site.Scan.subdir site.Scan.cwd in
      let placed_tests =
        List.filter_map described ~f:(fun (site, kind, _, _) ->
            match kind with Scan.Test -> Some (effective site) | _ -> None)
      in
      let tally_of =
        List.fold
          ~init:(Map.empty (module String))
          ~f:(fun counts key -> Map.update counts key ~f:(fun n -> 1 + Option.value n ~default:0))
      in
      let placed_test_cwds = tally_of placed_tests in
      Map.iteri (tally_of test_floor) ~f:(fun ~key:cwd ~data:in_text ->
          let in_walk = Option.value (Map.find placed_test_cwds cwd) ~default:0 in
          if in_walk < in_text then (
            Int.incr floor_holes;
            fail
              (Printf.sprintf
                 "%s: the raw text shows %d test %s running%s but the scan placed only %d -- it is \
                  reading the file with a hole in it"
                 dune_file in_text
                 (if in_text = 1 then "stanza" else "stanzas")
                 (if String.is_empty cwd then " in its own directory" else " in " ^ cwd)
                 in_walk)));
      (* Inline-test libraries are compared the same way, per directory: a `(subdir …)` moves where
         they run, so losing that placement would validate the dependency against the parent
         directory (Codex P2, round 8). *)
      let placed_inline =
        List.filter_map described ~f:(fun (site, kind, _, _) ->
            match kind with Scan.Inline_tests -> Some (effective site) | _ -> None)
        |> tally_of
      in
      Map.iteri (tally_of inline_floor) ~f:(fun ~key:dir ~data:in_text ->
          let in_walk = Option.value (Map.find placed_inline dir) ~default:0 in
          if in_walk < in_text then (
            Int.incr floor_holes;
            fail
              (Printf.sprintf
                 "%s: the raw text shows %d inline-test %s in%s but the scan placed only %d -- it \
                  is reading the file with a hole in it"
                 dune_file in_text
                 (if in_text = 1 then "library" else "libraries")
                 (if String.is_empty dir then " its own directory" else " " ^ dir)
                 in_walk)));
      (* A bare command under `(setenv PATH …)` is one the walk refuses to name -- PATH decides what
         it resolves to -- so it must have placed a site saying so. Compared as a total per file,
         since the walk's own site for it carries a composed name rather than the command (Codex P2,
         round 9). Any `Unreadable_directory` site counts: a `(bash …)` line makes one too, so the
         walk's tally is a superset and the comparison stays a floor. *)
      let unnameable_floor = List.concat_map raw ~f:(fun r -> r.Scan.raw_unnameable) in
      (* Matched on the site's OWN identity, not on unnameability in general: a `(bash …)` site is
         unnameable for an unrelated reason, and counting it here let it answer for a dropped
         PATH-rewritten one (Codex P2, round 10). *)
      let placed_rewritten =
        List.filter_map described ~f:(fun (site, _, _, _) ->
            if site.Scan.path_rewritten then Some (effective site) else None)
        |> tally_of
      in
      Map.iteri (tally_of unnameable_floor) ~f:(fun ~key:dir ~data:in_text ->
          let in_walk = Option.value (Map.find placed_rewritten dir) ~default:0 in
          if in_walk < in_text then (
            Int.incr floor_holes;
            fail
              (Printf.sprintf
                 "%s: the raw text runs %d bare %s under `(setenv PATH ...)`%s, and the scan \
                  placed only %d it refuses to name for that reason -- it is reading the file with \
                  a hole in it"
                 dune_file in_text
                 (if in_text = 1 then "command" else "commands")
                 (if String.is_empty dir then "" else " in " ^ dir)
                 in_walk)));
      (* The rules are compared as a MULTISET over (working directory, executable), which is the
         pair [sites] makes one site of. Neither coarser comparison works: a flat set would let five
         of the six rules that each run `profile_precedence.exe` be dropped with the sixth answering
         for all six, and dropping the directory would do the same to one rule running an executable
         under two `chdir`s, whose two sites resolve two different configs (Codex P2, rounds 2 and
         3). Read from the site's structured [executables] rather than from its display name, which
         joins several with ", " and so cannot be taken apart again where a path contains a
         comma. *)
      let occurrences pairs =
        List.fold pairs
          ~init:(Map.empty (module String))
          ~f:(fun counts (cwd, exe) ->
            let key = cwd ^ "\000" ^ exe in
            Map.update counts key ~f:(fun n -> 1 + Option.value n ~default:0))
      in
      let placed =
        occurrences
          (List.concat_map described ~f:(fun (site, _, _, _) ->
               List.map site.Scan.executables ~f:(fun exe -> (effective site, exe))))
      in
      let run_floor = List.concat_map raw ~f:(fun r -> r.Scan.raw_runs) in
      Map.iteri (occurrences run_floor) ~f:(fun ~key ~data:in_text ->
          let in_walk = Option.value (Map.find placed key) ~default:0 in
          if in_walk < in_text then (
            Int.incr floor_holes;
            let cwd, exe = String.lsplit2_exn key ~on:'\000' in
            fail
              (Printf.sprintf
                 "%s: the raw text runs `%s`%s in %d %s, and the scan placed it in only %d -- it \
                  is reading the file with a hole in it"
                 dune_file exe
                 (if String.is_empty cwd then "" else " in " ^ cwd)
                 in_text
                 (if in_text = 1 then "stanza" else "stanzas")
                 in_walk)));
      if
        (not (List.is_empty test_floor))
        || (not (List.is_empty inline_floor))
        || (not (List.is_empty run_floor))
        || not (List.is_empty unnameable_floor)
      then Int.incr floors_checked;
      (* The golden holds which KINDS a dune file has, not how many of each: a tally there made
         every test added anywhere in the repository churn this file, and put a single hot line in
         the way of every parallel branch (gh-ocannl-665). What blindness costs is still visible --
         a directory whose stanzas stop being seen loses its kinds from the line below, and within a
         directory the floor above holds the count up. The exact numbers go to stderr, which a
         `(test)` stanza does not diff. *)
      let present =
        [
          (tally Scan.Test, "tests");
          (tally Scan.Inline_tests, "inline-test libraries");
          (tally Scan.Runs_executable, "exe-running rules");
          (List.length exempted, "exempt rules (" ^ exempt_names ^ ")");
        ]
        |> List.filter_map ~f:(fun (n, what) -> if n = 0 then None else Some what)
      in
      printf "%s: %s\n" dune_file
        (if List.is_empty present then "nothing that runs a test executable"
         else String.concat ~sep:", " present);
      inventory :=
        Printf.sprintf
          "  %s: %d tests (floor %d), %d inline-test libraries (floor %d), %d exe-running rules \
           (%d named in the text), %d exempt"
          dune_file (tally Scan.Test) (List.length test_floor) (tally Scan.Inline_tests)
          (List.length inline_floor) (tally Scan.Runs_executable) (List.length run_floor)
          (List.length exempted)
        :: !inventory);
  let stale =
    Set.diff (Set.of_list (module String) (List.map exempt_sites ~f:fst)) !exemptions_used
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted sites that no rule runs any more -- drop them from the exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  (* The reviewable part: an exemption is a claim about what an executable links, which no scan can
     check, so it is printed rather than merely held. *)
  printf
    "\n\
     Stanza kinds that can run a test executable: %s (plus `test`, `tests` and a `library`'s \
     `inline_tests`, which dune runs itself).\n"
    (String.concat ~sep:", " Scan.action_heads);
  printf "Stanza kinds that cannot: %s. Anything else fails above.\n"
    (String.concat ~sep:", " Scan.inert_heads);
  printf "Actions that execute a program: %s.\n" (String.concat ~sep:", " Scan.program_actions);
  printf "Actions that do not: %s. Anything else fails above.\n"
    (String.concat ~sep:", " Scan.inert_actions);
  printf
    "\n\
     The ambient environment (gh-ocannl-1149) reaches a run through a copy of %s, which every rule \
     depending on it in these directories reads: %s; or through `%s` itself, depended on beside a \
     configuration checked in, copied from %s into: %s.\n"
    ambient_config_source
    (String.concat ~sep:", " (Set.to_list ambient_config_dirs))
    Scan.ambient_file ambient_home
    (String.concat ~sep:", " (Set.to_list (Set.remove ambient_record_dirs ambient_home)));
  printf "\nExempt from the dependency, running an executable that reads no configuration:\n";
  List.iter exempt_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  let count kind = Option.value (Hashtbl.find counts kind) ~default:0 in
  (* Not a golden line: stderr is where a number that moves with every added test can be read
     without being diffed (gh-ocannl-665). A `(test)` stanza diffs stdout only. *)
  eprintf
    "Stanzas that run a test executable, by dune file (not diffed -- see gh-ocannl-665):\n%s\n"
    (String.concat ~sep:"\n" (List.rev !inventory));
  eprintf
    "Totals: %d dune files; %d test stanzas, %d inline-test libraries and %d exe-running rules \
     declare %s; %d exempt.\n"
    (List.length dune_files) (count "test") (count "inline tests") (count "rule running")
    Scan.config_file (count "exempt");
  (* The blindness check, stated so that its passing reading is `true`: the sexp walk placed at
     least as many stanzas as a reader that shares none of its machinery finds in the raw text. This
     is what the pinned tallies used to be for, minus the churn -- and unlike a golden line it
     cannot be `dune promote`d into passing. *)
  printf "\n";
  Verdict.p
    "the scan places every test stanza, inline-test library and run executable the dune sources \
     show"
    (!floor_holes = 0);
  Verdict.p "that floor applies to more than one dune file" (!floors_checked > 1);
  Verdict.success_trailer ~blank_line:true
    [
      Printf.sprintf
        "every test stanza, inline-test library and exe-running rule in these files declares %s, \
         apart from the sites exempted above."
        Scan.config_file;
    ];
  Test_utils.Refusal_control_manifest.print "config_dep_completeness.ml"
