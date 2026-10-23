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
   reasons.

   This file reads the repository and words the verdicts; deciding them is
   [Test_utils.Config_dep_resolution.analyze], which takes the tree as data so that
   `dune_scan_cases` can hand it synthetic trees with known answers (gh-ocannl-603). The golden
   below can only say that today's tree is consistent; those cases say that a tree where X is wrong
   is rejected. *)

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
    ( "test/operations:python3, handed %{dep:bin_cells.py}, %{dep:../../benchmarks/bin_cells.py}, \
       %{dep:bin_cells_probe.exe}, %{dep:bench_design.expected}",
      "the hermetic bench driver's control (gh-ocannl-743) does launch an OCANNL-linked probe, but \
       only through the driver, which passes `--ocannl_no_config_file=true` on every cell's \
       command line -- disabling the config-file search outright -- and the control fails a cell \
       that resolved any setting from a config file, so no `ocannl_config` reaches what it checks"
    );
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

module Resolution = Test_utils.Config_dep_resolution

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune file or ocannl_config...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* Paths arrive relative to the rule's directory deep in the build tree; [Scan.base_dir] and
     [Scan.repo_relative] turn them into repository-relative ones (see their comment there). They
     are reported repository-relative and opened as dune handed them over. *)
  let base = Scan.base_dir Stdlib.Sys.argv.(1) in
  let paths =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let of_basename name =
    List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) name)
  in
  let dune_files =
    List.map (of_basename "dune") ~f:(fun (path, on_disk) -> (path, In_channel.read_all on_disk))
  in
  if List.is_empty dune_files then (
    Verdict.fail "no dune files among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  (* Everything from here to the printing is [Config_dep_resolution.analyze], which reads nothing
     from the filesystem, so `config_dep_resolution_cases` pins it on synthetic trees
     (gh-ocannl-603). What stays here is turning its verdicts into refusals and tallies. *)
  let report =
    Resolution.analyze ~exemption_keys:(List.map exempt_sites ~f:fst)
      ~config_files:(List.map (of_basename Scan.config_file) ~f:fst)
      dune_files
  in
  (match report.Resolution.ambient_producer with
  | Resolution.Producer_ok -> ()
  | Resolution.Environment_unrecorded ->
      fail
        (Printf.sprintf
           "%s/dune no longer produces the ambient record its tests depend on: it needs a rule \
            targeting `%s` that depends on `(universe)`, and one in `(subdir %s …)` targeting `%s` \
            that depends on `../%s` (gh-ocannl-1149)"
           Resolution.ambient_home Scan.ambient_file Resolution.ambient_subdir Scan.config_file
           Scan.ambient_file)
  | Resolution.Host_unrecorded ->
      fail
        (Printf.sprintf
           "%s/dune records the ambient environment but not the host: it needs a rule targeting \
            `%s` that depends on `(universe)`, and the rule producing `%s` must read it, or a \
            capability a backend probes on the host can move without rerunning a test \
            (gh-ocannl-1038)"
           Resolution.ambient_home Resolution.host_record_file Scan.ambient_file));
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
  List.iter report.Resolution.files ~f:(fun file ->
      let dune_file = file.Resolution.dune_file in
      (* A stanza kind the scan cannot place might carry an action that runs a test executable, so
         it is reported rather than counted as nothing (Codex P2, round 1). *)
      List.iter file.Resolution.unclassified_heads ~f:(fun head ->
          fail
            (Printf.sprintf
               "%s has a `(%s ...)` stanza, which this check has no classification for -- add it \
                to Dune_stanza_scan.action_heads if it can run an executable, or to inert_heads if \
                it cannot"
               dune_file head));
      List.iter file.Resolution.path_rewriting ~f:(fun head ->
          fail
            (Printf.sprintf
               "%s has a `(%s ...)` stanza that changes command resolution through PATH or a \
                binaries mapping -- here and in subdirectories. This check does not model \
                environment stanzas: run programs by path, or teach Dune_stanza_scan to carry the \
                override"
               dune_file head));
      List.iter file.Resolution.sites ~f:(fun r ->
          let { Scan.kind; name; _ } = r.Resolution.site in
          (* gh-ocannl-1359: what such a site runs, and so what it reads, is past this scan's reach
             -- a PATH interpreter handed a workspace script, a shell line -- and so is whether its
             hand-listed dependencies are all of them. Unsandboxed, an unlisted file is present
             whenever another rule happened to copy it, so the omission passes everywhere but a
             fresh build. Dune can enforce the list where this scan cannot, in a sandbox that
             copies: required whether or not the site is exempt from the config dep, which is a
             different question. *)
          if r.Resolution.needs_isolating_sandbox then
            fail
              (Printf.sprintf
                 "%s runs `%s`, whose reads this check cannot place, outside an isolating sandbox \
                  -- an unlisted file it reads passes whenever another rule copied it. Add \
                  `(sandbox always preserve_file_kind)` to its deps: `always` alone may pick the \
                  symlink sandbox, which an interpreter resolves straight back into _build \
                  (build-and-test.md#dune-mechanics)"
                 dune_file name);
          match r.Resolution.verdict with
          | Resolution.Exempt -> bump "exempt"
          | Resolution.Counted -> bump (Scan.kind_name kind)
          | Resolution.No_config_in_reach ->
              fail
                (Printf.sprintf
                   "%s runs %s in %s, which has no %s at or above it to depend on -- add \
                    `(copy_files ../config/%s)`"
                   dune_file name r.Resolution.directory Scan.config_file Scan.config_file)
          | Resolution.Unestablished_directory ->
              fail
                (Printf.sprintf
                   "%s runs `%s`, and a shell may change directory without dune knowing -- so \
                    which %s the process finds cannot be established here. Write the action as a \
                    `run` (with a `chdir` if it needs one), or exempt it by name with the reason"
                   dune_file name Scan.config_file)
          | Resolution.Ambient_unreached config_dir ->
              fail
                (Printf.sprintf
                   "%s: the %s %s reads %s/%s, which carries no record of the ambient environment, \
                    and depends on neither `%s` nor `(universe)` -- so a changed OCANNL_* variable \
                    serves its previous result. Copy the configuration from %s, or add `%s` \
                    (copied from %s) to its deps (gh-ocannl-1149)"
                   dune_file (Scan.kind_name kind) name config_dir Scan.config_file
                   Scan.ambient_file Resolution.ambient_config_source Scan.ambient_file
                   Resolution.ambient_home)
          | Resolution.Unreadable_undeclared ->
              fail
                (Printf.sprintf
                   "%s runs `%s`, which this check cannot read, and declares no %s -- declare it \
                    anyway, or name the executable in a way Dune_stanza_scan.classify_command \
                    places (a path, `%%{bin:…}`, or a named dependency)"
                   dune_file name Scan.config_file)
          | Resolution.Unclassified_undeclared ->
              fail
                (Printf.sprintf
                   "%s has a `(%s ...)` action this check has no classification for, and declares \
                    no %s -- declare it anyway, or add the head to \
                    Dune_stanza_scan.program_actions / inert_actions"
                   dune_file name Scan.config_file)
          | Resolution.Undeclared ->
              fail
                (Printf.sprintf "%s: the %s %s does not declare %s in its deps" dune_file
                   (Scan.kind_name kind) name Scan.config_file));
      (* The floor the raw text puts under the sexp walk ([Config_dep_resolution.floor_hole]): a
         walk that stops seeing stanzas is caught here instead of quietly reporting a smaller
         number. *)
      List.iter file.Resolution.floor_holes ~f:(fun hole ->
          Int.incr floor_holes;
          match hole with
          | Resolution.Test_hole { cwd; in_text; in_walk } ->
              fail
                (Printf.sprintf
                   "%s: the raw text shows %d test %s running%s but the scan placed only %d -- it \
                    is reading the file with a hole in it"
                   dune_file in_text
                   (if in_text = 1 then "stanza" else "stanzas")
                   (if String.is_empty cwd then " in its own directory" else " in " ^ cwd)
                   in_walk)
          | Resolution.Inline_hole { dir; in_text; in_walk } ->
              fail
                (Printf.sprintf
                   "%s: the raw text shows %d inline-test %s in%s but the scan placed only %d -- \
                    it is reading the file with a hole in it"
                   dune_file in_text
                   (if in_text = 1 then "library" else "libraries")
                   (if String.is_empty dir then " its own directory" else " " ^ dir)
                   in_walk)
          | Resolution.Unnameable_hole { dir; in_text; in_walk } ->
              fail
                (Printf.sprintf
                   "%s: the raw text runs %d bare %s under `(setenv PATH ...)`%s, and the scan \
                    placed only %d it refuses to name for that reason -- it is reading the file \
                    with a hole in it"
                   dune_file in_text
                   (if in_text = 1 then "command" else "commands")
                   (if String.is_empty dir then "" else " in " ^ dir)
                   in_walk)
          | Resolution.Run_hole { cwd; exe; in_text; in_walk } ->
              fail
                (Printf.sprintf
                   "%s: the raw text runs `%s`%s in %d %s, and the scan placed it in only %d -- it \
                    is reading the file with a hole in it"
                   dune_file exe
                   (if String.is_empty cwd then "" else " in " ^ cwd)
                   in_text
                   (if in_text = 1 then "stanza" else "stanzas")
                   in_walk));
      if file.Resolution.floor_applies then Int.incr floors_checked;
      let tally kind =
        List.count file.Resolution.sites ~f:(fun r ->
            Poly.equal r.Resolution.site.Scan.kind kind
            && not (Poly.equal r.Resolution.verdict Resolution.Exempt))
      in
      let exempted =
        List.filter file.Resolution.sites ~f:(fun r ->
            Poly.equal r.Resolution.verdict Resolution.Exempt)
      in
      let exempt_names =
        List.map exempted ~f:(fun r -> r.Resolution.site.Scan.name)
        |> List.dedup_and_sort ~compare:String.compare
        |> String.concat ~sep:", "
      in
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
          dune_file (tally Scan.Test) file.Resolution.test_floor (tally Scan.Inline_tests)
          file.Resolution.inline_floor (tally Scan.Runs_executable) file.Resolution.run_floor
          (List.length exempted)
        :: !inventory);
  let stale = report.Resolution.stale_exemptions in
  if not (List.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted sites that no rule runs any more -- drop them from the exemption list: %s"
         (String.concat ~sep:", " stale));
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
    Resolution.ambient_config_source
    (String.concat ~sep:", " report.Resolution.ambient_config_dirs)
    Scan.ambient_file Resolution.ambient_home
    (String.concat ~sep:", "
       (List.filter report.Resolution.ambient_record_dirs
          ~f:(Fn.non (String.equal Resolution.ambient_home))));
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
