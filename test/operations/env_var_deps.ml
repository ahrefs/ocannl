(* gh-ocannl-628: the ambient variables a dune rule declares, against the ones a run reads.

   A dune rule is invalidated by an environment variable only if it says so. Where that matters is
   everywhere: 213 stanzas in this repository declare `(env_var OCANNL_BACKEND)` so that changing
   the backend re-runs the test rather than serving the previous backend's output as a pass.

   Two ways that goes wrong, both silent, both checked here.

   {1 A spelling nothing reads}

   `Utils.read_env_var` consults ONE name per key, `OCANNL_<KEY>` (gh-ocannl-652). Before that it
   consulted the lowercase `ocannl_<key>` FIRST -- so `ocannl_backend=cuda` outranked
   `OCANNL_BACKEND` and decided which backend every test compiled and ran on, while not one dune
   file in the repository declared it: a developer who exported the lowercase form got stale targets
   served as passes, from rules written precisely to prevent that. gh-ocannl-628 swept the second
   spelling INTO 213 stanzas; gh-ocannl-652 dropped the spelling instead, and made setting one fatal
   so that the demotion could not be silent. What is left to check here is that a declaration names
   the spelling that is read -- a stanza declaring `(env_var ocannl_backend)` tracks a variable no
   run will consult, and invalidates nothing.

   {2 Suites a rejected spelling never reaches}

   Nothing declares the REJECTED spellings, by design, so nothing reruns for one either: a cached
   `@test/einsum/runtest` would serve its previous passes with `ocannl_backend=cuda` ambient and the
   fatal startup check never reached (gh-ocannl-652, Codex P1 round 2). Each test directory carries
   an `env_spelling_gate` whose `(universe)` dependency makes dune rerun it every invocation; this
   check is what keeps the set whole, since dune aliases are per directory and the next test
   directory would otherwise be added without one.

   Per ALIAS, not per directory (Codex P1 round 3). A `(test)` stanza is a runtest action and
   nothing else, so a directory whose gate is a test stanza is ungated for `@slow` -- a separately
   documented entry point that `dune build @slow` reaches without building one `(test)`. Asking the
   question per directory let a runtest gate vouch for the slow rules beside it, which is the shape
   of the hole it was written to prevent. And per EVERY alias a build can start from, not a fixed
   list of two: each slow rule sits on its own `slow-<name>` alias, so that one slow test can be
   rerun after a change without the ~30 minutes the whole `@slow` suite takes, and
   `ocannl_backend=cuda dune build @test/training/slow-cifar_conv` is the same entry point with the
   same hole. A gate "reaches" an alias when building the alias builds the gate: the gate's own
   alias, a rule whose `deps` name it (which runs the gate BEFORE the rule, so a rejected spelling
   fails before the slow run rather than beside it), or an `(alias (name …) (deps …))` stanza
   aggregating either. That last shape is what keeps `dune build @slow` the whole suite, and the
   check here is also what keeps the aggregation whole: a `slow-<name>` rule the `slow` alias does
   not list is one `@slow` skips, silently.

   {2 Gates the build does not see}

   The other direction: a variable that IS read and is declared nowhere. ppx_minidebug's per-module
   tracing gates (`OCANNL_LOG_LEVEL_ROW` and eighteen siblings) are read while PREPROCESSING, so a
   library that does not declare them hands back the modules it built with the trace statements
   stripped -- `OCANNL_LOG_LEVEL_ROW=9 dune build` returning a silent binary, which reads as "the
   trace shows nothing" rather than as "the trace was never compiled in". Each gate is checked
   against the library whose modules read it.

   {2 A stanza that declares neither spelling}

   The pairing check above sees a stanza that declares ONE spelling. What it cannot see is a stanza
   that declares NEITHER: a backend-sensitive test added with no `(env_var ...)` at all is invisible
   to a check phrased over the declarations that exist, and dune then serves the previous backend's
   output as a pass under `OCANNL_BACKEND=cuda dune build @…` -- the exact failure this file exists
   to prevent, arrived at from the other side (gh-ocannl-659, found on lukstafi/ocannl-staging#374
   where `simd_lane_choice` was added with no declaration; it is genuinely backend-free, and nothing
   in the build would have said otherwise had it been a `Context.auto` test).

   So the rule below is an exclusive or, asked of every stanza that runs an executable: either it
   declares `(env_var OCANNL_BACKEND)`, or it carries a marker comment inside its parentheses naming
   the backend it is pinned to -- or `none` -- and why. Both absent is the hole; both present is
   contradictory intent, since a stanza that names its backend has nothing to invalidate on.

   The conditional rule is kept, deliberately. Declaring the variable universally would be simpler
   to check and would claim a sensitivity most of these stanzas do not have: `test_einsum_parser`
   calls a parser, `test_metal_pool_bindings` pins Metal's emission, and the seven `Context.auto`
   rules in this directory pin `--ocannl_backend=cc` on the command line, which outranks the
   environment. A declaration on those would be noise, and noise is what the next reader learns to
   skip. What the marker costs instead is a sentence of classification per stanza, written where the
   next author will copy it from.

   {1 What decides "addressed to the configuration"}

   `Utils.classify_env_var`, the same function the startup check uses (gh-ocannl-629), so a name a
   rule tracks and a name a run reports cannot be classified two ways. It also supplies the reserved
   namespaces: `OCANNL_TOOL_...` is the tooling's, and `OCANNL_LOG_LEVEL_<MODULE>` is a gate,
   checked by the other half of this test. Both are uppercase-only, as configuration keys now are
   too. *)

open Base
open Stdio
open Verdict.Claims

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv

module Scan = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan
module Refusals = Test_utils.Refusal_control_scan
module Refusal_manifest = Test_utils.Refusal_control_manifest

let printf = Refusal_manifest.printf

(* Declarations of a name OCANNL does not read as a configuration key. Keyed by "<dune
   file>:<name>", and each entry earns its place on every run (see the staleness check below): a
   rule tracking a variable no key would be read from is normally a typo, which is the whole point
   of asking. *)
let exempt_declarations =
  [
    ( "test/operations/dune:ocannl_backedn",
      "the fixture behind the `config_var_warnings` golden, which captures the warning a mistyped \
       key draws; the rule tracks the name so that an ambient one arriving does not leave the \
       golden stale" );
    ( "test/operations/dune:OCANNL_BACKEDN",
      "the same fixture, in the casing OCANNL reads -- a lowercase one is reported rather than \
       read, so both write to the stream the golden holds" );
    ( "test/operations/dune:OCANNL_DEMO_KEY",
      "a synthetic key `config_var_spellings` looks up by name: no key OCANNL reads, so it cannot \
       be declared for its value -- but the run depends on it (gh-ocannl-749), an ambient one \
       drawing the unknown-key warning onto the stream those goldens capture" );
    ( "test/operations/dune:OCANNL_DASHED_ONLY_KEY",
      "the same fixture's second synthetic key, tracked for the same reason" );
    ( "test/operations/dune:ocannl-log_level",
      "the `config_var_fatal_spelling` fixture: a known key in the dashed spelling that \
       gh-ocannl-605 dropped, which since gh-ocannl-652 aborts the run rather than warning" );
  ]

(* Directories with runtest actions that carry no ambient gate, and why. Same shape as the
   declaration exemptions above: each is checked for still being needed. *)
let gateless_dirs =
  [
    ( "benchmarks/dune",
      "its one runtest action runs python3 over the benchmark orchestrator's own unit tests, which \
       import no OCANNL executable -- there is no startup check in reach to gate, the same reason \
       `config_dep_completeness` exempts it from the ocannl_config dependency" );
  ]

(* A gate is a stanza that depends on the state of the world: that is what makes dune rerun it
   rather than serve the previous run. Matched structurally rather than by the stanza's name, so a
   gate that is renamed or rewritten still counts. *)
let rec depends_on_universe = function
  | Sexp.List [ Sexp.Atom "universe" ] -> true
  | Sexp.List l -> List.exists l ~f:depends_on_universe
  | Sexp.Atom _ -> false

(* The lock a directory's actions share, where it has one. A gate added to such a directory has to
   take it too (Codex P1 round 4): not because the gate contends -- it links `arrayjit.utils` and
   starts no OpenMP pool -- but because the one unlocked action in a file of locked ones is what the
   next person copies when writing a real training test. Asked of the file rather than hard-coded
   per directory, so a directory that adopts the lock later brings its gate along. *)
let training_lock = "ocannl_training_test"

let rec takes_training_lock = function
  | Sexp.List (Sexp.Atom "locks" :: args) ->
      List.exists args ~f:(function Sexp.Atom a -> String.equal a training_lock | _ -> false)
  | Sexp.List l -> List.exists l ~f:takes_training_lock
  | Sexp.Atom _ -> false

(* The aliases a stanza's actions attach to: a rule's own `(alias …)` field -- not an `(alias …)`
   inside its `deps`, which is a dependency on another alias and is read by {!alias_deps} -- and
   `runtest` for a `(test)`/`(tests)` stanza, dune's shorthand for a runtest action that names no
   alias of its own. Asked per alias rather than per directory (Codex P1 round 3): a `(test)` stanza
   contributes to `runtest` alone, so a directory whose gate is a test stanza is ungated for `@slow`
   -- which is a separately documented entry point, and which `dune build @slow` reaches without
   building a single `(test)`. A gate on one alias must not vouch for another. *)
let aliases_of stanza =
  match stanza with
  | Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> [ "runtest" ]
  | Sexp.List (Sexp.Atom "rule" :: fields) ->
      (* Both spellings: `(alias A)` and dune's plural `(aliases A B)`, which is how a rule sits on
         `runtest` and on its own per-test alias at once (gh-ocannl-726). Reading only the singular
         would take those rules for attaching to nothing, and a rule attached to nothing is a rule
         no gate has to reach. *)
      List.concat_map fields ~f:(function
        | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom n ] -> [ n ]
        | Sexp.List (Sexp.Atom "aliases" :: args) ->
            List.filter_map args ~f:(function Sexp.Atom n -> Some n | _ -> None)
        | _ -> [])
  | _ -> []

(* The aliases a stanza's `deps` field names, which dune builds before it: before a rule's action
   runs, or whenever the alias an `(alias (name A) (deps …))` stanza defines is built. *)
let alias_deps stanza =
  match Scan.field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom n ] -> Some n
        | _ -> None)

(* The alias an `(alias (name A) …)` stanza defines: since dune 2.0 it carries no action of its own
   and only aggregates what its `deps` name. *)
let alias_stanza_name = function
  | Sexp.List (Sexp.Atom "alias" :: _) as stanza -> (
      match Scan.names_of stanza with [ n ] -> Some n | _ -> None)
  | _ -> None

(* A gate is a stanza with an action that depends on the state of the world. *)
let is_gate stanza = depends_on_universe stanza && not (List.is_empty (aliases_of stanza))

(* Every alias a build can start from: those rules and tests attach to, and those `(alias …)`
   stanzas define. *)
let entry_points stanzas =
  List.concat_map stanzas ~f:(fun s -> aliases_of s @ Option.to_list (alias_stanza_name s))
  |> List.dedup_and_sort ~compare:String.compare

(* The aliases `dune build @<alias>` runs a gate for: those a gate attaches to, closed under what
   building an alias builds -- the `deps` of every rule attached to it and of the `(alias …)` stanza
   defining it. A slow rule whose `deps` name the gate's alias is gated as much as one the gate sits
   beside, and runs it first. *)
let gated_aliases stanzas =
  let rec close gated =
    let next =
      List.fold stanzas ~init:gated ~f:(fun gated stanza ->
          if is_gate stanza || List.exists (alias_deps stanza) ~f:(Set.mem gated) then
            List.fold
              (aliases_of stanza @ Option.to_list (alias_stanza_name stanza))
              ~init:gated ~f:Set.add
          else gated)
    in
    if Set.equal next gated then gated else close next
  in
  close (Set.empty (module String))

(* The aliases building `@<alias>` reaches, through the same two routes -- what a suite
   aggregates. *)
let aliases_reached_from stanzas alias =
  let rec close reached =
    let next =
      List.fold stanzas ~init:reached ~f:(fun reached stanza ->
          let attached = aliases_of stanza @ Option.to_list (alias_stanza_name stanza) in
          if List.exists attached ~f:(Set.mem reached) then
            List.fold (alias_deps stanza) ~init:reached ~f:Set.add
          else reached)
    in
    if Set.equal next reached then reached else close next
  in
  close (Set.singleton (module String) alias)

(* The suite a per-test alias belongs to, by the naming convention: `<suite>-<name>` is one test and
   `<suite>` the suite, an `(alias (name <suite>) (deps …))` stanza listing its members -- so a
   member the list omits is one `dune build @<suite>` skips, silently. Three suites are built this
   way. `slow` is the one gh-ocannl-667 introduced. `train` carries the training integration runs --
   toy problems by intent, but serialized on the training lock, so on their own entry point (a macOS
   CI shard, a daily-sweep unit) rather than in `runtest` where they set every full run's wall-clock
   tail. `runtest` is the same arrangement for the golden-diff rules (gh-ocannl-726): dune generates
   `runtest-<name>` for `(test)`/`(tests)` stanzas and inline-test libraries only, so an
   `(executable)` plus a `(rule)` that diffs has to be given one -- and it has to be given it ALONE,
   since a rule attached to two aliases makes building either build both, which would put the whole
   directory behind every per-test alias. Aggregating the members here is then the only thing that
   keeps them in `dune runtest`. *)
let suites = [ "slow"; "train"; "runtest" ]
let member_of suite alias = String.is_prefix alias ~prefix:(suite ^ "-")

(* The repo-wide scans and the family alias that runs them (gh-ocannl-703). Which rules are in the
   family is DERIVED rather than listed here: a rule that recursively globs the repository, or takes
   its source tree for [Source_inventory], is reading the repository -- so a scan lands in the
   family the day it lands in the file, and this check cannot go stale against the stanza it is
   checking the way a second copy of the list would. *)
let scans_suite = "scans"

let rec inventories_repository = function
  | Sexp.List (Sexp.Atom ("glob_files_rec" | "source_tree") :: _) -> true
  | Sexp.List l -> List.exists l ~f:inventories_repository
  | Sexp.Atom _ -> false

let is_repo_wide_scan stanza =
  match stanza with
  | Sexp.List (Sexp.Atom "rule" :: _) ->
      Option.value_map (Scan.field stanza "deps") ~default:false ~f:(fun args ->
          List.exists args ~f:inventories_repository)
  | _ -> false

(* ... and the glob has to LEAVE the directory to be reading the repository (Codex P2, round 6). A
   test that recursively reads a fixture tree of its own uses the same form and is nobody's
   repo-wide scan; classifying it as one would demand a `scans` family beside it and fail the check
   until an unrelated suite appeared. *)
let rec escapes_directory = function
  | Sexp.List (Sexp.Atom ("glob_files_rec" | "source_tree") :: args) ->
      List.exists args ~f:(function
        | Sexp.Atom pattern -> String.is_prefix pattern ~prefix:"../"
        | _ -> false)
  | Sexp.List l -> List.exists l ~f:escapes_directory
  | Sexp.Atom _ -> false

let is_repo_wide_scan stanza =
  is_repo_wide_scan stanza
  && Option.value_map (Scan.field stanza "deps") ~default:false ~f:(fun args ->
      List.exists args ~f:escapes_directory)

(* A recursive glob is not the only corpus a repo-wide scan can have. One whose subject is a single
   file at the repository ROOT -- `CHANGES.md`, `AGENTS.md`, `.gitignore`, `ocannl_config.reference`
   -- reads the repository just as much: it is a rule in a test directory that answers for a
   document the whole repository shares, it goes stale on exactly the changes `@scans` exists to
   revalidate, and nothing else in the tree would run it. Left out of the derivation, such a scan
   sits in the family list by hand and drops out of it silently the day someone edits that list
   (Codex P2, round 5 on lukstafi/ocannl-staging#603).

   A root file is `../../<name>` with nothing further: `../../tools/sweep.sh` is a script a harness
   happens to drive, not a document the repository is about, and reading one says nothing about
   whether the rule scans anything. *)
let rec reads_repository_root_file = function
  | Sexp.Atom path ->
      let path = String.substr_replace_all path ~pattern:"\\" ~with_:"/" in
      let path =
        match String.chop_prefix path ~prefix:"%{dep:" with
        | Some rest -> String.chop_suffix_if_exists rest ~suffix:"}"
        | None -> path
      in
      Option.value_map (String.chop_prefix path ~prefix:"../../") ~default:false ~f:(fun name ->
          (not (String.is_empty name))
          && (not (String.contains name '/'))
          && not (String.contains name '*'))
  | Sexp.List (Sexp.Atom "glob_files_rec" :: _) -> false
  | Sexp.List l -> List.exists l ~f:reads_repository_root_file

let is_repo_wide_scan stanza =
  is_repo_wide_scan stanza
  ||
  match stanza with
  | Sexp.List (Sexp.Atom "rule" :: _) ->
      Option.value_map (Scan.field stanza "deps") ~default:false ~f:(fun args ->
          List.exists args ~f:reads_repository_root_file)
  | _ -> false

(* The focused aggregates beyond `scans` (gh-ocannl-783), and the same completeness question asked
   of them. A family is an `(alias (name <family>) (deps …))` stanza per directory, aggregating the
   per-test aliases of the tests that belong to it, so that `dune build @<family>` from the
   workspace root runs that class of test wherever it lives; a member the stanza omits is one the
   family skips, silently, which is exactly the failure gh-ocannl-703 closed for `scans`.

   Membership is DERIVED from what the member stanza itself declares -- never from a second copy of
   the list here, which could only confirm that the copy still says what it says. The derivation is
   a FLOOR: a family may list more (see `test_slab_free_on_grow` in `arrayjit/test/dune`), because
   what this check is for is the member that silently falls out, not the member someone chose to
   include. *)
type family = {
  family_alias : string;  (** the alias, spelled identically in every directory with members *)
  family_is : string;  (** what makes a stanza a member, for the diagnostic *)
  family_floor : int;
      (** how few members its own derivation may find across the repository before the derivation is
          taken to have stopped working. PER FAMILY, not one floor over the union: five healthy
          Metal members would otherwise satisfy a shared floor while the lifecycle derivation found
          nothing at all, and "every member is aggregated" would hold of the empty family exactly as
          loudly as of a complete one (Codex P2, round 3). Set well under the members there are, so
          that it says the derivation is about something rather than counting what it finds. *)
}

let metal_family =
  {
    family_alias = "metal-codegen";
    family_is =
      "names `metal` as its backend, so no other box can judge it -- the executed Metal-only \
       guards and the emitted-MSL structural tests";
    (* Five on 2026-08-27, in two directories. *)
    family_floor = 3;
  }

(* The library modules that exist to make resource lifetimes observable: a test that refers to one
   is asking about allocation, free or context lifetime across a cleanup seam, which is what this
   family collects. Derived from the instrumentation rather than from the test's name or directory,
   for the reason the whole arrangement is derived -- a probe added tomorrow is asked about the day
   it lands.

   QUALIFIED, because a bare module name carries no provenance: a local or third-party
   `Alloc_census` is not this one, and putting its user in the family would demand a focused alias
   for a test that touches no instrumentation (Codex P2, round 5). `Ir` is how every test outside
   the implementation reaches these -- `open Ocannl.Operation.DSL_modules` binds it -- and a test
   that aliases the path (`module AC = Ir.Alloc_census`) names it in the binding, which is what the
   derivation sees. *)
let lifecycle_modules = [ [ "Ir"; "Resource_fault_injection" ]; [ "Ir"; "Alloc_census" ] ]
let lifecycle_module_names = List.map lifecycle_modules ~f:(String.concat ~sep:".")

(* What a source has to SPELL for any reference to reach the instrumentation: the module's own name.
   The qualifier is not part of that -- `open Ir` then `Alloc_census.snapshot` writes a real
   reference and no `Ir.Alloc_census` anywhere (Codex P2, round 6) -- so the textual filter narrows
   on the last component alone and the parse decides. *)
let lifecycle_module_leaves = List.filter_map lifecycle_modules ~f:(fun path -> List.last path)

let lifecycle_family =
  {
    family_alias = "lifecycle";
    family_is =
      "has a module that reads the resource-lifecycle instrumentation ("
      ^ String.concat ~sep:", " lifecycle_module_names
      ^ "), so it answers for allocation, free or context lifetime across a cleanup seam";
    (* Two on 2026-08-27. One is the floor that matters -- the failure it guards against is the
       derivation finding NOTHING, and a floor equal to today's count would fail the day a probe is
       retired, which is the tally gh-ocannl-665 took out of the sibling goldens. *)
    family_floor = 1;
  }

let families = [ metal_family; lifecycle_family ]

(* A family member is not a STANZA but one of the things a stanza builds: `(tests (names a b))` is
   two tests with an alias each, and `(executables (names a b))` two executables with their own
   runners. Reading a stanza as one member accepts a family that reaches half of it, and asks a
   family to reach the half that is no member at all (Codex P2, round 2). So the unit of this whole
   check is the named test/executable/inline-test library -- or, for a rule, the rule itself, which
   is what an `(executable)`'s backend marker sits on. *)
type member_unit = {
  unit_subdir : string;  (** the `(subdir …)` it sits under, relative to the dune file *)
  unit_stanza : Sexp.t;
  unit_name : string option;  (** the name dune builds it under, where it has one *)
  unit_identity : string;  (** how the diagnostics and the report name it *)
  unit_aliases : (string * string) list;
      (** the aliases that reach THIS unit, each with the DIRECTORY it is defined in -- an alias is
          per directory, and the rule that runs an executable declared in `(subdir child …)` may
          perfectly well sit at the top level, where the family stanza aggregating it sits too
          (Codex P2, round 3). Any one of them will do: two rules running one executable are two
          ways of running it *)
}

(* Path arithmetic, so that a runner is credited to the executable it actually runs. `(chdir other
   (run ./probe.exe))` runs `other/probe.exe`, and a comparison that dropped either the cwd or the
   command's own directory would credit the LOCAL `probe` with it (Codex P2, rounds 1 and 2). *)
let normalize_path path =
  (* An absolute path keeps its root: `/probe.exe` is not this directory's `probe.exe`, and dropping
     the leading empty component made the two one identity (Codex P2, round 13). *)
  let root = if String.is_prefix path ~prefix:"/" then "/" else "" in
  root
  ^ (String.split path ~on:'/'
    |> List.fold ~init:[] ~f:(fun acc component ->
        match component with
        | "" | "." -> acc
        | ".." -> (
            match acc with
            | above :: rest when not (String.equal above "..") -> rest
            | _ -> ".." :: acc)
        | component -> component :: acc)
    |> List.rev |> String.concat ~sep:"/")

(* An executable's identities: the file dune builds, and the public name it installs under. A
   companion rule may run either -- `%{dep:probe.exe}` or `%{bin:pkg.probe}`, which
   `Scan.executables_run` reports as `Runs "probe.exe"` and `Runs "pkg.probe"` -- and accepting only
   the first leaves a correctly aggregated family reported as incomplete (Codex P2, round 3). The
   same pair `Scan.artifact_subjects` matches its runners on.

   The two are matched DIFFERENTLY, and it matters: the file is compared after resolving the
   action's cwd against it, so `(chdir nested (run ./probe.exe))` is another directory's file; the
   public name is a workspace-wide identifier that no cwd relocates, so it is compared as written. A
   fallback that compared the raw command against the file too would undo the cwd fix of round 2. *)
let executable_identities stanza ~subdir ~name =
  let declared =
    List.concat_map [ "public_name"; "public_names" ] ~f:(fun field ->
        match Scan.field stanza field with
        | None -> []
        | Some args -> List.filter_map args ~f:(function Sexp.Atom a -> Some a | _ -> None))
  in
  (* POSITIONALLY, which is how dune pairs them: `(executables (names a b) (public_names pa pb))`
     installs `a` as `pa`, and handing every unit the whole list would make a rule running `pa` a
     runner for `b` too (Codex P2, round 4). `-` is dune's placeholder for a name that is not
     installed, and a list that does not line up yields no public name at all -- the fail-closed
     direction, which reports rather than credits. *)
  let public =
    match List.findi (Scan.names_of stanza) ~f:(fun _ n -> String.equal n name) with
    | None -> []
    | Some (index, _) -> (
        match List.nth declared index with
        | Some public when not (String.equal public "-") -> [ public ]
        | _ -> [])
  in
  (normalize_path (Scan.in_subdir subdir (name ^ ".exe")), public)

(* The aliases of the rules that run one of those identities, WITH the directory each rule sits in.
   Searched over the whole dune file rather than one `(subdir …)` group: an executable declared in
   `(subdir child …)` is perfectly well run by a top-level rule naming `child/probe.exe`, and it is
   that rule's own directory whose family alias has to aggregate it (Codex P2, round 3). An
   `(executable)` has no alias of its own, so its runner's is the one a family lists -- the same
   placement the `ocannl_config` dep and the backend marker take. A command this comparison declines
   leaves the unit unaggregated, which is reported: the direction that asks the author to say what
   they meant rather than passing on a coincidence of names. *)
let runner_aliases file_stanzas ~identities:(file, public) =
  List.concat_map file_stanzas ~f:(fun (runner_subdir, stanza) ->
      if
        List.exists (Scan.executables_run stanza) ~f:(fun (cwd, command) ->
            match command with
            | Scan.Runs path ->
                let resolved =
                  normalize_path (Scan.in_subdir runner_subdir (Scan.in_subdir cwd path))
                in
                String.equal resolved file
            (* And a public name only where the command RESOLVED one. `(run ./pkg.probe)` and `(run
               %{bin:pkg.probe})` carry the same string and name different things -- a file here, an
               installed program -- so reading the first as a public-name runner would credit an
               unrelated rule with this executable (Codex P2, round 8). *)
            | Scan.Runs_public name -> List.mem public name ~equal:String.equal
            | _ -> false)
      then
        (* A `(test)` with a custom action can be the runner, and its focused entry point is the
           `runtest-<name>` dune generates -- `aliases_of` reports the directory-wide `runtest` for
           it, which is filtered out as a suite alias and would leave the executable reachable by
           nothing (Codex P2, round 15). *)
        let attached =
          match Scan.head stanza with
          | Some ("test" | "tests") ->
              List.map (Scan.names_of stanza) ~f:(fun name -> "runtest-" ^ name)
          | _ -> aliases_of stanza
        in
        List.map attached ~f:(fun alias -> (runner_subdir, alias))
      else [])

(* The units a stanza contributes, with the aliases that reach each. A directory-wide suite alias is
   never one of them: a family whose stanza depended on `(alias runtest)` would run the whole
   directory, which is precisely the run these aggregates exist to avoid -- and an arbitrary marked
   rule attached only to `runtest` would otherwise offer exactly that as its member alias (Codex P2,
   round 3). Such a rule is reported until it is given a dedicated alias. *)
let family_units file_stanzas ~subdir stanza =
  let head = Option.value (Scan.head stanza) ~default:"<not a stanza>" in
  let focused aliases =
    List.filter aliases ~f:(fun (_, alias) -> not (List.mem suites alias ~equal:String.equal))
  in
  let unit ?name aliases =
    {
      unit_subdir = subdir;
      unit_stanza = stanza;
      unit_name = name;
      unit_identity = Printf.sprintf "%s %s" head (Option.value name ~default:"<unnamed>");
      unit_aliases = focused aliases;
    }
  in
  let generated name = unit ~name [ (subdir, "runtest-" ^ name) ] in
  match Scan.head stanza with
  (* dune >= 3.20 generates `runtest-<name>` per `(test)`/`(tests)` name AND per inline-test library
     -- the namespace `generated_runtest_names` already knows. A Metal-marked inline-test library
     reaches its family through exactly that alias (Codex P2, round 2). *)
  | Some ("test" | "tests") -> List.map (Scan.names_of stanza) ~f:generated
  | Some "library" when Option.is_some (Scan.field stanza "inline_tests") ->
      List.map (Scan.names_of stanza) ~f:generated
  | Some ("executable" | "executables") ->
      List.map (Scan.names_of stanza) ~f:(fun name ->
          unit ~name
            (runner_aliases file_stanzas ~identities:(executable_identities stanza ~subdir ~name)))
  | Some _ -> [ unit (List.map (aliases_of stanza) ~f:(fun alias -> (subdir, alias))) ]
  | None -> []

(* The stanza kinds whose units can be asked whether their own modules read the lifecycle
   instrumentation. A plain `(library)` whose modules read it is the instrumentation's own
   implementation or a shared helper, not a probe anything runs; an inline-test library is a probe,
   and is included for that reason. *)
let has_family_modules stanza =
  match Scan.head stanza with
  | Some ("test" | "tests" | "executable" | "executables") -> true
  | Some "library" -> Option.is_some (Scan.field stanza "inline_tests")
  | _ -> false

(* Dune's named dependencies: `(deps (:golden foo.expected))` binds `%{golden}` to that path. A
   pform naming one carries no colon, so without the binding `golden_stem` would take the BINDING's
   name for the golden's -- rejecting the alias a reader would write and accepting one that names
   nothing (Codex P2, round 6). *)
let named_deps stanza =
  match Scan.field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List (Sexp.Atom name :: Sexp.Atom path :: _) when String.is_prefix name ~prefix:":"
          ->
            Some (String.drop_prefix name 1, path)
        | _ -> None)

(* A file NAMED by a pform -- `%{dep:verdict_ratchet.expected}`, dune's ordinary way of writing a
   dependency inline, or `%{golden}` for a named one -- still names a file. Unwrap the leading
   pform: what follows the last `:` inside the braces is the path, and a pform with no colon is a
   named dependency, resolvable only from the stanza that bound it (Codex P2, rounds 4 and 6).
   Shared by the alias check and by the scans family's target matching (round 7), so the two agree
   about what a name is. *)
let unwrap_pform ?(named = []) file =
  match String.chop_prefix file ~prefix:"%{" with
  | None -> file
  | Some rest -> (
      match String.substr_index rest ~pattern:"}" with
      | None -> rest
      | Some close ->
          let inside = String.prefix rest close in
          let after = String.drop_prefix rest (close + 1) in
          let path =
            match String.rindex inside ':' with
            | Some colon -> String.drop_prefix inside (colon + 1)
            (* An unbound named dependency resolves to nothing, which the callers refuse by name
               rather than guessing. *)
            | None -> Option.value (List.Assoc.find named inside ~equal:String.equal) ~default:""
          in
          path ^ after)

(* What a rule writes, so that the rule DIFFING it can be found: a scan produces `<name>.actual` and
   a second rule holds it against the golden. It is that second rule the family alias has to
   aggregate, since it is the one that fails when the scan reports something. *)
let targets_of stanza =
  List.concat_map [ "target"; "targets" ] ~f:(fun field ->
      match Scan.field stanza field with
      | None -> []
      | Some args -> List.filter_map args ~f:(function Sexp.Atom a -> Some a | _ -> None))

let rec mentions_atom text = function
  | Sexp.Atom a -> String.equal a text
  | Sexp.List items -> List.exists items ~f:(mentions_atom text)

(* A target a rule hands straight back to its OWN action is an input it wrote for itself, not a
   verdict anybody diffs. The one shape this covers is the response file a repo-wide scan writes to
   get its file list off the command line: dune's `echo` spawns nothing and so has no length limit,
   while `CreateProcess` on Windows caps a whole command line at 32,767 characters, which three of
   these scans had crossed (`test/support/scan_argv.ml`). Recognized by the `@<target>` REFERENCE in
   the rule's own action rather than by the target's name, so a rule cannot opt a real output out of
   the golden-diff requirement by calling it something. *)
let is_own_response_file stanza target =
  Option.value_map (Scan.field stanza "action") ~default:false ~f:(fun args ->
      List.exists args ~f:(mentions_atom ("@" ^ target)))

let rec diffs_file ?(named = []) target = function
  | Sexp.List (Sexp.Atom ("diff" | "diff?") :: args) ->
      (* Modulo the pform spelling: `(diff foo.expected %{dep:foo.actual})` is dune's ordinary way
         of naming the dependency, and comparing it literally against the producer's `foo.actual`
         would report a correctly aggregated scan as missing (Codex P2, round 7). *)
      List.exists args ~f:(function
        | Sexp.Atom a -> String.equal (unwrap_pform ~named a) target
        | _ -> false)
  | Sexp.List l -> List.exists l ~f:(diffs_file ~named target)
  | Sexp.Atom _ -> false

(* A rule whose action holds a golden against a run's output: the shape that has no alias of its own
   unless someone writes one, since dune generates `runtest-<name>` for `(test)`/`(tests)` stanzas
   and for inline-test libraries, and for nothing else (gh-ocannl-726). *)
let rec is_diff_action = function
  | Sexp.List (Sexp.Atom ("diff" | "diff?") :: _) -> true
  | Sexp.List l -> List.exists l ~f:is_diff_action
  | Sexp.Atom _ -> false

(* The golden a diff rule holds a run's output against: the first operand of its `diff`. What the
   alias is checked against below -- an alias whose name has nothing to do with its golden is an
   entry point nobody constructs, since what a reader has in hand is the golden that just failed
   (Codex P2, round 3, after `runtest-n3_fwd_with_prec` had shortened away the `-unoptimized` its
   golden carries). *)
let rec goldens_in = function
  | Sexp.List (Sexp.Atom ("diff" | "diff?") :: Sexp.Atom golden :: _) -> [ golden ]
  | Sexp.List l -> List.concat_map l ~f:goldens_in
  | Sexp.Atom _ -> []

(* The part of a golden's name a reader would type: everything before the first dune pform or
   extension, less a trailing `_expected` and any separator left dangling. So
   `verdict_ratchet.expected` is `verdict_ratchet`, `n3_fwd_with_prec-unoptimized.ll.expected` is
   `n3_fwd_with_prec-unoptimized`, `top_down_prec.%{read:…}.expected` is `top_down_prec`,
   `micrograd_demo_logging-%{read:…}-0-0.log.expected` is `micrograd_demo_logging`, and the ppx
   convention's `test_ppx_op_expected.ml` is `test_ppx_op`. The alias may go on to say WHICH golden
   of a subject it checks -- `-extension`, `-unoptimized`, `-ppx` -- which is why the relation asked
   for is a prefix rather than equality: one run can write several goldens, and each needs an alias
   of its own. *)
let golden_stem ?(named = []) golden =
  let cut_at s ~on =
    match String.substr_index s ~pattern:on with None -> s | Some i -> String.prefix s i
  in
  let golden = unwrap_pform ~named golden in
  (* A pform INSIDE the name goes the other way, and is cut before the basename is taken: a
     `%{read:config/…}` carries a path, so taking the basename first would leave the CONFIG file's
     name as the stem. *)
  let stem = cut_at (Stdlib.Filename.basename (cut_at golden ~on:"%{")) ~on:"." in
  let stem = Option.value (String.chop_suffix stem ~suffix:"_expected") ~default:stem in
  String.rstrip stem ~drop:(fun c -> Char.equal c '-' || Char.equal c '_')

let is_golden_diff stanza =
  match stanza with
  | Sexp.List (Sexp.Atom "rule" :: _) ->
      Option.value_map (Scan.field stanza "action") ~default:false ~f:(fun args ->
          List.exists args ~f:is_diff_action)
  | _ -> false

(* The names dune generates a `runtest-<name>` alias for by itself (>= 3.20): every `(test)`/
   `(tests)` stanza name, and every inline-test library's name. A hand-written alias must not reuse
   one (Codex P2, round 5): the two aliases MERGE, so the targeted entry point runs that test as
   well as the rule, which is the isolation this whole arrangement is for -- and once the rule also
   names `runtest`, dune calls the pair a dependency cycle. The `runtest-env_spelling_gate` rule is
   the deliberate exception, and is recognized structurally rather than by name: it is the ambient
   gate, whose whole purpose is to run the same binary the `(test)` stanza does. *)
let is_test_stanza = function Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> true | _ -> false

(* The generated names that belong to an ambient gate: a `(test)` stanza depending on `(universe)`
   is the gate, so a rule sharing ITS alias runs the same gate binary, which is the one deliberate
   collision. Recognized this way rather than by the literal name `env_spelling_gate` (which a
   rename would silently unexempt) and rather than by the rule alone (which let any
   universe-dependent rule claim the exemption -- Codex P2, rounds 6 and 7). *)
let gate_generated_names stanzas =
  List.concat_map stanzas ~f:(fun stanza ->
      if is_test_stanza stanza && depends_on_universe stanza then Scan.names_of stanza else [])
  |> Set.of_list (module String)

let generated_runtest_names stanzas =
  List.concat_map stanzas ~f:(fun stanza ->
      match stanza with
      | Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> Scan.names_of stanza
      | Sexp.List (Sexp.Atom "library" :: _) when Option.is_some (Scan.field stanza "inline_tests")
        ->
          Scan.names_of stanza
      | _ -> [])
  |> Set.of_list (module String)

(* The prefix `Utils.classify_env_var` reports for a per-module tracing gate. *)
let gate_prefix = "ocannl_log_level_"

(* A lower bound on how many sources call `Test_utils.Generated.init` (gh-ocannl-723). The rule
   below is a relationship between two answers, and if the source-side answer silently became "none"
   -- a ppxlib upgrade, a rename of the module, a glob that stopped reaching the test sources -- the
   relationship would hold vacuously over an empty set and the check would go green having stopped
   checking. A floor rather than a count, for the reason gh-ocannl-665 took the counts out of the
   sibling goldens: it must not move when a test is added or removed. There were 37 on
   2026-08-23. *)
let artifact_caller_floor = 20

(* The configuration key `OCANNL_BUILD_FILES_PREFIX` addresses, which is what a module reading it by
   name reads. *)
let artifact_config_key = "build_files_prefix"

(* The stanza kinds that name their own modules, and so can be asked what those modules read. *)
let module_stanzas = [ "library"; "test"; "tests"; "executable"; "executables" ]

(* The module that DEFINES the environment reader, and so names every configuration key there is
   while passing them to it. Every other library module reaching the reader is reported
   (gh-ocannl-749, Codex P2 round 6): a guard in a plain library is run by whatever links it, which
   puts the declaration on every such stanza -- a relationship nothing follows, and the same
   argument `Artifact_in_library` makes for the initializer. Named rather than derived, being one
   file with one reason -- and by its repository PATH, since a basename match would extend the
   exemption to any `utils.ml` a test directory adds, silently skipping its reads (Codex P2, round
   8). *)
let env_reader_home = "arrayjit/lib/utils.ml"

(* Dune resolves aliases, module defaults, and action paths within the directory a stanza is applied
   to. Keep that descent in one place: checks plug into [checks], and this iterator hands every one
   the same [(subdir, stanzas)] groups. A new check therefore starts per-directory instead of having
   to rediscover [(subdir ...)] traversal for itself (gh-ocannl-813). *)
let per_directory file_stanzas ~checks =
  List.stable_sort file_stanzas ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.group ~break:(fun (a, _) (b, _) -> not (String.equal a b))
  |> List.iter ~f:(fun group ->
      let subdir = fst (List.hd_exn group) in
      let stanzas = List.map group ~f:snd in
      List.iter checks ~f:(fun check -> check subdir stanzas))

(* A control of the iteration seam itself. The callback stands for a check added after the refactor:
   it contains no descent, yet receives the root and both nested directories from
   [per_directory]. *)
let per_directory_control () =
  let stanzas =
    Scan.stanzas
      {dune|(rule (target root))
(subdir child
 (rule (target child))
 (subdir grandchild
  (rule (target grandchild))))
|dune}
  in
  let file_stanzas = Scan.walk "" stanzas ~f:(fun subdir stanza -> [ (subdir, stanza) ]) in
  let seen = ref [] in
  let new_check subdir _stanzas = seen := subdir :: !seen in
  per_directory file_stanzas ~checks:[ new_check ];
  printf
    "The per-directory seam is put to a root stanza and two nested `(subdir ...)` groups. The\n\
     checker plugged into it contains no traversal of its own.\n\n";
  Verdict.p
    "a newly plugged-in check receives the root and every nested directory without adding descent"
    (List.equal String.equal (List.rev !seen) [ ""; "child"; "child/grandchild" ]);
  let one_stanza source = List.hd_exn (Scan.stanzas source) in
  Verdict.p "an escaping source_tree dependency identifies a repository-wide scan"
    (is_repo_wide_scan (one_stanza "(rule (deps (source_tree ../..)) (action (run scan.exe)))"));
  Verdict.p_none "a local source_tree fixture does not identify a repository-wide scan"
    [ one_stanza "(rule (deps (source_tree fixture)) (action (run scan.exe)))" ]
    ~f:is_repo_wide_scan;
  printf "\n"

(* gh-ocannl-800's own negative control. The repository is normally complete, so an empty orphan
   list there cannot tell the live relationship from one that stopped deciding anything. Put the
   relationship to one extracted refusal and two synthetic control corpora instead. *)
let refusal_control () =
  let source =
    {ocaml|let fail = Verdict.fail
let refuse name =
  fail
    (Printf.sprintf
       "%s: appears in no permanent control golden -- add a negative control"
       name)
|ocaml}
  in
  let diagnostics = Refusals.diagnostics source in
  let absent =
    Refusals.orphans ~control_text:"a legitimate control about something else" diagnostics
  in
  let covered =
    Refusals.orphans ~control_text:(Refusals.marker (List.hd_exn diagnostics)) diagnostics
  in
  let duplicate =
    let diagnostic = List.hd_exn diagnostics in
    Refusals.orphans ~control_text:(Refusals.marker diagnostic) [ diagnostic; diagnostic ]
  in
  printf
    "The refusal relationship is put to a synthesized `Verdict.fail` format. Its stable fragment\n\
     appears in no permanent control golden in the negative arm, and appears in the positive arm.\n\n";
  Verdict.p "a diagnostic absent from every control golden is an orphan" (List.length absent = 1);
  Verdict.p_empty "the same diagnostic fragment in a control golden is covered" ~over:diagnostics
    covered;
  Verdict.p "one control marker occurrence covers only one identical diagnostic"
    (List.length duplicate = 1);
  Verdict.p "a scanner absent from the manifest is detected by the population equality"
    (not (List.equal String.equal [ "scanner_a.ml" ] [ "scanner_a.ml"; "scanner_b.ml" ]));
  let same_basename = [ "a/scanner.ml"; "b/scanner.ml" ] in
  Verdict.p "repo-relative scanner paths keep equal basenames distinct"
    (Set.length (Set.of_list (module String) same_basename) = 2
    && Set.length (Set.of_list (module String) (List.map same_basename ~f:Stdlib.Filename.basename))
       = 1);
  printf "\n"

let main () =
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune file or .ml source...>\n" argv.(0);
    Stdlib.exit 1);
  let base = Scan.base_dir argv.(1) in
  let paths =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let dune_files =
    List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "dune")
  in
  (* Through `Sources.sources_among`, the same filter the sibling scans apply: dune's globs run over
     the BUILD tree, where a preprocessed `<name>.pp.ml` sits beside every ppx-using `<name>.ml`.
     Reading both would double the census, and a `.pp.ml` is not OCaml the compiler's own parser
     accepts -- it carries the ppx's output verbatim -- so the pair has to be resolved to the
     source, not merely deduplicated. *)
  let source_files =
    let all = List.filter paths ~f:(fun (path, _) -> String.is_suffix path ~suffix:".ml") in
    let kept = Set.of_list (module String) (Sources.sources_among (List.map all ~f:fst)) in
    List.filter all ~f:(fun (path, _) -> Set.mem kept path)
  in
  let expected_files =
    List.filter paths ~f:(fun (path, _) -> String.is_suffix path ~suffix:".expected")
  in
  let control_goldens =
    List.filter expected_files ~f:(fun (path, on_disk) ->
        (not (String.is_suffix path ~suffix:"refusal_control_scan_cases.expected"))
        && (String.is_suffix path ~suffix:"_cases.expected"
           || String.is_suffix path ~suffix:"_control.expected"
           || String.is_substring (In_channel.read_all on_disk) ~substring:"Synthetic controls:"))
  in
  let control_text =
    String.concat ~sep:"\n"
      (List.map control_goldens ~f:(fun (_, path) -> In_channel.read_all path))
  in
  let sources =
    List.map source_files ~f:(fun (path, on_disk) -> (String.lowercase path, on_disk))
  in
  if List.is_empty dune_files || List.is_empty sources then (
    Verdict.fail "no dune files or no sources among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  (* Directories whose sources this scan was handed. A library elsewhere cannot have its gates
     checked, and says so rather than passing for lack of evidence. *)
  let scanned_dirs =
    List.map sources ~f:(fun (path, _) -> Stdlib.Filename.dirname path)
    |> Set.of_list (module String)
  in
  let source_of ~dir module_name =
    List.Assoc.find sources ~equal:String.equal
      (String.lowercase (Scan.in_subdir dir (module_name ^ ".ml")))
  in
  (* gh-ocannl-783: every source that REFERS TO one of the resource-lifecycle instrumentation
     modules, keyed the way `source_of` looks a module up. What makes a test a lifecycle probe is
     that it asks the instrumentation something -- and reading that off the text would have made
     this very file a probe, since it has to spell the module names in order to look for them. *)
  let lifecycle_sources =
    List.filter_map source_files ~f:(fun (path, on_disk) ->
        let content = In_channel.read_all on_disk in
        (* Narrowed textually first -- the module has to be NAMED for any reference to reach it --
           and then PARSED, because a doc comment, a string literal or a longer identifier names it
           without reading it, and a family membership derived from a substring would demand a
           focused alias for a test that never touches the instrumentation (Codex P2, round 4). *)
        if
          not
            (List.exists lifecycle_module_leaves ~f:(fun m ->
                 String.is_substring content ~substring:m))
        then None
        else
          match Sources.module_references_in_source content ~paths:lifecycle_modules with
          | [] -> None
          | _ :: _ -> Some (String.lowercase path)
          | exception exn ->
              (* A source this scan cannot read is one it cannot answer for, and answering "no
                 references" for it would be the silent failure the whole check is against. *)
              Verdict.fail
                (Printf.sprintf
                   "%s names the resource-lifecycle instrumentation and does not parse, so whether \
                    it reads it cannot be established: %s"
                   path (Exn.to_string exn));
              None)
    |> Set.of_list (module String)
  in
  (* gh-ocannl-723: every source that calls `Test_utils.Generated.init`, keyed the way `source_of`
     looks a module up, so that a stanza's `(modules …)` field answers for its own sources. Narrowed
     textually before parsing -- the module has to be NAMED for any spelling of the call to reach it
     -- so the census costs a substring search over the repository and a parse of the few dozen
     files that could contain one. *)
  let artifact_callers =
    List.filter_map source_files ~f:(fun (path, on_disk) ->
        let content = In_channel.read_all on_disk in
        if not (Sources.could_call_generated_init content) then None
        else
          match Sources.generated_init_calls_in_source content with
          | [] -> None
          | _ :: _ -> Some path
          | exception exn ->
              (* A source this scan cannot read is one it cannot answer for, and answering "no
                 calls" for it would be the silent failure the whole check is against. *)
              Verdict.fail
                (Printf.sprintf
                   "%s names `Test_utils.Generated` and does not parse, so whether it calls the \
                    initializer cannot be established: %s"
                   path (Exn.to_string exn));
              None)
  in
  let artifact_caller_keys =
    Set.of_list (module String) (List.map artifact_callers ~f:String.lowercase)
  in
  (* Whether this run was handed the repository, established the way the sibling scans establish it:
     every scan root the globs are written for contributed its floor of sources. The relationship
     below is about whatever tree is in front of the scan and is checked either way; the CENSUS
     floor is a statement about the repository, so it is asked only of a run that has it. Which mode
     a run was in goes into the golden, so a glob that breaks flips that line rather than quietly
     retiring the floor. *)
  let repository_census = List.is_empty (Sources.floor_violations (List.map source_files ~f:fst)) in
  let artifact_claimed = ref (Set.empty (module String)) in
  let artifact_violations = ref 0 in
  (* One line per subject for stderr, and per dune file for the golden: what the golden holds is
     that a file still declares the variable where its modules call the initializer, not how many
     stanzas do -- the gh-ocannl-665 argument, since a count moves whenever a test is added. *)
  let artifact_table = ref [] in
  let artifact_by_file = ref [] in
  let exemptions = Map.of_alist_exn (module String) exempt_declarations in
  let exemptions_used = ref (Set.empty (module String)) in
  let gateless = Map.of_alist_exn (module String) gateless_dirs in
  let gateless_used = ref (Set.empty (module String)) in
  let gated = ref [] in
  (* The gh-ocannl-659 half: one line per stanza that runs an executable, and the per-file summary
     the golden holds. *)
  let classification = ref [] in
  let by_file = ref [] in
  (* gh-ocannl-783: one entry per derived family member -- the file, the family, the stanza, and
     whether that family's alias reaches it. The claim below is quantified over this population, so
     a derivation that stopped finding members reports an empty family rather than passing. *)
  let family_table = ref [] in
  let placed_subjects = ref 0 in
  let subject_floor = ref 0 in
  (* The two readers' POPULATIONS, each stanza by the file and line it opens at, so that the
     relationship checked below is "the same stanzas" and not "as many stanzas". Two totals compared
     as numbers say nothing about WHICH stanza either reader is alone on, and a gap of one absorbs a
     different stanza dropping out of enforcement (Codex P2, round 2), which is why gh-ocannl-690
     itemised the gap on stderr rather than leaving it as arithmetic. Identities rather than counts
     is also what keeps this off the churn treadmill gh-ocannl-701 took the scans off: a test added
     anywhere moves both lists together. *)
  let walk_places = ref [] in
  let floor_names = ref [] in
  (* Kept apart on purpose: a stanza declaring neither is the hole gh-ocannl-659 is about, while a
     marker the scan could not place is the scan going blind to it -- and a claim that conflated the
     two would pass while the second was true. *)
  let xor_violations = ref 0 in
  let marker_holes = ref 0 in
  let tracked_keys = ref (Set.empty (module String)) in
  let gate_table = ref [] in
  let read_table = ref [] in
  let guard_table = ref [] in
  (* The scanner sources are derived from the same repo-wide rule property that makes a rule a
     member of [@scans], but their directory descent comes from [per_directory] below. A new scan
     therefore joins the diagnostic check with no source list to update. *)
  let scanner_sources = ref (Set.empty (module String)) in
  (* Every `(env_var ...)` under a sexp, at any depth. *)
  let rec env_vars_in = function
    | Sexp.List [ Sexp.Atom "env_var"; Sexp.Atom name ] -> [ name ]
    | Sexp.List l -> List.concat_map l ~f:env_vars_in
    | Sexp.Atom _ -> []
  in
  (* The dependency fields, at any depth -- `(deps ...)` of a rule inside a `(subdir ...)` as much
     as of a top-level test, and `(preprocessor_deps ...)` of a library. Not recursing INTO one
     keeps the count below a partition of the file's declarations rather than a double count. *)
  let rec dep_fields = function
    | Sexp.List (Sexp.Atom (("deps" | "preprocessor_deps") as field) :: args) -> [ (field, args) ]
    | Sexp.List l -> List.concat_map l ~f:dep_fields
    | Sexp.Atom _ -> []
  in
  List.iter dune_files ~f:(fun (dune_file, on_disk) ->
      let dir = match Stdlib.Filename.dirname dune_file with "." -> "" | dir -> dir in
      let content = In_channel.read_all on_disk in
      let stanzas = Scan.stanzas content in
      (* gh-ocannl-659: the exclusive or, over every stanza that runs an executable. *)
      let marker_contract = Scan.backend_marker_contract content in
      let marked =
        List.map marker_contract.Scan.contract_stanzas ~f:(fun m -> m.Scan.marker_stanza)
      in
      let words = ref (Set.empty (module String)) in
      let any_declared = ref false in
      let what_of_stanza stanza =
        (* A `(rule ...)` has no `(name ...)`, so it is named by what it runs -- which is what the
           reader has to go and look at anyway. *)
        if not (String.is_empty stanza.Scan.marked_name) then stanza.Scan.marked_name
        else
          match
            List.map stanza.Scan.marked_sites ~f:(fun s -> s.Scan.name)
            |> List.dedup_and_sort ~compare:String.compare
          with
          | [] -> "<unnamed>"
          | names -> "running " ^ String.concat ~sep:", " names
      in
      let where_of_stanza stanza =
        Printf.sprintf "%s:%d, the %s %s" dune_file stanza.Scan.marked_line stanza.Scan.marked_head
          (what_of_stanza stanza)
      in
      List.iter marker_contract.Scan.contract_issues ~f:(function
        | Scan.Malformed_marker
            { issue_line = line; issue_text = text; issue_malformed = malformed } ->
            let why =
              Scan.marker_malformed_reason ~sentinel:Scan.marker_sentinel
                ~separator_subject:"backend"
                ~grammar:
                  (Printf.sprintf "%s <%s> -- <reason>" Scan.marker_sentinel
                     (String.concat ~sep:"|" Scan.marker_backends))
                malformed
            in
            Int.incr xor_violations;
            fail
              (Printf.sprintf
                 "%s:%d has a `%s` comment that does not parse as a marker: %s. The line reads \
                  `;%s`"
                 dune_file line Scan.marker_sentinel why text)
        | Scan.Marker_in_wrong_stanza { issue_line = line; issue_stanza = stanza; _ } ->
            Int.incr xor_violations;
            fail
              (Printf.sprintf
                 "%s carries a backend marker at line %d and runs no executable -- the marker \
                  belongs on the stanza that RUNS it, which for an `(executable)` is its companion \
                  rule, the same placement as the `%s` dep"
                 (where_of_stanza stanza) line Scan.config_file)
        | Scan.Marker_outside_stanza _ | Scan.Marker_outside_comment _ -> ());
      List.iter marker_contract.Scan.contract_stanzas ~f:(fun marked_stanza ->
          let stanza = marked_stanza.Scan.marker_stanza in
          let where = where_of_stanza stanza in
          (* The rule itself is [Scan.backend_rule_of]: this check owns the wording of the
             diagnostics and the tallies, and the DECISION lives with the scan so that it can be put
             to a stanza the repository does not contain (gh-ocannl-690). *)
          match Scan.backend_rule_of marked_stanza with
          | Scan.Runs_nothing -> ()
          | Scan.Names_twice line ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s carries more than one backend marker (the second at line %d) -- one stanza \
                    runs on one backend, so say so once"
                   where line)
          | Scan.Declares_and_names (line, m) ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s both declares `(env_var %s)` and carries a marker at line %d saying `%s` -- \
                    those are contradictory: a stanza that names its backend has nothing for the \
                    variable to invalidate, and a stanza that selects one has no business claiming \
                    otherwise. Keep whichever is true"
                   where Scan.backend_env_var line m.Scan.backend)
          | Scan.Declares_variable ->
              Int.incr placed_subjects;
              any_declared := true;
              classification :=
                Printf.sprintf "  %-58s declares %s" where Scan.backend_env_var :: !classification
          | Scan.Names_backend (_, m) ->
              Int.incr placed_subjects;
              List.iter (String.split m.Scan.backend ~on:',') ~f:(fun word ->
                  words := Set.add !words word);
              classification :=
                Printf.sprintf "  %-58s %s -- %s" where m.Scan.backend m.Scan.reason
                :: !classification
          | Scan.Names_neither ->
              Int.incr placed_subjects;
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s runs an executable and declares neither `(env_var %s)` nor a backend marker \
                    -- so `%s=cuda dune build @…` would serve this stanza's previous result as a \
                    pass. Add the declaration if the run SELECTS a backend, or the marker `; %s \
                    <%s> -- <reason>` if it names one or links none"
                   where Scan.backend_env_var Scan.backend_env_var Scan.marker_sentinel
                   (String.concat ~sep:"|" Scan.marker_backends)));
      List.iter marker_contract.Scan.contract_issues ~f:(function
        | Scan.Marker_outside_stanza { issue_line = line; issue_text = text } ->
            Int.incr marker_holes;
            fail
              (Printf.sprintf
                 "%s:%d has a backend marker that sits inside no stanza -- a comment between \
                  stanzas declares nothing; move it inside the parentheses of the stanza it is \
                  about. The line reads `;%s`"
                 dune_file line text)
        | Scan.Marker_outside_comment
            { issue_text_occurrences = in_text; issue_comment_occurrences = in_comments } ->
            Int.incr marker_holes;
            fail
              (Printf.sprintf
                 "%s spells `%s` %d times and only %d of them are in a comment this scan places -- \
                  a marker outside a comment declares nothing, and one in a comment this scan \
                  cannot see is one it will not read"
                 dune_file Scan.marker_sentinel in_text in_comments)
        | Scan.Malformed_marker _ | Scan.Marker_in_wrong_stanza _ -> ());
      (* The floor under the walk, read by the second reader that shares none of its classification
         machinery: a stanza the walk stops seeing is a stanza the rule above stops applying to,
         which looks exactly like a file with nothing to check (the gh-ocannl-665 argument, and
         config_dep_completeness' floors).

         Checked STANZA BY STANZA, not as two totals over the file. A total has slack in it, and the
         slack is not hypothetical: the raw reader recognises fewer shapes than `sites_of_stanza`
         does -- no `bash`/`system`, nothing under an unresolvable `chdir` -- so a stanza the walk
         places and this reader misses adds one to the walk's count and nothing to the floor, which
         is exactly enough to absorb a DIFFERENT stanza silently dropping out of enforcement (Codex
         P2, round 2). Asked per stanza, the two answers are about the same stanza and cannot be
         traded against a third; and the raw reader's narrower vocabulary degrades to a weaker floor
         for the stanzas it cannot see, rather than to a hole somewhere else in the file. *)
      List.iter marked ~f:(fun stanza ->
          let identity =
            Printf.sprintf "%s:%d, the %s %s" dune_file stanza.Scan.marked_line
              stanza.Scan.marked_head
              (if String.is_empty stanza.Scan.marked_name then "<unnamed>"
               else stanza.Scan.marked_name)
          in
          if stanza.Scan.marked_raw_subject && List.is_empty stanza.Scan.marked_sites then (
            Int.incr marker_holes;
            fail
              (Printf.sprintf
                 "%s: the raw text shows it running an executable and the walk placed no site for \
                  it -- it is reading the file with a hole in it, and a stanza it stops seeing is \
                  one this rule stops applying to"
                 identity));
          if stanza.Scan.marked_raw_subject then (
            Int.incr subject_floor;
            floor_names := identity :: !floor_names);
          if not (List.is_empty stanza.Scan.marked_sites) then
            walk_places :=
              ( identity,
                Printf.sprintf "%s running %s" identity
                  (String.concat ~sep:", "
                     (List.map stanza.Scan.marked_sites ~f:(fun s -> s.Scan.name))) )
              :: !walk_places);
      by_file :=
        ( dune_file,
          (if !any_declared then [ "declares " ^ Scan.backend_env_var ] else [])
          @
          match Set.to_list !words with
          | [] -> []
          | words -> [ "markers: " ^ String.concat ~sep:", " words ] )
        :: !by_file;
      (* The stanzas as dune applies them, each with the [(subdir ...)] it sits under.
         [marked_stanzas] makes the same descent as the backend subject scan; this is the one input
         to [per_directory] below, so every directory-scoped check consumes the same groups. *)
      let file_stanzas = List.map marked ~f:(fun m -> (m.Scan.marked_subdir, m.Scan.marked_sexp)) in
      let stanzas_in subdir =
        List.filter_map file_stanzas ~f:(fun (candidate, stanza) ->
            if String.equal candidate subdir then Some stanza else None)
      in
      let directory_checks = ref [] in
      let check_per_directory check = directory_checks := check :: !directory_checks in
      (* Runner identities are written relative to the DUNE FILE, so the raw `(subdir …)` path is
         what qualifies them -- not the repository-relative directory the modules are looked up
         in. *)
      (* Each candidate runner with the SUBDIRECTORY it was found in: the path a rule writes is
         relative to where the rule lives, so `(subdir a (rule … probe.exe))` and
         `(subdir b (rule … probe.exe))` name different programs and only the pair says which
         (gh-ocannl-747, Codex P2 round 3). *)
      let all_group_runners = file_stanzas in
      let artifact_subjects = ref [] in
      (* gh-ocannl-800: collect the executable each repository-wide scan rule runs. Dune's main
         module contract makes [foo.exe] come from [foo.ml]; the source lookup below holds that
         relationship against the files this run was actually handed. This is deliberately a
         callback on the gh-ocannl-813 seam: a scan rule under [(subdir ...)] cannot fall out by
         needing one more private descent. *)
      check_per_directory (fun subdir group ->
          List.iter group ~f:(fun stanza ->
              if repository_census && is_repo_wide_scan stanza then (
                let executables =
                  Scan.sites_of_stanza subdir stanza
                  |> List.concat_map ~f:(fun site -> site.Scan.executables)
                  |> List.dedup_and_sort ~compare:String.compare
                in
                if List.is_empty executables then
                  fail
                    (Printf.sprintf
                       "%s%s is a repository-wide scan rule and this check cannot name the scanner \
                        executable it runs -- a scanner source it cannot name is one whose refusal \
                        diagnostics it cannot cover"
                       dune_file
                       (if String.is_empty subdir then "" else " in `(subdir " ^ subdir ^ " ...)`"));
                List.iter executables ~f:(fun executable ->
                    let executable_path =
                      Scan.in_subdir (Scan.in_subdir dir subdir) executable |> normalize_path
                    in
                    match String.chop_suffix executable_path ~suffix:".exe" with
                    | None ->
                        fail
                          (Printf.sprintf
                             "%s%s runs `%s` from a repository-wide scan rule, which is not an \
                              executable name this check can relate to a scanner source"
                             dune_file
                             (if String.is_empty subdir then ""
                              else " in `(subdir " ^ subdir ^ " ...)`")
                             executable)
                    | Some stem ->
                        let source = String.lowercase (stem ^ ".ml") in
                        if List.Assoc.mem sources source ~equal:String.equal then
                          scanner_sources := Set.add !scanner_sources source
                        else
                          fail
                            (Printf.sprintf
                               "%s runs the repository scanner `%s`, and this check was handed no \
                                `%s` source to inspect for refusal diagnostics -- add the \
                                scanner's directory to the source globs"
                               dune_file executable source)))));
      (* gh-ocannl-723: the artifact-directory declaration, against the modules that read the key
         needing it. Both directions, since a declaration nothing reads for is the restatement this
         replaces rather than the relationship. This is the first consumer of the common
         per-directory seam; nested modules and executables cannot fall back to the dune file's
         directory. *)
      check_per_directory (fun subdir group ->
          let here = Scan.in_subdir dir subdir in
          let subjects =
            let key module_name = String.lowercase (Scan.in_subdir here (module_name ^ ".ml")) in
            let calls module_name = Set.mem artifact_caller_keys (key module_name) in
            (* A module that reads `build_files_prefix` some other way needs the variable tracked
               for the same reason a caller does, and is subject to the same rule (Codex P2, rounds
               2 and 3). Narrowed textually first, the way the caller census is: the key has to be
               SPELLED for either spelling of a read to name it, so only the sources that mention it
               are parsed. *)
            let reads_prefix module_name =
              match List.Assoc.find sources ~equal:String.equal (key module_name) with
              | None -> false
              | Some on_disk -> (
                  let content = In_channel.read_all on_disk in
                  String.is_substring content ~substring:artifact_config_key
                  && try Sources.source_reads_key content ~key:artifact_config_key with _ -> false)
            in
            (* Dune's default module set is the directory less what other stanzas claim, so the scan
               needs to know what the directory holds -- a `(test (name t))` with no `(modules …)`
               builds `t.ml` (Codex P2, round 2). Only the sources this scan was handed, which is
               what it can answer for; the census check below is what catches a caller no stanza
               claims either way. *)
            let directory = if String.is_empty here then "." else here in
            let directory_modules =
              List.filter_map source_files ~f:(fun (path, _) ->
                  if String.equal (Stdlib.Filename.dirname path) directory then
                    Some (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))
                  else None)
            in
            Scan.artifact_subjects ~directory_modules ~subdir ~runner_stanzas:all_group_runners
              group ~calls ~reads_prefix
          in
          List.iter subjects ~f:(fun subject ->
              let where =
                Printf.sprintf "%s, the %s %s" dune_file subject.Scan.artifact_head
                  subject.Scan.artifact_name
              in
              let needs = subject.Scan.artifact_callers @ subject.Scan.artifact_readers in
              let what =
                match (subject.Scan.artifact_callers, subject.Scan.artifact_readers) with
                | [], readers ->
                    String.concat ~sep:", " readers ^ " reads `" ^ artifact_config_key ^ "` by name"
                | callers, [] ->
                    String.concat ~sep:", " callers ^ " calls `Test_utils.Generated.init`"
                | callers, readers ->
                    String.concat ~sep:", " callers ^ " calls `Test_utils.Generated.init` and "
                    ^ String.concat ~sep:", " readers ^ " reads `" ^ artifact_config_key
                    ^ "` by name"
              in
              List.iter needs ~f:(fun m ->
                  artifact_claimed :=
                    Set.add !artifact_claimed (String.lowercase (Scan.in_subdir here (m ^ ".ml"))));
              artifact_table :=
                Printf.sprintf "  %-58s %s (%s)" where
                  (Scan.artifact_verdict_name subject.Scan.artifact_verdict)
                  (if List.is_empty needs then subject.Scan.artifact_deps_site
                   else String.concat ~sep:", " needs)
                :: !artifact_table;
              artifact_subjects := subject :: !artifact_subjects;
              match subject.Scan.artifact_verdict with
              | Scan.Artifact_declared | Scan.Artifact_other_reader -> ()
              | Scan.Artifact_undeclared ->
                  Int.incr artifact_violations;
                  fail
                    (Printf.sprintf
                       "%s: %s -- `%s` decides which directory the run's generated artifacts are \
                        read from, and %s does not declare `(env_var %s)`, so dune serves the \
                        previous run's result across a change of it. Add the declaration there"
                       where what artifact_config_key subject.Scan.artifact_deps_site
                       Scan.artifact_env_var)
              | Scan.Artifact_stale_declaration ->
                  Int.incr artifact_violations;
                  fail
                    (Printf.sprintf
                       "%s declares `(env_var %s)` in %s and no module of it reads `%s` at all -- \
                        neither through `Test_utils.Generated.init` nor by name. A declaration \
                        with nothing behind it is a restatement, not a relationship, and the next \
                        author copies it. Drop it, or read the generated artifacts through \
                        `Test_utils.Generated`, which is the one supported way to read them"
                       where Scan.artifact_env_var subject.Scan.artifact_deps_site
                       artifact_config_key)
              | Scan.Artifact_unrun ->
                  Int.incr artifact_violations;
                  fail
                    (Printf.sprintf
                       "%s: %s, and no stanza in this file runs the executable -- an \
                        `(executable)` has no `deps` field, so the `(env_var %s)` declaration goes \
                        on the rule that RUNS it, the same placement as the `%s` dep and the \
                        backend marker. This scan can find neither"
                       where what Scan.artifact_env_var Scan.config_file)
              | Scan.Artifact_in_library ->
                  Int.incr artifact_violations;
                  fail
                    (Printf.sprintf
                       "%s: %s, from a library module -- the initializer empties the artifact \
                        directory of the process that owns it, so it belongs to an executable's \
                        own modules. Called through a library it puts the `(env_var %s)` \
                        requirement on every stanza that links the library, where nothing follows \
                        it"
                       where what Scan.artifact_env_var)));
      (* The ambient gate, per directory AND per alias (gh-ocannl-652). Per SUBDIRECTORY too: a
         `(subdir child …)` group defines aliases in `child`, and a gate at the top level is a gate
         for the top level -- so a family alias written inside the group could otherwise serve its
         member from cache with a rejected spelling ambient, which is the whole failure
         gh-ocannl-652 closed (Codex P1, round 4). *)
      check_per_directory (fun subdir here ->
          let where =
            if String.is_empty subdir then dune_file
            else Printf.sprintf "%s, in `(subdir %s …)`" dune_file subdir
          in
          let gated_here = gated_aliases here in
          let entries = entry_points here in
          (* The lock (Codex P1 round 4): a gate in a directory whose actions take it has to take it
             too. Since this check receives a directory group, a group's unlocked gate cannot hide
             behind the outer [(subdir ...)] form (Codex P2, round 5). *)
          if List.exists here ~f:takes_training_lock then
            List.iter here ~f:(fun s ->
                if is_gate s && not (takes_training_lock s) then
                  fail
                    (Printf.sprintf
                       "%s%s serializes its actions on `%s` and its gate on `%s` does not take the \
                        lock -- the one unlocked action in a directory of locked ones is what the \
                        next training test gets copied from; add `(locks %s)` to it"
                       dune_file
                       (if String.is_empty subdir then ""
                        else Printf.sprintf ", in `(subdir %s …)`" subdir)
                       training_lock
                       (String.concat ~sep:", " (aliases_of s))
                       training_lock));
          List.iter entries ~f:(fun alias ->
              if Set.mem gated_here alias then gated := (where, alias) :: !gated
                (* The exemption is a statement about ONE directory -- `benchmarks/dune` runs
                   python3 over the orchestrator's own tests and links no OCANNL executable -- and a
                   `(subdir …)` group of that file is a different directory, whose stanzas the
                   recorded reason says nothing about (Codex P2, round 7). Applying it there would
                   exempt a nested OCANNL-linked test on the strength of its parent's reason. *)
              else if String.is_empty subdir && Map.mem gateless dune_file then
                gateless_used := Set.add !gateless_used dune_file
              else
                fail
                  (Printf.sprintf
                     "%s has actions on the `%s` alias and no ambient gate reaches it -- nothing \
                      there declares a rejected environment spelling, so `ocannl_backend=cuda dune \
                      build @%s` would serve that directory's cached results with the fatal \
                      startup check never reached; copy the `env_spelling_gate` stanza for that \
                      alias from a neighbour, depend on the gate's alias from the rule, or exempt \
                      the directory by name with the reason"
                     where alias
                     (if String.equal alias "runtest" then
                        Scan.in_subdir
                          (Stdlib.Filename.dirname dune_file)
                          (Scan.in_subdir subdir alias)
                      else alias)));
          (* Each suite's members, against what it aggregates -- in the directory that defines
             it. *)
          List.iter suites ~f:(fun suite ->
              let reaches = aliases_reached_from here suite in
              List.iter entries ~f:(fun alias ->
                  if member_of suite alias && not (Set.mem reaches alias) then
                    fail
                      (Printf.sprintf
                         "%s attaches a rule to `%s` that the `%s` alias does not aggregate -- \
                          `dune build @%s` would skip it silently; list `(alias %s)` in the \
                          `(alias (name %s) (deps …))` stanza"
                         dune_file alias suite suite alias suite))));
      (* The scans family, the same question asked of the repo-wide scans (gh-ocannl-703): the rule
         that diffs a scan's output against its golden is the one that fails, so it is the one
         `@scans` has to reach. The producers are recognized by what they read rather than by name,
         so a scan added tomorrow is asked about too. *)
      let scans_reaches = aliases_reached_from stanzas scans_suite in
      List.iter stanzas ~f:(fun producer ->
          if is_repo_wide_scan producer then
            (* A producer the family already aggregates directly needs no separate checker at all:
               declaring a target says where its diagnostics go, not that a second rule judges them
               -- a scan can assert by exit status and still write one (Codex P2, round 8). Asked
               before the target hunt, so both shapes are accepted the same way. *)
            (* The targets a second rule could be JUDGING. A response file the rule wrote for its
               own action is not one, and it is subtracted before the emptiness question rather
               than inside the iteration: a scan whose only target is its response file would
               otherwise reach the loop, iterate over nothing and pass -- the round-4 fail-open
               again, one shape over. *)
            let judged_targets =
              List.filter (targets_of producer) ~f:(Fn.non (is_own_response_file producer))
            in
            if List.exists (aliases_of producer) ~f:(Set.mem scans_reaches) then ()
            else if List.is_empty judged_targets then
              (* A scan that declares no target writes and checks in one action -- the `no-infer`
                 shape this repository uses elsewhere, or an assertion by exit status. There is no
                 second rule to look for, so the family has to aggregate THIS rule's own alias;
                 iterating over its (empty) target list would have passed it silently, which is the
                 fail-open Codex found in round 4. *)
              if not (List.exists (aliases_of producer) ~f:(Set.mem scans_reaches)) then
                fail
                  (Printf.sprintf
                     "%s has a rule that globs the repository -- a repo-wide scan -- and declares \
                      no target, so it checks its own output in its action, and the `%s` alias \
                      does not aggregate the alias it sits on (%s): `dune build @%s/%s` would skip \
                      it silently. List its alias in the `(alias (name %s) (deps …))` stanza"
                     dune_file scans_suite
                     (match aliases_of producer with
                     | [] -> "none"
                     | aliases -> String.concat ~sep:", " aliases)
                     (Stdlib.Filename.dirname dune_file)
                     scans_suite scans_suite)
              else ()
            else
              List.iter judged_targets ~f:(fun target ->
                  let checkers =
                    List.filter stanzas ~f:(fun s ->
                        is_golden_diff s
                        && Option.value_map (Scan.field s "action") ~default:false ~f:(fun args ->
                            List.exists args ~f:(diffs_file ~named:(named_deps s) target)))
                  in
                  let aliases = List.concat_map checkers ~f:aliases_of in
                  if not (List.exists aliases ~f:(Set.mem scans_reaches)) then
                    fail
                      (Printf.sprintf
                         "%s globs the repository to produce `%s` -- a repo-wide scan -- and no \
                          rule the `%s` alias aggregates diffs it against its golden: `dune build \
                          @%s/%s` would skip it silently. Give the diff rule `(alias \
                          runtest-<name>)` -- that alias and no other -- and list `(alias \
                          runtest-<name>)` in BOTH the `(alias (name runtest) (deps …))` and the \
                          `(alias (name %s) (deps …))` stanzas"
                         dune_file target scans_suite
                         (Stdlib.Filename.dirname dune_file)
                         scans_suite scans_suite)));
      (* The focused aggregates (gh-ocannl-783): the same completeness question, asked of two more
         families whose membership is derived from what the member stanza declares. `file_stanzas`
         and `stanzas_in` are the same per-directory readings the gate check above uses.

         Every unit the file builds, each in the subdirectory dune applies it to. The whole file's
         stanzas go in, since the rule that runs an executable need not sit in its group. *)
      let units =
        List.concat_map file_stanzas ~f:(fun (subdir, stanza) ->
            family_units file_stanzas ~subdir stanza)
      in
      (* The modules of ONE unit. Dune builds each name of a plural stanza as its own executable --
         its main module plus the stanza's shared ones, and NOT the other names' mains -- so asking
         the question of the stanza's whole module list makes one main's use of the instrumentation
         a claim about its neighbour too (Codex P2, round 2). *)
      let unit_module_sources { unit_subdir; unit_stanza; unit_name; _ } =
        let here = Scan.in_subdir dir unit_subdir in
        let directory = if String.is_empty here then "." else here in
        let directory_modules =
          List.filter_map source_files ~f:(fun (path, _) ->
              if String.equal (Stdlib.Filename.dirname path) directory then
                Some (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))
              else None)
        in
        (* Every main module this directory's stanzas name OTHER than this unit's own. Dune gives
           each named test and executable its own main and shares the rest, so a stanza that omits
           `(modules …)` does not build its neighbour's main -- and reading it as if it did made one
           main's use of the instrumentation a claim about every default-module stanza beside it
           (Codex P2, rounds 2 and 13). *)
        let siblings =
          match unit_name with
          | None -> []
          | Some name ->
              List.concat_map (stanzas_in unit_subdir) ~f:(fun stanza ->
                  if has_family_modules stanza then Scan.names_of stanza else [])
              |> List.filter ~f:(fun other -> not (String.equal other name))
              |> List.map ~f:String.lowercase
        in
        Scan.modules_of ~directory_modules (stanzas_in unit_subdir) unit_stanza
        |> List.filter ~f:(fun module_name ->
            not (List.mem siblings (String.lowercase module_name) ~equal:String.equal))
        |> List.map ~f:(fun module_name ->
            String.lowercase (Scan.in_subdir here (module_name ^ ".ml")))
      in
      (* The Metal derivation reads the MARKED stanza, whatever kind it is. For an `(executable)`
         the marker's required placement is the rule that runs it (gh-ocannl-659), so a Metal test
         in that form carries its marker on an unnamed `(rule …)`: asking the question of the
         executable stanza instead would find no marker and let the family omit the test while this
         scan passed (Codex P2, round 1). Whatever stanza carries the marker is the member -- every
         unit of it, since a marker is a statement about the stanza. *)
      let metal_stanzas =
        List.filter_map marker_contract.Scan.contract_stanzas ~f:(fun marked_stanza ->
            let stanza = marked_stanza.Scan.marker_stanza in
            match Scan.backend_rule_of marked_stanza with
            | Scan.Names_backend (_, body) | Scan.Declares_and_names (_, body) ->
                (* The marker admits several words, comma-separated, where a stanza honestly names
                   two backends -- so membership is "names metal among them", not "is spelled
                   metal". *)
                if List.mem (String.split body.Scan.backend ~on:',') "metal" ~equal:String.equal
                then Some (stanza.Scan.marked_subdir, stanza.Scan.marked_sexp)
                else None
            | Scan.Runs_nothing | Scan.Declares_variable | Scan.Names_twice _ | Scan.Names_neither
              ->
                None)
      in
      let metal_members =
        List.filter units ~f:(fun u ->
            List.exists metal_stanzas ~f:(fun (subdir, stanza) ->
                String.equal subdir u.unit_subdir && phys_equal stanza u.unit_stanza))
      in
      let lifecycle_members =
        List.filter units ~f:(fun u ->
            has_family_modules u.unit_stanza
            && List.exists (unit_module_sources u) ~f:(Set.mem lifecycle_sources))
      in
      let family_reaches = Hashtbl.create (module String) in
      let family_directory family subdir = family.family_alias ^ "\t" ^ subdir in
      check_per_directory (fun subdir here ->
          List.iter families ~f:(fun family ->
              Hashtbl.set family_reaches ~key:(family_directory family subdir)
                ~data:(aliases_reached_from here family.family_alias)));
      let check_family_members () =
        List.iter
          [ (metal_family, metal_members); (lifecycle_family, lifecycle_members) ]
          ~f:(fun (family, members) ->
            (* Reachability PER DIRECTORY, from the stanzas dune applies there: a `(subdir child …)`
               group may carry its own `(alias (name <family>) …)`, and the root recursive
               `@<family>` build reaches it, so a member there is correctly wired (Codex P2, round
               2). Which directory is asked comes from the ALIAS, not from the member: an executable
               in a subdirectory run by a top-level rule is aggregated by a top-level family stanza
               (Codex P2, round 3). What is never right is reaching an alias from a directory that
               does not define it, which is what asking the file as a whole would have allowed. *)
            let reached (subdir, alias) =
              match Hashtbl.find family_reaches (family_directory family subdir) with
              | Some reached -> Set.mem reached alias
              | None -> false
            in
            List.iter members ~f:(fun u ->
                let aggregated = List.exists u.unit_aliases ~f:reached in
                family_table :=
                  (dune_file, family.family_alias, u.unit_identity, aggregated) :: !family_table;
                if not aggregated then
                  fail
                    (Printf.sprintf
                       "%s: the %s%s %s, and the `%s` alias does not reach it -- `dune build @%s` \
                        would skip it silently. List %s, adding an `(alias (name %s) (deps …))` \
                        stanza in that directory if it has no member yet"
                       dune_file u.unit_identity
                       (if String.is_empty u.unit_subdir then ""
                        else Printf.sprintf " in `(subdir %s …)`" u.unit_subdir)
                       family.family_is family.family_alias family.family_alias
                       (match u.unit_aliases with
                       | [] ->
                           "the alias of a rule that runs it -- it has none of its own, being an \
                            `(executable)` no rule in this file runs under a dedicated alias"
                       | aliases ->
                           String.concat ~sep:" or "
                             (List.map aliases ~f:(fun (subdir, alias) ->
                                  if String.is_empty subdir then Printf.sprintf "`(alias %s)`" alias
                                  else Printf.sprintf "`(alias %s)` in `(subdir %s …)`" alias subdir)))
                       family.family_alias)))
      in
      (* A hand-written per-test alias must not reuse a name dune generates one for: the aliases
         merge, and the targeted run stops being one test (Codex P2, round 5). Asked of every rule
         in the file rather than of the golden diffs alone, since any rule can be given such an
         alias; the ambient gate is the one rule that means to share it. *)
      (* Per `(subdir …)` group, like the gate, lock and family checks: dune generates a stanza's
         alias in the directory it applies the stanza to, and a top-level reading of a file with a
         group sees neither the group's `(test)` names nor the rules that could collide with them
         (Codex P2, round 10). *)
      check_per_directory (fun subdir here ->
          let generated = generated_runtest_names here in
          let gate_names = gate_generated_names here in
          List.iter here ~f:(fun stanza ->
              List.iter (aliases_of stanza) ~f:(fun alias ->
                  match String.chop_prefix alias ~prefix:"runtest-" with
                  (* The one deliberate collision: a rule sharing the alias dune generates for a
                     `(test)` stanza BECAUSE IT RUNS THE SAME BINARY -- the ambient gate, whose rule
                     exists so that every per-test alias can depend on it. Three things have to
                     hold, and the third is what the earlier rounds were missing: the rule is a
                     gate, the name belongs to a gate `(test)` stanza, and the rule's action runs
                     that stanza's executable, so the merged alias runs one program either way
                     (Codex P2, rounds 6 to 8). *)
                  | Some name
                    when is_gate stanza && Set.mem gate_names name
                         && List.exists (Scan.executables_run stanza) ~f:(fun (cwd, command) ->
                             match command with
                             (* Resolved, not by basename: `(chdir other (run ./gate.exe))` runs
                                another directory's binary, and granting the exemption to it would
                                let the merged alias run the generated test AND something unrelated
                                (Codex P2, round 11). The same resolution the family's runner
                                matching makes. *)
                             | Scan.Runs path ->
                                 String.equal
                                   (normalize_path
                                      (Scan.in_subdir subdir (Scan.in_subdir cwd path)))
                                   (normalize_path (Scan.in_subdir subdir (name ^ ".exe")))
                             | _ -> false) ->
                      ()
                  | Some name when Set.mem generated name ->
                      fail
                        (Printf.sprintf
                           "%s attaches a rule to `%s`, the alias dune generates for the `%s` \
                            stanza in this directory -- the two merge, so `dune build @%s/%s` \
                            would run that test as well as this rule, and naming `runtest` beside \
                            it is a dependency cycle. Name the alias after the GOLDEN this rule \
                            checks, qualified where a run writes several"
                           (if String.is_empty subdir then dune_file
                            else Printf.sprintf "%s, in `(subdir %s …)`" dune_file subdir)
                           alias name
                           (Scan.in_subdir (Stdlib.Filename.dirname dune_file) subdir)
                           alias)
                  | _ -> ())));
      (* And one alias checks one golden: two golden diffs sharing an alias make the targeted run
         two tests, which is the isolation this arrangement is for -- and the prefix relation above
         admits the pair, since `foo.expected` and `foo-extension.expected` both accept
         `runtest-foo-extension` (Codex P2, round 7). A producer rule sharing its checker's alias is
         a different thing and stays allowed: it is what MAKES the output the checker reads. *)
      List.iter
        (List.concat_map stanzas ~f:(fun s ->
             (* One entry per GOLDEN, not per rule: a single rule whose `progn` diffs two goldens
                puts two checks behind one alias just as two rules would (Codex P2, round 8). *)
             if is_golden_diff s then
               List.concat_map (aliases_of s) ~f:(fun alias ->
                   List.map (goldens_in s) ~f:(fun _ -> alias))
             else [])
        |> List.sort ~compare:String.compare
        |> List.group ~break:(fun a b -> not (String.equal a b))
        |> List.filter ~f:(fun group -> List.length group > 1))
        ~f:(fun group ->
          fail
            (Printf.sprintf
               "%s checks %d goldens on the alias `%s` -- `dune build @%s/%s` would run them all, \
                so the alias no longer names one test. Give each its own, named after the golden \
                it checks"
               dune_file (List.length group) (List.hd_exn group)
               (Stdlib.Filename.dirname dune_file)
               (List.hd_exn group)));
      (* Every golden diff sits on a per-test alias, and on that alias ALONE (gh-ocannl-726). Dune
         generates `runtest-<name>` for `(test)`/`(tests)` stanzas and inline-test libraries and for
         nothing else, so a rule that diffs a golden and names `runtest` itself can only be run by
         running the whole directory -- and validating it targeted then means building its `.actual`
         and diffing by hand, outside dune, which fails open. Naming BOTH aliases does not fix it
         either, and is the trap worth checking for: a rule attached to two aliases makes building
         either one build both, so the per-test alias would drag the directory in behind it, and
         where the name is one dune generates the pair is a dependency cycle outright. *)
      List.iter stanzas ~f:(fun stanza ->
          if is_golden_diff stanza then
            match aliases_of stanza with
            (* One alias, and it names a member of a suite: that is the whole convention. Asked of
               the alias SET rather than of the name `runtest` alone (Codex P2, round 1), since a
               rule with no alias at all, or with an alias of its own invention, has neither the
               targeted entry point nor a place in a suite -- and its golden stops being checked
               without anything saying so. *)
            | [ alias ]
              when List.exists suites ~f:(fun suite -> member_of suite alias)
                   && List.for_all (goldens_in stanza) ~f:(fun golden ->
                       let suffix =
                         List.find_map suites ~f:(fun suite ->
                             String.chop_prefix alias ~prefix:(suite ^ "-"))
                         |> Option.value ~default:alias
                       in
                       let stem = golden_stem ~named:(named_deps stanza) golden in
                       (* The stem itself, or the stem and then a qualifier saying WHICH golden of
                          the run this is. A bare prefix would accept `runtest-foobar` for
                          `foo.expected`, leaving the alias a reader constructs empty (Codex P2,
                          round 8). *)
                       (not (String.is_empty stem))
                       && (String.equal suffix stem || String.is_prefix suffix ~prefix:(stem ^ "-")))
              ->
                ()
            | aliases ->
                let what =
                  match aliases with
                  | [] -> "attaches a golden diff to no alias at all"
                  | [ alias ] when not (List.exists suites ~f:(fun s -> member_of s alias)) ->
                      Printf.sprintf "attaches a golden diff to `%s`, which is no suite's member"
                        alias
                  | [ alias ]
                    when List.exists (goldens_in stanza) ~f:(fun g ->
                             String.is_empty (golden_stem ~named:(named_deps stanza) g)) ->
                      Printf.sprintf
                        "attaches `%s` to a golden this check cannot name -- %s reduces to an \
                         empty stem, so nothing constrains the alias. Spell the golden as a plain \
                         path, or teach `golden_stem` the form"
                        alias
                        (String.concat ~sep:", "
                           (List.filter (goldens_in stanza) ~f:(fun g ->
                                String.is_empty (golden_stem ~named:(named_deps stanza) g))))
                  | [ alias ] ->
                      Printf.sprintf
                        "attaches `%s` to a golden its name does not name: the goldens are %s, so \
                         the alias should begin `<suite>-%s`. A reader reaches for the alias with \
                         the failing GOLDEN in hand, and an alias that renames it is one they \
                         construct empty"
                        alias
                        (String.concat ~sep:", " (goldens_in stanza))
                        (String.concat ~sep:"` or `<suite>-"
                           (List.map (goldens_in stanza)
                              ~f:(golden_stem ~named:(named_deps stanza))))
                  | aliases ->
                      Printf.sprintf
                        "attaches a golden diff to %d aliases (%s) -- and a rule on two aliases \
                         makes building either one build both"
                        (List.length aliases) (String.concat ~sep:", " aliases)
                in
                fail
                  (Printf.sprintf
                     "%s %s. Give it `(alias <suite>-<name>)` -- that alias and no other, with \
                      <suite> one of %s -- and list it in this file's `(alias (name <suite>) (deps \
                      …))` stanza, which is what runs it as part of the suite. <name> is the \
                      golden the rule checks, and for `runtest` must not be the name of a `(test)` \
                      stanza in this directory, since dune generates `runtest-<name>` for those"
                     dune_file what (String.concat ~sep:", " suites)));
      let fields = List.concat_map stanzas ~f:dep_fields in
      let declared =
        List.concat_map fields ~f:(fun (_, args) -> List.concat_map args ~f:env_vars_in)
      in
      (* A declaration this scan did not look inside a dependency field for is one it cannot check,
         so the two counts have to agree. Dune admits `(env_var ...)` only in a dependency
         specification, so a disagreement means a field spelling this scan does not know. *)
      let all = List.concat_map stanzas ~f:env_vars_in in
      if List.length all <> List.length declared then
        fail
          (Printf.sprintf
             "%s declares %d `(env_var ...)` dependencies but only %d of them are in a `deps` or \
              `preprocessor_deps` field -- teach this check the field that holds the others"
             dune_file (List.length all) (List.length declared));
      List.iter fields ~f:(fun (_field, args) ->
          let names = List.concat_map args ~f:env_vars_in in
          List.iter names ~f:(fun name ->
              let key = dune_file ^ ":" ^ name in
              match Utils.classify_env_var name with
              (* Someone else's variable entirely (`PATH`, `HOME`): not this check's business. *)
              | Utils.Env_not_addressed -> ()
              | Utils.Env_config_key config_key -> tracked_keys := Set.add !tracked_keys config_key
              (* The gates are checked below, against the modules that read them. *)
              | Utils.Env_reserved _ -> ()
              | Utils.Env_unread_spelling _ | Utils.Env_unknown_key _ | Utils.Env_unread_reserved _
                ->
                  if Map.mem exemptions key then exemptions_used := Set.add !exemptions_used key
                  else
                    let what =
                      match Utils.classify_env_var name with
                      | Utils.Env_unread_reserved prefix ->
                          "is in the reserved " ^ prefix
                          ^ " namespace in a casing its reader does not consult"
                      | _ -> "addresses OCANNL and names no configuration key it reads"
                    in
                    fail
                      (Printf.sprintf
                         "%s declares `(env_var %s)`, which %s -- a variable nothing consults \
                          invalidates nothing; fix the spelling, or exempt it by name with the \
                          reason"
                         dune_file name what)));
      (* What a stanza's own modules read, against what the stanza declares: the gates, read while
         PREPROCESSING them, and the ambient variables they read by name at RUN time. *)
      (* Per `(subdir …)` group, like every other reading in this file: a nested stanza's modules
         live in that directory, and a top-level walk saw the `(subdir …)` wrapper instead of them
         -- so a child module could read a tracing gate or an ambient variable undeclared (Codex P2,
         round 16). *)
      check_per_directory (fun subdir stanzas ->
          let dir = Scan.in_subdir dir subdir in
          List.iter stanzas ~f:(fun stanza ->
              match (Scan.head stanza, Scan.field stanza "modules") with
              | Some kind, Some modules when List.mem module_stanzas kind ~equal:String.equal ->
                  let name =
                    match Scan.names_of stanza with name :: _ -> name | [] -> "<unnamed>"
                  in
                  let where =
                    Printf.sprintf "%s%s, %s %s" dune_file
                      (if String.is_empty subdir then ""
                       else Printf.sprintf " in `(subdir %s …)`" subdir)
                      kind name
                  in
                  let env_vars_of field =
                    match Scan.field stanza field with
                    | None -> []
                    | Some args -> List.concat_map args ~f:env_vars_in
                  in
                  let declared_gates =
                    List.filter (env_vars_of "preprocessor_deps") ~f:(fun name ->
                        match Utils.classify_env_var name with
                        | Utils.Env_reserved prefix -> String.equal prefix gate_prefix
                        | _ -> false)
                  in
                  let sources =
                    List.filter_map modules ~f:(function
                      | Sexp.Atom module_name ->
                          Option.map (source_of ~dir module_name) ~f:(fun on_disk ->
                              (module_name ^ ".ml", In_channel.read_all on_disk))
                      | _ -> None)
                  in
                  let read_gates =
                    List.concat_map sources ~f:(fun (source, content) ->
                        Sources.tracing_gates_in_source content
                        |> List.map ~f:(fun gate -> (gate, source)))
                  in
                  if (not (List.is_empty declared_gates)) && not (Set.mem scanned_dirs dir) then
                    fail
                      (Printf.sprintf
                         "%s declares tracing gates, and this check was handed no sources from %s \
                          to check them against -- add the directory to the rule\'s globs"
                         where
                         (if String.is_empty dir then "the repository root" else dir))
                  else (
                    List.iter read_gates ~f:(fun (gate, source) ->
                        if not (List.mem declared_gates gate ~equal:String.equal) then
                          fail
                            (Printf.sprintf
                               "%s/%s reads the tracing gate %s while being preprocessed, and %s \
                                does not declare it -- setting the variable then returns the \
                                modules already built without the trace statements"
                               dir source gate where)
                        else gate_table := (where, gate, dir ^ "/" ^ source) :: !gate_table);
                    List.iter declared_gates ~f:(fun gate ->
                        if not (List.Assoc.mem read_gates gate ~equal:String.equal) then
                          fail
                            (Printf.sprintf
                               "%s declares the tracing gate %s, which none of its modules reads \
                                -- drop it, or move it to the library whose modules do"
                               where gate)));
                  (* The run-time half (Codex P2, round 2). A gate is read while the module is
                     built; `Sys.getenv "OCANNL_TOOL_TEST_RESTRICT_MASK"` is read while it RUNS, and
                     the consequence of not declaring it is the same one this whole check is about
                     -- `test_cpu_topology` was reusable across a change of the mask that decides
                     what it does. Presence is checkable here and not for configuration keys at
                     large, which a test reaches through the library rather than by name: what makes
                     the difference is the literal in the source, which says exactly which variable
                     this module reads.

                     An `(executable)` stanza has no `deps` field at all -- its companion rule
                     carries them -- so for those the declaration is looked for anywhere in the dune
                     file, the same latitude `config_dep_completeness` gives the `ocannl_config`
                     dep. *)
                  let declared_for_stanza =
                    match Scan.field stanza "deps" with
                    | Some _ -> env_vars_of "deps"
                    | None -> declared
                  in
                  List.iter sources ~f:(fun (source, content) ->
                      Sources.env_var_reads_in_source content
                      |> List.iter ~f:(fun read ->
                          match Utils.classify_env_var read with
                          | Utils.Env_not_addressed -> ()
                          | _ ->
                              if not (List.mem declared_for_stanza read ~equal:String.equal) then
                                fail
                                  (Printf.sprintf
                                     "%s/%s reads the environment variable %s by name, and %s does \
                                      not declare it -- dune then reuses the previous result \
                                      across a change of the variable that decides what the run \
                                      does"
                                     dir source read where)
                              else read_table := (where, read, dir ^ "/" ^ source) :: !read_table))
              | _ -> ()));
      (* gh-ocannl-749: the same question, for the reads OCANNL's own environment reader makes.
         `Sys.getenv "NAME"` above says which variable it reads in the call itself;
         `Utils.read_env_var key` takes the key as a value, and the shape that matters takes it from
         a LIST -- an ambient-environment guard, refusing to run when a variable that would rewrite
         its golden is set. Three of them stand in this repository, and each was a hand-written list
         of keys standing in for its rule's `(env_var …)` declarations with nothing relating the two:
         the guard only RUNS when dune reruns the rule, which happens only for a variable the rule
         declares, so a key on the list and not in the deps is a key the guard never sees.
         gh-ocannl-628's hole, arrived at from the guard side.

         What is checked is the relationship, not the list: the keys are read out of the source and
         paired with the declarations of the rules dune runs it under, exactly as gh-ocannl-723 pairs
         a `Generated.init` caller with `OCANNL_BUILD_FILES_PREFIX` -- and through the same
         machinery, so the two cannot drift on who runs what.

         A pass of its own rather than a clause of the gate walk above, because the question is per
         PROGRAM and not per stanza, and because a stanza reaching for dune's default module set
         never entered that walk at all: it is guarded on `(modules …)` being written down, so a
         `(test (name guard))` whose implicit `guard.ml` reads the environment was accepted in
         silence (Codex P2, round 1). `Scan.modules_of` resolves the default set the way the artifact
         scan does.

         Over the SUBDIRECTORY groups the artifact check built, and not the top-level stanzas: a
         `(subdir gen …)` applies its stanzas to another directory, so a nested program's modules
         live there and its runners may sit at either level -- and a walk that saw only the top level
         read the wrapper as a stanza with no modules and skipped its body (Codex P2, round 2). *)
      (* `(include_subdirs unqualified)` puts a descendant directory's modules into a stanza's
         module set, which the per-directory census below does not model -- and the module would then
         be claimed by nobody, taking its environment reads out of the check silently (Codex P2,
         round 10). Refused rather than approximated: a scan that cannot place a file's modules
         should say so, which is the same answer it gives an unresolvable key. *)
      (* `(include_subdirs …)` is a STANZA, not a field of one: asking `Scan.field` for it searched
         the children of every other stanza and never fired, so the refusal this was supposed to be
         did not exist (Codex P2, round 11 -- my own round-10 fix, dead on arrival). *)
      (* Through `Scan.walk`, so a directive inside a `(subdir gen …)` is reached too: looping the
         top-level forms let a nested one bypass the refusal entirely (Codex P2, round 13 -- the
         second defect in this one refusal, the first being that it matched nothing at all). *)
      check_per_directory (fun _subdir stanzas ->
          List.iter stanzas ~f:(fun stanza ->
              match stanza with
              | Sexp.List (Sexp.Atom "include_subdirs" :: args)
                when not (List.mem (List.concat_map args ~f:Scan.atoms) "no" ~equal:String.equal) ->
                  fail
                    (Printf.sprintf
                       "%s declares `(include_subdirs %s)`, which puts a descendant directory's \
                        modules into its stanzas' module sets -- this check derives a stanza's \
                        modules from its own directory, so it cannot say which modules those \
                        stanzas own, nor which environment reads go with them. Teach it the mode, \
                        or keep the guard's modules beside their dune file"
                       dune_file
                       (String.concat ~sep:" " (List.concat_map args ~f:Scan.atoms)))
              | _ -> ()));
      check_per_directory (fun subdir group ->
          let here = Scan.in_subdir dir subdir in
          let directory = if String.is_empty here then "." else here in
          let directory_modules =
            List.filter_map source_files ~f:(fun (path, _) ->
                if String.equal (Stdlib.Filename.dirname path) directory then
                  Some (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))
                else None)
          in
          List.iter group ~f:(fun stanza ->
              let kind =
                match Scan.head stanza with
                | Some (("test" | "tests" | "executable" | "executables") as kind) -> Some kind
                (* EVERY library, and always as a refusal. An `(inline_tests (deps …))` declaration
                   invalidates the inline-test runner alone -- the library stays linkable, and a
                   standalone test that links it reuses its output across a change of the variable
                   the library module reads (Codex P2, round 7). So the inline-test case is not a
                   licence either, and the rule is the one `Artifact_in_library` already states: a
                   read that goes stale belongs to an executable's own modules. *)
                | Some "library" -> Some "library"
                | _ -> None
              in
              match kind with
              | None -> ()
              | Some kind ->
                  let modules = Scan.modules_of ~directory_modules group stanza in
                  let source_of_module module_name =
                    Option.map (source_of ~dir:here module_name) ~f:(fun on_disk ->
                        (module_name ^ ".ml", In_channel.read_all on_disk))
                  in
                  let path_of source = Scan.in_subdir here source in
                  let named () =
                    match Scan.names_of stanza with name :: _ -> name | [] -> "<unnamed>"
                  in
                  (* Where the declaration has to sit is dune's semantics, not one rule: a `(test)`
                     runs under its own `(deps …)` and an inline-test library under
                     `(inline_tests (deps …))`, while an `(executable)` has no `deps` field at all,
                     so every rule that RUNS it carries the declaration -- and EVERY one, since dune
                     invalidates each rule on its own deps and the undeclared one would serve its
                     previous result whatever its neighbours say. Reading the whole file's
                     declarations instead was the latitude `config_dep_completeness` gives the
                     `ocannl_config` dep, and it is too loose here: with six rules running
                     `profile_precedence.exe`, one of them dropping a key still passed (Codex P1,
                     round 1). *)
                  (* A stanza dune runs ITSELF may still pin: `(test … (action (setenv OCANNL_X ""
                     (run %{test}))))` fixes the value for every run of it, so demanding a
                     declaration besides would be asking for a dependency the run cannot have (Codex
                     P2, round 12). Derived from the stanza's own action, the same way an
                     executable's runners are. *)
                  let own_pins stanza =
                    match List.map (Scan.runs_of ~subdir stanza) ~f:(fun (_, pins) -> pins) with
                    | [] -> Set.empty (module String)
                    | first :: rest -> List.fold rest ~init:first ~f:Set.inter
                  in
                  let programs =
                    match kind with
                    | "executable" | "executables" ->
                        List.map
                          (Scan.program_runners ~subdir ~runner_stanzas:all_group_runners group
                             stanza) ~f:(fun (name, runners) ->
                            ( name,
                              Scan.program_modules stanza ~modules ~name,
                              name ^ ".exe",
                              (* The pins come with the runner, scoped to the runs of THIS program:
                                 a `setenv` around a helper beside the subject pins nothing here,
                                 and one around the subject is not undone by an unpinned helper
                                 (Codex P1 round 1, P2 round 4). *)
                              List.map runners ~f:(fun (r, pins) -> (Scan.field r "deps", pins)) ))
                    (* Against an EMPTY runner list, which is the same shape an executable nothing
                       runs gets and says the same thing: there is no `deps` field in reach that a
                       change of the variable could invalidate for every process that links it. *)
                    | "library" -> [ (named (), modules, "whatever links it", []) ]
                    | _ ->
                        [
                          (named (), modules, "it", [ (Scan.field stanza "deps", own_pins stanza) ]);
                        ]
                  in
                  List.iter programs ~f:(fun (name, own_modules, program, runners) ->
                      let where =
                        Printf.sprintf "%s%s, %s %s" dune_file
                          (if String.is_empty subdir then "" else " (subdir " ^ subdir ^ ")")
                          kind name
                      in
                      (* A module the scan was handed no source for is one it cannot answer for, and
                         reading it as a module with no reads is the silent direction: a new source
                         root added without extending this rule's globs would take its guards out of
                         the check without anything saying so (Codex P2, round 8). It is reported,
                         except where the directory itself was never scanned -- that boundary is
                         stated in the golden and reported by the gate half above. *)
                      (* A module dune is TOLD has no implementation is not a missing input: an
                         `.mli` performs no run-time read, so there is nothing for this check to
                         look at (Codex P2, round 9). *)
                      let without_implementation =
                        List.concat_map [ "modules_without_implementation"; "virtual_modules" ]
                          ~f:(fun field ->
                            match Scan.field stanza field with
                            | Some args ->
                                List.filter_map args ~f:(function
                                  | Sexp.Atom m -> Some (String.lowercase m)
                                  | _ -> None)
                            | None -> [])
                      in
                      List.iter own_modules ~f:(fun module_name ->
                          if
                            Option.is_none (source_of_module module_name)
                            && (not
                                  (List.mem without_implementation (String.lowercase module_name)
                                     ~equal:String.equal))
                            && Set.mem scanned_dirs directory
                          then
                            fail
                              (Printf.sprintf
                                 "%s names the module `%s`, and this check was handed no `%s.ml` \
                                  from %s to read -- a module it cannot see is one whose \
                                  environment reads it cannot check, which looks exactly like a \
                                  module that makes none. Add the directory to the rule's globs, \
                                  or name the generated source among them"
                                 where module_name (String.lowercase module_name) directory));
                      let sources =
                        List.filter_map own_modules ~f:source_of_module
                        |> List.filter ~f:(fun (source, _) ->
                            not (String.equal (path_of source) env_reader_home))
                      in
                      let reads =
                        List.map sources ~f:(fun (source, content) ->
                            ( source,
                              if Sources.could_read_env_var content then
                                Sources.env_reader_reads_in_source content
                              else { Sources.reader_keys = []; reader_unresolved = [] } ))
                      in
                      (* Every reach is resolved to a finite set of keys or REPORTED. A reach the
                         scan cannot follow used to fall back on the program's string literals,
                         which is a superset where the key list is written in the program and says
                         nothing where it is not -- so one incidental literal naming a real key made
                         an unresolved reach look answered, and the check reported success having
                         proven nothing about it (Codex P2, round 4). *)
                      List.iter reads ~f:(fun (source, r) ->
                          List.iter r.Sources.reader_unresolved ~f:(fun what ->
                              fail
                                (Printf.sprintf
                                   "%s: %s/%s reaches the environment reader with a key this scan \
                                    cannot resolve to a finite set -- %s. There is then nothing to \
                                    hold the `(env_var …)` declarations against, and a check that \
                                    carried on would report success having proven nothing. Iterate \
                                    a list of string literals this program's modules define, or \
                                    spell the key at the call"
                                   where directory source what)));
                      (* Normalized before the registry is consulted: `read_env_var` builds its
                         variable through `Utils.env_var_name`, which UPPERCASES, so `read_env_var
                         "PROFILE"` reads the same `OCANNL_PROFILE` as the lowercase spelling. A
                         case-sensitive membership test dropped it as an unknown key and asked for
                         no declaration -- a silent pass over a variable the guard does observe
                         (Codex P2, round 5).

                         Every resolved key is asked for, KNOWN OR NOT: the reader builds and
                         consults `OCANNL_<KEY>` whatever the registry says, so a misspelled key is
                         still a variable the run depends on, and filtering it out recorded neither
                         a requirement nor a refusal (Codex P2, round 11). An unknown one cannot be
                         declared -- the sibling check refuses a declaration naming no key OCANNL
                         reads -- so a pin is the way to satisfy it, which is what the synthetic
                         keys in `config_var_spellings` already do. *)
                      let keys =
                        List.concat_map reads ~f:(fun (_, r) -> r.Sources.reader_keys)
                        |> List.map ~f:String.lowercase
                        |> List.dedup_and_sort ~compare:String.compare
                      in
                      List.iter keys ~f:(fun key ->
                          let var = Utils.env_var_name key in
                          let answers (deps, pins) =
                            Scan.declares_env_var deps var
                            (* A variable the rule pins with `(setenv …)` cannot arrive from the
                               ambient environment, so that run does not depend on it. By SCOPE, and
                               at every run the action makes: a `progn` pinning one branch has not
                               pinned another. *)
                            || Set.mem pins var
                          in
                          match (runners, List.filter runners ~f:(fun r -> not (answers r))) with
                          | [], _ when String.equal kind "library" ->
                              fail
                                (Printf.sprintf
                                   "%s reads the configuration key `%s` straight from the \
                                    environment, from a library module -- every executable that \
                                    links the library reads it, so the requirement would fall on \
                                    every stanza that links it, where nothing follows it. An \
                                    `(inline_tests (deps …))` declaration is not a licence either: \
                                    it invalidates the inline-test runner alone, and leaves the \
                                    other linkers stale. Move the read into the executable's own \
                                    modules, where `(env_var %s)` can answer for it"
                                   where key var)
                          | [], _ ->
                              fail
                                (Printf.sprintf
                                   "%s reads the configuration key `%s` straight from the \
                                    environment, and no stanza in this file runs %s -- an \
                                    `(executable)` has no `deps` field, so the `(env_var %s)` \
                                    declaration goes on the rule that RUNS it, and this scan can \
                                    find none"
                                   where key program var)
                          | _, [] ->
                              guard_table :=
                                ( where,
                                  var,
                                  Printf.sprintf "%d run%s of %s" (List.length runners)
                                    (if List.length runners = 1 then "" else "s")
                                    program )
                                :: !guard_table
                          | _, missing ->
                              fail
                                (Printf.sprintf
                                   "%s reads the configuration key `%s` straight from the \
                                    environment through `Utils.%s`, and %d of the %d run%s of %s \
                                    neither declares `(env_var %s)` nor pins the variable -- dune \
                                    then serves that run's previous output across a change of it, \
                                    so the guard that would have reported the variable never runs. \
                                    Add the declaration to every rule that runs %s, or pin the \
                                    variable there with `(setenv %s …)`%s"
                                   where key Sources.env_reader (List.length missing)
                                   (List.length runners)
                                   (if List.length runners = 1 then "" else "s")
                                   program var program var
                                   (if Set.mem Utils.known_config_keys key then ""
                                    else
                                      Printf.sprintf
                                        ". `%s` is no configuration key OCANNL reads, so it cannot \
                                         be declared -- pin it, or fix the spelling"
                                        key))))));
      (* Every directory-scoped check above plugs into this one traversal. Keep the reverse: checks
         run in source order, so diagnostics and tables remain stable while the seam changes. *)
      per_directory file_stanzas ~checks:(List.rev !directory_checks);
      check_family_members ();
      artifact_by_file := (dune_file, !artifact_subjects) :: !artifact_by_file);
  (* Every static format a repository scanner hands to a Verdict refusal or claim, related to all
     permanent control goldens. The filename conventions are the repository's existing control
     vocabulary; a scan whose live golden embeds its controls announces the same thing with the
     heading [Synthetic controls:]. Production goldens without either signal cannot accidentally
     answer. *)
  let refusal_diagnostics =
    Set.to_list !scanner_sources
    |> List.concat_map ~f:(fun source ->
        match List.Assoc.find sources ~equal:String.equal source with
        | None -> [] (* already refused at the derivation site above *)
        | Some on_disk ->
            Refusals.diagnostics (In_channel.read_all on_disk)
            |> List.map ~f:(fun diagnostic -> (source, diagnostic)))
  in
  let refusal_exemptions : (string * string) list = [] in
  let exemption_map = Map.of_alist_exn (module String) refusal_exemptions in
  let refusal_exemptions_used = ref (Set.empty (module String)) in
  let refusal_key source diagnostic =
    Printf.sprintf "%s:%d:%s" source diagnostic.Refusals.line diagnostic.Refusals.identity
  in
  let refusal_coverage = Refusals.coverage ~control_text (List.map refusal_diagnostics ~f:snd) in
  let orphan_refusals =
    List.zip_exn refusal_diagnostics refusal_coverage
    |> List.filter_map ~f:(fun ((source, diagnostic), covered) ->
        if covered then None
        else
          let key = refusal_key source diagnostic in
          if Map.mem exemption_map key then (
            refusal_exemptions_used := Set.add !refusal_exemptions_used key;
            None)
          else Some (source, diagnostic))
  in
  List.iter orphan_refusals ~f:(fun (source, diagnostic) ->
      fail
        (Printf.sprintf
           "%s:%d scanner refusal `%s` appears in no permanent control golden -- exercise that \
            refusal in a *_cases.expected or *_control.expected test, add it to an embedded \
            `Synthetic controls:` section, or exempt the exact fragment with a reason"
           source diagnostic.Refusals.line diagnostic.fragment));
  let stale_refusal_exemptions =
    Set.diff
      (Set.of_list (module String) (List.map refusal_exemptions ~f:fst))
      !refusal_exemptions_used
  in
  if not (Set.is_empty stale_refusal_exemptions) then
    fail
      (Printf.sprintf
         "scanner-refusal exemptions no extracted diagnostic needs any more -- drop them: %s"
         (String.concat ~sep:", " (Set.to_list stale_refusal_exemptions)));
  let stale =
    Set.diff (Set.of_list (module String) (List.map exempt_declarations ~f:fst)) !exemptions_used
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted declarations no dune file makes any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  (* The coverage of the gate half, stated rather than assumed: a library outside these directories
     has no sources here to check its `preprocessor_deps` against, and says so above if it declares
     a gate at all -- but a gate it FAILS to declare is invisible from here, so the boundary belongs
     in the golden where widening it is a reviewable diff. *)
  printf "Directories whose sources this scan reads:\n";
  Set.iter scanned_dirs ~f:(fun dir -> printf "  %s\n" (if String.is_empty dir then "." else dir));
  printf "\nConfiguration keys tracked as ambient dependencies, in the one spelling OCANNL reads:\n";
  Set.iter !tracked_keys ~f:(printf "  %s\n");
  printf "\nPer-module tracing gates, and the library whose preprocessor_deps declares each:\n";
  List.sort !gate_table ~compare:(fun (_, a, _) (_, b, _) -> String.compare a b)
  |> List.iter ~f:(fun (where, gate, source) -> printf "  %-30s %s (%s)\n" gate where source);
  printf
    "\nAmbient variables a module reads by name at run time, and the stanza that declares each:\n";
  List.sort !read_table ~compare:(fun (_, a, _) (_, b, _) -> String.compare a b)
  |> List.iter ~f:(fun (where, read, source) -> printf "  %-30s %s (%s)\n" read where source);
  printf
    "\n\
     Configuration keys a program reads straight from the environment through `Utils.%s`, and how\n\
     many runs of it answer for each (gh-ocannl-749). A key read this way cannot be outranked by\n\
     a commandline flag or a config file, so every run depends on the variable unconditionally --\n\
     and EVERY one of them declares it or pins it with `setenv`, since dune invalidates each rule\n\
     on its own deps.\n"
    Sources.env_reader;
  List.sort !guard_table ~compare:(fun (wa, a, sa) (wb, b, sb) ->
      match String.compare wa wb with
      | 0 -> ( match String.compare a b with 0 -> String.compare sa sb | c -> c)
      | c -> c)
  |> List.iter ~f:(fun (where, var, source) -> printf "  %-38s %s (%s)\n" var where source);
  let stale_gateless =
    Set.diff (Set.of_list (module String) (List.map gateless_dirs ~f:fst)) !gateless_used
  in
  if not (Set.is_empty stale_gateless) then
    fail
      (Printf.sprintf
         "directories exempted from the ambient gate that no longer run tests -- drop them from \
          the exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale_gateless)));
  printf "\nAmbient environment gates, by dune file and every alias whose build runs one:\n";
  List.sort !gated ~compare:(fun (a, x) (b, y) ->
      match String.compare a b with 0 -> String.compare x y | c -> c)
  |> List.iter ~f:(fun (dune_file, alias) -> printf "  %-40s @%s\n" dune_file alias);
  List.iter gateless_dirs ~f:(fun (dir, why) -> printf "  %s -- no gate: %s\n" dir why);
  printf "\nDeclarations of a name OCANNL does not read as a configuration key, exempt by design:\n";
  List.iter exempt_declarations ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf
    "\n\
     Artifact-directory declarations, by dune file. A stanza whose modules call\n\
     `Test_utils.Generated.init` declares `(env_var %s)`\n\
     where dune runs it -- in its own `(deps ...)`, or, for an `(executable)` which has none, in\n\
     the rule that runs it (gh-ocannl-723). What is held here is which VERDICTS a file's stanzas\n\
     draw, not how many draw each: a tally would move whenever a test is added, and the per-stanza\n\
     table goes to stderr.\n"
    Scan.artifact_env_var;
  printf "  %s\n"
    (if repository_census then
       Printf.sprintf "the census covers every scan root, so the floor of %d callers applies to it"
         artifact_caller_floor
     else
       "the census does not cover the repository's scan roots, so only the relationship is asked \
        of it");
  List.sort !artifact_by_file ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.iter ~f:(fun (dune_file, subjects) ->
      if not (List.is_empty subjects) then
        printf "  %s: %s\n" dune_file
          (List.map subjects ~f:(fun s -> Scan.artifact_verdict_name s.Scan.artifact_verdict)
          |> List.dedup_and_sort ~compare:String.compare
          |> String.concat ~sep:", "));
  (* gh-ocannl-659. The golden holds which backend WORDS a dune file's markers use, not how many
     stanzas carry each: a tally there would move on every test added anywhere in the repository,
     which is the churn gh-ocannl-665 took out of `config_dep_completeness` for the same reason. The
     per-stanza classification is not centralized here at all -- it lives in the marker comment next
     to the stanza, which is the point of putting it there; what this file pins is that a directory
     did not quietly lose a whole class of them. *)
  printf
    "\n\
     Backend declarations, by dune file. Every stanza that runs an executable either declares\n\
     `(env_var %s)`, or carries the marker comment\n\
    \    ; %s <%s> -- <reason>\n\
     inside its parentheses. What is held here is which WORDS a file's markers use, not how many\n\
     stanzas carry each: the per-stanza reasons live next to the stanzas, where the next author\n\
     will copy them from, and the tallies go to stderr (gh-ocannl-659, gh-ocannl-665).\n"
    Scan.backend_env_var Scan.marker_sentinel
    (String.concat ~sep:"|" Scan.marker_backends);
  List.sort !by_file ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.iter ~f:(fun (dune_file, present) ->
      printf "  %s: %s\n" dune_file
        (if List.is_empty present then "nothing that runs a test executable"
         else String.concat ~sep:"; " present));
  (* gh-ocannl-783. The golden holds which FAMILIES a dune file has derived members of, not how many
     stanzas each has -- the same reason the two sections above hold words and verdicts rather than
     tallies: a count would move whenever a Metal test or a lifecycle probe is added, and what is
     worth a reviewable diff is a directory acquiring or losing a family. The per-member table, with
     the alias that reaches each, goes to stderr. *)
  printf
    "\n\
     Focused aggregate families, by dune file (gh-ocannl-783). Each family alias is spelled\n\
     identically in every directory that has members, so `dune build @<family>` from the workspace\n\
     root runs the whole family; membership is derived from what a member stanza declares, and a\n\
     member the family's `(alias (name <family>) (deps ...))` stanza omits fails this scan.\n";
  List.map !family_table ~f:(fun (dune_file, family, _, _) -> (dune_file, family))
  |> List.dedup_and_sort ~compare:Poly.compare
  |> List.group ~break:(fun (a, _) (b, _) -> not (String.equal a b))
  |> List.iter ~f:(fun group ->
      printf "  %s: %s\n"
        (fst (List.hd_exn group))
        (String.concat ~sep:", " (List.map group ~f:snd)));
  eprintf
    "Focused-aggregate members, and the family alias that reaches each (not diffed -- see \
     gh-ocannl-665):\n\
     %s\n"
    (String.concat ~sep:"\n"
       (List.map (List.sort !family_table ~compare:Poly.compare)
          ~f:(fun (dune_file, family, identity, aggregated) ->
            Printf.sprintf "  %-24s %-40s the %s%s" ("@" ^ family) dune_file identity
              (if aggregated then "" else " -- NOT AGGREGATED"))));
  eprintf
    "Backend classification of every stanza that runs an executable (not diffed -- see \
     gh-ocannl-665):\n\
     %s\n"
    (String.concat ~sep:"\n" (List.rev !classification));
  eprintf "Totals: %d such stanzas, against a raw-text floor of %d.\n" !placed_subjects
    !subject_floor;
  (* The two populations, compared as SORTED LISTS OF STANZAS rather than as totals. A number would
     say "one short" and leave which stanza to arithmetic; these say which file and line each reader
     is alone on, and the claim below is about the stanzas themselves (gh-ocannl-708). Both
     directions, because they fail differently: a stanza only the floor names is the walk going
     blind, and one only the walk places is a stanza whose enforcement nothing independent vouches
     for. *)
  let placed_identities = List.sort (List.map !walk_places ~f:fst) ~compare:String.compare in
  let floored_identities = List.sort !floor_names ~compare:String.compare in
  let detail = Map.of_alist_multi (module String) !walk_places in
  let describe identity =
    match Map.find detail identity with Some (what :: _) -> what | _ -> identity
  in
  (* As MULTISETS, not sets. Two stanzas opening on the same line of the same file share an
     identity, and comparing deduplicated lists would let the floored one answer for the other --
     the collapse `config_dep_completeness` compares (directory, executable) pairs with multiplicity
     to avoid (Codex P2, rounds 2 and 3 of PR #343). *)
  let counts identities =
    List.fold identities
      ~init:(Map.empty (module String))
      ~f:(fun tally identity ->
        Map.update tally identity ~f:(fun n -> 1 + Option.value n ~default:0))
  in
  let excess these those =
    let those = counts those in
    Map.to_alist (counts these)
    |> List.concat_map ~f:(fun (identity, mine) ->
        let theirs = Option.value (Map.find those identity) ~default:0 in
        List.init (Int.max 0 (mine - theirs)) ~f:(fun _ -> identity))
  in
  let walk_only = excess placed_identities floored_identities in
  let floor_only = excess floored_identities placed_identities in
  (* The claim is quantified over the UNION, through `Verdict.p_all`, which carries the
     non-emptiness guard with it: a scan that stopped reading dune files altogether would leave two
     empty populations, and "the same stanzas" holds vacuously of nothing (gh-ocannl-729). *)
  let population =
    List.dedup_and_sort (placed_identities @ floored_identities) ~compare:String.compare
  in
  let placed_counts = counts placed_identities and floored_counts = counts floored_identities in
  let named_by_both identity =
    Option.value (Map.find placed_counts identity) ~default:0
    = Option.value (Map.find floored_counts identity) ~default:0
  in
  (match walk_only with
  | [] -> eprintf "Every one of them has a second reader's floor under it.\n"
  | walk_only ->
      eprintf
        "The %d standing on the walk alone -- a site is placed and the raw reader names nothing \
         there. Teach `Scan.raw_stanza_of` the shape, or the rule applies to a stanza nothing \
         independent vouches for:\n\
         %s\n"
        (List.length walk_only)
        (String.concat ~sep:"\n" (List.map walk_only ~f:(fun i -> "  " ^ describe i))));
  (match floor_only with
  | [] -> ()
  | floor_only ->
      eprintf
        "And the %d the raw reader names with no site placed -- the walk reading the file with a \
         hole in it:\n\
         %s\n"
        (List.length floor_only)
        (String.concat ~sep:"\n" (List.map floor_only ~f:(fun i -> "  " ^ i))));
  printf "\n";
  Verdict.p
    "every stanza that runs an executable either declares the backend variable or says in place \
     why it does not"
    (!xor_violations = 0);
  Verdict.p
    "every marker the text spells was read as one, and every stanza the raw text shows running an \
     executable has a site placed for it"
    (!marker_holes = 0);
  Verdict.p_all
    "every stanza either reader names as running an executable is named by the other too" population
    ~f:named_by_both;
  (* The relationship, and a floor under the population that carries it (gh-ocannl-729): a
     derivation that stopped finding members -- a marker grammar change, a renamed instrumentation
     module, a glob that stopped reaching the sources -- would leave "every member is aggregated"
     true of nothing and print the line a healthy repository prints. The floor is a statement about
     the REPOSITORY, so it is asked only of a run that has it: the control's synthetic trees
     legitimately contain one member or none, and the relationship is checked over them either way.
     Same shape, and the same reasoning, as the artifact census floor below. *)
  let family_members = List.sort !family_table ~compare:Poly.compare in
  let family_unaggregated = List.filter family_members ~f:(fun (_, _, _, reached) -> not reached) in
  let found family =
    List.count family_members ~f:(fun (_, alias, _, _) -> String.equal alias family.family_alias)
  in
  let family_floor_met =
    (not repository_census) || List.for_all families ~f:(fun f -> found f >= f.family_floor)
  in
  List.iter families ~f:(fun family ->
      if repository_census && found family < family.family_floor then
        fail
          (Printf.sprintf
             "the repository's `%s` derivation finds %d member%s, against a floor of %d -- that \
              derivation has stopped finding them, and a family whose membership is empty is \
              aggregated completely by an empty stanza"
             family.family_alias (found family)
             (if found family = 1 then "" else "s")
             family.family_floor));
  eprintf "Focused-aggregate members derived, per family, against each family's own floor:\n%s\n"
    (String.concat ~sep:"\n"
       (List.map families ~f:(fun family ->
            Printf.sprintf "  %-20s %d (floor %d)" ("@" ^ family.family_alias) (found family)
              family.family_floor)));
  (* Quantified over the dune files scanned -- non-empty in every run, the controls' synthetic trees
     included -- rather than over the members, which a control tree legitimately has none of: the
     member population's own floor is the repository-gated one folded in beside it. *)
  let unaggregated_files =
    Set.of_list
      (module String)
      (List.map family_unaggregated ~f:(fun (dune_file, _, _, _) -> dune_file))
  in
  Verdict.p_all
    "every focused-aggregate member is reached by its family alias, and a repository-wide \
     derivation finds enough of them for the rule to be about something"
    dune_files ~f:(fun (dune_file, _) ->
      (not (Set.mem unaggregated_files dune_file)) && family_floor_met);
  eprintf
    "Artifact-directory verdict of every stanza whose modules call Test_utils.Generated.init, or \
     which declares %s without one (not diffed -- see gh-ocannl-665):\n\
     %s\n"
    Scan.artifact_env_var
    (String.concat ~sep:"\n" (List.rev !artifact_table));
  let unclaimed =
    List.filter artifact_callers ~f:(fun path ->
        not (Set.mem !artifact_claimed (String.lowercase path)))
  in
  List.iter unclaimed ~f:(fun path ->
      fail
        (Printf.sprintf
           "%s calls `Test_utils.Generated.init` and no stanza's `(modules ...)` claims it -- the \
            rule that would require `(env_var %s)` of it never reaches it. Name the module in the \
            stanza that builds it, or hand this scan that directory's dune file"
           path Scan.artifact_env_var));
  let floor_met =
    (not repository_census) || List.length artifact_callers >= artifact_caller_floor
  in
  if not floor_met then
    fail
      (Printf.sprintf
         "the repository's census finds %d source%s calling `Test_utils.Generated.init`, against a \
          floor of %d -- the census has stopped finding them, and a relationship checked over an \
          empty set holds for nothing"
         (List.length artifact_callers)
         (if List.length artifact_callers = 1 then "" else "s")
         artifact_caller_floor);
  eprintf "Sources calling Test_utils.Generated.init: %d, against a floor of %d.\n"
    (List.length artifact_callers) artifact_caller_floor;
  (* Over the sources scanned, for the reason the family claim above gives: a control tree may
     legitimately hold no caller at all, and the callers' own floor is the repository-gated one. *)
  let unclaimed_sources = Set.of_list (module String) unclaimed in
  Verdict.p_all
    "every source that calls Test_utils.Generated.init is claimed by some stanza's modules, and a \
     repository-wide census finds enough of them for the rule to be about something"
    source_files ~f:(fun (path, _) -> (not (Set.mem unclaimed_sources path)) && floor_met);
  Verdict.p
    "every stanza whose modules call Test_utils.Generated.init declares OCANNL_BUILD_FILES_PREFIX \
     where dune runs it, and every declaration of it has a caller behind it"
    (!artifact_violations = 0);
  if repository_census then (
    let derived_scanner_sources =
      Set.to_list !scanner_sources |> List.sort ~compare:String.compare
    in
    let manifest_sources = List.sort Refusal_manifest.sources ~compare:String.compare in
    printf
      "\n\
       Scanner refusal controls (gh-ocannl-800). Sources come from repository-wide scan rules;\n\
       controls are *_cases.expected, *_control.expected, or a live scan golden with an explicit\n\
       `Synthetic controls:` section. Printf substitutions do not decide coverage.\n";
    printf "  statically extracted diagnostics: %d (details on stderr)\n"
      (List.length refusal_diagnostics);
    printf "  named exemptions with reasons:\n";
    if List.is_empty refusal_exemptions then printf "    (none)\n"
    else List.iter refusal_exemptions ~f:(fun (key, reason) -> printf "    %s -- %s\n" key reason);
    eprintf "Repository scanner refusal diagnostics (not diffed):\n";
    List.iter2_exn refusal_diagnostics refusal_coverage ~f:(fun (source, diagnostic) covered ->
        eprintf "  %s:%d  %s%s\n" source diagnostic.Refusals.line diagnostic.fragment
          (if covered then ""
           else if Map.mem exemption_map (refusal_key source diagnostic) then " -- EXEMPT"
           else " -- ORPHAN"));
    Verdict.p_all ~min:10 "repository-wide scan rules resolve to scanner sources"
      (Set.to_list !scanner_sources) ~f:(fun source ->
        List.Assoc.mem sources source ~equal:String.equal);
    Verdict.p "the refusal-control manifest source set equals the derived scanner census"
      (List.equal String.equal manifest_sources derived_scanner_sources);
    Verdict.p_all ~min:10 "the permanent control-golden corpus is present" control_goldens
      ~f:(fun (_path, on_disk) -> not (String.is_empty (In_channel.read_all on_disk)));
    let orphan_keys =
      Set.of_list
        (module String)
        (List.map orphan_refusals ~f:(fun (source, diagnostic) -> refusal_key source diagnostic))
    in
    Verdict.p_all
      "every statically recoverable scanner refusal diagnostic appears in a control golden or has \
       a named reasoned exemption"
      refusal_diagnostics ~f:(fun (source, diagnostic) ->
        (not (Set.mem orphan_keys (refusal_key source diagnostic)))
        && Set.is_empty stale_refusal_exemptions));
  if not (Verdict.any_failed ()) then
    printf
      "\n\
       OK: every `(env_var ...)` addressed to OCANNL names a spelling a run reads, every test \
       directory carries the ambient gate, and every per-module tracing gate is declared by the \
       library whose modules read it.\n";
  if repository_census then
    List.Assoc.find sources "test/operations/env_var_deps.ml" ~equal:String.equal
    |> Option.iter ~f:(fun source -> Refusal_manifest.print source)

(* gh-ocannl-723's negative control, and why it runs the checker rather than inspecting it.

   A control written from today's corpus encodes the ABSENCE of the shape it is about: every stanza
   in this repository that calls `Test_utils.Generated.init` declares the variable, so a check that
   reported nothing and a check that decided nothing would both pass over it, and the second is what
   an unexercised rule quietly becomes. So the rule is put to a stanza/source pair the repository
   does not contain: a synthetic tree of four dune files and one source, handed to THIS executable
   in a child process, once with the declaration and once without.

   Everything but the one declaration is held fixed between the two runs, so the difference in
   verdict is the rule's and nothing else's. The tree is built to satisfy the file's other rules --
   the fixtures for the exemption and gateless lists are DERIVED from those lists, so they cannot
   drift from them -- which buys the sharper claim: the violating tree exits 1 and names the stanza,
   and the legitimate one exits 0. *)

let control_root_paths = [ "t/dune"; "t/probe.ml"; "t/nested/probe2.ml" ]
let control_probe = "let () = Test_utils.Generated.init ~backend_name:\"cc\"\n"

(* The subject: an `(executable)` whose one module calls the initializer, plus the rule that runs it
   -- an executable has no `deps` field, so the rule is where the declaration has to go, and putting
   the pair in the control is what keeps that placement checked. *)
let control_subject ~declares =
  Printf.sprintf
    {dune|(executable
 (name probe)
 (modules probe)
 (libraries test_utils))

(rule
 ; ocannl-backend: none -- a synthetic control fixture, which runs on no device at all.
 (target probe.actual)
 (deps
  ocannl_config
%s  %%{dep:probe.exe})
 (action
  (with-stdout-to
   %%{target}
   (run ./probe.exe))))

; A second, always-correct pair inside a `(subdir …)`. It is not the pair under control -- it
; declares in both runs -- but it is what makes the LEGITIMATE run's exit status depend on the walk
; descending: a scan that only read the top level would leave `nested/probe2.ml` claimed by nobody,
; which the census check reports (Codex P2, round 3).
(subdir
 nested
 (executable
  (name probe2)
  (modules probe2)
  (libraries test_utils))
 (rule
  ; ocannl-backend: none -- the same fixture, one directory down.
  (target probe2.actual)
  (deps
   ocannl_config
   (env_var %s)
   %%{dep:probe2.exe})
  (action
   (with-stdout-to
    %%{target}
    (run ./probe2.exe)))))
|dune}
    (if declares then Printf.sprintf "  (env_var %s)\n" Scan.artifact_env_var else "")
    Scan.artifact_env_var

(* The rest of the tree exists only so that the two runs differ in ONE verdict. Both lists below are
   checked for staleness against the files that make them necessary, so a tree without those files
   fails for reasons that have nothing to do with the rule under control. *)
let control_context () =
  let by_file =
    List.map exempt_declarations ~f:(fun (key, _) ->
        match String.lsplit2 key ~on:':' with
        | Some (file, name) -> (file, name)
        | None -> (key, key))
    |> Map.of_alist_multi (module String)
  in
  let exempt_files =
    List.map (Map.to_alist by_file) ~f:(fun (file, names) ->
        let declarations =
          String.concat ~sep:"" (List.map names ~f:(Printf.sprintf "  (env_var %s)\n"))
        in
        ( file,
          "(rule\n (target exempt.fixture)\n (deps\n" ^ declarations
          ^ " )\n (action\n  (with-stdout-to\n   %{target}\n   (echo \"\"))))\n" ))
  in
  let gateless_files =
    List.map gateless_dirs ~f:(fun (file, _) ->
        ( file,
          Printf.sprintf "(test\n (name gateless)\n (deps\n  (env_var %s))\n (modules gateless))\n"
            Scan.backend_env_var ))
  in
  exempt_files @ gateless_files

let write_file path data =
  let dir = Stdlib.Filename.dirname path in
  let rec mkdirs dir =
    if not (String.equal dir Stdlib.Filename.current_dir_name || Stdlib.Sys.file_exists dir) then (
      mkdirs (Stdlib.Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdirs dir;
  Out_channel.write_all path ~data

(* The tree is this process's to remove: it was made by `Filename.temp_dir`, which creates a fresh
   directory nothing else holds. Removal is best effort -- a control that failed to tidy up is not a
   control that decided wrongly, and the temporary directory is the operating system's to reclaim
   either way. *)
let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

(* Through temporary FILES rather than pipes, for the reason `generated_provenance` gives: the child
   writes to both streams, and reading two pipes in sequence deadlocks as soon as the one not being
   read fills its buffer. *)
let run_checker ~root ~exe args =
  let capture suffix = Stdlib.Filename.temp_file "evd_control" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture p = Unix.openfile p [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let here = Stdlib.Sys.getcwd () in
  Stdlib.Sys.chdir root;
  let pid = Unix.create_process exe (Array.of_list (exe :: args)) Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Stdlib.Sys.chdir here;
  Unix.close out;
  Unix.close err;
  let text = In_channel.read_all out_path ^ In_channel.read_all err_path in
  (try Unix.unlink out_path with Unix.Unix_error _ -> ());
  (try Unix.unlink err_path with Unix.Unix_error _ -> ());
  (status, text)

let control () =
  (* Absolute before the chdir: `Sys.executable_name` resolves the name this process was started
     with, and a relative one would name nothing from inside the temporary tree. *)
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_control" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  write_file (Stdlib.Filename.concat root "t/probe.ml") control_probe;
  write_file (Stdlib.Filename.concat root "t/nested/probe2.ml") control_probe;
  let paths = control_root_paths @ List.map context ~f:fst in
  let run ~declares =
    write_file (Stdlib.Filename.concat root "t/dune") (control_subject ~declares);
    run_checker ~root ~exe ("." :: paths)
  in
  (* The exact sentence the rule produces, so that the control observes THIS rule failing and not
     merely the child's misfortune -- the argument gh-ocannl-692 made for `generated_provenance`. *)
  let diagnostic = "calls `Test_utils.Generated.init`" in
  let report label (status, text) =
    eprintf "the control's %s run %s. Its captured output:\n%s\n" label (describe_status status)
      text
  in
  let violating = run ~declares:false in
  let legitimate = run ~declares:true in
  let violating_reported =
    (match fst violating with Unix.WEXITED 1 -> true | _ -> false)
    && String.is_substring (snd violating) ~substring:diagnostic
    && String.is_substring (snd violating) ~substring:Scan.artifact_env_var
  in
  let legitimate_passed =
    (match fst legitimate with Unix.WEXITED 0 -> true | _ -> false)
    && not (String.is_substring (snd legitimate) ~substring:diagnostic)
  in
  if not violating_reported then report "violating" violating;
  if not legitimate_passed then report "legitimate" legitimate;
  printf
    "The rule is put to a stanza this repository does not contain: an `(executable)` whose one\n\
     module calls `Test_utils.Generated.init`, run by a rule that does or does not declare\n\
     `(env_var %s)`. Nothing else differs between the two runs.\n\n"
    Scan.artifact_env_var;
  Verdict.p
    "the checker reports the stanza and exits 1 when the rule running it omits the declaration"
    violating_reported;
  Verdict.p "the same tree with the declaration added passes and says nothing about it"
    legitimate_passed;
  try remove_tree root with Unix.Unix_error _ -> ()

(* gh-ocannl-708's control, and why it is a second tree rather than a stanza added to the one above.

   The rule under control here is the RELATIONSHIP between the two readers: a stanza the walk places
   a site for is a stanza the raw-text floor names. This repository contains exactly one stanza of
   the shape that used to break it -- `benchmarks/dune` running its orchestrator as `(run python3
   %{dep:test_orchestrate.py})` -- so a control read off today's corpus would pass whether the floor
   learned the shape or the shape left the repository. Put to a tree of its own, the claim is about
   the rule: the same tool handed a file this workspace builds is seen by both readers, and handed
   nothing of ours is seen by neither. *)

let floor_subject ~handed ~declares =
  Printf.sprintf
    {dune|(rule
%s (target orchestrated.actual)
 (deps ocannl_config %s)
 (action
  (with-stdout-to
   %%{target}
   (run python3 %s))))
|dune}
    (if declares then
       " ; ocannl-backend: none -- hands a script to python3, which links no backend.\n"
     else "")
    (if handed then "%{dep:orchestrate.py}" else "orchestrate.py")
    (if handed then "%{dep:orchestrate.py}" else "orchestrate.py")

let floor_control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_floor" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  (* One source, so that the run is handed both a dune file and a source: the checker refuses a tree
     with neither, since its globs matching nothing is the failure it reports first. It calls
     nothing and reads no configuration key, which keeps the tree's only subject the rule above. *)
  write_file (Stdlib.Filename.concat root "t/noop.ml") "let () = ()\n";
  let paths = "t/dune" :: "t/noop.ml" :: List.map context ~f:fst in
  let run ~handed ~declares =
    write_file (Stdlib.Filename.concat root "t/dune") (floor_subject ~handed ~declares);
    run_checker ~root ~exe ("." :: paths)
  in
  let report label (status, text) =
    eprintf "the floor control's %s run %s. Its captured output:\n%s\n" label
      (describe_status status) text
  in
  let exited n (status, _) = match status with Unix.WEXITED m -> m = n | _ -> false in
  (* The rule reaches the stanza at all: without a marker or a declaration, the checker reports it
     by name. That is the walk's half. *)
  let reported = run ~handed:true ~declares:false in
  (* And the floor's half: with the marker, the run passes AND says every stanza it placed has a
     second reader's floor under it -- the sentence that named this stanza as the exception before
     the two readers shared the pform lists. *)
  let floored = run ~handed:true ~declares:true in
  (* The negative control. The same command handed nothing this workspace provides is a stanza
     NEITHER reader sees, so the rule does not apply and no marker is asked for. A floor that
     over-claimed here would fail this correct tree. *)
  let invisible = run ~handed:false ~declares:false in
  let unfloored_sentence = "standing on the walk alone" in
  let floored_sentence = "second reader's floor under it" in
  let reported_ok =
    exited 1 reported
    && String.is_substring (snd reported) ~substring:"runs an executable and declares neither"
    && String.is_substring (snd reported) ~substring:"t/dune"
  in
  let floored_ok =
    exited 0 floored
    && String.is_substring (snd floored) ~substring:floored_sentence
    && not (String.is_substring (snd floored) ~substring:unfloored_sentence)
  in
  let invisible_ok =
    exited 0 invisible
    && (not
          (String.is_substring (snd invisible) ~substring:"runs an executable and declares neither"))
    && not (String.is_substring (snd invisible) ~substring:unfloored_sentence)
  in
  if not reported_ok then report "reported" reported;
  if not floored_ok then report "floored" floored;
  if not invisible_ok then report "invisible" invisible;
  printf
    "\n\
     The relationship is put to a tree of one rule, which runs `python3` on a file that is or is\n\
     not one this workspace builds. Nothing else differs between the three runs.\n\n";
  Verdict.p
    "an external command handed a file this workspace builds is a stanza the rule reaches, \
     reported by name when it declares neither"
    reported_ok;
  Verdict.p "the same stanza, declaring its backend, passes with the raw-text floor naming it too"
    floored_ok;
  Verdict.p "the same command handed nothing of this workspace is a stanza neither reader sees"
    invisible_ok;
  try remove_tree root with Unix.Unix_error _ -> ()

(* gh-ocannl-783's control, and why it is a third tree.

   Every family in this repository is complete -- that is what the rule is for -- so a control read
   off today's corpus would pass whether the rule decides anything or not, which is the argument
   both controls above make. Put to a tree of its own, what is asserted is the rule: a stanza the
   derivation calls a member is reported when its family alias does not reach it, the same tree with
   the alias listing it passes, and a stanza the derivation calls no member is asked for nothing.

   Both derivations are exercised, because they are independent: the Metal one reads the stanza's
   backend marker, the lifecycle one reads its modules' sources, and a control that ran only the
   first would leave the second able to stop finding anything. *)

let family_gate ~elsewhere =
  Printf.sprintf
    {dune|(test
 ; ocannl-backend: none -- the ambient gate of this synthetic tree; it runs on no device.
 (name gate)
 (modules gate)
 (deps ocannl_config (universe))
 (libraries base))

(rule
 ; ocannl-backend: none -- the same gate, on the alias the family stanza depends on.
 (alias runtest-gate)
 (deps ocannl_config (universe))
 (action
  %s))

(alias
 (name runtest)
 (deps (alias runtest-gate)))
|dune}
    (if elsewhere then "(chdir nested (run ./gate.exe))" else "(run %{dep:gate.exe})")

(* The three shapes a member stanza takes, because the derivation reads each of them differently and
   the differences are where it went wrong (Codex P2, round 1). A `(test)` carries its marker itself
   and dune generates its alias; an `(executable)` runs nothing, so its marker is REQUIRED to sit on
   the rule that runs it and the alias to list is that rule's; and a `(tests)` is several tests
   behind one stanza, each with an alias the family has to reach. *)
type family_shape =
  | Single_test  (** `(test (name probe) …)` *)
  | Exe_with_runner  (** `(executable (name probe) …)` plus the `(rule)` that runs it *)
  | Runner_elsewhere  (** the same pair, whose rule `(chdir nested …)` runs ANOTHER `probe.exe` *)
  | Plural_tests  (** `(tests (names probe probe2) …)` -- two units behind one stanza *)
  | Inline_library  (** `(library (name probelib) (inline_tests …))`, whose alias dune generates *)
  | Subdir_member  (** the member, and its family alias, inside a `(subdir child …)` group *)
  | Public_name_runner  (** an installed executable, run by its `(public_name …)` *)
  | Public_names_crossed
      (** two installed executables, and a rule running the SECOND one's public name *)
  | Public_name_by_path
      (** an installed executable, and a rule running a FILE that shares its public name *)
  | Subdir_exe_top_runner  (** the executable in a group, the rule that runs it at the top level *)
  | Runtest_only_rule  (** a marked rule whose only alias is the directory-wide `runtest` *)
  | Subdir_ungated  (** the member and its family alias in a group with no gate of its own *)
  | Subdir_unlocked_gate
      (** a group whose actions take the training lock and whose gate does not *)
  | Subdir_alias_collision
      (** a group with a `(test (name probe))` and a hand-written rule on `runtest-probe` *)
  | Gate_elsewhere
      (** the rule on `runtest-gate` runs ANOTHER directory's `gate.exe`, so it is no gate *)
  | Sibling_defaults  (** two stanzas that omit `(modules …)`, each with its own main *)
  | Test_stanza_runner  (** an `(executable)` run by a `(test)` stanza's custom action *)
  | Runner_absolute  (** a rule running an ABSOLUTE path that ends in the local executable's name *)

let family_marker ~metal =
  Printf.sprintf
    " ; ocannl-backend: %s -- a synthetic control fixture, judged on this marker alone.\n"
    (if metal then "metal" else "cc")

let family_member_stanza ~shape ~metal =
  let marker = family_marker ~metal in
  match shape with
  | Single_test | Gate_elsewhere ->
      Printf.sprintf
        "(test\n%s (name probe)\n (modules probe)\n (deps ocannl_config)\n (libraries base))\n"
        marker
  | Plural_tests ->
      Printf.sprintf
        "(tests\n\
         %s (names probe probe2)\n\
        \ (modules probe probe2)\n\
        \ (deps ocannl_config)\n\
        \ (libraries base))\n"
        marker
  | Inline_library ->
      Printf.sprintf
        "(library\n\
         %s (name probelib)\n\
        \ (modules probe)\n\
        \ (inline_tests\n\
        \  (deps ocannl_config))\n\
        \ (libraries base))\n"
        marker
  | Subdir_member | Subdir_ungated | Subdir_unlocked_gate | Subdir_alias_collision ->
      (* Dune lets a `(subdir …)` group carry its own alias stanza, and the recursive `@<family>`
         build from the root reaches it. The family stanza therefore goes INSIDE the group here,
         which is the arrangement the check used to reject -- and the group needs an ambient gate of
         its OWN, since a gate at the top level gates the top level's aliases and nothing else. The
         `Subdir_ungated` variant omits exactly that gate. *)
      let gate =
        match shape with
        | Subdir_ungated -> ""
        | _ ->
            " (test\n\
            \  ; ocannl-backend: none -- this group's ambient gate; it runs on no device.\n\
            \  (name childgate)\n\
            \  (modules childgate)\n\
            \  (deps ocannl_config (universe))\n\
            \  (libraries base))\n\
            \ (rule\n\
            \  ; ocannl-backend: none -- the same gate, on the alias the group's aliases depend on.\n\
            \  (alias runtest-childgate)\n\
            \  (deps ocannl_config (universe))\n\
            \  (action\n\
            \   (run %{dep:childgate.exe})))\n\
            \ (alias\n\
            \  (name runtest)\n\
            \  (deps (alias runtest-childgate)))\n"
      in
      let gate_dep = match shape with Subdir_ungated -> "" | _ -> "(alias runtest-childgate) " in
      (* The lock the training tests serialize on. Taken by the member and by nothing else in the
         `Subdir_unlocked_gate` variant, which is the arrangement the repository's own rule forbids:
         one unlocked action in a directory of locked ones. *)
      let locks =
        match shape with Subdir_unlocked_gate -> "  (locks ocannl_training_test)\n" | _ -> ""
      in
      (* The collision: a hand-written rule on the alias dune generates for the group's own `(test)`
         stanza. Building that alias would run both. *)
      let collision =
        match shape with
        | Subdir_alias_collision ->
            " (rule\n\
            \  ; ocannl-backend: none -- a synthetic control fixture, judged on this marker alone.\n\
            \  (alias runtest-probe)\n\
            \  (deps ocannl_config (alias runtest-childgate))\n\
            \  (action\n\
            \   (run %{dep:childgate.exe})))\n"
        | _ -> ""
      in
      Printf.sprintf
        "(subdir\n\
        \ child\n\
         %s (test\n\
         %s  (name probe)\n\
        \  (modules probe)\n\
         %s  (deps ocannl_config)\n\
        \  (libraries base))\n\
         %s (alias\n\
        \  (name %s)\n\
        \  (deps %s(alias runtest-probe))))\n"
        gate
        (String.substr_replace_all marker ~pattern:" ; " ~with_:"  ; ")
        locks collision metal_family.family_alias gate_dep
  | Public_names_crossed ->
      (* Two installed executables behind one stanza, and a rule running the SECOND one's public
         name. Nothing here runs `probe`, so a family listing that rule aggregates `probe2` and
         nothing else. *)
      Printf.sprintf
        "(executables\n\
        \ (names probe probe2)\n\
        \ (public_names pkg.probe pkg.probe2)\n\
        \ (modules probe probe2)\n\
        \ (libraries base))\n\n\
         (rule\n\
         %s (alias probe_run)\n\
        \ (deps ocannl_config (alias runtest-gate))\n\
        \ (action\n\
        \  (run %%{bin:pkg.probe2})))\n"
        marker
  | Test_stanza_runner ->
      (* A `(test)` with a custom action as the runner: dune's focused entry point for it is the
         `runtest-harness` alias it generates. *)
      Printf.sprintf
        "(executable\n\
        \ (name probe)\n\
        \ (modules probe)\n\
        \ (libraries base))\n\n\
         (test\n\
         %s (name harness)\n\
        \ (modules harness)\n\
        \ (deps ocannl_config %%{dep:probe.exe})\n\
        \ (libraries base)\n\
        \ (action\n\
        \  (run ./probe.exe)))\n"
        marker
  | Sibling_defaults ->
      (* Two tests, neither naming its modules: dune gives each its own main and shares the rest, so
         only the one whose main reads the instrumentation is a member. *)
      Printf.sprintf
        "(test\n\
         %s (name probe)\n\
        \ (deps ocannl_config)\n\
        \ (libraries base))\n\n\
         (test\n\
         %s (name probe2)\n\
        \ (deps ocannl_config)\n\
        \ (libraries base))\n"
        marker marker
  | Runner_absolute ->
      (* The same basename, reached by an absolute path: `/probe.exe` is the system's, not ours. *)
      Printf.sprintf
        "(executable\n\
        \ (name probe)\n\
        \ (modules probe)\n\
        \ (libraries base))\n\n\
         (rule\n\
         %s (alias probe_run)\n\
        \ (deps ocannl_config (alias runtest-gate))\n\
        \ (action\n\
        \  (run /probe.exe)))\n"
        marker
  | Public_name_by_path ->
      (* The same public name, run as a path: `(run ./pkg.probe)` names a file in this directory,
         not the installed program, and `classify_command` reports both as the same string. *)
      Printf.sprintf
        "(executable\n\
        \ (name probe)\n\
        \ (public_name pkg.probe)\n\
        \ (modules probe)\n\
        \ (libraries base))\n\n\
         (rule\n\
         %s (alias probe_run)\n\
        \ (deps ocannl_config (alias runtest-gate))\n\
        \ (action\n\
        \  (run ./pkg.probe)))\n"
        marker
  | Public_name_runner ->
      (* An installed executable, whose companion rule runs it by the name it installs under --
         which is what `Scan.executables_run` reports, and not the `.exe` file. *)
      Printf.sprintf
        "(executable\n\
        \ (name probe)\n\
        \ (public_name pkg.probe)\n\
        \ (modules probe)\n\
        \ (libraries base))\n\n\
         (rule\n\
         %s (alias probe_run)\n\
        \ (deps ocannl_config (alias runtest-gate))\n\
        \ (action\n\
        \  (run %%{bin:pkg.probe})))\n"
        marker
  | Subdir_exe_top_runner ->
      (* The executable in a group, the rule that runs it at the top level -- so the family stanza
         that aggregates it belongs at the top level too, where the rule is. *)
      Printf.sprintf
        "(subdir\n\
        \ child\n\
        \ (executable\n\
        \  (name probe)\n\
        \  (modules probe)\n\
        \  (libraries base)))\n\n\
         (rule\n\
         %s (alias probe_run)\n\
        \ (deps ocannl_config (alias runtest-gate) %%{dep:child/probe.exe})\n\
        \ (action\n\
        \  (run ./child/probe.exe)))\n"
        marker
  | Runtest_only_rule ->
      (* A marked rule whose only alias is the directory-wide suite: it has no focused alias for a
         family to list, and a family listing `runtest` would run the directory. *)
      Printf.sprintf
        "(executable\n\
        \ (name probe)\n\
        \ (modules probe)\n\
        \ (libraries base))\n\n\
         (rule\n\
         %s (alias runtest)\n\
        \ (deps ocannl_config %%{dep:probe.exe})\n\
        \ (action\n\
        \  (run ./probe.exe)))\n"
        marker
  | Exe_with_runner | Runner_elsewhere ->
      (* The runner depends on the gate's alias, since its own alias is a build entry point like any
         other -- the same reason the family stanzas below do. *)
      Printf.sprintf
        "(executable\n\
        \ (name probe)\n\
        \ (modules probe)\n\
        \ (libraries base))\n\n\
         (rule\n\
         %s (alias probe_run)\n\
        \ (deps ocannl_config (alias runtest-gate) %%{dep:probe.exe})\n\
        \ (action\n\
        \  %s))\n"
        marker
        (match shape with
        | Runner_elsewhere -> "(chdir nested (run ./probe.exe))"
        | _ -> "(run ./probe.exe)")

(* Which aliases the family stanza lists, when there is one. `Every` is what a correct dune file
   writes; `First_only` lists one alias of a plural stanza, which is both the half-listed error and
   -- when only one of the two mains is a member -- the correct listing. *)
type family_listing = Every | First_only

let family_listed_aliases ~shape ~listing =
  match (shape, listing) with
  | (Subdir_member | Subdir_ungated | Subdir_unlocked_gate | Subdir_alias_collision), _ -> []
  | Runtest_only_rule, _ -> [ "runtest" ]
  | Sibling_defaults, _ -> [ "runtest-probe" ]
  | Test_stanza_runner, _ -> [ "runtest-harness" ]
  | Runner_absolute, _ -> [ "probe_run" ]
  | Gate_elsewhere, _ -> [ "runtest-probe" ]
  | ( ( Exe_with_runner | Runner_elsewhere | Public_name_runner | Public_names_crossed
      | Public_name_by_path | Subdir_exe_top_runner ),
      _ ) ->
      [ "probe_run" ]
  | Single_test, _ -> [ "runtest-probe" ]
  | Inline_library, _ -> [ "runtest-probelib" ]
  | Plural_tests, First_only -> [ "runtest-probe" ]
  | Plural_tests, Every -> [ "runtest-probe"; "runtest-probe2" ]

(* The subject: one member stanza which is, or is not, a member of the family named -- by its
   backend marker for `metal-codegen`, by what `probe.ml` names for `lifecycle` -- and, optionally,
   the family alias stanza that aggregates it. Everything else is held fixed across the runs. *)
let family_subject ~shape ~metal ~family ~listing =
  Printf.sprintf "%s\n%s%s"
    (family_gate ~elsewhere:(match shape with Gate_elsewhere -> true | _ -> false))
    (family_member_stanza ~shape ~metal)
    (match (family, family_listed_aliases ~shape ~listing) with
    | None, _ | _, [] -> ""
    | Some family, aliases ->
        Printf.sprintf "\n(alias\n (name %s)\n (deps\n  (alias runtest-gate)\n%s))\n" family
          (String.concat ~sep:"\n" (List.map aliases ~f:(Printf.sprintf "  (alias %s)"))))

(* The lifecycle derivation reads the module's SOURCE, so the two spellings of `probe.ml` are what
   makes the stanza a member or not. Syntactically valid OCaml either way: the checker parses the
   sources it is handed for the variables they read, and a source it cannot read is one it reports
   rather than passes over. *)
type family_probe_source =
  | Reads  (** a real reference to the instrumentation *)
  | Mentions_only  (** the module named in a comment and a string literal, and nowhere else *)
  | Same_named  (** a module of the same LAST name, reached through another qualifier *)
  | Opened  (** `open Ir`, then the module named without its qualifier *)
  | Qualifier_aliased  (** `module I = Ir`, then the module named through the alias *)
  | Scoped_open  (** the qualifier opened inside a nested module, and named OUTSIDE it *)
  | Qualifier_shadowed  (** the qualifier's NAME rebound to another module, then used *)
  | Nested_qualifier  (** somebody else's module of the qualifier's name, opened and aliased *)
  | Nested_path  (** somebody else's module of the qualifier's name, written out in full *)
  | Leaf_shadowed  (** the qualifier opened, and then the LEAF rebound to another module *)
  | Alias_nested  (** the qualifier aliased, and the alias then reached through another module *)
  | Functor_parameter  (** the qualifier's name introduced as a functor's parameter *)
  | Include_wrapper  (** the qualifier re-exported by a structure, then used through it *)
  | Include_with_definition
      (** the same wrapper, with an unrelated definition beside the include *)
  | Open_not_export  (** the qualifier OPENED in a wrapper that includes somebody else *)
  | Module_type_functor  (** the qualifier's name as a MODULE TYPE functor's parameter *)
  | Include_then_override  (** the qualifier included, and then one of its leaves redefined *)
  | Signature_module  (** the qualifier's name declared as a module INSIDE a signature *)
  | Override_then_include  (** a definition, and then an include that supersedes it *)
  | Signature_open  (** the qualifier opened inside a signature, then a leaf named unqualified *)
  | Recursive_module  (** the qualifier's name taken by a recursive module *)
  | Rebound_wrapper  (** an alias of the qualifier, rebound to a wrapper that overrides the leaf *)
  | Recursive_group  (** a recursive group binding an alias BEFORE it takes the qualifier's name *)
  | Include_then_include  (** the qualifier included, and then somebody else included after it *)
  | Signature_alias  (** a manifest module alias inside a signature, used by a later declaration *)
  | Neither

let family_probe = function
  | Reads -> "let () = ignore (Ir.Alloc_census.snapshot ())\n"
  | Same_named ->
      (* Somebody else's `Alloc_census`, and a bare one: real references, to a module whose
         provenance is not the instrumentation's (Codex P2, round 5). *)
      "module Alloc_census = Foo.Alloc_census\nlet () = ignore (Foo.Alloc_census.snapshot ())\n"
  | Opened ->
      (* The same reference with the qualifier opened rather than written: no `Ir.Alloc_census`
         anywhere in the text (Codex P2, round 6). *)
      "open Ir\n\nlet () = ignore (Alloc_census.snapshot ())\n"
  | Qualifier_aliased ->
      (* And with the qualifier itself bound to a name of the source's choosing. *)
      "module I = Ir\n\nlet () = ignore (I.Alloc_census.snapshot ())\n"
  | Qualifier_shadowed ->
      (* The qualifier's name rebound: what `Ir.Alloc_census` names after this is Other's (Codex P2,
         round 8). *)
      "module Ir = Other\n\nlet () = ignore (Ir.Alloc_census.snapshot ())\n"
  | Nested_qualifier ->
      (* `Vendor.Ir` is Vendor's, whatever it is called. Both spellings that would bind it: opened,
         so a bare `Alloc_census` would be in scope, and aliased under the qualifier's own name
         (Codex P2, round 9). *)
      "module Ir = Vendor.Ir\n\n\
       let () = ignore (Ir.Alloc_census.snapshot ())\n\n\
       module Also = struct\n\
      \  open Vendor.Ir\n\n\
      \  let () = ignore (Alloc_census.snapshot ())\n\
       end\n"
  | Nested_path ->
      (* And written out rather than bound: the path names Vendor's module, and our components sit
         inside it (Codex P2, round 10). *)
      "let () = ignore (Vendor.Ir.Alloc_census.snapshot ())\n"
  | Leaf_shadowed ->
      (* The open is ours and the leaf is not: after the rebinding, `Alloc_census` is Foo's (Codex
         P2, round 11). *)
      "open Ir\n\n\
       module Alloc_census = Foo.Alloc_census\n\n\
       let () = ignore (Alloc_census.snapshot ())\n"
  | Alias_nested ->
      (* The alias is ours and the path is not: `Vendor.I.Alloc_census` resolves from Vendor (Codex
         P2, round 12). *)
      "module I = Ir\n\nlet () = ignore (Vendor.I.Alloc_census.snapshot ())\n"
  | Functor_parameter ->
      (* The qualifier's NAME as a functor parameter: inside the body it is the parameter. *)
      "module M (Ir : S) = struct\n  let () = ignore (Ir.Alloc_census.snapshot ())\nend\n"
  | Include_wrapper ->
      (* A structure that only re-exports the qualifier IS the qualifier for this purpose (Codex P2,
         round 13). *)
      "module I = struct\n  include Ir\nend\n\nlet () = ignore (I.Alloc_census.snapshot ())\n"
  | Include_with_definition ->
      (* An include exports its contents whatever sits next to it (Codex P2, round 14). *)
      "module I = struct\n\
      \  include Ir\n\n\
      \  let helper = ()\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n\
       let () = I.helper\n"
  | Open_not_export ->
      (* An `open` changes lookup inside the structure and exports nothing, so what `I` re-exports
         is Vendor's (Codex P2, round 14). *)
      "module I = struct\n\
      \  open Ir\n\n\
      \  include Vendor\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n"
  | Module_type_functor ->
      (* The module-type spelling of a functor: the `Ir` in the result signature is the
         parameter. *)
      "module type F = functor (Ir : S) -> sig\n  val x : Ir.Alloc_census.t\nend\n"
  | Include_then_override ->
      (* The wrapper re-exports the qualifier and then overrides the leaf, so a reference through
         the wrapper to THAT leaf is Vendor's (Codex P2, round 15). *)
      "module I = struct\n\
      \  include Ir\n\n\
      \  module Alloc_census = Vendor.Alloc_census\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n"
  | Signature_module ->
      (* A signature binds module names for the items after it. *)
      "module type S = sig\n  module Ir : X\n\n  val x : Ir.Alloc_census.t\nend\n"
  | Override_then_include ->
      (* The include comes LAST, so what `I.Alloc_census` names is the qualifier's (Codex P2, round
         16). A member. *)
      "module I = struct\n\
      \  module Alloc_census = Vendor.Alloc_census\n\n\
      \  include Ir\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n"
  | Signature_open ->
      (* A signature's open reaches the items after it, so `AC` is `Ir.Alloc_census`. A member. *)
      "module type S = sig\n  open Ir\n\n  module AC = Alloc_census\nend\n"
  | Recursive_module ->
      (* The recursive group's name is in scope throughout it and after it. *)
      "module rec Ir : sig\n\
      \  val x : int\n\
       end = struct\n\
      \  let x = 0\n\
       end\n\n\
       let () = ignore (Ir.Alloc_census.snapshot ())\n"
  | Rebound_wrapper ->
      (* The name first aliases the qualifier and is then rebound to a wrapper that overrides the
         leaf: the second binding is what stands. *)
      "module I = Ir\n\n\
       module I = struct\n\
      \  include Ir\n\n\
      \  module Alloc_census = Vendor.Alloc_census\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n"
  | Recursive_group ->
      (* Every name of the group is in scope in every body, so `I` here is bound to the group's own
         `Ir`, not to the qualifier (Codex P2, round 17). *)
      "module rec I : sig\n\
      \  val x : int\n\
       end = Ir\n\n\
       and Ir : sig\n\
      \  val x : int\n\
       end = struct\n\
      \  let x = 0\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n"
  | Include_then_include ->
      (* The later include decides the leaf, and which leaves it brings cannot be read off the tree
         -- so the wrapper stops naming the qualifier (Codex P2, round 17). *)
      "module I = struct\n\
      \  include Ir\n\n\
      \  include Vendor\n\
       end\n\n\
       let () = ignore (I.Alloc_census.snapshot ())\n"
  | Signature_alias ->
      (* A manifest alias binds, so the later declaration is a real reference. *)
      "module type S = sig\n  module I = Ir\n\n  val x : I.Alloc_census.t\nend\n"
  | Scoped_open ->
      (* The open is real and so is the reference, and they are in different scopes: what
         `Alloc_census` names outside `Elsewhere` is somebody else's module (Codex P2, round 7). *)
      "module Elsewhere = struct\n  open Ir\nend\n\nlet () = ignore (Alloc_census.snapshot ())\n"
  | Mentions_only ->
      (* The module NAMED where naming it reads nothing -- the shape a substring derivation calls a
         probe (Codex P2, round 4). Also as a longer identifier, since that is the third way a text
         scan mistakes a mention for a use. *)
      "(* See Ir.Alloc_census for what this test does NOT do. *)\n\
       let alloc_census_note = \"Ir.Alloc_census\"\n\
       let () = ignore alloc_census_note\n"
  | Neither -> "let () = ()\n"

let family_control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_family" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  write_file (Stdlib.Filename.concat root "t/gate.ml") "let () = ()\n";
  (* The plural shape's second module. Written once and always present: which stanza CLAIMS it is
     what differs between the shapes, and an unclaimed source is not itself a finding here. *)
  write_file (Stdlib.Filename.concat root "t/probe2.ml") "let () = ()\n";
  (* The remaining shapes' own modules. A stanza names its modules, and a module this scan is handed
     no source for is a finding of its own (gh-ocannl-749) -- so a fixture tree that declares one
     without providing it fails for a reason that has nothing to do with the family relationship.
     Inert, present always, and claimed only by the shape that names them. *)
  write_file (Stdlib.Filename.concat root "t/harness.ml") "let () = ()\n";
  write_file (Stdlib.Filename.concat root "t/child/childgate.ml") "let () = ()\n";
  let paths =
    "t/dune" :: "t/gate.ml" :: "t/probe.ml" :: "t/probe2.ml" :: "t/child/probe.ml" :: "t/harness.ml"
    :: "t/child/childgate.ml" :: List.map context ~f:fst
  in
  let run ?(shape = Single_test) ?(listing = Every) ?probe ~metal ~lifecycle ~family () =
    let probe =
      match probe with Some probe -> probe | None -> if lifecycle then Reads else Neither
    in
    write_file
      (Stdlib.Filename.concat root "t/dune")
      (family_subject ~shape ~metal ~family ~listing);
    (* The same source in both directories, so that a shape putting the member in `(subdir child …)`
       is put the same question as one at the top level. *)
    List.iter [ "t/probe.ml"; "t/child/probe.ml" ] ~f:(fun path ->
        write_file (Stdlib.Filename.concat root path) (family_probe probe));
    run_checker ~root ~exe ("." :: paths)
  in
  let report label (status, text) =
    eprintf "the family control's %s run %s. Its captured output:\n%s\n" label
      (describe_status status) text
  in
  let exited n (status, _) = match status with Unix.WEXITED m -> m = n | _ -> false in
  let unreached = "alias does not reach" in
  let metal = Some metal_family.family_alias and lifecycle = Some lifecycle_family.family_alias in
  let metal_omitted = run ~metal:true ~lifecycle:false ~family:None () in
  let metal_listed = run ~metal:true ~lifecycle:false ~family:metal () in
  let lifecycle_omitted = run ~metal:false ~lifecycle:true ~family:None () in
  let lifecycle_listed = run ~metal:false ~lifecycle:true ~family:lifecycle () in
  (* The marker's placement on an `(executable)` is the RULE that runs it, so the derivation has to
     read it there and the alias to list is the rule's. *)
  let runner_omitted = run ~shape:Exe_with_runner ~metal:true ~lifecycle:false ~family:None () in
  let runner_listed = run ~shape:Exe_with_runner ~metal:true ~lifecycle:false ~family:metal () in
  (* A plural stanza is several units: a marker covers both of them, so listing one alias leaves the
     other out of the family -- while the lifecycle derivation belongs to whichever main actually
     names the instrumentation, so listing that one alias is COMPLETE. Same stanza, same listing,
     opposite verdicts: what differs is which derivation put it in the family. *)
  let plural_half =
    run ~shape:Plural_tests ~listing:First_only ~metal:true ~lifecycle:false ~family:metal ()
  in
  let plural_whole = run ~shape:Plural_tests ~metal:true ~lifecycle:false ~family:metal () in
  let plural_one_main =
    run ~shape:Plural_tests ~listing:First_only ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* An inline-test library's alias is generated too, so a Metal-marked one reaches its family
     through `runtest-<library-name>`. *)
  let library_omitted = run ~shape:Inline_library ~metal:true ~lifecycle:false ~family:None () in
  let library_listed = run ~shape:Inline_library ~metal:true ~lifecycle:false ~family:metal () in
  (* A `(subdir …)` group carrying its own family alias is correctly wired, and the recursive build
     from the root reaches it. *)
  let subdir_listed = run ~shape:Subdir_member ~metal:true ~lifecycle:false ~family:None () in
  (* And the runner that runs SOMEONE ELSE's executable: listing its alias does not put this
     directory's lifecycle probe in the family, however alike the two basenames are. *)
  let runner_elsewhere =
    run ~shape:Runner_elsewhere ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  let runner_here = run ~shape:Exe_with_runner ~metal:false ~lifecycle:true ~family:lifecycle () in
  (* An installed executable run by its public name, and one declared in a group whose runner sits
     at the top level: both are correct wirings whose runner link the earlier readings dropped. *)
  let public_runner =
    run ~shape:Public_name_runner ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  let cross_group_runner =
    run ~shape:Subdir_exe_top_runner ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* And the alias a family must never be given: the directory-wide suite. *)
  let runtest_only = run ~shape:Runtest_only_rule ~metal:true ~lifecycle:false ~family:metal () in
  (* A public name belongs to ITS executable: a rule running the second one's public name runs the
     second one, whatever the first is called. *)
  let crossed_public =
    run ~shape:Public_names_crossed ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* Naming the instrumentation is not reading it: a doc comment, a string literal and a longer
     identifier put the name in the source and put nothing in the family. *)
  let mention_only = run ~probe:Mentions_only ~metal:false ~lifecycle:true ~family:None () in
  (* The group's own ambient gate: an alias defined inside `(subdir child …)` is `child`'s, and a
     gate at the top level does not reach it. *)
  let subdir_ungated = run ~shape:Subdir_ungated ~metal:true ~lifecycle:false ~family:None () in
  (* Nor does a top-level reading see a group's training lock: `is_gate` is false of the `(subdir
     …)` form, so the group's unlocked gate passed unseen. *)
  let subdir_unlocked =
    run ~shape:Subdir_unlocked_gate ~metal:true ~lifecycle:false ~family:None ()
  in
  (* And a module of the same last name reached through another qualifier: a real reference, to
     something that is not the instrumentation. *)
  let same_named = run ~probe:Same_named ~metal:false ~lifecycle:true ~family:None () in
  (* The two spellings that reach the instrumentation without writing its qualifier. Both ARE
     members, so the tree without a family stanza is the reported one -- the discriminating
     direction, since a derivation that missed them would pass this tree silently. *)
  let opened_probe = run ~probe:Opened ~metal:false ~lifecycle:true ~family:None () in
  let aliased_probe = run ~probe:Qualifier_aliased ~metal:false ~lifecycle:true ~family:None () in
  (* And the same open, one scope away from the reference. *)
  let scoped_open = run ~probe:Scoped_open ~metal:false ~lifecycle:true ~family:None () in
  (* And the qualifier's own name rebound to another module. *)
  let shadowed_qualifier =
    run ~probe:Qualifier_shadowed ~metal:false ~lifecycle:true ~family:None ()
  in
  (* Somebody else's module of the qualifier's name, opened and aliased. *)
  let nested_qualifier = run ~probe:Nested_qualifier ~metal:false ~lifecycle:true ~family:None () in
  let nested_path = run ~probe:Nested_path ~metal:false ~lifecycle:true ~family:None () in
  (* The qualifier opened and the LEAF then rebound: what follows is Foo's. *)
  let leaf_shadowed = run ~probe:Leaf_shadowed ~metal:false ~lifecycle:true ~family:None () in
  (* The alias reached through another module, and the qualifier's name as a functor parameter. *)
  let alias_nested = run ~probe:Alias_nested ~metal:false ~lifecycle:true ~family:None () in
  let functor_parameter =
    run ~probe:Functor_parameter ~metal:false ~lifecycle:true ~family:None ()
  in
  (* A structure that only re-exports the qualifier IS one, so its user is a member and the tree
     without a family stanza is the reported one. *)
  let include_wrapper = run ~probe:Include_wrapper ~metal:false ~lifecycle:true ~family:None () in
  (* The same wrapper with a definition beside the include: still a re-export, still a member. *)
  let include_with_definition =
    run ~probe:Include_with_definition ~metal:false ~lifecycle:true ~family:None ()
  in
  (* An `open` in a wrapper that includes somebody else re-exports the somebody else. *)
  let open_not_export = run ~probe:Open_not_export ~metal:false ~lifecycle:true ~family:None () in
  (* And the module-type spelling of a functor parameter. *)
  let module_type_functor =
    run ~probe:Module_type_functor ~metal:false ~lifecycle:true ~family:None ()
  in
  (* A wrapper that re-exports the qualifier and then overrides the leaf, and a signature that binds
     the qualifier's name: neither reference is ours. *)
  let include_then_override =
    run ~probe:Include_then_override ~metal:false ~lifecycle:true ~family:None ()
  in
  let signature_module = run ~probe:Signature_module ~metal:false ~lifecycle:true ~family:None () in
  (* Two spellings that ARE references and were missed, and two that are not and were accepted. *)
  let override_then_include =
    run ~probe:Override_then_include ~metal:false ~lifecycle:true ~family:None ()
  in
  let signature_open = run ~probe:Signature_open ~metal:false ~lifecycle:true ~family:None () in
  let recursive_module = run ~probe:Recursive_module ~metal:false ~lifecycle:true ~family:None () in
  let rebound_wrapper = run ~probe:Rebound_wrapper ~metal:false ~lifecycle:true ~family:None () in
  let recursive_group = run ~probe:Recursive_group ~metal:false ~lifecycle:true ~family:None () in
  let include_then_include =
    run ~probe:Include_then_include ~metal:false ~lifecycle:true ~family:None ()
  in
  let signature_alias = run ~probe:Signature_alias ~metal:false ~lifecycle:true ~family:None () in
  (* And a `(test)` stanza as the runner of a lifecycle executable. *)
  let test_stanza_runner =
    run ~shape:Test_stanza_runner ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* Two default-module tests, only one of whose mains reads the instrumentation: listing that one
     alias is complete. *)
  let sibling_defaults =
    run ~shape:Sibling_defaults ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* And an absolute path that ends in the local executable's name. *)
  let runner_absolute =
    run ~shape:Runner_absolute ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* And a rule on the gate's generated alias that runs another directory's binary: not the gate, so
     the collision is not the deliberate one. *)
  let gate_elsewhere = run ~shape:Gate_elsewhere ~metal:true ~lifecycle:false ~family:metal () in
  (* And the alias collision one directory down: dune generates a `(test)` stanza's alias in the
     directory it applies the stanza to, so a rule in the same group can merge with it. *)
  let subdir_collision =
    run ~shape:Subdir_alias_collision ~metal:true ~lifecycle:false ~family:None ()
  in
  (* A rule running a FILE that shares the executable's public name is not its runner. *)
  let public_by_path =
    run ~shape:Public_name_by_path ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* The negative control: neither derivation calls this stanza a member, so no family alias is
     asked for. A derivation that over-claimed would fail this correct tree. *)
  let no_member = run ~metal:false ~lifecycle:false ~family:None () in
  let omitted_ok family (result : Unix.process_status * string) =
    exited 1 result
    && String.is_substring (snd result) ~substring:unreached
    && String.is_substring (snd result) ~substring:family
    && String.is_substring (snd result) ~substring:"t/dune"
  in
  let listed_ok result =
    exited 0 result && not (String.is_substring (snd result) ~substring:unreached)
  in
  let metal_omitted_ok = omitted_ok metal_family.family_alias metal_omitted in
  let lifecycle_omitted_ok = omitted_ok lifecycle_family.family_alias lifecycle_omitted in
  let runner_omitted_ok = omitted_ok metal_family.family_alias runner_omitted in
  let plural_half_ok =
    omitted_ok metal_family.family_alias plural_half
    && String.is_substring (snd plural_half) ~substring:"runtest-probe2"
  in
  let metal_listed_ok = listed_ok metal_listed in
  let lifecycle_listed_ok = listed_ok lifecycle_listed in
  let runner_listed_ok = listed_ok runner_listed in
  let plural_whole_ok = listed_ok plural_whole in
  let plural_one_main_ok = listed_ok plural_one_main in
  let library_omitted_ok = omitted_ok metal_family.family_alias library_omitted in
  let library_listed_ok = listed_ok library_listed in
  let subdir_listed_ok = listed_ok subdir_listed in
  let runner_elsewhere_ok = omitted_ok lifecycle_family.family_alias runner_elsewhere in
  let runner_here_ok = listed_ok runner_here in
  let public_runner_ok = listed_ok public_runner in
  let cross_group_runner_ok = listed_ok cross_group_runner in
  let runtest_only_ok = omitted_ok metal_family.family_alias runtest_only in
  let crossed_public_ok = omitted_ok lifecycle_family.family_alias crossed_public in
  let mention_only_ok = listed_ok mention_only in
  let subdir_ungated_ok =
    exited 1 subdir_ungated
    && String.is_substring (snd subdir_ungated) ~substring:"no ambient gate reaches it"
    && String.is_substring (snd subdir_ungated) ~substring:"(subdir child"
  in
  let subdir_unlocked_ok =
    exited 1 subdir_unlocked
    && String.is_substring (snd subdir_unlocked) ~substring:"does not take the lock"
    && String.is_substring (snd subdir_unlocked) ~substring:"(subdir child"
  in
  let same_named_ok = listed_ok same_named in
  let opened_probe_ok = omitted_ok lifecycle_family.family_alias opened_probe in
  let aliased_probe_ok = omitted_ok lifecycle_family.family_alias aliased_probe in
  let scoped_open_ok = listed_ok scoped_open in
  let shadowed_qualifier_ok = listed_ok shadowed_qualifier in
  let public_by_path_ok = omitted_ok lifecycle_family.family_alias public_by_path in
  let nested_qualifier_ok = listed_ok nested_qualifier in
  let nested_path_ok = listed_ok nested_path in
  let leaf_shadowed_ok = listed_ok leaf_shadowed in
  let alias_nested_ok = listed_ok alias_nested in
  let functor_parameter_ok = listed_ok functor_parameter in
  let include_wrapper_ok = omitted_ok lifecycle_family.family_alias include_wrapper in
  let include_with_definition_ok =
    omitted_ok lifecycle_family.family_alias include_with_definition
  in
  let open_not_export_ok = listed_ok open_not_export in
  let module_type_functor_ok = listed_ok module_type_functor in
  let include_then_override_ok = listed_ok include_then_override in
  let signature_module_ok = listed_ok signature_module in
  let override_then_include_ok = omitted_ok lifecycle_family.family_alias override_then_include in
  let signature_open_ok = omitted_ok lifecycle_family.family_alias signature_open in
  let recursive_module_ok = listed_ok recursive_module in
  let rebound_wrapper_ok = listed_ok rebound_wrapper in
  let recursive_group_ok = listed_ok recursive_group in
  let include_then_include_ok = listed_ok include_then_include in
  let signature_alias_ok = omitted_ok lifecycle_family.family_alias signature_alias in
  let test_stanza_runner_ok = listed_ok test_stanza_runner in
  let sibling_defaults_ok = listed_ok sibling_defaults in
  let runner_absolute_ok = omitted_ok lifecycle_family.family_alias runner_absolute in
  let gate_elsewhere_ok =
    exited 1 gate_elsewhere
    && String.is_substring (snd gate_elsewhere) ~substring:"the alias dune generates"
  in
  let subdir_collision_ok =
    exited 1 subdir_collision
    && String.is_substring (snd subdir_collision) ~substring:"the alias dune generates"
    && String.is_substring (snd subdir_collision) ~substring:"(subdir child"
  in
  let no_member_ok = listed_ok no_member in
  if not metal_omitted_ok then report "metal, family stanza omitted" metal_omitted;
  if not metal_listed_ok then report "metal, family stanza listing it" metal_listed;
  if not lifecycle_omitted_ok then report "lifecycle, family stanza omitted" lifecycle_omitted;
  if not lifecycle_listed_ok then report "lifecycle, family stanza listing it" lifecycle_listed;
  if not runner_omitted_ok then report "executable + runner, family stanza omitted" runner_omitted;
  if not runner_listed_ok then report "executable + runner, runner listed" runner_listed;
  if not plural_half_ok then report "plural stanza, one alias listed" plural_half;
  if not plural_whole_ok then report "plural stanza, both aliases listed" plural_whole;
  if not plural_one_main_ok then report "plural stanza, one main is the probe" plural_one_main;
  if not library_omitted_ok then report "inline-test library, family stanza omitted" library_omitted;
  if not library_listed_ok then report "inline-test library, generated alias listed" library_listed;
  if not subdir_listed_ok then report "member and family alias inside a subdir group" subdir_listed;
  if not runner_elsewhere_ok then report "runner running another directory's exe" runner_elsewhere;
  if not runner_here_ok then report "runner running this directory's exe" runner_here;
  if not public_runner_ok then report "runner naming the public name" public_runner;
  if not cross_group_runner_ok then report "subdir executable, top-level runner" cross_group_runner;
  if not runtest_only_ok then report "marked rule whose only alias is runtest" runtest_only;
  if not crossed_public_ok then report "runner naming the other unit's public name" crossed_public;
  if not mention_only_ok then report "the instrumentation named but not read" mention_only;
  if not subdir_ungated_ok then report "family alias in a group with no gate" subdir_ungated;
  if not subdir_unlocked_ok then report "locked group whose gate takes no lock" subdir_unlocked;
  if not same_named_ok then report "a same-named module of another provenance" same_named;
  if not opened_probe_ok then report "the qualifier opened rather than written" opened_probe;
  if not aliased_probe_ok then report "the qualifier bound to another name" aliased_probe;
  if not scoped_open_ok then report "the qualifier opened in another scope" scoped_open;
  if not shadowed_qualifier_ok then report "the qualifier's name rebound" shadowed_qualifier;
  if not public_by_path_ok then report "a file sharing the public name" public_by_path;
  if not nested_qualifier_ok then report "another module named like the qualifier" nested_qualifier;
  if not nested_path_ok then report "that module's path written out" nested_path;
  if not leaf_shadowed_ok then report "the opened module's leaf rebound" leaf_shadowed;
  if not alias_nested_ok then report "the alias reached through another module" alias_nested;
  if not functor_parameter_ok then
    report "the qualifier's name as a functor parameter" functor_parameter;
  if not include_wrapper_ok then report "the qualifier re-exported by a structure" include_wrapper;
  if not include_with_definition_ok then
    report "a re-export beside a definition" include_with_definition;
  if not open_not_export_ok then report "an open mistaken for a re-export" open_not_export;
  if not module_type_functor_ok then report "a module-type functor parameter" module_type_functor;
  if not include_then_override_ok then report "an overridden leaf" include_then_override;
  if not signature_module_ok then report "a module bound inside a signature" signature_module;
  if not override_then_include_ok then report "an include after a definition" override_then_include;
  if not signature_open_ok then report "an open inside a signature" signature_open;
  if not recursive_module_ok then
    report "a recursive module of the qualifier's name" recursive_module;
  if not rebound_wrapper_ok then report "an alias rebound to a wrapper" rebound_wrapper;
  if not recursive_group_ok then
    report "a recursive group taking the qualifier's name" recursive_group;
  if not include_then_include_ok then report "an include after the qualifier's" include_then_include;
  if not signature_alias_ok then report "a manifest alias in a signature" signature_alias;
  if not test_stanza_runner_ok then report "a test stanza as the runner" test_stanza_runner;
  if not sibling_defaults_ok then report "two default-module stanzas" sibling_defaults;
  if not runner_absolute_ok then report "an absolute runner path" runner_absolute;
  if not gate_elsewhere_ok then report "a gate alias running another binary" gate_elsewhere;
  if not subdir_collision_ok then report "an alias collision inside a group" subdir_collision;
  if not no_member_ok then report "neither derivation's member" no_member;
  printf
    "\n\
     The focused-aggregate rule is put to a tree of one member stanza and a family alias that does\n\
     or does not list it. Nothing else differs between the runs of a pair; the member takes each of\n\
     the shapes dune builds a member in -- a test, an executable with its runner, a plural stanza,\n\
     an inline-test library, a `(subdir …)` group -- and the last run is a stanza neither\n\
     derivation calls a member.\n\n";
  Verdict.p
    "a stanza whose backend marker names metal is reported, naming its family, when the \
     metal-codegen alias does not reach it"
    metal_omitted_ok;
  Verdict.p "the same tree with the member listed in the family stanza passes" metal_listed_ok;
  Verdict.p
    "a stanza whose modules read the resource-lifecycle instrumentation is reported the same way \
     when the lifecycle alias does not reach it"
    lifecycle_omitted_ok;
  Verdict.p "the same tree with that member listed passes too" lifecycle_listed_ok;
  Verdict.p
    "an executable whose RUNNER carries the metal marker is a member too, reported when the family \
     does not reach that rule"
    runner_omitted_ok;
  Verdict.p "the same tree with the runner's own alias listed passes" runner_listed_ok;
  Verdict.p
    "a plural stanza with only one of its two generated aliases listed is reported, naming the one \
     still missing"
    plural_half_ok;
  Verdict.p "the same plural stanza with both listed passes" plural_whole_ok;
  Verdict.p
    "the same one-alias listing is COMPLETE when only that main reads the instrumentation, so the \
     lifecycle family is not asked for its neighbour"
    plural_one_main_ok;
  Verdict.p
    "an inline-test library carrying the metal marker is a member, reported when the family does \
     not reach the alias dune generates for it"
    library_omitted_ok;
  Verdict.p "the same library listed under its generated alias passes" library_listed_ok;
  Verdict.p
    "a member inside a `(subdir …)` group is aggregated by a family alias in that same group, \
     which the recursive build from the root reaches"
    subdir_listed_ok;
  Verdict.p
    "a runner that runs another directory's executable of the same name does not aggregate this \
     directory's member"
    runner_elsewhere_ok;
  Verdict.p "the same runner running this directory's executable does" runner_here_ok;
  Verdict.p
    "a runner naming the executable's `(public_name …)` aggregates it as much as one naming its \
     .exe"
    public_runner_ok;
  Verdict.p
    "an executable declared in a `(subdir …)` group is aggregated by the top-level family stanza \
     when the rule that runs it sits at the top level"
    cross_group_runner_ok;
  Verdict.p
    "a marked rule whose only alias is the directory-wide `runtest` has no focused alias to offer, \
     and a family listing `runtest` does not aggregate it"
    runtest_only_ok;
  Verdict.p
    "a public name belongs to its own executable, so a rule running the second unit's public name \
     does not aggregate the first"
    crossed_public_ok;
  Verdict.p
    "a source that names the instrumentation in a comment, a string and a longer identifier reads \
     none of it, and is no member"
    mention_only_ok;
  Verdict.p
    "a family alias defined inside a `(subdir …)` group needs that group's own ambient gate, and \
     is reported without one"
    subdir_ungated_ok;
  Verdict.p
    "a `(subdir …)` group whose actions take the training lock and whose gate does not is reported \
     there too"
    subdir_unlocked_ok;
  Verdict.p
    "a module of the same last name reached through another qualifier is not the instrumentation, \
     and its user is no member"
    same_named_ok;
  Verdict.p
    "`open Ir` and then a bare `Alloc_census.snapshot` is a reference to the instrumentation, and \
     its test is a member"
    opened_probe_ok;
  Verdict.p "so is `module I = Ir` and then `I.Alloc_census.snapshot`" aliased_probe_ok;
  Verdict.p
    "an `open Ir` inside a nested module does not make a bare `Alloc_census` outside it a \
     reference to the instrumentation"
    scoped_open_ok;
  Verdict.p
    "`module Ir = Other` rebinds the qualifier, so a later `Ir.Alloc_census` is not the \
     instrumentation"
    shadowed_qualifier_ok;
  Verdict.p
    "a rule running a file that shares the executable's public name is not its runner, however \
     alike the two strings are"
    public_by_path_ok;
  Verdict.p
    "`open Vendor.Ir` and `module Ir = Vendor.Ir` bind Vendor's module, not the qualifier, so \
     neither makes their file a member"
    nested_qualifier_ok;
  Verdict.p
    "nor does writing `Vendor.Ir.Alloc_census` out in full: a path names the module it starts at"
    nested_path_ok;
  Verdict.p
    "`open Ir` and then `module Alloc_census = Foo.Alloc_census` rebinds the leaf, so what follows \
     is not the instrumentation"
    leaf_shadowed_ok;
  Verdict.p
    "`module I = Ir` does not make `Vendor.I.Alloc_census` ours: an alias-qualified path resolves \
     from where it starts"
    alias_nested_ok;
  Verdict.p "a functor parameter named `Ir` shadows the qualifier inside the functor's body"
    functor_parameter_ok;
  Verdict.p
    "`module I = struct include Ir end` re-exports the qualifier, so `I.Alloc_census` is a \
     reference to the instrumentation"
    include_wrapper_ok;
  Verdict.p "and goes on doing so when the structure defines something beside the include"
    include_with_definition_ok;
  Verdict.p
    "`open Ir` inside a wrapper that includes somebody else re-exports the somebody else, so its \
     user is no member"
    open_not_export_ok;
  Verdict.p
    "a MODULE TYPE functor's parameter named `Ir` shadows the qualifier in its result signature"
    module_type_functor_ok;
  Verdict.p
    "a wrapper that includes the qualifier and then redefines a leaf does not lend us that leaf"
    include_then_override_ok;
  Verdict.p
    "`module Ir : X` inside a signature binds the name for the items after it, so a later \
     `Ir.Alloc_census.t` is the signature's"
    signature_module_ok;
  Verdict.p
    "an include AFTER a definition supersedes it, so the wrapper's leaf is the qualifier's again"
    override_then_include_ok;
  Verdict.p
    "an `open` inside a signature reaches the items after it, so `module AC = Alloc_census` there \
     is a reference"
    signature_open_ok;
  Verdict.p
    "a recursive module taking the qualifier's name shadows it, in its own group and after it"
    recursive_module_ok;
  Verdict.p
    "rebinding an alias replaces what it meant, so the earlier binding does not keep the later \
     reference alive"
    rebound_wrapper_ok;
  Verdict.p
    "every name of a recursive group is in scope in every body, so an alias declared before the \
     group takes `Ir` is bound to the group's"
    recursive_group_ok;
  Verdict.p
    "a wrapper that includes the qualifier and then includes somebody else stops naming it, since \
     which leaves the second include brings cannot be read off the tree"
    include_then_include_ok;
  Verdict.p "`module I = Ir` inside a signature binds, so a later `I.Alloc_census.t` is a reference"
    signature_alias_ok;
  Verdict.p
    "an executable run by a `(test)` stanza's custom action is aggregated through the \
     `runtest-<name>` dune generates for that test"
    test_stanza_runner_ok;
  Verdict.p
    "two stanzas that omit `(modules …)` get a main each, so only the one whose main reads the \
     instrumentation is a member"
    sibling_defaults_ok;
  Verdict.p "an absolute path ending in the local executable's name is not the local executable"
    runner_absolute_ok;
  Verdict.p
    "a rule on a `(test)`'s generated alias that runs ANOTHER directory's binary is not the \
     deliberate gate collision, and is reported"
    gate_elsewhere_ok;
  Verdict.p
    "a rule reusing the alias dune generates for a `(test)` in the same `(subdir …)` group is \
     reported there too"
    subdir_collision_ok;
  Verdict.p "a stanza neither derivation calls a member is asked for no family alias" no_member_ok;
  try remove_tree root with Unix.Unix_error _ -> ()

(* gh-ocannl-749's control, and why it is a third tree.

   The rule is that a key a module reads straight from the environment is one the stanza running it
   declares. Every guard in this repository now declares its keys -- which is what this change did
   -- so a control drawn from the corpus would pass whether the rule has teeth or has stopped
   deciding. Put to a tree of its own it is the rule that is claimed about: the same source, once
   under a rule that declares the key, once under one that does not, and once under one that pins
   the variable with `setenv` instead.

   The source is a GUARD, not a literal read: the key reaches the reader through a list, which is
   the shape the hand-written lists took and the shape the earlier check could not see. One key is a
   configuration key and one is not, so the tree also says that the candidate set is intersected
   with the registry rather than taken whole -- a scan demanding `OCANNL_NOT_A_CONFIG_KEY` would
   fail the legitimate run, and the declaration it asked for would then be reported by the sibling
   check as naming no key OCANNL reads. *)

let guard_key = "virtualize_max_visits"
let guard_non_key = "not_a_config_key"

(* The same guard with its key SHOUTED. `read_env_var` uppercases to build the variable, so this
   reads the very same `OCANNL_<KEY>`; a case-sensitive look-up against the registry dropped it as
   an unknown key and asked for nothing (Codex P2, round 5). *)
let shouting_probe =
  Printf.sprintf
    "let guarded = [ %S ]\n\
     let () =\n\
    \  List.iter\n\
    \    (fun arg_name ->\n\
    \      match Utils.read_env_var arg_name with Some _ -> exit 1 | None -> ())\n\
    \    guarded\n"
    (String.uppercase guard_key)

let guard_of key =
  Printf.sprintf
    "let guarded = [ %S ]\n\
     let () =\n\
    \  List.iter\n\
    \    (fun arg_name ->\n\
    \      match Utils.read_env_var arg_name with Some _ -> exit 1 | None -> ())\n\
    \    guarded\n"
    key

let guard_probe = guard_of guard_key

(* A guard on a key the registry does not know. The reader builds and consults `OCANNL_<KEY>`
   whatever the registry says, so the run depends on it -- and it cannot be DECLARED, the sibling
   check refusing a declaration naming no key OCANNL reads, so a pin is the only way to answer for
   it (Codex P2, round 11 of PR #484). *)
let unknown_key_probe = guard_of guard_non_key

(* A dynamic reach whose key list is in ANOTHER compilation unit -- the reviewer's own example. The
   source names no configuration key at all, so the candidate fallback resolves to nothing and the
   check would have run its loop over an empty list, passing while checking nothing. *)
let opaque_probe = "let () = List.iter (fun k -> ignore (Utils.read_env_var k)) Shared.guarded\n"

(* One rule running the guard, answering for the variable in one of the three ways a rule can. The
   declared spelling is built from the key rather than written out, so the control cannot drift from
   `Utils.env_var_name`. *)
let guard_rule ~target ~declares ~pins =
  Printf.sprintf
    {dune|(rule
 ; ocannl-backend: none -- a synthetic control fixture, which runs on no device at all.
 (target %s)
 (deps
  ocannl_config
%s  %%{dep:guard.exe})
 (action
  (with-stdout-to
   %%{target}
%s)))
|dune}
    target
    (if declares then Printf.sprintf "  (env_var %s)\n" (Utils.env_var_name guard_key) else "")
    (match pins with
    | `No -> "  (run ./guard.exe)"
    | `Around_the_run ->
        Printf.sprintf "  (setenv %s 1\n   (run ./guard.exe))" (Utils.env_var_name guard_key)
    | `Around_the_run_unknown ->
        Printf.sprintf "  (setenv %s 1\n   (run ./guard.exe))" (Utils.env_var_name guard_non_key)
    (* The pin on a SIBLING branch of the same action: `setenv` scopes over what it wraps, so this
       rule's run of the guard is as exposed to the ambient variable as an unpinned one. A check
       reading the rule's setenvs as a flat set would credit it (Codex P1, round 1). *)
    | `Elsewhere_in_the_action ->
        Printf.sprintf "  (progn\n   (setenv %s 1\n    (run ./other.exe))\n   (run ./guard.exe))"
          (Utils.env_var_name guard_key)
    (* The inverse: the guard IS pinned, and an unpinned run of a helper stands beside it. The pin
       belongs to the guard's run, so an intersection taken over every command in the action --
       which is what this check did first -- demanded a declaration the run cannot need (Codex P2,
       round 4). *)
    | `Around_the_run_beside_a_helper ->
        Printf.sprintf "  (progn\n   (setenv %s 1\n    (run ./guard.exe))\n   (run ./other.exe))"
          (Utils.env_var_name guard_key))

(* The arms. Everything but the one thing each is about is held fixed, so a difference in verdict is
   that thing's and nothing else's. *)
let guard_subject ~arm =
  let executable modules =
    Printf.sprintf "(executable\n (name guard)\n%s (libraries arrayjit.utils))\n\n" modules
  in
  let named = " (modules guard)\n" in
  let one ~declares ~pins = guard_rule ~target:"guard.actual" ~declares ~pins in
  match arm with
  | `Declares | `Unresolvable -> executable named ^ one ~declares:true ~pins:`No
  | `Pins -> executable named ^ one ~declares:false ~pins:`Around_the_run
  | `Pins_beside_a_helper ->
      executable named ^ one ~declares:false ~pins:`Around_the_run_beside_a_helper
  | `Pins_elsewhere -> executable named ^ one ~declares:false ~pins:`Elsewhere_in_the_action
  | `Neither -> executable named ^ one ~declares:false ~pins:`No
  (* An `(executable)` reaching for dune's DEFAULT module set: the same tree with the `(modules …)`
     field left off, which is the common shape and the one the check skipped entirely while it was a
     clause of a walk guarded on that field being written down (Codex P2, round 1). *)
  | `Implicit_modules -> executable "" ^ one ~declares:false ~pins:`No
  (* A module the stanza NAMES and the scan was handed no source for. Reading it as a module with no
     reads is the silent direction (Codex P2, round 8). *)
  | `Module_without_source ->
      Printf.sprintf
        "(executable\n (name guard)\n (modules guard helper)\n (libraries arrayjit.utils))\n\n%s"
        (guard_rule ~target:"guard.actual" ~declares:true ~pins:`No)
  (* A test directory's own `utils.ml`: the exemption belongs to the module that DEFINES the reader,
     at its repository path, and a basename match would extend it to this one (Codex P2, round
     8). *)
  | `Utils_lookalike ->
      Printf.sprintf
        "(executable\n (name guard)\n (modules guard utils)\n (libraries arrayjit.utils))\n\n%s"
        (guard_rule ~target:"guard.actual" ~declares:false ~pins:`No)
  (* The same program one directory down. A `(subdir …)` applies its stanzas to another directory,
     so the modules live there and the runner may sit at either level -- and a walk over the
     top-level forms alone read the wrapper as a stanza with no modules (Codex P2, round 2). *)
  | `In_a_subdir ->
      Printf.sprintf "(subdir gen\n %s)\n\n%s"
        (String.strip (executable " (modules guard)\n"))
        (String.substr_replace_all
           (guard_rule ~target:"guard.actual" ~declares:false ~pins:`No)
           ~pattern:"guard.exe" ~with_:"gen/guard.exe")
  (* A `(library)` with inline tests is RUN by dune, under `(inline_tests (deps …))`, so a module of
     it reading the environment goes stale the same way an executable's does. A plain `(library)` is
     out of scope, `Utils` being where the reader lives. *)
  (* And a PLAIN library, run by nothing of its own: the requirement would fall on every stanza that
     links it, so the read is reported rather than attributed (Codex P2, round 6). *)
  (* The mode this scan refuses to model. It shipped dead once -- matched as a FIELD of a stanza
     rather than as a stanza -- so it has an arm of its own now: a refusal with no control is a
     refusal nobody has seen happen (Codex P2, round 11). *)
  (* The unknown-key guard, once answered for and once not. *)
  | `Unknown_key -> executable named ^ one ~declares:false ~pins:`No
  | `Unknown_key_pinned ->
      executable named
      ^ guard_rule ~target:"guard.actual" ~declares:false ~pins:`Around_the_run_unknown
  (* A `(test)` dune runs itself, with an action that PINS the variable: the run cannot observe the
     ambient one, so no declaration answers for it (Codex P2, round 12). *)
  | `Self_running_pin ->
      Printf.sprintf
        "(test\n\
        \ ; ocannl-backend: none -- a synthetic control fixture, which runs on no device at all.\n\
        \ (name guard)\n\
        \ (modules guard)\n\
        \ (deps ocannl_config)\n\
        \ (libraries arrayjit.utils)\n\
        \ (action\n\
        \  (setenv %s 1\n\
        \   (run %%{test}))))\n\n\
         (rule\n\
        \ (alias runtest)\n\
        \ (deps ocannl_config (universe))\n\
        \ (action\n\
        \  (progn)))\n"
        (Utils.env_var_name guard_key)
  (* The same directive one level down, which the top-level loop did not reach. *)
  | `Include_subdirs_nested ->
      "(subdir gen\n (include_subdirs unqualified))\n\n"
      ^ Printf.sprintf
          "(executable\n (name guard)\n (modules guard)\n (libraries arrayjit.utils))\n\n%s"
          (guard_rule ~target:"guard.actual" ~declares:true ~pins:`No)
  | `Include_subdirs ->
      "(include_subdirs unqualified)\n\n"
      ^ Printf.sprintf
          "(executable\n (name guard)\n (modules guard)\n (libraries arrayjit.utils))\n\n%s"
          (guard_rule ~target:"guard.actual" ~declares:true ~pins:`No)
  | `Plain_library -> "(library\n (name guard)\n (modules guard)\n (libraries arrayjit.utils))\n"
  | `Inline_tests_library ->
      "(library\n\
       ; ocannl-backend: none -- a synthetic control fixture, which runs on no device at all.\n\
      \ (name guard)\n\
      \ (modules guard)\n\
      \ (libraries arrayjit.utils)\n\
      \ (inline_tests))\n"
  (* An `(inline_tests (deps …))` declaration is not a licence: it invalidates the inline runner
     alone, while the library stays linkable by executables that declare nothing (Codex P2, round
     7). *)
  | `Inline_tests_library_declares ->
      Printf.sprintf
        "(library\n\
         ; ocannl-backend: none -- a synthetic control fixture, which runs on no device at all.\n\
        \ (name guard)\n\
        \ (modules guard)\n\
        \ (libraries arrayjit.utils)\n\
        \ (inline_tests\n\
        \  (deps\n\
        \   (env_var %s))))\n"
        (Utils.env_var_name guard_key)
  (* TWO rules running the same executable. Dune invalidates each on its own deps, so one of them
     declaring says nothing about the other's run -- the file-wide latitude this check started with
     accepted exactly this tree (Codex P1, round 1). *)
  | `Second_runner_bare ->
      executable named ^ one ~declares:true ~pins:`No ^ "\n"
      ^ guard_rule ~target:"guard2.actual" ~declares:false ~pins:`No
  | `Second_runner_declares ->
      executable named ^ one ~declares:true ~pins:`No ^ "\n"
      ^ guard_rule ~target:"guard2.actual" ~declares:true ~pins:`No

let guard_control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_guard" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  let paths =
    "t/dune" :: "t/guard.ml" :: "t/gen/guard.ml" :: "t/utils.ml" :: List.map context ~f:fst
  in
  (* The probe goes where the stanza's modules live, which for the subdir arm is one level down.
     Both places are handed to the checker on every run, and the arm that is not using one writes an
     inert source there, so the argument list -- and hence which globs the checker believes it was
     given -- is the same in every arm. *)
  let run ?(probe = guard_probe) ?(at = "t/guard.ml") arm =
    List.iter [ "t/guard.ml"; "t/gen/guard.ml"; "t/utils.ml" ] ~f:(fun path ->
        write_file
          (Stdlib.Filename.concat root path)
          (if String.equal path at then probe else "let () = ()\n"));
    write_file (Stdlib.Filename.concat root "t/dune") (guard_subject ~arm);
    run_checker ~root ~exe ("." :: paths)
  in
  let report label (status, text) =
    eprintf "the guard control's %s run %s. Its captured output:\n%s\n" label
      (describe_status status) text
  in
  let exited n (status, _) = match status with Unix.WEXITED m -> m = n | _ -> false in
  let undeclared = run `Neither in
  let declared = run `Declares in
  let pinned = run `Pins in
  let pinned_elsewhere = run `Pins_elsewhere in
  let pinned_beside_helper = run `Pins_beside_a_helper in
  let implicit = run `Implicit_modules in
  let in_a_subdir = run `In_a_subdir ~at:"t/gen/guard.ml" in
  let module_without_source = run `Module_without_source in
  let utils_lookalike = run `Utils_lookalike ~at:"t/utils.ml" in
  let unknown_key = run `Unknown_key ~probe:unknown_key_probe in
  let unknown_key_pinned = run `Unknown_key_pinned ~probe:unknown_key_probe in
  let self_running_pin = run `Self_running_pin in
  let include_subdirs = run `Include_subdirs in
  let include_subdirs_nested = run `Include_subdirs_nested in
  let plain_library = run `Plain_library in
  let inline_library = run `Inline_tests_library in
  let inline_library_declares = run `Inline_tests_library_declares in
  let second_bare = run `Second_runner_bare in
  let second_declares = run `Second_runner_declares in
  let unresolvable = run `Unresolvable ~probe:opaque_probe in
  let shouted = run `Neither ~probe:shouting_probe in
  (* A fragment of the failure and of nothing else. The report's own heading names the reader and
     the key too, so a substring drawn from there would be found in every run (Codex's round-one
     lesson on the sibling controls, met here on the first try). *)
  let diagnostic = "so the guard that would have reported the variable never runs" in
  let unresolved = "with a key this scan cannot resolve" in
  let says text substring = String.is_substring text ~substring in
  let names_the_key text = says text (Utils.env_var_name guard_key) in
  let reports (result : Unix.process_status * string) fragment =
    exited 1 result && says (snd result) fragment
  in
  let passes result fragment = exited 0 result && not (says (snd result) fragment) in
  let undeclared_ok = reports undeclared diagnostic && names_the_key (snd undeclared) in
  let declared_ok = passes declared diagnostic in
  let pinned_ok = passes pinned diagnostic in
  let pinned_elsewhere_ok = reports pinned_elsewhere diagnostic in
  let pinned_beside_helper_ok = passes pinned_beside_helper diagnostic in
  let implicit_ok = reports implicit diagnostic && names_the_key (snd implicit) in
  let in_a_subdir_ok = reports in_a_subdir diagnostic && names_the_key (snd in_a_subdir) in
  let module_without_source_ok = reports module_without_source "handed no `helper.ml`" in
  let utils_lookalike_ok =
    reports utils_lookalike diagnostic && names_the_key (snd utils_lookalike)
  in
  let unknown_key_ok =
    reports unknown_key "no configuration key OCANNL reads"
    && String.is_substring (snd unknown_key) ~substring:(Utils.env_var_name guard_non_key)
  in
  let unknown_key_pinned_ok = passes unknown_key_pinned diagnostic in
  let self_running_pin_ok = passes self_running_pin diagnostic in
  let include_subdirs_ok = reports include_subdirs "include_subdirs unqualified" in
  let include_subdirs_nested_ok = reports include_subdirs_nested "include_subdirs unqualified" in
  let plain_library_ok =
    reports plain_library "from a library module" && names_the_key (snd plain_library)
  in
  let inline_library_ok =
    reports inline_library "from a library module" && names_the_key (snd inline_library)
  in
  let inline_library_declares_ok =
    reports inline_library_declares "from a library module"
    && names_the_key (snd inline_library_declares)
  in
  (* The count is what says the SECOND runner is what failed: one of the two, not both, and not the
     stanza as a whole. *)
  let second_bare_ok =
    reports second_bare diagnostic && says (snd second_bare) "1 of the 2 runs of guard.exe"
  in
  let second_declares_ok = passes second_declares diagnostic in
  let unresolvable_ok = reports unresolvable unresolved in
  let shouted_ok = reports shouted diagnostic && names_the_key (snd shouted) in
  if not undeclared_ok then report "undeclared" undeclared;
  if not declared_ok then report "declared" declared;
  if not pinned_ok then report "pinned" pinned;
  if not pinned_elsewhere_ok then report "pinned-elsewhere" pinned_elsewhere;
  if not implicit_ok then report "implicit-modules" implicit;
  if not pinned_beside_helper_ok then report "pinned-beside-a-helper" pinned_beside_helper;
  if not in_a_subdir_ok then report "in-a-subdir" in_a_subdir;
  if not module_without_source_ok then report "module-without-source" module_without_source;
  if not utils_lookalike_ok then report "utils-lookalike" utils_lookalike;
  if not unknown_key_ok then report "unknown-key" unknown_key;
  if not unknown_key_pinned_ok then report "unknown-key-pinned" unknown_key_pinned;
  if not self_running_pin_ok then report "self-running-pin" self_running_pin;
  if not include_subdirs_ok then report "include-subdirs" include_subdirs;
  if not include_subdirs_nested_ok then report "include-subdirs-nested" include_subdirs_nested;
  if not plain_library_ok then report "plain-library" plain_library;
  if not inline_library_ok then report "inline-tests-library" inline_library;
  if not inline_library_declares_ok then
    report "inline-tests-library-declares" inline_library_declares;
  if not second_bare_ok then report "second-runner-bare" second_bare;
  if not second_declares_ok then report "second-runner-declares" second_declares;
  if not unresolvable_ok then report "unresolvable" unresolvable;
  if not shouted_ok then report "shouted-key" shouted;
  printf
    "\n\
     The guard rule is put to a tree of one `(executable)` whose module reads a configuration key\n\
     from a LIST it hands to `Utils.read_env_var`. The arms differ in how the rules running it\n\
     answer for `(env_var %s)`, in whether the stanza writes its `(modules …)` down, and in whether\n\
     the key list is resolvable at all. Nothing else differs between the runs.\n\n"
    (Utils.env_var_name guard_key);
  Verdict.p
    "the checker reports the key and exits 1 when the rule running the guard neither declares nor \
     pins it"
    undeclared_ok;
  Verdict.p "the same tree with the declaration added passes and says nothing about it" declared_ok;
  Verdict.p "and so does the same tree pinning the variable with `setenv` instead" pinned_ok;
  Verdict.p
    "a `setenv` over a SIBLING branch of the same action does not pin this run, and is reported"
    pinned_elsewhere_ok;
  Verdict.p "and an unpinned run of a HELPER beside a pinned guard does not un-pin it"
    pinned_beside_helper_ok;
  Verdict.p
    "a stanza reaching for dune's default module set is judged too, not skipped for lack of a \
     `(modules …)` field"
    implicit_ok;
  Verdict.p
    "a program declared inside a `(subdir …)` is reached, not read as a stanza with no modules"
    in_a_subdir_ok;
  Verdict.p
    "a module the stanza names and this check was handed no source for is reported, not read as \
     one that makes no reads"
    module_without_source_ok;
  Verdict.p
    "a test directory's own `utils.ml` is not the module that defines the reader, and is not exempt"
    utils_lookalike_ok;
  Verdict.p "a key the registry does not know is still asked for, and is reported as undeclarable"
    unknown_key_ok;
  Verdict.p "and pinning it is the way to answer for one" unknown_key_pinned_ok;
  Verdict.p
    "a stanza dune runs itself pins through its own action, and is not asked to declare besides"
    self_running_pin_ok;
  Verdict.p "a dune file whose module sets this scan cannot place is refused, not approximated"
    include_subdirs_ok;
  Verdict.p "and the same directive inside a `(subdir …)` does not slip past the refusal"
    include_subdirs_nested_ok;
  Verdict.p
    "a guard in a plain library is reported, there being no `deps` field in reach to declare it"
    plain_library_ok;
  Verdict.p "a library with inline tests is refused the same way, being linkable all the same"
    inline_library_ok;
  Verdict.p
    "and declaring the variable in `(inline_tests (deps …))` is no licence: that invalidates the \
     inline runner alone"
    inline_library_declares_ok;
  Verdict.p
    "a second rule running the same executable answers for its own run: one declaring does not \
     cover the other"
    second_bare_ok;
  Verdict.p "and the same pair with both declaring passes" second_declares_ok;
  Verdict.p
    "a dynamic reach whose keys resolve to nothing is refused rather than passed over in silence"
    unresolvable_ok;
  Verdict.p
    "a key spelled in upper case names the same variable, and is asked for under the same name"
    shouted_ok;
  try remove_tree root with Unix.Unix_error _ -> ()

let () =
  match Array.to_list argv with
  | _ :: [ "--control" ] ->
      per_directory_control ();
      refusal_control ();
      control ();
      floor_control ();
      guard_control ();
      family_control ();
      (* Dune's repository-wide rule hands the same source to [main] as [./env_var_deps.ml] after a
         full build has materialized the local build-tree copy. Exercise that spelling here too: the
         manifest identity is repository-relative even when the file used to extract the diagnostics
         is local to the action's cwd. *)
      Refusal_manifest.print "./env_var_deps.ml"
  | _ -> main ()
