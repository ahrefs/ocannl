(** The resolution half of [test/operations/config_dep_completeness] (gh-ocannl-603): given the dune
    files of a tree and where its [ocannl_config] files are, which configuration each test-running
    site's process will read, whether the site's dependencies name THAT one, whether a change of the
    ambient environment reaches the run, and which exemptions were spent.

    The scanner half, {!Dune_stanza_scan}, is pinned by [test/operations/dune_scan_cases] on text
    built to break it. This half used to live inside the check's [let () =] main, where its only
    test was a golden over the live repository -- which can say "today's tree is consistent" and
    never "a tree where X is wrong gets rejected". Twenty-odd review rounds of resolution fixes (a
    sibling directory's config, a cross-directory [copy_files], exemption-after-availability
    ordering, the root normalization and the root exclusion) were each validated by planting files
    by hand and reverting them, and none left a test behind. Here the resolution reads nothing from
    the filesystem: it takes the tree as data and returns its verdicts as data, so
    [test/operations/config_dep_resolution_cases] can hand it synthetic trees with known answers.

    What it returns is typed rather than worded: the check's main turns each verdict into its
    refusal, keeping every diagnostic a literal in the scanner source, where the refusal-control
    catalogue ([Refusal_control_manifest]) finds and holds it. *)

open Base
module Scan = Dune_stanza_scan

(** Where the ambient record and the configuration carrying it are produced (gh-ocannl-1149). *)
let ambient_home = "test/config"

let ambient_subdir = "ambient"

let ambient_config_source =
  String.concat ~sep:"/" [ ambient_home; ambient_subdir; Scan.config_file ]

let ambient_record_source = ambient_home ^ "/" ^ Scan.ambient_file

(** The repository root in one spelling: [""], where [Filename.dirname] says ["."]. *)
let root_normalized directory = match directory with "." -> "" | directory -> directory

(** [path], written relative to the repository-relative directory [dir], made repository-relative.
    [dir] is split into its components first: handed over whole, as one component, a [..] in [path]
    would pop all of [test/a] rather than [a] alone, so a [(chdir ../b ...)] in [test/a] would run
    in a top-level [b]. That held wherever the dune file sits one level deep, which is where the
    sibling [chdir] fixes were validated by hand, and no deeper (gh-ocannl-603). *)
let under dir path = Scan.repo_relative (Scan.split_path dir) path

(** The directories a process running in [directory] searches for its configuration, nearest first:
    [directory] and every ancestor BELOW the repository root, as [Utils.config_file_args] walks
    them.

    The root itself is not one of those stops, even though the executable's own search does end
    there. A root [ocannl_config] is gitignored personal dev settings, so letting one satisfy a site
    would make the check pass on the developer's machine and fail in CI, for the very directory
    whose missing [(copy_files ../config/ocannl_config)] it exists to catch
    (lukstafi/ocannl-staging#346). Only a config a checked-in stanza puts in reach counts. *)
let search_path directory =
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
  ancestors (root_normalized directory)

(** The first directory on [directory]'s {!search_path} that [has_config]: both the file that must
    exist and the one the dependency must name, whether the process runs in a descendant of the
    stanza's directory, a sibling, or the stanza's own (Codex P2, rounds 9 to 11 of PR #343). *)
let nearest_config ~has_config directory = List.find (search_path directory) ~f:has_config

(** Whether one of [declared] -- dependency paths as the stanza in [stanza_dir] writes them -- names
    the config in [nearest]: the one the process will actually read, not merely some ancestor's
    (Codex P2, round 10 of PR #343). Paths are resolved from the stanza's directory before
    comparing. *)
let declares_nearest ~stanza_dir ~nearest declared =
  List.exists declared ~f:(fun path ->
      String.equal
        (Stdlib.Filename.dirname (under stanza_dir path))
        (if String.is_empty nearest then "." else nearest))

type verdict =
  | Counted  (** declares the config its process reads, and the ambient environment reaches it *)
  | Exempt  (** an exemption was spent on it *)
  | No_config_in_reach  (** no directory on its search path has a config to depend on *)
  | Unestablished_directory
      (** a shell action, not exempt: which config the process finds cannot be established *)
  | Ambient_unreached of string
      (** declares the config in this directory (["."] for the root), which carries no record of the
          ambient environment, and depends on neither that record nor [(universe)] *)
  | Unreadable_undeclared  (** runs a command the scan cannot read, and declares no config *)
  | Unclassified_undeclared  (** has an action head the scan cannot place, and declares no config *)
  | Undeclared  (** declares no config, or not the one its process reads *)

type site_report = {
  site : Scan.site;
  key : string;  (** ["<directory the process runs in>:<name>"], what an exemption is keyed by *)
  directory : string;  (** the directory the process runs in, repository-relative *)
  nearest : string option;  (** {!nearest_config} of [directory] *)
  needs_isolating_sandbox : bool;
      (** what it runs is past the scan's reach, and it runs outside a sandbox that copies
          (gh-ocannl-1359) -- whether or not it is exempt from the config dep *)
  verdict : verdict;
}

(** Where the raw-text reader ({!Scan.raw_stanzas}) saw more than the sexp walk placed: the walk is
    reading the file with a hole in it. *)
type floor_hole =
  | Test_hole of { cwd : string; in_text : int; in_walk : int }
  | Inline_hole of { dir : string; in_text : int; in_walk : int }
  | Unnameable_hole of { dir : string; in_text : int; in_walk : int }
  | Run_hole of { cwd : string; exe : string; in_text : int; in_walk : int }

type file_report = {
  dune_file : string;
  unclassified_heads : string list;
  path_rewriting : string list;
  sites : site_report list;
  floor_holes : floor_hole list;
  floor_applies : bool;  (** whether the raw text showed anything for the floors to hold *)
  test_floor : int;
  inline_floor : int;
  run_floor : int;
}

type report = {
  ambient_producer_ok : bool;
      (** [test/config/dune] still produces the ambient record and the configuration carrying it *)
  ambient_config_dirs : string list;  (** where a copy of {!ambient_config_source} lands *)
  ambient_record_dirs : string list;
      (** where {!ambient_record_source} is in reach, [ambient_home] included *)
  files : file_report list;
  stale_exemptions : string list;  (** exemption keys no site spent *)
}

let string_set = Set.of_list (module String)

(** The ambient producer (gh-ocannl-1149): [test/config/dune] records the ambient [OCANNL_*]
    environment in [ocannl_ambient], under [(universe)], and appends that record to the shared
    configuration as [test/config/ambient/ocannl_config]. The whole guarantee rests on both rules,
    so they are checked rather than trusted. *)
let ambient_producer_ok dune_files =
  let producer =
    List.find dune_files ~f:(fun (path, _) -> String.equal path (ambient_home ^ "/dune"))
  in
  let rules =
    match producer with
    | None -> []
    | Some (_, content) ->
        Scan.walk "" (Scan.stanzas content) ~f:(fun subdir stanza ->
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
  produces ~subdir:"" ~target:Scan.ambient_file ~deps_ok:Scan.depends_on_universe
  && produces ~subdir:ambient_subdir ~target:Scan.config_file ~deps_ok:(fun deps ->
      List.mem
        (Scan.declared_config_paths ~name:Scan.ambient_file deps)
        ("../" ^ Scan.ambient_file) ~equal:String.equal)

let tally_of keys =
  List.fold keys
    ~init:(Map.empty (module String))
    ~f:(fun counts key -> Map.update counts key ~f:(fun n -> 1 + Option.value n ~default:0))

(** [analyze ~exemption_keys ~config_files dune_files]: [dune_files] are [(path, content)] pairs and
    [config_files] paths, all repository-relative; [exemption_keys] are the {!site_report.key}s a
    site may spend instead of declaring the config. *)
let analyze ~exemption_keys ~config_files dune_files =
  let dune_files =
    List.dedup_and_sort dune_files ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  (* Where an `ocannl_config` resolves: a file present next to the dune file -- checked in, or one a
     `(copy_files ...)` put there, since the glob that supplies these paths matches dune's targets
     as well as its sources -- or one a stanza copies there, read from the stanzas so that a
     directory whose config has simply never been built still counts. Gathered over every dune file
     before anything is checked: a stanza in one directory may run something in another, whose
     config a THIRD file materializes (Codex P2, round 14 of PR #343). *)
  let config_dirs =
    List.map config_files ~f:(fun path -> root_normalized (Stdlib.Filename.dirname path))
    |> string_set
  in
  let copied_config_dirs =
    List.concat_map dune_files ~f:(fun (dune_file, content) ->
        let dir = Stdlib.Filename.dirname dune_file in
        Scan.config_copy_dirs content |> List.map ~f:(fun subdir -> under dir subdir))
    |> string_set
  in
  let has_config directory =
    Set.mem config_dirs directory || Set.mem copied_config_dirs directory
  in
  let copies name =
    List.concat_map dune_files ~f:(fun (dune_file, content) ->
        let dir = Stdlib.Filename.dirname dune_file in
        Scan.copy_sources content ~name
        |> List.map ~f:(fun (subdir, source) ->
            let into = under dir subdir in
            (into, under into source)))
  in
  let copied_from name source =
    List.filter_map (copies name) ~f:(fun (into, from) ->
        Option.some_if (String.equal from source) into)
    |> string_set
  in
  (* A copy of the ambient configuration changes whenever the environment does, so a rule depending
     on it reruns; a configuration checked in beside its tests carries no record, and a rule reading
     one depends on `ocannl_ambient` itself. *)
  let ambient_config_dirs = copied_from Scan.config_file ambient_config_source in
  let ambient_record_dirs =
    Set.add (copied_from Scan.ambient_file ambient_record_source) ambient_home
  in
  let exemptions = string_set exemption_keys in
  let spent = ref (Set.empty (module String)) in
  let files =
    List.map dune_files ~f:(fun (dune_file, content) ->
        let dir = Stdlib.Filename.dirname dune_file in
        (* Both displacements matter and they compose: a `(subdir …)` stanza applies elsewhere, and
           a `chdir` action runs elsewhere again. The config an executable finds is the one in the
           directory the PROCESS runs in. *)
        let stanza_dir site = under dir site.Scan.subdir in
        let effective site = Scan.in_subdir site.Scan.subdir site.Scan.cwd in
        let sites =
          List.map (Scan.sites content) ~f:(fun site ->
              let directory = under dir (effective site) in
              let key = directory ^ ":" ^ site.Scan.name in
              let nearest = nearest_config ~has_config directory in
              let declares_config =
                Option.exists nearest ~f:(fun nearest ->
                    declares_nearest ~stanza_dir:(stanza_dir site) ~nearest
                      site.Scan.declared_config_paths)
              in
              (* Whether a change of the environment reaches this run (gh-ocannl-1149): through the
                 configuration it depends on, when that is a copy carrying the ambient record;
                 through the record itself, depended on beside a configuration checked in; or
                 through `(universe)`, which reruns the rule on every invocation anyway. *)
              let ambient_covered =
                site.Scan.depends_on_universe
                || Option.exists nearest ~f:(Set.mem ambient_config_dirs)
                || List.exists site.Scan.declared_ambient_paths ~f:(fun declared ->
                    Set.mem ambient_record_dirs
                      (Stdlib.Filename.dirname (under (stanza_dir site) declared)))
              in
              let kind = site.Scan.kind in
              (* An exemption is spent only where the dep is actually absent, so declaring it anyway
                 makes the entry stale and the list gets pruned rather than growing quietly. A shell
                 action needs one even where the dep is declared, since the dep cannot be shown to
                 be the right one. *)
              let exempt =
                Set.mem exemptions key
                &&
                match kind with
                | Scan.Unreadable_directory -> true
                | Scan.Runs_executable -> not declares_config
                | _ -> false
              in
              if exempt then spent := Set.add !spent key;
              (* What the check requires is the dependency; an extra one harms nothing. So a stanza
                 the scan cannot read through matters only where the dependency is ABSENT. Except
                 when what is unknown is WHERE it runs: no dependency can be shown to build the
                 config that process will find, so declaring one settles nothing (Codex P2, round
                 8). An exempt site needs no config where it runs -- that is what the exemption says
                 -- so availability is checked after it, not over every site (Codex P2, round
                 11). *)
              let verdict =
                match (declares_config, exempt, kind) with
                | _, true, _ -> Exempt
                | _, false, _ when Option.is_none nearest -> No_config_in_reach
                | _, false, Scan.Unreadable_directory -> Unestablished_directory
                | true, _, _ when not ambient_covered ->
                    Ambient_unreached
                      (Option.value_map nearest ~default:"." ~f:(fun d ->
                           if String.is_empty d then "." else d))
                | true, _, _ -> Counted
                | false, false, Scan.Unreadable_command -> Unreadable_undeclared
                | false, false, Scan.Unclassified_action -> Unclassified_undeclared
                | false, false, _ -> Undeclared
              in
              {
                site;
                key;
                directory;
                nearest;
                needs_isolating_sandbox =
                  (match kind with
                  | Scan.Unreadable_directory -> not site.Scan.isolating_sandbox
                  | _ -> false);
                verdict;
              })
        in
        (* The floor the raw text puts under the sexp walk: a second reader sharing none of the
           walk's machinery. Sites count including exempt ones -- an exemption is about the config
           dependency, not about whether the stanza was seen. Tests and inline-test libraries
           compare per working directory (a test action can run `%{test}` under several `chdir`s,
           each its own site), PATH-rewritten bare commands per directory on the site's own
           identity, and rules as a MULTISET over (working directory, executable), the pair the walk
           makes one site of -- coarser fails to detect, finer fails a correct scan (Codex P2,
           rounds 1 to 10 of PR #343). *)
        let raw = Scan.raw_stanzas content in
        let placed ~f = List.filter_map sites ~f:(fun r -> f r.site) |> tally_of in
        let holes floor ~placed ~hole =
          Map.to_alist (tally_of floor)
          |> List.filter_map ~f:(fun (key, in_text) ->
              let in_walk = Option.value (Map.find placed key) ~default:0 in
              Option.some_if (in_walk < in_text) (hole key ~in_text ~in_walk))
        in
        let test_floor = List.concat_map raw ~f:(fun r -> r.Scan.raw_test_cwds) in
        let inline_floor =
          List.filter_map raw ~f:(fun r ->
              Option.some_if
                (String.equal r.Scan.raw_head "library" && r.Scan.raw_inline_tests)
                r.Scan.raw_subdir)
        in
        let unnameable_floor = List.concat_map raw ~f:(fun r -> r.Scan.raw_unnameable) in
        let run_key (cwd, exe) = cwd ^ "\000" ^ exe in
        let run_floor = List.concat_map raw ~f:(fun r -> r.Scan.raw_runs) in
        let floor_holes =
          holes test_floor
            ~placed:
              (placed ~f:(fun site ->
                   match site.Scan.kind with Scan.Test -> Some (effective site) | _ -> None))
            ~hole:(fun cwd ~in_text ~in_walk -> Test_hole { cwd; in_text; in_walk })
          @ holes inline_floor
              ~placed:
                (placed ~f:(fun site ->
                     match site.Scan.kind with
                     | Scan.Inline_tests -> Some (effective site)
                     | _ -> None))
              ~hole:(fun dir ~in_text ~in_walk -> Inline_hole { dir; in_text; in_walk })
          @ holes unnameable_floor
              ~placed:
                (placed ~f:(fun site -> Option.some_if site.Scan.path_rewritten (effective site)))
              ~hole:(fun dir ~in_text ~in_walk -> Unnameable_hole { dir; in_text; in_walk })
          @ holes (List.map run_floor ~f:run_key)
              ~placed:
                (List.concat_map sites ~f:(fun r ->
                     List.map r.site.Scan.executables ~f:(fun exe ->
                         run_key (effective r.site, exe)))
                |> tally_of)
              ~hole:(fun key ~in_text ~in_walk ->
                let cwd, exe = String.lsplit2_exn key ~on:'\000' in
                Run_hole { cwd; exe; in_text; in_walk })
        in
        {
          dune_file;
          unclassified_heads = Scan.unclassified_heads content;
          path_rewriting = Scan.path_rewriting_stanzas content;
          sites;
          floor_holes;
          floor_applies =
            not
              (List.is_empty test_floor && List.is_empty inline_floor && List.is_empty run_floor
             && List.is_empty unnameable_floor);
          test_floor = List.length test_floor;
          inline_floor = List.length inline_floor;
          run_floor = List.length run_floor;
        })
  in
  {
    ambient_producer_ok = ambient_producer_ok dune_files;
    ambient_config_dirs = Set.to_list ambient_config_dirs;
    ambient_record_dirs = Set.to_list ambient_record_dirs;
    files;
    stale_exemptions = Set.to_list (Set.diff exemptions !spent);
  }
