(* gh-ocannl-1748: no rule spawns a command line Windows cannot start.

   Windows caps a whole command line at 32,767 characters. Past it [CreateProcess] fails and dune
   reports [Error: CreateProcess(): No such file or directory], naming neither the length nor the
   program. gh-ocannl-1720 was that failure: [verdict_ratchet] put every test source on its command
   line through [%{deps}], the line had reached 32,524 characters on the last green Windows run, and
   ordinary merges elsewhere pushed it over -- on the scheduled Windows run, a day late, attributed
   to a neighbouring rule. Linux and macOS have no such cap, so per-PR CI could not see it coming.
   This scan makes it visible there.

   {1 What it reads}

   Every [dune] file (and [dune-file], dune's alternative name) in dune's own source tree: the rule
   declares [(sandbox always)] and [(source_tree ../..)], and [Test_scan.Source_inventory] lists
   what dune copied, so a directory dune excludes is excluded here too. Each file is read through
   [Test_scan.Dune_stanza_scan], which refuses one that sexplib and dune would read differently. In
   every stanza, at any depth, it reads each [(action ...)] field -- rules, aliases, tests, a
   [(preprocess (action ...))] -- and a rule's short form, whose body IS its action. It walks dune's
   action language: [run], [dynamic-run], [system] and [bash] spawn a process, [chdir] moves where
   paths are spelled from, and every other head of [Dune_stanza_scan.inert_actions] nests further
   actions without spawning. A head on neither list is refused, as is a [chdir] destination it
   cannot resolve and an [include] stanza, whose rules live in a file it does not read.

   {1 How it measures: bounded, or refused}

   For each spawned command it computes an UPPER BOUND on the line Windows is handed, from what each
   atom expands to: literal text as written; a path pform ([%{dep:}], [%{exe:}], [%{target}],
   [%{targets}], [%{test}], [%{workspace_root}], a named dependency bound to files) as the path from
   the action's working directory, plus two for a leading [./]; an install or toolchain location
   ([%{lib:}], [%{bin:}], [%{ocamlc}], [%{ocaml-config:}]) as {!install_path_allowance}; and
   [%{deps}] as the sum of its entries. Every argument gains three for its separator and Windows'
   quoting plus one per quote or backslash, the program position gains {!install_path_allowance}
   (dune resolves a bare name to an absolute path), and a shell line gains {!shell_allowance} for
   the shell's own argv. A command whose bound reaches {!limit} fails, and a [(system ...)] line
   fails at {!cmd_limit}, since cmd.exe runs it and caps it at 8,191.

   It does not model a dependency SET. [(glob_files ...)], [(glob_files_rec ...)], [(source_tree
   ...)] and [(package ...)] expand to whatever dune finds when the rule runs, and dune's globs
   match GENERATED files as well as sources: about half of what [env_var_deps.filelist] holds is
   [.pp.ml] renderings, which no reading of the source tree sees, beside [.exe]s, copied configs and
   the targets of every rule. A bound computed from sources would be a lower bound passed off as an
   upper one, so such an expansion on a spawned command line is REFUSED, whatever its size today: it
   grows with the repository, and the only reading that cannot go stale is that it may not appear
   there. The fix is gh-ocannl-1720's: write [%{deps}] to a response file in a non-spawning [(echo
   ...)], hand the program [@<file>], and read it through [Test_scan.Scan_argv.expand]. Equally
   refused: a pform with no bound on this list, [%{read:}] of a file {!judged_reads} does not name,
   and a dependency form it does not know.

   {1 What it deliberately does not read}

   - The command lines dune builds for itself -- compiling, linking, running a ppx driver. They are
   dune's, not this repository's, and the scan has no way to read them. - The environment block.
   Windows caps it separately; [(setenv ...)] is walked through for the action it wraps. - Escapes.
   A [\%{...}] that dune reads as literal text is read here as a live pform -- a refusal or an
   over-count, the loud direction, so it is left that way. - [enabled_if]. A rule disabled on this
   host is measured all the same, since Windows is where the length matters.

   The live tree is normally clean, so the refusals are otherwise dead code. [control] runs this
   executable as a child over synthetic trees -- one passing, gh-ocannl-1720's own shape among the
   refused ones -- and asserts each verdict and its diagnostic. *)

open Base
open Stdio
module Scan = Test_scan.Dune_stanza_scan
module Inventory = Test_scan.Source_inventory

(** The bound at which a spawned command line fails: 4,767 characters short of Windows' 32,767, so a
    rule fails on Linux while the repository has a few hundred paths of growth left. *)
let limit = 28_000

let windows_cap = 32_767

(** The same margin for a [(system ...)] line, which dune hands to [cmd.exe /c] on Windows: cmd.exe
    caps the command it is given at 8,191 characters, a quarter of what [CreateProcess] takes. *)
let cmd_limit = 7_000

let cmd_cap = 8_191

(** What an install-location or toolchain pform may expand to: Windows' classic [MAX_PATH]. *)
let install_path_allowance = 260

(** The shell's own argv in front of a [(system ...)] or [(bash ...)] line: its absolute path and
    flags ([cmd /c], [bash -e -u -o pipefail -c]). *)
let shell_allowance = 300

(** Files whose contents a spawned command line may carry through [%{read:...}], each with a bound
    and why it holds. Judged, not derived: the reader cannot know what a generated file will hold. A
    row no spawned command reads any more fails as stale. *)
let judged_reads =
  [
    ( "test/config/ocannl_backend.txt",
      64,
      "one backend name, printed by `ocannl_read_config --read=backend`" );
    ( "test/config/ocannl_backend_extension.txt",
      64,
      "one generated-code file extension, printed by `ocannl_read_config --read=backend_extension`"
    );
  ]

(* Paths as component lists from the workspace root. *)

let components path =
  String.split_on_chars path ~on:[ '/'; '\\' ]
  |> List.filter ~f:(fun c -> not (String.is_empty c || String.equal c "."))

let resolve dir path =
  List.fold (components path) ~init:(List.rev dir) ~f:(fun acc c ->
      match (c, acc) with "..", _ :: rest -> rest | "..", [] -> [] | c, acc -> c :: acc)
  |> List.rev

(** The path dune spells for [target] from [cwd]. *)
let relpath ~cwd target =
  let rec drop a b =
    match (a, b) with x :: a', y :: b' when String.equal x y -> drop a' b' | _ -> (a, b)
  in
  let up, down = drop cwd target in
  let parts = List.map up ~f:(fun _ -> "..") @ down in
  if List.is_empty parts then "." else String.concat ~sep:"/" parts

(** A path's length on the command line, with room for a leading [./]. *)
let path_cost ~cwd target = String.length (relpath ~cwd target) + 2

type stanza = {
  file : string;  (** the dune file, from the workspace root *)
  line : int;
  dir : string list;  (** the directory the stanza applies to, after [subdir] *)
  sexp : Sexp.t;
}

let identity st =
  let head = Option.value (Scan.head st.sexp) ~default:"?" in
  let named =
    List.find_map [ "name"; "names"; "target"; "targets"; "alias" ] ~f:(fun f ->
        Option.map (Scan.field st.sexp f) ~f:(fun args ->
            String.concat ~sep:" " (List.concat_map args ~f:Scan.atoms)))
  in
  Printf.sprintf "%s:%d (%s%s)" st.file st.line head
    (match named with Some n -> " " ^ n | None -> "")

let has_pform s = String.is_substring s ~substring:"%{"
let ( let* ) r f = Result.bind r ~f

let sum_results results =
  List.fold_result results ~init:0 ~f:(fun acc r -> Result.map r ~f:(( + ) acc))

let unbounded_dependency form =
  Error
    (Printf.sprintf
       "`%s` expands to a dependency set that grows with the repository, including the generated \
        files dune's globs also match, so no reading of the source tree bounds it"
       (Sexp.to_string form))

(* One dependency entry, as it contributes to [%{deps}] or to a named dependency's expansion. *)
let rec entry_cost st ~cwd ~pform entry =
  match entry with
  | Sexp.Atom a when not (has_pform a) -> Ok (path_cost ~cwd (resolve st.dir a) + 1)
  | Sexp.Atom a -> (
      match Scan.pieces a with
      | [ Scan.Pform p ] -> Result.map (pform st ~cwd p) ~f:(( + ) 1)
      | _ ->
          Error
            (Printf.sprintf "dependency `%s` mixes a pform with text, which this scan cannot bound"
               a))
  | Sexp.List (Sexp.Atom name :: entries) when String.is_prefix name ~prefix:":" ->
      sum_results (List.map entries ~f:(entry_cost st ~cwd ~pform))
  | Sexp.List [ Sexp.Atom "file"; Sexp.Atom a ] when not (has_pform a) ->
      Ok (path_cost ~cwd (resolve st.dir a) + 1)
  | Sexp.List (Sexp.Atom ("alias" | "alias_rec" | "env_var" | "universe" | "sandbox") :: _) -> Ok 0
  | Sexp.List (Sexp.Atom ("glob_files" | "glob_files_rec" | "source_tree" | "package") :: _) ->
      unbounded_dependency entry
  | _ ->
      Error
        (Printf.sprintf "dependency form `%s` is one this scan cannot bound" (Sexp.to_string entry))

let deps_of st = Option.value (Scan.field st.sexp "deps") ~default:[]

let named_dep st name =
  List.find_map (deps_of st) ~f:(function
    | Sexp.List (Sexp.Atom n :: entries) when String.equal n (":" ^ name) -> Some entries
    | _ -> None)

let field_paths st ~cwd name =
  match Scan.field st.sexp name with
  | Some atoms when List.for_all atoms ~f:(function Sexp.Atom a -> not (has_pform a) | _ -> false)
    ->
      Ok
        (List.sum
           (module Int)
           atoms
           ~f:(function Sexp.Atom a -> path_cost ~cwd (resolve st.dir a) + 1 | _ -> 0))
  | Some _ -> Error (Printf.sprintf "`%%{%s}` names a target spelled with a pform" name)
  | None -> Error (Printf.sprintf "`%%{%s}` in a stanza with no `(%s ...)` field" name name)

(* The judged reads a command line used, for the stale-row check. *)
let reads_used = Hash_set.create (module String)

(** The upper bound on one pform's expansion, or why it has none. *)
let rec pform_cost st ~cwd p =
  let name, arg =
    match String.lsplit2 p ~on:':' with Some (n, a) -> (n, Some a) | None -> (p, None)
  in
  match (name, arg) with
  | "deps", None -> sum_results (List.map (deps_of st) ~f:(entry_cost st ~cwd ~pform:pform_cost))
  | ("dep" | "exe" | "path" | "path-no-dep"), Some a
    when not (String.is_empty a || String.is_substring a ~substring:"%{") ->
      Ok (path_cost ~cwd (resolve st.dir a))
  | "target", None -> field_paths st ~cwd "target"
  | "targets", None -> field_paths st ~cwd "targets"
  | "test", None ->
      let names =
        match (Scan.field st.sexp "name", Scan.field st.sexp "names") with
        | Some names, _ | None, Some names -> List.concat_map names ~f:Scan.atoms
        | None, None -> []
      in
      if List.is_empty names then Error "`%{test}` in a stanza that names no test"
      else
        Ok
          (List.fold names ~init:0 ~f:(fun acc n ->
               max acc (path_cost ~cwd (resolve st.dir (n ^ ".exe")))))
  | ("workspace_root" | "project_root"), None -> Ok (path_cost ~cwd [])
  | ("lib" | "lib-private" | "libexec" | "libexec-private" | "bin" | "ocaml-config"), Some _
  | ("ocaml" | "ocamlc" | "ocamlopt"), None ->
      Ok install_path_allowance
  | "read", Some a -> (
      let path = String.concat ~sep:"/" (resolve st.dir a) in
      match List.find judged_reads ~f:(fun (p, _, _) -> String.equal p path) with
      | Some (_, bound, _) ->
          Hash_set.add reads_used path;
          Ok bound
      | None ->
          Error
            (Printf.sprintf
               "`%%{%s}` puts the contents of `%s` on the command line, a file `judged_reads` does \
                not bound"
               p path))
  | name, None when Option.is_some (named_dep st name) ->
      sum_results
        (List.map (Option.value_exn (named_dep st name)) ~f:(entry_cost st ~cwd ~pform:pform_cost))
  | _ -> Error (Printf.sprintf "`%%{%s}` is a pform this scan has no bound for" p)

let atom_cost st ~cwd atom =
  let* text =
    sum_results
      (List.map (Scan.pieces atom) ~f:(function
        | Scan.Literal s -> Ok (String.length s)
        | Scan.Pform p -> pform_cost st ~cwd p))
  in
  Ok (text + 3 + String.count atom ~f:(fun c -> Char.equal c '"' || Char.equal c '\\'))

type spawn = {
  stanza : stanza;
  command : string;
  cost : (int, string) Result.t;
  limit : int * int * string;  (** where it fails, the cap that limit keeps clear of, and whose *)
}

let limit_of = function
  | Sexp.List (Sexp.Atom "system" :: _) -> (cmd_limit, cmd_cap, "cmd.exe")
  | _ -> (limit, windows_cap, "Windows")

let spawn_cost st ~cwd sexp =
  match sexp with
  | Sexp.List (Sexp.Atom ("run" | "dynamic-run") :: (Sexp.Atom _ :: _ as argv)) ->
      let* costs =
        Result.all
          (List.map argv ~f:(function
            | Sexp.Atom a -> atom_cost st ~cwd a
            | form ->
                Error
                  (Printf.sprintf "argument `%s` is a list, which this scan cannot bound"
                     (Sexp.to_string form))))
      in
      Ok (List.sum (module Int) costs ~f:Fn.id + install_path_allowance)
  | Sexp.List [ Sexp.Atom ("system" | "bash"); Sexp.Atom line ] ->
      Result.map (atom_cost st ~cwd line) ~f:(( + ) shell_allowance)
  | form ->
      Error (Printf.sprintf "command `%s` is not one this scan can read" (Sexp.to_string_hum form))

(* Where a [chdir] destination puts the working directory. *)
let chdir_target st ~cwd dir =
  match Scan.pieces dir with
  | [ Scan.Pform ("workspace_root" | "project_root") ] -> Ok []
  | [ Scan.Pform ("workspace_root" | "project_root"); Scan.Literal rest ] -> Ok (resolve [] rest)
  | _ when not (has_pform dir) -> Ok (resolve cwd dir)
  | _ ->
      Error
        (Printf.sprintf "`(chdir %s ...)` in %s moves to a directory this scan cannot resolve" dir
           (identity st))

let rec spawns_in st ~cwd sexp =
  match sexp with
  | Sexp.List (Sexp.Atom h :: _) when List.mem Scan.program_actions h ~equal:String.equal ->
      [
        Ok
          {
            stanza = st;
            command = Sexp.to_string sexp;
            cost = spawn_cost st ~cwd sexp;
            limit = limit_of sexp;
          };
      ]
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest) -> (
      match chdir_target st ~cwd dir with
      | Ok cwd -> List.concat_map rest ~f:(spawns_in st ~cwd)
      | Error e -> [ Error e ])
  | Sexp.List (Sexp.Atom h :: rest) when List.mem Scan.inert_actions h ~equal:String.equal ->
      List.concat_map rest ~f:(function
        | Sexp.List _ as a -> spawns_in st ~cwd a
        | Sexp.Atom _ -> [])
  | Sexp.List (Sexp.Atom h :: _) ->
      [
        Error
          (Printf.sprintf "action head `%s` in %s is on neither of the action lists this scan reads"
             h (identity st));
      ]
  | _ -> []

let is_action_head h =
  List.mem Scan.program_actions h ~equal:String.equal
  || List.mem Scan.inert_actions h ~equal:String.equal

(* Every action a stanza carries, at any depth: an [(action ...)] field, or a rule's short form. *)
let actions_of sexp =
  let rec fields = function
    | Sexp.List (Sexp.Atom "action" :: body) -> body
    | Sexp.List l -> List.concat_map l ~f:fields
    | Sexp.Atom _ -> []
  in
  match (sexp, fields sexp) with
  | Sexp.List (Sexp.Atom "rule" :: body), [] ->
      List.filter body ~f:(fun f ->
          match Scan.head f with Some h -> is_action_head h | None -> false)
  | _, found -> found

let rec stanzas_in ~file ~content ~dir pairs =
  List.concat_map pairs ~f:(fun (sexp, (raw : Scan.raw_form)) ->
      match sexp with
      | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body) ->
          stanzas_in ~file ~content ~dir:(resolve dir sub)
            (List.zip_exn body (List.drop raw.raw_children 2))
      | _ -> [ { file; line = Scan.line_of content raw.raw_start; dir; sexp } ])

type findings = {
  dune_files : string list;
  unreadable : string list;
  spawns : spawn list;
  refused : string list;  (** sites with no bound, and stanzas the scan cannot place *)
}

let scan_tree inventory =
  let dune_files =
    Inventory.select inventory ~f:(fun p ->
        List.mem [ "dune"; "dune-file" ] (Stdlib.Filename.basename p) ~equal:String.equal)
  in
  let unreadable = ref [] and spawns = ref [] and refused = ref [] in
  List.iter dune_files ~f:(fun (f : Inventory.file) ->
      let content = In_channel.read_all f.on_disk in
      match Scan.spelled_stanzas content with
      | exception exn ->
          unreadable := Printf.sprintf "%s: %s" f.path (Exn.to_string exn) :: !unreadable
      | pairs ->
          let dir = List.drop_last_exn (components f.path) in
          List.iter (stanzas_in ~file:f.path ~content ~dir pairs) ~f:(fun st ->
              match Scan.head st.sexp with
              | Some ("include" | "dynamic_include") ->
                  refused :=
                    Printf.sprintf "%s includes stanzas from a file this scan does not read"
                      (identity st)
                    :: !refused
              | _ ->
                  List.iter (actions_of st.sexp) ~f:(fun action ->
                      List.iter (spawns_in st ~cwd:st.dir action) ~f:(function
                        | Ok s -> spawns := s :: !spawns
                        | Error e -> refused := e :: !refused))));
  {
    dune_files = List.map dune_files ~f:(fun f -> f.path);
    unreadable = List.rev !unreadable;
    spawns = List.rev !spawns;
    refused = List.rev !refused;
  }

let scan ~root ~generated ~live =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let t = scan_tree inventory in
  let unbounded =
    List.filter_map t.spawns ~f:(fun s ->
        match s.cost with
        | Error e ->
            Some
              (Printf.sprintf "%s: %s, in `%s`" (identity s.stanza) e (String.prefix s.command 200))
        | Ok _ -> None)
  in
  let over =
    List.filter_map t.spawns ~f:(fun s ->
        let limit, cap, whose = s.limit in
        match s.cost with
        | Ok n when n >= limit ->
            Some
              (Printf.sprintf
                 "%s: a spawned command line of up to %d characters, at or over this scan's limit \
                  of %d (%s refuses past %d): `%s`"
                 (identity s.stanza) n limit whose cap (String.prefix s.command 200))
        | _ -> None)
  in
  let refusals = t.refused @ unbounded in
  List.iter t.unreadable ~f:(eprintf "%s\n");
  List.iter refusals ~f:(eprintf "%s\n");
  if not (List.is_empty refusals) then
    eprintf
      "A dependency set belongs in a response file: `(with-stdout-to <x>.filelist (echo \
       \"%%{deps}\"))`, then `(run <exe> @<x>.filelist)`, read through \
       `Test_scan.Scan_argv.expand` (gh-ocannl-1720).\n";
  List.iter over ~f:(eprintf "%s\n");
  (* The floors keep a scan that stopped finding dune files or spawns from passing; a synthetic
     control tree is far smaller than the repository, so they apply to the live run only. *)
  let file_floor = if live then 20 else 1 and spawn_floor = if live then 200 else 1 in
  Verdict.p_empty ~min:file_floor
    "every dune file in the source tree reads the same to sexplib and to dune" ~over:t.dune_files
    t.unreadable;
  Verdict.p_empty ~min:spawn_floor
    "every spawned command line has a bounded expansion and every stanza is one this scan places"
    ~over:t.spawns refusals;
  Verdict.p_all ~min:spawn_floor "every bounded spawned command line stays under this scan's limit"
    t.spawns ~f:(fun s ->
      let limit, _, _ = s.limit in
      match s.cost with Ok n -> n < limit | Error _ -> true);
  if live then (
    let stale = List.filter judged_reads ~f:(fun (p, _, _) -> not (Hash_set.mem reads_used p)) in
    List.iter stale ~f:(fun (p, _, _) ->
        eprintf "judged_reads: no spawned command line reads `%s` any more; delete its row\n" p);
    Verdict.p_empty "every judged `%{read:...}` file is still read on a spawned command line"
      ~over:judged_reads stale;
    let bounded =
      List.filter_map t.spawns ~f:(fun s -> Result.ok s.cost |> Option.map ~f:(fun n -> (n, s)))
      |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare b a)
    in
    eprintf "%d dune files, %d spawned command lines (not part of the golden)\n"
      (List.length t.dune_files) (List.length t.spawns);
    List.iter (List.take bounded 5) ~f:(fun (n, s) ->
        eprintf "  up to %6d characters: %s (not part of the golden)\n" n (identity s.stanza)))

(* Synthetic controls. *)

let write_file path data =
  let rec mkdirs dir =
    if not (String.equal dir Stdlib.Filename.current_dir_name || Stdlib.Sys.file_exists dir) then (
      mkdirs (Stdlib.Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdirs (Stdlib.Filename.dirname path);
  Out_channel.write_all path ~data

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let control () =
  let exe = Fresh_process.executable () in
  let fixture = Stdlib.Filename.temp_dir "command line control " "" in
  let run name files =
    let root = Stdlib.Filename.concat fixture name in
    List.iter files ~f:(fun (path, data) -> write_file (Stdlib.Filename.concat root path) data);
    Fresh_process.run ~exe [ "--scan-only"; root ]
  in
  let report label child = Fresh_process.report ~label child in
  let passed label child =
    let ok = Fresh_process.matches ~exit:0 ~contains:[] child in
    if not ok then report label child;
    ok
  in
  let refused label ~messages child =
    let ok = Fresh_process.matches ~exit:1 ~contains:messages child in
    if not ok then report label child;
    ok
  in
  let project = ("dune-project", "(lang dune 3.20)\n") in
  let rule action = Printf.sprintf "(rule (target out) (deps a.ml) (action %s))\n" action in
  let legitimate =
    run "legitimate"
      [
        project;
        ( "a/dune",
          {|(rule
 (target listed.actual)
 (deps x.ml y.ml (:runner p.exe) (alias other) (env_var HOME) (universe) (sandbox always))
 (action (with-stdout-to %{target} (run %{runner} %{workspace_root} %{deps}))))
(rule
 (target files.actual)
 (deps (glob_files_rec ../*.ml) (source_tree ..))
 (action
  (progn
   (with-stdout-to files.filelist (echo "%{deps}"))
   (with-stdout-to %{target} (run %{dep:p.exe} @files.filelist)))))
(test (name t) (deps (glob_files *.ml)))
(tests (names u v) (action (run %{test} --flag)))
(rule
 (alias reads)
 (action
  (chdir %{workspace_root}
   (run %{bin:tool} %{lib:base:base.cmi} "--x=%{read:../test/config/ocannl_backend.txt}"))))
(subdir nested (rule (with-stdout-to short.txt (bash "echo short"))))
(rule (alias sys) (action (system "echo fine")))
|}
        );
      ]
  in
  let glob_on_command_line =
    run "glob on the command line"
      [
        project;
        ( "a/dune",
          "(rule (target out) (deps (glob_files_rec ../*.ml)) (action (with-stdout-to %{target} \
           (run %{dep:p.exe} %{deps}))))\n" );
      ]
  in
  let named_glob =
    run "named glob"
      [
        project;
        ( "a/dune",
          "(rule (target out) (deps (:srcs (glob_files *.ml))) (action (run %{dep:p.exe} %{srcs})))\n"
        );
      ]
  in
  let long_name i = Printf.sprintf "dependency_%03d_%s.ml" i (String.make 90 'x') in
  let over_long_listed =
    run "over-long listed deps"
      [
        project;
        ( "a/dune",
          Printf.sprintf "(rule (target out) (deps %s) (action (run %%{dep:p.exe} %%{deps})))\n"
            (String.concat ~sep:" " (List.init 300 ~f:long_name)) );
      ]
  in
  let over_long_shell =
    run "over-long shell line"
      [ project; ("a/dune", rule (Printf.sprintf "(bash \"echo %s\")" (String.make limit 'y'))) ]
  in
  let over_long_system =
    run "over-long system line"
      [
        project; ("a/dune", rule (Printf.sprintf "(system \"echo %s\")" (String.make cmd_limit 'z')));
      ]
  in
  let unbounded_pform =
    run "unbounded pform" [ project; ("a/dune", rule "(run %{dep:p.exe} %{env:HOME=})") ]
  in
  let unjudged_read =
    run "unjudged read" [ project; ("a/dune", rule "(run %{dep:p.exe} %{read:notes.txt})") ]
  in
  let unplaced_head = run "unplaced head" [ project; ("a/dune", rule "(frobnicate %{deps})") ] in
  let unresolved_chdir =
    run "unresolved chdir" [ project; ("a/dune", rule "(chdir %{env:D=.} (run %{dep:p.exe}))") ]
  in
  let included = run "included" [ project; ("a/dune", "(include dune.inc)\n") ] in
  let unreadable =
    run "unreadable" [ project; ("a/dune", "(rule #; (hidden) (action (run p)))\n") ]
  in
  printf
    "Synthetic controls run the shipping scanner over a passing tree and over each refused shape;\n\
     refusal output is captured and matched below.\n\n";
  Verdict.p
    "listed deps, a response file, default and named test actions, a judged read under chdir, a \
     short-form rule in a subdir and a short system line pass"
    (passed "legitimate" legitimate);
  Verdict.p "gh-ocannl-1720's shape, a recursive glob in %{deps} on a command line, is refused"
    (refused "glob on the command line"
       ~messages:[ "a/dune:1 (rule out)"; "`(glob_files_rec ../*.ml)` expands to a dependency set" ]
       glob_on_command_line);
  Verdict.p "a named dependency bound to a glob is refused where the command line expands it"
    (refused "named glob"
       ~messages:[ "`(glob_files *.ml)` expands to a dependency set" ]
       named_glob);
  Verdict.p "listed deps whose expansion reaches the limit are refused with their bound"
    (refused "over-long listed deps"
       ~messages:[ "a/dune:1 (rule out): a spawned command line of up to" ]
       over_long_listed);
  Verdict.p "a shell line that reaches the limit is refused"
    (refused "over-long shell line" ~messages:[ "at or over this scan's limit" ] over_long_shell);
  Verdict.p
    "a system line under the process limit but past cmd.exe's is refused at cmd.exe's own limit"
    (refused "over-long system line"
       ~messages:
         [
           Printf.sprintf "at or over this scan's limit of %d (cmd.exe refuses past %d)" cmd_limit
             cmd_cap;
         ]
       over_long_system);
  Verdict.p "a pform with no bound is refused"
    (refused "unbounded pform"
       ~messages:[ "`%{env:HOME=}` is a pform this scan has no bound for" ]
       unbounded_pform);
  Verdict.p "a read of a file no judgment bounds is refused"
    (refused "unjudged read" ~messages:[ "a file `judged_reads` does not bound" ] unjudged_read);
  Verdict.p "an action head on neither list is refused"
    (refused "unplaced head" ~messages:[ "action head `frobnicate`" ] unplaced_head);
  Verdict.p "a chdir to a directory the scan cannot resolve is refused"
    (refused "unresolved chdir"
       ~messages:[ "moves to a directory this scan cannot resolve" ]
       unresolved_chdir);
  Verdict.p "an include stanza is refused"
    (refused "included"
       ~messages:[ "includes stanzas from a file this scan does not read" ]
       included);
  Verdict.p "a dune file sexplib would read with a hole in it is refused"
    (refused "unreadable" ~messages:[ "a/dune: " ] unreadable);
  Test_scan.Refusal_control_manifest.print "command_line_length_scan.ml";
  remove_tree fixture

let usage () =
  eprintf "Usage: %s <declared-source root> <generated exclusions...> | --scan-only <root>\n"
    Stdlib.Sys.argv.(0);
  Stdlib.exit 2

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--scan-only"; root ] -> scan ~root ~generated:[] ~live:false
  | _ :: root :: generated when not (String.is_prefix root ~prefix:"--") ->
      eprintf
        "Limit: %d characters per spawned command line, Windows refuses past %d (not part of the \
         golden)\n"
        limit windows_cap;
      scan ~root ~generated ~live:true;
      printf "\n";
      control ()
  | _ -> usage ()
