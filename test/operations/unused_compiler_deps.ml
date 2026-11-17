(* gh-ocannl-2118: no stanza declares a compiler-reaching library it never uses.

   A library whose dependency closure holds [compiler-libs.common] forces the compiler archive's
   units into every executable that declares it, used or not (gh-ocannl-2010 found 1.4 GiB of build
   per checkout that nobody had asked for). [test_utils_linkage] guards one direction: the cheap
   runtime helpers it names never reach the compiler. This guards the other: a stanza that declares
   [test_scan], [ppxlib], [compiler-libs.*], [einsum_parser], ... must reference it from its own
   compiled modules.

   {1 Sources of truth}

   DECLARATIONS: each stanza's own [(libraries ...)] field, read from the dune files the rule hands
   over, through [Test_scan.Dune_stanza_scan]. Not [dune describe]: it omits [(test)] and [(tests)]
   stanzas, and its [requires] adds the ppx runtime libraries the stanza never wrote ([ppxlib.ast],
   from metaquot, is itself compiler-reaching).

   IDENTITY AND REACH: [dune describe workspace --format csexp], run by this program against the
   source root ([DUNE_SOURCEROOT]) with a private [--build-dir] under a per-process temporary root,
   so it never shares the running build's lock. It is dune's own resolution of public and private
   names and of external findlib libraries with their [requires]; an external library only a
   [(test)] stanza declares is absent there, and is resolved the same way through a throwaway probe
   project. A library is COMPILER-REACHING when its [requires] closure holds [compiler-libs.common],
   the library that owns [ocamlcommon] -- the definition of the property, not a list: every scanner
   library falls out of the walk.

   OWNERSHIP: a library with its own include directory owns its [.cmi] units. For compiler-reaching
   libraries sharing an include directory, dune-resolved [ocamlfind query -predicates byte] supplies
   their own absolute archive paths and dune-resolved [ocamlobjinfo] supplies [Unit name:]
   definitions. An imported interface never credits a unit defined in another archive. For an
   interface-only unit (a shared-directory [.cmi] with no archive definition), the candidates are
   archives whose [Interfaces imported:] sections name it. A candidate depending on another
   candidate cannot own it: exactly one remaining candidate must exist. Ambiguous or absent
   candidates refuse when that unit is referenced; imports outside the library's include directories
   contribute nothing. Imported entries have two tab-separated fields (digest and unit); malformed
   entries refuse. This preserves type-only units such as [Parsetree] without crediting its users.
   The interface-only heuristic can misattribute a unit if its defining library never imports it but
   a consumer does, producing a false unused-dependency finding. Archive-less findlib packages own
   no units. A failed query, unreadable archive or output without definitions refuses without
   directory fallback. Shared local libraries refuse: findlib cannot resolve their archives and
   [@check] does not build them. Non-compiler-reaching declarations stay outside this check's scope.

   USE: the typed tree of every [.cmt]/[.cmti] in the stanza's byte object directory
   ([Cmt_references]): post-preprocessing, so generated modules, a [select]-chosen source and a
   ppx-produced reference count; resolved, so a wrapped or opened name reads as its library's unit,
   and a unit only another library's interface loads does not. The rule depends on the workspace's
   [@check], which writes those files.

   {1 Recognised forms (closed)}

   Stanzas: [library], [executable], [executables], [test], [tests], also inside [(subdir ...)]. An
   [include] or [dynamic_include] stanza is refused (its stanzas live in a file not read), as is any
   other stanza carrying a [libraries] field.

   [libraries] elements: a library name; [(re_export <name>)], exempt by form (the export serves the
   dependants, which this check reads in their own right); [(select <target> from (<libs> -> <file>)
   ... (-> <file>))], whose first arm dune can satisfy contributes its positive libraries. Anything
   else, and any [%{...}] in the field, is refused.

   Object directories: [<dir>/.<name>.objs/byte] for a library, [<dir>/.<first name>.eobjs/byte] for
   the executables kinds. A stanza without one is inactive in this build context only when it
   carries [enabled_if] or [optional]; otherwise it is refused, as is a directory holding no [.cmt],
   an unreadable [.cmt], a declared name nothing resolves, malformed describe output, and a describe
   entry or field outside [build_context], [root], [library], [executables] and the library fields
   read here.

   It does not judge libraries that are not compiler-reaching (an unused [stdio] is outside its
   scope), or stanzas inactive in this build context (another backend's, an optional library's). A
   compiler-reaching library declared only for a side effect -- a deriver linked so it registers --
   takes an exact row in [exemptions], held earned on every run; a local [(kind ppx_rewriter)] or
   [(kind ppx_deriver)] library is exempt by form, since a rewriter in [libraries] is there to
   register with the driver.

   Every refusal path runs in [controls] below, against synthetic input; the fixtures in
   [unused_compiler_deps_fixtures/] are its controls over compiled code: each fixture stanza's
   expected finding is pinned in [fixture_expectations], and that table must name exactly the
   fixture directory's stanzas. *)

open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Dune_stanza_scan

let source = "unused_compiler_deps.ml"
let fixture_dir = "test/operations/unused_compiler_deps_fixtures"

(* Each fixture stanza and the libraries the check must report for it, by declared spelling. *)
let fixture_expectations =
  [
    ("ucd_flag_unused_test_scan", [ "test_scan" ]);
    ("ucd_flag_ppxlib_beside_test_scan", [ "ppxlib" ]);
    ("ucd_flag_select_arm_unused", [ "test_scan" ]);
    ("ucd_flag_ppxlib_metaquot_only", [ "ppxlib" ]);
    ("ucd_pass_wrapped_name", []);
    ("ucd_pass_public_unwrapped", []);
    ("ucd_pass_toplevel", []);
    ("ucd_pass_interface_unit", []);
    ("ucd_flag_interface_unit_toplevel", [ "compiler-libs.toplevel" ]);
    ("ucd_pass_interface_only", []);
    ("ucd_pass_generated_module", []);
    ("ucd_pass_ppx_generated", []);
    ("ucd_pass_select_arm_used", []);
    ("ucd_pass_re_export", []);
    ("ucd_pass_rewriter_registration", []);
    ("ucd_gap_shared_directory", [ "compiler-libs.toplevel" ]);
  ]

let compiler_root = "compiler-libs.common"

(* Declarations linked for a side effect nothing names: (directory, stanza, library, reason). Each
   row must still be earned -- the stanza declares the library and would otherwise be reported -- or
   the run fails, so a row cannot outlive its reason. *)
let exemptions =
  [
    ( "test/operations",
      "dead_export_scan_cases",
      "ppx_hash",
      "an external deriver linked so its archive registers the [@@deriving hash] expander the \
       fixture corpus is expanded with" );
  ]

(* Every refusal lands here, and one claim holds the list empty; [controls] proves each path records
   one. *)
let refusals = ref []
let refuse fmt = Printf.ksprintf (fun msg -> refusals := msg :: !refusals) fmt

exception Malformed of string

let malformed fmt = Printf.ksprintf (fun msg -> raise (Malformed msg)) fmt

(* {1 Canonical s-expressions} -- dune's machine format: [<length>:<bytes>] atoms and parentheses,
   no escapes, so there is nothing to approximate. *)

let parse_csexp text =
  let len = String.length text in
  let rec items pos acc =
    if pos >= len then (List.rev acc, pos)
    else
      match text.[pos] with
      | ')' -> (List.rev acc, pos)
      | _ ->
          let item, pos = one pos in
          items pos (item :: acc)
  and one pos =
    match text.[pos] with
    | '(' ->
        let inner, pos = items (pos + 1) [] in
        if pos >= len || not (Char.equal text.[pos] ')') then malformed "csexp: an unclosed list";
        (Sexp.List inner, pos + 1)
    | c when Char.is_digit c ->
        let colon =
          match String.index_from text pos ':' with
          | Some i when String.for_all (String.sub text ~pos ~len:(i - pos)) ~f:Char.is_digit -> i
          | _ -> malformed "csexp: an atom length without its ':'"
        in
        let n = Int.of_string (String.sub text ~pos ~len:(colon - pos)) in
        if colon + 1 + n > len then malformed "csexp: an atom runs past the end";
        (Sexp.Atom (String.sub text ~pos:(colon + 1) ~len:n), colon + 1 + n)
    | c -> malformed "csexp: unexpected byte %C at %d" c pos
  in
  let all, pos = items 0 [] in
  if pos <> len then malformed "csexp: an unbalanced ')'";
  all

(* {1 The workspace description} *)

type library = {
  lib_name : string;
  uid : string;
  local : bool;
  requires : string list;
  include_dirs : string list;
}

(* Dune can give concurrent actions the same TMPDIR. Each invocation owns its nested builds and tool
   output. *)
let temp_root = lazy (Stdlib.Filename.temp_dir "unused-compiler-deps-" "")
let temp name = Stdlib.Filename.concat (Lazy.force temp_root) name

(* [dune describe workspace] of the project rooted at [root], in a build directory of its own. *)
let run_describe ~dune ~root ~tag =
  let build_dir = temp ("ucd-" ^ tag ^ "-build") in
  let out = temp ("ucd-" ^ tag ^ ".csexp") in
  (* The nested dune must not believe it is an action of the running one. *)
  let env =
    Unix.environment ()
    |> Array.filter ~f:(fun kv ->
        not
          (List.exists [ "INSIDE_DUNE="; "DUNE_ACTION_TRACE_DIR=" ] ~f:(fun prefix ->
               String.is_prefix kv ~prefix)))
  in
  let fd = Unix.openfile out [ O_WRONLY; O_CREAT; O_TRUNC ] 0o644 in
  let devnull = Unix.openfile Stdlib.Filename.null [ O_WRONLY ] 0 in
  let pid =
    Unix.create_process_env dune
      [|
        dune;
        "describe";
        "workspace";
        "--lang";
        "0.1";
        "--format";
        "csexp";
        "--root";
        root;
        "--build-dir";
        build_dir;
      |]
      env Unix.stdin fd devnull
  in
  Unix.close fd;
  Unix.close devnull;
  let _, status = Unix.waitpid [] pid in
  match status with
  | WEXITED 0 -> Ok (In_channel.read_all out, build_dir)
  | _ -> Error (Printf.sprintf "dune describe workspace of %s did not exit 0" tag)

(* The workspace description lists external libraries only where a library or executable it
   describes requires them; a library only a [(test)] stanza declares is absent. Those are resolved
   by dune too: a throwaway project with one [optional] library per name, described the same way. An
   unavailable name drops out there as it would in the build. *)
let probe_project ~tag dune_text =
  let root = temp ("ucd-" ^ tag) in
  Unix.mkdir root 0o755;
  Out_channel.write_all (root ^ "/dune-project") ~data:"(lang dune 3.20)\n";
  Out_channel.write_all (root ^ "/dune") ~data:dune_text;
  root

let probe_name i = Printf.sprintf "ucd_probe_%d" i

let probe_stanzas names =
  List.mapi names ~f:(fun i name ->
      Printf.sprintf "(library (name %s) (optional) (modules) (libraries %s))\n" (probe_name i) name)
  |> String.concat

let library_fields = [ "name"; "uid"; "local"; "requires"; "source_dir"; "modules"; "include_dirs" ]

let read_library ~map_path fields =
  let get name =
    List.find_map fields ~f:(function
      | Sexp.List [ Sexp.Atom f; v ] when String.equal f name -> Some v
      | _ -> None)
  in
  List.iter fields ~f:(function
    | Sexp.List [ Sexp.Atom f; _ ] when List.mem library_fields f ~equal:String.equal -> ()
    | form ->
        refuse "describe: a library field outside the recognised forms: %s" (Sexp.to_string form));
  let atom name =
    match get name with
    | Some (Sexp.Atom a) -> a
    | _ -> malformed "describe: a library without an atom %s" name
  in
  let atoms name =
    match get name with
    | Some (Sexp.List l) ->
        List.map l ~f:(function
          | Sexp.Atom a -> a
          | _ -> malformed "describe: a library's %s holds a list" name)
    | _ -> malformed "describe: a library without a %s list" name
  in
  {
    lib_name = atom "name";
    uid = atom "uid";
    local = String.equal (atom "local") "true";
    requires = atoms "requires";
    include_dirs = List.map (atoms "include_dirs") ~f:map_path;
  }

(* The libraries of one description, local paths moved from the private build directory to the
   running build's. Malformed output raises [Malformed]; an unknown entry or field is refused. *)
let read_description ~workspace_root ~build_dir text =
  let prefix = build_dir ^ "/default/" in
  let map_path p =
    match String.chop_prefix p ~prefix with
    | Some rel -> workspace_root ^ "/" ^ rel
    | None ->
        if String.is_prefix p ~prefix:build_dir then
          malformed "describe: a path in a context other than default: %s" p;
        p
  in
  match parse_csexp text with
  | [ Sexp.List entries ] ->
      List.filter_map entries ~f:(function
        | Sexp.List [ Sexp.Atom "library"; Sexp.List fields ] ->
            Some (read_library ~map_path fields)
        | Sexp.List (Sexp.Atom ("build_context" | "root" | "executables") :: _) -> None
        | form ->
            refuse "describe: an entry outside the recognised forms: %s"
              (String.prefix (Sexp.to_string form) 120);
            None)
  | _ -> malformed "describe: not one list of entries"

(* {1 Ownership and reach} *)

let cmi_units dir =
  match Stdlib.Sys.readdir dir with
  | exception Sys_error _ -> None
  | files ->
      Some
        (Array.to_list files
        |> List.filter_map ~f:(fun f ->
            Option.map (String.chop_suffix f ~suffix:".cmi") ~f:String.capitalize))

(* These tools are action arguments resolved by dune, not shell commands found on PATH. *)
let capture ~stderr tool args =
  let out = temp "ucd-tool-output" in
  try
    let fd = Unix.openfile out [ O_WRONLY; O_CREAT; O_TRUNC ] 0o644 in
    let pid =
      Exn.protect
        ~finally:(fun () -> Unix.close fd)
        ~f:(fun () -> Unix.create_process tool (Array.of_list (tool :: args)) Unix.stdin fd stderr)
    in
    let _, status = Unix.waitpid [] pid in
    match status with WEXITED 0 -> Ok (In_channel.read_all out) | _ -> Error "tool did not exit 0"
  with Unix.Unix_error (e, _, _) -> Error (Unix.error_message e)

type archive_info = { definitions : string list; interfaces : string list }

let empty_archive = { definitions = []; interfaces = [] }

let read_archive ~archive text =
  let lines = String.split_lines text in
  let definitions =
    List.filter_map lines ~f:(fun line -> String.chop_prefix line ~prefix:"Unit name: ")
  in
  if List.is_empty definitions || List.exists definitions ~f:String.is_empty then (
    refuse "archive %s: no recognised unit definitions" archive;
    empty_archive)
  else
    let _, interfaces =
      List.fold lines ~init:(false, []) ~f:(fun (reading, units) line ->
          if String.equal line "Interfaces imported:" then (true, units)
          else if reading && String.is_prefix line ~prefix:"\t" then (
            match String.split line ~on:'\t' |> List.filter ~f:(Fn.non String.is_empty) with
            | [ digest; unit ]
              when String.length digest = 32
                   && (String.equal digest (String.make 32 '-')
                      || String.for_all digest ~f:(String.mem "0123456789abcdef")) ->
                (true, unit :: units)
            | _ ->
                refuse "archive %s: malformed interface import" archive;
                (true, units))
          else (false, units))
    in
    { definitions; interfaces }

let archive_units ?(stderr = Unix.stderr) ~objinfo archive =
  match capture ~stderr objinfo [ archive ] with
  | Error e ->
      refuse "archive %s: %s" archive e;
      empty_archive
  | Ok text -> read_archive ~archive text

let shared_units ?(stderr = Unix.stderr) ~findlib ~objinfo l =
  if l.local then (
    refuse "shared include directory: cannot resolve local archives for %s" l.lib_name;
    empty_archive)
  else
    match
      capture ~stderr findlib [ "query"; "-predicates"; "byte"; "-format"; "%+a"; l.lib_name ]
    with
    | Error e ->
        refuse "archive query for %s: %s" l.lib_name e;
        empty_archive
    | Ok text ->
        String.split_lines text
        |> List.filter ~f:(fun line -> not (String.is_empty line))
        |> List.fold ~init:empty_archive ~f:(fun acc path ->
            let info = archive_units ~stderr ~objinfo path in
            {
              definitions = info.definitions @ acc.definitions;
              interfaces = info.interfaces @ acc.interfaces;
            })

(* Interface imports also occur in consumers. Only a unique dependency-minimal archive can own an
   interface-only unit. No such archive, or incomparable candidates, is a refusal, not shared
   credit. *)
let interface_owner ~depends_on ~name unit candidates =
  let candidates = List.dedup_and_sort candidates ~compare:String.compare in
  let minimal =
    List.filter candidates ~f:(fun uid ->
        not
          (List.exists candidates ~f:(fun other ->
               (not (String.equal uid other)) && depends_on uid other)))
  in
  match minimal with
  | [ uid ] -> Some uid
  | _ ->
      refuse "interface-only unit %s: no unique archive owner among [%s]" unit
        (String.concat ~sep:", " (List.map candidates ~f:name));
      None

let depends_on by_uid uid target =
  let rec walk seen = function
    | [] -> false
    | u :: rest when Set.mem seen u -> walk seen rest
    | u :: rest ->
        if String.equal u target then true
        else
          let deps =
            Option.value_map (Hashtbl.find by_uid u) ~default:[] ~f:(fun l -> l.requires)
          in
          walk (Set.add seen u) (deps @ rest)
  in
  walk (Set.empty (module String)) [ uid ]

(* A shortest path from [uid] to the compiler root through [requires], if there is one. *)
let reach_path by_uid uid =
  let name u = match Hashtbl.find by_uid u with Some l -> l.lib_name | None -> u in
  let rec bfs seen = function
    | [] -> None
    | (u, path) :: rest ->
        if String.equal (name u) compiler_root then Some (List.rev path)
        else
          let next =
            match Hashtbl.find by_uid u with
            | Some l ->
                List.filter l.requires ~f:(fun r -> not (Set.mem seen r))
                |> List.dedup_and_sort ~compare:String.compare
            | None -> []
          in
          let seen = List.fold next ~init:seen ~f:Set.add in
          bfs seen (rest @ List.map next ~f:(fun r -> (r, name r :: path)))
  in
  bfs (Set.singleton (module String) uid) [ (uid, [ name uid ]) ]

(* {1 Stanzas} *)

type kind = Library | Executables

type stanza = {
  dir : string;
  kind : kind;
  name : string;  (** the library name, or the first executable name *)
  form : Sexp.t;
}

let declaring_heads = [ "library"; "executable"; "executables"; "test"; "tests" ]

let rec collect_stanzas ~dir forms =
  List.concat_map forms ~f:(fun form ->
      match form with
      | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body) ->
          collect_stanzas ~dir:(Scan.in_subdir dir sub) body
      | Sexp.List (Sexp.Atom head :: _) when List.mem declaring_heads head ~equal:String.equal -> (
          let kind = if String.equal head "library" then Library else Executables in
          match Scan.names_of form with
          | name :: _ -> [ { dir; kind; name; form } ]
          | [] ->
              refuse "%s/dune: a %s stanza without a name" dir head;
              [])
      | Sexp.List (Sexp.Atom (("include" | "dynamic_include") as head) :: _) ->
          (* The stanzas behind it live in a file this check does not read. *)
          refuse "%s/dune: an %s stanza, whose stanzas this check cannot read" dir head;
          []
      | Sexp.List (Sexp.Atom head :: _) ->
          if Option.is_some (Scan.field form "libraries") then
            refuse "%s/dune: a %s stanza carries a libraries field" dir head;
          []
      | _ -> [])

let object_dir ~workspace_root s =
  let leaf =
    match s.kind with Library -> "." ^ s.name ^ ".objs" | Executables -> "." ^ s.name ^ ".eobjs"
  in
  String.concat ~sep:"/" [ workspace_root; s.dir; leaf; "byte" ]

let conditional s =
  Option.is_some (Scan.field s.form "enabled_if") || Option.is_some (Scan.field s.form "optional")

let public_names s =
  match Scan.field s.form "public_name" with Some [ Sexp.Atom public ] -> [ public ] | _ -> []

type declared = Plain of string | Re_exported of string

(* The stanza's declarations, with [select] resolved: the first arm whose positive libraries all
   resolve and whose [!]-negated ones do not, as dune picks it. *)
let declarations ~available s =
  let where = Printf.sprintf "%s/dune (%s)" s.dir s.name in
  Option.value (Scan.field s.form "libraries") ~default:[]
  |> List.concat_map ~f:(function
    | Sexp.Atom a when String.is_substring a ~substring:"%{" ->
        refuse "%s: a pform in libraries: %s" where a;
        []
    | Sexp.Atom a -> [ Plain a ]
    | Sexp.List [ Sexp.Atom "re_export"; Sexp.Atom a ] -> [ Re_exported a ]
    | Sexp.List (Sexp.Atom "select" :: Sexp.Atom _ :: Sexp.Atom "from" :: arms) as form -> (
        let parsed =
          List.map arms ~f:(function
            | Sexp.List terms -> (
                let condition, rest =
                  List.split_while terms ~f:(fun t -> not (Sexp.equal t (Sexp.Atom "->")))
                in
                match rest with
                | [ Sexp.Atom "->"; Sexp.Atom _ ]
                  when List.for_all condition ~f:(function Sexp.Atom _ -> true | _ -> false) ->
                    Some (List.concat_map condition ~f:Scan.atoms)
                | _ -> None)
            | _ -> None)
        in
        if List.exists parsed ~f:Option.is_none then (
          refuse "%s: a select arm outside the recognised forms: %s" where (Sexp.to_string form);
          [])
        else
          let holds condition =
            List.for_all condition ~f:(fun l ->
                match String.chop_prefix l ~prefix:"!" with
                | Some negated -> not (available negated)
                | None -> available l)
          in
          match List.find (List.filter_opt parsed) ~f:holds with
          | Some condition ->
              List.filter_map condition ~f:(fun l ->
                  if String.is_prefix l ~prefix:"!" then None else Some (Plain l))
          | None ->
              refuse "%s: no select arm holds" where;
              [])
    | form ->
        refuse "%s: a libraries element outside the recognised forms: %s" where
          (Sexp.to_string form);
        [])

let referenced_units dir =
  match Stdlib.Sys.readdir dir with
  | exception Sys_error _ -> None
  | files ->
      let annots =
        Array.to_list files
        |> List.filter ~f:(fun f ->
            String.is_suffix f ~suffix:".cmt" || String.is_suffix f ~suffix:".cmti")
        |> List.sort ~compare:String.compare
      in
      if List.is_empty annots then Some (Error "holds no .cmt or .cmti")
      else
        Some
          (List.fold_result annots
             ~init:(Set.empty (module String))
             ~f:(fun acc f ->
               match Cmt_references.referenced_units (dir ^ "/" ^ f) with
               | Ok units -> Ok (Set.union acc (Set.of_list (module String) units))
               | Error e -> Error e))

type finding = { stanza : stanza; library : string; path : string list }

(* An exemption row is earned while the check would report its declaration without it. *)
let earned findings (dir, name, library, _reason) =
  List.exists findings ~f:(fun f ->
      String.equal f.stanza.dir dir && String.equal f.stanza.name name
      && String.equal f.library library)

(* What the check knows of the libraries, built from the descriptions and the dune files. *)
type world = {
  workspace_root : string;
  resolve : string -> library option;
  available : string -> bool;
  reach_of : string -> string list option;
  owners : string -> string list;
  rewriters : Set.M(String).t;
}

type verdict = Inactive | Unread | Checked of finding list

(* One stanza's verdict. Refusals are recorded through [refuse]. *)
let judge w s =
  let objdir = object_dir ~workspace_root:w.workspace_root s in
  let reaching =
    declarations ~available:w.available s
    |> List.filter_map ~f:(function
      | Re_exported _ -> None
      | Plain name when Set.mem w.rewriters name -> None
      | Plain name -> (
          match w.resolve name with
          | None ->
              if not (conditional s) then
                refuse "%s/dune (%s): the declared library %s resolves to nothing" s.dir s.name name;
              None
          | Some l -> Option.map (w.reach_of l.uid) ~f:(fun path -> (name, l.uid, path))))
  in
  match referenced_units objdir with
  | None ->
      if conditional s then Inactive
      else (
        refuse "%s/dune (%s): no object directory %s" s.dir s.name objdir;
        Unread)
  | Some (Error e) ->
      refuse "%s/dune (%s): %s" s.dir s.name e;
      Unread
  | Some (Ok units) ->
      let referenced_owners =
        if List.is_empty reaching then Set.empty (module String)
        else Set.to_list units |> List.concat_map ~f:w.owners |> Set.of_list (module String)
      in
      Checked
        (List.filter_map reaching ~f:(fun (name, uid, path) ->
             let used = Set.mem referenced_owners uid in
             if used then None else Some { stanza = s; library = name; path }))

(* {1 Controls} -- each refusal path, run against synthetic input. *)

let refuses f =
  let saved = !refusals in
  refusals := [];
  let raised = match f () with () -> false | exception Malformed _ -> true in
  let recorded = match !refusals with [] -> false | _ :: _ -> true in
  refusals := saved;
  raised || recorded

let controls ~dune ~objinfo ~tool_stderr =
  let stanza_of text =
    match collect_stanzas ~dir:"synthetic" (Scan.stanzas text) with
    | [ s ] -> s
    | _ -> failwith "a control stanza must be one stanza"
  in
  let empty_dir = temp "ucd-control-empty" in
  let corrupt_dir = temp "ucd-control-corrupt" in
  List.iter [ empty_dir; corrupt_dir ] ~f:(fun d -> Unix.mkdir d 0o755);
  Out_channel.write_all (corrupt_dir ^ "/m.cmt") ~data:"not a cmt";
  p "an unreadable archive is refused"
    (refuses (fun () ->
         ignore
           (archive_units ~stderr:tool_stderr ~objinfo (empty_dir ^ "/missing.cma") : archive_info)));
  p "an archive tool producing no unit definitions is refused"
    (refuses (fun () ->
         ignore (archive_units ~stderr:tool_stderr ~objinfo:dune "--version" : archive_info)));
  p "malformed archive interface entries are refused"
    (refuses (fun () ->
         ignore
           (read_archive ~archive:"control"
              "Unit name: Defined\nInterfaces imported:\n\tbad\tTypes_only\n"
             : archive_info)));
  let shared =
    { lib_name = "missing"; uid = "missing"; local = false; requires = []; include_dirs = [] }
  in
  p "a failed archive query is refused"
    (refuses (fun () ->
         ignore (shared_units ~stderr:tool_stderr ~findlib:dune ~objinfo shared : archive_info)));
  p "a shared local library is refused"
    (refuses (fun () ->
         ignore (shared_units ~findlib:dune ~objinfo { shared with local = true } : archive_info)));
  p "an interface-only unit belongs to its unique dependency-minimal archive"
    (Option.equal String.equal
       (interface_owner
          ~depends_on:(fun a b -> String.equal a "consumer" && String.equal b "owner")
          ~name:Fn.id "Types_only" [ "consumer"; "owner" ])
       (Some "owner"));
  p_all "missing, incomparable and cyclic interface-only owners are refused"
    [ ([], false); ([ "a"; "b" ], false); ([ "a"; "b" ], true) ]
    ~f:(fun (candidates, cyclic) ->
      refuses (fun () ->
          ignore
            (interface_owner ~depends_on:(fun _ _ -> cyclic) ~name:Fn.id "Types_only" candidates
              : string option)));
  let w =
    {
      workspace_root = temp "ucd-control-root";
      resolve = (fun _ -> None);
      available = (fun _ -> false);
      reach_of = (fun _ -> None);
      owners = (fun _ -> []);
      rewriters = Set.empty (module String);
    }
  in
  let judged text () = ignore (judge w (stanza_of text) : verdict) in
  p_all
    "the csexp reader refuses an unclosed list, a bare length, a short atom, a stray byte and a \
     stray ')'" [ "(3:abc"; "12"; "(5:ab)"; "(x)"; "()3:abc)" ] ~f:(fun text ->
      refuses (fun () -> ignore (parse_csexp text : Sexp.t list)));
  p "a description that is not one list of entries is refused"
    (refuses (fun () ->
         ignore (read_description ~workspace_root:"." ~build_dir:"/b" "4:atom" : library list)));
  p "a description entry of an unknown head is refused"
    (refuses (fun () ->
         ignore (read_description ~workspace_root:"." ~build_dir:"/b" "((5:frobs))" : library list)));
  let library extra =
    Printf.sprintf
      "((7:library(%s(4:name1:x)(3:uid1:u)(5:local4:true)(8:requires())(12:include_dirs()))))" extra
  in
  p "the control library description itself is read"
    (not
       (refuses (fun () ->
            ignore
              (read_description ~workspace_root:"." ~build_dir:"/b" (library "") : library list))));
  p "a library field outside the recognised forms is refused"
    (refuses (fun () ->
         ignore
           (read_description ~workspace_root:"." ~build_dir:"/b" (library "(4:kind3:ppx)")
             : library list)));
  p "a library path in a context other than default is refused"
    (refuses (fun () ->
         ignore
           (read_description ~workspace_root:"." ~build_dir:"/b"
              "((7:library((4:name1:x)(3:uid1:u)(5:local4:true)(8:requires())(12:include_dirs(7:/b/cc/x)))))"
             : library list)));
  p "a library without its uid is refused"
    (refuses (fun () ->
         ignore
           (read_description ~workspace_root:"." ~build_dir:"/b"
              "((7:library((4:name1:x)(5:local4:true)(8:requires())(12:include_dirs()))))"
             : library list)));
  p "a library whose requires holds a list is refused"
    (refuses (fun () ->
         ignore
           (read_description ~workspace_root:"." ~build_dir:"/b"
              "((7:library((4:name1:x)(3:uid1:u)(5:local4:true)(8:requires((1:a)))(12:include_dirs()))))"
             : library list)));
  p "a library without its requires list is refused"
    (refuses (fun () ->
         ignore
           (read_description ~workspace_root:"." ~build_dir:"/b"
              "((7:library((4:name1:x)(3:uid1:u)(5:local4:true)(12:include_dirs()))))"
             : library list)));
  p "a stanza without a name is refused"
    (refuses (fun () ->
         ignore
           (collect_stanzas ~dir:"synthetic" (Scan.stanzas "(library (modules a))") : stanza list)));
  p "another stanza carrying a libraries field is refused"
    (refuses (fun () ->
         ignore
           (collect_stanzas ~dir:"synthetic" (Scan.stanzas "(frob (libraries a))") : stanza list)));
  p "an include stanza, hiding stanzas in a file not read, is refused"
    (refuses (fun () ->
         ignore (collect_stanzas ~dir:"synthetic" (Scan.stanzas "(include dune.inc)") : stanza list)));
  (let stanza = stanza_of "(library (name a) (libraries b))" in
   let row = ("synthetic", "a", "b", "a control row") in
   let finding library = { stanza; library; path = [ library ] } in
   p "an exemption row is earned while its declaration would be reported"
     (earned [ finding "b" ] row);
   p_none "an exemption row whose declaration is used, or gone, is no longer earned"
     [ []; [ finding "c" ] ]
     ~f:(fun findings -> earned findings row));
  p "a pform in libraries is refused" (refuses (judged "(library (name a) (libraries %{x}))"));
  p "a libraries element of another form is refused"
    (refuses (judged "(library (name a) (libraries (frob x)))"));
  p "a select arm outside the recognised forms is refused"
    (refuses (judged "(library (name a) (libraries (select a.ml from (x -> a.x.ml y))))"));
  p "a select with no arm that holds is refused"
    (refuses (judged "(library (name a) (libraries (select a.ml from (x -> a.x.ml))))"));
  p "a declared name nothing resolves is refused"
    (refuses (judged "(library (name a) (libraries nowhere))"));
  p "a stanza with no object directory and no enabled_if is refused"
    (refuses (judged "(library (name a))"));
  p "a stanza with no object directory under enabled_if is inactive, not refused"
    (not (refuses (judged "(library (name a) (enabled_if false))")));
  p "an object directory holding no typed tree is refused"
    (match referenced_units empty_dir with Some (Error _) -> true | _ -> false);
  p "an object directory holding an unreadable typed tree is refused"
    (match referenced_units corrupt_dir with Some (Error _) -> true | _ -> false);
  p "a description dune refuses to produce is refused"
    (Result.is_error
       (run_describe ~dune ~root:(probe_project ~tag:"control-bad" "(library") ~tag:"control-bad"))

let () =
  Test_utils.set_binary_stdout ();
  let argv = Array.to_list (Test_scan.Scan_argv.expand Stdlib.Sys.argv) in
  let workspace_root, dune, findlib, objinfo, deps =
    match argv with
    | _ :: root :: dune :: findlib :: objinfo :: deps -> (root, dune, findlib, objinfo, deps)
    | _ -> failwith "usage: unused_compiler_deps WORKSPACE_ROOT DUNE OCAMLFIND OCAMLOBJINFO @DEPS"
  in
  let source_root =
    match Stdlib.Sys.getenv_opt "DUNE_SOURCEROOT" with
    | Some r -> r
    | None -> failwith "DUNE_SOURCEROOT is unset: run this through its dune rule"
  in
  let (), control_stderr =
    Stderr_capture.capture ~temp_dir:(Lazy.force temp_root) (fun () ->
        Out_channel.with_file Stdlib.Filename.null ~f:(fun sink ->
            controls ~dune ~objinfo ~tool_stderr:(Unix.descr_of_out_channel sink)))
  in
  p "the synthetic tool calls leave action stderr empty" (String.is_empty control_stderr);
  let describe ~root ~tag =
    match run_describe ~dune ~root ~tag with
    | Error e ->
        refuse "%s" e;
        []
    | Ok (text, build_dir) -> (
        try read_description ~workspace_root ~build_dir text
        with Malformed m ->
          refuse "%s" m;
          [])
  in
  let libraries = describe ~root:source_root ~tag:"workspace" in
  let by_uid = Hashtbl.create (module String) in
  let by_name = Hashtbl.create (module String) in
  List.iter libraries ~f:(fun l ->
      Hashtbl.set by_uid ~key:l.uid ~data:l;
      Hashtbl.set by_name ~key:l.lib_name ~data:l);
  p "the workspace description names compiler-libs.common" (Hashtbl.mem by_name compiler_root);
  (* Dune files: every stanza. *)
  let base = Scan.base_dir workspace_root in
  let stanzas =
    List.filter deps ~f:(fun d -> String.equal (Stdlib.Filename.basename d) "dune")
    |> List.sort ~compare:String.compare
    |> List.concat_map ~f:(fun path ->
        let dir = Stdlib.Filename.dirname (Scan.repo_relative base path) in
        let dir = if String.equal dir "." then "" else dir in
        collect_stanzas ~dir (Scan.stanzas (In_channel.read_all path)))
  in
  let local_libraries =
    List.filter stanzas ~f:(fun s -> match s.kind with Library -> true | Executables -> false)
  in
  let public_of =
    List.concat_map local_libraries ~f:(fun s ->
        List.map (public_names s) ~f:(fun p -> (s.name, p)))
    |> Map.of_alist_reduce (module String) ~f:(fun a _ -> a)
  in
  let local_names =
    List.concat_map local_libraries ~f:(fun s -> s.name :: public_names s)
    |> Set.of_list (module String)
  in
  (* External names no description lists yet, resolved through the probe project. *)
  let unresolved =
    List.concat_map stanzas ~f:(fun s ->
        Option.value_map (Scan.field s.form "libraries") ~default:[]
          ~f:(List.concat_map ~f:Scan.atoms))
    |> List.filter_map ~f:(fun a ->
        let a = Option.value (String.chop_prefix a ~prefix:"!") ~default:a in
        if
          List.mem [ "select"; "from"; "->"; "re_export" ] a ~equal:String.equal
          || String.is_suffix a ~suffix:".ml" || String.is_suffix a ~suffix:".mli"
          || String.is_substring a ~substring:"%{"
          || Set.mem local_names a || Hashtbl.mem by_name a
        then None
        else Some a)
    |> List.dedup_and_sort ~compare:String.compare
  in
  if not (List.is_empty unresolved) then (
    let probed =
      describe ~root:(probe_project ~tag:"probe" (probe_stanzas unresolved)) ~tag:"probe"
    in
    List.iter probed ~f:(fun l ->
        if not l.local then (
          if not (Hashtbl.mem by_uid l.uid) then Hashtbl.set by_uid ~key:l.uid ~data:l;
          if not (Hashtbl.mem by_name l.lib_name) then Hashtbl.set by_name ~key:l.lib_name ~data:l));
    (* A findlib name dune resolves to a library of another name ([ctypes.foreign] is
       [ctypes-foreign]) is bound to what its probe library requires. *)
    List.iteri unresolved ~f:(fun i name ->
        if not (Hashtbl.mem by_name name) then
          match List.find probed ~f:(fun l -> String.equal l.lib_name (probe_name i)) with
          | None -> ()
          | Some probe -> (
              match List.filter_map probe.requires ~f:(Hashtbl.find by_uid) with
              | [ l ] -> Hashtbl.set by_name ~key:name ~data:l
              | _ -> refuse "probe: %s resolves to no single library" name)));
  let resolve name =
    match Hashtbl.find by_name name with
    | Some l -> Some l
    | None -> Option.bind (Map.find public_of name) ~f:(Hashtbl.find by_name)
  in
  let directories = Hashtbl.create (module String) in
  Hashtbl.iter by_uid ~f:(fun l ->
      List.iter l.include_dirs ~f:(fun d -> Hashtbl.add_multi directories ~key:d ~data:l.uid));
  let owners = Hashtbl.create (module String) in
  let interface_candidates = Hashtbl.create (module String) in
  let shared_cmis = ref (Set.empty (module String)) in
  Hashtbl.iter by_uid ~f:(fun l ->
      if Option.is_some (reach_path by_uid l.uid) then
        let shared =
          List.exists l.include_dirs ~f:(fun d ->
              List.exists (Hashtbl.find_multi directories d) ~f:(fun uid ->
                  not (String.equal uid l.uid)))
        in
        let cmis =
          List.concat_map l.include_dirs ~f:(fun d -> Option.value (cmi_units d) ~default:[])
          |> Set.of_list (module String)
        in
        let definitions =
          if shared then (
            shared_cmis := Set.union !shared_cmis cmis;
            let info = shared_units ~findlib ~objinfo l in
            List.iter info.interfaces ~f:(fun u ->
                if Set.mem cmis u then Hashtbl.add_multi interface_candidates ~key:u ~data:l.uid);
            info.definitions)
          else Set.to_list cmis
        in
        List.iter definitions ~f:(fun u -> Hashtbl.add_multi owners ~key:u ~data:l.uid));
  let interface_cache = Hashtbl.create (module String) in
  let owners_of unit =
    match Hashtbl.find_multi owners unit with
    | [] when Set.mem !shared_cmis unit ->
        Hashtbl.find_or_add interface_cache unit ~default:(fun () ->
            interface_owner ~depends_on:(depends_on by_uid)
              ~name:(fun uid -> (Hashtbl.find_exn by_uid uid).lib_name)
              unit
              (Hashtbl.find_multi interface_candidates unit))
        |> Option.to_list
    | owners -> owners
  in
  let reach = Hashtbl.create (module String) in
  let w =
    {
      workspace_root;
      resolve;
      available =
        (fun name ->
          match resolve name with
          | None -> false
          | Some l -> (not l.local) || List.exists l.include_dirs ~f:Stdlib.Sys.file_exists);
      reach_of =
        (fun uid -> Hashtbl.find_or_add reach uid ~default:(fun () -> reach_path by_uid uid));
      owners = owners_of;
      (* A rewriter in [libraries] is linked for its registration with the driver -- nothing names
         it, by design -- so a declared local one is exempt by form. *)
      rewriters =
        List.filter local_libraries ~f:(fun s ->
            match Scan.field s.form "kind" with
            | Some (Sexp.Atom ("ppx_rewriter" | "ppx_deriver") :: _)
            | Some (Sexp.List (Sexp.Atom ("ppx_rewriter" | "ppx_deriver") :: _) :: _) ->
                true
            | _ -> false)
        |> List.concat_map ~f:(fun s -> s.name :: public_names s)
        |> Set.of_list (module String);
    }
  in
  let verdicts = List.map stanzas ~f:(fun s -> (s, judge w s)) in
  let checked =
    List.filter_map verdicts ~f:(fun (s, v) -> match v with Checked _ -> Some s | _ -> None)
  in
  let findings =
    List.concat_map verdicts ~f:(fun (_, v) -> match v with Checked f -> f | _ -> [])
  in
  let inactive =
    List.count verdicts ~f:(fun (_, v) -> match v with Inactive -> true | _ -> false)
  in
  let in_fixtures s = String.equal s.dir fixture_dir in
  let real, planted = List.partition_tf findings ~f:(fun f -> not (in_fixtures f.stanza)) in
  let real_checked = List.filter checked ~f:(Fn.non in_fixtures) in
  let exempted f =
    List.exists exemptions ~f:(fun (dir, name, library, _) ->
        String.equal f.stanza.dir dir && String.equal f.stanza.name name
        && String.equal f.library library)
  in
  eprintf
    "unused_compiler_deps: %d stanzas checked, %d inactive in this build context (not part of the \
     golden)\n"
    (List.length checked) inactive;
  List.iter (List.rev !refusals) ~f:(fun r -> eprintf "refused: %s\n" r);
  let reported = List.filter real ~f:(Fn.non exempted) in
  List.iter reported ~f:(fun f ->
      eprintf "unused: %s/dune (%s) declares %s, reaching the compiler through %s\n" f.stanza.dir
        f.stanza.name f.library
        (String.concat ~sep:" -> " f.path));
  p_empty "every dune file and the workspace description lie within the recognised forms"
    ~over:stanzas (List.rev !refusals);
  p_all "every exemption row is still earned: its stanza declares the library and uses none of it"
    exemptions ~f:(earned real);
  p_empty "no stanza declares a compiler-reaching library its modules never reference"
    ~over:real_checked reported;
  (* The compiled controls. *)
  let fixture_stanzas =
    List.filter stanzas ~f:in_fixtures
    |> List.map ~f:(fun s -> s.name)
    |> List.sort ~compare:String.compare
  in
  p "the fixture table names exactly the fixture directory's stanzas"
    (List.equal String.equal fixture_stanzas
       (List.map fixture_expectations ~f:fst |> List.sort ~compare:String.compare));
  p_all "every fixture library the check must report is derived as compiler-reaching"
    (List.concat_map fixture_expectations ~f:snd |> List.dedup_and_sort ~compare:String.compare)
    ~f:(fun name -> Option.is_some (Option.bind (resolve name) ~f:(fun l -> w.reach_of l.uid)));
  p_all "every fixture stanza is read and reported for exactly its expected libraries"
    fixture_expectations ~f:(fun (name, expected) ->
      let got =
        List.filter_map planted ~f:(fun f ->
            if String.equal f.stanza.name name then Some f.library else None)
        |> List.sort ~compare:String.compare
      in
      let read = List.exists checked ~f:(fun s -> in_fixtures s && String.equal s.name name) in
      let holds =
        read && List.equal String.equal got (List.sort expected ~compare:String.compare)
      in
      printf "  %s: %s\n" name
        (if not read then "not read"
         else if List.is_empty got then "reported for nothing"
         else "reported for " ^ String.concat ~sep:", " got);
      holds);
  Test_scan.Refusal_control_manifest.print source
