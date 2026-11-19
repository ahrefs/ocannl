(** The reader behind [test/operations/sweep_owner_scan] (gh-ocannl-926): every [`Outside_sweep]
    skip names the matrix that owns its execution ({!Verdict.sweep_owner}), and that matrix still
    exists.

    An [`Outside_sweep] skip leaves both sweep intersections, so a claim recorded that way is
    covered only by whatever runs it elsewhere. Before the owner was an argument, that relationship
    lived in comments: deleting the compiler-trace CI step would have left [cc_backend_trace_name]'s
    level-3 claim excluded from the backend and the environment dimension alike, with nothing
    turning red.

    {1 What it reads}

    - {b The corpus}: every [.ml] file of the source tree whose text spells [Outside_sweep], parsed
      with ppxlib. A use is an EXPRESSION [`Outside_sweep <arg>] (a pattern or a type is not a use).
      Its argument must be a literal owner: a constructor whose last path component is [Ci_step] or
      [Plain_run] (the type checker has already made it {!Verdict.sweep_owner}'s) applied to a
      record of string constants, every field of the constructor present. Anything else -- a
      variable, a function call, a computed string, the tag with no argument -- is refused. A file
      in {!exemptions} spells fixture owners, not the tree's: its uses are neither required to be
      literal nor resolved, and an exemption whose file has no use is refused as stale.
    - {b An owner's alias} must lie in the use's own directory (only a stanza there can compile the
      source) and be defined by exactly one stanza of that directory's [dune] file: a
      [(test)]/[(tests)] stanza through its per-test alias [runtest-<name>], or a [(rule)] through
      its [(alias …)]/[(aliases …)]. That stanza must RUN the use's source.
    - {b Running the source} means a command in command position ([Dune_stanza_scan.commands_in]): a
      test stanza compiling it (its [(modules …)] list the file's module, or a [(select …)] names
      the file as an arm) runs itself, through its default action or its own [(run %{test} …)]; any
      stanza's [(run %{dep:X.exe} …)] or [(run %{exe:X.exe} …)] runs [X] when [X] is an executable
      or test stanza of the file compiling it. A module a rule of the file copies the source into
      ([(copy# F X.ml)], [(copy F X.ml)]) is the same source. A stanza that only names an executable
      -- copies it, depends on it -- runs nothing.
    - {b A [Ci_step] owner}: [.github/workflows/<workflow>] holds exactly one list item whose
      [name:] (on the item's dash line or at the item's own key indentation, quotes stripped) is the
      step, and one of that item's lines, a trailing YAML or shell comment cut off, spells
      [dune build] and after it [@<alias>] followed by a character that cannot continue an alias;
      and the item's own [env:] block sets one of the [(env_var …)] dependencies the owning stanza
      declares -- the gate that makes the step's run differ from the sweep's. A stanza declaring
      none is refused: nothing the scan reads would tell the two runs apart.
    - {b A [Plain_run] owner}: none of the owning stanza's runs of the source passes the atom
      [forcing], and some stanza of the same [dune] file runs the source with it -- the forced run
      whose skips the owner answers for.

    {1 What it deliberately does not read}

    - Whether a workflow step runs at all: its [if:], its job's matrix and triggers are not
      evaluated, and a [dune build @<alias>] inside an [echo] or a quoted string reads as a build.
      The gate's VALUE is not read (that it opens the gate is the test's), nor a job- or
      workflow-level [env:], nor a variable set inside the [run:] script: a gate set those ways is
      refused, the loud direction. A build spelled any other way -- through [tools/test-run.sh], or
      with the alias on a continuation line -- is refused, the loud direction.
    - Whether the owning run evaluates the same CLAIMS: the claim identity differs between the two
      runs anyway (a forced copy has its own executable name), and labels are the test's to keep
      equal. The scan pins that the owner exists and compiles the same source, not what it prints.
    - Owners spelled outside the source tree's [.ml] files, and [`Outside_sweep] reached through a
      ppx or a functor: none exists, and the tag must be spelled for the type to be inhabited. *)

open Base
open Ppxlib

(** Files whose owners are fixtures rather than the tree's, each with the reason. *)
let exemptions =
  [
    ( "test/operations/verdict_skip_probe.ml",
      "replays fixture records for sweep_harness.sh: its owners are the fixture's, read back \
       through Verdict.sweep_owner_of_field, not the tree's" );
    ( "test/operations/verdict_quantified.ml",
      "its children hand Verdict owners the record grammar cannot carry, to pin that the skip \
       fails the run and writes no record" );
  ]

let tag = "Outside_sweep"

type use = { file : string; line : int; owner : (Verdict.sweep_owner, string) Result.t }

let parse ~file source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf file;
  Parse.implementation lexbuf

let last_component (lid : longident) =
  match lid with Lident name | Ldot (_, name) -> Some name | Lapply _ -> None

let string_constant (e : expression) =
  match e.pexp_desc with Pexp_constant (Pconst_string (s, _, _)) -> Some s | _ -> None

(* The literal owner [arg] spells, or why it is not one. *)
let literal_owner (arg : expression option) : (Verdict.sweep_owner, string) Result.t =
  match arg with
  | None -> Error "carries no owner"
  | Some { pexp_desc = Pexp_construct ({ txt; _ }, payload); _ } -> (
      let fields =
        match payload with
        | Some { pexp_desc = Pexp_record (fields, None); _ } ->
            Some
              (List.map fields ~f:(fun ({ txt; _ }, value) ->
                   (last_component txt, string_constant value)))
        | _ -> None
      in
      let get fields name =
        match
          List.filter fields ~f:(fun (field, _) -> Option.equal String.equal field (Some name))
        with
        | [ (_, Some value) ] -> Ok value
        | [ (_, None) ] -> Error (Printf.sprintf "field %s is not a string literal" name)
        | _ -> Error (Printf.sprintf "field %s is not given exactly once" name)
      in
      match (last_component txt, fields) with
      | Some "Ci_step", Some fields -> (
          match (get fields "workflow", get fields "step", get fields "alias") with
          | Ok workflow, Ok step, Ok alias -> Ok (Verdict.Ci_step { workflow; step; alias })
          | Error e, _, _ | _, Error e, _ | _, _, Error e -> Error ("its Ci_step " ^ e))
      | Some "Plain_run", Some fields -> (
          match (get fields "alias", get fields "forcing") with
          | Ok alias, Ok forcing -> Ok (Verdict.Plain_run { alias; forcing })
          | Error e, _ | _, Error e -> Error ("its Plain_run " ^ e))
      | _ -> Error "carries an owner that is not a literal Ci_step or Plain_run record")
  | Some _ -> Error "carries an owner that is not a literal Ci_step or Plain_run record"

(** Every expression [`Outside_sweep …] of one source. Raises on a source that does not parse. *)
let uses ~file source =
  if not (String.is_substring source ~substring:tag) then []
  else
    let found = ref [] in
    let iter =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_variant (label, arg) when String.equal label tag ->
              found :=
                { file; line = e.pexp_loc.loc_start.pos_lnum; owner = literal_owner arg } :: !found
          | _ -> ());
          super#expression e
      end
    in
    iter#structure (parse ~file source);
    List.rev !found

(* ---- Resolution against the dune file ---- *)

module Stanza = Dune_stanza_scan

let stanza_kind = function Sexp.List (Sexp.Atom kind :: _) -> Some kind | _ -> None

(* [file] (a basename) and every module a rule of [stanzas] copies it into ([(copy# F X.ml)], [(copy
   F X.ml)]): the forced missing-toolkit arm runs such a copy. *)
let copies_of stanzas ~file =
  file
  :: List.concat_map stanzas ~f:(fun stanza ->
      match (stanza_kind stanza, Stanza.field stanza "action") with
      | ( Some "rule",
          Some [ Sexp.List [ Sexp.Atom ("copy#" | "copy"); Sexp.Atom src; Sexp.Atom dst ] ] )
        when String.equal src file ->
          [ dst ]
      | _ -> [])

(* Whether [stanza] compiles one of [sources] (basenames): its modules list one's module, or a
   select of its libraries names one as an arm. *)
let compiles stanza ~sources =
  let modules = List.map sources ~f:(String.chop_suffix_if_exists ~suffix:".ml") in
  let listed =
    match Stanza.field stanza "modules" with
    | Some args ->
        List.exists args ~f:(function
          | Sexp.Atom m -> List.mem modules (String.uncapitalize m) ~equal:String.equal
          | Sexp.List _ -> false)
    | None -> false
  in
  listed
  || List.exists (Stanza.select_arms stanza) ~f:(fun (_, arm) ->
      List.mem sources arm ~equal:String.equal)

let builds_executables stanza =
  match stanza_kind stanza with
  | Some ("executable" | "executables" | "test" | "tests") -> true
  | _ -> false

(* The executable a command names in command position: [%{dep:X.exe}] or [%{exe:X.exe}] as the whole
   atom. *)
let command_executable cmd =
  match Stanza.pieces cmd with
  | [ Stanza.Pform pform ] -> (
      match String.lsplit2 pform ~on:':' with
      | Some (("dep" | "exe"), path) when String.is_suffix path ~suffix:".exe" ->
          Some (String.chop_suffix_exn path ~suffix:".exe")
      | _ -> None)
  | _ -> None

(* The argument lists with which [stanza] RUNS an executable compiling one of [sources]: a test
   stanza compiling them runs itself, through its default action (no arguments) or each [(run
   %{test} …)] of its own; any stanza's [(run %{dep:X.exe} …)] or [(run %{exe:X.exe} …)] in command
   position runs [X] when [X] is an executable or test stanza of [stanzas] compiling them. A rule
   that only names the executable -- copies it, depends on it -- runs nothing. *)
let runs_of stanzas stanza ~sources =
  let action = Stanza.field stanza "action" in
  let programs =
    List.concat_map (Option.value action ~default:[]) ~f:(fun sexp -> Stanza.commands_in sexp)
    |> List.filter_map ~f:(function
      | _, _, Stanza.Program (cmd, args) -> Some (cmd, args)
      | _ -> None)
  in
  let self =
    (match stanza_kind stanza with Some ("test" | "tests") -> true | _ -> false)
    && compiles stanza ~sources
  in
  let compiled_executable exe =
    List.exists stanzas ~f:(fun other ->
        builds_executables other
        && List.mem (Stanza.names_of other) exe ~equal:String.equal
        && compiles other ~sources)
  in
  (if self && Option.is_none action then [ [] ] else [])
  @ List.filter_map programs ~f:(fun (cmd, args) ->
      if self && String.equal cmd "%{test}" then Some args
      else
        match command_executable cmd with
        | Some exe when compiled_executable exe -> Some args
        | _ -> None)

let defines_alias stanza alias =
  match stanza_kind stanza with
  | Some ("test" | "tests") ->
      List.exists (Stanza.names_of stanza) ~f:(fun name -> String.equal ("runtest-" ^ name) alias)
  | Some "rule" -> List.mem (Stanza.aliases_of stanza) alias ~equal:String.equal
  | _ -> false

type owning = {
  stanzas : Sexp.t list;  (** the use's directory's dune file *)
  sources : string list;  (** the skipping source and the copies made of it *)
  runs : string list list;  (** the argument lists the owning stanza runs the source with *)
  env_vars : string list;  (** the [(env_var …)] dependencies the owning stanza declares *)
}

(* The variables a stanza's [(deps …)] declares through [(env_var NAME)]. *)
let env_vars_of stanza =
  Option.value (Stanza.field stanza "deps") ~default:[]
  |> List.filter_map ~f:(function
    | Sexp.List [ Sexp.Atom "env_var"; Sexp.Atom name ] -> Some name
    | _ -> None)

(** The stanza owning [alias] for the use at [file], when it runs [file]. *)
let owning_stanza ~read ~file alias_path =
  let dir = Stdlib.Filename.dirname file and base = Stdlib.Filename.basename file in
  let alias_dir = Stdlib.Filename.dirname alias_path
  and alias = Stdlib.Filename.basename alias_path in
  let dune = dir ^ "/dune" in
  if not (String.equal alias_dir dir) then
    Error
      (Printf.sprintf
         "owner alias @%s lies outside %s, the directory of the skipping source, where no stanza \
          can compile it"
         alias_path dir)
  else
    match read dune with
    | None -> Error (Printf.sprintf "%s is not among the files read" dune)
    | Some content -> (
        let stanzas = Stanza.stanzas content in
        let sources = copies_of stanzas ~file:base in
        match List.filter stanzas ~f:(fun stanza -> defines_alias stanza alias) with
        | [] -> Error (Printf.sprintf "no stanza of %s defines @%s" dune alias_path)
        | _ :: _ :: _ as several ->
            Error
              (Printf.sprintf "%d stanzas of %s define @%s; the owner names one run"
                 (List.length several) dune alias_path)
        | [ stanza ] -> (
            match runs_of stanzas stanza ~sources with
            | [] ->
                Error
                  (Printf.sprintf
                     "the stanza of %s defining @%s does not run an executable compiling %s" dune
                     alias_path base)
            | runs -> Ok { stanzas; sources; runs; env_vars = env_vars_of stanza }))

(* ---- Resolution against the workflow ---- *)

let indentation line = String.length line - String.length (String.lstrip line)

let unquote value =
  let value = String.strip value in
  let n = String.length value in
  if
    n >= 2
    && Char.equal value.[0] value.[n - 1]
    && (Char.equal value.[0] '"' || Char.equal value.[0] '\'')
  then String.sub value ~pos:1 ~len:(n - 2)
  else value

let is_comment line = String.is_prefix (String.lstrip line) ~prefix:"#"

(* [line] without a trailing comment: YAML's and the shell's alike start at a [#] after whitespace.
   A [#] inside a quoted string is cut too, which can only hide a mention -- the refusing
   direction. *)
let uncommented line =
  if is_comment line then ""
  else
    match String.substr_index line ~pattern:" #" with
    | Some at -> String.prefix line at
    | None -> line

(* Every YAML list item of [lines] as its (start line, lines): the dash line and the lines indented
   past the dash, blank ones included. *)
let list_items lines =
  let lines = Array.of_list lines in
  let n = Array.length lines in
  List.filter_map (List.range 0 n) ~f:(fun i ->
      let line = lines.(i) in
      if String.is_prefix (String.lstrip line) ~prefix:"- " && not (is_comment line) then (
        let dash = indentation line in
        let j = ref (i + 1) in
        while
          !j < n && (String.is_empty (String.strip lines.(!j)) || indentation lines.(!j) > dash)
        do
          Int.incr j
        done;
        Some (i + 1, dash, Array.to_list (Array.sub lines ~pos:i ~len:(!j - i))))
      else None)

let item_name ~dash item =
  let key_column = dash + 2 in
  List.find_mapi item ~f:(fun k line ->
      let text =
        if k = 0 then String.drop_prefix line (dash + 2)
        else if indentation line = key_column then String.lstrip line
        else ""
      in
      Option.map (String.chop_prefix text ~prefix:"name:") ~f:unquote)

let alias_char c = Char.is_alphanum c || List.mem [ '_'; '-'; '.'; '/' ] c ~equal:Char.equal

(* Whether [line] builds [@alias]: the token, ended by a character that cannot continue an alias, is
   preceded on the same line by a [dune build] invocation. *)
let builds_alias line alias =
  let pattern = "@" ^ alias in
  match String.substr_index line ~pattern:"dune build" with
  | None -> false
  | Some build ->
      List.exists (String.substr_index_all line ~may_overlap:false ~pattern) ~f:(fun at ->
          let after = at + String.length pattern in
          at > build && (after >= String.length line || not (alias_char line.[after])))

(* The variables an item's own [env:] block sets: the key lines indented past the item's [env:] key,
   which sits at the item's key column. *)
let item_env ~dash item =
  let key_column = dash + 2 in
  let rec collect inside = function
    | [] -> []
    | line :: rest ->
        let text = String.lstrip line in
        let column = indentation line in
        if String.is_empty (String.strip line) || is_comment line then collect inside rest
        else if column = key_column || (column = dash && String.is_prefix text ~prefix:"- ") then
          let key = if column = dash then String.drop_prefix text 2 else text in
          collect (String.equal (String.strip key) "env:") rest
        else if inside && column > key_column then
          match String.lsplit2 text ~on:':' with
          | Some (name, _) -> String.strip name :: collect inside rest
          | None -> collect inside rest
        else collect inside rest
  in
  collect false item

let ci_step_builds ~read ~workflow ~step ~alias ~env_vars =
  let path = ".github/workflows/" ^ workflow in
  if String.mem workflow '/' then
    Error (Printf.sprintf "workflow %S is not a file directly under .github/workflows" workflow)
  else
    match read path with
    | None -> Error (Printf.sprintf "%s is not among the files read" path)
    | Some content -> (
        let items =
          List.filter
            (list_items (String.split_lines content))
            ~f:(fun (_, dash, item) -> Option.equal String.equal (item_name ~dash item) (Some step))
        in
        match items with
        | [] -> Error (Printf.sprintf "%s has no step named %S" path step)
        | _ :: _ :: _ -> Error (Printf.sprintf "%s has several steps named %S" path step)
        | [ (start, dash, item) ] -> (
            let set = item_env ~dash item in
            let gates = List.filter env_vars ~f:(List.mem set ~equal:String.equal) in
            if not (List.exists item ~f:(fun line -> builds_alias (uncommented line) alias)) then
              Error
                (Printf.sprintf "step %S of %s (line %d) no longer builds @%s" step path start alias)
            else
              match (env_vars, gates) with
              | [], _ ->
                  Error
                    (Printf.sprintf
                       "the stanza of @%s declares no (env_var ...) dependency, so no setting of \
                        step %S of %s tells its run from the sweep's"
                       alias step path)
              | _, [] ->
                  Error
                    (Printf.sprintf
                       "step %S of %s (line %d) builds @%s but its env: sets none of %s, the \
                        variables the alias's stanza declares, so it runs what the sweep runs"
                       step path start alias
                       (String.concat ~sep:", " env_vars))
              | _, gates ->
                  Ok
                    (Printf.sprintf "step %S of %s builds @%s with %s set" step path alias
                       (String.concat ~sep:", " gates))))

(** The resolution of one literal owner of a use at [file]. *)
let resolve ~read ~file (owner : Verdict.sweep_owner) =
  match Verdict.sweep_owner_problem owner with
  | Some why -> Error ("the record grammar cannot carry it: " ^ why)
  | None -> (
      match owner with
      | Verdict.Ci_step { workflow; step; alias } ->
          Result.bind (owning_stanza ~read ~file alias) ~f:(fun { env_vars; _ } ->
              ci_step_builds ~read ~workflow ~step ~alias ~env_vars)
      | Verdict.Plain_run { alias; forcing } ->
          Result.bind (owning_stanza ~read ~file alias) ~f:(fun { stanzas; sources; runs; _ } ->
              let forced args = List.mem args forcing ~equal:String.equal in
              if List.exists runs ~f:forced then
                Error (Printf.sprintf "@%s itself passes the forcing argument `%s`" alias forcing)
              else if
                not
                  (List.exists stanzas ~f:(fun stanza ->
                       List.exists (runs_of stanzas stanza ~sources) ~f:forced))
              then
                Error
                  (Printf.sprintf
                     "no stanza beside @%s runs the source with `%s`, so the forced skip this \
                      owner answers for never runs"
                     alias forcing)
              else Ok (Printf.sprintf "@%s runs the source without `%s`" alias forcing)))

type report = {
  resolved : string list;  (** [<file>: <owner field> -- <how it resolved>], one per use. *)
  findings : string list;  (** Each refusal, as [<file>:<line>: <message>]. *)
}

(** The whole check over [uses] (every use of the corpus); [read] looks up a repository file.
    [?exemptions] is for the synthetic trees of the cases test. *)
let check ?(exemptions = exemptions) ~read uses =
  let exempt file = List.Assoc.mem exemptions file ~equal:String.equal in
  let per_use =
    List.map uses ~f:(fun use ->
        let where = Printf.sprintf "%s:%d" use.file use.line in
        match use.owner with
        | _ when exempt use.file -> `Exempt use.file
        | Error why -> `Finding (Printf.sprintf "%s: an `Outside_sweep skip %s" where why)
        | Ok owner -> (
            let field = Verdict.sweep_owner_field owner in
            match resolve ~read ~file:use.file owner with
            | Ok how -> `Resolved (Printf.sprintf "%s: %s -- %s" use.file field how)
            | Error why ->
                `Finding
                  (Printf.sprintf "%s: the outside-sweep owner %s does not resolve: %s" where field
                     why)))
  in
  let stale =
    List.filter_map exemptions ~f:(fun (file, _) ->
        if List.exists per_use ~f:(function `Exempt f -> String.equal f file | _ -> false) then None
        else
          Some
            (Printf.sprintf
               "%s: exempted as spelling fixture owners, but it spells no outside-sweep owner" file))
  in
  {
    resolved = List.filter_map per_use ~f:(function `Resolved line -> Some line | _ -> None);
    findings =
      List.filter_map per_use ~f:(function `Finding line -> Some line | _ -> None) @ stale;
  }
