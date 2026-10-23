(** Which aliases re-run a file that pins the text of generated code (gh-ocannl-1533).

    [codegen_text_inventory] says WHAT a codegen change must re-run; this says HOW, as the per-test
    aliases [dune build] (or [tools/test-run.sh run build]) takes. Knowing the file is not knowing
    the alias: the test stanza [runtest-top_down_prec] runs that test and diffs nothing but its own
    [.expected], while the kernel snapshots [top_down_prec.<ext>.expected] are compared only by the
    hand-written [runtest-top_down_prec-extension] rule beside it -- and which of the four snapshots
    that rule compares depends on the backend the build resolves. A codegen author who runs the test
    by its own name sees green and leaves every snapshot unchecked (staging#1326).

    So the aliases are read from the dune files rather than written down beside the inventory, by
    the same reader the other dune scans share ({!Dune_stanza_scan}):

    - A golden is re-run by the rule whose [(diff …)] (or [(diff? …)]) names it as the expected
      side, matched with each [%{…}] standing for the text the build substitutes there, and the
      substitution is reported -- [%{read:../config/ocannl_backend_extension.txt}] is [cu] for the
      CUDA snapshot. A [(test)]/[(tests)] stanza re-runs the [<name>.expected] beside it, under
      dune's own [runtest-<name>].
    - A source is re-run by whatever runs the stanza owning its module: a [(test)] stanza under
      [runtest-<name>], an [(executable)] under the alias of each rule that runs it -- or, for a
      rule that only writes targets, the alias of each rule that reads one -- and an inline-test
      library under its per-module runner rule (gh-ocannl-1037), or dune's [runtest-<library>] where
      it has none. A [(select)] arm is the module its target names.

    The reading stops at the dune files it is handed and at the shapes above. What it cannot place
    it answers with nothing, and the caller refuses an empty answer: a member nobody can re-run by
    name is a gap in the census, never a quiet omission from it. *)

open Base
module Dune = Dune_stanza_scan

type stanza = { dir : string; stanza : Sexp.t }
(** A stanza with the repository-relative directory it applies to: its dune file's, or the one a
    [(subdir …)] wrapper names. *)

(** Every stanza of the given dune files, each as [(directory, content)]. Raises as
    {!Dune_stanza_scan.stanzas} does on a file the two readers disagree about. *)
let stanzas_of files =
  List.concat_map files ~f:(fun (dir, content) ->
      Dune.walk dir (Dune.stanzas content) ~f:(fun dir stanza -> [ { dir; stanza } ]))

type route = { alias : string; substitutions : (string * string) list }
(** An alias, spelled as [dune build] takes it ([@dir/name]), with the pform substitutions under
    which it re-runs the member: empty unless the rule's golden path is spelled with a pform. *)

let render { alias; substitutions } =
  match substitutions with
  | [] -> alias
  | substitutions ->
      alias ^ " when "
      ^ String.concat ~sep:", "
          (List.map substitutions ~f:(fun (pform, value) -> "%{" ^ pform ^ "} is " ^ value))

let at dir alias = "@" ^ Dune.in_subdir dir alias
let normalize path = Dune.repo_relative [] path

(* [path] relative to [dir], when it lies beneath it. *)
let relative ~dir path =
  if String.is_empty dir then Some path else String.chop_prefix path ~prefix:(dir ^ "/")

(* Whether [name] is what [atom] names once dune substitutes its pforms. Each pform stands for a
   non-empty run of text without a [/] -- a substituted file NAME, which is every use a golden path
   makes of one -- and the bindings that made it match come back in order. *)
let match_atom atom name =
  let n = String.length name in
  let rec go pieces pos bindings =
    match pieces with
    | [] -> if pos = n then Some (List.rev bindings) else None
    | Dune.Literal text :: rest ->
        if String.is_substring_at name ~pos ~substring:text then
          go rest (pos + String.length text) bindings
        else None
    | Dune.Pform pform :: rest ->
        let rec extend len =
          if pos + len > n then None
          else
            let value = String.sub name ~pos ~len in
            if String.contains value '/' then None
            else
              match go rest (pos + len) ((pform, value) :: bindings) with
              | Some _ as found -> found
              | None -> extend (len + 1)
        in
        extend 1
  in
  go (Dune.pieces (Dune.program_path atom)) 0 []

let is_rule s = Option.equal String.equal (Dune.head s.stanza) (Some "rule")
let rule_aliases s = if is_rule s then Dune.aliases_of s.stanza else []

(* The expected side of every [(diff …)]/[(diff? …)] in a stanza's action. *)
let rec diffed = function
  | Sexp.List [ Sexp.Atom ("diff" | "diff?"); Sexp.Atom expected; _ ] -> [ expected ]
  | Sexp.List items -> List.concat_map items ~f:diffed
  | Sexp.Atom _ -> []

let routes_sorted routes =
  List.dedup_and_sort routes ~compare:(fun a b -> String.compare (render a) (render b))

(** The aliases that compare [path], a repository-relative golden, against what a test produced. *)
let for_golden stanzas ~path =
  List.concat_map stanzas ~f:(fun s ->
      match relative ~dir:s.dir path with
      | None -> []
      | Some rel -> (
          match Dune.head s.stanza with
          | Some ("test" | "tests") ->
              List.filter_map (Dune.names_of s.stanza) ~f:(fun name ->
                  if String.equal rel (name ^ ".expected") then
                    Some { alias = at s.dir ("runtest-" ^ name); substitutions = [] }
                  else None)
          | _ ->
              let action = Option.value (Dune.field s.stanza "action") ~default:[] in
              List.concat_map (List.concat_map action ~f:diffed) ~f:(fun expected ->
                  match match_atom expected rel with
                  | None -> []
                  | Some substitutions ->
                      List.map (rule_aliases s) ~f:(fun alias ->
                          { alias = at s.dir alias; substitutions }))))
  |> routes_sorted

(* What a stanza's [(modules …)] field says of its modules, uncapitalized: the names it lists, the
   names it excludes after a backslash, and whether it takes [:standard] -- every module of its
   directory no other stanza lists. No field at all is [:standard], as it is to dune. *)
let modules_of stanza =
  let rec read ~excluding (listed, excluded, standard) = function
    | [] -> (listed, excluded, standard)
    | Sexp.Atom "\\" :: rest -> read ~excluding:true (listed, excluded, standard) rest
    | Sexp.Atom ":standard" :: rest -> read ~excluding (listed, excluded, true) rest
    | Sexp.Atom name :: rest ->
        let name = String.uncapitalize name in
        read ~excluding
          (if excluding then (listed, name :: excluded, standard)
           else (name :: listed, excluded, standard))
          rest
    | Sexp.List inner :: rest ->
        read ~excluding (read ~excluding (listed, excluded, standard) inner) rest
  in
  match Dune.field stanza "modules" with
  | None -> ([], [], true)
  | Some args -> read ~excluding:false ([], [], false) args

(* The stanzas that own modules. *)
let owner s =
  match Dune.head s.stanza with
  | Some ("test" | "tests" | "executable" | "executables" | "library") -> true
  | _ -> false

(* Whether [s] owns [module_name] among the stanzas of its directory. *)
let owns stanzas s ~module_name =
  let listed, excluded, standard = modules_of s.stanza in
  let mem = List.mem ~equal:String.equal in
  mem listed module_name
  || standard
     && (not (mem excluded module_name))
     && not
          (List.exists stanzas ~f:(fun other ->
               (not (phys_equal other s))
               && String.equal other.dir s.dir && owner other
               &&
               let listed, _, _ = modules_of other.stanza in
               mem listed module_name))

(* The module a source file is: the target of a [(select)] it is an arm of, else its own name. *)
let module_of stanzas ~dir ~base =
  List.find_map stanzas ~f:(fun s ->
      if String.equal s.dir dir then
        List.find_map (Dune.select_arms s.stanza) ~f:(fun (target, arm) ->
            if String.equal arm base then String.chop_suffix target ~suffix:".ml" else None)
      else None)
  |> Option.value ~default:(String.chop_suffix_if_exists base ~suffix:".ml")

(* The files a rule writes. *)
let targets_of stanza =
  List.concat_map [ "target"; "targets" ] ~f:(fun f ->
      List.filter_map
        (Option.value (Dune.field stanza f) ~default:[])
        ~f:(function Sexp.Atom a when not (String.contains a '%') -> Some a | _ -> None))

(* Whether an atom of a stanza in [dir] names the repository-relative [file]: spelled bare, or as
   the path of a [%{dep:…}]. *)
let names_file ~dir ~file atom =
  let as_path p = String.equal (normalize (Dune.in_subdir dir (Dune.program_path p))) file in
  match Dune.pieces atom with
  | [ Dune.Literal p ] -> as_path p
  | [ Dune.Pform pform ] -> (
      match String.lsplit2 pform ~on:':' with Some ("dep", p) -> as_path p | _ -> false)
  | _ -> false

(* The aliases a rule's work reaches: its own, or -- for a rule that only writes targets -- those of
   every rule that reads one of them, followed until an alias is found. *)
let rec reached stanzas ~seen s =
  match rule_aliases s with
  | _ :: _ as aliases -> List.map aliases ~f:(at s.dir)
  | [] ->
      let files = List.map (targets_of s.stanza) ~f:(fun t -> normalize (Dune.in_subdir s.dir t)) in
      List.concat_map stanzas ~f:(fun reader ->
          if
            is_rule reader
            && (not (List.mem seen reader ~equal:phys_equal))
            && List.exists (Dune.atoms reader.stanza) ~f:(fun atom ->
                List.exists files ~f:(fun file -> names_file ~dir:reader.dir ~file atom))
          then reached stanzas ~seen:(reader :: seen) reader
          else [])

(* The aliases of every rule that runs the repository-relative executable [exe]. *)
let runners stanzas ~exe =
  List.concat_map stanzas ~f:(fun s ->
      if not (is_rule s) then []
      else
        let runs =
          List.exists (Dune.executables_run s.stanza) ~f:(fun (cwd, command) ->
              match command with
              | Dune.Runs p ->
                  String.equal (normalize (Dune.in_subdir s.dir (Dune.in_subdir cwd p))) exe
              | Dune.Runs_dependency p -> String.equal (normalize (Dune.in_subdir s.dir p)) exe
              | _ -> false)
        in
        if runs then reached stanzas ~seen:[ s ] s else [])

(* The rules that hand an inline-test runner [-only-test <module>.ml]. *)
let module_runners stanzas ~dir ~module_name =
  List.concat_map stanzas ~f:(fun s ->
      let rec adjacent = function
        | "-only-test" :: file :: _ when String.equal file (module_name ^ ".ml") -> true
        | _ :: rest -> adjacent rest
        | [] -> false
      in
      if String.equal s.dir dir && adjacent (Dune.atoms s.stanza) then
        List.map (rule_aliases s) ~f:(at s.dir)
      else [])

(** The aliases that run [path], a repository-relative test source. *)
let for_source stanzas ~path =
  let dir, base =
    match String.rsplit2 path ~on:'/' with Some (dir, base) -> (dir, base) | None -> ("", path)
  in
  let module_name = module_of stanzas ~dir ~base in
  List.concat_map stanzas ~f:(fun s ->
      if not (String.equal s.dir dir && owner s && owns stanzas s ~module_name) then []
      else
        let names = Dune.names_of s.stanza in
        let exes names = List.map names ~f:(fun n -> Dune.in_subdir dir (n ^ ".exe")) in
        let run_by names = List.concat_map (exes names) ~f:(fun exe -> runners stanzas ~exe) in
        match Dune.head s.stanza with
        | Some ("test" | "tests") ->
            let names =
              if List.mem names module_name ~equal:String.equal then [ module_name ] else names
            in
            List.map names ~f:(fun n -> at dir ("runtest-" ^ n)) @ run_by names
        | Some ("executable" | "executables") -> run_by names
        | Some "library" when Option.is_some (Dune.field s.stanza "inline_tests") -> (
            match module_runners stanzas ~dir ~module_name with
            | [] -> List.map names ~f:(fun n -> at dir ("runtest-" ^ n))
            | aliases -> aliases)
        | _ -> [])
  |> List.map ~f:(fun alias -> { alias; substitutions = [] })
  |> routes_sorted
