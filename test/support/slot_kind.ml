(** Which backends a dune invocation can run a stanza on whatever the configuration says, and
    whether it reaches a stanza that reads the configuration at all (gh-ocannl-1004, gh-ocannl-1066,
    gh-ocannl-1095).

    tools/batch-backends.sh resolves the backends a [tools/test-run.sh] batch can hold, and from
    them both the dune width the batch runs at and the fleet slot it takes ([--cpu] only when it
    holds no GPU). The resolved configuration answers that for every stanza that selects its backend
    from it -- the ones declaring [(env_var OCANNL_BACKEND)] -- but not for the ones that NAME
    theirs: a stanza carrying [; ocannl-backend: cuda -- …] goes at the CUDA backend by name, and
    its [select] arm follows the library's availability, not the environment. So
    [dune runtest test/operations] on a box with hipjit runs the hip-marked stanzas on the GPU while
    both configurations say cc. Those markers are complete by construction -- [env_var_deps] fails a
    stanza that runs an executable and carries neither the variable nor a marker (gh-ocannl-659) --
    so the stanzas a run can reach, and their markers, are the whole answer. Every backend a reached
    marker names is reported, a CPU one included: the width a batch runs at is the tightest any of
    its backends meets on the box, and on the fleet's rog-nv-linux a CPU batch has a width of its
    own.

    The configuration's own backend counts unless it is PROVEN unread (gh-ocannl-1095): nothing the
    batch builds declares [(env_var OCANNL_BACKEND)] or carries neither declaration (which
    [env_var_deps] fails, and which is read the way it would run: from the configuration), and
    everything it builds is modelled exactly (the section on what a batch builds, below). A run
    reaching only stanzas that name theirs -- [; ocannl-backend: none] ones included, as
    [@test/operations/scans] does -- then holds only what they name, whatever the configurations
    say. A stanza naming a backend twice is not read as naming none: its file is unreadable, which
    is [Unknown].

    What a run reaches is read from a CLOSED set of argv shapes, the ones the runner is used with;
    any other word is unmodelled, and answered as [Unknown] -- every backend. (Codex review rounds
    3-4 on PR #803 found a new corner of dune's CLI each round, so this stopped trying to model the
    CLI.)
    - [build] with alias targets only: [@dir/alias] builds [alias] in [dir] and every directory
      below it, [@@dir/alias] in [dir] alone, [dir] plain (no [.], [..], leading or trailing [/]).
      An alias reaches what its stanzas attach to it -- the [runtest-<name>] dune generates for a
      [(test)]/[(tests)] stanza and an inline-test library included -- closed under everything those
      build first (the section on what building a stanza builds first, below). No target at all is
      the default alias everywhere, taken as reaching every stanza.
    - [runtest]/[test] with plain directories runs [@runtest] under each, and with none under the
      root.
    - Options only from the listed harmless ones, and no [--].
    - [promote] and [clean] build nothing and reach nothing; any other subcommand is unmodelled
      ([install] builds what it installs). [exec] is answered before this is asked (its program may
      pick any backend), and so is a command-line backend flag.

    A dune file this cannot read is itself an [Unknown] answer, for the same reason. *)

open Base
module Scan = Dune_stanza_scan

let gpu_backends = [ "cuda"; "hip"; "metal" ]

(** What building a stanza builds first (gh-ocannl-1095): the aliases its dependency fields name,
    and the files it mentions, which the rules producing them must build. *)
type need =
  | Alias_need of { dir : string; alias : string; recursive : bool }
  | File_need of { in_dir : string option; base : string }
      (** a file: its basename, in the directory its path resolves to, or anywhere when that cannot
          be told *)
  | Glob_need of { under : (string * bool) option; pattern : string }
      (** a basename pattern, matched in one directory (or below it, when recursive), or anywhere
          when the directory cannot be resolved *)

type role =
  | Runs  (** the stanza itself: its action, run on the alias it sits on or for its targets *)
  | Compiles  (** what compiling a library, an executable or a test's program builds first *)

type stanza = {
  dir : string;  (** the directory dune applies it in, repository-relative, [""] for the root *)
  role : role;
  attached : string list;  (** the aliases it attaches to or defines *)
  sexp : Sexplib.Sexp.t;
  named : string list;  (** the backends its marker names, [none] left out *)
  reads_config : bool;  (** whether it selects its backend from the configuration *)
  unmarked : bool;  (** whether the marker contract found nothing it runs, so it declares nothing *)
  targets : string list;  (** the basenames (as patterns) of the files a rule produces *)
  source_like : bool;
      (** whether one of them is a file dune may build to compile or load anything *)
  needs : need list;
  inexact : string option;
      (** the first construct in it this does not model exactly, which makes a batch reaching it
          every backend *)
  foreign_config : string option;
      (** for a [copy_files], a configuration it copies from a directory the runner does not read:
          what a reader in its directory, or below, reads instead *)
  overrides : string option;
      (** a backend it sets for what it runs, past the configuration: a batch reaching it can hold
          any backend *)
}

(** The aliases a stanza sits on, including the per-stanza one dune generates for a test and an
    inline-test library, which {!Scan.aliases_of} leaves to its callers. *)
let attached_aliases sexp =
  let generated =
    match Scan.head sexp with
    | Some ("test" | "tests") -> List.map (Scan.names_of sexp) ~f:(fun n -> "runtest-" ^ n)
    | Some "library" when Option.is_some (Scan.field sexp "inline_tests") ->
        "runtest" :: List.map (Scan.names_of sexp) ~f:(fun n -> "runtest-" ^ n)
    | _ -> []
  in
  Scan.aliases_of sexp @ Option.to_list (Scan.alias_stanza_name sexp) @ generated

(** What a stanza's backend declaration says it holds: the backends its marker names, and whether it
    reads the configuration. Each rule [env_var_deps] fails is read the widening way -- a stanza
    both declaring and naming holds both, one declaring neither reads the configuration -- except a
    second marker, which raises like any other marker the contract refuses. *)
let backend_of_rule rule =
  let named { Scan.backend; _ } =
    String.split backend ~on:',' |> List.filter ~f:(fun b -> not (String.equal b "none"))
  in
  match rule with
  | Scan.Runs_nothing -> ([], false)
  | Scan.Declares_variable | Scan.Names_neither -> ([], true)
  | Scan.Names_backend (_, m) -> (named m, false)
  | Scan.Declares_and_names (_, m) -> (named m, true)
  | Scan.Names_twice _ -> failwith "a stanza carrying two backend markers"

let join dir sub = match (dir, sub) with "", s -> s | d, "" -> d | d, s -> d ^ "/" ^ s

(** {1 What a batch builds, and when that is proven}

    The configuration's backend is taken off a batch only on a PROOF that nothing the batch builds
    reads it. Two review rounds on PR #1027 (Codex GPT-6.1 Sol) each found dune shapes a closure
    over dune's dependency semantics missed, so the claim is the inverted one: the closure below is
    trusted only where every stanza in it is built from constructs it models EXACTLY. Any other
    construct makes the batch [Unknown] -- every backend: what it builds is then unread, the
    backends it names included (review round 5) -- and so does a backend set past the configuration.
    Within an exact closure, the named backends and the configuration readers are the whole answer.

    The closure, seeded by the argv's aliases:
    - an [(alias …)]/[(alias_rec …)] in a dependency field (a stanza's [deps], an inline-test
      library's [(inline_tests (deps …))]), resolved against the stanza's directory, through the
      same attached-alias inventory the seeds use, generated per-test aliases included;
    - a file: every atom anywhere in the stanza, and the payload of every path pform ([%{dep:…}],
      [%{read:…}], …), taken as a file it may need -- matched by BASENAME against every rule's
      targets in any directory, a pform elsewhere in the atom read as a wildcard, a [glob_files]
      pattern matched as a glob (a character class or an alternative matching everything). A copy
      keeps the basename, so this follows it to the original producer without modelling it;
    - compilation, for every batch that builds anything: every library, executable, test program,
      lexer, parser and [env] in the tree, its files followed the same way (a test's [deps] and
      [action] are its run, not its compilation), and every rule producing a source-like file. Which
      program links which library is not read, so all of them are.

    Exactly modelled, and nothing else: the stanza heads listed below (an [(include …)], a [cram]
    test or any other head is a hole in the inventory); in a dependency field, files, [(file …)],
    named bindings, [glob_files]/[glob_files_rec], [source_tree], [env_var], [universe], [sandbox]
    (each with atom arguments only), and an alias whose path stays in the tree and carries no pform;
    the pforms listed below ([%{env:…}] only inside an [enabled_if]); no [dynamic-run]; and no
    preprocessing action. A directory target is taken as producing every file. A program dune runs
    to preprocess ([pps]) is taken not to start a backend. *)

let basename p = match String.rsplit2 p ~on:'/' with Some (_, b) -> b | None -> p

(** [path] resolved against [dir], repository-relative; [None] when it leaves the tree, is absolute
    or carries a pform. *)
let resolve ~dir path =
  if String.is_substring path ~substring:"%{" || String.is_prefix path ~prefix:"/" then None
  else
    List.fold_result (String.split path ~on:'/')
      ~init:(List.rev (if String.is_empty dir then [] else String.split dir ~on:'/'))
      ~f:(fun acc c ->
        match (c, acc) with
        | ("" | "."), _ -> Ok acc
        | "..", [] -> Error ()
        | "..", _ :: up -> Ok up
        | c, _ -> Ok (c :: acc))
    |> Result.ok
    |> Option.map ~f:(fun parts -> String.concat ~sep:"/" (List.rev parts))

(* The pforms whose payload is a path dune builds before expanding it. *)
let path_pforms = [ "dep"; "exe"; "path"; "read"; "read-lines"; "read-strings" ]

(* The pforms with a payload that name no file of the batch's own, or one compilation builds. An
   [%{env:…}] can expand to any path, so it is exact only inside an [enabled_if], which adds no
   dependency (Codex review on PR #1027). *)
let valued_pforms = [ "bin"; "lib"; "lib-available"; "ocaml-config"; "version" ]

(* The variables that name no file of their own: the stanza's targets and deps, or the context. *)
let variable_pforms =
  [
    "target";
    "targets";
    "deps";
    "test";
    "workspace_root";
    "system";
    "ocaml";
    "ocamlc";
    "ocamlopt";
    "arch_sixtyfour";
    "ext_obj";
    "ext_exe";
    "ext_lib";
    "ext_dll";
    "context_name";
    "profile";
    "architecture";
    "os_type";
    "model";
    "ocaml_version";
    "ocaml_bin";
    "null";
    "cc";
    "cxx";
  ]

let is_wild c = Char.equal c '*' || Char.equal c '?'

(** An atom as a basename pattern: each pform a wildcard. [None] for one naming no file of its own
    -- a bare variable such as [%{deps}] or [%{target}], whose files the stanza names elsewhere. *)
let pattern_of atom =
  let b =
    basename
      (String.concat
         (List.map (Scan.pieces atom) ~f:(function Scan.Literal l -> l | Scan.Pform _ -> "*")))
  in
  if String.is_empty b || String.for_all b ~f:(Char.equal '*') then None else Some b

(** The files a stanza may need: every atom, and the payload of every path pform, read as a path. A
    path pform's payload is relative to the stanza's directory; a literal atom is too, unless the
    stanza runs anything under a [chdir], when it may be relative to anywhere (Codex review on PR
    #1027: a basename matched anywhere let a copy in one directory answer for a need in another). *)
let file_needs ~dir sexp =
  let chdir = List.mem (Scan.atoms sexp) "chdir" ~equal:String.equal in
  let placed ~relative path =
    let in_dir, base =
      match String.rsplit2 path ~on:'/' with Some (d, b) -> (d, b) | None -> ("", path)
    in
    ((if relative then resolve ~dir in_dir else None), base)
  in
  List.concat_map (Scan.atoms sexp) ~f:(fun atom ->
      let payloads =
        List.filter_map (Scan.pieces atom) ~f:(function
          | Scan.Pform p -> (
              match String.lsplit2 p ~on:':' with
              | Some (k, v) when List.mem path_pforms k ~equal:String.equal -> Some (v, true)
              | Some ("lib", v) ->
                  Some (Option.value_map (String.rsplit2 v ~on:':') ~f:snd ~default:v, false)
              | _ -> None)
          | Scan.Literal _ -> None)
      in
      List.filter_map ((atom, not chdir) :: payloads) ~f:(fun (a, relative) ->
          Option.map (pattern_of a) ~f:(fun b ->
              if String.exists b ~f:is_wild then Glob_need { under = None; pattern = b }
              else
                let in_dir, base = placed ~relative a in
                let in_dir = if String.equal base b then in_dir else None in
                File_need { in_dir; base = b })))

(** The first pform in [sexp] this does not model exactly: one outside the lists above, a named
    binding aside. *)
let inexact_pform ~bindings sexp =
  let rec atoms_with ~in_enabled_if = function
    | Sexp.Atom a -> [ (a, in_enabled_if) ]
    | Sexp.List (Sexp.Atom "enabled_if" :: rest) ->
        List.concat_map rest ~f:(atoms_with ~in_enabled_if:true)
    | Sexp.List l -> List.concat_map l ~f:(atoms_with ~in_enabled_if)
  in
  List.find_map (atoms_with ~in_enabled_if:false sexp) ~f:(fun (atom, in_enabled_if) ->
      List.find_map (Scan.pieces atom) ~f:(function
        | Scan.Literal _ -> None
        | Scan.Pform p ->
            let exact =
              match String.lsplit2 p ~on:':' with
              | Some ("env", _) -> in_enabled_if
              | Some (k, _) ->
                  List.mem path_pforms k ~equal:String.equal
                  || List.mem valued_pforms k ~equal:String.equal
              | None ->
                  List.mem variable_pforms p ~equal:String.equal
                  || List.mem bindings p ~equal:String.equal
            in
            if exact then None else Some (Printf.sprintf "pform %%{%s}" p)))

(* The fields written in dune's dependency specification, wherever they sit: a stanza's [deps], an
   inline-test library's [(inline_tests (deps …))], an executable's [link_deps], a preprocessor's
   [preprocessor_deps], a foreign stub's [extra_deps] (Codex review on PR #1027). *)

(** A stanza's dependency fields: [deps], and an inline-test library's [(inline_tests (deps …))]. *)
let dep_field_names = [ "deps"; "link_deps"; "preprocessor_deps"; "extra_deps" ]

let rec dep_fields = function
  | Sexp.Atom _ -> []
  | Sexp.List (Sexp.Atom h :: args) when List.mem dep_field_names h ~equal:String.equal -> [ args ]
  | Sexp.List l -> List.concat_map l ~f:dep_fields

(** The first [(alias …)]/[(alias_rec …)] form in [sexp] outside a dependency field -- one this
    cannot place, such as a [copy_files] attaching its copies to an alias -- other than a rule's own
    [(alias …)] field. *)
let stray_alias sexp =
  let rec inner = function
    | Sexp.Atom _ -> None
    | Sexp.List (Sexp.Atom h :: _) when List.mem dep_field_names h ~equal:String.equal -> None
    | Sexp.List (Sexp.Atom ("alias" | "alias_rec") :: _) as form ->
        Some (Printf.sprintf "alias form %s" (Sexp.to_string form))
    | Sexp.List l -> List.find_map l ~f:inner
  in
  match sexp with
  | Sexp.List (Sexp.Atom head :: fields) ->
      List.find_map fields ~f:(function
        (* A rule's own [(alias …)] is the alias it sits on. *)
        | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom _ ] when String.equal head "rule" -> None
        | field -> inner field)
  | other -> inner other

(** What a stanza's dependency fields need beyond its atoms -- the aliases they name and the globs
    they match -- with the names their bindings give, or the first form this does not model exactly.
*)
let dep_needs ~dir sexp =
  let rec item = function
    | Sexp.Atom _ -> Ok ([], [])
    | Sexp.List (Sexp.Atom h :: rest) when String.is_prefix h ~prefix:":" ->
        Result.map
          (Result.all (List.map rest ~f:item))
          ~f:(fun parts ->
            (List.concat_map parts ~f:fst, String.drop_prefix h 1 :: List.concat_map parts ~f:snd))
    | Sexp.List [ Sexp.Atom (("alias" | "alias_rec") as h); Sexp.Atom spec ] -> (
        match resolve ~dir spec with
        | Some p ->
            let dir, alias = Option.value (String.rsplit2 p ~on:'/') ~default:("", p) in
            Ok ([ Alias_need { dir; alias; recursive = String.equal h "alias_rec" } ], [])
        | None -> Error (Printf.sprintf "dependency (%s %s)" h spec))
    | Sexp.List [ Sexp.Atom (("glob_files" | "glob_files_rec") as h); Sexp.Atom spec ] ->
        (* In the directory it names (and below, for [glob_files_rec]); every pform a wildcard, and
           a bare [*] kept -- it matches every target there (Codex review on PR #1027). *)
        let in_dir, base =
          match String.rsplit2 spec ~on:'/' with Some (d, b) -> (d, b) | None -> ("", spec)
        in
        let pattern =
          String.concat
            (List.map (Scan.pieces base) ~f:(function Scan.Literal l -> l | Scan.Pform _ -> "*"))
        in
        let under =
          Option.map (resolve ~dir in_dir) ~f:(fun d -> (d, String.equal h "glob_files_rec"))
        in
        Ok
          ([ Glob_need { under; pattern = (if String.is_empty pattern then "*" else pattern) } ], [])
    (* Each in its one shape: atoms only, so a form wrapping another dependency -- a [sandbox]
       around an alias, say -- is not read as carrying none (Codex review on PR #1027). *)
    | Sexp.List (Sexp.Atom ("file" | "source_tree" | "env_var" | "universe" | "sandbox") :: args)
      when List.for_all args ~f:(function Sexp.Atom _ -> true | Sexp.List _ -> false) ->
        Ok ([], [])
    | other -> Error (Printf.sprintf "dependency %s" (Sexp.to_string other))
  in
  Result.map
    (Result.all (List.concat_map (dep_fields sexp) ~f:(List.map ~f:item)))
    ~f:(fun parts -> (List.concat_map parts ~f:fst, List.concat_map parts ~f:snd))

(** The basenames, as patterns, of the files a rule produces: its [(target …)]/[(targets …)], or,
    without them and without an alias, every atom it carries -- dune infers such a rule's targets
    from its action, and reading every atom is the wide side of that inference. *)
let targets_of sexp =
  match Scan.head sexp with
  | Some "rule" -> (
      match (Scan.field sexp "targets", Scan.field sexp "target") with
      | Some args, _ | None, Some args ->
          (* A directory target produces every path below it, which a basename cannot name: it is
             taken as producing anything (Codex review on PR #1027). *)
          List.concat_map args ~f:(function
            | Sexp.Atom a -> Option.to_list (pattern_of a)
            | Sexp.List _ -> [ "*" ])
      | None, None ->
          (* Inferred from the action, an alias beside it or not (Codex review on PR #1027): the
             file each writing form writes. *)
          let rec written = function
            | Sexp.List
                [
                  Sexp.Atom ("with-stdout-to" | "with-stderr-to" | "with-outputs-to" | "write-file");
                  Sexp.Atom file;
                  body;
                ] ->
                Option.to_list (pattern_of file) @ written body
            | Sexp.List
                [ Sexp.Atom ("copy" | "copy#" | "copy-and-add-line-directive"); _; Sexp.Atom file ]
              ->
                Option.to_list (pattern_of file)
            | Sexp.List l -> List.concat_map l ~f:written
            | Sexp.Atom _ -> []
          in
          written sexp)
  | _ -> []

(* A rule target compilation may need without naming it is source-like: any target but these, the
   extensions of files only read by name -- inverted after a review on PR #1027 found a source
   dialect ([.re]) the old list of source suffixes missed. A dialect declared in [dune-project] is
   assumed not to take one of these extensions. *)
let data_suffixes =
  [
    ".actual";
    ".expected";
    ".txt";
    ".filelist";
    ".generators";
    ".generated";
    ".sources";
    ".raw";
    ".log";
    ".tsv";
    ".csv";
    ".json";
    ".out";
    ".dat";
  ]

let source_named target =
  (* The configuration file is read by name too. *)
  (not (String.equal target Scan.config_file))
  && not (List.exists data_suffixes ~f:(fun suffix -> String.is_suffix target ~suffix))

(* A pattern is judged by its suffix too: [*-0-0.log] names logs, [*] or [x.*] anything. *)
let source_like = source_named

(* The stanza heads the inventory models: the ones that run something on an alias or for a target,
   the ones that compile, and the ones that build nothing a test runs. Not [install] nor
   [documentation]: dune's generated [@install] and [@doc] build what they list, which no alias here
   carries, so each is a hole in the inventory (Codex review on PR #1027). *)
let running_heads = [ "rule"; "alias"; "test"; "tests"; "library" ]

let compiling_heads =
  [
    "library";
    "executable";
    "executables";
    "test";
    "tests";
    "ocamllex";
    "ocamlyacc";
    "menhir";
    "env";
    "foreign_library";
  ]

let inert_heads = [ "copy_files"; "copy_files#"; "dirs"; "data_only_dirs"; "vendored_dirs" ]

(* The configuration keys that decide the backend: the backend itself, and [no_config_file], which
   drops the configuration (no backend at all is [Context.auto]'s, GPUs first). *)
let backend_keys = [ "backend"; "no_config_file" ]

(** The first place [sexp] sets the backend past the configuration -- one of {!backend_keys} as an
    [OCANNL_] variable bound by a [setenv], an [env]'s [env_vars]/[env-vars], an [env] command or a
    shell assignment, a rule generating an [ocannl_config] (one the tracked configurations the
    runner reads cannot show), or, where the stanza reads the configuration, a command-line flag in
    any spelling -- which the resolved configuration cannot answer for, so a batch reaching it can
    hold any backend (Codex review on PR #1027). *)
let backend_override ~reads_config sexp =
  let normal w = String.lowercase (String.map w ~f:(function '-' -> '_' | c -> c)) in
  let is_backend name =
    List.exists backend_keys ~f:(fun k -> String.equal (normal name) ("ocannl_" ^ k))
  in
  let rec go = function
    | Sexp.Atom a ->
        (* In a stanza that reads the configuration, any word of any atom -- a shell line's
           included: the variable in any spelling, bare or assigned ([env OCANNL_BACKEND=hip], a
           [system] line), or the flag in every spelling [Utils.cmdline_var_names] accepts (one or
           two dashes, the [ocannl] prefix or none, any case, either separator). *)
        let words =
          String.split_on_chars a ~on:[ ' '; '\t'; '\n'; ';'; '&'; '|'; '('; ')'; '"'; '\'' ]
        in
        Option.some_if
          (reads_config
          && List.exists words ~f:(fun w ->
              let n = normal w in
              let key =
                let bare = String.lstrip n ~drop:(Char.equal '_') in
                Option.value (String.chop_prefix bare ~prefix:"ocannl_") ~default:bare
              in
              (* [env] rewrites the environment it runs a program in ([-i] empties it, [-u] unsets a
                 variable), so a reader run through it is not judged by the ambient one. *)
              String.equal w "env" || String.is_suffix w ~suffix:"/env"
              || List.exists backend_keys ~f:(fun k ->
                  String.equal n ("ocannl_" ^ k)
                  || String.is_prefix n ~prefix:("ocannl_" ^ k ^ "=")
                  || (String.is_prefix w ~prefix:"-" && String.is_prefix key ~prefix:k))))
          (Printf.sprintf "word in %s" a)
    (* The declaration that the stanza reads the variable is not a setting of it. *)
    | Sexp.List [ Sexp.Atom "env_var"; _ ] -> None
    (* Only a stanza that reads the configuration is moved by it: one naming its backend (a [none]
       scan setting [OCANNL_NO_CONFIG_FILE] to read its own fixture, say) holds what it names. An
       [env] stanza's variables are not modelled at all ([env_unmodelled]). *)
    | Sexp.List (Sexp.Atom "setenv" :: Sexp.Atom name :: _) when reads_config && is_backend name ->
        Some (Printf.sprintf "(setenv %s …)" name)
    | Sexp.List (Sexp.Atom (("env_vars" | "env-vars") as field) :: bindings) when reads_config ->
        List.find_map (List.concat_map bindings ~f:Scan.atoms) ~f:(fun name ->
            Option.some_if (is_backend name) (Printf.sprintf "(%s (%s …))" field name))
    | Sexp.List l -> List.find_map l ~f:go
  in
  if List.exists (targets_of sexp) ~f:(fun t -> Scan.glob_could_match t ~name:Scan.config_file) then
    Some "generated ocannl_config"
  else go sexp

(* The directories whose [ocannl_config] the runner resolves -- tools/batch-backends.sh's
   BATCH_CONFIG_DIRS, which tools/test-test-run.sh leg 58 holds equal to this. A copy of any other
   directory's configuration (the gitignored root one, say) is a backend the runner never reads
   (Codex review on PR #1027). *)
let config_dirs = [ "test/config"; "arrayjit/test" ]

(** What a [copy_files] stanza copies, as a target pattern here and the glob it reads: its short
    form's path, or its long form's [(files …)]. The copy keeps the basename, so a file need is
    followed past it to the original producer anyway; this is what a GLOB in the copy's directory
    needs, since globs are matched where they point (Codex review on PR #1027). *)
let copies_of ~dir sexp =
  let spec =
    match sexp with
    | Sexp.List [ Sexp.Atom _; Sexp.Atom spec ] -> Some spec
    | _ -> ( match Scan.field sexp "files" with Some [ Sexp.Atom spec ] -> Some spec | _ -> None)
  in
  match spec with
  | None -> ([], [])
  | Some spec ->
      let in_dir, base =
        match String.rsplit2 spec ~on:'/' with Some (d, b) -> (d, b) | None -> ("", spec)
      in
      let pattern =
        String.concat
          (List.map (Scan.pieces base) ~f:(function Scan.Literal l -> l | Scan.Pform _ -> "*"))
      in
      ( [ pattern ],
        [ Glob_need { under = Option.map (resolve ~dir in_dir) ~f:(fun d -> (d, false)); pattern } ]
      )

(* The settings an [env] stanza's profiles may carry and stay exact: compiler and linker flags. Any
   other -- [env_vars] in either spelling, [binaries], ... -- changes what the actions it covers
   build or see, and is not modelled (Codex review on PR #1027). *)
let env_settings =
  [ "flags"; "ocamlc_flags"; "ocamlopt_flags"; "c_flags"; "cxx_flags"; "link_flags" ]

let env_unmodelled = function
  | Sexp.List (Sexp.Atom "env" :: profiles) ->
      List.find_map profiles ~f:(function
        | Sexp.List (Sexp.Atom _ :: settings) ->
            List.find_map settings ~f:(function
              | Sexp.List (Sexp.Atom h :: _) when List.mem env_settings h ~equal:String.equal ->
                  None
              | other -> Some (Printf.sprintf "env setting %s" (Sexp.to_string other)))
        | other -> Some (Printf.sprintf "env profile %s" (Sexp.to_string other)))
  | _ -> None

(* The action heads the closure reads: the ones that write a file it infers ([targets_of]), and the
   ones that write none. Any other -- [format-dune-file], say -- may write what this cannot name. *)
let action_heads =
  [
    "run";
    "dynamic-run";
    "system";
    "bash";
    "progn";
    "concurrent";
    "echo";
    "cat";
    "copy";
    "copy#";
    "copy-and-add-line-directive";
    "write-file";
    "with-stdout-to";
    "with-stderr-to";
    "with-outputs-to";
    "with-stdin-from";
    "with-accepted-exit-codes";
    "ignore-stdout";
    "ignore-stderr";
    "ignore-outputs";
    "chdir";
    "setenv";
    "diff";
    "diff?";
    "cmp";
    "no-infer";
    "pipe-stdout";
    "pipe-stderr";
    "pipe-outputs";
  ]

(** The first action head in a stanza's [action] (or a short-form rule's) this does not read. *)
let unknown_action_head sexp =
  let rec action = function
    | Sexp.Atom _ -> None
    | Sexp.List (Sexp.Atom "with-accepted-exit-codes" :: _codes :: rest) ->
        List.find_map rest ~f:action
    | Sexp.List (Sexp.Atom h :: args) ->
        if List.mem action_heads h ~equal:String.equal then List.find_map args ~f:action
        else Some (Printf.sprintf "action (%s …)" h)
    | Sexp.List _ as other -> Some (Printf.sprintf "action %s" (Sexp.to_string other))
  in
  let rule_fields =
    [
      "targets";
      "target";
      "deps";
      "action";
      "alias";
      "aliases";
      "package";
      "locks";
      "mode";
      "fallback";
      "enabled_if";
    ]
  in
  let stray_form =
    (* A rule's other forms: its short-form action, or a field this does not know. *)
    match sexp with
    | Sexp.List (Sexp.Atom "rule" :: fields) ->
        List.find_map fields ~f:(function
          | Sexp.List (Sexp.Atom h :: _) when List.mem rule_fields h ~equal:String.equal -> None
          | form -> Some (action form))
    | _ -> None
  in
  match (Scan.field sexp "action", stray_form) with
  | _, Some found -> found
  | Some [ a ], None -> action a
  | Some _, None -> Some "action field of more than one form"
  | None, None -> None

(** One stanza, as the closure reads it: run, and -- for a head that compiles -- compiled, which is
    a stanza of its own here, seeded for every batch. *)
let views_of ~dir ~named ~reads_config ~unmarked sexp =
  let head = Option.value (Scan.head sexp) ~default:"<not a stanza>" in
  let known =
    List.exists [ running_heads; compiling_heads; inert_heads ] ~f:(fun l ->
        List.mem l head ~equal:String.equal)
  in
  let unknown = Option.some_if (not known) (Printf.sprintf "stanza (%s …)" head) in
  let run =
    let copies, copied =
      if String.is_prefix head ~prefix:"copy_files" then copies_of ~dir sexp else ([], [])
    in
    let produced = targets_of sexp in
    let targets = produced @ copies in
    let deps, inexact =
      match dep_needs ~dir sexp with
      | Ok (needs, bindings) -> (needs @ copied, inexact_pform ~bindings sexp)
      | Error form -> ([], Some form)
    in
    let inexact =
      List.find_map ~f:Fn.id
        [
          inexact;
          stray_alias sexp;
          unknown_action_head sexp;
          Option.some_if
            (List.mem (Scan.atoms sexp) "dynamic-run" ~equal:String.equal)
            "dynamic-run action";
        ]
    in
    {
      dir;
      role = Runs;
      attached = attached_aliases sexp;
      sexp;
      named;
      reads_config;
      unmarked;
      targets;
      (* A copy's wildcard is the glob it copies, not a target this cannot name. *)
      source_like = List.exists produced ~f:source_like || List.exists copies ~f:source_named;
      needs = List.dedup_and_sort (deps @ file_needs ~dir sexp) ~compare:Poly.compare;
      inexact;
      overrides = backend_override ~reads_config sexp;
      foreign_config =
        List.find_map copied ~f:(function
          | Glob_need { under; pattern } when Scan.glob_could_match pattern ~name:Scan.config_file
            -> (
              match under with
              | Some (src, _) when List.mem config_dirs src ~equal:String.equal -> None
              | Some (src, _) -> Some (Printf.sprintf "a copy of the ocannl_config in %s" src)
              | None -> Some "a copy of an ocannl_config from a directory this cannot name")
          | _ -> None);
    }
  in
  let compiled ~needs inexact =
    {
      dir;
      role = Compiles;
      attached = [];
      sexp;
      named = [];
      reads_config = false;
      unmarked = false;
      targets = [];
      source_like = false;
      needs;
      inexact;
      overrides = backend_override ~reads_config:false sexp;
      foreign_config = None;
    }
  in
  let compile =
    (* A head the inventory does not model is a hole in it -- an [(include …)] brings in stanzas
       this never reads -- so it is seeded like a compilation, for every batch. *)
    if not known then [ compiled ~needs:[] unknown ]
    else if not (List.mem compiling_heads head ~equal:String.equal) then
      (* An inert head attaching anything to an alias -- a [copy_files] long form, say -- is a hole
         too: no alias here carries it (Codex review on PR #1027). *)
      if
        String.equal head "rule"
        && Option.is_none (Scan.field sexp "targets")
        && Option.is_none (Scan.field sexp "target")
        && ((List.is_empty (Scan.aliases_of sexp) && List.is_empty (targets_of sexp))
           || Option.is_some (unknown_action_head sexp))
      then
        (* A rule on no alias whose targets no writing form names produces what this cannot name. *)
        (* ... and so does one whose action has a head this does not read. *)
        [ compiled ~needs:[] (Some "rule with targets no action form names") ]
      else if List.mem running_heads head ~equal:String.equal then []
      else
        Option.to_list (Option.map (stray_alias sexp) ~f:(fun why -> compiled ~needs:[] (Some why)))
    else
      (* A test's [deps] and [action], and an inline-test library's [inline_tests], are its run. *)
      let fields =
        match sexp with
        | Sexp.List (h :: fields) ->
            Sexp.List
              (h
              :: List.filter fields ~f:(function
                | Sexp.List (Sexp.Atom ("deps" | "action" | "inline_tests") :: _) -> false
                | _ -> true))
        | other -> other
      in
      let aliases, bindings, malformed =
        match dep_needs ~dir fields with
        | Ok (needs, bindings) -> (needs, bindings, None)
        | Error form -> ([], [], Some form)
      in
      let inexact =
        List.find_map ~f:Fn.id
          [
            malformed;
            stray_alias fields;
            Option.some_if
              (List.mem (Scan.atoms fields) "action" ~equal:String.equal)
              "preprocessing action";
            (* ctypes stubs run generator programs as they build; a compiler flag naming a program
               ([-pp], [-ppx], [-cc]) or code ([-plugin]) runs it, from any flags field or an
               env's. *)
            Option.map (Scan.field fields "ctypes") ~f:(fun _ -> "ctypes field");
            List.find_map (Scan.atoms fields) ~f:(fun a ->
                Option.some_if
                  (List.exists [ "-pp"; "-ppx"; "-cc"; "-plugin" ] ~f:(fun f ->
                       String.equal a f
                       || String.is_prefix a ~prefix:(f ^ "=")
                       || String.is_prefix a ~prefix:(f ^ " ")))
                  (Printf.sprintf "compiler flag %s" a));
            (* An included flags file says what it says only once built: not read. *)
            Option.some_if
              (List.mem (Scan.atoms fields) ":include" ~equal:String.equal)
              "(:include …) form";
            env_unmodelled fields;
            inexact_pform ~bindings fields;
          ]
      in
      [
        compiled
          ~needs:(List.dedup_and_sort (aliases @ file_needs ~dir fields) ~compare:Poly.compare)
          inexact;
      ]
  in
  run :: compile

(** The stanzas of one dune file, in [dir]. A marker the contract refuses is not read as absent: it
    raises, and the caller takes the unreadable file as a GPU answer. *)
let stanzas_of ~dir content =
  let contract = Scan.backend_marker_contract content in
  if not (List.is_empty contract.Scan.contract_issues) then
    failwith "a backend marker the env_var_deps contract refuses";
  List.concat_map contract.Scan.contract_stanzas ~f:(fun marked ->
      let st = marked.Scan.marker_stanza in
      let rule = Scan.backend_rule_of marked in
      let named, reads_config = backend_of_rule rule in
      let unmarked = match rule with Scan.Runs_nothing -> true | _ -> false in
      views_of ~dir:(join dir st.Scan.marked_subdir) ~named ~reads_config ~unmarked
        st.Scan.marked_sexp)

type target = Alias of { dir : string; alias : string; recursive : bool }

(* dune's options, read closed rather than open (Codex review round 3 on PR #803): an option this
   does not model could change WHAT is built -- [--alias-rec runtest] names an alias as the next
   word, [--root] and [--workspace] swap the tree being built for another -- so the only options let
   through are those listed here as changing nothing about which rules run. Anything else, an option
   this does not know included, makes the whole argv unmodelled, which the caller takes as a GPU.
   The lists follow [dune build --help] (dune 3.24). *)
let harmless_flags =
  [
    "-f";
    "--force";
    "-w";
    "--watch";
    "--passive-watch-mode";
    "--stop-on-first-error";
    "--wait-for-filesystem-clock";
    "--always-show-command-line";
    "--auto-promote";
    "--display-separate-messages";
    "--debug-backtraces";
    "--debug-dependency-path";
    "--debug-package-logs";
    "--disable-promotion";
    "--ignore-promoted-rules";
    "--no-buffer";
    "--no-print-directory";
    "--release";
    "--verbose";
    "--store-orig-source-dir";
    "--no-config";
  ]

(* Not [--build-dir] (a context-rooted alias such as [@out/default/runtest] then reads as a source
   directory), [--instrument-with] (it adds and runs a ppx) or [--diff-command] (it runs a program):
   each is unmodelled (Codex review on PR #1027). *)
let harmless_valued =
  [
    "-j";
    "--jobs";
    "-p";
    "--for-release-of-packages";
    "--only-packages";
    "--profile";
    "--display";
    "--cache";
    "--cache-check-probability";
    "--cache-storage-mode";
    "--sandbox";
    "--error-reporting";
    "--action-stdout-on-success";
    "--action-stderr-on-success";
    "--file-watcher";
    "--terminal-persistence";
    "--trace-file";
    "--dump-gc-stats";
    "--watch-exclusions";
    "--config-file";
  ]

type word = Target of string | Unmodelled of string

(** The argv's words past its subcommand: targets, and the first word this does not model. Dune's
    own [--] is one of those (past it, dune still reads targets), as is any option not listed above
    -- the alias-naming ones included, which the runner is never handed. *)
let rec words acc = function
  | [] -> List.rev acc
  | opt :: rest when String.is_prefix opt ~prefix:"-" ->
      let name, inline =
        match String.lsplit2 opt ~on:'=' with
        | Some (n, v) when String.is_prefix n ~prefix:"--" -> (n, Some v)
        | _ -> (opt, None)
      in
      (* `-j8`: a short option with its value attached. *)
      let name, inline =
        if
          String.length name > 2
          && (not (String.is_prefix name ~prefix:"--"))
          && Option.is_none inline
        then (String.prefix name 2, Some (String.drop_prefix name 2))
        else (name, inline)
      in
      if List.mem harmless_flags name ~equal:String.equal && Option.is_none inline then
        words acc rest
      else if List.mem harmless_valued name ~equal:String.equal then
        match (inline, rest) with
        | Some _, _ -> words acc rest
        | None, _ :: rest' -> words acc rest'
        | None, [] -> List.rev (Unmodelled opt :: acc)
      else List.rev (Unmodelled opt :: acc)
  | word :: rest -> words (Target word :: acc) rest

(* A directory in the one spelling this reads: relative, its components plain names -- no `.`, `..`
   or empty one, so no leading `/`, `./` or trailing `/`. The repository root is spelled by giving
   no directory at all. Any other spelling is unmodelled rather than normalised (Codex review round
   4 on PR #803): normalising is exactly where a spelling dune reads one way gets read another. No
   component starts with [_] or [.] either: those are the directories dune does not read as source
   ([_build], [.git]), so [@_build/default/runtest] -- a build-context root, which runs the whole
   context's suite -- names no directory the inventory holds, and is unmodelled rather than read as
   reaching nothing (Codex review on PR #1027). *)
let plain_dir d =
  (not (String.is_empty d))
  && List.for_all (String.split d ~on:'/') ~f:(fun c ->
      (not (String.is_empty c))
      && (not (String.is_prefix c ~prefix:"_"))
      && not (String.is_prefix c ~prefix:"."))

(* An alias target in the spellings this reads: [@alias], [@@alias], [@dir/alias], [@@dir/alias],
   with [dir] plain. A path target, and anything else, is unmodelled. *)
let target_of w =
  let alias_in ~recursive spec =
    match String.rsplit2 spec ~on:'/' with
    | None when plain_dir spec -> Some (Alias { dir = ""; alias = spec; recursive })
    | Some (dir, alias) when plain_dir dir && plain_dir alias ->
        Some (Alias { dir; alias; recursive })
    | _ -> None
  in
  match String.chop_prefix w ~prefix:"@@" with
  | Some spec -> alias_in ~recursive:false spec
  | None -> Option.bind (String.chop_prefix w ~prefix:"@") ~f:(alias_in ~recursive:true)

(** The targets a dune argv builds: [Ok None] for a subcommand known to build nothing ([promote],
    [clean]), [Error word] for one carrying a word this does not model -- any other subcommand
    included: [install], say, builds what it installs (Codex review on PR #1027). *)
let targets argv =
  let split ~target rest =
    let ws = words [] rest in
    match List.find_map ws ~f:(function Unmodelled o -> Some o | Target _ -> None) with
    | Some o -> Error o
    | None ->
        List.fold_result ws ~init:[] ~f:(fun acc -> function
          | Target w -> ( match target w with Some t -> Ok (t :: acc) | None -> Error w)
          | Unmodelled o -> Error o)
        |> Result.map ~f:List.rev
  in
  match argv with
  | ("runtest" | "test") :: rest ->
      Result.map
        (split rest ~target:(fun w ->
             if plain_dir w then Some (Alias { dir = w; alias = "runtest"; recursive = true })
             else None))
        ~f:(fun ts ->
          Some
            (if List.is_empty ts then [ Alias { dir = ""; alias = "runtest"; recursive = true } ]
             else ts))
  | "build" :: rest ->
      Result.map (split rest ~target:target_of) ~f:(fun ts ->
          Some
            (if List.is_empty ts then [ Alias { dir = ""; alias = "default"; recursive = true } ]
             else ts))
  | ("promote" | "clean") :: _ -> Ok None
  | sub :: _ -> Error sub
  | [] -> Error "<no subcommand>"

let in_scope ~recursive ~root dir =
  String.equal root dir
  || (recursive && (String.is_empty root || String.is_prefix dir ~prefix:(root ^ "/")))

(** Every stanza building [targets] builds, in the order the tree lists them: the closure of the
    argv's aliases, every compilation and every source-like producer under {!need}s. *)
let reached stanzas targets =
  let arr = Array.of_list stanzas in
  let producers =
    List.concat_mapi stanzas ~f:(fun i s -> List.map s.targets ~f:(fun t -> (i, t)))
  in
  let seen = Array.create ~len:(Array.length arr) false in
  let queue = Queue.create () in
  let add i =
    if not seen.(i) then (
      seen.(i) <- true;
      Queue.enqueue queue i)
  in
  let requested = Hash_set.Poly.create () in
  let request ~root ~alias ~recursive =
    (* [default]'s implicit definition is [(alias_rec all)], so even [@@default] recurses. *)
    let recursive = recursive || String.equal alias "default" in
    if not (Hash_set.mem requested (root, alias, recursive)) then (
      Hash_set.add requested (root, alias, recursive);
      Array.iteri arr ~f:(fun i s ->
          match s.role with
          | Compiles -> ()
          | Runs ->
              if in_scope ~recursive ~root s.dir then
                (* `default` and `all` build every target in the directory, not an alias's
                   members. *)
                if String.equal alias "default" || String.equal alias "all" then (
                  if not (List.is_empty s.attached && List.is_empty s.targets) then add i)
                else if List.mem s.attached alias ~equal:String.equal then add i))
  in
  let produce matches =
    List.iter producers ~f:(fun (i, t) -> if matches arr.(i).dir t then add i)
  in
  List.iter targets ~f:(fun (Alias { dir; alias; recursive }) ->
      request ~root:dir ~alias ~recursive);
  Array.iteri arr ~f:(fun i s ->
      match s.role with Compiles -> add i | Runs -> if s.source_like then add i);
  while not (Queue.is_empty queue) do
    List.iter arr.(Queue.dequeue_exn queue).needs ~f:(function
      | Alias_need { dir; alias; recursive } -> request ~root:dir ~alias ~recursive
      | File_need { in_dir; base } ->
          produce (fun dir t ->
              (match in_dir with None -> true | Some d -> String.equal d dir)
              && Scan.glob_could_match t ~name:base)
      | Glob_need { under; pattern } ->
          produce (fun dir t ->
              (match under with
                | None -> true
                | Some (root, recursive) -> in_scope ~recursive ~root dir)
              && (String.exists t ~f:is_wild || Scan.glob_could_match pattern ~name:t)))
  done;
  List.filteri stanzas ~f:(fun i _ -> seen.(i))

(** Where a reached stanza is, for a reason: its names (or a rule's aliases or targets) and its
    directory. *)
let describe s =
  let what =
    match (Scan.names_of s.sexp, Scan.aliases_of s.sexp) with
    | [], [] when not (List.is_empty s.targets) ->
        "the rule producing " ^ String.concat ~sep:"," s.targets
    | [], [] -> "the " ^ Option.value (Scan.head s.sexp) ~default:"unnamed" ^ " stanza"
    | [], aliases -> "the rule on " ^ String.concat ~sep:"," aliases
    | names, _ -> String.concat ~sep:"," names
  in
  let dir = if String.is_empty s.dir then "." else s.dir in
  match s.role with
  | Runs -> Printf.sprintf "it reaches %s in %s" what dir
  | Compiles
    when List.mem compiling_heads (Option.value (Scan.head s.sexp) ~default:"") ~equal:String.equal
    ->
      Printf.sprintf "it compiles %s in %s" what dir
  | Compiles -> Printf.sprintf "it reads the dune file in %s" dir

(* Run from inside [_build], the walk shares its tree with the rules dune runs beside it, which
   create and delete their outputs while it reads: an entry can vanish between the listing that
   names it and the [stat] that classifies it, or a directory between the [stat] and its own listing
   (gh-ocannl-1227). What vanished is no directory and holds nothing -- the source tree dune reads
   never loses an entry mid-build, so only build outputs, which dune does not read as dune files,
   are passed over (one a rule recreates right after is no exception). *)

(** The entries of [path], sorted; [[]] once [path] no longer exists. The verdict is the failing
    call's own errno, never a second probe after it -- a probe opens a window of its own, in which a
    rule can recreate what the listing found gone. [Sys.readdir] carries no errno ([Sys_error] holds
    only a message), hence [Unix]. Any failure but [ENOENT] raises: the caller takes an unreadable
    tree as every backend. *)
let listing path =
  match Unix.opendir path with
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> []
  | handle ->
      let rec read acc =
        match Unix.readdir handle with
        | "." | ".." -> read acc
        | entry -> read (entry :: acc)
        | exception End_of_file -> acc
      in
      Exn.protect
        ~f:(fun () -> List.sort (read []) ~compare:String.compare)
        ~finally:(fun () -> Unix.closedir handle)

(** The [entries] of [path] that are directories, following links as dune does; one that no longer
    exists, or a link to nothing, is not. Any other stat failure raises, as in {!listing}. *)
let subdirectories path entries =
  List.filter entries ~f:(fun e ->
      match Unix.stat (Stdlib.Filename.concat path e) with
      | { Unix.st_kind = Unix.S_DIR; _ } -> true
      | _ -> false
      | exception Unix.Unix_error (Unix.ENOENT, _, _) -> false)

(** Every dune file under [root] that dune itself would read, as [(dir, content)] with [dir]
    relative to [root] ([""] for [root] itself). Into a directory's subdirectories as dune goes: by
    default the ones whose name starts with neither [.] nor [_] (not _build, _opam, .git, ...), and
    where a [(dirs …)] stanza says otherwise, the ones it names -- [:standard] for that default, a
    name or a glob for the subdirectories it matches (the root's [(dirs :standard .github .claude)]
    admits [.claude], whose own [(dirs skills)] keeps its worktrees out; Codex review on PR #1027),
    less the ones a [(data_only_dirs …)] names -- each stanza applying where dune applies it, a
    [(subdir …)] scoping it to that subdirectory. A directory stanza in any other shape raises: the
    caller takes an unreadable tree as every backend. *)
let dune_files ?(workspace_root = true) ~root () =
  let standard e = not (String.is_prefix e ~prefix:"." || String.is_prefix e ~prefix:"_") in
  let matches args e =
    (* Every entry is checked before any is matched: an ordered-set operator ([\\], another [:name])
       changes what the rest means, so a set carrying one is not read at all. *)
    List.iter args ~f:(function
      | Sexp.Atom ":standard" -> ()
      | Sexp.Atom name
        when not (String.is_prefix name ~prefix:"\\" || String.is_prefix name ~prefix:":") ->
          ()
      | other ->
          failwith
            (Printf.sprintf "a directory-set entry this does not read: %s" (Sexp.to_string other)));
    List.exists args ~f:(function
      | Sexp.Atom ":standard" -> standard e
      | Sexp.Atom name -> Scan.glob_could_match name ~name:e
      | Sexp.List _ -> false)
  in
  (* The directory stanzas that apply to a directory: its own dune file's, and those a [(subdir …)]
     in an ancestor's scopes to it (Codex review on PR #1027), keyed by directory. *)
  let scoped = Hashtbl.create (module String) in
  let rec collect dir = function
    | Sexp.List (Sexp.Atom (("dirs" | "data_only_dirs") as kind) :: args) ->
        Hashtbl.add_multi scoped ~key:dir ~data:(kind, args)
    | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body) ->
        List.iter body ~f:(collect (join dir sub))
    | _ -> ()
  in
  let admitted dir entries =
    let stanzas = Hashtbl.find_multi scoped dir in
    let dirs =
      match List.filter_map stanzas ~f:(function "dirs", a -> Some a | _ -> None) with
      | [] -> List.filter entries ~f:standard
      | [ args ] -> List.filter entries ~f:(matches args)
      | _ -> failwith (Printf.sprintf "more than one (dirs …) stanza for %s" dir)
    in
    (* A data-only directory's dune file is not read by dune, so it is not read here. *)
    List.fold stanzas ~init:dirs ~f:(fun dirs -> function
      | "data_only_dirs", args ->
          (* Matched exactly, since it takes directories away: a character class or an alternative,
             which [Scan.glob_could_match] reads as matching anything, is refused. *)
          List.iter args ~f:(function
            | Sexp.Atom a when String.exists a ~f:(fun c -> Char.equal c '[' || Char.equal c '{') ->
                failwith (Printf.sprintf "a data_only_dirs pattern this does not read: %s" a)
            | _ -> ());
          List.filter dirs ~f:(fun e -> not (matches args e))
      | _ -> dirs)
  in
  (* What dune reads beyond the [dune] files is not modelled: a workspace naming more than its
     language (a context's environment can set the backend), and the alternative [dune-file] name a
     [dune-project] can enable. Either makes the tree unreadable (Codex review on PR #1027). *)
  (match Stdlib.Sys.getenv_opt "DUNE_WORKSPACE" with
  | Some w when not (String.is_empty w) -> failwith "DUNE_WORKSPACE names a workspace"
  | _ -> ());
  (* Every project the walk enters, the root's and any nested one (Codex review on PR #1027). *)
  let check_project path =
    let project = Stdlib.Filename.concat path "dune-project" in
    if Stdlib.Sys.file_exists project then
      List.iter
        (Scan.stanzas (Stdio.In_channel.read_all project))
        ~f:(function
          | Sexp.List (Sexp.Atom (("dialect" | "accept_alternative_dune_file_name") as h) :: _) ->
              (* A dialect can make any extension a source one; the data list assumes none does. *)
              failwith (Printf.sprintf "%s declares (%s …)" project h)
          | _ -> ())
  in
  (* The root dune builds is the outermost ancestor holding a dune-workspace, or failing one the
     outermost holding a dune-project: an ancestor that would take the root from [root] (a nested
     worktree without its own dune-workspace, say) means dune builds a tree this did not read (Codex
     review on PR #1027). A workspace file itself names its language only. The ancestors are skipped
     where [root] is read as a copy, not as the workspace ([~workspace_root:false]). *)
  let absolute =
    (* The working directory itself for [.], so that its parent is the first ancestor. *)
    if List.mem [ "."; "" ] root ~equal:String.equal then Stdlib.Sys.getcwd ()
    else if Stdlib.Filename.is_relative root then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) root
    else root
  in
  let check_workspace file =
    if Stdlib.Sys.file_exists file then
      List.iter
        (Scan.stanzas (Stdio.In_channel.read_all file))
        ~f:(function
          | Sexp.List (Sexp.Atom "lang" :: _) -> ()
          | other -> failwith (Printf.sprintf "%s carries %s" file (Sexp.to_string other)))
  in
  check_workspace (Stdlib.Filename.concat absolute "dune-workspace");
  (if workspace_root then
     let own = Stdlib.Sys.file_exists (Stdlib.Filename.concat absolute "dune-workspace") in
     let rec ancestors dir =
       let up = Stdlib.Filename.dirname dir in
       if not (String.equal up dir) then (
         List.iter [ "dune-workspace"; "dune-project" ] ~f:(fun f ->
             if
               Stdlib.Sys.file_exists (Stdlib.Filename.concat up f)
               && (String.equal f "dune-workspace" || not own)
             then failwith (Printf.sprintf "%s holds a %s, which takes dune's root" up f));
         ancestors up)
     in
     ancestors absolute);
  let rec under dir =
    let path = if String.is_empty dir then root else Stdlib.Filename.concat root dir in
    (* The root itself must be there: only a directory found inside the walk can have vanished. *)
    let entries =
      if String.is_empty dir then
        Stdlib.Sys.readdir path |> Array.to_list |> List.sort ~compare:String.compare
      else listing path
    in
    if List.mem entries "dune-file" ~equal:String.equal then
      failwith (Printf.sprintf "%s holds a dune-file" path);
    check_project path;
    (* A [.t] file or directory is a cram test dune discovers with no stanza naming it. *)
    Option.iter
      (List.find entries ~f:(String.is_suffix ~suffix:".t"))
      ~f:(fun t -> failwith (Printf.sprintf "%s holds the cram test %s" path t));
    let content =
      if List.mem entries "dune" ~equal:String.equal then
        Some (Stdio.In_channel.read_all (Stdlib.Filename.concat path "dune"))
      else None
    in
    Option.iter content ~f:(fun c -> List.iter (Scan.stanzas c) ~f:(collect dir));
    let subdirs = subdirectories path entries in
    Option.to_list (Option.map content ~f:(fun c -> (dir, c)))
    @ List.concat_map (admitted dir subdirs) ~f:(fun e -> under (join dir e))
  in
  under ""

(** What a dune argv can hold. [Reaches] lists in [named] each backend a reached stanza's marker
    names, once, with the first stanza that names it, and gives in [reads_config] the first reached
    stanza that selects its backend from the configuration -- [None] only on the proof that nothing
    it builds reads it, or when it runs nothing. [Unknown why] is an argv this does not model, a
    dune file it could not read, a construct in what the batch builds that it does not model
    exactly, or a backend set past the configuration, which the caller takes as every backend. *)
type answer =
  | Reaches of { named : (string * string) list; reads_config : string option }
  | Unknown of string

(* The libraries through which a program can reach a backend: OCANNL's own, and the GPU bindings. *)
let backend_capable lib =
  List.exists [ "arrayjit"; "ocannl"; "neural_nets_lib"; "cudajit"; "hipjit"; "metal" ] ~f:(fun l ->
      String.equal lib l || String.is_prefix lib ~prefix:(l ^ "."))

(** The first preprocessor a stanza names that can reach a backend: a [pps]/[staged_pps] entry that
    is a backend-capable library or a workspace library whose libraries, transitively, include one.
    A ppx runs while dune compiles, so one that could start a backend is not taken on trust (Codex
    review on PR #1027). *)
let ppx_reaching_backend stanzas =
  let libraries = Hashtbl.create (module String) in
  List.iter stanzas ~f:(fun s ->
      match (s.role, Scan.head s.sexp) with
      | Runs, Some "library" ->
          let deps =
            Option.value_map (Scan.field s.sexp "libraries") ~default:[]
              ~f:(List.concat_map ~f:Scan.atoms)
          in
          let public =
            Option.value_map (Scan.field s.sexp "public_name") ~default:[]
              ~f:(List.concat_map ~f:Scan.atoms)
          in
          (* Every definition of a name kept, so a private name two projects each define is read as
             both (Codex review on PR #1027). *)
          List.iter
            (Scan.names_of s.sexp @ public)
            ~f:(fun n -> Hashtbl.add_multi libraries ~key:n ~data:deps)
      | _ -> ());
  let rec reaches seen lib =
    backend_capable lib
    || (not (Set.mem seen lib))
       && List.exists
            (List.concat (Hashtbl.find_multi libraries lib))
            ~f:(reaches (Set.add seen lib))
  in
  let rec ppxs = function
    | Sexp.List (Sexp.Atom ("pps" | "staged_pps") :: args) ->
        List.filter_map args ~f:(function
          | Sexp.Atom a when not (String.is_prefix a ~prefix:"-") -> Some a
          | _ -> None)
    | Sexp.List l -> List.concat_map l ~f:ppxs
    | Sexp.Atom _ -> []
  in
  List.find_map stanzas ~f:(fun s ->
      Option.map
        (List.find (ppxs s.sexp) ~f:(reaches (Set.empty (module String))))
        ~f:(fun ppx -> (s, ppx)))

(* The environment variables that run a program or add what is built: dune's, as their options do,
   and the compiler's [OCAMLPARAM], which can name a preprocessor or a ppx. *)
let unmodelled_dune_env = [ "DUNE_DIFF_COMMAND"; "DUNE_INSTRUMENT_WITH"; "DUNE_ROOT"; "OCAMLPARAM" ]

(* Dune's own aliases this models: [runtest] through the stanzas it attaches, [default] and [all] as
   every target, and the ones that build what every batch's compilations already cover. *)
let builtin_aliases =
  [ "runtest"; "default"; "all"; "check"; "install"; "doc"; "doc-private"; "fmt"; "lint" ]

let answer ?(getenv = Stdlib.Sys.getenv_opt) ~dune_files argv =
  let build_dir = Option.value (getenv "DUNE_BUILD_DIR") ~default:"" in
  (* With the build directory moved to a relative [out], [@out/default/runtest] is a context root,
     not a source directory (Codex review on PR #1027). *)
  let build_root =
    (* Normalized first ([./out] is [out]); an absolute one, or one leaving the tree, roots no alias
       here. *)
    if String.is_empty build_dir || not (Stdlib.Filename.is_relative build_dir) then None
    else Option.bind (resolve ~dir:"" build_dir) ~f:(fun d -> List.hd (String.split d ~on:'/'))
  in
  let targets argv =
    Result.bind (targets argv) ~f:(function
      | None -> Ok None
      | Some ts -> (
          match
            List.find ts ~f:(fun (Alias { dir; _ }) ->
                match build_root with
                | Some b -> String.equal (List.hd_exn (String.split dir ~on:'/')) b
                | None -> false)
          with
          | Some (Alias { dir; alias; _ }) -> Error (Printf.sprintf "@%s/%s" dir alias)
          | None -> Ok (Some ts)))
  in
  match
    List.find unmodelled_dune_env ~f:(fun v ->
        not (String.is_empty (Option.value (getenv v) ~default:"")))
  with
  | Some v -> Unknown (Printf.sprintf "%s is set, which this does not model" v)
  | None -> (
      match targets argv with
      | Error opt ->
          Unknown
            (Printf.sprintf
               "it carries `%s`, which this does not model (it could change what is built)" opt)
      | Ok None -> Reaches { named = []; reads_config = None }
      | Ok (Some targets) -> (
          let read =
            List.fold_result dune_files ~init:[] ~f:(fun acc (dir, content) ->
                match stanzas_of ~dir content with
                | stanzas -> Ok (stanzas :: acc)
                | exception exn ->
                    Error
                      (Printf.sprintf "the dune file in %s is unreadable here (%s)"
                         (if String.is_empty dir then "." else dir)
                         (Exn.to_string exn)))
          in
          match read with
          | Error why -> Unknown why
          | Ok stanzas -> (
              let stanzas = List.concat (List.rev stanzas) in
              (* An alias the argv names that no stanza in its scope carries is one of dune's own,
                 and only the ones listed are modelled ([revdep-runtest] builds what depends on the
                 directory; Codex review on PR #1027). *)
              let unmodelled_alias =
                List.find_map targets ~f:(fun (Alias { dir = root; alias; recursive }) ->
                    Option.some_if
                      ((not (List.mem builtin_aliases alias ~equal:String.equal))
                      && not
                           (List.exists stanzas ~f:(fun s ->
                                in_scope ~recursive ~root s.dir
                                && List.mem s.attached alias ~equal:String.equal)))
                      (Printf.sprintf "@%s%s"
                         (if String.is_empty root then "" else root ^ "/")
                         alias))
              in
              match unmodelled_alias with
              | Some a ->
                  Unknown
                    (Printf.sprintf "no stanza carries %s, and it is no dune alias this models" a)
              | None -> (
                  let found = reached stanzas targets in
                  (* A stanza the marker contract found running nothing may still run a WORKSPACE
                     program by its public name -- a bare [(run name)] or [%{bin:name}] resolves to
                     it -- which then declares no backend at all: not modelled. *)
                  let programs =
                    List.concat_map stanzas ~f:(fun s ->
                        match (s.role, Scan.head s.sexp) with
                        | Runs, Some ("executable" | "executables" | "test" | "tests") ->
                            List.concat_map
                              (List.filter_map [ "public_name"; "public_names" ]
                                 ~f:(Scan.field s.sexp))
                              ~f:(List.concat_map ~f:Scan.atoms)
                        | _ -> [])
                  in
                  let found =
                    List.map found ~f:(fun s ->
                        if Option.is_some s.inexact || not s.unmarked then s
                        else
                          let names =
                            List.concat_map (Scan.atoms s.sexp) ~f:(fun a ->
                                a
                                :: List.filter_map (Scan.pieces a) ~f:(function
                                  | Scan.Pform p -> String.chop_prefix p ~prefix:"bin:"
                                  | Scan.Literal _ -> None))
                          in
                          match List.find names ~f:(List.mem programs ~equal:String.equal) with
                          | Some prog ->
                              {
                                s with
                                inexact =
                                  Some
                                    (Printf.sprintf "run of the workspace program %s, unmarked" prog);
                              }
                          | None -> s)
                  in
                  (* A reader reads the configuration in its own directory, or the nearest above it:
                     one a [copy_files] brings there from a directory the runner does not read is a
                     backend the runner never sees (Codex review on PR #1027). *)
                  let foreign =
                    List.filter_map stanzas ~f:(fun s ->
                        Option.map s.foreign_config ~f:(fun why -> (s.dir, why)))
                  in
                  let found =
                    List.map found ~f:(fun s ->
                        match
                          List.find foreign ~f:(fun (d, _) ->
                              s.reads_config && in_scope ~recursive:true ~root:d s.dir)
                        with
                        | Some (d, why) when Option.is_none s.overrides ->
                            {
                              s with
                              overrides = Some (Printf.sprintf "configuration, %s (in %s)" why d);
                            }
                        | _ -> s)
                  in
                  let found =
                    (* Every compilation is in every batch's closure, so a ppx reaching a backend is
                       too. *)
                    match ppx_reaching_backend stanzas with
                    | Some (s, ppx) ->
                        {
                          s with
                          inexact = Some (Printf.sprintf "ppx %s (it can reach a backend)" ppx);
                        }
                        :: found
                    | None -> found
                  in
                  match
                    List.find_map found ~f:(fun s ->
                        match (s.overrides, s.inexact) with
                        | Some o, _ ->
                            Some
                              (Printf.sprintf "%s, whose %s sets the backend past the configuration"
                                 (describe s) o)
                        | None, Some c ->
                            Some
                              (Printf.sprintf "%s, whose %s this does not model exactly"
                                 (describe s) c)
                        | None, None -> None)
                  with
                  | Some why -> Unknown why
                  | None ->
                      let named =
                        List.concat_map found ~f:(fun s ->
                            let why =
                              Printf.sprintf "%s, which names %s" (describe s)
                                (String.concat ~sep:"," s.named)
                            in
                            List.map s.named ~f:(fun b -> (b, why)))
                        |> List.fold ~init:[] ~f:(fun acc (b, why) ->
                            if List.Assoc.mem acc b ~equal:String.equal then acc
                            else (b, why) :: acc)
                        |> List.rev
                      in
                      let reads_config =
                        List.find_map found ~f:(fun s ->
                            Option.some_if s.reads_config
                              (describe s ^ ", which reads the configuration"))
                      in
                      Reaches { named; reads_config }))))

(** Whether [answer] can hold a GPU by name: an unknown answer can. Whether a configuration a
    reached stanza reads names one is the caller's question. *)
let holds_gpu = function
  | Unknown _ -> true
  | Reaches { named; _ } ->
      List.exists named ~f:(fun (b, _) -> List.mem gpu_backends b ~equal:String.equal)
