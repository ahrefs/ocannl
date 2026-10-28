(** Which backends a dune invocation can run a stanza on whatever the configuration says, and
    whether the configuration counts at all (gh-ocannl-1004, gh-ocannl-1066, gh-ocannl-1227).

    tools/batch-backends.sh resolves the backends a [tools/test-run.sh] batch can hold, and from
    them both the dune width the batch runs at and the fleet slot it takes ([--cpu] only when it
    holds no GPU). The resolved configuration answers that for every stanza that selects its backend
    from it -- the ones declaring [(env_var OCANNL_BACKEND)] -- but not for the ones that NAME
    theirs: a stanza carrying [; ocannl-backend: cuda -- …] goes at the CUDA backend by name, and
    its [select] arm follows the library's availability, not the environment. Those markers are
    complete by construction -- [env_var_deps] fails a stanza that runs an executable and carries
    neither the variable nor a marker (gh-ocannl-659) -- so the stanzas a run can reach, and their
    markers, answer for what it names. Every backend a reached marker names is reported, a CPU one
    included: the width a batch runs at is the tightest any of its backends meets on the box.

    {b Whether the configuration counts is a policy, not a proof} (gh-ocannl-1227, the maintainer's
    decision of 2026-10-18). Two argv shapes leave it out, and every other one counts it:
    - every target is on {!config_free_aliases}, a closed, short list of aliases whose whole closure
      reads no configuration -- which [slot_kind_live] checks on the repository's own tree
      ({!refusals}), so a reader joining such an alias fails the scan rather than passing as
      backend-free;
    - or a plain compilation product: [build [./]<dir>/<name>.exe] (or [.bc]) of an executable or
      test stanza declaring [<name>] there, or [<dir>/<name>.cma]/[.cmxa] of such a library. A plain
      build compiles and links and runs nothing; what it builds first -- every compilation, every
      rule producing a source-like file, and the files they need -- reaches no reader and names no
      backend, which [slot_kind_live] checks too. gh-ocannl-1095 had instead taken the configuration
      off any batch it could PROVE read none, which needed an exact model of every dune construct a
      batch could build; each review round found another (response files, pforms in command
      positions, [%{bin:env}]). The allow-list needs none of that: no argv outside it is ever proven
      unread.

    What a run reaches is read from a CLOSED set of argv shapes, the ones the runner is used with;
    any other word is unmodelled, and answered as [Unknown] -- every backend. (Codex review rounds
    3-4 on PR #803 found a new corner of dune's CLI each round, so this stopped trying to model the
    CLI.)
    - [build] with alias targets: [@dir/alias] builds [alias] in [dir] and every directory below it,
      [@@dir/alias] in [dir] alone, [dir] plain (no [.], [..], leading or trailing [/]). An alias
      reaches what its stanzas attach to it -- the [runtest-<name>] dune generates for a
      [(test)]/[(tests)] stanza and an inline-test library included -- closed under what those build
      first (the section on what a batch builds, below). No target at all is the default alias
      everywhere, taken as reaching every stanza. And [build] with plain compilation products,
      above.
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

(** What building a stanza builds first: the aliases its dependency fields name, and the files it
    mentions, which the rules producing them must build. *)
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
  targets : string list;  (** the basenames (as patterns) of the files a rule produces *)
  source_like : bool;
      (** whether one of them is a file dune may build to compile or load anything *)
  needs : need list;
  unplaced : string list;
      (** the dependencies whose members the closure cannot follow: an [(alias …)] whose path does
          not resolve, and an [(include …)] of a dependency file. A batch reaching one is [Unknown],
          and {!refusals} refuses one inside a checked closure *)
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
    second marker, which {!marker_refusals} refuses before any stanza is read. *)
let backend_of_rule rule =
  let named { Scan.backend; _ } =
    String.split backend ~on:',' |> List.filter ~f:(fun b -> not (String.equal b "none"))
  in
  match rule with
  | Scan.Runs_nothing -> ([], false)
  | Scan.Declares_variable | Scan.Names_neither -> ([], true)
  | Scan.Names_backend (_, m) -> (named m, false)
  | Scan.Declares_and_names (_, m) -> (named m, true)
  | Scan.Names_twice _ ->
      failwith "a stanza carrying two backend markers, which marker_refusals refuses"

let join dir sub = match (dir, sub) with "", s -> s | d, "" -> d | d, s -> d ^ "/" ^ s

(** {1 What a batch builds}

    The closure collecting the named backends, seeded by the argv's aliases. It is an
    OVER-approximation with a stated boundary, not a proof: nothing is taken off a batch because the
    closure missed it -- the configuration counts outside the allow-list anyway -- and a construct
    it cannot follow inside a checked closure is refused by {!refusals}.
    - an [(alias …)]/[(alias_rec …)] anywhere in a dependency field (a stanza's [deps], an
      inline-test library's [(inline_tests (deps …))], [link_deps], ...), resolved against the
      stanza's directory, through the same attached-alias inventory the seeds use, generated
      per-test aliases included; one whose path does not resolve, and an included dependency file,
      is [unplaced] -- a batch reaching one is [Unknown], every backend, since what it builds there
      is unread;
    - a file: every atom anywhere in the stanza, and the payload of every path pform ([%{dep:…}],
      [%{read:…}], …), taken as a file it may need -- matched by BASENAME against every rule's
      targets in any directory, a pform elsewhere in the atom read as a wildcard, a [glob_files]
      pattern matched as a glob (a character class or an alternative matching everything). A copy
      keeps the basename, so this follows it to the original producer without modelling it;
    - compilation, for every batch that builds anything: every library, executable, test program,
      lexer, parser and [env] in the tree, its files followed the same way (a test's [deps] and
      [action] are its run, not its compilation), and every rule producing a source-like file. Which
      program links which library is not read, so all of them are. A directory target, and a rule
      naming neither a target nor an alias whose action writes no file this reads, are taken as
      producing every file. A backend fixed past the configuration -- a [setenv], an [env-vars], a
      [--ocannl_backend] flag -- is the marker's to name: [env_var_deps] fails a stanza whose action
      pins a backend its marker does not name, a pin it cannot read, and an [env-vars] binding of
      the backend (gh-ocannl-1518, [Dune_stanza_scan.backend_pins]). *)

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

(* The fields written in dune's dependency specification, wherever they sit: a stanza's [deps], an
   inline-test library's [(inline_tests (deps …))], an executable's [link_deps], a preprocessor's
   [preprocessor_deps], a foreign stub's [extra_deps] (Codex review on PR #1027). *)
let dep_field_names = [ "deps"; "link_deps"; "preprocessor_deps"; "extra_deps" ]

let rec dep_fields = function
  | Sexp.Atom _ -> []
  | Sexp.List (Sexp.Atom h :: args) when List.mem dep_field_names h ~equal:String.equal -> [ args ]
  | Sexp.List l -> List.concat_map l ~f:dep_fields

(** What a stanza's dependency fields need beyond its atoms -- the aliases they name and the globs
    they match, found at any depth (inside a named binding or a [sandbox] too) -- and the
    dependencies whose members this cannot follow: an alias whose path does not resolve, and an
    included dependency file, whose specifications exist only once built (Codex review round 1 on
    staging#1318). *)
let dep_needs ~dir sexp =
  let rec item = function
    | Sexp.Atom _ -> ([], [])
    | Sexp.List [ Sexp.Atom (("alias" | "alias_rec") as h); Sexp.Atom spec ] -> (
        match resolve ~dir spec with
        | Some p ->
            let dir, alias = Option.value (String.rsplit2 p ~on:'/') ~default:("", p) in
            ([ Alias_need { dir; alias; recursive = String.equal h "alias_rec" } ], [])
        | None -> ([], [ Printf.sprintf "(%s %s)" h spec ]))
    | Sexp.List [ Sexp.Atom "include"; Sexp.Atom file ] -> ([], [ "(include " ^ file ^ ")" ])
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
        ([ Glob_need { under; pattern = (if String.is_empty pattern then "*" else pattern) } ], [])
    | Sexp.List l ->
        let parts = List.map l ~f:item in
        (List.concat_map parts ~f:fst, List.concat_map parts ~f:snd)
  in
  let parts = List.concat_map (dep_fields sexp) ~f:(List.map ~f:item) in
  (List.concat_map parts ~f:fst, List.concat_map parts ~f:snd)

(** The basenames, as patterns, of the files a rule produces: its [(target …)]/[(targets …)], or,
    without them, the file each writing form of its action writes. *)
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
          let found = written sexp in
          (* A rule on no alias whose action writes nothing this reads produces what this cannot
             name: taken as producing anything. *)
          if List.is_empty found && List.is_empty (Scan.aliases_of sexp) then [ "*" ] else found)
  | _ -> []

(* A rule target compilation may need without naming it is source-like: any target but these, the
   extensions of files only read by name -- inverted after a review on PR #1027 found a source
   dialect ([.re]) the old list of source suffixes missed. A dialect declared in [dune-project] is
   refused by {!check_project}. *)
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
   carries -- {!refusals} refuses any head outside these lists (Codex review on PR #1027). *)
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

(* The directories whose [ocannl_config] the runner resolves -- tools/batch-backends.sh's
   BATCH_CONFIG_DIRS, which tools/test-test-run.sh leg 58 holds equal to this. A reader reading any
   other directory's configuration (the gitignored root one, say) reads a backend the runner never
   sees, which {!refusals} refuses (gh-ocannl-1249). *)
let config_dirs = [ "test/config"; "arrayjit/test" ]

(* gh-ocannl-1149: where test/config/dune writes the shared configuration with the ambient
   environment appended as comments, which is what every test directory copies. It resolves the same
   backend as test/config's -- the same keys, read from the same environment -- so a copy of it is
   no foreign configuration, as long as the rule writing it has exactly the shape
   {!ambient_augmentation} reads. *)
let ambient_config_dir = "test/config/ambient"

(** Whether [sexp], in [dir], is the rule writing {!ambient_config_dir}'s configuration as
    test/config's followed by the ambient record and nothing else. *)
let ambient_augmentation ~dir sexp =
  String.equal dir ambient_config_dir
  && Option.equal String.equal (Scan.head sexp) (Some "rule")
  && (match Scan.field sexp "target" with
    | Some [ Sexp.Atom target ] -> String.equal target Scan.config_file
    | _ -> false)
  && Option.is_none (Scan.field sexp "targets")
  &&
  match Scan.field sexp "action" with
  | Some
      [
        Sexp.List
          [
            Sexp.Atom "with-stdout-to";
            Sexp.Atom "%{target}";
            Sexp.List
              [
                Sexp.Atom "progn";
                Sexp.List [ Sexp.Atom "cat"; Sexp.Atom config ];
                Sexp.List [ Sexp.Atom "cat"; Sexp.Atom record ];
              ];
          ];
      ] ->
      String.equal config ("../" ^ Scan.config_file)
      && String.equal record ("../" ^ Scan.ambient_file)
  | _ -> false

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

(** One stanza, as the closure reads it: run, and -- for a head that compiles -- compiled, which is
    a stanza of its own here, seeded for every batch. *)
let views_of ~dir ~named ~reads_config sexp =
  let head = Option.value (Scan.head sexp) ~default:"<not a stanza>" in
  let run =
    let copies, copied =
      if String.is_prefix head ~prefix:"copy_files" then copies_of ~dir sexp else ([], [])
    in
    let produced = targets_of sexp in
    let deps, unplaced = dep_needs ~dir sexp in
    (* A long-form [copy_files] can attach its copies to an alias, which then builds them and so
       what they copy (Codex review round 4 on staging#1318). *)
    let copy_aliases =
      if String.is_prefix head ~prefix:"copy_files" then
        Option.value_map (Scan.field sexp "alias") ~default:[] ~f:(List.concat_map ~f:Scan.atoms)
      else []
    in
    {
      dir;
      role = Runs;
      attached = attached_aliases sexp @ copy_aliases;
      sexp;
      named;
      reads_config;
      targets = produced @ copies;
      (* A copy's wildcard is the glob it copies, not a target this cannot name. *)
      source_like = List.exists produced ~f:source_like || List.exists copies ~f:source_named;
      needs = List.dedup_and_sort (deps @ copied @ file_needs ~dir sexp) ~compare:Poly.compare;
      (* A [dynamic-run] discovers what it needs as it runs (Codex review round 3 on
         staging#1318). *)
      unplaced =
        (unplaced
        @
        if List.mem (Scan.atoms sexp) "dynamic-run" ~equal:String.equal then [ "(dynamic-run …)" ]
        else []);
    }
  in
  let compile =
    if not (List.mem compiling_heads head ~equal:String.equal) then []
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
      let deps, unplaced = dep_needs ~dir fields in
      [
        {
          dir;
          role = Compiles;
          attached = [];
          sexp;
          named = [];
          reads_config = false;
          targets = [];
          source_like = false;
          needs = List.dedup_and_sort (deps @ file_needs ~dir fields) ~compare:Poly.compare;
          unplaced;
        };
      ]
  in
  run :: compile

(** The head of a stanza the inventory does not read, if it is one: an [(include …)] brings in
    stanzas never read, a [(cram …)] test runs commands no marker covers. *)
let unread_head s =
  match s.role with
  | Compiles -> None
  | Runs ->
      let head = Option.value (Scan.head s.sexp) ~default:"<not a stanza>" in
      Option.some_if
        (not
           (List.exists [ running_heads; compiling_heads; inert_heads ] ~f:(fun l ->
                List.mem l head ~equal:String.equal)))
        head

(** Each backend marker of the dune file [dune_file] that [env_var_deps] refuses, as
    [<dune file>:<line>: <why>]: the markers the shared contract ({!Scan.backend_marker_contract})
    refuses -- one that does not parse, one inside no stanza, one on a stanza that runs nothing, one
    outside a comment (which has no line) -- and a second marker on one stanza. Empty when every
    marker reads. The decisions are the contract's and {!Scan.backend_rule_of}'s, the ones
    [env_var_deps] fails the tree on, so the two cannot disagree about which markers are bad. *)
let marker_refusals ~dune_file (contract : _ Scan.marker_contract) =
  let at line why = Printf.sprintf "%s:%d: %s" dune_file line why in
  List.map contract.Scan.contract_issues ~f:(function
    | Scan.Malformed_marker { issue_line; issue_malformed; _ } ->
        at issue_line
          ("a backend marker that does not parse: "
          ^ Scan.backend_marker_malformed_reason issue_malformed)
    | Scan.Marker_outside_stanza { issue_line; _ } ->
        at issue_line "a backend marker inside no stanza, which declares nothing"
    | Scan.Marker_in_wrong_stanza { issue_line; issue_why; _ } ->
        at issue_line ("a backend marker on a stanza it cannot declare for: " ^ issue_why)
    | Scan.Marker_outside_comment
        { issue_text_occurrences = in_text; issue_comment_occurrences = in_comments } ->
        Printf.sprintf "%s: spells `%s` %d times and only %d of them are in a comment" dune_file
          Scan.marker_sentinel in_text in_comments)
  @ List.filter_map contract.Scan.contract_stanzas ~f:(fun marked ->
      match Scan.backend_rule_of marked with
      | Scan.Names_twice line -> Some (at line "a second backend marker on one stanza")
      | _ -> None)

(** The stanzas of one dune file, in [dir]; or, when {!marker_refusals} refuses any of its markers,
    those refusals. A refused marker is read neither as absent nor as every backend: the caller
    refuses the batch over it (gh-ocannl-1652's follow-up). *)
let stanzas_of ~dir content =
  let contract = Scan.backend_marker_contract content in
  match marker_refusals ~dune_file:(join dir "dune") contract with
  | _ :: _ as refused -> Error refused
  | [] ->
      Ok
        (List.concat_map contract.Scan.contract_stanzas ~f:(fun marked ->
             let st = marked.Scan.marker_stanza in
             let named, reads_config = backend_of_rule (Scan.backend_rule_of marked) in
             views_of ~dir:(join dir st.Scan.marked_subdir) ~named ~reads_config st.Scan.marked_sexp))

type target =
  | Alias of { dir : string; alias : string; recursive : bool }
  | Product of { dir : string; file : string }
      (** a compilation product of a stanza in [dir]: [<name>.exe], [.bc], [.cma] or [.cmxa] *)

(** The aliases whose whole closure reads no configuration, so that a batch of them holds only what
    their stanzas name (gh-ocannl-1227). Closed and short: an entry is the alias in one directory,
    with one recursion, never a basename -- a [scans] alias elsewhere is no member. [slot_kind_live]
    checks each against the repository's tree ({!refusals}). Not [@lifecycle]: its resource probes
    ([schedule_cache_device], [alloc_census_peak], the [autotune_*] ones, ...) read the
    configuration's backend on purpose. *)
let config_free_aliases =
  [
    Alias { dir = "test/operations"; alias = "scans"; recursive = true };
    Alias { dir = ""; alias = "metal-codegen"; recursive = true };
  ]

let spell = function
  | Alias { dir; alias; recursive } ->
      Printf.sprintf "%s%s%s"
        (if recursive then "@" else "@@")
        (if String.is_empty dir then "" else dir ^ "/")
        alias
  | Product { dir; file } -> if String.is_empty dir then file else dir ^ "/" ^ file

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

(* The extensions of an executable's products, and of a library's. *)
let exe_products = [ ".exe"; ".bc" ]
let library_products = [ ".cma"; ".cmxa" ]

(* An alias target in the spellings this reads: [@alias], [@@alias], [@dir/alias], [@@dir/alias],
   with [dir] plain; or a compilation product, [<dir>/<name>.exe] and its siblings, with one leading
   [./] allowed (tools/test-run.sh bisect spells its build phase so). Anything else is
   unmodelled. *)
let target_of w =
  let alias_in ~recursive spec =
    match String.rsplit2 spec ~on:'/' with
    | None when plain_dir spec -> Some (Alias { dir = ""; alias = spec; recursive })
    | Some (dir, alias) when plain_dir dir && plain_dir alias ->
        Some (Alias { dir; alias; recursive })
    | _ -> None
  in
  let product path =
    let path = Option.value (String.chop_prefix path ~prefix:"./") ~default:path in
    let dir, file =
      match String.rsplit2 path ~on:'/' with Some (d, f) -> (d, f) | None -> ("", path)
    in
    Option.some_if
      ((String.is_empty dir || plain_dir dir)
      && plain_dir file
      && List.exists (exe_products @ library_products) ~f:(fun suffix ->
          String.is_suffix file ~suffix && String.length file > String.length suffix))
      (Product { dir; file })
  in
  match String.chop_prefix w ~prefix:"@@" with
  | Some spec -> alias_in ~recursive:false spec
  | None -> (
      match String.chop_prefix w ~prefix:"@" with
      | Some spec -> alias_in ~recursive:true spec
      | None -> product w)

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

(** Every stanza building the [aliases] builds, in the order the tree lists them: the closure of the
    aliases, every compilation and every source-like producer under {!need}s. [reached stanzas []]
    is what every build reaches -- a plain compilation's whole closure. *)
let reached stanzas aliases =
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
  List.iter aliases ~f:(fun (dir, alias, recursive) -> request ~root:dir ~alias ~recursive);
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
  | Compiles -> Printf.sprintf "it compiles %s in %s" what dir

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

(** Raises on a [dune-project] (at [path], with [content]) declaring what the closure assumes away:
    a [dialect], which can make any extension a source one where {!data_suffixes} assumes none does,
    or the alternative [dune-file] name. *)
let check_project ~path content =
  List.iter (Scan.stanzas content) ~f:(function
    | Sexp.List (Sexp.Atom (("dialect" | "accept_alternative_dune_file_name") as h) :: _) ->
        failwith (Printf.sprintf "%s declares (%s …)" path h)
    | _ -> ())

(** Raises on a [dune-workspace] (at [path], with [content]) naming more than its language: a
    context's environment can set the backend, and a second context builds the tree twice. *)
let check_workspace ~path content =
  List.iter (Scan.stanzas content) ~f:(function
    | Sexp.List (Sexp.Atom "lang" :: _) -> ()
    | other -> failwith (Printf.sprintf "%s carries %s" path (Sexp.to_string other)))

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
      check_project ~path:project (Stdio.In_channel.read_all project)
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
    if Stdlib.Sys.file_exists file then check_workspace ~path:file (Stdio.In_channel.read_all file)
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

(** Every stanza reading the configuration -- declaring [(env_var OCANNL_BACKEND)], or neither it
    nor a marker -- whose nearest configuration is not one the runner resolves (gh-ocannl-1249): a
    refusal each. Shared by {!refusals} and, failing closed, by {!answer} (Codex review round 5 on
    staging#1318). [config_files] are the repository-relative paths of the [ocannl_config] files
    present in the tree. *)
let unresolved_readers ~config_files ~stanzas dune_files =
  let paths = List.map dune_files ~f:(fun (dir, content) -> (join dir "dune", content)) in
  let report = Config_dep_resolution.analyze ~exemption_keys:[] ~config_files paths in
  (* Where each directory's copied configuration comes from. *)
  let copied_from =
    List.concat_map paths ~f:(fun (dune_file, content) ->
        let dir = Config_dep_resolution.root_normalized (Stdlib.Filename.dirname dune_file) in
        List.map (Scan.copy_sources content ~name:Scan.config_file) ~f:(fun (subdir, source) ->
            let into = Config_dep_resolution.under dir subdir in
            ( Config_dep_resolution.root_normalized into,
              Config_dep_resolution.root_normalized
                (Stdlib.Filename.dirname (Config_dep_resolution.under into source)) )))
  in
  (* The ambient copy is test/config's only while its exact producer stands, and it is the
     producer's own output rather than a copy of anything (Codex review round 4 on staging#1318). *)
  let ambient_trusted =
    List.exists stanzas ~f:(fun s ->
        Poly.equal s.role Runs && ambient_augmentation ~dir:s.dir s.sexp)
    && not (List.exists copied_from ~f:(fun (into, _) -> String.equal into ambient_config_dir))
  in
  let resolved = if ambient_trusted then ambient_config_dir :: config_dirs else config_dirs in
  (* The directories a configuration in [d] comes from, following copies of copies to where they
     start; a cycle stops where it closes. *)
  let rec origins ~seen d =
    match
      List.filter_map copied_from ~f:(fun (into, from) -> Option.some_if (String.equal into d) from)
    with
    | [] -> [ d ]
    | sources ->
        List.concat_map sources ~f:(fun from ->
            if List.mem seen from ~equal:String.equal then [ from ]
            else origins ~seen:(from :: seen) from)
  in
  (* A stanza declaring neither the variable nor a marker reads the configuration too, the widening
     way [backend_of_rule] reads it: [env_var_deps] fails it, but that may not have run yet (Codex
     review round 3 on staging#1318). *)
  let undeclared =
    List.map paths ~f:(fun (dune_file, content) ->
        ( dune_file,
          List.concat_map (Scan.backend_marker_contract content).Scan.contract_stanzas
            ~f:(fun marked ->
              match Scan.backend_rule_of marked with
              | Scan.Names_neither -> marked.Scan.marker_stanza.Scan.marked_sites
              | _ -> []) ))
  in
  List.concat_map report.Config_dep_resolution.files ~f:(fun file ->
      let neither =
        Option.value ~default:[]
          (List.Assoc.find undeclared file.Config_dep_resolution.dune_file ~equal:String.equal)
      in
      List.filter_map file.Config_dep_resolution.sites ~f:(fun r ->
          let site = r.Config_dep_resolution.site in
          if not (site.Scan.declares_backend || List.mem neither site ~equal:Poly.equal) then None
          else
            let at = r.Config_dep_resolution.directory in
            let at = if String.is_empty at then "." else at in
            let who = Printf.sprintf "the configuration: %s running in %s" site.Scan.name at in
            match (site.Scan.kind, r.Config_dep_resolution.nearest) with
            | Scan.Unreadable_directory, _ -> Some (who ^ ", in a directory this cannot establish")
            | _, None ->
                Some
                  (who
                 ^ ", with no checked-in configuration in reach: it reads the gitignored root one, \
                    or none")
            | _, Some nearest -> (
                let sources = origins ~seen:[ nearest ] nearest in
                match
                  List.find sources ~f:(fun d -> not (List.mem resolved d ~equal:String.equal))
                with
                | None -> None
                | Some d ->
                    Some
                      (Printf.sprintf
                         "%s, which reads the configuration in %s: not one the runner resolves" who
                         (if String.is_empty d then "." else d)))))

(** What a dune argv can hold. [Reaches] lists in [named] each backend a reached stanza's marker
    names, once, with the first stanza that names it, and gives in [reads_config] why the
    configuration counts -- [None] when every target is on {!config_free_aliases} or a plain
    compilation product, or when it builds nothing. [Unknown why] is an argv this does not model or
    a dune file it could not read, which the caller takes as every backend. [Refused refusals] is a
    tree carrying a backend marker [env_var_deps] refuses, each [<dune file>:<line>: <why>]: the
    caller refuses the batch rather than guess what the marker meant. *)
type answer =
  | Reaches of { named : (string * string) list; reads_config : string option }
  | Unknown of string
  | Refused of string list

(* The environment variables that run a program or add what is built: dune's, as their options do,
   and the compiler's [OCAMLPARAM], which can name a preprocessor or a ppx. *)
let unmodelled_dune_env = [ "DUNE_DIFF_COMMAND"; "DUNE_INSTRUMENT_WITH"; "DUNE_ROOT"; "OCAMLPARAM" ]

(* Dune's own aliases this models: [runtest] through the stanzas it attaches, [default] and [all] as
   every target, and the ones that build what every batch's compilations already cover. *)
let builtin_aliases =
  [ "runtest"; "default"; "all"; "check"; "install"; "doc"; "doc-private"; "fmt"; "lint" ]

(** Every dune file's stanzas; or [`Refused] with every refused marker of the tree, so that one
    refusal names them all; or [`Unreadable] for the first file that could not be read. *)
let read_stanzas dune_files =
  List.fold_result dune_files ~init:([], []) ~f:(fun (acc, refused) (dir, content) ->
      match stanzas_of ~dir content with
      | Ok stanzas -> Ok (stanzas :: acc, refused)
      | Error r -> Ok (acc, refused @ r)
      | exception exn ->
          Error
            (`Unreadable
               (Printf.sprintf "the dune file in %s is unreadable here (%s)"
                  (if String.is_empty dir then "." else dir)
                  (Exn.to_string exn))))
  |> Result.bind ~f:(function
    | l, [] -> Ok (List.concat (List.rev l))
    | _, refused -> Error (`Refused refused))

(** The stanza in [dir] declaring the compilation product [file], if any: an executable or test for
    [.exe]/[.bc], a library for [.cma]/[.cmxa]. *)
let declares_product stanzas ~dir ~file =
  let declared heads suffixes =
    List.exists suffixes ~f:(fun suffix ->
        match String.chop_suffix file ~suffix with
        | None -> false
        | Some name ->
            List.exists stanzas ~f:(fun s ->
                Poly.equal s.role Runs && String.equal s.dir dir
                && List.mem heads (Option.value (Scan.head s.sexp) ~default:"") ~equal:String.equal
                && List.mem (Scan.names_of s.sexp) name ~equal:String.equal))
  in
  declared [ "executable"; "executables"; "test"; "tests" ] exe_products
  || declared [ "library" ] library_products

let answer ?(getenv = Stdlib.Sys.getenv_opt) ?(config_free = config_free_aliases) ?config_files
    ~dune_files argv =
  let build_dir = Option.value (getenv "DUNE_BUILD_DIR") ~default:"" in
  (* With the build directory moved to a relative [out], [@out/default/runtest] is a context root,
     not a source directory (Codex review on PR #1027). *)
  let build_root =
    (* Normalized first ([./out] is [out]); an absolute one, or one leaving the tree, roots no alias
       here. *)
    if String.is_empty build_dir || not (Stdlib.Filename.is_relative build_dir) then None
    else Option.bind (resolve ~dir:"" build_dir) ~f:(fun d -> List.hd (String.split d ~on:'/'))
  in
  let in_build_root dir =
    match build_root with
    | Some b -> String.equal (List.hd_exn (String.split dir ~on:'/')) b
    | None -> false
  in
  let targets argv =
    Result.bind (targets argv) ~f:(function
      | None -> Ok None
      | Some ts -> (
          match
            List.find ts ~f:(function Alias { dir; _ } | Product { dir; _ } -> in_build_root dir)
          with
          | Some t -> Error (spell t)
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
          match read_stanzas dune_files with
          | Error (`Unreadable why) -> Unknown why
          | Error (`Refused refused) -> Refused refused
          | Ok stanzas when List.exists stanzas ~f:(fun s -> Option.is_some (unread_head s)) ->
              (* A hole in the inventory is every batch's: what an [(include …)] brings in may sit
                 on any alias (Codex review round 3 on staging#1318). *)
              let s = List.find_exn stanzas ~f:(fun s -> Option.is_some (unread_head s)) in
              Unknown
                (Printf.sprintf
                   "the dune file in %s carries a (%s …) stanza, which this does not read"
                   (if String.is_empty s.dir then "." else s.dir)
                   (Option.value_exn (unread_head s)))
          | Ok stanzas
            when Option.exists config_files ~f:(fun config_files ->
                     not (List.is_empty (unresolved_readers ~config_files ~stanzas dune_files))) ->
              (* A reader of a configuration the runner does not resolve: what it selects is unread,
                 so every batch is every backend until the tree is fixed -- the scan refuses it, but
                 may not have run yet (Codex review round 5 on staging#1318). *)
              Unknown
                (List.hd_exn
                   (unresolved_readers ~config_files:(Option.value_exn config_files) ~stanzas
                      dune_files))
          | Ok stanzas -> (
              (* An alias the argv names that no stanza in its scope carries is one of dune's own,
                 and only the ones listed are modelled ([revdep-runtest] builds what depends on the
                 directory; Codex review on PR #1027). A product no stanza declares may be a rule's
                 target, which runs whatever produces it. *)
              let unmodelled =
                List.find_map targets ~f:(function
                  | Alias { dir = root; alias; recursive } as t ->
                      Option.some_if
                        ((not (List.mem builtin_aliases alias ~equal:String.equal))
                        && not
                             (List.exists stanzas ~f:(fun s ->
                                  in_scope ~recursive ~root s.dir
                                  && List.mem s.attached alias ~equal:String.equal)))
                        (Printf.sprintf "no stanza carries %s, and it is no dune alias this models"
                           (spell t))
                  | Product { dir; file } as t ->
                      Option.some_if
                        (not (declares_product stanzas ~dir ~file))
                        (Printf.sprintf
                           "it builds %s, which no executable, test or library stanza in %s \
                            declares (a rule could produce it)"
                           (spell t)
                           (if String.is_empty dir then "." else dir)))
              in
              let found =
                lazy
                  (reached stanzas
                     (List.filter_map targets ~f:(function
                       | Alias { dir; alias; recursive } -> Some (dir, alias, recursive)
                       | Product _ -> None)))
              in
              (* A dependency the closure cannot follow leaves what it builds unread: its named
                 backends included, so counting the configuration does not cover it (Codex review
                 round 1 on staging#1318). *)
              let unmodelled =
                match unmodelled with
                | Some _ -> unmodelled
                | None ->
                    List.find_map (Lazy.force found) ~f:(fun s ->
                        Option.map (List.hd s.unplaced) ~f:(fun form ->
                            Printf.sprintf "%s, whose dependency %s this cannot follow" (describe s)
                              form))
              in
              match unmodelled with
              | Some why -> Unknown why
              | None ->
                  let found = Lazy.force found in
                  let named =
                    List.concat_map found ~f:(fun s ->
                        let why =
                          Printf.sprintf "%s, which names %s" (describe s)
                            (String.concat ~sep:"," s.named)
                        in
                        List.map s.named ~f:(fun b -> (b, why)))
                    |> List.fold ~init:[] ~f:(fun acc (b, why) ->
                        if List.Assoc.mem acc b ~equal:String.equal then acc else (b, why) :: acc)
                    |> List.rev
                  in
                  (* The policy first; then, failing closed, the closure itself: a reader reached by
                     a configuration-free batch is one [slot_kind_live] would refuse, but that scan
                     may not have run on this checkout yet, so the configuration counts here and now
                     (Codex review round 2 on staging#1318). *)
                  let reads_config =
                    match
                      List.find_map targets ~f:(function
                        | Product _ -> None
                        | Alias _ as t ->
                            Option.some_if
                              (not (List.mem config_free t ~equal:Poly.equal))
                              (Printf.sprintf
                                 "it builds %s, which is not on the configuration-free alias list"
                                 (spell t)))
                    with
                    | Some why -> Some why
                    | None ->
                        List.find_map found ~f:(fun s ->
                            Option.some_if s.reads_config
                              (Printf.sprintf
                                 "%s, which reads the configuration (a configuration-free batch \
                                  should not: slot_kind_live refuses it)"
                                 (describe s)))
                  in
                  Reaches { named; reads_config })))

(** The [ocannl_config] files checked into the tree [dune_files] was read from, as the run-time
    resolver hands them to {!answer}: the one in each directory holding a dune file, or above one,
    below [root]. A configuration only a [chdir] reaches, in a directory with no dune file at or
    below it, is not seen, so a reader there reads as having nothing in reach: the loud direction.
*)
let config_files_of ~root dune_files =
  let rec ancestors dir =
    if String.is_empty dir then []
    else dir :: (match String.rsplit2 dir ~on:'/' with Some (up, _) -> ancestors up | None -> [])
  in
  List.concat_map dune_files ~f:(fun (dir, _) -> ancestors dir)
  |> List.dedup_and_sort ~compare:String.compare
  |> List.filter_map ~f:(fun dir ->
      let path = join dir Scan.config_file in
      Option.some_if (Stdlib.Sys.file_exists (Stdlib.Filename.concat root path)) path)

(** Whether [answer] can hold a GPU by name: an unknown answer can. Whether a configuration that
    counts names one is the caller's question. *)
let holds_gpu = function
  | Unknown _ | Refused _ -> true
  | Reaches { named; _ } ->
      List.exists named ~f:(fun (b, _) -> List.mem gpu_backends b ~equal:String.equal)

(** [answer] on one line, as the slot_kind tests' goldens record it: [names <backend>,…] in
    [named]'s order, or [names nothing], then [ + reads config] when the configuration counts; or
    [unknown: <why>], or [refused: <refusal>; …]. *)
let summary = function
  | Unknown why -> "unknown: " ^ why
  | Refused refused -> "refused: " ^ String.concat ~sep:"; " refused
  | Reaches { named; reads_config } ->
      (match named with
        | [] -> "names nothing"
        | _ -> "names " ^ String.concat ~sep:"," (List.map named ~f:fst))
      ^ if Option.is_some reads_config then " + reads config" else ""

(** {1 Keeping the list honest}

    What [slot_kind_live] checks of the repository's tree, so that the two configuration-free
    answers above stay true (gh-ocannl-1227, gh-ocannl-1249). Each refusal names what failed and
    where; an empty list is a tree the answers hold for. The checks read markers over closures and
    never analyse an action: a backend an action fixes is its marker's to name, which [env_var_deps]
    enforces (gh-ocannl-1518).

    - Each listed alias's closure holds no stanza reading the configuration, and no alias dependency
      whose path the closure cannot follow -- a missing alias is refused too.
    - What every build reaches (every compilation, every source-like producer, and what they need)
      holds no reader and names no backend: a plain compilation runs nothing that opens a device.
    - Every stanza reading the configuration runs where the configuration it finds is one the runner
      resolves: in {!config_dirs}, or copied from one of them or from {!ambient_config_dir} -- whose
      producer has exactly {!ambient_augmentation}'s shape, and no other rule anywhere writes an
      [ocannl_config]. A reader with no configuration in reach of the checked-in tree would read the
      gitignored root one (gh-ocannl-1249).
    - The inventory is complete: no stanza head outside the modelled ones (an [include] brings in
      stanzas never read, a [cram] test runs commands no marker covers) and no [dynamic-run].

    [config_files] are the repository-relative paths of the [ocannl_config] files in the tree, as
    {!Config_dep_resolution.analyze} reads them. *)
let refusals ?(config_free = config_free_aliases) ~config_files dune_files =
  match read_stanzas dune_files with
  | Error (`Unreadable why) -> [ why ]
  | Error (`Refused refused) -> refused
  | Ok stanzas ->
      let closure_refusals ~what found =
        List.concat_map found ~f:(fun s ->
            (if s.reads_config then
               [ Printf.sprintf "%s: %s, which reads the configuration" what (describe s) ]
             else [])
            @ List.map s.unplaced ~f:(fun form ->
                Printf.sprintf "%s: %s, whose dependency %s this cannot follow" what (describe s)
                  form))
      in
      let listed =
        List.concat_map config_free ~f:(function
          | Product _ as t -> [ Printf.sprintf "%s is no alias" (spell t) ]
          | Alias { dir = root; alias; recursive } as t ->
              let what = Printf.sprintf "the listed %s" (spell t) in
              if
                not
                  (List.exists stanzas ~f:(fun s ->
                       in_scope ~recursive ~root s.dir
                       && List.mem s.attached alias ~equal:String.equal))
              then [ what ^ ": no stanza carries it" ]
              else closure_refusals ~what (reached stanzas [ (root, alias, recursive) ]))
      in
      let plain =
        let what = "a plain compilation" in
        let found = reached stanzas [] in
        closure_refusals ~what found
        @ List.filter_map found ~f:(fun s ->
            Option.some_if
              (not (List.is_empty s.named))
              (Printf.sprintf "%s: %s, which names %s" what (describe s)
                 (String.concat ~sep:"," s.named)))
      in
      let inventory =
        List.filter_map stanzas ~f:(fun s ->
            match s.role with
            | Compiles -> None
            | Runs -> (
                let in_dir = if String.is_empty s.dir then "." else s.dir in
                match unread_head s with
                | Some head ->
                    Some
                      (Printf.sprintf "the inventory: a (%s …) stanza in %s, which it does not read"
                         head in_dir)
                | None ->
                    if List.mem (Scan.atoms s.sexp) "dynamic-run" ~equal:String.equal then
                      Some
                        (Printf.sprintf "the inventory: %s, whose dynamic-run it cannot follow"
                           (describe s))
                    else None))
      in
      let generated =
        List.filter_map stanzas ~f:(fun s ->
            Option.some_if
              (Poly.equal s.role Runs
              && (not
                    (String.is_prefix
                       (Option.value (Scan.head s.sexp) ~default:"")
                       ~prefix:"copy_files"))
              && List.exists s.targets ~f:(fun t -> Scan.glob_could_match t ~name:Scan.config_file)
              && not (ambient_augmentation ~dir:s.dir s.sexp))
              (Printf.sprintf "the configuration: %s, which can write an ocannl_config" (describe s)))
      in
      let readers = unresolved_readers ~config_files ~stanzas dune_files in
      listed @ plain @ inventory @ generated @ readers
