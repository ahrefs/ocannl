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

    The configuration's own backend counts only when some reached stanza reads it: one declaring
    [(env_var OCANNL_BACKEND)], or one carrying neither declaration (which [env_var_deps] fails, and
    which is read the way it would run: from the configuration). A run reaching only stanzas that
    name theirs -- [; ocannl-backend: none] ones included, as [@test/operations/scans] does -- holds
    only what they name, whatever the configurations say (gh-ocannl-1095). A stanza naming a backend
    twice is not read as naming none: its file is unreadable, which is [Unknown].

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
    - Any other subcommand runs no test and reaches nothing. [exec] is answered before this is asked
      (its program may pick any backend), and so is a command-line backend flag.

    A dune file this cannot read is itself an [Unknown] answer, for the same reason. *)

open Base
module Scan = Dune_stanza_scan

let gpu_backends = [ "cuda"; "hip"; "metal" ]

(** What building a stanza builds first (gh-ocannl-1095): the aliases its dependency fields name,
    and the files it mentions, which the rules producing them must build. *)
type need =
  | Alias_need of { dir : string; alias : string; recursive : bool }
  | File_need of string  (** a file's basename *)
  | Glob_need of string  (** a basename pattern, in [*] and [?] *)

type stanza = {
  dir : string;  (** the directory dune applies it in, repository-relative, [""] for the root *)
  attached : string list;  (** the aliases it attaches to or defines *)
  sexp : Sexplib.Sexp.t;
  named : string list;  (** the backends its marker names, [none] left out *)
  reads_config : bool;  (** whether it selects its backend from the configuration *)
  targets : string list;  (** the basenames (as patterns) of the files a rule produces *)
  source_like : bool;
      (** whether one of them is a file dune may build to compile or load anything *)
  needs : need list;
  opaque : string option;  (** a way it builds something first that this does not follow *)
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

(** {1 What building a stanza builds first}

    Taking the configuration's backend off a batch that reaches no configuration reader is only as
    sound as the reached set is complete (Codex GPT-6.1 Sol review on PR #1027): a diff alias
    consumes an [.actual] whose producing rule reads the configuration, an alias's deps name an
    alias in another directory, a test's generated [runtest-<name>] carries deps of its own. So the
    reached set is the closure of the argv's aliases under every way dune builds one thing before
    another:
    - an [(alias …)]/[(alias_rec …)] in a dependency field (a stanza's [deps], an inline-test
      library's [(inline_tests (deps …))]), resolved against the stanza's directory, through the
      same attached-alias inventory the seeds use, generated per-test aliases included;
    - a file: every atom anywhere in the stanza, and the payload of every path pform ([%{dep:…}],
      [%{read:…}], …), taken as a file it may need -- matched by BASENAME against every rule's
      targets in any directory, a pform elsewhere in the atom read as a wildcard. Over-wide on
      purpose: a copy ([copy_files], [(copy …)]) keeps the basename, so this follows it to the
      original producer without modelling it, and a word that names no target matches nothing;
    - a rule producing a source-like file (an OCaml or C source, a header, a dune include, a sexp)
      is taken as always built: compiling or loading anything may need it, and which compilation
      does is not read here.

    A dependency form this does not read -- [(package …)], [(include …)], an alias path leaving the
    tree or carrying a pform, any other head -- and a [dynamic-run] make a reached stanza OPAQUE,
    which answers [Unknown]: every backend. *)

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

let file_needs sexp =
  List.concat_map (Scan.atoms sexp) ~f:(fun atom ->
      let payloads =
        List.filter_map (Scan.pieces atom) ~f:(function
          | Scan.Pform p -> (
              match String.lsplit2 p ~on:':' with
              | Some (k, v) when List.mem path_pforms k ~equal:String.equal -> Some v
              | _ -> None)
          | Scan.Literal _ -> None)
      in
      List.filter_map (atom :: payloads) ~f:(fun a ->
          Option.map (pattern_of a) ~f:(fun b ->
              if String.exists b ~f:is_wild then Glob_need b else File_need b)))

(** The aliases a stanza's dependency fields name, or the first dependency form this does not read.
*)
let alias_needs ~dir sexp =
  let fields =
    Option.to_list (Scan.field sexp "deps")
    @
    match Scan.field sexp "inline_tests" with
    | Some args -> Option.to_list (Scan.field_in args "deps")
    | None -> []
  in
  let rec item = function
    | Sexp.Atom _ -> Ok []
    | Sexp.List (Sexp.Atom h :: rest) when String.is_prefix h ~prefix:":" ->
        Result.map (Result.all (List.map rest ~f:item)) ~f:List.concat
    | Sexp.List [ Sexp.Atom (("alias" | "alias_rec") as h); Sexp.Atom spec ] -> (
        match resolve ~dir spec with
        | Some p ->
            let dir, alias = Option.value (String.rsplit2 p ~on:'/') ~default:("", p) in
            Ok [ Alias_need { dir; alias; recursive = String.equal h "alias_rec" } ]
        | None -> Error (Printf.sprintf "(%s %s)" h spec))
    | Sexp.List
        (Sexp.Atom
           ( "file" | "glob_files" | "glob_files_rec" | "source_tree" | "env_var" | "universe"
           | "sandbox" )
        :: _) ->
        Ok []
    | other -> Error (Sexp.to_string other)
  in
  Result.map (Result.all (List.concat_map fields ~f:(List.map ~f:item))) ~f:List.concat

(** The basenames, as patterns, of the files a rule produces: its [(target …)]/[(targets …)], or,
    without them and without an alias, every atom it carries -- dune infers such a rule's targets
    from its action, and reading every atom is the wide side of that inference. *)
let targets_of sexp =
  match Scan.head sexp with
  | Some "rule" -> (
      match (Scan.field sexp "targets", Scan.field sexp "target") with
      | Some args, _ | None, Some args ->
          List.filter_map (List.concat_map args ~f:Scan.atoms) ~f:pattern_of
      | None, None ->
          if List.is_empty (Scan.aliases_of sexp) then
            List.filter_map (Scan.atoms sexp) ~f:pattern_of
          else [])
  | _ -> []

let source_suffixes =
  [
    ".ml";
    ".mli";
    ".mll";
    ".mly";
    ".c";
    ".h";
    ".cc";
    ".cpp";
    ".cxx";
    ".hpp";
    ".s";
    ".S";
    ".inc";
    ".sexp";
  ]

let source_like target =
  String.exists target ~f:is_wild
  || List.exists source_suffixes ~f:(fun suffix -> String.is_suffix target ~suffix)

(** The stanzas of one dune file, in [dir]. A marker the contract refuses is not read as absent: it
    raises, and the caller takes the unreadable file as a GPU answer. *)
let stanzas_of ~dir content =
  let contract = Scan.backend_marker_contract content in
  if not (List.is_empty contract.Scan.contract_issues) then
    failwith "a backend marker the env_var_deps contract refuses";
  List.map contract.Scan.contract_stanzas ~f:(fun marked ->
      let st = marked.Scan.marker_stanza in
      let sexp = st.Scan.marked_sexp in
      let named, reads_config = backend_of_rule (Scan.backend_rule_of marked) in
      let dir = join dir st.Scan.marked_subdir in
      let targets = targets_of sexp in
      let aliases, opaque =
        match alias_needs ~dir sexp with
        | Ok needs -> (needs, None)
        | Error form -> ([], Some (Printf.sprintf "dependency %s" form))
      in
      let opaque =
        if Option.is_none opaque && List.mem (Scan.atoms sexp) "dynamic-run" ~equal:String.equal
        then Some "dynamic-run action"
        else opaque
      in
      {
        dir;
        attached = attached_aliases sexp;
        sexp;
        named;
        reads_config;
        targets;
        source_like = List.exists targets ~f:source_like;
        needs = List.dedup_and_sort (aliases @ file_needs sexp) ~compare:Poly.compare;
        opaque;
      })

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
    "--build-dir";
    "--sandbox";
    "--diff-command";
    "--error-reporting";
    "--action-stdout-on-success";
    "--action-stderr-on-success";
    "--file-watcher";
    "--terminal-persistence";
    "--trace-file";
    "--dump-gc-stats";
    "--watch-exclusions";
    "--instrument-with";
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
   4 on PR #803): normalising is exactly where a spelling dune reads one way gets read another. *)
let plain_dir d =
  (not (String.is_empty d))
  && List.for_all (String.split d ~on:'/') ~f:(fun c ->
      (not (String.is_empty c)) && (not (String.equal c ".")) && not (String.equal c ".."))

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

(** The targets a dune argv builds: [Ok None] for a subcommand that runs no test, [Error word] for
    one carrying a word this does not model. *)
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
  | _ -> Ok None

let in_scope ~recursive ~root dir =
  String.equal root dir
  || (recursive && (String.is_empty root || String.is_prefix dir ~prefix:(root ^ "/")))

(** Every stanza building [targets] builds, in the order the tree lists them: the closure of the
    argv's aliases under {!need}s, seeded with every source-like producer. *)
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
    if not (Hash_set.mem requested (root, alias, recursive)) then (
      Hash_set.add requested (root, alias, recursive);
      Array.iteri arr ~f:(fun i s ->
          if in_scope ~recursive ~root s.dir then
            (* `default` builds every target in the directory rather than an alias's members. *)
            if String.equal alias "default" then (
              if not (List.is_empty s.attached && List.is_empty s.targets) then add i)
            else if List.mem s.attached alias ~equal:String.equal then add i))
  in
  let produce matches = List.iter producers ~f:(fun (i, t) -> if matches t then add i) in
  List.iter targets ~f:(fun (Alias { dir; alias; recursive }) ->
      request ~root:dir ~alias ~recursive);
  Array.iteri arr ~f:(fun i s -> if s.source_like then add i);
  while not (Queue.is_empty queue) do
    List.iter arr.(Queue.dequeue_exn queue).needs ~f:(function
      | Alias_need { dir; alias; recursive } -> request ~root:dir ~alias ~recursive
      | File_need f -> produce (fun t -> Scan.glob_could_match t ~name:f)
      | Glob_need g ->
          produce (fun t -> String.exists t ~f:is_wild || Scan.glob_could_match g ~name:t))
  done;
  List.filteri stanzas ~f:(fun i _ -> seen.(i))

(** Where a reached stanza is, for a reason: its names (or a rule's aliases) and its directory. *)
let describe s =
  let what =
    match (Scan.names_of s.sexp, Scan.aliases_of s.sexp) with
    | [], [] -> "the rule producing " ^ String.concat ~sep:"," s.targets
    | [], aliases -> "the rule on " ^ String.concat ~sep:"," aliases
    | names, _ -> String.concat ~sep:"," names
  in
  Printf.sprintf "it reaches %s in %s" what (if String.is_empty s.dir then "." else s.dir)

(** What a dune argv can hold. [Reaches] lists in [named] each backend a reached stanza's marker
    names, once, with the first stanza that names it, and gives in [reads_config] the first reached
    stanza that selects its backend from the configuration, if any -- both empty when the run
    reaches no such stanza, or runs no test. [Unknown why] is an argv this does not model or a dune
    file it could not read, which the caller takes as every backend. *)
type answer =
  | Reaches of { named : (string * string) list; reads_config : string option }
  | Unknown of string

let answer ~dune_files argv =
  match targets argv with
  | Error opt ->
      Unknown
        (Printf.sprintf "it carries `%s`, which this does not model (it could change what is built)"
           opt)
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
          let found = reached (List.concat (List.rev stanzas)) targets in
          match List.find found ~f:(fun s -> Option.is_some s.opaque) with
          | Some s ->
              Unknown
                (Printf.sprintf "%s, whose %s this does not follow" (describe s)
                   (Option.value_exn s.opaque))
          | None ->
              let found =
                List.filter found ~f:(fun s -> (not (List.is_empty s.named)) || s.reads_config)
              in
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
              let reads_config =
                List.find_map found ~f:(fun s ->
                    Option.some_if s.reads_config (describe s ^ ", which reads the configuration"))
              in
              Reaches { named; reads_config }))

(** Whether [answer] can hold a GPU by name: an unknown answer can. Whether a configuration a
    reached stanza reads names one is the caller's question. *)
let holds_gpu = function
  | Unknown _ -> true
  | Reaches { named; _ } ->
      List.exists named ~f:(fun (b, _) -> List.mem gpu_backends b ~equal:String.equal)
