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
      An alias reaches what its stanzas attach to it, closed under the [(alias …)] its [deps] name
      ({!Dune_stanza_scan.aliases_reached_from}), plus the [runtest-<name>] dune generates for a
      [(test)]/[(tests)] stanza and an inline-test library. No target at all is the default alias
      everywhere, taken as reaching every stanza.
    - [runtest]/[test] with plain directories runs [@runtest] under each, and with none under the
      root.
    - Options only from the listed harmless ones, and no [--].
    - Any other subcommand runs no test and reaches nothing. [exec] is answered before this is asked
      (its program may pick any backend), and so is a command-line backend flag.

    A dune file this cannot read is itself an [Unknown] answer, for the same reason. *)

open Base
module Scan = Dune_stanza_scan

let gpu_backends = [ "cuda"; "hip"; "metal" ]

type stanza = {
  dir : string;  (** the directory dune applies it in, repository-relative, [""] for the root *)
  attached : string list;  (** the aliases it attaches to or defines *)
  sexp : Sexplib.Sexp.t;
  named : string list;  (** the backends its marker names, [none] left out *)
  reads_config : bool;  (** whether it selects its backend from the configuration *)
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
      {
        dir = join dir st.Scan.marked_subdir;
        attached = attached_aliases sexp;
        sexp;
        named;
        reads_config;
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

(** Every stanza [target] reaches that names a backend or reads the configuration, in directory
    order. *)
let reached stanzas target =
  let by_dir = List.sort_and_group stanzas ~compare:(fun a b -> String.compare a.dir b.dir) in
  List.concat_map by_dir ~f:(fun group ->
      let dir = (List.hd_exn group).dir in
      match target with
      | Alias { dir = root; alias; recursive } ->
          if not (in_scope ~recursive ~root dir) then []
          else
            let sexps = List.map group ~f:(fun s -> s.sexp) in
            (* `default` builds every target in the directory rather than an alias's members, so it
               reaches every stanza there; any other alias reaches its closure. *)
            let reached =
              if String.equal alias "default" then
                Set.of_list (module String) (List.concat_map group ~f:(fun s -> s.attached))
              else Scan.aliases_reached_from sexps alias
            in
            List.filter group ~f:(fun s ->
                ((not (List.is_empty s.named)) || s.reads_config)
                && List.exists s.attached ~f:(Set.mem reached)))

(** Where a reached stanza is, for a reason: its names (or a rule's aliases) and its directory. *)
let describe s =
  let what =
    match Scan.names_of s.sexp with
    | [] -> "the rule on " ^ String.concat ~sep:"," (Scan.aliases_of s.sexp)
    | names -> String.concat ~sep:"," names
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
      | Ok stanzas ->
          let stanzas = List.concat stanzas in
          let found = List.concat_map targets ~f:(reached stanzas) in
          let named =
            List.concat_map found ~f:(fun s ->
                let why =
                  Printf.sprintf "%s, which names %s" (describe s) (String.concat ~sep:"," s.named)
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
          Reaches { named; reads_config })

(** Whether [answer] can hold a GPU by name: an unknown answer can. Whether a configuration a
    reached stanza reads names one is the caller's question. *)
let holds_gpu = function
  | Unknown _ -> true
  | Reaches { named; _ } ->
      List.exists named ~f:(fun (b, _) -> List.mem gpu_backends b ~equal:String.equal)
