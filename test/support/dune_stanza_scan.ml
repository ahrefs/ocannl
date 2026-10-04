(** Reading dune files for the places that run a test executable.

    A test executable resolves its configuration by walking UP from its working directory, which
    under dune is [_build/default/<test dir>] — so it reads the [ocannl_config] that
    [(copy_files ../config/ocannl_config)] materialized there, but only if the stanza that runs it
    DEPENDS on that file. Nothing is sandboxed, so a stanza that omits the dep does not fail: it
    reads whatever happens to be in the directory when it runs, which makes both the run and any
    [.exe.output] probe order-dependent (gh-ocannl-586, and gh-ocannl-597 for why a convention in
    prose was not enough).

    This module answers, for one dune file, the two questions that check has to ask: which of its
    stanzas run a test executable, and whether each of them declares the dependency.

    {1 Reading dune files as s-expressions}

    Dune's syntax is s-expressions with [;] line comments and quoted strings, which is what sexplib
    reads — so this parses rather than splitting on parens. The lesson is {!Config_key_scan}'s: an
    approximation of a grammar has no natural stopping point, and every mistake here is silent in
    the same way, since a stanza the scan fails to recognise looks exactly like a stanza that does
    not exist.

    Sexplib is not dune's own reader, and the two disagree in exactly two places, both handled:

    - [#|…|#] and [#;] are comments to sexplib and ordinary atom characters to dune. A file
      containing either would be read with a hole in it, so {!stanzas} refuses it instead.
    - Dune's multi-line string blocks (a quoted string opening with a backslash and a bar) use an
      escape sexplib rejects, so such a file raises rather than being misread. No dune file in this
      repository uses one; if one appears, the failure names the file. *)

open Base

let config_file = "ocannl_config"

(* Dune runs a scanning rule from the rule's own directory inside the build tree and hands it paths
   relative to that directory. [%{workspace_root}] arrives as the way back out ("../.."), so the
   number of its components says how many of the working directory's trailing components name the
   rule's directory -- which turns those paths into repository-relative ones without a scan having
   to assume what the build directory is called. Shared by the checks that read dune files, so that
   two of them cannot disagree about what a path names. *)
let split_path path = String.split_on_chars path ~on:[ '/'; '\\' ]

let base_dir workspace_root =
  let depth = List.count (split_path workspace_root) ~f:(String.equal "..") in
  let cwd = List.filter (split_path (Stdlib.Sys.getcwd ())) ~f:(Fn.non String.is_empty) in
  List.drop cwd (max 0 (List.length cwd - depth))

let repo_relative base path =
  let components =
    List.fold
      (base @ split_path path)
      ~init:[]
      ~f:(fun acc component ->
        match component with
        | "" | "." -> acc
        | ".." -> ( match acc with _ :: rest -> rest | [] -> [])
        | component -> component :: acc)
  in
  String.concat ~sep:"/" (List.rev components)

(** [in_subdir parent child] joins two relative directories, either of which may be empty or [.] —
    [(chdir . …)] and a plain [./] say "here", and saying it must not make a directory look like a
    different one. *)
let in_subdir parent child =
  let clean d =
    if String.is_empty d || String.equal d "." then ""
    else Option.value (String.chop_prefix d ~prefix:"./") ~default:d
  in
  match (clean parent, clean child) with
  | "", child -> child
  | parent, "" -> parent
  | parent, child -> parent ^ "/" ^ child

type comment = { comment_start : int; comment_text : string }
(** A [;] line comment, at the offset of its [;] and with everything after it up to the end of the
    line. Comments are what sexplib throws away, and gh-ocannl-659 put a machine-checked declaration
    in one, so this scan has to keep them. *)

type raw_form = {
  raw_start : int;  (** offset of the form's first character *)
  raw_stop : int;  (** one past its last, so a list's range includes its closing parenthesis *)
  raw_atom : string option;  (** the atom as the source spells it; [None] for a list *)
  raw_quoted : bool;  (** whether that atom arrived in quotes, and so may carry escapes *)
  raw_children : raw_form list;
}
(** A form and where it sits, which is what turns "this comment is inside that stanza" into a
    question the file's own structure answers. *)

(* The raw reader: every form and every comment, by nothing but parentheses, quotes and tokens.

   This is a CROSS-CHECK, not a second parser. What it produces is compared against sexplib's tree
   SHAPE FOR SHAPE, so the two readers cannot quietly disagree; an earlier version compared only how
   many forms there are in total, which was already enough to catch the disagreement that matters
   and is strictly weaker than comparing the trees.

   The disagreement it exists to catch is `#|…|#` and `#;`, which sexplib reads as comments and dune
   does not, so sexplib could swallow a whole stanza: `(progn (echo #|) (run ./probe.exe) (echo
   |#))` is one stanza to both readers while sexplib drops the middle of it, and the executable
   action with it (Codex P2, round 17 of PR #343). An even earlier version refused any file
   CONTAINING those two characters, which is wrong in the ordinary case -- inside a quoted argument
   or after a `;`, sexplib does not treat them as comments either, and refusing there would take the
   whole suite down over an unrelated string (round 12).

   Positions are the reason this reader now returns a tree rather than a number. gh-ocannl-659's
   marker is a comment INSIDE a stanza's parentheses, and containment is the one attribution rule
   that no whitespace convention can defeat: this repository's dune files habitually separate a
   comment block from the stanza below it with a blank line, so "the comment above the stanza" would
   have to guess how far above, and would hand a marker to the wrong stanza the first time someone
   left a note between two rules. *)
let read_raw content =
  let length = String.length content in
  let comments = ref [] in
  let pos = ref 0 in
  let delimiter c = Char.is_whitespace c || List.mem [ '('; ')'; ';'; '"' ] c ~equal:Char.equal in
  let rec skip_trivia () =
    if !pos < length then
      match content.[!pos] with
      | ';' ->
          let start = !pos in
          let stop = ref (start + 1) in
          while !stop < length && not (Char.equal content.[!stop] '\n') do
            Int.incr stop
          done;
          comments :=
            {
              comment_start = start;
              comment_text = String.sub content ~pos:(start + 1) ~len:(!stop - start - 1);
            }
            :: !comments;
          pos := !stop;
          skip_trivia ()
      | c when Char.is_whitespace c ->
          Int.incr pos;
          skip_trivia ()
      | _ -> ()
  in
  let rec form () =
    skip_trivia ();
    if !pos >= length then None
    else
      match content.[!pos] with
      | ')' -> None
      | '(' ->
          let start = !pos in
          Int.incr pos;
          let children = forms () in
          (* An unbalanced file is refused below rather than patched over here. *)
          if !pos < length && Char.equal content.[!pos] ')' then Int.incr pos;
          Some
            {
              raw_start = start;
              raw_stop = !pos;
              raw_atom = None;
              raw_quoted = false;
              raw_children = children;
            }
      | '"' ->
          let start = !pos in
          Int.incr pos;
          let closed = ref false in
          while (not !closed) && !pos < length do
            (match content.[!pos] with '\\' -> Int.incr pos | '"' -> closed := true | _ -> ());
            Int.incr pos
          done;
          Some
            {
              raw_start = start;
              raw_stop = !pos;
              raw_atom = Some (String.sub content ~pos:start ~len:(!pos - start));
              raw_quoted = true;
              raw_children = [];
            }
      | _ ->
          (* A bare atom is a form too, at the top level as much as inside a stanza: dune would
             reject one there and sexplib happily returns it, so the two still agree and the caller
             reports it. *)
          let start = !pos in
          while !pos < length && not (delimiter content.[!pos]) do
            Int.incr pos
          done;
          Some
            {
              raw_start = start;
              raw_stop = !pos;
              raw_atom = Some (String.sub content ~pos:start ~len:(!pos - start));
              raw_quoted = false;
              raw_children = [];
            }
  and forms () = match form () with None -> [] | Some f -> f :: forms () in
  let top = forms () in
  skip_trivia ();
  if !pos < length then
    failwith
      (Printf.sprintf "dune file has a stray `%c` at offset %d -- it is not balanced" content.[!pos]
         !pos);
  (top, List.rev !comments)

(* Shape for shape, and text for text wherever the text is unambiguous. A quoted atom's escapes are
   decoded by sexplib and kept verbatim here, so only its ATOM-ness is compared; everything else --
   how many children a list has, at every depth, and what each bare atom spells -- has to match. *)
let rec shapes_agree raw sexp =
  match (raw.raw_atom, sexp) with
  | Some _, Sexp.Atom _ when raw.raw_quoted -> true
  | Some atom, Sexp.Atom parsed -> String.equal atom parsed
  | None, Sexp.List children ->
      List.length raw.raw_children = List.length children
      && List.for_all2_exn raw.raw_children children ~f:shapes_agree
  | _ -> false

(** Raises if [content] is not something both readers agree on: a scan that cannot read its input
    must say so rather than report a file with a hole in it. *)
let stanzas content =
  let parsed = Sexplib.Sexp.scan_sexps (Lexing.from_string content) in
  let raw, _comments = read_raw content in
  if List.length raw <> List.length parsed || not (List.for_all2_exn raw parsed ~f:shapes_agree)
  then
    failwith
      (Printf.sprintf
         "dune file parses as %d top-level forms and reads as %d, or the two trees differ in shape \
          -- sexplib and dune disagree about what is a comment here (`#|…|#` and `#;` are comments \
          to the one and atoms to the other), so this scan would read the file with a hole in it"
         (List.length parsed) (List.length raw));
  parsed

(** The 1-based line an offset falls on, for a diagnostic that a reader can go to. *)
let line_of content offset =
  let stop = min offset (String.length content) in
  1 + String.count (String.sub content ~pos:0 ~len:stop) ~f:(Char.equal '\n')

let head = function Sexp.List (Sexp.Atom h :: _) -> Some h | _ -> None

(** The arguments of a stanza's [(<name> …)] field, where it has one. *)
let field_in fields name =
  List.find_map fields ~f:(function
    | Sexp.List (Sexp.Atom f :: args) when String.equal f name -> Some args
    | _ -> None)

let field stanza name =
  match stanza with Sexp.List (_ :: fields) -> field_in fields name | _ -> None

let rec atoms = function Sexp.Atom a -> [ a ] | Sexp.List l -> List.concat_map l ~f:atoms

(** Whether a [copy_files] pattern could name [name] — the literal spelling, or a wildcard that
    covers it. Only [*] and [?] are interpreted; a pattern using dune's set syntax ([{a,b}], [[ab]])
    is taken as possibly matching, since this decides where a config EXISTS and guessing wide only
    risks accepting a directory that has one anyway, while guessing narrow rejects a correctly
    configured one (Codex P2, round 12 of PR #343). *)
let glob_could_match pattern ~name =
  if String.exists pattern ~f:(fun c -> List.mem [ '{'; '[' ] c ~equal:Char.equal) then true
  else
    let pattern = String.to_array pattern and name = String.to_array name in
    let rec go p n =
      if p = Array.length pattern then n = Array.length name
      else
        match pattern.(p) with
        | '*' -> go (p + 1) n || (n < Array.length name && go p (n + 1))
        | '?' -> n < Array.length name && go (p + 1) (n + 1)
        | c -> n < Array.length name && Char.equal c name.(n) && go (p + 1) (n + 1)
    in
    go 0 0

(** Dependency-specification forms that really depend on the file they name, so that the config
    named inside one is built before the test runs.

    Only [(file …)] and a bare path. Dune's dependency language has forms that name something else
    entirely — [(alias ocannl_config)], [(env_var ocannl_config)], [(package ocannl_config)] — and
    reading those as a file dependency lets a stanza pass while depending on no config at all (Codex
    P2, round 3 of PR #343). The GLOB forms are excluded for a subtler reason of the same kind
    (round 7): they match the source tree, and in a directory whose config arrives through
    [(copy_files …)] the file is a generated target, so [(deps (glob_files ocannl_config))] matches
    nothing and builds nothing. Checked against dune 3.18 in a cleaned build directory, where such a
    rule fails with "cat: ocannl_config: No such file or directory" exactly as a rule with no deps
    at all does.

    Anything not listed does not count, which fails safe: a form that does depend on the file fails
    loudly and is added, rather than one that does not silently passing. *)
let file_dep_forms = [ "file" ]

(** Forms that name paths, for finding what a [(:name …)] binding might point at. Wider than
    {!file_dep_forms} on purpose: here the question is what an executable could be called, and
    recognising more spellings only means recognising more of the executables that get run. *)
let path_bearing_forms = [ "file"; "glob_files"; "glob_files_rec"; "source_tree" ]

(** Every config file a [(deps …)] field really depends on, as the path is written — relative to the
    stanza's directory.

    Which of them is the one that MATTERS this module cannot say. OCANNL walks UP from the process
    directory and reads the first config it finds ([Utils.config_file_args]), so the dependency has
    to name that one: an ancestor's will do while no nearer directory has its own, and a rule that
    chdirs out of the stanza's subtree may legitimately name a common parent (Codex P2, rounds 9 to
    11 of PR #343). Where the files exist, and hence which directory wins, is the caller's
    knowledge. *)
let declared_config_paths args =
  let rec collect sexp =
    match sexp with
    | Sexp.Atom atom ->
        if String.equal (Stdlib.Filename.basename atom) config_file then [ atom ] else []
    | Sexp.List (Sexp.Atom head :: rest)
      when String.is_prefix head ~prefix:":" || List.mem file_dep_forms head ~equal:String.equal ->
        List.concat_map rest ~f:collect
    | Sexp.List _ -> []
  in
  match args with
  | None -> []
  | Some args -> List.concat_map args ~f:collect |> List.dedup_and_sort ~compare:String.compare

(** The one spelling OCANNL reads the backend under (gh-ocannl-652 dropped the lowercase form). *)
let backend_env_var = "OCANNL_BACKEND"

(** Whether [args] — ONE dependency field's arguments — declare [(env_var name)].

    Scoped to a field rather than searched over a whole stanza, which is the whole point of it: a
    stanza may carry several dependency fields and an action runs under exactly one of them, so a
    declaration in a neighbouring field reruns nothing that matters while reading, to a whole-stanza
    search, as a declaration (Codex P2, round 3). Recursion is through the dependency forms only —
    the same shapes {!declared_config_paths} descends — so an [(env_var …)] appearing somewhere that
    is not a dependency at all is not mistaken for one. *)
let declares_env_var args name =
  let rec collect sexp =
    match sexp with
    | Sexp.List [ Sexp.Atom "env_var"; Sexp.Atom declared ] -> String.equal declared name
    | Sexp.List (Sexp.Atom head :: rest)
      when String.is_prefix head ~prefix:":" || List.mem file_dep_forms head ~equal:String.equal ->
        List.exists rest ~f:collect
    | Sexp.List _ | Sexp.Atom _ -> false
  in
  match args with None -> false | Some args -> List.exists args ~f:collect

(** The names a [(name …)] or [(names …)] field gives, in order. *)
let names_of stanza =
  match field stanza "name" with
  | Some [ Sexp.Atom name ] -> [ name ]
  | _ -> (
      match field stanza "names" with
      | Some args -> List.filter_map args ~f:(function Sexp.Atom n -> Some n | _ -> None)
      | _ -> [])

(** {1 Aliases} *)

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
  match field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom n ] -> Some n
        | _ -> None)

(* The alias an `(alias (name A) …)` stanza defines: since dune 2.0 it carries no action of its own
   and only aggregates what its `deps` name. *)
let alias_stanza_name = function
  | Sexp.List (Sexp.Atom "alias" :: _) as stanza -> (
      match names_of stanza with [ n ] -> Some n | _ -> None)
  | _ -> None

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

(** A command atom, split at dune's [%{…}] boundaries and nowhere else.

    An earlier version tokenized on an allowlist of "characters a path may contain", which quietly
    cut valid filenames in half: [%{dep:helper+pp.exe}] came out as [pp.exe] and matched an
    exemption written for a different executable (Codex P2, round 4 of PR #343). A filename
    allowlist is a guess about the filesystem; [%{] and [}] are dune's own delimiters, and they are
    all this needs to know. *)
type piece = Literal of string | Pform of string

let pieces atom =
  let length = String.length atom in
  let rec go acc position =
    let literal_from start =
      if start < length then [ Literal (String.subo atom ~pos:start) ] else []
    in
    match String.substr_index atom ~pos:position ~pattern:"%{" with
    | None -> List.rev acc @ literal_from position
    | Some start -> (
        let acc =
          if start > position then
            Literal (String.sub atom ~pos:position ~len:(start - position)) :: acc
          else acc
        in
        match String.index_from atom (start + 2) '}' with
        (* Unterminated: not a pform at all, so the rest is text. *)
        | None -> List.rev acc @ literal_from start
        | Some stop ->
            go (Pform (String.sub atom ~pos:(start + 2) ~len:(stop - start - 2)) :: acc) (stop + 1))
  in
  go [] 0

(** What a [(run …)] action's command names.

    {1 Why command position, and why an unrecognized command fails}

    Looking for a [.exe] anywhere in the stanza is the tempting rule and the wrong one in both
    directions: it counts a rule that merely copies an executable, and it misses every way of naming
    one that does not spell the extension — [%{bin:probe}] the first among them (Codex P2, round 2
    of PR #343), which is the same fall-through as round 1's [(alias …)] one spelling further in.
    Patching spellings one at a time invites a further round, so what this reads is the command
    position of a program action, and a command it cannot place FAILS rather than counting as
    nothing.

    The spellings it places, all of them present in this repository or in dune's own manual: a
    literal path ending in [.exe]; [%{dep:…}] and [%{exe:…}], which name a path the same way;
    [%{bin:name}], which resolves a PUBLIC executable — conservatively a site, because dune resolves
    it from this workspace before PATH, and an external tool that reads no configuration is what the
    exemption list is for; [%{name}] bound by a named dependency [(:name pp.exe)], which the action
    reaches without ever spelling the file; and the toolchain pforms, which run a compiler. A bare
    word is a tool on PATH ([python3], [diff]): not something this repository builds, so not a site.
*)
type command =
  | Runs of string  (** a literal executable path, relative to the action's working directory *)
  | Runs_dependency of string
      (** an executable dependency expansion, relative to the stanza's directory. Dune adjusts the
          expanded path under [chdir]; applying the action's cwd again changes its identity
          (gh-ocannl-982). Named dependency bindings have the same origin. *)
  | Runs_public of string
      (** the executable, by the PUBLIC name [%{bin:…}] gave. Kept apart from {!Runs} because the
          two carry the same string and mean different things: [%{bin:pkg.probe}] resolves a public
          name, [(run ./pkg.probe)] names a file, and a consumer matching an
          [(executable (public_name pkg.probe))] against the first must not accept the second
          (gh-ocannl-783, Codex P2 round 8) *)
  | External  (** a tool on PATH or in the toolchain, which this repository does not build *)
  | Unrecognized of string  (** command position this scan cannot read — reported, never ignored *)
  | Unknown_directory of string
  | Path_rewritten of string
      (** a bare command under [(setenv PATH …)]: unnameable for a REASON of its own, kept apart
          from the other unnameable programs so a check can tell them apart. Grouping them all as
          {!Unknown_directory} let a surviving [(bash …)] site answer for a dropped one (Codex P2,
          round 10). *)
(** a shell command line: not only is the program unreadable, the directory it runs in is too,
    because [cd] inside the line moves it without dune knowing (Codex P2, round 8) *)

(** Pforms naming a program that is part of the toolchain rather than of this repository.

    DATA both readers consult, not machinery either of them owns. The raw-text floor has to know
    which pforms name something this workspace provides in order to see
    [(run python3 %{dep:orchestrate.py})] at all: an external command handed a file we build, whose
    only evidence is in its ARGUMENT. What it must NOT do is re-derive {!classify_command} to find
    out — a second reader that runs the first reader's classifier is a copy, and the floor exists to
    be a second opinion. A list is inert: it says which spellings mean what, and leaves each reader
    to decide what to do about it (gh-ocannl-708). *)
let toolchain_pforms = [ "ocaml"; "ocamlc"; "ocamlopt"; "cc"; "cxx"; "make" ]

(** Pform prefixes that expand to the PATH of something dune builds or knows about in this
    workspace: [%{dep:x}] and its synonyms. Shared for the reason above. *)
let path_pforms = [ "dep"; "exe"; "path"; "file" ]

(** The pform that names a PUBLIC executable of this workspace by name rather than by path. Dune
    resolves it here before it looks at PATH, which is why it counts as ours. *)
let binary_pform = "bin"

(** How a [(test)] stanza's custom action names the test binary. *)
let test_pform = "%{test}"

(* The path AS WRITTEN, less a leading [./] that only says "here". Reducing it to a basename would
   make `%{dep:../../tools/pp.exe}` and a local `pp.exe` the same identity, and an exemption naming
   one would cover the other -- the same collapse the config scanner's duplicate-basename check
   exists to prevent (Codex P2, round 3, and #340 round 10 before it). *)
let program_path path = Option.value (String.chop_prefix path ~prefix:"./") ~default:path
let is_executable path = String.is_suffix path ~suffix:".exe"

(* An explicit RELATIVE path is not a PATH lookup. `./probe` and `../tools/probe` name something
   this repository produced, whatever their extension, and stripping the `./` before asking loses
   the one thing that distinguishes them from `python3` (Codex P2, round 8). An ABSOLUTE one names
   something the system provides -- `/usr/bin/python3` is no more ours than `python3` is (round
   18). *)
let is_absolute path =
  String.is_prefix path ~prefix:"/"
  || String.is_prefix path ~prefix:"\\"
  || (String.length path >= 2 && Char.equal path.[1] ':')

let is_explicit_path path =
  (not (is_absolute path)) && (String.is_prefix path ~prefix:"./" || String.contains path '/')

let is_path_lookup_token token =
  match pieces token with
  | [ Literal _ ] -> (not (is_absolute token)) && not (is_explicit_path token)
  | _ -> false

let classify_command ~named_deps cmd =
  let explicit = is_explicit_path cmd in
  let cmd = program_path cmd in
  match pieces cmd with
  | [ Literal path ] -> if is_executable path || explicit then Runs path else External
  | [ Pform pform ] -> (
      match String.lsplit2 pform ~on:':' with
      | Some (prefix, path) when List.mem path_pforms prefix ~equal:String.equal ->
          if is_executable path then Runs_dependency (program_path path) else Unrecognized cmd
      | Some (prefix, name) when String.equal prefix binary_pform -> Runs_public name
      | Some _ -> Unrecognized cmd
      | None -> (
          if String.equal pform "test" then Runs test_pform
          else if List.mem toolchain_pforms pform ~equal:String.equal then External
          else
            (* [./%{pp}] with [(deps (:pp pp.exe))]: the action names the dependency, not the
               file. *)
            match List.Assoc.find named_deps pform ~equal:String.equal with
            | Some paths -> (
                match List.find paths ~f:is_executable with
                | Some path -> Runs_dependency (program_path path)
                (* A binding that resolves to no executable is not evidence of an external tool: the
                   action runs whatever it binds, and this scan did not recognise it. *)
                | None -> Unrecognized cmd)
            | None -> Unrecognized cmd))
  (* Text and pforms mixed, or a pform inside a path: rather than guess where the program's name
     begins, say so. *)
  | _ -> Unrecognized cmd

(** The [(:name …)] bindings of a stanza's [(deps …)] field, with the paths each one binds.

    The paths are collected through the dependency forms that wrap them, because
    [(:runner (file probe.exe))] binds an executable as surely as [(:runner probe.exe)] does, and
    keeping only the bare atoms lost it — after which the binding looked empty and the command
    reading it looked external (Codex P2, round 6 of PR #343). *)
let named_deps_of stanza =
  let rec paths sexp =
    match sexp with
    | Sexp.Atom a -> [ a ]
    | Sexp.List (Sexp.Atom head :: rest) when List.mem path_bearing_forms head ~equal:String.equal
      ->
        List.concat_map rest ~f:paths
    | Sexp.List _ -> []
  in
  match field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List (Sexp.Atom name :: values) when String.is_prefix name ~prefix:":" ->
            Some (String.drop_prefix name 1, List.concat_map values ~f:paths)
        | _ -> None)

(** Actions that execute a program. [dynamic-run] is here because dune runs one there too (Codex P2,
    round 3); [system] and [bash] hand a command line to a shell. *)
let program_actions = [ "run"; "dynamic-run"; "system"; "bash" ]

(** Every other head dune's action language admits, including the predicate heads that appear in a
    [(with-accepted-exit-codes …)] test. None of them executes a program of its own — they nest
    actions, move bytes around, or compare files.

    The list is here so that {!unclassified_action_heads} can report a head on neither list. Three
    review rounds found this scan missing a way to run something — a stanza kind, then a command
    spelling, then [dynamic-run] — each patched instance leaving the next one waiting. What ends
    that is the fall-through, not the third patch: dune's action vocabulary is closed and short, so
    the scan can say what it knows and fail on the rest. *)
let inert_actions =
  [
    "progn";
    "concurrent";
    "chdir";
    "setenv";
    "with-stdout-to";
    "with-stderr-to";
    "with-outputs-to";
    "with-stdin-from";
    "with-accepted-exit-codes";
    "ignore-stdout";
    "ignore-stderr";
    "ignore-outputs";
    "no-infer";
    "echo";
    "write-file";
    "cat";
    "copy";
    "copy#";
    "copy-and-add-line-directive";
    "diff";
    "diff?";
    "cmp";
    "pipe-stdout";
    "pipe-stderr";
    "pipe-outputs";
    "format-dune-file";
    "or";
    "and";
    "not";
  ]

(** What an action puts in a position where a program could be named. [Elsewhere] wraps another of
    these with the destination of a [chdir] this scan cannot resolve. *)
type command_site =
  | Program of string * string list  (** the command, and the arguments it was given *)
  | Shell of string
  | Elsewhere of string * command_site
      (** a destination this scan cannot resolve, wrapping what runs there *)
  | Unnameable of string * command_site
      (** a rewritten PATH, wrapping what runs under it: there a bare command name no longer says
          what it names *)

(* Every command an action runs, at any depth: [(with-stdout-to … (run …))], [(no-infer (progn (run
   …) …))] and the rest nest the one that matters. Each comes with the directory the process will
   run in, relative to the stanza's own: [chdir] moves it, and the configuration an OCANNL
   executable finds is the one it searches upward from THERE, not the one next to the dune file
   (Codex P2, round 6 of PR #343). *)
let rec commands_in ?(cwd = "") ?(pinned = Set.empty (module String)) sexp =
  let nested =
    match sexp with Sexp.List l -> List.concat_map l ~f:(commands_in ~cwd ~pinned) | _ -> []
  in
  match sexp with
  (* A destination built out of a pform ([%{workspace_root}]) is not a path this scan can resolve,
     and treating it as a literal directory name would have the process searching from a directory
     that does not exist -- possibly landing back on the stanza's own config (Codex P2, round 12).
     Everything under it is reported as running somewhere unestablished. *)
  (* Rewriting PATH changes what a bare command name means, so nothing under it can be placed by
     reading the atom: `(setenv PATH . (run env probe))` may launch a local `probe` this scan would
     otherwise call a tool (Codex P2, round 16). Modelling the environment is not on the table;
     saying so is. *)
  | Sexp.List (Sexp.Atom "setenv" :: Sexp.Atom name :: value :: rest)
    when String.equal (String.uppercase name) "PATH" ->
      let value = match value with Sexp.Atom v -> v | _ -> "..." in
      List.concat_map rest ~f:(commands_in ~cwd ~pinned:(Set.add pinned "PATH"))
      (* The directory a nested `chdir` chose is still where the process runs; PATH says nothing
         about it (Codex P2, round 17). *)
      |> List.map ~f:(fun (inner_cwd, inner_pinned, command) ->
          let named =
            match command with
            | Program (cmd, _) -> cmd
            | Shell line -> "shell: " ^ line
            | Elsewhere (what, _) | Unnameable (what, _) -> what
          in
          ( inner_cwd,
            inner_pinned,
            Unnameable (Printf.sprintf "%s, under `(setenv PATH %s ...)`" named value, command) ))
  (* Any other `setenv` pins that variable over WHAT IT WRAPS, and nothing else: a `progn` that pins
     one branch has not pinned the run in another (gh-ocannl-749, Codex P1 round 1), and a pin
     around a helper does not protect the subject beside it (Codex P2, round 4). Carried per run
     site rather than per stanza, so both directions fall out of the same traversal. *)
  | Sexp.List (Sexp.Atom "setenv" :: Sexp.Atom name :: _value :: rest) ->
      List.concat_map rest ~f:(commands_in ~cwd ~pinned:(Set.add pinned name))
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest)
    when String.is_substring dir ~substring:"%{" ->
      List.concat_map rest ~f:(commands_in ~cwd ~pinned)
      |> List.map ~f:(fun (_, inner_pinned, command) ->
          let named =
            match command with
            | Program (cmd, _) -> cmd
            | Shell line -> "shell: " ^ line
            | Elsewhere (what, _) | Unnameable (what, _) -> what
          in
          (* Kept as the command it is, tagged with the destination: an unresolvable directory only
             matters for something that might read configuration there, and a PATH tool reads none
             wherever it runs (Codex P2, round 13). *)
          ( cwd,
            inner_pinned,
            Elsewhere (Printf.sprintf "%s, under `(chdir %s ...)`" named dir, command) ))
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest) ->
      List.concat_map rest ~f:(commands_in ~cwd:(in_subdir cwd dir) ~pinned)
  | Sexp.List (Sexp.Atom ("run" | "dynamic-run") :: Sexp.Atom cmd :: args) ->
      ( cwd,
        pinned,
        Program (cmd, List.filter_map args ~f:(function Sexp.Atom a -> Some a | _ -> None)) )
      :: nested
  (* A shell action hands a command line to a shell, and this scan does not parse shell. Splitting
     it on whitespace looked like reading it and was not: `if ready; then ./probe.exe; fi` yields
     `./probe.exe;`, which ends in no extension and passes for an external tool -- so the rule runs
     a test executable and the check says nothing (Codex P2, round 5). A shell line is reported
     whole instead, and as something stronger than an unreadable command: `cd ../sibling &&
     ./probe.exe` moves the working directory with no dune `chdir` to show for it, so the directory
     whose config the process will find is unknown too (round 8). *)
  | Sexp.List (Sexp.Atom ("bash" | "system") :: args) ->
      List.filter_map args ~f:(function Sexp.Atom a -> Some (cwd, pinned, Shell a) | _ -> None)
      @ nested
  | _ -> nested

(** Heads inside a stanza's [(action …)] that are on neither action list, each once. *)
let unclassified_action_heads stanza =
  (* With the directory it sits in: an unknown action may run an OCANNL program, and a `chdir`
     around it moves where that would read its configuration (Codex P2, round 18). *)
  let rec walk_action ~cwd sexp =
    match sexp with
    | Sexp.Atom _ -> []
    (* A destination built out of a pform is no directory this scan can resolve, here as much as in
       `commands_in`: taking it literally would put the site in a directory whose ancestors include
       the stanza's own, and accept the stanza-local dependency for a process running elsewhere
       (Codex P2, round 19 of PR #343). *)
    | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest)
      when String.is_substring dir ~substring:"%{" ->
        List.map
          (List.concat_map rest ~f:(walk_action ~cwd))
          ~f:(fun (_, head) -> (None, Printf.sprintf "%s, under `(chdir %s ...)`" head dir))
    | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest) ->
        List.concat_map rest ~f:(walk_action ~cwd:(in_subdir cwd dir))
    | Sexp.List (Sexp.Atom head :: args) ->
        let nested =
          (* A program action's arguments are its command line, not further actions. *)
          if List.mem program_actions head ~equal:String.equal then []
          else List.concat_map args ~f:(walk_action ~cwd)
        in
        if
          List.mem program_actions head ~equal:String.equal
          || List.mem inert_actions head ~equal:String.equal
        then nested
        else (Some cwd, head) :: nested
    | Sexp.List l -> List.concat_map l ~f:(walk_action ~cwd)
  in
  match field stanza "action" with
  | None -> []
  | Some args ->
      List.concat_map args ~f:(walk_action ~cwd:"") |> List.dedup_and_sort ~compare:Poly.compare

(** Every command site in a stanza, including commands classified as external tools. The original
    site, directory, enclosing environment pins, and classification are preserved so a closed-world
    consumer can distinguish a verified no-argument helper from an opaque launcher. *)
let classified_command_sites_with_pins_preserving_multiplicity stanza =
  let named_deps = named_deps_of stanza in
  let rec classify command =
    match command with
    | Program (cmd, args) -> (
        match classify_command ~named_deps cmd with
        (* A PATH tool is only the end of the story while it is handed nothing this repository
           builds. `env probe.exe` launches it; `diff old.exe new.exe` reads it; `env -C ../s
           probe.exe` launches it SOMEWHERE ELSE -- and dune's grammar says which of the three it is
           no more than a list of launchers would (Codex P2, rounds 12 to 14). So the strongest of
           the three is assumed: something may run, in a directory this scan cannot establish. A
           command that IS a workspace executable answers for itself, and its arguments are its
           data. *)
        | External -> (
            let handed =
              List.filter args ~f:(fun arg ->
                  match classify_command ~named_deps arg with
                  | Runs _ | Runs_dependency _ | Runs_public _ | Unrecognized _ -> true
                  | External | Unknown_directory _ | Path_rewritten _ -> false)
            in
            match handed with
            | [] -> External
            | handed ->
                Unknown_directory
                  (Printf.sprintf "%s, handed %s" cmd (String.concat ~sep:", " handed)))
        | classified -> classified)
    | Shell line -> Unknown_directory ("shell: " ^ line)
    (* Only what could read a configuration cares which directory it runs in: a PATH tool reads none
       wherever it is (Codex P2, round 13). *)
    | Elsewhere (what, command) -> (
        match classify command with External -> External | _ -> Unknown_directory what)
    (* Under a rewritten PATH no BARE literal can be trusted, including [probe.exe]: the suffix
       makes it workspace-shaped but does not stop PATH from selecting another file. A path or pform
       still names what it names (gh-ocannl-874, Codex P2 round 4). *)
    | Unnameable (what, Program (cmd, _)) when is_path_lookup_token cmd -> Path_rewritten what
    | Unnameable (what, command) -> (
        match classify command with External -> Path_rewritten what | other -> other)
  in
  List.map (commands_in stanza) ~f:(fun (cwd, pinned, command) ->
      (cwd, pinned, command, classify command))

let classified_commands_with_pins_preserving_multiplicity stanza =
  List.map (classified_command_sites_with_pins_preserving_multiplicity stanza)
    ~f:(fun (cwd, pinned, _site, command) -> (cwd, pinned, command))

(** What a stanza runs from this workspace. This form preserves one entry per command site;
    consumers enforcing execution multiplicity must use it rather than the deduplicated census. *)
let executables_run_with_pins_preserving_multiplicity stanza =
  List.filter (classified_commands_with_pins_preserving_multiplicity stanza)
    ~f:(fun (_, _, command) -> match command with External -> false | _ -> true)

(** The set-like reading used by directory and dependency checks, where two identical command sites
    ask the same question. *)
let executables_run_with_pins stanza =
  executables_run_with_pins_preserving_multiplicity stanza
  |> List.dedup_and_sort ~compare:Poly.compare

(** What a stanza runs, each with the directory it runs in — the shape every caller but the runner
    machinery wants. *)
let executables_run stanza =
  List.map (executables_run_with_pins stanza) ~f:(fun (cwd, _pinned, command) -> (cwd, command))
  |> List.dedup_and_sort ~compare:Poly.compare

type kind =
  | Test  (** a [(test)] or [(tests)] stanza, which dune runs itself *)
  | Inline_tests  (** a [(library)] with an [(inline_tests)] field, ditto *)
  | Runs_executable
      (** a [(rule)] that runs an executable — where an [(executable)] stanza's dependencies have to
          live, there being no [deps] field on one *)
  | Unreadable_command
      (** a [(run …)] whose command this scan cannot place: reported, so that what it runs is
          settled by a reader rather than by silence *)
  | Unreadable_directory
      (** a shell action: the directory the process ends up in is unknown, so no dependency of this
          stanza's can be shown to be the right one *)
  | Unclassified_action
      (** an action head on neither {!program_actions} nor {!inert_actions} — it might run a
          program, so it is reported too *)

type site = {
  kind : kind;
  name : string;
  declares_config : bool;
      (** whether the deps depend on any config file at all; WHICH one had to be named is
          {!declared_config_paths}, since only the caller knows where configs exist *)
  declared_config_paths : string list;
  declares_backend : bool;
      (** whether THE DEPS THIS SITE RUNS UNDER declare [(env_var OCANNL_BACKEND)] — read from the
          same field as {!declares_config}, and for the same reason. A stanza can carry dependency
          fields that the action does not run under: an inline-test library's own
          [(preprocessor_deps …)] is not [(inline_tests (deps …))], and a declaration in the former
          leaves the test action just as undeclared as none at all while looking, to anything that
          searches the stanza as a whole, exactly like a declaration (Codex P2, round 3). Scoping it
          per site is what keeps that answer tied to the thing dune will actually rerun. *)
  path_rewritten : bool;
      (** whether this site is a program the walk refuses to name because [(setenv PATH …)] decides
          what it resolves to — as opposed to the other reasons a site can be unnameable, which a
          caller must be able to tell apart (Codex P2, round 10). *)
  executables : string list;
      (** for a {!Runs_executable} site, the executables it covers, each as {!classify_command}
          named it — the identities kept apart, rather than the display [name] that joins them with
          ", ". A caller matching sites against executables must read them from here: recovering
          them by splitting [name] loses a path that itself contains a comma (Codex P2, round 2).
          Empty for every other kind. *)
  subdir : string;
  cwd : string;
}
(** [subdir] is the directory the stanza applies to, relative to the dune file's own: empty at the
    top level, and the path a [(subdir …)] wrapper names inside one. A wrapped stanza runs
    elsewhere, so it is that directory's config it needs — test/operations/dune configures
    test/operations/config this way.

    [cwd] is the directory the process runs in, relative to the stanza's: empty unless a [chdir]
    moved it. The two compose — [in_subdir subdir cwd] is the directory whose configuration the
    executable will actually find, and the one that has to have one. *)

(* [(subdir <dir> <stanza>…)] applies its body to another directory. Descending into it is what
   keeps its stanzas subject to the same rules; ignoring it would drop them silently, which is the
   one thing this scan must not do. *)
let rec walk dir stanzas ~f =
  List.concat_map stanzas ~f:(fun stanza ->
      match stanza with
      | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body) -> walk (in_subdir dir sub) body ~f
      | stanza -> f dir stanza)

let kind_name = function
  | Test -> "test"
  | Inline_tests -> "inline tests"
  | Runs_executable -> "rule running"
  | Unreadable_command -> "rule whose command this scan cannot read:"
  | Unreadable_directory -> "rule whose working directory this scan cannot establish:"
  | Unclassified_action -> "rule with an action this scan cannot place:"

(** Stanza heads that carry an action, so one of them may run a test executable. [alias] is here for
    the same reason [rule] is: it took an [action] field before dune 2.0, and it can still depend on
    an executable. *)
let action_heads = [ "rule"; "alias" ]

(** Stanza heads classified as running no test executable. Declaring things to build ([executable],
    [ocamllex]), placing files ([copy_files], [install]), or describing the directory ([env],
    [dirs]) — none of them run anything of their own.

    The list exists so that {!unclassified_heads} can report what is on neither list. A head nobody
    has classified might carry an action, and passing over it in silence is the failure this whole
    scan exists to end (Codex P2, round 1 of PR #343) — [cram] is the live example: dune runs cram
    tests, this repository has none, and the day one appears the scan says so rather than counting
    it as nothing. *)
let inert_heads =
  [
    "executable";
    "executables";
    "copy_files";
    "copy_files#";
    "install";
    "env";
    "dirs";
    "data_only_dirs";
    "vendored_dirs";
    "include_subdirs";
    "documentation";
    "ocamllex";
    "ocamlyacc";
    "menhir";
    "toplevel";
    "deprecated_library_name";
  ]

type raw_stanza = {
  raw_head : string;  (** the atom the stanza opens with *)
  raw_inline_tests : bool;  (** whether it has an [(inline_tests …)] field of its own *)
  raw_unnameable : string list;
      (** the directories of the bare commands the stanza runs under a [(setenv PATH …)] — each one
          a program the walk refuses to name for that reason, and so a site it must have placed
          carrying {!site.path_rewritten}. *)
  raw_subdir : string;
      (** the directory the stanza's [(subdir …)] nesting puts it in — where dune runs what it
          declares, and so where the config is resolved from. [""] for a top-level stanza. *)
  raw_runs : (string * string) list;
      (** the executables it runs, each with the directory the process ends up in — the stanza's
          [(subdir …)] composed with the [chdir] actions around the command, which is the directory
          whose config the executable will find. Deduplicated within the stanza, since {!sites}
          reports one site per distinct executable per directory, and named as {!classify_command}
          names a [Runs]. *)
  raw_test_cwds : string list;
      (** for a [test]/[tests] stanza, the directories its own binary runs in: one per [chdir]
          branch running [%{test}], and [[""]] composed with the subdirectory when the stanza has no
          custom action — which is how {!sites} counts its [Test] sites, one per directory. Empty
          for every other head. *)
  raw_opaque : string list;
      (** what the stanza demonstrably runs and this reader will not name: the command line of a
          [(bash …)] or [(system …)] action, which it does not parse; a command sitting under a
          [(chdir %{…} …)] whose destination it cannot resolve; and a command it cannot name handed
          something this workspace provides, which the walk reads as a program that may run
          somewhere it cannot establish (gh-ocannl-708). Recorded rather than dropped — THAT
          something runs is the whole of what gh-ocannl-659's per-stanza floor asks, and none of the
          three can be placed any further without parsing shell, resolving a pform, or deciding
          which of `env probe.exe` and `diff old.exe new.exe` is a launcher (gh-ocannl-690).

          Carried without a directory, deliberately. {!sites} places each of these as an
          [Unreadable_directory] site precisely BECAUSE the directory is unknown, so a per-directory
          floor built on them would be holding the walk's refusal to guess against this reader's
          guess. Only the per-stanza floor consumes them. *)
}
(** One stanza as a SECOND reader sees it. *)

(* One thing a stanza's actions run, as this reader sees it before deciding what to record of it. A
   variant rather than the tuple this used to be, because the two shapes gh-ocannl-690 added carry
   weaker evidence than a placed command does: a [(run …)] answers "what, and where", while a shell
   line answers only "something". *)
type raw_ran =
  | Raw_command of {
      cwd : string;  (** where the process ends up, when that is knowable *)
      token : string;  (** the command position, as written *)
      args : string list;
          (** the atoms the command was handed, as written. A command this reader cannot name is not
              the end of the story while one of them names something this workspace provides:
              [(run python3 %{dep:orchestrate.py})] runs a file we build, and the walk places a site
              for exactly that reading (gh-ocannl-708). *)
      under_path : bool;  (** whether a [(setenv PATH …)] encloses it *)
      unresolved : string option;
          (** [Some dir] when a [(chdir dir …)] whose destination holds a pform encloses it, which
              makes [cwd] a fiction — the command is still evidence that something runs *)
    }
  | Raw_opaque of string  (** something runs here and this reader will not say what *)

(* The raw reading of ONE stanza, lifted out of {!raw_stanzas} so that a caller holding a stanza can
   ask this reader about THAT stanza rather than about a whole file. {!raw_stanzas} is this over a
   document; gh-ocannl-659's floor pairs it with the walk stanza by stanza, which needs both
   classifiers reachable from one place (Codex P2, round 2).

   Sharing the traversal costs nothing this module was protecting: the independence it documents
   lives in the CLASSIFICATION -- how a stanza is decided to run something -- and that is still two
   separate pieces of machinery, answering the same question from different sides. *)

(** [raw_stanzas content] reads [content] as a second opinion on {!sites}: the stanzas it declares,
    with what each of them runs and where.

    This is the independent floor the checks are held to. A check phrased in terms of {!sites}
    cannot notice that walk going blind — a stanza it stops recognising looks exactly like a stanza
    that is not there — so the numbers it reports need a source that cannot go blind with it.

    {2 Where the independence lies, and where it deliberately does not}

    In the CLASSIFICATION, not in the parsing. What can go blind is {!sites}' own machinery — its
    traversal, {!executables_run}, {!classify_command} — and none of that is shared here: this
    module answers the same questions with its own small traversal, so a regression in one shows up
    as a disagreement with the other.

    The LEXING is not re-derived, and an earlier version of this reader that did re-derive it is why
    the point is worth stating. Dune's syntax has quoted atoms, comments, and whitespace wherever an
    atom may begin — and a hand-rolled scan of the raw text got each of those wrong in turn, in one
    position after another, every mistake either failing a correct scan or silently covering
    nothing. That is this module's own opening lesson, arrived at from the other side: an
    approximation of a grammar has no natural stopping point. So the text is PARSED, by the same
    reader dune's own syntax admits, and only the meaning is worked out here.

    {2 Agreeing with the walk about scope}

    Independent, but not free to disagree about what it is looking at:

    - STANZA POSITION. A stanza is a top-level form, or one inside a [(subdir …)], recursively.
      [(env (test (flags …)))] declares a build PROFILE named [test], and {!sites} rightly makes no
      test site of it. [inline_tests] counts only as a direct field of its stanza.
    - WHAT RUNS THINGS. Only [test], [tests] and the {!action_heads} — a [(run …)] under a library's
      [(preprocess …)] is a build-time action that {!sites} makes no site of.
    - WHERE IT RUNS. [(subdir …)] and [(chdir …)] both move the process, and {!sites} emits one site
      per resulting directory because each resolves a different config. The two compose through
      {!in_subdir}, and what is recorded here is the composition, ready to compare against
      [in_subdir site.subdir site.cwd].
    - WHAT CANNOT BE RESOLVED. Under a [chdir] whose destination holds a pform neither reader can
      say where the process runs: the walk emits an {!Unreadable_directory} site carrying no
      executables, and this one records the command in {!raw_stanza.raw_opaque} — tagged rather than
      dropped, since THAT something runs there is what the per-stanza floor rests on. A command that
      is external wherever it runs is passed over by both, the walk placing no site for it either.
    - IDENTITY. Commands are recognised as {!classify_command} recognises them, [(:name …)] bindings
      included, and normalised to the same string.

    What it still declines to NAME is a [(bash …)] or [(system …)] command line, and an external
    command handed something this workspace provides: the text does not say what those run. Both are
    recorded as running something unnamed, which is the whole of what the per-stanza floor asks of
    them. Where it under-reports it does so knowingly: a floor may under-claim, and then holds a
    weaker statement about that one stanza, but may never over-claim, which would report a hole in a
    correct scan. *)
let raw_stanza_of =
  let counted_heads = "test" :: "tests" :: action_heads in
  let test_heads = [ "test"; "tests" ] in
  (* Every `(:name …)` the stanza binds, with the first executable it names -- the binding may wrap
     its path in a dependency form, and it is found wherever in the stanza it sits. *)
  (* A binding's paths, the way [named_deps_of] reads them: atoms, and the forms that CARRY paths.
     Descending into any form instead would take `(:runner (alias fake.exe) (file real.exe))` for
     the alias -- which dune does not run and the walk does not name (Codex P2, round 9). *)
  let rec first_executable sexp =
    match sexp with
    | Sexp.Atom a when is_executable a -> Some (program_path a)
    | Sexp.Atom _ -> None
    | Sexp.List (Sexp.Atom head :: rest) when List.mem path_bearing_forms head ~equal:String.equal
      ->
        List.find_map rest ~f:first_executable
    | Sexp.List _ -> None
  in
  (* And only from the stanza's own `deps` field, which is the only place [named_deps_of] looks. *)
  let bindings_of fields =
    match
      List.find_map fields ~f:(function
        | Sexp.List (Sexp.Atom "deps" :: args) -> Some args
        | _ -> None)
    with
    | None -> []
    | Some args ->
        List.filter_map args ~f:(function
          | Sexp.List (Sexp.Atom name :: values)
            when String.is_prefix name ~prefix:":" && String.length name > 1 -> (
              match List.find_map values ~f:first_executable with
              | Some path -> Some (String.drop_prefix name 1, path)
              | None -> None)
          | _ -> None)
  in
  (* The program a command token names, mirroring [classify_command] for the spellings the stanza
     itself settles: a `.exe`, any explicit relative path (`./probe` is ours whatever its
     extension), the pforms naming a path in this workspace, and a `%{name}` the deps bind. *)
  let program ~bindings token =
    let bare = program_path token in
    match String.chop_prefix bare ~prefix:"%{" with
    | None when String.is_substring bare ~substring:"%{" -> None
    | None ->
        if is_executable token || is_explicit_path token then Some (program_path token) else None
    | Some rest -> (
        match String.chop_suffix rest ~suffix:"}" with
        | None -> None
        | Some inner -> (
            match String.lsplit2 inner ~on:':' with
            | Some (prefix, path) when List.mem path_pforms prefix ~equal:String.equal ->
                if is_executable path then Some (program_path path) else None
            | Some (prefix, name) when String.equal prefix binary_pform -> Some name
            | Some _ -> None
            | None -> List.Assoc.find bindings inner ~equal:String.equal))
  in
  (* A command name PATH decides the meaning of: no pform and no explicit path to read it off. An
     [.exe] suffix makes the name workspace-shaped but does not stop PATH from selecting another
     file. Under a rewritten PATH the walk refuses to vouch for such a name, so it places a site for
     it wherever it runs -- which is what the [raw_unnameable] floor and the opaque one both rest
     on, so they read one predicate rather than restating it. *)
  let is_bare_name cmd =
    (not (String.is_substring cmd ~substring:"%{"))
    && (not (is_absolute cmd))
    && not (is_explicit_path cmd)
  in
  (* Whether a command-line word names something this workspace provides, as far as the RAW TEXT can
     tell: a `.exe`, an explicit relative path, or a pform that is not one of the toolchain's.

     This is what lets the floor see a command it cannot NAME running something of ours -- `(run
     python3 %{dep:orchestrate.py})`, and `env -C ../sibling ./probe.exe` in the same shape. The
     walk reaches the same stanzas by asking {!classify_command} about every argument; this asks a
     coarser question of the text, and shares only the LISTS the answer is written in
     ({!toolchain_pforms} first among them, which is what keeps `(run tool %{ocaml})` invisible to
     both). Coarser is the safe direction: a floor may under-claim -- it then holds a weaker
     statement about that one stanza -- and may not over-claim, which would report a hole in a
     correct scan. The `%{…}` boundaries are dune's own delimiters and are read with {!pieces}, the
     way {!is_explicit_path} and {!is_executable} are already read from one place: what stays apart
     between the two readers is the judgement, not the lexing. *)
  let names_workspace_file arg =
    is_executable arg || is_explicit_path arg
    || List.exists (pieces arg) ~f:(function
      | Literal _ -> false
      | Pform pform ->
          let head =
            match String.lsplit2 pform ~on:':' with Some (head, _) -> head | None -> pform
          in
          not (List.mem toolchain_pforms head ~equal:String.equal))
  in
  (* Every command the stanza runs, with the directory it runs in. Its own traversal, deliberately:
     this is the question [executables_run] answers, and answering it twice is the point. *)
  let rec commands ~cwd ~unresolved ~under_path sexp =
    match sexp with
    | Sexp.Atom _ -> []
    (* `(setenv PATH …)` changes what a BARE command name resolves to, so the walk stops vouching
       for where such a program runs. Descending through it as an ordinary form would lose that
       (Codex P2, round 9); what is beneath it is marked instead. *)
    | Sexp.List (Sexp.Atom "setenv" :: Sexp.Atom name :: _value :: rest)
      when String.equal (String.uppercase name) "PATH" ->
        List.concat_map rest ~f:(commands ~cwd ~unresolved ~under_path:true)
    | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest) ->
        if String.is_substring dir ~substring:"%{" then
          List.concat_map rest ~f:(commands ~cwd ~unresolved:(Some dir) ~under_path)
        else List.concat_map rest ~f:(commands ~cwd:(in_subdir cwd dir) ~unresolved ~under_path)
    (* A shell action hands a command line to a shell, and this reader parses shell no more than
       [commands_in] does -- splitting it on whitespace would be reading it, and reading it wrong.
       What the text does say is that the stanza runs SOMETHING, which is the whole of what the
       per-stanza floor asks; before gh-ocannl-690 it said nothing, and a stanza running its test
       through `bash` had no floor under it at all. The arguments are that command line rather than
       nested actions, so the traversal stops here. *)
    | Sexp.List (Sexp.Atom (("bash" | "system") as action) :: args) ->
        List.filter_map args ~f:(function
          | Sexp.Atom line -> Some (Raw_opaque (Printf.sprintf "(%s %s)" action line))
          | _ -> None)
    (* Kept whatever encloses it: what an unresolvable `chdir` costs is the DIRECTORY, and
       [of_stanza] is where that decides which list the command lands in. *)
    | Sexp.List (Sexp.Atom ("run" | "dynamic-run") :: Sexp.Atom cmd :: args) ->
        let args = List.filter_map args ~f:(function Sexp.Atom a -> Some a | _ -> None) in
        [ Raw_command { cwd; token = cmd; args; under_path; unresolved } ]
    | Sexp.List l -> List.concat_map l ~f:(commands ~cwd ~unresolved ~under_path)
  in
  let of_stanza ~subdir sexp =
    match sexp with
    | Sexp.Atom _ | Sexp.List [] | Sexp.List (Sexp.List _ :: _) -> []
    | Sexp.List (Sexp.Atom head :: fields) ->
        let inline =
          List.exists fields ~f:(function
            | Sexp.List (Sexp.Atom "inline_tests" :: _) -> true
            | _ -> false)
        in
        let ran =
          if List.mem counted_heads head ~equal:String.equal then
            commands ~cwd:subdir ~unresolved:None ~under_path:false sexp
          else []
        in
        let bindings = bindings_of fields in
        (* What this reader can place: everything but the shell lines and what an unresolvable
           `chdir` moved. Both of those are evidence that something runs and no evidence of WHERE,
           so letting them through here would put a fabricated directory into the per-directory
           floors `config_dep_completeness` builds out of [raw_runs] and [raw_unnameable]. *)
        let placed =
          List.filter_map ran ~f:(function
            | Raw_command { cwd; token; under_path; unresolved = None; _ } ->
                Some (cwd, token, under_path)
            | Raw_command _ | Raw_opaque _ -> None)
        in
        let is_test = List.mem test_heads head ~equal:String.equal in
        let test_cwds =
          if not is_test then []
          else
            match
              List.filter_map placed ~f:(fun (cwd, cmd, _) ->
                  (* `./%{test}` is the test binary too: the `./` says only "here". *)
                  if String.equal (program_path cmd) test_pform then Some cwd else None)
            with
            | [] -> [ subdir ]
            | cwds -> List.dedup_and_sort cwds ~compare:String.compare
        in
        [
          {
            raw_head = head;
            raw_inline_tests = inline;
            raw_subdir = subdir;
            raw_runs =
              List.filter_map placed ~f:(fun (cwd, cmd, under_path) ->
                  if under_path && is_bare_name cmd then None
                  else if String.equal (program_path cmd) test_pform then None
                  else
                    match program ~bindings cmd with Some path -> Some (cwd, path) | None -> None)
              |> List.dedup_and_sort ~compare:(fun (c1, e1) (c2, e2) ->
                  match String.compare c1 c2 with 0 -> String.compare e1 e2 | n -> n);
            raw_test_cwds = test_cwds;
            (* A bare name under `(setenv PATH …)`: the only shape whose site the walk is certain to
               make {!Unreadable_directory}, since a command it CAN name stays a [Runs] even there.
               Deduplicated by (directory, command), which is how the walk's own sites collapse. *)
            raw_unnameable =
              List.filter_map placed ~f:(fun (cwd, cmd, under_path) ->
                  if under_path && is_bare_name cmd then Some (cwd, cmd) else None)
              |> List.dedup_and_sort ~compare:Poly.compare
              |> List.map ~f:fst;
            raw_opaque =
              (* An unresolvable `chdir` costs the DIRECTORY and nothing else, so it tags whatever
                 it encloses rather than replacing it. *)
              (let where unresolved what =
                 match unresolved with
                 | None -> what
                 | Some dir -> Printf.sprintf "%s, under `(chdir %s ...)`" what dir
               in
               List.filter_map ran ~f:(function
                 | Raw_opaque what -> Some what
                 (* Under an unresolvable `chdir` what is lost is the DIRECTORY, not the evidence
                    that something runs -- and evidence from two enclosing forms must not cancel
                    out. Several things can supply it: a command this reader could otherwise name, a
                    bare name under a rewritten PATH, which the walk places a site for precisely
                    because PATH may point it at a workspace executable, and a command handed
                    something of ours. Reading only the first left a command enclosed by BOTH with
                    no floor under it, in either nesting order (Codex P2, round 1 of PR #422). *)
                 | Raw_command { token; args; unresolved; under_path; _ } -> (
                     match program ~bindings token with
                     (* A named command is already in [raw_runs] where its directory is known; only
                        an unresolvable `chdir` moves it here. *)
                     | Some exe -> Option.map unresolved ~f:(fun dir -> where (Some dir) exe)
                     (* The test binary is the stanza's own, and [raw_test_cwds] is where it is
                        floored -- with the directory it runs in, which is more than this list can
                        carry. *)
                     | None when String.equal (program_path token) test_pform -> None
                     | None -> (
                         if under_path && is_bare_name token then
                           (* Likewise [raw_unnameable], which carries the directory when there is
                              one. *)
                           Option.map unresolved ~f:(fun dir ->
                               Printf.sprintf "%s, under `(setenv PATH ...)` and `(chdir %s ...)`"
                                 token dir)
                         else
                           (* gh-ocannl-708: a command this reader cannot name, handed something
                              this workspace provides. The walk reads that as a program that may
                              run, in a directory it cannot establish -- `env -C ../sibling
                              ./probe.exe` is the shape that makes it more than a tool reading our
                              files -- and places a site accordingly. A PATH tool handed nothing of
                              ours stays invisible to both readers. *)
                           match List.filter args ~f:names_workspace_file with
                           | handed when not (List.is_empty handed) ->
                               Some
                                 (where unresolved
                                    (Printf.sprintf "%s, handed %s" token
                                       (String.concat ~sep:", " handed)))
                           (* Or spelled, in command position, as something out of this workspace
                              that this reader cannot resolve to a program: `%{dep:x.py}` names a
                              file we build without naming an executable. *)
                           | _ ->
                               if names_workspace_file token then
                                 Some
                                   (where unresolved
                                      (Printf.sprintf "%s, itself named out of this workspace" token))
                               else None))))
              |> List.dedup_and_sort ~compare:String.compare;
          };
        ]
  in
  fun ~subdir sexp -> of_stanza ~subdir sexp

(** Whether the raw reader thinks this stanza runs something — the floor's side of "is this stanza
    subject to the backend rule", kept in ONE place so a caller cannot drift from it.

    Deliberately a lower bound, and safe to be one — but a lower bound over what a stanza RUNS, not
    over what this reader can name. It reads [run] and [dynamic-run] commands, [bash] and [system]
    actions, and commands under a [chdir] it cannot resolve; the last two it can only report as
    {!raw_stanza.raw_opaque}, and that is enough, because the question here is whether the stanza is
    subject to the rule and not which program answers for it (gh-ocannl-690).

    What still passes it by is an action head nobody has classified and which encloses no [(run …)]:
    {!sites_of_stanza} places an {!Unclassified_action} site where this reader, which knows [run]
    and the shell actions by name, sees nothing at all. Under-claiming is harmless exactly as long
    as the comparison is made STANZA BY STANZA: it weakens the floor for that stanza alone. Compared
    in aggregate it is not harmless at all — a stanza the walk counts and this reader does not
    contributes slack that hides a DIFFERENT stanza dropping out of enforcement (Codex P2, round 2).
*)
let raw_runs_something r =
  (not (List.is_empty r.raw_runs))
  || (not (List.is_empty r.raw_test_cwds))
  || r.raw_inline_tests
  || (not (List.is_empty r.raw_unnameable))
  || not (List.is_empty r.raw_opaque)

let raw_stanzas content =
  (* Parsed rather than scanned. [sites] has already refused any file the two readers disagree
     about, so what arrives here is a file both read the same way. *)
  let parsed = Sexplib.Sexp.scan_sexps (Lexing.from_string content) in
  (* A `(subdir …)` holds stanzas, and the directory it names is where they run. *)
  let rec walk_stanzas ~subdir sexp =
    match sexp with
    | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom dir :: rest) ->
        raw_stanza_of ~subdir sexp
        @ List.concat_map rest ~f:(walk_stanzas ~subdir:(in_subdir subdir dir))
    | sexp -> raw_stanza_of ~subdir sexp
  in
  List.concat_map parsed ~f:(walk_stanzas ~subdir:"")

(** Every place ONE stanza runs a test executable, given the directory a [(subdir …)] nesting puts
    it in. {!sites} is this over a whole file; a caller that needs to ask a stanza something else at
    the same time — gh-ocannl-659 asks what comments sit inside it — walks the file itself and calls
    this per stanza, rather than re-deriving what counts as a site. *)
let sites_of_stanza subdir stanza =
  let deps () = field stanza "deps" in
  let site ?(cwd = "") ?deps:(deps_field = deps ()) ?(executables = []) ?(path_rewritten = false)
      kind name =
    [
      {
        kind;
        name;
        declares_config = not (List.is_empty (declared_config_paths deps_field));
        declared_config_paths = declared_config_paths deps_field;
        declares_backend = declares_env_var deps_field backend_env_var;
        path_rewritten;
        executables;
        subdir;
        cwd;
      };
    ]
  in
  let stanza_name () = String.concat ~sep:", " (names_of stanza) in
  (* Everything a stanza's actions run, each with the directory it runs in. A `(test)` may carry a
     custom action, so this serves both branches; the difference is only WHICH of the commands is
     the test itself. *)
  let run = executables_run stanza in
  let sites_for ~is_test =
    List.map run ~f:fst
    |> List.dedup_and_sort ~compare:String.compare
    |> List.concat_map ~f:(fun cwd ->
        let for_cwd f =
          List.filter_map run ~f:(fun (c, command) ->
              if String.equal c cwd then f command else None)
        in
        let exes =
          for_cwd (function
            (* In a test stanza, `%{test}` is the test binary itself, reported as the Test site
               rather than as something the action also runs. *)
            | Runs name when is_test && String.equal name test_pform -> None
            (* A public-name run launches a program exactly as a path does; only a CONSUMER matching
               it against a declared executable has to tell the two apart. *)
            | Runs name | Runs_dependency name | Runs_public name -> Some name
            | _ -> None)
        in
        let unreadable = for_cwd (function Unrecognized cmd -> Some cmd | _ -> None) in
        let unlocatable = for_cwd (function Unknown_directory cmd -> Some cmd | _ -> None) in
        let rewritten = for_cwd (function Path_rewritten cmd -> Some cmd | _ -> None) in
        (if List.is_empty exes then []
         else site ~cwd ~executables:exes Runs_executable (String.concat ~sep:", " exes))
        @ List.concat_map unreadable ~f:(fun cmd -> site ~cwd Unreadable_command cmd)
        @ List.concat_map unlocatable ~f:(fun cmd -> site ~cwd Unreadable_directory cmd)
        @ List.concat_map rewritten ~f:(fun cmd ->
            site ~cwd ~path_rewritten:true Unreadable_directory cmd))
  in
  match head stanza with
  | Some ("test" | "tests") ->
      (* Where the TEST runs, which is where its own command runs -- not where a helper in the same
         action happens to be sent (Codex P2, round 10). With no custom action, dune runs it in the
         stanza's directory. *)
      let test_cwds =
        List.filter_map run ~f:(function
          | cwd, Runs name when String.equal name test_pform -> Some cwd
          | _ -> None)
        |> List.dedup_and_sort ~compare:String.compare
      in
      let test_cwds = if List.is_empty test_cwds then [ "" ] else test_cwds in
      List.concat_map test_cwds ~f:(fun cwd -> site ~cwd Test (stanza_name ()))
      @ sites_for ~is_test:true
  | Some "library" -> (
      match field stanza "inline_tests" with
      | None -> []
      | Some inline -> site ~deps:(field_in inline "deps") Inline_tests (stanza_name ()))
  | Some h when List.mem action_heads h ~equal:String.equal ->
      (* One site per directory the rule runs something in: what each needs is that directory's
         config, declared by the path that reaches it from here. *)
      sites_for ~is_test:false
      @ List.concat_map (unclassified_action_heads stanza) ~f:(function
        | Some cwd, head -> site ~cwd Unclassified_action head
        | None, what -> site Unreadable_directory what)
  | _ -> []

(** Every place in [content] that runs a test executable.

    An [(executable)] stanza is not one: it declares something to build, and dune runs it only where
    a rule says so — which is why a diagnostic executable such as [bench_circles_step] or a tutorial
    such as [gpt2_generate] needs no exemption from the check built on this. It is structurally not
    a site, rather than a name on a list someone has to keep true.

    What a rule runs is read from the command position of its [run] actions, in every spelling
    {!classify_command} places — and a command it cannot place becomes an {!Unreadable_command}
    site, which the caller fails on. *)
let sites content = walk "" (stanzas content) ~f:sites_of_stanza

(** Stanza heads in [content] that {!sites} has no classification for, each once. The caller fails
    on them: see {!inert_heads} for why silence is not an option here. *)
let unclassified_heads content =
  walk "" (stanzas content) ~f:(fun _subdir stanza ->
      match head stanza with
      | Some h
        when List.mem action_heads h ~equal:String.equal
             || List.mem inert_heads h ~equal:String.equal
             || List.mem [ "test"; "tests"; "library"; "subdir" ] h ~equal:String.equal ->
          []
      | Some h -> [ h ]
      (* A bare atom or a list that does not start with one is not a stanza dune would accept, so it
         is reported the same way rather than passed over. *)
      | None -> [ "<not a stanza>" ])
  |> List.dedup_and_sort ~compare:String.compare

(** Stanzas that override command resolution for a whole directory: PATH bindings and
    [(env (_ (binaries …)))] mappings.

    There a bare command name may resolve to something this repository builds, so every
    classification that reads one off an atom is unreliable -- and unlike the action-local
    [(setenv PATH …)], the effect reaches other dune files, since an [env] stanza applies to
    subdirectories too. A [binaries] mapping poses the same problem without spelling PATH.
    [path_rewriting_stanza_scopes] retains the [(subdir …)] placement so a caller can refuse exactly
    the affected directory tree; [path_rewriting_stanzas] is the older head-only census used by
    checks that reject every occurrence. This repository has no [env] stanza at all, and the day one
    overrides command resolution those checks say so rather than quietly reading bare names as tools
    (Codex P2, round 17 of PR #343; PR #623 rounds 12-13). *)
let path_rewriting_stanza_scopes content =
  (* The NAME position of an `env-vars` binding, not any atom in the stanza: setting some other
     variable to the literal value `PATH` rewrites nothing. `binaries` is itself a command lookup
     override (Codex P2, round 18 of PR #343; round 13 of PR #623). *)
  let rec overrides_resolution sexp =
    match sexp with
    | Sexp.List (Sexp.Atom "env-vars" :: bindings) ->
        List.exists bindings ~f:(function
          | Sexp.List (Sexp.Atom name :: _) -> String.equal (String.uppercase name) "PATH"
          | _ -> false)
    | Sexp.List (Sexp.Atom "binaries" :: _) -> true
    | Sexp.List l -> List.exists l ~f:overrides_resolution
    | Sexp.Atom _ -> false
  in
  walk "" (stanzas content) ~f:(fun subdir stanza ->
      match head stanza with Some "env" when overrides_resolution stanza -> [ subdir ] | _ -> [])
  |> List.dedup_and_sort ~compare:String.compare

let path_rewriting_stanzas content =
  path_rewriting_stanza_scopes content
  |> List.map ~f:(fun _subdir -> "env")
  |> List.dedup_and_sort ~compare:String.compare

(** The directories this dune file materializes the shared configuration into with a
    [(copy_files …ocannl_config)] stanza, relative to its own as in {!site}. The other way for a
    directory to have one is a file checked in next to the dune file, which this cannot see and the
    caller supplies. *)
let config_copy_dirs content =
  walk "" (stanzas content) ~f:(fun subdir stanza ->
      match head stanza with
      (* [copy_files#] is the preprocessing spelling of the same stanza. A wildcard is read as
         possibly matching the config: this decides where a config EXISTS, so guessing wide only
         risks accepting a directory that has one by another route, while guessing narrow would
         reject a correctly configured one (Codex P2, round 12). *)
      | Some ("copy_files" | "copy_files#")
        when List.exists (atoms stanza) ~f:(fun atom ->
                 glob_could_match (Stdlib.Filename.basename atom) ~name:config_file) ->
          [ subdir ]
      | _ -> [])

(** {1 The backend marker (gh-ocannl-659)}

    [env_var_deps] (gh-ocannl-628) checks that a stanza declaring one spelling of an ambient
    variable declares both. What it could not see is a stanza declaring NEITHER: a backend-sensitive
    test added with no [(env_var OCANNL_BACKEND)] at all is invisible to a pairing check, and dune
    then serves the previous backend's output as a pass under [OCANNL_BACKEND=cuda dune build @…] —
    the exact failure the declaration exists to prevent.

    The rule that closes it is an exclusive or, over every stanza that runs an executable: either
    the stanza declares [(env_var OCANNL_BACKEND)], or it carries a marker comment saying which
    backend it is pinned to — or none at all — and why. Both absent is the hole; both present is
    contradictory intent, since a stanza that names its backend has nothing to invalidate on.

    The marker is a COMMENT, in place, rather than a line in a central list. A central per-stanza
    list is a churn and conflict magnet (the gh-ocannl-665 lesson), and — the reason that decides it
    — the next author copies the stanza next to the one they are writing, so the teaching text has
    to live there rather than in a file they will never open (the recurrence mechanism gh-ocannl-668
    diagnosed). *)

(* [backend_env_var] and [declares_env_var] are defined up with [declared_config_paths]: the
   declaration is read from a site's own dependency field, so it has to be available where a site is
   built. *)

(** What makes a comment a marker rather than prose. Distinctive enough that a check can also ask
    whether every occurrence of it in a dune file became a marker, which is how a misplaced or
    misspelled one is caught rather than silently reading as no marker at all. *)
let marker_sentinel = "ocannl-backend:"

(** The words the marker admits in the backend position.

    [none] says the run does not depend on the configured backend AT ALL — usually because it links
    none, and sometimes because it links one and reaches no context. It is about the RUN, not the
    link line, since what the declaration protects is the output: a test that links [ocannl] for its
    DSL modules and calls a parser has nothing for [OCANNL_BACKEND] to invalidate.

    The rest are the backends OCANNL has, for a stanza that NAMES one instead of selecting it.
    Adding a backend adds a word here — and a marker naming a word that is not one of these fails,
    which is the point: [; ocannl-backend: metl -- …] would otherwise read as a truthful exemption.

    Kept as text rather than taken from [Backends.get_backend] on purpose: the scanning tests link
    [arrayjit.utils] and the source scanners, and pulling the whole backend closure — Metal and CUDA
    bindings included — into a check that reads dune files would trade a six-word list for a link
    line that has to resolve on every platform.

    That leaves a restatement, and a restatement nothing relates to its original goes stale in the
    worse direction: a backend added here would make the TRUTHFUL marker for it fail as malformed,
    whose remedy is to reach for [none] — a lie this grammar accepts. So the relationship is
    asserted where the link cost is already paid rather than here:
    [test/operations/marker_backend_vocabulary] holds this list equal to the names of
    [Backends.all_of_backend] plus ["none"], and fails whichever side moves alone (gh-ocannl-689).
*)
let marker_backends = [ "none"; "cc"; "multidev_cc"; "cuda"; "hip"; "metal" ]

(** The separators the marker admits between the backend and the reason. The em dash is what this
    repository's prose uses; [--] is what an ASCII keyboard produces, and refusing it would be a
    grammar that fails for a reason nobody can see in a diff. *)
let marker_separators = [ "--"; "\xe2\x80\x94" ]

type marker_malformed =
  | Repeated_sentinel
  | Missing_reason_separator
  | Short_reason of string
  | Declaration_error of string

type 'a marker_text =
  | Not_a_marker
  | Parsed_marker of 'a
  | Malformed_marker_text of marker_malformed list

let marker_malformed_reason ~sentinel ~separator_subject ~grammar = function
  | Repeated_sentinel ->
      Printf.sprintf
        "two `%s` declarations in one comment -- the second would be read as part of the first's \
         reason; put each on its own line"
        sentinel
  | Missing_reason_separator ->
      Printf.sprintf "no `--` separating the %s from the reason -- the grammar is `; %s`"
        separator_subject grammar
  | Short_reason reason -> Printf.sprintf "the reason `%s` is one word -- say why, not what" reason
  | Declaration_error why -> why

(** [parse_marker_text] is the grammar shared by every machine-checked Dune comment marker.

    A marker announces itself by containing [sentinel] anywhere in a [;] comment. It contains that
    sentinel exactly once, separates its declaration from a reason with the earliest admitted
    spelling of [--] or an em dash, and gives a reason of at least two words. [parse_declaration]
    owns only the convention-specific declaration to the left of that separator.

    Nothing malformed is normalised: a second sentinel, a missing separator, or a one-word reason is
    returned as [Malformed_marker_text]. This is the common outer grammar; the contained-marker
    contract below adds comment, stanza, and subject attribution. *)
let parse_marker_text ~sentinel ~parse_declaration text =
  let trimmed = String.strip text in
  match String.substr_index trimmed ~pattern:sentinel with
  | None -> Not_a_marker
  | Some at -> (
      let rest = String.strip (String.subo trimmed ~pos:(at + String.length sentinel)) in
      if Option.is_some (String.substr_index rest ~pattern:sentinel) then
        Malformed_marker_text [ Repeated_sentinel ]
      else
        let split =
          List.filter_map marker_separators ~f:(fun separator ->
              Option.map (String.substr_index rest ~pattern:separator) ~f:(fun index ->
                  (index, separator)))
          |> List.min_elt ~compare:(fun (a, _) (b, _) -> Int.compare a b)
          |> Option.map ~f:(fun (index, separator) ->
              ( String.strip (String.sub rest ~pos:0 ~len:index),
                String.strip (String.subo rest ~pos:(index + String.length separator)) ))
        in
        match split with
        | None -> Malformed_marker_text [ Missing_reason_separator ]
        | Some (declaration, reason) -> (
            if
              List.length
                (String.split_on_chars reason ~on:[ ' '; '\t' ]
                |> List.filter ~f:(Fn.non String.is_empty))
              < 2
            then Malformed_marker_text [ Short_reason reason ]
            else
              match parse_declaration ~declaration ~reason with
              | Ok marker -> Parsed_marker marker
              | Error why -> Malformed_marker_text (List.map why ~f:(fun s -> Declaration_error s)))
      )

type marker_body = { backend : string; reason : string }
(** [backend] is the comma-separated list as normalised by {!parse_marker}: the words with their
    spacing removed, so a caller can split it on [','] without re-trimming. *)

type marker =
  | Marker of marker_body
  | Malformed of string  (** what is wrong with it, phrased for the author of the comment *)

(** [parse_marker text] reads the text of one [;] comment (everything after the semicolon).

    [None] means it is not a marker at all — ordinary prose, which a dune file is full of.
    [Some (Malformed …)] means it announced itself as one and does not parse, which is a failure
    rather than a shrug: a marker the grammar rejects would otherwise leave its stanza declaring
    nothing, reported as if the author had written no marker.

    The grammar, all of it:
    {v ; ocannl-backend: <backend>[,<backend>…] -- <reason> v}
    where each [<backend>] is one of {!marker_backends}, the separator is one of
    {!marker_separators}, and [<reason>] is at least two words. A reason is required even for
    [none], and a one-word reason is a label rather than a reason: writing down WHICH backend and
    WHY is the friction that stops a reflexive exemption, and a grammar that accepts
    [; ocannl-backend: none -- pure] has given that away. A reason too long for one line continues
    as an ordinary comment on the next, which needs no grammar of its own.

    Everything the grammar refuses it refuses OUT LOUD, and nothing it can repair does it repair.
    That distinction is the whole value of the construct: this is the one comment in the tree whose
    job is to be checkable, so an empty entry between commas, a backend named twice, or a second
    declaration sharing the line are all {!Malformed} rather than normalised away — silently reading
    [cc,] as [cc] would hand back a clean answer for a marker its author got wrong. *)
let parse_backend_declaration ~declaration:backend ~reason =
  (* A stanza may name more than one backend -- `data_parallel` runs the same model on cc and on
     multidev_cc -- and writing both is more truthful than picking one. `none` makes no such pair.

     Every rejection is a rejection rather than a normalisation. Dropping an empty entry would read
     `cc,` and `cc,,metal` as a clean `cc`/`cc,metal`, and deduplicating would read `cc,cc` as `cc`
     -- silently repairing a typo in the one place whose purpose is to be wrong out loud. *)
  let named = String.split backend ~on:',' |> List.map ~f:String.strip in
  if String.is_empty backend then Error [ "no backend named before the reason" ]
  else if List.exists named ~f:String.is_empty then
    Error
      [
        Printf.sprintf
          "`%s` has an empty entry between commas -- name each backend, or drop the comma" backend;
      ]
  else if List.exists named ~f:(fun word -> not (List.mem marker_backends word ~equal:String.equal))
  then
    Error
      [ Printf.sprintf "`%s` is not one of %s" backend (String.concat ~sep:", " marker_backends) ]
  else if List.contains_dup named ~compare:String.compare then
    Error [ Printf.sprintf "`%s` names the same backend twice" backend ]
  else if List.mem named "none" ~equal:String.equal && List.length named > 1 then
    Error
      [
        Printf.sprintf
          "`%s` says both that the run depends on a backend and that it depends on none" backend;
      ]
  else Ok { backend = String.concat ~sep:"," named; reason }

let parse_marker text =
  match
    parse_marker_text ~sentinel:marker_sentinel ~parse_declaration:parse_backend_declaration text
  with
  | Not_a_marker -> None
  | Parsed_marker marker -> Some (Marker marker)
  | Malformed_marker_text malformed ->
      Some
        (Malformed
           (List.map malformed
              ~f:
                (marker_malformed_reason ~sentinel:marker_sentinel ~separator_subject:"backend"
                   ~grammar:
                     (Printf.sprintf "%s <%s> -- <reason>" marker_sentinel
                        (String.concat ~sep:"|" marker_backends)))
           |> String.concat ~sep:"; "))

type marked_stanza = {
  marked_head : string;  (** the atom the stanza opens with, or ["<not a stanza>"] *)
  marked_name : string;  (** its [(name …)]/[(names …)], joined, for a diagnostic *)
  marked_line : int;  (** the line its opening parenthesis sits on *)
  marked_sites : site list;  (** what it runs; empty means it is not subject to the rule *)
  marked_raw_subject : bool;
      (** what {!raw_runs_something} — the SECOND reader — makes of the same stanza. Carried here so
          the floor can be checked per stanza against [marked_sites] instead of as a total: two
          answers about one stanza cannot be traded off against a third stanza the way two counts
          over a file can. *)
  marked_declares_backend : bool;  (** whether it declares [(env_var OCANNL_BACKEND)] *)
  marked_comments : (int * string) list;
      (** the comments inside its parentheses, each with the line it sits on — not those of a
          [(subdir …)] this walk descends past, since those belong to no stanza *)
  marked_subdir : string;
      (** the [(subdir …)] path this stanza was found under, relative to the dune file, or [""] at
          the top level — so a caller can resolve its modules and its aliases in the directory dune
          actually applies it to *)
  marked_sexp : Sexp.t;
      (** the stanza itself. Carried so that a caller which decided something FROM the marker can go
          on to ask the stanza the ordinary structural questions — which aliases it attaches to,
          which modules it names — without having to find it again by name, which for a [(rule …)]
          (the placement a marker takes on an [(executable)]'s runner) is not possible at all. *)
}

(** [marked_stanzas content] is {!sites} again, per stanza and with the comments each stanza
    encloses — the two questions gh-ocannl-659's rule asks of one stanza at once.

    A comment belongs to a stanza when it sits BETWEEN ITS PARENTHESES. That is the whole
    attribution rule, and it is deliberately not "the comment above the stanza": this repository's
    dune files habitually leave a blank line between a comment block and the stanza it introduces,
    so an adjacency rule would have to guess how far above to look, and would hand a marker to the
    wrong stanza the first time someone left a note between two rules. Containment is decided by the
    file's own structure and cannot be moved by whitespace. *)
let marked_stanzas content =
  let parsed = stanzas content in
  let raw, comments = read_raw content in
  let enclosed form =
    List.filter_map comments ~f:(fun c ->
        if c.comment_start >= form.raw_start && c.comment_start < form.raw_stop then
          Some (line_of content c.comment_start, c.comment_text)
        else None)
  in
  let rec go dir form sexp =
    match (sexp, form.raw_children) with
    (* A `(subdir …)` holds stanzas and is not one: the walk descends, exactly as {!walk} does, so
       what is reported is the stanzas dune will apply -- and a comment sitting in the subdir but in
       none of its stanzas is attributed to nothing, which is what makes a misplaced marker
       reportable rather than invisible. *)
    | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body), _ :: _ :: body_forms
      when List.length body_forms = List.length body ->
        List.concat (List.map2_exn body_forms body ~f:(go (in_subdir dir sub)))
    | _ ->
        let sites = sites_of_stanza dir sexp in
        [
          {
            marked_head = (match head sexp with Some h -> h | None -> "<not a stanza>");
            marked_name = String.concat ~sep:", " (names_of sexp);
            marked_line = line_of content form.raw_start;
            marked_sites = sites;
            (* The same stanza, put to the other reader. `raw_stanza_of` returns nothing for a form
               that is not a stanza at all, which is itself an honest "runs nothing". *)
            marked_raw_subject = List.exists (raw_stanza_of ~subdir:dir sexp) ~f:raw_runs_something;
            (* Read from the SITES' own dependency fields, not from the stanza as a whole. Which
               field carries a site's deps is already worked out per site -- an inline-test library
               declares under `(inline_tests (deps …))`, not in the library stanza at large -- and
               asking the stanza instead would certify a test action against a declaration it does
               not run under (Codex P2, round 3). A stanza that runs nothing declares nothing, which
               is what the XOR's "a marker here declares nothing" arm already says of it. *)
            marked_declares_backend = List.exists sites ~f:(fun s -> s.declares_backend);
            marked_comments = enclosed form;
            marked_subdir = dir;
            marked_sexp = sexp;
          };
        ]
  in
  List.concat (List.map2_exn raw parsed ~f:(go ""))

type 'a contained_marker = {
  contained_line : int;
  contained_text : string;
  contained_value : 'a;
  containing_stanza : marked_stanza;
}
(** A successfully parsed marker and the stanza whose parentheses contain it. *)

type 'a marker_stanza = { marker_stanza : marked_stanza; stanza_markers : 'a contained_marker list }
(** A stanza paired with the valid markers the contained-marker contract attributed to it. *)

(** Every refusal the generic contained-marker contract can make.

    Convention-specific declaration errors arrive as [Malformed_marker]; the other constructors are
    the reusable structural contract. A well-formed marker must be in a Dune comment, inside one
    stanza, and accepted by that convention's [belongs] predicate. *)
type marker_issue =
  | Malformed_marker of {
      issue_line : int;
      issue_text : string;
      issue_malformed : marker_malformed;
    }
  | Marker_outside_stanza of { issue_line : int; issue_text : string }
  | Marker_in_wrong_stanza of {
      issue_line : int;
      issue_text : string;
      issue_stanza : marked_stanza;
      issue_why : string;
    }
  | Marker_outside_comment of { issue_text_occurrences : int; issue_comment_occurrences : int }

type 'a marker_contract = {
  contract_stanzas : 'a marker_stanza list;
  contract_issues : marker_issue list;
  contract_text_occurrences : int;
  contract_comment_occurrences : int;
  contract_stanza_occurrences : int;
      (** how many of the [contract_comment_occurrences] sit in a comment the walk attributed to a
          stanza — the count a placement diagnostic actually wants, since a marker in a comment
          between stanzas declares nothing. It is the contract's own answer rather than a
          subtraction the caller performs: deriving it as "comments minus the ones the
          [Marker_outside_stanza] issues mention" makes every consumer re-implement the relationship
          between an issue and the occurrences it accounts for, and quietly returns the wrong number
          the moment an issue carries more than one occurrence or the contract gains another
          placement refusal. Malformed and wrong-stanza markers are counted here: they were placed
          inside a stanza, and what is wrong with them is said by their own issues. *)
}

let sentinel_occurrences ~sentinel text =
  let rec count from found =
    match String.substr_index text ~pos:from ~pattern:sentinel with
    | None -> found
    | Some at -> count (at + String.length sentinel) (found + 1)
  in
  if String.is_empty sentinel then 0 else count 0 0

(** [contained_marker_contract] applies the complete reusable marker contract to one Dune file.

    [parse_declaration] receives the containing stanza, the convention-specific declaration, and the
    already-validated reason. [belongs] states which containing stanza may carry the parsed marker.
    The result always retains every stanza; invalid markers become issues and are not attributed as
    valid markers.

    Sentinel accounting deliberately compares the dumb text count with the lexer-derived comment
    count. This makes a marker in a quoted action argument or ordinary field a refusal rather than
    invisible text. Comments inside a [(subdir ...)] wrapper but no child stanza likewise become
    [Marker_outside_stanza], because {!marked_stanzas} attributes by actual parentheses. The three
    counts nest — [contract_stanza_occurrences] is at most [contract_comment_occurrences], which is
    at most [contract_text_occurrences] — so a consumer states whichever placement it cares about by
    comparing two of them rather than by reconstructing one. *)
let contained_marker_contract content ~sentinel ~parse_declaration ~belongs =
  let stanzas = marked_stanzas content in
  let _, comments = read_raw content in
  let comment_occurrences =
    List.sum
      (module Int)
      comments
      ~f:(fun comment -> sentinel_occurrences ~sentinel comment.comment_text)
  in
  let issues = ref [] in
  let markers = ref [] in
  let stanza_occurrences = ref 0 in
  let stanza_of_comment line text =
    List.find stanzas ~f:(fun stanza ->
        List.exists stanza.marked_comments ~f:(fun (candidate_line, candidate_text) ->
            Int.equal line candidate_line && String.equal text candidate_text))
  in
  List.iter comments ~f:(fun comment ->
      let line = line_of content comment.comment_start in
      let text = comment.comment_text in
      let occurrences = sentinel_occurrences ~sentinel text in
      if occurrences > 0 then
        match stanza_of_comment line text with
        | None ->
            issues := Marker_outside_stanza { issue_line = line; issue_text = text } :: !issues
        | Some stanza -> (
            stanza_occurrences := !stanza_occurrences + occurrences;
            match
              parse_marker_text ~sentinel ~parse_declaration:(parse_declaration stanza) text
            with
            | Not_a_marker -> ()
            | Malformed_marker_text malformed ->
                List.iter malformed ~f:(fun issue_malformed ->
                    issues :=
                      Malformed_marker { issue_line = line; issue_text = text; issue_malformed }
                      :: !issues)
            | Parsed_marker value -> (
                match belongs stanza value with
                | Error reasons ->
                    List.iter reasons ~f:(fun issue_why ->
                        issues :=
                          Marker_in_wrong_stanza
                            {
                              issue_line = line;
                              issue_text = text;
                              issue_stanza = stanza;
                              issue_why;
                            }
                          :: !issues)
                | Ok () ->
                    markers :=
                      {
                        contained_line = line;
                        contained_text = text;
                        contained_value = value;
                        containing_stanza = stanza;
                      }
                      :: !markers)));
  let text_occurrences = sentinel_occurrences ~sentinel content in
  if not (Int.equal text_occurrences comment_occurrences) then
    issues :=
      Marker_outside_comment
        {
          issue_text_occurrences = text_occurrences;
          issue_comment_occurrences = comment_occurrences;
        }
      :: !issues;
  let markers = List.rev !markers in
  {
    contract_stanzas =
      List.map stanzas ~f:(fun marker_stanza ->
          {
            marker_stanza;
            stanza_markers =
              List.filter markers ~f:(fun marker ->
                  phys_equal marker.containing_stanza marker_stanza);
          });
    contract_issues = List.rev !issues;
    contract_text_occurrences = text_occurrences;
    contract_comment_occurrences = comment_occurrences;
    contract_stanza_occurrences = !stanza_occurrences;
  }

(** How gh-ocannl-659's rule reads ONE stanza: whether it is subject to the rule, and which of the
    two declarations it carries.

    Only the DECISION lives here. The diagnostics stay with the check, which owns their wording and
    their counters — what moves is the part that has to be put to a stanza the repository does not
    contain, so that "a stanza running its test through [bash] and declaring nothing is reported"
    can be asserted on synthetic text rather than inferred from the live tree (gh-ocannl-690; the
    same shape gh-ocannl-603 asks for on [config_dep_completeness]'s resolution half). *)
type backend_rule =
  | Runs_nothing  (** not subject to the rule, and carrying no marker: nothing to say of it *)
  | Declares_variable  (** subject, and declares [(env_var OCANNL_BACKEND)] *)
  | Names_backend of int * marker_body  (** subject, and carrying exactly one well-formed marker *)
  | Declares_and_names of int * marker_body
      (** subject, and carrying both — a stanza that names its backend has nothing for the variable
          to invalidate *)
  | Names_twice of int  (** subject, and carrying more than one marker: the line of the second *)
  | Names_neither  (** subject, and carrying neither — the hole gh-ocannl-659 closed *)

(** Apply the shared contained-marker contract to the backend declaration. A marker on a stanza that
    runs nothing is a wrong-stanza issue and is not handed to {!backend_rule_of}. *)
let backend_marker_contract content =
  contained_marker_contract content ~sentinel:marker_sentinel
    ~parse_declaration:(fun _stanza -> parse_backend_declaration)
    ~belongs:(fun stanza _marker ->
      if List.is_empty stanza.marked_sites then Error [ "the containing stanza runs no executable" ]
      else Ok ())

let backend_rule_of marked =
  let stanza = marked.marker_stanza in
  let well_formed =
    List.map marked.stanza_markers ~f:(fun marker ->
        (marker.contained_line, marker.contained_value))
  in
  let subject = not (List.is_empty stanza.marked_sites) in
  match (subject, stanza.marked_declares_backend, well_formed) with
  | false, _, [] -> Runs_nothing
  | false, _, _ :: _ -> assert false
  | true, _, _ :: (line, _) :: _ -> Names_twice line
  | true, true, [ (line, m) ] -> Declares_and_names (line, m)
  | true, true, [] -> Declares_variable
  | true, false, [ (line, m) ] -> Names_backend (line, m)
  | true, false, [] -> Names_neither

(** {1 The artifact-directory declaration (gh-ocannl-723)}

    [Test_utils.Generated.init] reads the [build_files_prefix] configuration key: it decides whether
    the artifact directory is this process's to empty, and refuses the run outright where it is not.
    So a stanza whose executable calls it and does not declare [(env_var OCANNL_BUILD_FILES_PREFIX)]
    is one dune will not rerun when that variable changes — the gh-ocannl-628 hole, one key over,
    and with the same consequence: the previous run's result served as a pass.

    The declaration was a convention held by copying a neighbour, and on 2026-08-22 two PRs shipped
    without it and were caught by a reviewer's eye rather than by the build. What this section adds
    is the relationship itself, asked where the link cost is already paid: a stanza names its
    modules, a module's source either calls the initializer or does not, and the two answers have to
    agree.

    Where the declaration has to sit is what dune's own semantics decide, and it is not the same
    place for every stanza. A [(test)]/[(tests)] stanza dune runs itself, under its own [(deps …)];
    an inline-test library runs under [(inline_tests (deps …))]; an [(executable)] has no [deps]
    field at all, so the rule that RUNS it is what a change of the variable has to invalidate — the
    same placement as the [ocannl_config] dep and the backend marker. *)

let artifact_env_var = "OCANNL_BUILD_FILES_PREFIX"

(** {2 Which rules run which program}

    Shared by every check phrased as "the stanza dune runs this module under declares X": the
    artifact one below, and the ambient-guard one in [env_var_deps] (gh-ocannl-749). Lifted out of
    {!artifact_subjects} rather than copied, since a second copy of the identity rules is exactly
    the restatement those checks exist to replace. *)

(* The path AS WRITTEN, which is the executable's identity here for the reason {!program_path}
   gives: `../support/probe.exe` and a local `probe.exe` are different programs, and reducing both
   to a basename made a rule running the first count as the runner of the second -- crediting a
   local executable with a declaration made elsewhere, and hiding that nothing runs it (Codex P2,
   round 2 of PR #457). *)
let exes_run stanza =
  List.filter_map (executables_run stanza) ~f:(fun (_cwd, command) ->
      match command with
      | Runs path | Runs_dependency path -> Some (`File path)
      | Runs_public name -> Some (`Public name)
      | _ -> None)
  |> List.dedup_and_sort ~compare:Poly.compare

(** A directory-qualified path with its [.] and [..] segments resolved, so that two spellings of one
    program compare equal and two programs never do. [..] above the root is kept, since a path
    leaving the workspace names something this scan cannot place and must not conflate with a local
    program. *)
let normalize_path path =
  let parts = String.split path ~on:'/' in
  List.fold parts ~init:[] ~f:(fun acc part ->
      match part with
      | "" | "." -> acc
      | ".." -> ( match acc with ".." :: _ | [] -> ".." :: acc | _ :: rest -> rest)
      | part -> part :: acc)
  |> List.rev |> String.concat ~sep:"/"

(** A file command's path relative to its stanza. Literal paths follow [chdir]; dependency
    expansions retain the stanza-relative path Dune resolves before the action runs. This says
    nothing about the process's configuration search, which still starts from its action cwd. *)
let command_file_path ~cwd = function
  | Runs path -> Some (in_subdir cwd path)
  | Runs_dependency path -> Some path
  | Runs_public _ | External | Unrecognized _ | Unknown_directory _ | Path_rewritten _ -> None

(** Every program a stanza runs, resolved to a workspace-relative path, with the variables pinned
    around THAT run.

    For literal paths the working directory is part of the identity: [(chdir ../a (run probe.exe))]
    in a rule under [b] runs [a]'s program, not [b]'s, so resolving only the rule's own subdirectory
    would report the real program as unrun and credit a same-named local one with this rule's
    declarations (Codex P2, round 4 of PR #484). [commands_in] already tracked the directory for the
    configuration search; this is the same fact answering a second question. Dependency expansions
    instead keep the stanza-relative path from which Dune resolves them. *)
let runs_of_with_multiplicity ~subdir stanza =
  List.filter_map (executables_run_with_pins_preserving_multiplicity stanza)
    ~f:(fun (cwd, pinned, command) ->
      match command with
      (* A FILE is resolved to a workspace-relative path -- the working directory is part of its
         identity, since `(chdir ../a (run probe.exe))` in a rule under `b` runs `a`'s program. A
         PUBLIC name is not a path and is carried as written: `%{bin:pkg.probe}` names the same
         program from anywhere, and `classify_command` keeps the two apart precisely so a `(run
         ./pkg.probe)` here is not credited to the executable installed under that name
         (gh-ocannl-783). *)
      | (Runs _ | Runs_dependency _) as command ->
          Option.map (command_file_path ~cwd command) ~f:(fun path ->
              (`File (normalize_path (in_subdir subdir path)), pinned))
      | Runs_public name -> Some (`Public name, pinned)
      | _ -> None)

(** The set-like counterpart to {!runs_of_with_multiplicity}. *)
let runs_of ~subdir stanza =
  runs_of_with_multiplicity ~subdir stanza |> List.dedup_and_sort ~compare:Poly.compare

(** The public names a stanza gives, in the order it gives them. [(public_names a b)] pairs
    POSITIONALLY with [(names a b)], which is what lets one name of an [(executables …)] be asked
    about on its own (gh-ocannl-747); [-] is dune's placeholder for a name that is not installed. *)
let public_names stanza =
  match field stanza "public_names" with
  | Some args -> List.filter_map args ~f:(function Sexp.Atom p -> Some p | _ -> None)
  | None -> ( match field stanza "public_name" with Some [ Sexp.Atom p ] -> [ p ] | _ -> [])

(* Both identities an executable can be run under: the local `probe.exe` a `%{dep:…}` names, and the
   public name a `%{bin:pkg.probe}` resolves to, which `classify_command` already records as `Runs
   "pkg.probe"`. Searching only for the first left a public-name runner unrecognised, and its
   executable reported as though nothing ran it (Codex P2, round 3 of PR #457).

   A rule OUTSIDE a `(subdir gen …)` runs the executable declared inside it as `gen/probe.exe`, so
   the executable answers to both spellings (Codex P2, round 4). *)
let program_public_name stanza ~index =
  match List.nth (public_names stanza) index with
  | Some public when not (String.equal public "-") -> Some public
  | _ -> None

(** Each PROGRAM an [(executable)]/[(executables)] stanza declares, paired with the stanzas that run
    it. One name is one program: a rule running `b.exe` is not a runner of `a` (gh-ocannl-747).

    [runner_stanzas] pairs each candidate runner with the SUBDIRECTORY it was found in, and defaults
    to [stanzas] in [subdir]. A caller descending into a [(subdir …)] passes the whole file, since a
    top-level rule may run a nested executable — and it has to pass each rule's own subdirectory
    along with it, because the path a rule writes is relative to where the rule lives. Resolving
    both sides to a directory-qualified path is what tells `(subdir a (rule … probe.exe))` from
    `(subdir b (rule … probe.exe))`: comparing the written path against an unqualified `probe.exe`
    made each a runner of the other's program, so an unrun executable in [a] could inherit [b]'s
    declaration (Codex P2, round 3 of PR #484). It is also what keeps `../support/probe.exe` a
    different program from the local one, which is why the path is matched AS WRITTEN and not by
    basename (Codex P2, round 2 of PR #457).

    A public name is not a path and is compared as written: [%{bin:pkg.probe}] names the same
    program from anywhere, which is what `classify_command` records as {!Runs_public}
    (gh-ocannl-783). *)
let program_runners ?(subdir = "") ?runner_stanzas stanzas stanza =
  let runner_stanzas =
    match runner_stanzas with
    | Some runners -> runners
    | None -> List.map stanzas ~f:(fun s -> (subdir, s))
  in
  let names = match names_of stanza with [] -> [ "<unnamed>" ] | names -> names in
  List.mapi names ~f:(fun index name ->
      let canonical = normalize_path (in_subdir subdir (name ^ ".exe")) in
      let public = program_public_name stanza ~index in
      ( name,
        List.filter_map runner_stanzas ~f:(fun (runner_subdir, s) ->
            (* The runs of THIS program, and the variables pinned around each of them. A pin counts
               only where it is in force at every run of this program: an unpinned run of a HELPER
               beside it says nothing about the subject (Codex P2, round 4), and a pin around the
               helper alone protects nothing here (Codex P1, round 1). *)
            let mine =
              List.filter_map (runs_of ~subdir:runner_subdir s) ~f:(fun (identity, pinned) ->
                  match (identity, public) with
                  | `File resolved, _ when String.equal resolved canonical -> Some pinned
                  | `Public name, Some public when String.equal name public -> Some pinned
                  | _ -> None)
            in
            match mine with
            | [] -> None
            | first :: rest -> Some (s, List.fold rest ~init:first ~f:Set.inter)) ))

(** Dune's own main-module rule (gh-ocannl-747): the executable named [a] is built from the module
    [a] of the stanza's module set, and every module that is no name's main module is linked into
    all of them. Matched case-insensitively, since a module name is the capitalized source basename
    and the [(names …)] field spells the file's.

    What it decides is attribution. Before it, an [(executables (names a b) (modules a b))] stanza
    combined both programs into one subject, so a rule running EITHER counted as a runner of both:
    with `a.ml` calling the initializer and only `b.exe`'s rule omitting the declaration, `a` was
    reported undeclared over a rule that runs neither its main module nor its initializer. *)
let main_module_of modules name =
  List.find modules ~f:(fun m -> String.equal (String.lowercase m) (String.lowercase name))

(** The modules of [stanza] that belong to the program called [name]: its own main module, plus
    every module that is no name's main module and so is linked into all of them. *)
let program_modules stanza ~modules ~name =
  let mains = List.filter_map (names_of stanza) ~f:(main_module_of modules) in
  Option.to_list (main_module_of modules name)
  @ List.filter modules ~f:(fun m -> not (List.mem mains m ~equal:String.equal))

(** The stanza heads that name their own modules, and so can be asked what those modules call. *)
let module_bearing_heads = [ "test"; "tests"; "executable"; "executables"; "library" ]

(** The module names a [(modules …)] field lists EXPLICITLY, or [None] where the field is absent or
    reaches for dune's default set.

    Both of those mean the same thing to a reader of the file alone: the stanza's modules are
    whatever the directory holds and no other stanza claims. [(test (name t))] with no [(modules …)]
    at all is the common shape of it — dune builds [t.ml] — and [(modules :standard \ helper)] is
    the same default written down (Codex P2, round 2). Treating either as "names no modules" made a
    stanza own nothing, which turned a required declaration into a stale one. *)
type module_set =
  | Named of string list  (** the field lists them, and no default is involved *)
  | Default_less of string list
      (** the default set, less the modules subtracted from it. [(modules :standard \ helper)] is
          this with [helper] subtracted, and an absent field is this with nothing subtracted.
          Resolving the subtraction matters as much as resolving the default: a stanza that EXCLUDES
          a module does not link it, so demanding a declaration of it would be a demand about a
          module the test never builds (Codex P2, round 3). *)

(* Dune's ordered-set language, evaluated with its GROUPING rather than flattened.

   `\` is a binary difference between what stands to its left and what stands to its right, WITHIN
   the parentheses that hold it -- so `(modules (:standard \ helper) guard)` subtracts `helper` from
   the default set and then adds `guard`, and flattening the whole field to `:standard \ helper
   guard` subtracted `guard` as well (Codex P2, round 6 of PR #484). A module wrongly subtracted is
   a module no stanza claims, which takes its reads out of every check phrased over a stanza's
   modules -- silently.

   The result of evaluating a set is what it includes, what it subtracts, and whether `:standard` is
   among its terms; a nested set contributes all three to the one that holds it. Over-subtracting
   remains the safe direction where a term cannot be evaluated, since what falls out of one stanza's
   claim is caught by the census over sources no stanza claims. *)
type ordered_set = { standard : bool; included : string list; excluded : string list }

let empty_set = { standard = false; included = []; excluded = [] }

(* A term that INCLUDES a name cancels an exclusion of it made elsewhere in the same set: `(modules
   (main guard \ guard) guard)` links `guard`, the nested difference having been undone by the term
   beside it. Keeping both and subtracting at the end dropped it (Codex P2, round 8). *)
let union a b =
  {
    standard = a.standard || b.standard;
    included = a.included @ b.included;
    excluded =
      List.filter a.excluded ~f:(fun m -> not (List.mem b.included m ~equal:String.equal))
      @ List.filter b.excluded ~f:(fun m -> not (List.mem a.included m ~equal:String.equal));
  }

let difference left right =
  {
    standard = left.standard;
    (* What the right-hand side would have ADDED is what the left loses; what it subtracts in turn
       is a nesting this language admits and dune resolves the same way. *)
    included =
      List.filter left.included ~f:(fun m -> not (List.mem right.included m ~equal:String.equal));
    (* An exclusion is only meaningful against `:standard`, whose members are not known here. Where
       the left side is a CONCRETE set the subtraction has already happened above, and carrying a
       symbolic exclusion out of it let `(guard \ guard)` -- an empty set -- take `guard` out of a
       `:standard` beside it (Codex P2, round 9 of PR #484). *)
    excluded = (if left.standard then left.excluded @ right.included else left.excluded);
  }

let rec eval_ordered_set terms =
  match List.split_while terms ~f:(fun t -> not (Sexp.equal t (Sexp.Atom "\\"))) with
  | left, [] -> List.fold left ~init:empty_set ~f:(fun acc term -> union acc (eval_term term))
  | left, _ :: right -> difference (eval_ordered_set left) (eval_ordered_set right)

and eval_term = function
  | Sexp.Atom ":standard" -> { empty_set with standard = true }
  | Sexp.Atom name -> { empty_set with included = [ name ] }
  | Sexp.List terms -> eval_ordered_set terms

let explicit_modules stanza =
  match field stanza "modules" with
  | None -> Default_less []
  | Some args ->
      let set = eval_ordered_set args in
      (* A subtraction has already removed what it removes, so what is left is the set. Under
         `:standard` the exclusions are what a caller needs, the default set not being known
         here. *)
      if set.standard then Default_less set.excluded
      else Named (List.dedup_and_sort set.included ~compare:String.compare)

(** The modules a stanza owns, given every module the directory holds. Dune's default set is the
    directory less what other stanzas claim, which is what makes an explicit list elsewhere in the
    file narrow this one — and less whatever this stanza itself subtracts. *)
let modules_of ?(directory_modules = []) stanzas stanza =
  match explicit_modules stanza with
  | Named modules -> modules
  | Default_less excluded ->
      let claimed =
        List.concat_map stanzas ~f:(fun other ->
            match explicit_modules other with Named modules -> modules | Default_less _ -> [])
        @ excluded
        |> List.map ~f:String.lowercase
        |> Set.of_list (module String)
      in
      List.filter directory_modules ~f:(fun m -> not (Set.mem claimed (String.lowercase m)))

type artifact_verdict =
  | Artifact_declared  (** the deps this stanza's run happens under name the variable *)
  | Artifact_undeclared  (** they do not, which is the hole *)
  | Artifact_stale_declaration
      (** they do, and no module of this stanza reads [build_files_prefix] at all: a declaration
          with nothing behind it, which is the restatement this check exists to replace *)
  | Artifact_other_reader
      (** they do, no module calls the initializer, and one reads the key directly — a declaration
          this check has no business removing. Calling the initializer is the usual reason to need
          the variable tracked, not the only one (Codex P2, round 2). *)
  | Artifact_unrun
      (** an [(executable)] whose modules call the initializer and which no stanza in this dune file
          runs — so there is no [deps] field anywhere that answers for it *)
  | Artifact_in_library
      (** a plain [(library)] whose modules call it. The initializer empties the artifact directory
          of the process that owns it, so it belongs to an executable's own modules; reached through
          a library it would put the requirement on every stanza that links the library, which is
          not a relationship this scan — or any other — follows. *)

type artifact_subject = {
  artifact_head : string;
  artifact_name : string;
  artifact_callers : string list;
      (** the stanza's modules that call the initializer, in the order [(modules …)] lists them *)
  artifact_readers : string list;
      (** its modules that read [build_files_prefix] by name and do NOT call the initializer. They
          need the variable tracked for the same reason and are subject to the same rule: the
          initializer is the usual way to read the key, not the only one (Codex P2, rounds 2 and 3).
      *)
  artifact_deps_site : string;
      (** where the declaration was looked for, in words a diagnostic can use: the stanza's own
          dependency field, or the rules that run its executable *)
  artifact_verdict : artifact_verdict;
}

(** Every stanza in [stanzas] the rule has an opinion about, given [calls], which answers whether
    one module name's source calls the initializer. A stanza with no caller among its modules and no
    declaration of its own is not a subject and is not reported. *)
let artifact_subjects ?(directory_modules = []) ?(subdir = "") ?runner_stanzas stanzas ~calls
    ~reads_prefix =
  (* Each candidate runner with the subdirectory it was found in; see {!program_runners}. *)
  let runner_stanzas =
    Option.value runner_stanzas ~default:(List.map stanzas ~f:(fun s -> (subdir, s)))
  in
  (* Whether a stanza RUNS something, in the widest sense {!executables_run} admits -- a named
     executable, a command it could not place, a program under an unresolvable `chdir`. That is what
     decides whether the converse question below is this stanza's to answer: a stanza that runs an
     executable of this file is judged through that executable's own verdict, and one that runs
     something this scan cannot name is not judged at all, since the modules behind it are not
     visible from here. What is left -- a stanza that declares the variable and runs nothing
     whatever -- has nothing behind its declaration by construction. *)
  let runs_something stanza = not (List.is_empty (executables_run stanza)) in
  (* And a stanza that names the variable somewhere OTHER than its dependency field is acting on it
     -- `(setenv OCANNL_BUILD_FILES_PREFIX "" …)` is how `generated_provenance`'s rule pins the
     default -- so its declaration answers for something this scan can see, whatever it runs. *)
  let acts_on_the_variable stanza =
    match stanza with
    | Sexp.List (_ :: fields) ->
        List.exists fields ~f:(function
          | Sexp.List (Sexp.Atom "deps" :: _) -> false
          | field -> List.mem (atoms field) artifact_env_var ~equal:String.equal)
    | Sexp.List [] | Sexp.Atom _ -> false
  in
  let module_subjects =
    List.concat_map stanzas ~f:(fun stanza ->
        match head stanza with
        | Some h when List.mem module_bearing_heads h ~equal:String.equal -> (
            let modules = modules_of ~directory_modules stanzas stanza in
            let callers = List.filter modules ~f:calls in
            let readers = List.filter modules ~f:(fun m -> (not (calls m)) && reads_prefix m) in
            let name = match names_of stanza with n :: _ -> n | [] -> "<unnamed>" in
            let subject ?(as_name = name) ?(callers = callers) ?(readers = readers) artifact_verdict
                artifact_deps_site =
              Some
                {
                  artifact_head = h;
                  artifact_name = as_name;
                  artifact_callers = callers;
                  artifact_readers = readers;
                  artifact_deps_site;
                  artifact_verdict;
                }
            in
            (* [all] is what makes the stanza declared -- every run of it has to be invalidated, so a
             second rule running the same executable without the declaration leaves that run stale.
             [any] is what makes a declaration present at all, and so what a stale one is judged
             by. The two coincide for everything dune runs itself. *)
            (* What makes the stanza subject to the rule is that some module of it READS the key --
             through the initializer or by name. Asking only about the initializer permitted a
             declaration for a direct reader without ever requiring one, which leaves exactly the
             stale run this check is about (Codex P2, round 3). Which of the two it is decides only
             the wording of the verdict. *)
            let decide ?as_name ?(callers = callers) ?(readers = readers) ~all ~any site =
              let subject = subject ?as_name ~callers ~readers in
              match (callers, readers, all, any) with
              | [], [], _, false -> None
              | [], [], _, true -> subject Artifact_stale_declaration site
              | [], _ :: _, true, _ -> subject Artifact_other_reader site
              | _ :: _, _, true, _ -> subject Artifact_declared site
              | _, _, false, _ -> subject Artifact_undeclared site
            in
            let declares args = declares_env_var args artifact_env_var in
            match h with
            | "library" -> (
                Option.to_list
                @@
                if
                  (* A library module CALLING the initializer is prohibited whether or not the
                     library also has inline tests: `init` empties the artifact directory of
                     whatever process links the module, and an `(inline_tests (deps …))` declaration
                     invalidates the inline-test runner alone -- not the other executables that link
                     the same library and initialize through it (Codex P2, round 4). Reading the key
                     by NAME is an ordinary thing for a library module to do, so a reader is judged
                     where the library's own tests run, and not judged at all where it has none. *)
                  not (List.is_empty callers)
                then subject Artifact_in_library "-"
                else
                  match field stanza "inline_tests" with
                  | None -> None
                  | Some inline ->
                      let declared = declares (field_in inline "deps") in
                      decide ~all:declared ~any:declared "its `(inline_tests (deps …))`")
            | "executable" | "executables" ->
                (* One subject per NAME, since one name is one program: its own main module plus
                   every module no name claims, judged against the rules that run IT
                   (gh-ocannl-747). For the single-name shape -- everything in this repository --
                   that is the same partition as before, the whole module set against every
                   runner. *)
                let programs =
                  List.map (program_runners ~subdir ~runner_stanzas stanzas stanza)
                    ~f:(fun (name, name_runners) ->
                      let own = program_modules stanza ~modules ~name in
                      let callers = List.filter own ~f:calls in
                      let readers =
                        List.filter own ~f:(fun m -> (not (calls m)) && reads_prefix m)
                      in
                      (name, name_runners, callers, readers))
                in
                (* A declaration belongs to the RULE, so whether it is stale is the rule's question
                   and not the program's: a rule running both `a` and `b` where only `a` reads the
                   key is justified, and neither program is carrying a stale declaration. Judged per
                   runner against every program of the stanza it launches, rather than by asking
                   each program whether ALL of its runners are justified -- which reported `b` stale
                   as soon as it also had a dedicated undeclared rule of its own (Codex P2, rounds 2
                   and 4 of PR #484). *)
                let needed_by_some =
                  List.concat_map programs ~f:(fun (_, runners, callers, readers) ->
                      if List.is_empty callers && List.is_empty readers then []
                      else List.map runners ~f:fst)
                in
                let unjustified (runner, _) =
                  declares (field runner "deps")
                  && not (List.mem needed_by_some runner ~equal:Sexp.equal)
                in
                List.concat_map programs ~f:(fun (name, name_runners, callers, readers) ->
                    let decide = decide ~as_name:name ~callers ~readers in
                    Option.to_list
                      (match name_runners with
                      | [] ->
                          if List.is_empty callers && List.is_empty readers then None
                          else subject ~as_name:name ~callers ~readers Artifact_unrun "-"
                      | runners ->
                          let declared =
                            List.map runners ~f:(fun (r, _) -> declares (field r "deps"))
                          in
                          decide ~all:(List.for_all declared ~f:Fn.id)
                            ~any:(List.exists runners ~f:unjustified)
                            (Printf.sprintf "the `(deps …)` of the %d rule%s running %s.exe"
                               (List.length runners)
                               (if List.length runners = 1 then "" else "s")
                               name)))
            | _ ->
                let declared = declares (field stanza "deps") in
                Option.to_list (decide ~all:declared ~any:declared "its `(deps …)`"))
        | _ -> [])
  in
  (* The converse over the stanzas the question above does not reach. A `(rule …)` names no modules,
     so it is a subject only through the executable it runs -- and a rule that declares the variable
     and runs nothing at all was outside the check entirely, which is the copied declaration the
     converse direction exists to catch (Codex P2, round 1). Heads that DO name modules are excluded
     here, having been decided above, so nothing is reported twice. *)
  let stale_subjects =
    List.filter_map stanzas ~f:(fun stanza ->
        match head stanza with
        | Some h
          when (not (List.mem module_bearing_heads h ~equal:String.equal))
               && declares_env_var (field stanza "deps") artifact_env_var
               && (not (runs_something stanza))
               && not (acts_on_the_variable stanza) ->
            let name =
              match (names_of stanza, exes_run stanza) with
              | name :: _, _ -> name
              | [], [] -> "<unnamed>"
              | [], exes ->
                  "running "
                  ^ String.concat ~sep:", "
                      (List.map exes ~f:(function `File path -> path | `Public name -> name))
            in
            Some
              {
                artifact_head = h;
                artifact_name = name;
                artifact_callers = [];
                artifact_readers = [];
                artifact_deps_site = "its `(deps …)`";
                artifact_verdict = Artifact_stale_declaration;
              }
        | _ -> None)
  in
  module_subjects @ stale_subjects

(** The verdict in one word, for a check's tables and a cases test's expectations. *)
let artifact_verdict_name = function
  | Artifact_declared -> "declared"
  | Artifact_undeclared -> "undeclared"
  | Artifact_stale_declaration -> "stale declaration"
  | Artifact_other_reader -> "declared for a direct read"
  | Artifact_unrun -> "unrun"
  | Artifact_in_library -> "in a library"
