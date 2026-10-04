(** Scanning OCaml sources for the schedule-cache directories they name.

    An autotune run writes its saved schedules into a directory named relative to the working
    directory, so a test that names one leaves that directory wherever the executable ran. Under
    [dune runtest] that is inside [_build/], which is why a missing root [.gitignore] entry used to
    stay missing: the directory materialises in the repository root only when someone runs the test
    executable there by hand, and then it is untracked on that one machine.

    The ignore list answers that with a single glob over a name prefix rather than an entry per
    directory, so it cannot fall behind. What has to hold instead is that every directory a source
    names carries the prefix, which is what this module reads sources to establish.

    {1 Reading these sources as OCaml}

    Parsed, not grepped, for the reasons {!Config_key_scan} sets out at length — and this scan needs
    the parse tree more than that one does, on two counts. The name is usually not written at the
    call site: of the two dozen arguments that name a directory, most arrive through a binding
    ([let cache_dir = "…" in … Autotune.tune ~cache_dir]) rather than as a literal, so a scan
    matching [~cache_dir:"…"] would see a minority of them and vouch for nothing about the rest. And
    the second spelling, a direct [Schedule_cache] operation's [~dir], can only be told from any
    other [~dir] in the repository by resolving the module alias it is called through — which the
    tests bind three different ways.

    {1 What is resolved, and how far}

    Names resolve through [Lexical_scope] at each use. A labelled or optional [cache_dir] parameter
    is reported as forwarded; every other unresolved argument fails loudly in the consuming check.
    Only exact [Schedule_cache], [Ir.Schedule_cache] and [Ocannl.Ir.Schedule_cache] module paths
    (and their lexical aliases) identify direct cache operations. Unknown opens bring in no names;
    qualified directory values are unresolved rather than borrowed from a same-named local. The
    library's [Autotune.resolve_cache_dir] preserves forwarding when its argument is a lexical
    parameter: it picks that parameter or the separately censused configuration default. Other
    computations of a directory remain unresolved. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes

(** The two labels whose argument names a schedule cache directory: [~cache_dir], which
    [Autotune.tune] and [Train.tune_placements] take, and the [~dir] of a direct
    {!Ir.Schedule_cache} operation. Both create the directory — [Schedule_cache.store] runs
    [ensure_dir] over the path — so a census of one is not a census of caches. A test that only
    seeds an entry, never tuning against it, is an established pattern here and would otherwise
    leave an unignored directory with this check green (Codex P2, round 1). *)
let tune_label = "cache_dir"

let cache_module = "Schedule_cache"
let store_label = "dir"

(** The configuration key whose default value is the directory a search uses when no [~cache_dir] is
    passed. Nothing in a source names that directory, so it has to be read out of the library that
    defines it: the prefix rule holding over every explicit argument says nothing about the one
    every implicit search uses (Codex P2, round 3). *)
let default_config_key = "autotune_cache_dir"

(** The prefix every such directory carries, so that one [.gitignore] glob covers all of them. It is
    also, today, the built-in default's own name — but that is a fact to be checked rather than
    relied on, which is what {!default_config_key} is read for. *)
let required_prefix = "autotune_cache"

let string_literal = Config_key_scan.string_literal
let structure_of = Config_key_scan.structure_of
let longident_of = Config_key_scan.longident_of
let pattern_name = Config_key_scan.pattern_name
let flatten_longident = Config_key_scan.flatten_longident

let label_name = function
  | Asttypes.Labelled name | Asttypes.Optional name -> Some name
  | Asttypes.Nolabel -> None

(* The string literal an application passes under a given label, where it passes one. *)
let labelled_literal args wanted =
  List.find_map args ~f:(fun (lbl, argument) ->
      match label_name lbl with
      | Some name when String.equal name wanted -> string_literal argument
      | _ -> None)

(** Whether a directory name is one the ignore glob covers: a single path component carrying the
    prefix. The component matters as much as the prefix — [Schedule_cache.ensure_dir] walks the path
    it is given, so ["autotune_cache/../leaked_cache"] carries the prefix and creates [leaked_cache]
    in the working directory, which [/autotune_cache*/] does not match (Codex P2, round 1). A glob
    segment does not cross a separator, which is exactly the property required here. *)
let covered_by_glob name =
  String.is_prefix name ~prefix:required_prefix
  && not (String.exists name ~f:(fun c -> Char.equal c '/' || Char.equal c '\\'))

(** How a [~cache_dir] argument names its directory. [Names] carries a directory the prefix rule
    applies to; [Disabled] is the empty string, which turns the disk cache off and creates nothing;
    [Forwarded] is an identifier some function in this file takes as a parameter, whose value
    arrives from a call site scanned in its own right; [Unresolved] is everything else, which the
    check reports rather than assumes. *)
type resolution = Names of string | Disabled | Forwarded of string | Unresolved of string

type use = { resolution : resolution; line : int; spelling : string }
(** [spelling] is the label as written ([~cache_dir] or [~dir]), so a failure names the argument the
    author has to change rather than a canonical one they never typed. *)

let describe = function
  | Names name -> "names " ^ name
  | Disabled -> "disables the cache"
  | Forwarded name -> "forwards the parameter " ^ name
  | Unresolved how -> "names " ^ how

type value_denotes = Parameter | Unknown | Resolver | Literal of string
type module_denotes = Cache | Ir | Ocannl | Autotune | Other

let module_path env path =
  let rec resolve = function
    | Ppxlib.Longident.Lident name -> (
        match Map.find env.Lexical_scope.modules name with
        | Some denotation -> denotation
        | None -> (
            match name with
            | "Schedule_cache" -> Cache
            | "Ir" -> Ir
            | "Ocannl" -> Ocannl
            | "Autotune" -> Autotune
            | _ -> Other))
    | Ldot (path, "Ir") when Poly.equal (resolve path) Ocannl -> Ir
    | Ldot (path, "Schedule_cache") when Poly.equal (resolve path) Ir -> Cache
    | _ -> Other
  in
  Some (resolve path)

type report = {
  uses : use list;
      (** every argument that names a directory: the [~cache_dir] of a tuning call, and the [~dir]
          of a direct {!cache_module} operation *)
  builtin_defaults : string list;
      (** the non-empty [~default:] literals of this file's reads of {!default_config_key} *)
}

(** What one source says about schedule cache directories. Parses once: both questions are asked of
    every file in the repository, and each is a walk over the same tree. *)
let read ?(source = "") content =
  let ast = structure_of content in
  let resolve env ~disabling_allowed argument =
    let literal value =
      if String.is_empty value && disabling_allowed then Disabled else Names value
    in
    match string_literal argument with
    | Some value -> literal value
    | None -> (
        match argument.pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some (Literal value) -> literal value
            | Some Parameter -> Forwarded name
            | _ -> Unresolved ("`" ^ name ^ "`"))
        | Pexp_ident { txt; _ } ->
            Unresolved ("`" ^ String.concat ~sep:"." (flatten_longident txt) ^ "`")
        | _ -> Unresolved "an expression")
  in
  let calls_cache_module env callee =
    match callee.pexp_desc with
    | Pexp_ident { txt = Ppxlib.Longident.Ldot (qualifier, _); _ } ->
        Poly.equal (module_path env qualifier) (Some Cache)
    | _ -> false
  in
  let found = ref [] and defaults = ref [] in
  let iterator =
    object (self)
      inherit [value_denotes, module_denotes] Lexical_scope.scoped as super
      method local = Unknown

      method! bind_parameter env label pattern =
        let denotes =
          if Option.equal String.equal (label_name label) (Some tune_label) then Parameter
          else Unknown
        in
        self#forget
          (Lexical_scope.bind_values env (Lexical_scope.pattern_vars [ pattern ]) denotes)
          (Lexical_scope.pattern_unpacks [ pattern ])

      method! shadowed = Some Other
      method module_path env path = module_path env path
      val mutable resolver_seen = false

      method! define ~top env rec_flag bindings ~walk =
        let denotes = super#define ~top env rec_flag bindings ~walk in
        List.map2_exn bindings denotes ~f:(fun binding value ->
            if
              top
              && String.equal source "arrayjit/lib/autotune.ml"
              && Option.equal String.equal (pattern_name binding.pvb_pat) (Some "resolve_cache_dir")
            then (
              let first = not resolver_seen in
              resolver_seen <- true;
              if first then Resolver else Unknown)
            else value)

      method! let_denotes binding =
        match (pattern_name binding.pvb_pat, string_literal binding.pvb_expr) with
        | Some _, Some value -> Literal value
        | _ -> Unknown

      method! binding_denotes env binding =
        match self#let_denotes binding with
        | Literal _ as value -> value
        | _ -> (
            match binding.pvb_expr.pexp_desc with
            | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } ->
                Option.value (Lexical_scope.lookup env name) ~default:Unknown
            | Pexp_apply (callee, args) ->
                let resolver =
                  match callee.pexp_desc with
                  | Pexp_ident { txt = Ppxlib.Longident.Lident "resolve_cache_dir"; _ } ->
                      String.equal source "arrayjit/lib/autotune.ml"
                      && Poly.equal (Lexical_scope.lookup env "resolve_cache_dir") (Some Resolver)
                  | Pexp_ident { txt = Ldot (path, "resolve_cache_dir"); _ } ->
                      Poly.equal (module_path env path) (Some Autotune)
                  | _ -> false
                in
                let forwarding =
                  List.exists args ~f:(fun (label, arg) ->
                      Option.equal String.equal (label_name label) (Some tune_label)
                      &&
                      match arg.pexp_desc with
                      | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } ->
                          Poly.equal (Lexical_scope.lookup env name) (Some Parameter)
                      | _ -> false)
                in
                if resolver && forwarding then Parameter else Unknown
            | _ -> Unknown)

      method! attribute _ attr = attr

      method! expression env expr =
        (match expr.pexp_desc with
        | Pexp_apply (callee, args) ->
            (* The built-in default: the [~default:] literal of a read of the key by name. The
               library reads it twice and only one carries the directory -- the other asks merely
               whether the key was set, and defaults to the empty string. *)
            if
              Option.equal String.equal (labelled_literal args "arg_name") (Some default_config_key)
            then
              Option.iter (labelled_literal args "default") ~f:(fun value ->
                  if not (String.is_empty value) then defaults := value :: !defaults);
            let into_cache = calls_cache_module env callee in
            List.iter args ~f:(fun (lbl, argument) ->
                (* [Some disabling_allowed] where this argument names a directory. *)
                let names_a_directory =
                  match label_name lbl with
                  | Some name when String.equal name tune_label -> Some true
                  | Some name when String.equal name store_label && into_cache -> Some false
                  | _ -> None
                in
                Option.iter names_a_directory ~f:(fun disabling_allowed ->
                    found :=
                      {
                        resolution = resolve env ~disabling_allowed argument;
                        line = argument.pexp_loc.loc_start.pos_lnum;
                        spelling = "~" ^ Option.value (label_name lbl) ~default:tune_label;
                      }
                      :: !found))
        | _ -> ());
        super#expression env expr
    end
  in
  ignore
    (iterator#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  { uses = List.rev !found; builtin_defaults = List.rev !defaults }

type ignore_line = { pattern : string; negated : bool }

(** A line with git's whitespace rules applied: trailing spaces dropped unless backslash-quoted,
    LEADING whitespace kept. The asymmetry is git's, and it is not decorative — an accidentally
    indented [ /autotune_cache*/] has the space as part of the pattern and ignores nothing, so a
    parser that stripped both ends would report coverage git does not give (Codex P2, round 4;
    checked against `git check-ignore`, which agrees on both halves). *)
let strip_trailing_spaces line =
  let n = String.length line in
  let rec last_kept i =
    if i <= 0 then 0
    else if not (Char.equal line.[i - 1] ' ') then i
    else if i >= 2 && Char.equal line.[i - 2] '\\' then i
    else last_kept (i - 1)
  in
  String.prefix line (last_kept n)

(** The patterns of an ignore file, in order: comments and blank lines dropped, a leading [!]
    recorded rather than swallowed. Order is kept because gitignore's rule is last-match-wins, so a
    set of patterns is not enough to answer whether anything is ignored. A [#] opens a comment only
    at the start of the line, which is why the comment test comes before no stripping at all. *)
let ignore_patterns content =
  String.split_lines content
  |> List.filter_map ~f:(fun line ->
      if String.is_prefix line ~prefix:"#" then None
      else
        let line = strip_trailing_spaces line in
        if String.is_empty line then None
        else
          match String.chop_prefix line ~prefix:"!" with
          | Some rest -> Some { pattern = rest; negated = true }
          | None -> Some { pattern = line; negated = false })

(** The glob that has to be in the root [.gitignore] for the prefix rule to ignore anything.
    Root-anchored and directory-only: a cache directory is only ever created in the working
    directory, and an unanchored pattern would hide a stray copy anywhere in the tree — which is how
    three of them once reached [test/config/] and were committed. *)
let required_glob = "/" ^ required_prefix ^ "*/"

(** Whether an ignore file carries [required_glob] as an ignore rather than a negation. Read line by
    line rather than as a glob engine: what is asked is whether this exact rule is present, which is
    what keeps the ignore list from creeping back into a name-by-name list even while
    {!effectively_ignored} would be satisfied by bespoke entries. *)
let declares_required_glob content =
  String.split_lines content
  |> List.exists ~f:(fun line -> String.equal (strip_trailing_spaces line) required_glob)

(* A glob over one path component: [*] and [?], a backslash making the next character literal, and
   no separators to consider. Bounded by the pattern and name lengths, both tiny here.

   The escape is git's and it is load-bearing in both directions: [\_] is an underscore, so
   [!/autotune_cache\_test/] really does expose [autotune_cache_test] while a matcher reading the
   backslash literally sees no match and reports the directory still ignored; and [\*] is a literal
   asterisk rather than a wildcard, so ignoring the escape over-matches as readily as it
   under-matches. Both checked against `git check-ignore` (Codex P2, round 5). *)
(* A character class, git's own reading of one: members and [a-z] ranges, negated by a leading [!]
   or [^], a [\\] escaping the next character, and a []] in the first position taken literally.
   [None] for an unterminated [[], which git does NOT read as a literal bracket -- the pattern then
   matches nothing at all, `[abc` included. Every one of those readings is measured against
   `git check-ignore` rather than assumed, and the unterminated one is where the assumption would
   have been wrong; the cases are pinned in [cache_dir_scan_cases]. *)
let parse_class pattern at =
  let n = String.length pattern in
  let negated, start =
    let start = at + 1 in
    if start < n && (Char.equal pattern.[start] '!' || Char.equal pattern.[start] '^') then
      (true, start + 1)
    else (false, start)
  in
  let rec scan k acc ~first =
    if k >= n then None
    else if Char.equal pattern.[k] ']' && not first then Some (negated, List.rev acc, k + 1)
    else if Char.equal pattern.[k] '\\' && k + 1 < n then
      scan (k + 2) (`Char pattern.[k + 1] :: acc) ~first:false
    else if k + 2 < n && Char.equal pattern.[k + 1] '-' && not (Char.equal pattern.[k + 2] ']') then
      scan (k + 3) (`Range (pattern.[k], pattern.[k + 2]) :: acc) ~first:false
    else scan (k + 1) (`Char pattern.[k] :: acc) ~first:false
  in
  scan start [] ~first:true

let class_matches members c =
  List.exists members ~f:(function
    | `Char member -> Char.equal member c
    | `Range (low, high) -> Char.between c ~low ~high)

let glob_matches pattern name =
  let np = String.length pattern and nn = String.length name in
  let rec go i j =
    if i = np then j = nn
    else
      match pattern.[i] with
      | '\\' when i + 1 < np -> j < nn && Char.equal name.[j] pattern.[i + 1] && go (i + 2) (j + 1)
      | '*' -> go (i + 1) j || (j < nn && go i (j + 1))
      | '?' -> j < nn && go (i + 1) (j + 1)
      | '[' -> (
          match parse_class pattern i with
          | Some (negated, members, next) ->
              j < nn && Bool.equal negated (not (class_matches members name.[j])) && go next (j + 1)
          | None -> false)
      | c -> j < nn && Char.equal name.[j] c && go (i + 1) (j + 1)
  in
  go 0 0

(* Whether a pattern contains a separator that git reads as one -- an escaped [\/] is a literal
   slash in a name, not the anchoring separator, so the two cannot be counted together. *)
let has_unescaped_slash pattern =
  let n = String.length pattern in
  let rec go i =
    if i >= n then false
    else if Char.equal pattern.[i] '\\' then go (i + 2)
    else if Char.equal pattern.[i] '/' then true
    else go (i + 1)
  in
  go 0

(** The glob a pattern imposes on a root-level DIRECTORY name, where it can match one at all.
    gitignore anchors a pattern that contains a slash anywhere but the end to the ignore file's own
    directory, so [docs/*.log] cannot match a bare root-level name; a pattern without one matches by
    basename at any depth, the root included. A trailing slash restricts a pattern to directories,
    which every candidate here is. *)
let rec drop_any_depth_prefix p =
  match String.chop_prefix p ~prefix:"**/" with
  | Some rest -> drop_any_depth_prefix rest
  | None -> p

let root_directory_glob pattern =
  let p = Option.value (String.chop_suffix pattern ~suffix:"/") ~default:pattern in
  let p = Option.value (String.chop_prefix p ~prefix:"/") ~default:p in
  (* A leading [**/] matches any number of directories INCLUDING ZERO, so [**/foo] and [/**/foo]
     both reach a root-level [foo] -- rejecting them for containing a slash read them as applying to
     nothing, and the unreadable-pattern report, sharing this conversion, did not catch them either
     (Codex P2, round 6; `git check-ignore` confirms all three spellings expose the directory).
     After the prefix is gone, a [**] left inside one component is git's "consecutive asterisks are
     regular asterisks", which {!glob_matches} already treats as [*] does. *)
  let p = drop_any_depth_prefix p in
  if has_unescaped_slash p || String.is_empty p then None else Some p

(** Patterns that could bear on a root-level directory name and that {!glob_matches} cannot read.
    Reported by the caller rather than silently treated as non-matching: a scan that cannot read its
    input has to say so, and "not ignored" and "not understood" are different answers. *)
let unreadable_patterns content =
  List.filter_map content ~f:(fun { pattern; negated = _ } ->
      match root_directory_glob pattern with
      | Some glob when String.is_substring glob ~substring:"**" -> Some pattern
      | _ -> None)

(** Whether git ignores a root-level directory of this name, by gitignore's own rule: every pattern
    is considered in order and the LAST one that matches decides, so a later [!] un-ignores what an
    earlier line ignored. Reading only for the required glob's presence would report coverage that a
    subsequent negation has taken away (Codex P2, round 2). *)
let effectively_ignored patterns name =
  List.fold patterns ~init:false ~f:(fun ignored { pattern; negated } ->
      match root_directory_glob pattern with
      | Some glob when glob_matches glob name -> not negated
      | _ -> ignored)
