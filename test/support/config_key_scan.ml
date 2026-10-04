(** Scanning OCaml sources for the configuration keys they read.

    Shared by the two consistency tests that hold the configuration honest:
    [test_config_consistency] (every key documented and registered) and [digest_completeness] (every
    key classified against the cache identity, gh-ocannl-572). Both need the same fact — which keys
    a source file reads — so they read it the same way.

    {1 Reading these sources as OCaml}

    This module parses. It does not match text, and it does not match token shapes.

    That is the conclusion of five review rounds on PR #340, each finding the same kind of defect
    one level down: prose read as a call site, a quoted string inside a comment read as a nested
    comment, a character literal opening a string, an escaped literal losing its value, and finally
    [let () = …] not counting as a definition while [?(arg_name : string = "k")] did not count as a
    literal. Every one was silent by construction — keys that vanish from a scan look exactly like
    keys that were never read — and every fix left the next unhandled spelling waiting.

    An approximation of OCaml has no natural stopping point; the grammar does. On the parse tree a
    labelled argument is one node however it was spelled, a string literal carries its decoded value
    whatever escapes produced it, and a structure item covers exactly the source it covers. *)

open Base
open Ppxlib.Parsetree

(* ppxlib's parse tree rather than [compiler-libs]'. Both read a source with the same parser --
   ppxlib calls the compiler's -- but ppxlib then migrates the result to an AST of its own version,
   which does not move when the compiler's does.

   That is what a scanner needs, and what the compiler's own tree could not give it: these modules
   are the one part of this repository a compiler release can break, and the 5.4/5.5 boundary broke
   them twice -- [Ldot] gaining located components, then [let module M = … in] ceasing to be
   [Pexp_letmodule] and becoming an ordinary structure item inside an expression. Neither was a
   change in what the sources say; both were changes in how the tree spells it.

   The selected AST is [Astlib.Ast_502], BELOW the floor the opam files declare, so a parse on any
   supported compiler is a downgrade and the question is what a downgrade costs a scanner.

   Not data: the 5.x chain performs no migration_error at all. What a newer AST spells differently
   is either mapped onto the older constructor -- 5.5's structure-item-in-an-expression becomes the
   [Pexp_letmodule] that {!Cache_dir_scan} matches -- or carried across in an attribute encoding.
   And not recognition either, for what is matched here: applications, labelled arguments, string
   constants, field accesses and module bindings all predate 5.2 and map across directly.

   Checked rather than argued: the censuses built on these scanners are byte-identical to what
   matching the compiler's own tree produced, over every source in the repository.

   What this DOES rest on is the AST staying [Ast_502], which is a property of ppxlib rather than of
   the compiler -- hence the upper bound on ppxlib in [dune-project], whose note explains why the
   bound rather than a pinned versioned AST. *)
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
module Location = Ppxlib.Location
module Longident = Ppxlib.Longident
module Parse = Ppxlib.Parse

(** The label whose argument names a configuration key. *)
let label = "arg_name"

let is_our_label = function
  | Asttypes.Labelled name | Asttypes.Optional name -> String.equal name label
  | Asttypes.Nolabel -> false

(** The value of a string literal as the parser resolved it: [{|k|}], ["k"] and a continuation
    spelled over two lines all arrive here decoded. *)
let string_literal expr =
  match expr.pexp_desc with Pexp_constant (Pconst_string (value, _, _)) -> Some value | _ -> None

(* The library's own function rather than a match on the constructors. Matching [Ldot] would pin a
   constructor shape into this file, which is the coupling the header has just moved off: ppxlib
   guarantees the shape of the tree, not that any given spelling of it stays convenient.

   [Longident.flatten_exn] fatal-errors on a functor application, and it is worth recording why that
   cannot arise here rather than guarding against it (Codex P2 on PR #342, whose premise this is):
   in EXPRESSION position OCaml gives no [Pexp_ident] an applied path. [Set.Make(String).empty] and
   [F(X).Utils.settings.large_models] both parse as [Pexp_field] over a constructor application,
   with a qualified label -- checked against the parser, not assumed. Since {!longident_of} runs on
   [Pexp_ident] alone, and a record-field label is never an application, every path reaching here is
   [Lident] or [Ldot]. A future parser that changed this would take the test down loudly, which
   beats a fallback silently dropping a read. *)
let flatten_longident = Longident.flatten_exn

let longident_of expr =
  match expr.pexp_desc with Pexp_ident { txt; _ } -> Some (flatten_longident txt) | _ -> None

(** Raises if [content] does not parse: a scan that cannot read its input must say so rather than
    report an empty census. *)
let structure_of content = Parse.implementation (Lexing.from_string content)

(** The ppx_minidebug extension whose argument names a module's compile-time tracing gate. *)
let tracing_gate_extension = "global_debug_log_level_from_env_var"

(** The environment variables [content] reads at RUN time by name: the string literal argument of a
    [Sys.getenv] / [Sys.getenv_opt], under any receiver path ([Stdlib.Sys.getenv_opt], [Sys.getenv]
    with [Base] in scope). A dynamic argument is deliberately not reported -- the reader that takes
    the name as a parameter is {!Utils.read_env_var} itself, whose keys are a different question
    (gh-ocannl-628, Codex P2 round 2 on PR #371).

    Walks expressions, not just structure items, because these reads sit anywhere in a function
    body. *)
let env_var_reads_in_source content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr with
        | [%expr [%e? callee] [%e? argument]] -> (
            match (longident_of callee, string_literal argument) with
            | Some path, Some name -> (
                match List.last path with
                | Some ("getenv" | "getenv_opt") -> found := name :: !found
                | _ -> ())
            | _ -> ())
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !found

(** The environment variables ppx_minidebug reads while preprocessing [content]: the argument of
    each [%%global_debug_log_level_from_env_var] at the top of the file. What a library must declare
    in its [preprocessor_deps] for the gate to invalidate anything (gh-ocannl-628).

    Parsed like everything else here rather than grepped: this file's own doc comments name the
    extension, and a text scan would read a mention as a declaration -- exactly the class of defect
    that made this module parse in the first place. *)
let tracing_gates_in_source content =
  structure_of content
  |> List.filter_map ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_extension (({ txt; _ }, payload), _) when String.equal txt tracing_gate_extension -> (
          match payload with
          | PStr [ { pstr_desc = Pstr_eval (expr, _); _ } ] -> string_literal expr
          | _ -> None)
      | _ -> None)

(** {1 Resolving a receiver} *)

(** The simple name a pattern binds, where it has one. Best-effort by design: a pattern this does
    not recognise yields no name, and every client of it treats a nameless binding as the answer
    that costs least. *)
let rec pattern_name pattern =
  match pattern.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint (inner, _) | Ppat_alias (inner, _) -> pattern_name inner
  | _ -> None

(* A path is flattened defensively here, unlike {!longident_of}: this one runs over MODULE paths,
   where a functor application is a shape the language admits, and a scan that raised on one would
   refuse a file for containing an unrelated [module M = F (X)]. *)
let flatten_module_path txt = try Some (flatten_longident txt) with _ -> None

let path_ends_in name txt =
  match flatten_module_path txt with
  | Some path -> ( match List.last path with Some m -> String.equal m name | None -> false)
  | None -> false

(** Every local name under which the module called [wanted] is reachable in [structure], and whether
    its contents were put in scope unqualified by an [open] or an [include].

    Every way a module's contents can be given a local name, in BOTH the structure and the
    expression grammar. OCaml spells each of binding, opening and including twice — [module G = M]
    against [let module G = M in …], [open M] against [let open M in …], [include M] against nothing
    — and a pass that knew only the structure spellings would read
    [let open Test_utils.Generated in init …] as a call to somebody else's [init] (Codex P2, round 1
    of PR #457). Which is the shape a scan must not get wrong quietly.

    The aliases are collected in this pass and the call sites matched against them in a second —
    OCaml lets neither be used before it is bound, so two passes cost nothing and save the walk from
    depending on that.

    Two answers about the [open]s, because the two clients want opposite defaults. [opened] says one
    exists anywhere in the file, which is the over-reading direction and the safe one for the
    initializer census: a caller too many makes dune rerun a stanza it need not have, one too few is
    the stale run. [open_ranges] says WHERE each is in scope — an [open] at structure level reaching
    to the end of the structure that holds it, a [let open … in body] reaching over the body — which
    is what a client wanting the other direction needs (Codex P2, round 3 of PR #484).

    Shared by the two scans that match a call by its receiver — the initializer census below and the
    environment reader above — because a second copy of these rules is exactly the restatement this
    module exists to avoid. *)
let module_bindings_of structure ~wanted =
  let aliases = ref [] and opened = ref false in
  let open_ranges = ref [] in
  (* The end of the structure currently being walked, which is how far a structure-level `open`
     reaches. Kept as a stack, so an `open` inside a `module M = struct … end` scopes to that struct
     and not to the file. *)
  let structure_end = ref 0 in
  let bind_module_expr ?(scope = fun () -> (0, Int.max_value)) alias module_expr =
    let names_it txt =
      path_ends_in wanted txt
      ||
      (* An alias of an alias: `module H = G` where `G` was bound to the module. The binders run in
         source order and OCaml lets neither be used before it is bound, so consulting what is
         already recorded is all a chain needs (Codex P2, round 2). *)
      match flatten_module_path txt with
      | Some path -> (
          match List.last path with
          | Some m -> List.mem !aliases m ~equal:String.equal
          | None -> false)
      | None -> false
    in
    (* A signature constraint wraps the path without changing which module it names, so `module G :
       module type of Test_utils.Generated = Test_utils.Generated` binds `G` as surely as the bare
       form does. Unwrapped recursively rather than one level: nesting them is legal and a scan that
       stopped at one would be the same defect one layer down (Codex P2, round 4). *)
    let rec unwrap module_expr =
      match module_expr.pmod_desc with
      | Pmod_constraint (inner, _) -> unwrap inner
      | _ -> module_expr
    in
    match (unwrap module_expr).pmod_desc with
    | Pmod_ident { txt; _ } when names_it txt -> (
        match alias with
        | Some alias -> aliases := alias :: !aliases
        | None ->
            opened := true;
            open_ranges := scope () :: !open_ranges)
    | _ -> ()
  in
  let binders =
    object
      inherit Ast_traverse.iter as super

      method! structure items =
        let saved = !structure_end in
        (match List.last items with
        | Some last -> structure_end := last.pstr_loc.loc_end.pos_cnum
        | None -> ());
        super#structure items;
        structure_end := saved

      method! structure_item item =
        (match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = alias; _ }; pmb_expr; _ } ->
            bind_module_expr alias pmb_expr
        | Pstr_open { popen_expr; _ } ->
            bind_module_expr
              ~scope:(fun () -> (item.pstr_loc.loc_end.pos_cnum, !structure_end))
              None popen_expr
        (* `include Test_utils.Generated` puts the contents in scope under no name of their own,
           which is the same situation an `open` leaves. *)
        | Pstr_include { pincl_mod; _ } ->
            bind_module_expr
              ~scope:(fun () -> (item.pstr_loc.loc_end.pos_cnum, !structure_end))
              None pincl_mod
        | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_letmodule ({ txt = alias; _ }, module_expr, _) -> bind_module_expr alias module_expr
        | Pexp_open ({ popen_expr; _ }, body) ->
            bind_module_expr
              ~scope:(fun () -> (body.pexp_loc.loc_start.pos_cnum, body.pexp_loc.loc_end.pos_cnum))
              None popen_expr
        | _ -> ());
        super#expression expr
    end
  in
  binders#structure structure;
  (!aliases, !opened, List.rev !open_ranges)

(** Whether an offset falls inside any of [ranges]. *)
let within ranges offset =
  List.exists ranges ~f:(fun (start, stop) -> start <= offset && offset < stop)

(** Where a binding of the simple name [fn] is in scope in [structure]: a [let fn = …] at structure
    level reaching to the end of the structure that holds it, a [let fn = … in body] reaching over
    the body, and a parameter called [fn] reaching over its function.

    What it is for is shadowing. A file that both opens a module and defines its own [fn] spells the
    bare name for its own, and a scan that read the [open] alone would attribute the call to the
    module — which for a check that asks for an [(env_var …)] declaration means failing a correct
    stanza over a variable the program never reads (Codex P2, round 3 of PR #484).

    Best-effort in the direction that costs least: a pattern {!pattern_name} does not recognise
    yields no range, and a call inside BOTH an open's scope and a shadow's is read as shadowed
    whichever is inner. Both leave a bare call unattributed rather than wrongly attributed, which is
    the safe direction for the client that asks. *)
let shadow_ranges_of structure ~fn =
  let ranges = ref [] in
  let structure_end = ref 0 in
  let binds bindings =
    List.exists bindings ~f:(fun b ->
        Option.value_map (pattern_name b.pvb_pat) ~default:false ~f:(String.equal fn))
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! structure items =
        let saved = !structure_end in
        (match List.last items with
        | Some last -> structure_end := last.pstr_loc.loc_end.pos_cnum
        | None -> ());
        super#structure items;
        structure_end := saved

      method! structure_item item =
        (match item.pstr_desc with
        | Pstr_value (_, bindings) when binds bindings ->
            ranges := (item.pstr_loc.loc_end.pos_cnum, !structure_end) :: !ranges
        | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_let (_, bindings, body) when binds bindings ->
            ranges := (body.pexp_loc.loc_start.pos_cnum, body.pexp_loc.loc_end.pos_cnum) :: !ranges
        | Pexp_function (params, _, _)
          when List.exists params ~f:(fun p ->
                   match p.pparam_desc with
                   | Pparam_val (_, _, pattern) ->
                       Option.value_map (pattern_name pattern) ~default:false ~f:(String.equal fn)
                   | _ -> false) ->
            ranges := (expr.pexp_loc.loc_start.pos_cnum, expr.pexp_loc.loc_end.pos_cnum) :: !ranges
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure structure;
  List.rev !ranges

(** Whether [path] names [fn] of the module [wanted], given that module's local bindings: written
    out or reached through an alias, or spelled bare under an [open]/[include] of it. *)
let names_function_of ~wanted ~aliases ~opened ~fn path =
  match List.rev path with
  | last :: receivers when String.equal last fn -> (
      match receivers with
      | [] -> opened
      | receiver :: _ ->
          String.equal receiver wanted || List.mem aliases receiver ~equal:String.equal)
  | _ -> false

(** {1 Configuration read straight from the environment (gh-ocannl-749)} *)

(** The one function that reads a configuration key's environment variable and nothing else.
    [Utils.get_global_arg] consults the commandline and the config file too, so a value it returns
    can be pinned where the environment cannot reach it; this one cannot be outranked, which is what
    makes a read through it an unconditional dependency on [OCANNL_<KEY>]. *)
let env_reader = "read_env_var"

(** The module that owns it. *)
let utils_module = "Utils"

type env_reader_reads = {
  reader_keys : string list;
      (** every key the reader is reached with, sorted and deduplicated: a string literal at the
          call, and the elements of a list an iteration hands it one at a time *)
  reader_unresolved : string list;
      (** one line per reach whose key this scan could not resolve to a finite set of literals,
          naming what it saw. A caller must treat a non-empty list as a refusal: the alternative is
          a check that runs its loop over whatever keys happened to be resolvable and reports
          success, which is a pass proving nothing (Codex P2, round 4 of PR #484). *)
}

(** Guard-key contract (gh-ocannl-797).

    A dynamic guard uses exactly this local expression:
    [let guarded_keys = [ "key"; ... ] in Base.List.iter guarded_keys ~f:(fun arg_name -> match
     Utils.read_env_var arg_name with ...)].

    The list, iteration, callback parameter and match scrutinee are adjacent AST nodes. No list
    binding, projection, concatenation, callback scope or helper is inferred. Each other dynamic
    reach refuses with the required shape. Match arms may contain arbitrary reporting code, but
    additional reader reaches in them are checked independently. Literal reads remain supported;
    reader identification shares the module's existing receiver census.

    [Base] is a reserved spelling in this contract: a file declaring a module with that name is
    refused conservatively, without resolving its scope. This is a maintenance contract rather than
    an adversarial OCaml interpreter. *)

let rec guard_key_literals expr =
  match expr.pexp_desc with
  | Pexp_construct ({ txt = Lident "[]"; _ }, None) -> Some []
  | Pexp_construct ({ txt = Lident "::"; _ }, Some { pexp_desc = Pexp_tuple [ head; tail ]; _ }) ->
      Option.both (string_literal head) (guard_key_literals tail)
      |> Option.map ~f:(fun (key, keys) -> key :: keys)
  | _ -> None

let env_reader_reads_in_source content =
  let structure = structure_of content in
  let keys = ref [] and unresolved = ref [] in
  let blessed = Hash_set.create (module Int) in
  let contracted = Hash_set.create (module Int) in
  let aliases, _opened, open_ranges = module_bindings_of structure ~wanted:utils_module in
  let shadows = shadow_ranges_of structure ~fn:env_reader in
  let names_the_reader ~offset path =
    match List.rev path with
    | last :: [] when String.equal last env_reader ->
        within open_ranges offset && not (within shadows offset)
    | _ -> names_function_of ~wanted:utils_module ~aliases ~opened:false ~fn:env_reader path
  in
  let base_declared = ref false in
  let declarations =
    object
      inherit Ast_traverse.iter as super

      method! module_binding binding =
        if Option.equal String.equal binding.pmb_name.txt (Some "Base") then base_declared := true;
        super#module_binding binding

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_letmodule ({ txt = Some "Base"; _ }, _, _) -> base_declared := true
        | _ -> ());
        super#expression expr

      method! module_expr expr =
        (match expr.pmod_desc with
        | Pmod_functor (Named ({ txt = Some "Base"; _ }, _), _) -> base_declared := true
        | _ -> ());
        super#module_expr expr
    end
  in
  declarations#structure structure;
  let required_shape =
    "expected `let guarded_keys = [\"key\"; ...] in Base.List.iter guarded_keys ~f:(fun arg_name \
     -> match Utils.read_env_var arg_name with ...)`"
  in
  let refuse_at expr detail =
    let pos = expr.pexp_loc.loc_start in
    unresolved :=
      Printf.sprintf "line %d, column %d: %s; %s%s" pos.pos_lnum (pos.pos_cnum - pos.pos_bol) detail
        required_shape
        (if !base_declared then " (this source declares the reserved module Base)" else "")
      :: !unresolved
  in
  let exact_path expr path =
    Option.value_map (longident_of expr) ~default:false ~f:(List.equal String.equal path)
  in
  let record_contract expr =
    let record_iteration literals iteration =
      (* A following relationship assertion or reporting expression does not change the guard. Only
         the first expression may be the iteration; there is no binding to follow. *)
      let iteration =
        match iteration.pexp_desc with Pexp_sequence (first, _) -> first | _ -> iteration
      in
      match iteration.pexp_desc with
      | Pexp_apply
          ( iterator,
            [
              (Asttypes.Nolabel, source);
              ( Asttypes.Labelled "f",
                {
                  pexp_desc =
                    Pexp_function
                      ( [
                          {
                            pparam_desc =
                              Pparam_val
                                ( Asttypes.Nolabel,
                                  None,
                                  { ppat_desc = Ppat_var { txt = parameter; _ }; _ } );
                            _;
                          };
                        ],
                        None,
                        Pfunction_body
                          {
                            pexp_desc =
                              Pexp_match
                                ( ({
                                     pexp_desc =
                                       Pexp_apply (reader, [ (Asttypes.Nolabel, argument) ]);
                                     _;
                                   } as read),
                                  _ );
                            _;
                          } );
                  _;
                } );
            ] )
        when exact_path iterator [ "Base"; "List"; "iter" ]
             && exact_path source [ "guarded_keys" ]
             && exact_path reader [ "Utils"; "read_env_var" ]
             && exact_path argument [ parameter ] && not !base_declared ->
          Option.iter (guard_key_literals literals) ~f:(fun guarded_keys ->
              Hash_set.add contracted read.pexp_loc.loc_start.pos_cnum;
              keys := guarded_keys @ !keys)
      | _ -> ()
    in
    match expr.pexp_desc with
    | Pexp_let
        ( Asttypes.Nonrecursive,
          [
            {
              pvb_pat = { ppat_desc = Ppat_var { txt = "guarded_keys"; _ }; _ };
              pvb_expr = literals;
              _;
            };
          ],
          iteration ) ->
        record_iteration literals iteration
    | _ -> ()
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        record_contract expr;
        (match expr.pexp_desc with
        | Pexp_apply (f, args) -> (
            match longident_of f with
            | Some path when names_the_reader ~offset:f.pexp_loc.loc_start.pos_cnum path -> (
                Hash_set.add blessed f.pexp_loc.loc_start.pos_cnum;
                if not (Hash_set.mem contracted expr.pexp_loc.loc_start.pos_cnum) then
                  match args with
                  | [ (Asttypes.Nolabel, argument) ] -> (
                      match string_literal argument with
                      | Some key -> keys := key :: !keys
                      | None -> refuse_at expr "dynamic reader outside the guard contract")
                  | _ -> refuse_at expr "reader applied to no single key")
            | _ -> ())
        | Pexp_ident { txt; _ }
          when names_the_reader ~offset:expr.pexp_loc.loc_start.pos_cnum (flatten_longident txt) ->
            if not (Hash_set.mem blessed expr.pexp_loc.loc_start.pos_cnum) then
              refuse_at expr "reader handed on as a value"
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure structure;
  {
    reader_keys = List.dedup_and_sort !keys ~compare:String.compare;
    reader_unresolved = List.dedup_and_sort !unresolved ~compare:String.compare;
  }

(** Whether [content] could possibly reach the reader: the function has to be NAMED for any spelling
    of a call, an alias or an [open] to reach it, so the substring is a necessary condition and a
    file without it is skipped unparsed. The same narrowing filter {!could_call_generated_init} is,
    and only a filter — what decides is {!env_reader_reads_in_source}. *)
let could_read_env_var content = String.is_substring content ~substring:env_reader

(** {1 The artifact-freshness initializer (gh-ocannl-723)} *)

(** The module that owns the freshness-checked reads of [build_files/], and the function of it that
    a test must call before its first compile. *)
let generated_module = "Generated"

let generated_init = "init"

(** Every spelling of a [Test_utils.Generated.init] call in [content], deduplicated and in the order
    they first appear; empty when the source does not call it.

    Three spellings reach the same function and all three are read, because the difference between
    them is a matter of taste and the rule built on this is not: [Test_utils.Generated.init] written
    out, [Generated.init] through a [module Generated = Test_utils.Generated] alias (which is what
    most tests here do), and a bare [init] under an [open] or [include] of the module. Each of those
    has an expression spelling too ([let module G = … in], [let open … in]), and both are collected:
    the difference between them is a matter of taste as well. The alias may be spelled anything, so
    the aliases and opens are collected in a first pass and the call sites matched against them in a
    second — OCaml lets neither be used before it is bound, so two passes cost nothing and save the
    walk from depending on that. An [open] is taken to reach the whole file rather than its own
    scope, which is the over-reading direction and the safe one (see below).

    Parsed rather than grepped, for the reason the rest of this module is:
    [test/support/generated.ml] names its own [Generated.init] in half a dozen doc comments and
    error messages, and [generated_provenance.ml] quotes one of them in a string literal it asserts
    on. A text scan would read every one of those as a call.

    The receiver is matched by NAME, so a module bound to [Generated] that is not this one is read
    as if it were. That is the safe direction of the two, and deliberately so: a declaration too
    many makes dune rerun a stanza that need not have been rerun, while one too few is the stale run
    the rule built on this exists to prevent. *)
let generated_init_calls_in_source content =
  let structure = structure_of content in
  let aliases, opened, _open_ranges = module_bindings_of structure ~wanted:generated_module in
  let is_the_call path =
    names_function_of ~wanted:generated_module ~aliases ~opened ~fn:generated_init path
  in
  let found = ref [] in
  let calls =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match longident_of expr with
        | Some path when is_the_call path ->
            let spelling = String.concat ~sep:"." path in
            if not (List.mem !found spelling ~equal:String.equal) then found := spelling :: !found
        | _ -> ());
        super#expression expr
    end
  in
  calls#structure structure;
  List.rev !found

(** Whether [content] could possibly call the initializer: the module has to be NAMED for any of the
    three spellings to reach it, alias and [open] included, so the substring is a necessary
    condition and skipping a file without it skips no call. Only a narrowing filter — what decides
    is {!generated_init_calls_in_source} — but it keeps a repository-wide census from parsing every
    source in the tree. *)
let could_call_generated_init content = String.is_substring content ~substring:generated_module

(** Which of [paths] the source actually REFERS TO, deduplicated and in the order they first appear.
    Each path is spelled in components — [[ "Ir"; "Alloc_census" ]] — and matches wherever those
    components appear CONSECUTIVELY inside a real identifier's module path:
    [Ir.Alloc_census.snapshot ()], [module AC = Ir.Alloc_census], a type [Ir.Alloc_census.t].

    Parsed rather than grepped, for the reason the rest of this module is parsed: a source that
    names the module in a doc comment, in a string literal, or inside a longer identifier is not
    reading it, and a derivation built on a substring reads all three as uses. The same argument
    {!generated_init_calls_in_source} makes, and it applies with more force here, since what is
    derived from the answer is which focused aggregate a test belongs to (gh-ocannl-783).

    Qualified rather than by bare name, and that is what the path is for: a local or third-party
    module that happens to be called [Alloc_census] is not the instrumentation, and matching the
    last component alone would put its user in a family it has nothing to do with. The qualifier
    carries the provenance the AST does not — the parser resolves no paths — so the caller states
    how much of it to insist on.

    Every longident in the structure is visited, so a reference in an expression, a type, a pattern
    or a module expression counts alike. What does not count is a module ALIASED to one of these
    paths and then used under its alias — but the binding that introduces the alias names the path,
    which this sees, so a file using the alias has already been counted. And a module bound to the
    qualifier itself ([module Ir = Somewhere_else]) is read as if it were the real one: the same
    over-reading direction {!receiver_is_generated} accepts, and the safe one here too. *)
let module_references_in_source content ~paths =
  let structure = structure_of content in
  let rec starts_with components path =
    match (components, path) with
    | _, [] -> true
    | [], _ :: _ -> false
    | c :: components, p :: path -> String.equal c p && starts_with components path
  in
  (* The qualifiers the caller's paths are written under -- [["Ir"]] for [Ir.Alloc_census] -- since
     a source may reach the same module without writing one: [open Ir] then [Alloc_census.snapshot],
     or [module I = Ir] then [I.Alloc_census.t]. Both spell a real reference to the same module, and
     a match that insisted on the literal qualifier would call neither a use (Codex P2, round 6).

     Tracked in ONE traversal, IN SCOPE: a structure's bindings take effect where they sit and end
     with the structure, and [let open … in] / [let module … in] reach their body and nothing else.
     A separate collecting pass would have made [module Elsewhere = struct open Ir end] open the
     qualifier over the whole file, so an unrelated [Alloc_census] outside that module would be read
     as this one (Codex P2, round 7) -- and here, unlike in {!generated_init_calls_in_source}, the
     answer decides which focused aggregate a test belongs to, so an over-reading is a family
     demanding a test that has nothing to do with it. *)
  let qualifiers =
    List.filter_map paths ~f:(fun path ->
        match List.rev path with
        | _ :: (_ :: _ as rev_qualifier) -> Some (List.rev rev_qualifier)
        | _ -> None)
    |> List.dedup_and_sort ~compare:(List.compare String.compare)
  in
  let aliases = ref [] and opened = ref [] and shadowed = ref [] in
  (* A name is the qualifier only while nothing has rebound it. [module Ir = Other] shadows it, and
     a later [Ir.Alloc_census] is Other's -- the same argument the qualified path makes about
     [Foo.Alloc_census], one component further left (Codex P2, round 8). *)
  let is_shadowed name = List.mem !shadowed name ~equal:String.equal in
  (* What a BINDER's target has to be for the name it binds to denote the qualifier: the qualifier
     itself, not a path that happens to contain it. `open Vendor.Ir` opens Vendor's `Ir`, and
     reading it as ours would make every bare `Alloc_census` in the file a reference to the
     instrumentation (Codex P2, round 9). A binding is a WIDE claim -- it decides what a whole scope
     of unqualified names mean -- so it is the place to be exact. *)
  let names_qualifier qualifier components =
    (List.equal String.equal components qualifier && not (List.exists qualifier ~f:is_shadowed))
    ||
    match components with
    | [ single ] ->
        List.exists !aliases ~f:(fun (alias, of_qualifier, _) ->
            String.equal alias single && List.equal String.equal of_qualifier qualifier)
    | _ -> false
  in
  (* A name bound to something that is not the qualifier stops denoting it for the rest of its scope
     -- whether it was bound by a module binding or introduced as a functor's parameter. *)
  let shadow_name name =
    aliases := List.filter !aliases ~f:(fun (a, _, _) -> not (String.equal a name));
    if not (is_shadowed name) then shadowed := name :: !shadowed
  in
  (* Binding by PATH, for the spellings that carry one rather than a module expression: a
     signature's `open Ir` (Codex P2, round 16). *)
  let bind_path alias components =
    let bound = ref false in
    List.iter qualifiers ~f:(fun qualifier ->
        if names_qualifier qualifier components then (
          bound := true;
          match alias with
          | Some alias ->
              shadowed := List.filter !shadowed ~f:(Fn.non (String.equal alias));
              aliases :=
                (alias, qualifier, [])
                :: List.filter !aliases ~f:(fun (a, _, _) -> not (String.equal a alias))
          | None -> opened := (qualifier, []) :: !opened));
    match alias with Some alias when not !bound -> shadow_name alias | _ -> ()
  in
  let bind_module_expr alias module_expr =
    let rec unwrap module_expr =
      match module_expr.pmod_desc with
      | Pmod_constraint (inner, _) -> unwrap inner
      | _ -> module_expr
    in
    (* The paths a module expression re-exports. A path names itself; and an `include` inside a
       structure re-exports what it names, so `module I = struct include Ir end` binds `I` to the
       qualifier as surely as `module I = Ir` does (Codex P2, round 13) -- and goes on doing so when
       the structure defines other things beside it, since an include exports its contents whatever
       sits next to it (round 14).

       `open` is NOT one of these. It changes what unqualified names mean INSIDE the structure and
       exports nothing, so `module I = struct open Ir include Vendor end` re-exports Vendor and
       reading the open as an export would attribute Vendor's `Alloc_census` to us (round 14). *)
    (* And the names it REDEFINES after including: `struct include Ir module Alloc_census = Vendor.
       Alloc_census end` re-exports the qualifier and then overrides one of its leaves, so a
       reference through the wrapper to that leaf is Vendor's (Codex P2, round 15). Carried with the
       binding, since it is a fact about this wrapper and not about the qualifier. *)
    let rec exports module_expr =
      match (unwrap module_expr).pmod_desc with
      | Pmod_ident { txt; _ } -> (Option.to_list (flatten_module_path txt), [])
      | Pmod_structure items ->
          (* IN ORDER: a definition overrides what an earlier include exported, and a LATER include
             overrides the definitions before it (Codex P2, round 16). So an include resets the
             override set to whatever it brings, and a definition adds to it. *)
          List.fold items ~init:([], []) ~f:(fun (paths, overridden) item ->
              match item.pstr_desc with
              | Pstr_include { pincl_mod = inner; _ } ->
                  (* The LAST include decides, and it decides everything: which leaves a later
                     include supersedes cannot be read off the parse tree, so `struct include Ir
                     include Vendor end` is Vendor's and the wrapper stops naming the qualifier
                     (Codex P2, round 17). That is the loud direction -- a reference through such a
                     wrapper is reported as no member rather than credited. *)
                  ignore paths;
                  exports inner
              | Pstr_module { pmb_name = { txt = Some name; _ }; _ } -> (paths, name :: overridden)
              | Pstr_recmodule bindings ->
                  ( paths,
                    List.filter_map bindings ~f:(fun binding -> binding.pmb_name.txt) @ overridden
                  )
              | _ -> (paths, overridden))
      | _ -> ([], [])
    in
    let paths, overridden = exports module_expr in
    let names_any = ref false in
    List.iter paths ~f:(fun components ->
        List.iter qualifiers ~f:(fun qualifier ->
            if names_qualifier qualifier components then (
              names_any := true;
              match alias with
              | Some alias ->
                  shadowed := List.filter !shadowed ~f:(Fn.non (String.equal alias));
                  (* Replacing, not prepending: rebinding a name to another wrapper of the same
                     qualifier would otherwise leave both meanings live, and `matches` takes the
                     kinder of the two (Codex P2, round 16). *)
                  aliases :=
                    (alias, qualifier, overridden)
                    :: List.filter !aliases ~f:(fun (a, _, _) -> not (String.equal a alias))
              | None -> opened := (qualifier, overridden) :: !opened)));
    (* Whatever else it was bound to, the NAME now denotes that instead. *)
    match alias with
    | Some alias when not !names_any -> shadow_name alias
    | _ -> ()
  in
  let found = ref [] in
  let matches components path =
    let qualifier, name =
      match List.rev path with
      | name :: rev_qualifier -> (List.rev rev_qualifier, name)
      | [] -> ([], "")
    in
    (* ANCHORED at the qualifier: `Vendor.Ir.Alloc_census` is Vendor's, and finding our path inside
       a longer one would call its user a probe (Codex P2, round 10). The same exactness the binders
       get, for the same reason -- a path names one module, and which one is decided by where it
       starts. *)
    (starts_with components path && not (List.exists qualifier ~f:is_shadowed))
    (* Anchored at the alias for the same reason the direct branch anchors at the qualifier:
       `Vendor.I.Alloc_census` resolves from `Vendor`, whatever `I` means here (Codex P2, round
       12). *)
    || List.exists !aliases ~f:(fun (alias, of_qualifier, overridden) ->
        List.equal String.equal of_qualifier qualifier
        && (not (List.mem overridden name ~equal:String.equal))
        && starts_with components [ alias; name ])
    (* Under an `open`, the reference begins with the module's own name -- and only while that name
       still means the opened module: `open Ir` followed by `module Alloc_census = Foo.Alloc_census`
       rebinds the leaf, and what follows is Foo's (Codex P2, round 11). *)
    || List.exists !opened ~f:(fun (of_qualifier, overridden) ->
           List.equal String.equal of_qualifier qualifier
           && not (List.mem overridden name ~equal:String.equal))
       && starts_with components [ name ]
       && not (is_shadowed name)
  in
  let walk =
    object (self)
      inherit Ast_traverse.iter as super

      (* A structure is a scope, and its items bind in source order: each is visited under what the
         items before it bound, and the whole set is dropped when the structure ends. Nested
         structures go through this method too, which is what confines a `module M = struct open Ir
         end` to `M`. *)
      method! structure items =
        let saved_aliases = !aliases and saved_opened = !opened and saved_shadowed = !shadowed in
        List.iter items ~f:(fun item ->
            (* A recursive group's names are in scope throughout it, so they bind BEFORE their own
               bodies are walked (Codex P2, round 16). *)
            (match item.pstr_desc with
            | Pstr_recmodule bindings ->
                (* EVERY name first: all of them are in scope in all the bodies, so resolving one
                   member's target before its neighbours are bound would resolve it against the
                   outer meaning of a name the group takes (Codex P2, round 17). *)
                List.iter bindings ~f:(fun binding ->
                    Option.iter binding.pmb_name.txt ~f:shadow_name);
                List.iter bindings ~f:(fun binding ->
                    bind_module_expr binding.pmb_name.txt binding.pmb_expr)
            | _ -> ());
            self#structure_item item;
            match item.pstr_desc with
            | Pstr_module { pmb_name = { txt = alias; _ }; pmb_expr; _ } ->
                bind_module_expr alias pmb_expr
            | Pstr_open { popen_expr; _ } -> bind_module_expr None popen_expr
            | Pstr_include { pincl_mod; _ } -> bind_module_expr None pincl_mod
            | _ -> ());
        aliases := saved_aliases;
        opened := saved_opened;
        shadowed := saved_shadowed

      (* The expression spellings reach their BODY. Visited by hand rather than through `super`, so
         that the binding is in force for the body and gone after it. *)
      method! expression expr =
        let scoped ~bind ~body visit =
          visit ();
          let saved_aliases = !aliases and saved_opened = !opened and saved_shadowed = !shadowed in
          bind ();
          self#expression body;
          aliases := saved_aliases;
          opened := saved_opened;
          shadowed := saved_shadowed
        in
        match expr.pexp_desc with
        | Pexp_open (declaration, body) ->
            scoped
              ~bind:(fun () -> bind_module_expr None declaration.popen_expr)
              ~body
              (fun () -> self#open_declaration declaration)
        | Pexp_letmodule ({ txt = alias; _ }, module_expr, body) ->
            scoped
              ~bind:(fun () -> bind_module_expr alias module_expr)
              ~body
              (fun () -> self#module_expr module_expr)
        | _ -> super#expression expr

      (* A functor's parameter is a module name bound inside its body, and it can be the qualifier's
         name: `module M (Ir : S) = struct … Ir.Alloc_census … end` refers to the parameter (Codex
         P2, round 12). Scoped like the other binders -- in force for the body, gone after it. *)
      method! module_expr module_expr =
        match module_expr.pmod_desc with
        | Pmod_functor (parameter, body) ->
            let saved_aliases = !aliases
            and saved_opened = !opened
            and saved_shadowed = !shadowed in
            (match parameter with
            | Named ({ txt = name; _ }, module_type) ->
                self#module_type module_type;
                Option.iter name ~f:shadow_name
            | Unit -> ());
            self#module_expr body;
            aliases := saved_aliases;
            opened := saved_opened;
            shadowed := saved_shadowed
        | _ -> super#module_expr module_expr

      (* And the module-type spelling of a functor, whose parameter shadows inside the result
         signature exactly as a module functor's does (Codex P2, round 14). *)
      method! module_type module_type =
        match module_type.pmty_desc with
        | Pmty_functor (parameter, result) ->
            let saved_aliases = !aliases
            and saved_opened = !opened
            and saved_shadowed = !shadowed in
            (match parameter with
            | Named ({ txt = name; _ }, argument) ->
                self#module_type argument;
                Option.iter name ~f:shadow_name
            | Unit -> ());
            self#module_type result;
            aliases := saved_aliases;
            opened := saved_opened;
            shadowed := saved_shadowed
        | _ -> super#module_type module_type

      (* A signature is a scope too, and `module Ir : X` inside one binds that name for the items
         after it: the `Ir` in a later `val x : Ir.Alloc_census.t` is the signature's (Codex P2,
         round 15). Same arrangement as `structure`. *)
      method! signature signature =
        let saved_aliases = !aliases and saved_opened = !opened and saved_shadowed = !shadowed in
        List.iter signature ~f:(fun item ->
            (* A recursive group's names are in scope throughout it, so they bind BEFORE their own
               declarations are walked (Codex P2, round 16). *)
            (match item.psig_desc with
            | Psig_recmodule declarations ->
                List.iter declarations ~f:(fun declaration ->
                    Option.iter declaration.pmd_name.txt ~f:shadow_name)
            | _ -> ());
            self#signature_item item;
            match item.psig_desc with
            (* A manifest alias in a signature binds its name to the path it names, exactly as a
               structure's `module I = Ir` does; only a declaration with no manifest is an opaque
               module that merely takes the name (Codex P2, round 17). *)
            | Psig_module
                {
                  pmd_name = { txt = Some name; _ };
                  pmd_type = { pmty_desc = Pmty_alias { txt; _ }; _ };
                  _;
                } ->
                Option.iter (flatten_module_path txt) ~f:(bind_path (Some name))
            | Psig_module { pmd_name = { txt = Some name; _ }; _ }
            | Psig_modsubst { pms_name = { txt = name; _ }; _ } ->
                shadow_name name
            (* A signature's `open` reaches the items after it exactly as a structure's does. *)
            | Psig_open { popen_expr = { txt; _ }; _ } ->
                Option.iter (flatten_module_path txt) ~f:(bind_path None)
            | _ -> ());
        aliases := saved_aliases;
        opened := saved_opened;
        shadowed := saved_shadowed

      method! longident lid =
        (match flatten_module_path lid with
        | Some components ->
            List.iter paths ~f:(fun path ->
                let spelling = String.concat ~sep:"." path in
                if matches components path && not (List.mem !found spelling ~equal:String.equal)
                then found := spelling :: !found)
        | None -> ());
        super#longident lid
    end
  in
  walk#structure structure;
  List.rev !found

type label_use = { key : string option; offset : int }
(** Every place the [arg_name] label names a configuration key, and what it names it with. [key] is
    [Some k] when the argument is a string literal — the convention both consistency tests rely on
    to find a read — and [None] when it is anything else: a variable, a punned parameter, an
    expression. [offset] is where the argument sits, for a caller that wants to report the line.

    Both positions count, because both name keys: an argument at an application site
    ([~arg_name:"key"], [?arg_name:"key"]) and a parameter's default ([?(arg_name = "key")], with or
    without a type annotation). A label in a {e type} is not a use of anything and does not appear.
*)

let label_uses content =
  let uses = ref [] in
  let record key (loc : Location.t) = uses := { key; offset = loc.loc_start.pos_cnum } :: !uses in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_apply (_, args) ->
            List.iter args ~f:(fun (lbl, arg) ->
                if is_our_label lbl then record (string_literal arg) arg.pexp_loc)
        | Pexp_function (params, _, _) ->
            List.iter params ~f:(fun param ->
                match param.pparam_desc with
                | Pparam_val (lbl, default, _) when is_our_label lbl ->
                    record (Option.bind default ~f:string_literal) param.pparam_loc
                | _ -> ())
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !uses

type definition = { start : int; stop : int; name : string option; top_level : bool }
(** Every function in the file: the source range it covers, the name of the binding it is the body
    of (if any), and whether that binding is top-level.

    {2 Why functions and not bindings}

    An exemption says a named top-level function is trusted plumbing. The question is therefore
    which function a use sits in, and the answer is the innermost enclosing {e lambda} — not the
    innermost enclosing binding, which is a different thing that merely correlates with it.

    Three rounds of review found the correlation breaking, each in a new spelling: a local
    [let hidden name = …], then a nameless [let result, source = …] that had to stay transparent
    because the real plumbing forwards its key from inside one, then
    [let (hidden as alias) = fun name -> …], whose pattern yields no simple name. Keying on the
    lambda answers all three at once and does not care how the helper was introduced:

    - the plumbing's own [let result, source = resolve_config_value … ~arg_name:n in …] is no lambda
      at all, so the use stays with the host function, as it must;
    - a local helper is a lambda, so the use is its own, whatever pattern binds it;
    - an inline [(fun name -> … ~arg_name:name)] is a lambda with no binding at all, so it has no
      name and can be exempt nowhere — a hole that was open until this round and that no list of
      binding forms would have closed.

    [name] is best-effort and fails safe: a pattern this does not recognise yields no name, and a
    nameless function is refused rather than exempted. The module path is kept for the reader, as
    [top_level] is what an exemption may rely on. *)

let definitions content =
  let ast = structure_of content in
  let root_bindings =
    List.concat_map ast ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.map bindings ~f:(fun binding -> binding.pvb_loc.loc_start.pos_cnum)
        | _ -> [])
    |> Set.of_list (module Int)
  in
  (* Which functions are the body of a named binding, and whether that binding is top-level. *)
  let named = Hashtbl.create (module Int) in
  let path = ref [] in
  let qualify name = String.concat ~sep:"." (List.rev (name :: !path)) in
  let within name f =
    let saved = !path in
    path := name :: !path;
    f ();
    path := saved
  in
  let naming =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = name; _ }; _ } ->
            within (Option.value name ~default:"_") (fun () -> super#structure_item item)
        (* Anonymous nesting has no name to borrow; the reader gets a placeholder. Correctness does
           not ride on this list being complete -- [top_level] does that. *)
        | Pstr_recmodule _ | Pstr_open _ | Pstr_include _ | Pstr_extension _ ->
            within "_" (fun () -> super#structure_item item)
        | _ -> super#structure_item item

      method! value_binding binding =
        (match (pattern_name binding.pvb_pat, binding.pvb_expr.pexp_desc) with
        | Some name, Pexp_function _ ->
            Hashtbl.set named ~key:binding.pvb_expr.pexp_loc.loc_start.pos_cnum
              ~data:(qualify name, Set.mem root_bindings binding.pvb_loc.loc_start.pos_cnum)
        | _ -> ());
        super#value_binding binding
    end
  in
  naming#structure ast;
  let items = ref [] in
  let collecting =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_function _ ->
            let start = expr.pexp_loc.loc_start.pos_cnum in
            let name, top_level =
              match Hashtbl.find named start with
              | Some (name, top_level) -> (Some name, top_level)
              | None -> (None, false)
            in
            items := { start; stop = expr.pexp_loc.loc_end.pos_cnum; name; top_level } :: !items
        | _ -> ());
        super#expression expr
    end
  in
  collecting#structure ast;
  List.rev !items

(** The function an offset sits in: the smallest one containing it, so a local helper wins over the
    function that defines it. *)
let definition_at definitions offset =
  List.filter definitions ~f:(fun d -> d.start <= offset && offset < d.stop)
  |> List.min_elt ~compare:(fun a b -> Int.compare (a.stop - a.start) (b.stop - b.start))

(** The keys [content] reads through the label.

    A key that reaches the lookup any other way — through a helper taking the name as a parameter —
    is invisible to this scan, and hence to both tests built on it. That is why
    [test_config_consistency] separately fails any non-literal use of the label outside the handful
    of named functions that implement the lookup. *)
let keys_in_source content =
  List.filter_map (label_uses content) ~f:(fun use -> use.key)
  (* The empty string names no setting, so it is not a key and does not enter a census. It does not
     get to pass unnoticed either: [test_config_consistency] reports it from {!label_uses}, where a
     literal is still visible as a literal. *)
  |> List.filter ~f:(fun key -> not (String.is_empty key))

type key_reach = {
  reach : (string * string list) list;
      (** Every top-level value of the module, with the keys it reaches. *)
  unplaced : string list;
      (** Keys read outside every NAMED top-level value binding -- in a nested module, or under
          [let () = …] -- which no entry is credited with, and which a caller has to refuse rather
          than lose. *)
}

(** The configuration keys each top-level value of a module reaches: the literal [~arg_name:"k"]
    reads in its own body, and those of every top-level value of the same module it names,
    transitively (gh-ocannl-1149).

    What it is for is a requirement on a program that NAMES one of these values from outside: the
    keys reached are the ones its run depends on, so the rules running it declare them. A reach is a
    superset where it errs, and errs that way on purpose: a local binding that shadows a top-level
    name is still read as a use of it, and an optional argument a caller supplies -- which spares
    the read it defaults to -- is not modeled. The cost of the excess is a rerun when the variable
    changes; the cost of a miss is a stale result served as a pass.

    In source order, so a top-level name refers to the binding before it, as in OCaml, and a
    [let rec] group is closed to its fixpoint. A nested [module M = struct … end] is walked too, its
    values reported as [M.v] and resolved innermost scope first; a functor's body is not. A
    QUALIFIED value no local one answers to -- [Utils.debug_log_from_routines], [Tnode.Placements.f]
    -- contributes whatever [external_keys] answers for its path, after a top-level [module X = Y]
    alias of this module has been expanded; the caller decides which modules it can follow, since
    this function sees one file. A field of [Utils.settings] contributes what [settings_field_keys]
    answers for it (the field's own name by default). *)
let top_level_key_reach ?(external_keys = fun (_ : string list) -> [])
    ?(settings_field_keys = fun field -> [ field ]) content =
  let structure = structure_of content in
  let aliases =
    List.filter_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_module
            {
              pmb_name = { txt = Some alias; _ };
              pmb_expr = { pmod_desc = Pmod_ident { txt; _ }; _ };
              _;
            } ->
            Option.map (flatten_module_path txt) ~f:(fun path -> (alias, path))
        | _ -> None)
  in
  let expand = function
    | head :: rest -> (
        match List.Assoc.find aliases head ~equal:String.equal with
        | Some path -> path @ rest
        | None -> head :: rest)
    | [] -> []
  in
  let placed = ref (Set.empty (module Int)) in
  (* What a binding's body reads itself, and every value path it names -- resolved later, against
     the scope the binding sits in. *)
  let direct_of expr =
    let keys = ref [] and names = ref [] in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression expr =
          (match expr.pexp_desc with
          | Pexp_ident { txt; _ } ->
              Option.iter (flatten_module_path txt) ~f:(fun path -> names := path :: !names)
          (* The record spelling of a read, recognised as {!settings_keys_in_source} recognises it:
             a field of [Utils.settings], answered for by every key that initializes the field. *)
          | Pexp_field ({ pexp_desc = Pexp_ident { txt; _ }; _ }, { txt = field; _ }) -> (
              match Option.map (flatten_module_path txt) ~f:expand with
              | Some path
                when List.length path >= 2
                     && List.equal String.equal
                          (List.drop path (List.length path - 2))
                          [ "Utils"; "settings" ] ->
                  Option.iter
                    (List.last (flatten_longident field))
                    ~f:(fun field -> keys := settings_field_keys field @ !keys)
              | _ -> ())
          | Pexp_apply (_, args) ->
              List.iter args ~f:(fun (lbl, arg) ->
                  if is_our_label lbl then (
                    placed := Set.add !placed arg.pexp_loc.loc_start.pos_cnum;
                    Option.iter (string_literal arg) ~f:(fun key -> keys := key :: !keys)))
          | _ -> ());
          super#expression expr
      end
    in
    iterator#expression expr;
    (!keys, !names)
  in
  let dotted = String.concat ~sep:"." in
  (* A path named inside [scope] (the nested modules a binding sits in) means the innermost value of
     that spelling defined so far, as OCaml resolves it; a path no local value answers to is another
     module's, whatever [external_keys] says of it. *)
  let resolve ~env ~scope path =
    let rec local depth =
      if depth < 0 then None
      else
        match Map.find env (dotted (List.take scope depth @ path)) with
        | Some keys -> Some keys
        | None -> local (depth - 1)
    in
    match local (List.length scope) with
    | Some keys -> keys
    | None -> (
        match path with
        | [ _ ] -> Set.empty (module String)
        | _ -> Set.of_list (module String) (external_keys (expand path)))
  in
  let reach_of ~env ~scope (keys, names) =
    List.fold names
      ~init:(Set.of_list (module String) keys)
      ~f:(fun acc path -> Set.union acc (resolve ~env ~scope path))
  in
  let rec unwrap module_expr =
    match module_expr.pmod_desc with Pmod_constraint (inner, _) -> unwrap inner | _ -> module_expr
  in
  let rec walk ~scope env items =
    List.fold items ~init:env ~f:(fun env item ->
        match item.pstr_desc with
        | Pstr_value (rec_flag, bindings) -> (
            let group =
              List.filter_map bindings ~f:(fun binding ->
                  Option.map (pattern_name binding.pvb_pat) ~f:(fun name ->
                      (dotted (scope @ [ name ]), direct_of binding.pvb_expr)))
            in
            let rec close env =
              let env' =
                List.fold group ~init:env ~f:(fun acc (name, direct) ->
                    Map.set acc ~key:name ~data:(reach_of ~env:acc ~scope direct))
              in
              if Map.equal Set.equal env env' then env else close env'
            in
            match rec_flag with
            | Recursive ->
                close
                  (List.fold group ~init:env ~f:(fun acc (name, _) ->
                       Map.set acc ~key:name ~data:(Set.empty (module String))))
            | Nonrecursive ->
                List.fold group ~init:env ~f:(fun acc (name, direct) ->
                    Map.set acc ~key:name ~data:(reach_of ~env ~scope direct)))
        (* A nested structure's values are reached as [M.v], from inside it and from after it. A
           functor's body is not walked: what its values read depends on its argument. *)
        | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } -> (
            match (unwrap pmb_expr).pmod_desc with
            | Pmod_structure inner -> walk ~scope:(scope @ [ name ]) env inner
            | _ -> env)
        | _ -> env)
  in
  let env = walk ~scope:[] (Map.empty (module String)) structure in
  let unplaced =
    List.filter_map (label_uses content) ~f:(fun use ->
        if Set.mem !placed use.offset then None else use.key)
  in
  {
    reach = Map.to_alist env |> List.map ~f:(fun (name, keys) -> (name, Set.to_list keys));
    unplaced;
  }

(** Every configuration key that initializes each field of a record called [settings]: the literal
    [~arg_name] reads in the value of each [settings.field <- …] assignment, so that a field a
    compatibility alias can also set ([large_models], from [big_models]) is answered for by both
    keys. *)
let settings_field_initializers content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_setfield ({ pexp_desc = Pexp_ident { txt; _ }; _ }, { txt = field; _ }, value)
          when Option.value_map (flatten_module_path txt) ~default:false ~f:(fun path ->
                   String.equal (List.last_exn path) "settings") -> (
            let keys = ref [] in
            let reads =
              object
                inherit Ast_traverse.iter as super

                method! expression e =
                  (match e.pexp_desc with
                  | Pexp_apply (_, args) ->
                      List.iter args ~f:(fun (lbl, arg) ->
                          if is_our_label lbl then
                            Option.iter (string_literal arg) ~f:(fun key -> keys := key :: !keys))
                  | _ -> ());
                  super#expression e
              end
            in
            reads#expression value;
            match List.last (flatten_longident field) with
            | Some field -> found := (field, !keys) :: !found
            | None -> ())
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !found

(** The library sources among a rule's dependencies, sorted and deduplicated.

    Both consistency tests are handed [%{deps}] — the whole dependency set of the rule that runs
    them, flattened into one argument list — because the files they scan are globbed rather than
    enumerated (gh-ocannl-592: a hand-written list says nothing about the file that falls off it,
    and three files did, each staying green because their keys were read from somewhere else too).

    Two kinds of argument are therefore not sources. Anything that is not a [.ml] file: the config
    file the rule depends on, the reference file, the executable itself. And dune's preprocessed
    twin [x.pp.ml] of an [x.ml] that is in the list — those are the ppx expansion of a file already
    scanned, so they would double every census, and they exist only where the library that owns them
    is built, which is what would make the census differ between a machine with the CUDA toolchain
    and one without. A twin is dropped only when its original is present, so a source genuinely
    named [x.pp.ml] is not silently lost.

    Nothing else is filtered, and both tests print how many files they scanned: a glob that stops
    matching shows up as a diff rather than as a quietly smaller census. *)
let sources_among args =
  let sources =
    List.filter args ~f:(String.is_suffix ~suffix:".ml")
    |> List.dedup_and_sort ~compare:String.compare
  in
  let present = Set.of_list (module String) sources in
  List.filter sources ~f:(fun path ->
      match String.chop_suffix path ~suffix:".pp.ml" with
      | Some stem -> not (Set.mem present (stem ^ ".ml"))
      | None -> true)

(** The scan ROOTS the two consistency scanners glob, each with a lower bound on how many sources it
    must contribute.

    The bounds are what is left of an exact count. A count was there as a tripwire — it proved the
    globs found files rather than silently matching nothing, which is the failure a scanning test
    cannot report on its own (a scan reading zero files reports OK). But an exact count is also a
    tally of the repository, so it moved on every correct addition anywhere under these six
    directories, and two branches adding a file to the SAME directory wrote identical text on that
    line: they merged cleanly to a total that was wrong by one, and the next unrelated PR inherited
    the red (gh-ocannl-701, the same genre as gh-ocannl-665).

    A floor keeps the tripwire and drops the tally: a glob that breaks goes to zero, which any floor
    catches, while a file added moves nothing. They sit well below the counts of the day they were
    written, so that ordinary deletions do not fail either — the number to raise them to is never
    "today's count", which would restore the tally. What they are NOT is a second reader of the
    corpus (the stronger form gh-ocannl-665 needed for stanzas): the corpus here is dune's own
    dependency set, so there is nothing to re-derive, and the independence a hand-written constant
    carries is that no scan of ours can move it.

    Roots rather than directories, because the rules glob with [glob_files_rec]: a source in a
    subdirectory of one of these belongs to that root's census and to that root's floor, not to a
    bucket of its own (Codex P2, round 1). A bucket of its own would be a new line in both goldens —
    the promote round this change exists to remove — under no floor at all. *)
let scan_root_floors =
  [
    ("arrayjit/lib", 30);
    ("benchmarks/runners/ocannl", 5);
    ("bin", 4);
    ("lib", 6);
    ("tensor", 10);
    ("tools", 2);
  ]

(** The scan root [path] sits under. See {!Scan_floors.root_of}. *)
let scan_root_of path = Scan_floors.root_of ~floors:scan_root_floors path

(** [files] counted per scan root, sorted by root. *)
let counts_by_scan_root files = Scan_floors.counts_by_root ~floors:scan_root_floors files

(** Which scan roots the census actually came from, by name and not by count. *)
let scan_roots files = Scan_floors.roots ~floors:scan_root_floors files

(** The floors [files] fails, itemised. *)
let floor_violations files =
  Scan_floors.violations ~floors:scan_root_floors ~noun:"source"
    ~floors_name:"Config_key_scan.scan_root_floors" files

(** The per-root counts and the total, on stderr. *)
let report_counts files =
  Scan_floors.report ~floors:scan_root_floors ~noun:"source" ~what:"Sources scanned" files

(** Basenames that more than one of [files] carries.

    Both tests key something by basename — the forwarding exemptions in [test_config_consistency],
    the codegen-stage module list in [digest_completeness] — which is unambiguous only while the
    files scanned have distinct names. With the scan list enumerated by hand that could not happen
    by accident; with globs over whole directories it can, the day someone adds a [tensor/utils.ml]
    (Codex P2, PR #340 round 10). So it fails where the fix is, rather than silently lending one
    file's exemptions to another. *)
let duplicate_basenames files =
  List.map files ~f:Stdlib.Filename.basename
  |> List.sort_and_group ~compare:String.compare
  |> List.filter_map ~f:(function name :: _ :: _ -> Some name | _ -> None)

(** [keys_in_source] over each file, as a set. Call sites only — this is what
    [test_config_consistency] means by "every key a source file asks for is documented and
    registered". *)
let keys_in_files files =
  List.concat_map files ~f:(fun fname -> keys_in_source (Stdio.In_channel.read_all fname))
  |> Set.of_list (module String)

(** The [Utils] predicates over the settings record that fold a [log_level] threshold into a field
    read, each with the keys a call of it implies.

    One table, because a predicate is two facts in two places and they must agree: the keys its call
    contributes to the census ({!settings_keys_in_source}), and the name that must not be handed
    around as a value where the census could not follow it ({!unqualified_settings_reads}). Written
    twice, the drift that matters was silent — a third predicate added to the second list and not to
    the first loses its keys from the census both consistency tests rest on, and a census that stops
    seeing a key looks exactly like a key nobody reads (gh-ocannl-750). The other direction was
    always loud, which is the asymmetry that let the two lists sit forty lines apart.

    Each entry's key list is what the predicate reads: the threshold it hides, and the field it
    gates. They are not derivable from the name — [with_runtime_debug] gates
    [output_debug_files_in_build_directory] — so the table is the definition, and
    [config_scan_lexing] pins each entry in both positions. *)
let settings_predicates =
  [
    ("debug_log_from_routines", [ "log_level"; "debug_log_from_routines" ]);
    ("with_runtime_debug", [ "log_level"; "output_debug_files_in_build_directory" ]);
  ]

(** The names alone, for the position that cares which identifiers are predicates rather than what
    they read. *)
let settings_predicate_names = List.map settings_predicates ~f:fst

(** The keys a call of the predicate named by the last component of [path] implies, if it is one. *)
let settings_predicate_keys path =
  match List.last path with
  | Some name -> List.Assoc.find settings_predicates name ~equal:String.equal
  | None -> None

(** The other spelling of a configuration read: a field of the startup-resolved [Utils.settings]
    record, whose field names {e are} the config keys, and the predicates over it that fold in the
    [log_level > 1] threshold ({!settings_predicates}). A census built from [arg_name] literals
    alone would miss every one of these — [large_models] is read as [Utils.settings.large_models] in
    the codegen, so a future misclassification of it could pass unchallenged (Codex P2 on PR #337).

    A field {e read} and a predicate {e call}: naming either in prose is not a use of it. *)
let settings_keys_in_source content =
  let keys = ref [] in
  let ends_with path suffix =
    let extra = List.length path - List.length suffix in
    extra >= 0 && List.equal String.equal (List.drop path extra) suffix
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr with
        (* Qualified: [Low_level.virtualize_settings] and friends are records of the same shape
           whose field names are NOT config keys ([max_visits] against [virtualize_max_visits]), so
           an unqualified match would attribute reads to keys that do not exist. *)
        (* The field label may itself be module-qualified -- [r.Utils.large_models] is how one
           disambiguates a field name -- so the read is named by its LAST component, not by the
           whole label (Codex P2, round 7). The receiver check stays as it is. *)
        | { pexp_desc = Pexp_field (record, { txt = field; _ }); _ } -> (
            match longident_of record with
            | Some path when ends_with path [ "Utils"; "settings" ] -> (
                match List.last (flatten_longident field) with
                | Some field -> keys := field :: !keys
                | None -> ())
            | _ -> ())
        (* One arm over the table rather than an arm per predicate: the guard is the same question
           every time -- does this path END in a predicate's name -- and asking it once is what
           keeps the table the only place a predicate is named (gh-ocannl-750). *)
        | [%expr [%e? f] ()] -> (
            match Option.bind (longident_of f) ~f:settings_predicate_keys with
            | Some implied -> keys := implied @ !keys
            | None -> ())
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !keys

(** Where a file reads a field of a record called [settings] without the [Utils.] receiver that
    {!settings_keys_in_source} recognises — a bare [settings.large_models], or one through an alias
    ([module U = Utils], [let open Utils in …], [include Utils], at any depth).

    That receiver is the convention the settings census rests on, and a read spelled any other way
    vanishes from it: a key read only that way at codegen could then be classified code-borne and
    pass [digest_completeness] unchallenged, which is the misclassification that test exists to
    catch (Codex P2, rounds 14 and 15 of PR #343).

    Resolving aliases and opens is the other way to close this, and a bigger machine than the
    convention needs — no source spells a read that way today, so the convention is checked instead,
    exactly as the string-literal one for [arg_name] is. Policing the READ rather than the scope is
    what makes it precise: [module Lazy = Utils.Lazy] and a qualified record expression such as
    [Utils.{ value; unique_id }] introduce no such read and do not appear here, while
    [Low_level.virtualize_settings.max_visits] is a different record and never did.

    [offset] is where the read sits, for the caller's report. *)
let unqualified_settings_reads content =
  let found = ref [] in
  let is_utils_settings path =
    List.length path >= 2
    && List.equal String.equal (List.drop path (List.length path - 2)) [ "Utils"; "settings" ]
  in
  (* Where `Utils.settings` is used AS a record rather than read through: the receiver of a field
     access is the one place it may appear. Everything else -- `let s = Utils.settings`, `read
     Utils.settings`, storing it, returning it -- puts the reads out of the census's reach, since
     they are then spelled against a name this scan has no reason to know (Codex P2, rounds 16 and
     17 of PR #343). Blessing that one position covers the whole class at once, where naming the
     ways to alias it did not. *)
  let blessed = Hash_set.create (module Int) in
  (* The predicates {!settings_keys_in_source} folds thresholds into are recognised as CALLS, so
     handing one around as a value loses its keys the same way handing the record around does (Codex
     P2, round 20). Their one visible position is the function of an application.

     The same table names them there, so a predicate cannot be added to one site and not the other
     (gh-ocannl-750). *)
  let predicates = settings_predicate_names in
  let ends_in names path =
    List.last path |> Option.value_map ~default:false ~f:(List.mem names ~equal:String.equal)
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_apply (f, _) -> (
            match longident_of f with
            | Some path when ends_in predicates path ->
                Hash_set.add blessed f.pexp_loc.loc_start.pos_cnum
            | _ -> ())
        | _ -> ());
        (match expr.pexp_desc with
        (* A write is not a read -- the census counts none of these -- but it is the same qualified
           use of the record, and `Utils.settings.k <- v` is how train.ml sets a few. *)
        | Pexp_field (receiver, _) | Pexp_setfield (receiver, _, _) -> (
            match longident_of receiver with
            | Some path when is_utils_settings path ->
                Hash_set.add blessed receiver.pexp_loc.loc_start.pos_cnum
            | Some path
              when List.last path |> Option.value_map ~default:false ~f:(String.equal "settings") ->
                (* A field read of something CALLED settings but not `Utils.settings`: a bare
                   `settings.k` under an open, or `U.settings.k` under an alias. *)
                found := expr.pexp_loc.loc_start.pos_cnum :: !found
            | _ -> ())
        | Pexp_ident { txt; _ } when ends_in predicates (flatten_longident txt) ->
            (* Blessed above if this is the function of an application. *)
            found := expr.pexp_loc.loc_start.pos_cnum :: !found
        | Pexp_ident { txt; _ }
          when List.last (flatten_longident txt)
               |> Option.value_map ~default:false ~f:(String.equal "settings") ->
            (* Any identifier ending in `settings`, not only the qualified one: under a local open
               the record is spelled bare, and `read settings` hands it on just as `read
               Utils.settings` does (Codex P2, round 18). Recorded now and discarded below if it
               turns out to be a blessed receiver -- a field access is visited before its
               receiver. *)
            found := expr.pexp_loc.loc_start.pos_cnum :: !found
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !found |> List.filter ~f:(fun offset -> not (Hash_set.mem blessed offset))

(** Every configuration read of a file — [arg_name] call sites and {!settings_keys_in_source} —
    keyed by file basename, for tests that care {e where} a key is read. Field names that are not
    config keys come along; callers intersect with the registry. *)
let keys_by_file files =
  List.map files ~f:(fun fname ->
      let content = Stdio.In_channel.read_all fname in
      ( Stdlib.Filename.basename fname,
        Set.of_list (module String) (keys_in_source content @ settings_keys_in_source content) ))

(** Whether [content] reads configuration key [key] by name, through either spelling this module
    recognises: an [~arg_name:"key"] call site, or a field of the startup-resolved [Utils.settings].

    What it is for: a stanza declaring [OCANNL_<KEY>] is justified by SOME read of that key among
    its modules, and calling the initializer is only one of them. A converse check that knew only
    the initializer would report a test reading the key directly as carrying a stale declaration,
    and so would make the documented way of pinning a key unusable for it (Codex P2, round 2). *)
let source_reads_key content ~key =
  let has keys = List.mem keys key ~equal:String.equal in
  has (keys_in_source content) || has (settings_keys_in_source content)
