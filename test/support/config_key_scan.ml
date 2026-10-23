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

    Shared by the initializer census below and [dead_export_scan], which match a reference by its
    receiver the same way, because a second copy of these rules is exactly the restatement this
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
    as if it were: the safe direction for the rule built on this, [env_var_deps]' refusal of the
    initializer in a library module, where a false call costs a moved module and a missed one an
    initializer running in every executable that links the library. *)
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

(** Every site where an OCaml source names one of the requested paths under the scanner's lexical
    module model: the path as the source SPELLS it -- [F.small_menu] under
    [module F = Fissioned_chain] -- beside the requested path it resolves to, so a caller can follow
    one particular reference (gh-ocannl-1368's width-pinning helpers).

    Used for env_var_deps' derivation of the programs starting an autotune search (gh-ocannl-1368).
    Lifecycle membership instead uses {!Lifecycle_scan}'s resolution-free Dune contract
    (gh-ocannl-798). This source pass remains an approximation rather than a typed OCaml
    environment. *)
let module_reference_sites content ~paths =
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
                if matches components path then
                  found := (components, String.concat ~sep:"." path) :: !found)
        | None -> ());
        super#longident lid
    end
  in
  walk#structure structure;
  List.rev !found

(** Which requested paths an OCaml source names: {!module_reference_sites}, by requested path. *)
let module_references_in_source content ~paths =
  List.fold (module_reference_sites content ~paths) ~init:[] ~f:(fun found (_, spelling) ->
      if List.mem found spelling ~equal:String.equal then found else spelling :: found)
  |> List.rev

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

type keyed_read = { read_key : string; reader : string list option; defaulted : bool; at : int }
(** A configuration key named by a string literal, with what reads it (gh-ocannl-1332): at an
    application site [~arg_name:"k"], [reader] is the applied function's path ([None] when it is no
    plain identifier) and [defaulted] whether the same application passes [~default] or [?default];
    as a parameter's default ([?(arg_name = "k")]), [reader] is [None] and [defaulted] is [true] --
    a helper that reads [k] unless told otherwise. [at] is where the literal sits. *)

let keyed_reads content =
  let reads = ref [] in
  let is_default = function
    | Asttypes.Labelled "default" | Asttypes.Optional "default" -> true
    | _ -> false
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_apply (fn, args) ->
            let defaulted = List.exists args ~f:(fun (lbl, _) -> is_default lbl) in
            List.iter args ~f:(fun (lbl, arg) ->
                if is_our_label lbl then
                  Option.iter (string_literal arg) ~f:(fun read_key ->
                      reads :=
                        {
                          read_key;
                          reader = longident_of fn;
                          defaulted;
                          at = arg.pexp_loc.loc_start.pos_cnum;
                        }
                        :: !reads))
        | Pexp_function (params, _, _) ->
            List.iter params ~f:(fun param ->
                match param.pparam_desc with
                | Pparam_val (lbl, Some default, _) when is_our_label lbl ->
                    Option.iter (string_literal default) ~f:(fun read_key ->
                        reads :=
                          {
                            read_key;
                            reader = None;
                            defaulted = true;
                            at = param.pparam_loc.loc_start.pos_cnum;
                          }
                          :: !reads)
                | _ -> ())
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !reads

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

    What it is for is a question about a program that NAMES one of these values from outside: which
    keys -- or, through [external_keys], which tagged entry points -- its run reaches. A reach is a
    superset where it errs, and errs that way on purpose: a local binding that shadows a top-level
    name is still read as a use of it, and an optional argument a caller supplies -- which spares
    the read it defaults to -- is not modeled.

    In source order, so a top-level name refers to the binding before it, as in OCaml, and a
    [let rec] group is closed to its fixpoint. A nested [module M = struct … end] is walked too, its
    values reported as [M.v] and resolved innermost scope first; a functor's body is not. A
    QUALIFIED value no local one answers to -- [Utils.debug_log_from_routines], [Tnode.Placements.f]
    -- contributes whatever [external_keys] answers for its path, after a top-level [module X = Y]
    alias of this module has been expanded; the caller decides which modules it can follow, since
    this function sees one file. A field of [Utils.settings] contributes what [settings_field_keys]
    answers for it (the field's own name by default).

    {b Scope of the name resolution} (gh-ocannl-1430). Aliases follow source order like values do: a
    top-level [module X = Y] expands [X.v] only in the items after it, a later [module X = …]
    rebinds [X] from that point on, and [Y] is resolved where the alias is written, so
    [module X = W] before [module W = Public] keeps naming the outer [W]. What is followed:
    top-level value bindings, nested [module M = struct … end] (as [M.v]), [include struct … end],
    [include M] of a structure this file defines or of a top-level alias, and a top-level
    [include Other] for the bare names no local value answers to. What is not, each read as naming
    nothing local: an [open] (a bare name under [open Other] resolves only through an [include]), a
    [let module X = Y in] or a [module X = Y] inside a nested structure, a first-class module, and a
    functor's body or application. Unlike the shadowing above, an unfollowed form errs SHORT: a key
    reached only through one is missing from the reach every caller reads ([env_var_deps]' search
    wrappers, [digest_completeness]' search-shaping reach).

    {b Lazies} (gh-ocannl-1535). [on_lazy] is called for every [lazy e] or [Lazy.from_fun e] (also
    as [Lazy.from_fun @@ f] and [f |> Lazy.from_fun]) that no function encloses in the right-hand
    side of a value binding or a top-level expression of a walked structure, with the holder (the
    binding's name, qualified by its nested modules) and the keys its body reaches in the scope it
    sees: the environment at that point of the walk -- the group's own for a [let rec] -- plus the
    initializer's local [let] bindings around it. [functor_bodies] walks a functor's body as a
    nested structure too, its parameters naming nothing local. [opaque] names values whose own reads
    are not credited to them. *)
let top_level_key_reach ?(external_keys = fun (_ : string list) -> [])
    ?(settings_field_keys = fun field -> [ field ]) ?(opaque = Set.empty (module String))
    ?(functor_bodies = false) ?(on_lazy = fun ~holder:_ ~at:_ (_ : string list) -> ()) content =
  (* The names a binding is credited under: its simple name. *)
  let names_of pattern = Option.to_list (pattern_name pattern) in
  let structure = structure_of content in
  (* The top-level [module X = Y] aliases in scope at the item being walked, each target already
     expanded through the aliases in scope where it was written (gh-ocannl-1430): so one step of
     lookup is the whole expansion, and an alias written later can neither reach back to an earlier
     use nor redirect an earlier alias's target. *)
  let aliases = ref [] in
  let expand path =
    match path with
    | head :: rest -> (
        match List.Assoc.find !aliases head ~equal:String.equal with
        | Some target -> target @ rest
        | None -> path)
    | [] -> path
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
  (* The other modules a top-level [include Other] brings into scope (gh-ocannl-1331: [Autotune]
     includes [Sketch_families]), whose values a bare name no local value answers to may be. *)
  let included = ref [] in
  let resolve ~env ~scope path =
    let rec local_as path depth =
      if depth < 0 then None
      else
        match Map.find env (dotted (List.take scope depth @ path)) with
        | Some keys -> Some keys
        | None -> local_as path (depth - 1)
    in
    (* As written, then through a top-level [module W = Widths] alias of a structure this file
       defines: the alias names a local module as surely as an external one (gh-ocannl-1320, Codex
       P2 round 9). *)
    let local depth =
      match local_as path depth with
      | Some keys -> Some keys
      | None ->
          let expanded = expand path in
          if List.equal String.equal expanded path then None else local_as expanded depth
    in
    match local (List.length scope) with
    | Some keys -> keys
    | None -> (
        match path with
        | [ _ ] ->
            List.fold !included
              ~init:(Set.empty (module String))
              ~f:(fun acc m ->
                Set.union acc (Set.of_list (module String) (external_keys (m @ path))))
        | _ -> Set.of_list (module String) (external_keys (expand path)))
  in
  let reach_of ~env ~scope (keys, names) =
    List.fold names
      ~init:(Set.of_list (module String) keys)
      ~f:(fun acc path -> Set.union acc (resolve ~env ~scope path))
  in
  (* gh-ocannl-1535: the lazies [expr] creates when it is evaluated once -- every [lazy e] or
     [Lazy.from_fun e] no function of [expr] encloses -- each handed to [on_lazy] with the keys its
     body reaches in [env]. *)
  (* Off while a recursive module group is walked to seed its members (below). *)
  let reporting = ref true in
  let report_lazies ~env ~scope ~holder expr =
    let under_function = ref 0 in
    (* The initializer's own local bindings in scope, innermost first, each with what it reaches: a
       lazy naming one is credited with it (round 2 of PR #1366). A local that shadows a top-level
       name credits both, the long direction. *)
    let locals = ref [] in
    let reach_here (keys, names) =
      List.fold names
        ~init:(reach_of ~env ~scope (keys, names))
        ~f:(fun acc path ->
          match path with
          | [ name ] -> (
              match List.Assoc.find !locals name ~equal:String.equal with
              | Some local -> Set.union acc local
              | None -> acc)
          | _ -> acc)
    in
    let is_from_fun fn =
      match longident_of fn with
      | Some path -> (
          match List.drop path (List.length path - 2) with
          | [ "Lazy"; "from_fun" ] -> true
          | _ -> false)
      | None -> false
    in
    let is_operator name fn =
      match longident_of fn with
      | Some [ op ] | Some [ "Stdlib"; op ] -> String.equal op name
      | _ -> false
    in
    (* The body [Lazy.from_fun] is applied to, however the application is spelled: directly, or
       through the standard application operators [@@] and [|>]. *)
    let from_fun_body expr =
      match expr.pexp_desc with
      | Pexp_apply (fn, [ (Asttypes.Nolabel, body) ]) when is_from_fun fn -> Some body
      | Pexp_apply (op, [ (Asttypes.Nolabel, fn); (Asttypes.Nolabel, body) ])
        when is_operator "@@" op && is_from_fun fn ->
          Some body
      | Pexp_apply (op, [ (Asttypes.Nolabel, body); (Asttypes.Nolabel, fn) ])
        when is_operator "|>" op && is_from_fun fn ->
          Some body
      | _ -> None
    in
    let report body (loc : Location.t) =
      if !reporting then
        on_lazy ~holder ~at:loc.loc_start.pos_cnum (Set.to_list (reach_here (direct_of body)))
    in
    let finder =
      object (self)
        inherit Ast_traverse.iter as super

        (* An object's method runs when it is called, like a function body; its [val] initializers
           run when the object is built (round 5 of PR #1366). *)
        method! class_field field =
          match field.pcf_desc with
          | Pcf_method _ ->
              Int.incr under_function;
              super#class_field field;
              Int.decr under_function
          | _ -> super#class_field field

        method! expression expr =
          match expr.pexp_desc with
          | Pexp_function _ ->
              Int.incr under_function;
              super#expression expr;
              Int.decr under_function
          | Pexp_lazy body when !under_function = 0 ->
              report body expr.pexp_loc;
              super#expression expr
          | Pexp_let (rec_flag, bindings, body) ->
              let saved = !locals in
              let bind () =
                List.iter bindings ~f:(fun binding ->
                    Option.iter (pattern_name binding.pvb_pat) ~f:(fun name ->
                        locals := (name, reach_here (direct_of binding.pvb_expr)) :: !locals))
              in
              (match rec_flag with
              | Recursive ->
                  (* The group closed to its fixpoint, each member in scope in every body. *)
                  let members =
                    List.filter_map bindings ~f:(fun binding ->
                        Option.map (pattern_name binding.pvb_pat) ~f:(fun name ->
                            (name, binding.pvb_expr)))
                  in
                  let rec close current =
                    locals := current @ saved;
                    let next =
                      List.map members ~f:(fun (name, rhs) -> (name, reach_here (direct_of rhs)))
                    in
                    if not (List.equal (fun (_, a) (_, b) -> Set.equal a b) next current) then
                      close next
                  in
                  close (List.map members ~f:(fun (name, _) -> (name, Set.empty (module String))));
                  List.iter bindings ~f:(fun binding -> self#expression binding.pvb_expr)
              | Nonrecursive ->
                  List.iter bindings ~f:(fun binding -> self#expression binding.pvb_expr);
                  bind ());
              self#expression body;
              locals := saved
          | _ -> (
              match from_fun_body expr with
              | Some body when !under_function = 0 ->
                  report body expr.pexp_loc;
                  super#expression expr
              | _ -> super#expression expr)
      end
    in
    finder#expression expr
  in
  let rec unwrap module_expr =
    match module_expr.pmod_desc with Pmod_constraint (inner, _) -> unwrap inner | _ -> module_expr
  in
  (* A functor's body, walked only under [functor_bodies]: its parameters' values are another
     module's, so a key reached through one is not found. *)
  let rec functor_body module_expr =
    match (unwrap module_expr).pmod_desc with
    | Pmod_functor (_, body) -> functor_body body
    | Pmod_structure items -> Some items
    | _ -> None
  in
  let rec walk ~scope env items =
    List.fold items ~init:env ~f:(fun env item ->
        match item.pstr_desc with
        | Pstr_value (rec_flag, bindings) ->
            (* Every name the binding is credited under ([names_of] above). *)
            let group =
              List.concat_map bindings ~f:(fun binding ->
                  match names_of binding.pvb_pat with
                  | [] -> []
                  | names ->
                      let direct = direct_of binding.pvb_expr in
                      List.map names ~f:(fun name ->
                          let name = dotted (scope @ [ name ]) in
                          (name, if Set.mem opaque name then ([], []) else direct)))
            in
            let rec close env =
              let env' =
                List.fold group ~init:env ~f:(fun acc (name, direct) ->
                    Map.set acc ~key:name ~data:(reach_of ~env:acc ~scope direct))
              in
              if Map.equal Set.equal env env' then env else close env'
            in
            let env' =
              match rec_flag with
              | Recursive ->
                  close
                    (List.fold group ~init:env ~f:(fun acc (name, _) ->
                         Map.set acc ~key:name ~data:(Set.empty (module String))))
              | Nonrecursive ->
                  List.fold group ~init:env ~f:(fun acc (name, direct) ->
                      Map.set acc ~key:name ~data:(reach_of ~env ~scope direct))
            in
            (* gh-ocannl-1535: the lazies this item creates at initialization, each resolved in the
               scope its body sees -- the group itself when it is recursive, what came before it
               otherwise. *)
            let lazy_env = match rec_flag with Recursive -> env' | Nonrecursive -> env in
            List.iter bindings ~f:(fun binding ->
                let holder =
                  Option.map (pattern_name binding.pvb_pat) ~f:(fun n -> dotted (scope @ [ n ]))
                in
                report_lazies ~env:lazy_env ~scope ~holder binding.pvb_expr);
            env'
        | Pstr_eval (expr, _) ->
            report_lazies ~env ~scope ~holder:None expr;
            env
        (* A nested structure's values are reached as [M.v], from inside it and from after it. A
           functor's body is not walked, unless [functor_bodies] asks: what its values read depends
           on its argument. *)
        | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
            (* Whatever the module is, from here on [name] means it and not an earlier alias: a new
               alias's target, expanded through the aliases before it ([module W = W] keeps an
               earlier [W]'s meaning), or nothing. From AFTER the binding: inside the body of
               [module W = struct … end], [W] still names what it named before (Codex P2, round 1 of
               PR #1324), so the body is walked first. *)
            let module_expr = unwrap pmb_expr in
            let env =
              match module_expr.pmod_desc with
              | Pmod_structure inner -> walk ~scope:(scope @ [ name ]) env inner
              | Pmod_functor _ when functor_bodies -> (
                  match functor_body module_expr with
                  | Some inner -> walk ~scope:(scope @ [ name ]) env inner
                  | None -> env)
              | _ -> env
            in
            (if List.is_empty scope then
               let target =
                 match module_expr.pmod_desc with
                 | Pmod_ident { txt; _ } -> Option.map (flatten_module_path txt) ~f:expand
                 | _ -> None
               in
               aliases :=
                 match target with
                 | Some target -> List.Assoc.add !aliases name target ~equal:String.equal
                 | None -> List.Assoc.remove !aliases name ~equal:String.equal);
            env
        (* An [include struct … end] exports its values into the enclosing scope, so they are walked
           there (gh-ocannl-1320, Codex P2 round 6). An included functor application is not: what
           its values read depends on its argument, as for a functor's body. *)
        (* An extension's structure payload ([let%debug_sexp f … = …], ppx_minidebug's) is the
           code its expansion keeps, so its items are walked in place (round 3 of PR #1366). *)
        | Pstr_extension ((_, PStr items), _) -> walk ~scope env items
        (* A recursive module's body is walked as a nested structure's, like [module M = struct …
           end] (round 3 of PR #1366); its name is no alias. The group is walked twice, the first
           time silently, so a member's lazy sees every member's values (round 4) -- its siblings'
           only: the replay of a member starts without what the silent pass bound inside it, so its
           own body is resolved in source order again (round 5). *)
        | Pstr_recmodule bindings ->
            let walk_group ~replay env =
              List.fold bindings ~init:env ~f:(fun env binding ->
                  match binding.pmb_name.txt with
                  | None -> env
                  | Some name -> (
                      let env =
                        if not replay then env
                        else
                          let own = dotted (scope @ [ name ]) ^ "." in
                          Map.filter_keys env ~f:(fun key -> not (String.is_prefix key ~prefix:own))
                      in
                      let module_expr = unwrap binding.pmb_expr in
                      match module_expr.pmod_desc with
                      | Pmod_structure inner -> walk ~scope:(scope @ [ name ]) env inner
                      | Pmod_functor _ when functor_bodies -> (
                          match functor_body module_expr with
                          | Some inner -> walk ~scope:(scope @ [ name ]) env inner
                          | None -> env)
                      | _ -> env))
            in
            let outer = !reporting in
            reporting := false;
            let seeded = walk_group ~replay:false env in
            reporting := outer;
            walk_group ~replay:true seeded
        | Pstr_include { pincl_mod; _ } -> (
            match (unwrap pincl_mod).pmod_desc with
            | Pmod_structure inner -> walk ~scope env inner
            (* [include M] of a structure this file walked as [M.v] re-exports each [v] here (Codex
               P2, round 7). *)
            | Pmod_ident { txt; _ } -> (
                (* Through a top-level [module W = Public] alias too (Codex P2, round 10). *)
                match Option.map (flatten_module_path txt) ~f:expand with
                | Some path ->
                    if List.is_empty scope then included := path :: !included;
                    (* The module named, resolved innermost scope first: inside [Make]'s body,
                       [include Helpers] names an enclosing [Helpers] as surely as a sibling one
                       (round 3 of PR #1366). *)
                    let named depth = dotted (List.take scope depth @ path) ^ "." in
                    let rec resolve_from depth =
                      if depth <= 0 then named 0
                      else if
                        Map.existsi env ~f:(fun ~key ~data:_ ->
                            String.is_prefix key ~prefix:(named depth))
                      then named depth
                      else resolve_from (depth - 1)
                    in
                    let prefix = resolve_from (List.length scope) in
                    Map.fold env ~init:env ~f:(fun ~key ~data acc ->
                        (* Every descendant, nested submodules' values included (Codex P2, round
                           8). *)
                        match String.chop_prefix key ~prefix with
                        | Some rest -> Map.set acc ~key:(dotted (scope @ [ rest ])) ~data
                        | None -> acc)
                | None -> env)
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

type held_read = { held_key : string; holder : string; held_at : int }
(** A configuration key a module holds for the process's lifetime: [holder] is the module-level
    binding the lazy sits in ([Multidev.num_devices] inside a nested module or functor, ["_"] where
    no binding names it), [held_at] the offset of the lazy. *)

(** The configuration keys whose FIRST resolved value the process keeps (gh-ocannl-1535): the keys
    read when a [lazy] created at module initialization is forced. Such a lazy is forced once, and
    its value -- or the exception its first force raised, which OCaml re-raises on every later force
    -- answers for the rest of the process, so a setting a test states in-process after the first
    force ([Hashtbl.set Utils.config_file_args]) never reaches it.

    {b Recognised forms}, a closed list:
    - a lazy is [lazy e], or [Lazy.from_fun e] under any receiver path ending in [Lazy.from_fun];
    - it is created at module initialization when no [fun]/[function] encloses it inside the
      right-hand side of a value binding or a top-level expression of a structure
      {!top_level_key_reach} walks -- the file's, a nested [module M = struct … end]'s, an included
      structure's, and here a functor body's too (a functor is applied once per backend, and
      backends are process-wide singletons): a binding's right-hand side, a closure's environment
      ([let f = let v = lazy … in fun () -> …]), an argument at module level
      ([stream_switch (lazy …)]), and the body of an enclosing module-level lazy;
    - the keys it reads are what {!top_level_key_reach} credits to its body, resolved in the scope
      the body sees: every literal [~arg_name] read, lambdas included, a field of [Utils.settings],
      and the reach of each same-source value it names, as bound at that point of the walk (the
      group itself for a [let rec]), through the walk's module aliases, includes and nested scopes.
      [opaque] names values not followed: the lookup plumbing, which reads the key it is handed
      rather than one of its own.

    [external_keys] and [settings_field_keys] are {!top_level_key_reach}'s: with a
    {!library_reach}'s, a call into another module of the library is followed through that module's
    own reach (a lazy in [Schedulers] calling [Cc_backend.effective_pool_width]), and a field of
    [Utils.settings] stands for every key its initializer reads.

    Where it errs, it errs as {!top_level_key_reach} does: long where a local binding shadows a
    top-level name and where a value is only mentioned, short where that model names what it does
    not follow -- an [open], a [let module], a first-class module, a functor's parameter, a module
    outside [external_keys]. Beyond that it does not see a lazy in a class or object body, a memo
    built some other way (a [ref] cell, a table), or a value computed EAGERLY at initialization,
    which is a different class: it is read before any test code runs. *)
let process_lifetime_reads ?opaque ?external_keys ?settings_field_keys content =
  let found = ref [] in
  let on_lazy ~holder ~at keys =
    List.iter keys ~f:(fun held_key ->
        if not (String.is_empty held_key) then
          found := { held_key; holder = Option.value holder ~default:"_"; held_at = at } :: !found)
  in
  ignore
    (top_level_key_reach ?external_keys ?settings_field_keys ?opaque ~functor_bodies:true ~on_lazy
       content
      : key_reach);
  List.rev !found

(** The heading of the agent-note section listing the process-lifetime reads. *)
let process_lifetime_heading = "### Process-lifetime settings"

(** The entries of that section in [lines] (a note's text): each bullet whose text opens with a code
    span naming an [.ml] file -- [- `schedule.ml`: `key`, `key`] -- paired with the code spans after
    it, a bullet running on over its indented continuation lines. The section runs from its heading
    to the next heading; any other bullet in it is prose. No section, no entries. *)
let process_lifetime_entries lines =
  let rec section = function
    | [] -> []
    | line :: rest when String.equal (String.rstrip line) process_lifetime_heading ->
        List.take_while rest ~f:(fun l -> not (String.is_prefix l ~prefix:"#"))
    | _ :: rest -> section rest
  in
  let code_spans text = String.split text ~on:'`' |> List.filteri ~f:(fun i _ -> i % 2 = 1) in
  let rec bullets acc = function
    | [] -> List.rev acc
    | line :: rest when String.is_prefix line ~prefix:"- " ->
        let continuation, rest =
          List.split_while rest ~f:(fun l ->
              String.is_prefix l ~prefix:" " && not (String.is_empty (String.strip l)))
        in
        bullets (String.concat ~sep:" " (line :: continuation) :: acc) rest
    | _ :: rest -> bullets acc rest
  in
  List.filter_map
    (bullets [] (section lines))
    ~f:(fun bullet ->
      if not (String.is_prefix bullet ~prefix:"- `") then None
      else
        match code_spans bullet with
        | file :: keys when String.is_suffix file ~suffix:".ml" -> Some (file, keys)
        | _ -> None)

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

(** Every value path [content] names, flattened: [C.Cc_backend.vector_bytes_setting] contributes
    [["C"; "Cc_backend"; "vector_bytes_setting"]]. *)
let value_paths_in_source content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_ident { txt; _ } ->
            Option.iter (flatten_module_path txt) ~f:(fun path -> found := path :: !found)
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure (structure_of content);
  List.rev !found

(** Which top-level values of a module STATE one of [literals] -- set a setting rather than read it
    (gh-ocannl-1368: which support values pin the cc vector file, the derivation its blessed helpers
    are held equal to).

    A statement is a WRITE that carries the literal in its arguments: an application of one of
    [table_writes] ([Hashtbl.set], ...) to one of [tables] ([Utils.config_file_args]), or of one of
    [env_writes] ([Unix.putenv]). A value also states the literal when it hands it to a top-level
    value of the same module that writes whatever it is given -- [Fissioned_chain.small_menu] passes
    ["cc_vector_bytes"] in a list to its local [state], whose body is a [Hashtbl.set] on the table
    -- and when it names a value that states it. A literal outside a write's arguments is not a
    statement: a [Hashtbl.mem] on the table, an [~arg_name] read, a format string, a table of names
    (Codex P2, round 1 of staging#1263).

    A write whose value is a string literal [pins] rejects states nothing -- ["-1"] and [""] leave
    the cc width on its auto-probe -- and neither does a [(literal, value)] pair whose value is one;
    a computed value is trusted.

    Flow-insensitive within a value, and an over-read where it errs: a value that sets the key and
    later unsets it states it. A literal reaching a write through a local binding, or nested modules
    and functor bodies, are not followed. *)
let stating_reach ~literals ~tables ~table_writes ~env_writes ~pins content =
  let structure = structure_of content in
  let ident_path expr =
    match expr.pexp_desc with Pexp_ident { txt; _ } -> flatten_module_path txt | _ -> None
  in
  let ends_in names expr =
    Option.exists (ident_path expr) ~f:(fun path ->
        List.mem names (List.last_exn path) ~equal:String.equal)
  in
  let is_literal expr =
    Option.exists (string_literal expr) ~f:(fun s -> List.mem literals s ~equal:String.equal)
  in
  let unpinning value = Option.exists (string_literal value) ~f:(Fn.non pins) in
  let positional (lbl, _) = match lbl with Asttypes.Nolabel -> true | _ -> false in
  (* Whether an argument list carries a literal a write could state: anywhere in it, except under an
     [~arg_name] argument, and except a key paired with a literal value [pins] rejects. *)
  let carries args =
    let found = ref false in
    let iterator =
      object (self)
        inherit Ast_traverse.iter as super

        method! expression expr =
          match expr.pexp_desc with
          | Pexp_apply (fn, args) ->
              self#expression fn;
              List.iter args ~f:(fun (lbl, arg) ->
                  if not (is_our_label lbl) then self#expression arg)
          | Pexp_tuple [ key; value ] when is_literal key ->
              if not (unpinning value) then found := true;
              self#expression value
          | _ ->
              if is_literal expr then found := true;
              super#expression expr
      end
    in
    List.iter args ~f:(fun (lbl, arg) -> if not (is_our_label lbl) then iterator#expression arg);
    !found
  in
  (* A write, and the value it sets: its [~data], or else its last positional argument. *)
  let write_value fn args =
    if
      ends_in table_writes fn
      && List.exists args ~f:(fun ((_, arg) as a) -> positional a && ends_in tables arg)
      || ends_in env_writes fn
    then
      Some
        (match
           List.find args ~f:(fun (lbl, _) ->
               match lbl with Asttypes.Labelled "data" -> true | _ -> false)
         with
        | Some (_, value) -> Some value
        | None -> Option.map (List.last (List.filter args ~f:positional)) ~f:snd)
    else None
  in
  let direct_of expr =
    let writes = ref false and states = ref false and calls = ref [] and names = ref [] in
    let iterator =
      object (self)
        inherit Ast_traverse.iter as super

        method! expression expr =
          match expr.pexp_desc with
          | Pexp_apply (fn, args) ->
              (match write_value fn args with
              | Some value ->
                  writes := true;
                  if carries args && not (Option.exists value ~f:unpinning) then states := true
              | None ->
                  if carries args then
                    Option.iter (ident_path fn) ~f:(fun p -> calls := p :: !calls));
              self#expression fn;
              List.iter args ~f:(fun (lbl, arg) ->
                  if not (is_our_label lbl) then self#expression arg)
          | _ ->
              Option.iter (ident_path expr) ~f:(fun path -> names := path :: !names);
              super#expression expr
      end
    in
    iterator#expression expr;
    (!writes, !states, !calls, !names)
  in
  let named =
    List.concat_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.filter_map bindings ~f:(fun b ->
                Option.map (pattern_name b.pvb_pat) ~f:(fun name -> (name, direct_of b.pvb_expr)))
        | _ -> [])
  in
  let local name = List.Assoc.mem named name ~equal:String.equal in
  (* Both closed to their fixpoint over the whole file: a name refers to any top-level value of that
     spelling, the flow-insensitive over-read the header admits. *)
  let fixpoint step =
    let rec go set =
      let set' =
        List.fold named ~init:set ~f:(fun acc (name, direct) ->
            if Set.mem acc name || not (step acc direct) then acc else Set.add acc name)
      in
      if Set.length set' = Set.length set then set else go set'
    in
    go (Set.empty (module String))
  in
  let names_any set names =
    List.exists names ~f:(function [ name ] -> Set.mem set name && local name | _ -> false)
  in
  (* The values that write whatever they are handed: a write in their body, or a call of one. *)
  let writing = fixpoint (fun set (writes, _, _, names) -> writes || names_any set names) in
  let states_with set (_, states, calls, names) =
    states || names_any writing calls || names_any set names
  in
  Set.to_list (fixpoint states_with)

type initialization_item = {
  runs : bool;
      (** Whether evaluating the item runs code at module initialization: a bare expression, or a
          binding of anything but a function. *)
  paths : string list list;
      (** Every value path the item names, nested modules and function bodies included. *)
  called : string list list;
      (** The functions the item CALLS in statement position, which a run of it reaches: the
          elements of a sequence, a [let]'s bound expressions and body, the branches of an [if],
          through type constraints and local [open]s. A path in a closure the item builds, or in an
          argument, is not one. *)
  literals : string list;  (** Every string literal the item spells. *)
}

(** Each top-level item of [content], in source order (gh-ocannl-1368: whether a program pins the cc
    vector file before its first search, and keeps it pinned). *)
let initialization_items content =
  let rec statements expr =
    match expr.pexp_desc with
    | Pexp_sequence (first, rest) -> statements first @ statements rest
    | Pexp_let (_, bindings, body) ->
        List.concat_map bindings ~f:(fun b -> statements b.pvb_expr) @ statements body
    | Pexp_ifthenelse (_, then_, else_) ->
        statements then_ @ Option.value_map else_ ~default:[] ~f:statements
    | Pexp_constraint (inner, _) | Pexp_open (_, inner) -> statements inner
    | Pexp_apply ({ pexp_desc = Pexp_ident { txt; _ }; _ }, _) ->
        Option.to_list (flatten_module_path txt)
    | _ -> []
  in
  List.map (structure_of content) ~f:(fun item ->
      let paths = ref [] and literals = ref [] in
      let iterator =
        object
          inherit Ast_traverse.iter as super

          method! expression expr =
            (match expr.pexp_desc with
            | Pexp_ident { txt; _ } ->
                Option.iter (flatten_module_path txt) ~f:(fun path -> paths := path :: !paths)
            | _ -> Option.iter (string_literal expr) ~f:(fun s -> literals := s :: !literals));
            super#expression expr
        end
      in
      iterator#structure_item item;
      let evaluated =
        match item.pstr_desc with
        | Pstr_eval (expr, _) -> [ expr ]
        | Pstr_value (_, bindings) ->
            List.filter_map bindings ~f:(fun b ->
                match b.pvb_expr.pexp_desc with Pexp_function _ -> None | _ -> Some b.pvb_expr)
        | _ -> []
      in
      {
        runs = not (List.is_empty evaluated);
        paths = List.rev !paths;
        called = List.concat_map evaluated ~f:statements;
        literals = List.rev !literals;
      })

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

type library_reach = {
  external_keys : string list -> string list;
      (** The keys the value a qualified path names reaches: a [Utils] settings predicate's
          ({!settings_predicate_keys}), or a top-level value of one of the library's modules, read
          by {!top_level_key_reach} with this same function following its own qualified calls. A
          leading component among [wrappers] names a dune library's wrapper module and is dropped
          ([Ir.Schedule.f] is [Schedule.f]). Any other path reaches nothing. *)
  settings_field_keys : string -> string list;
      (** Every key that initializes a field of [Utils.settings] (its [restore_settings]
          assignment), so a compatibility alias is answered for beside the key; the field's own name
          where nothing does. *)
}

(** The key reach of a library given as [(module name, file on disk)] pairs, each module read once
    and only when a path reaches it (gh-ocannl-1149; shared with gh-ocannl-1331's check that the
    search-shaping keys are read within [Autotune.tune]'s reach). OCaml admits no cycle between a
    library's modules, so the in-progress guard only keeps a malformed tree from looping. [opaque]
    names, per module, the values not followed (gh-ocannl-1535: the lookup plumbing, which reads the
    key it is handed). *)
let library_reach ?(wrappers = []) ?(opaque = fun (_ : string) -> Set.empty (module String))
    ~library_modules () =
  let settings_initializers =
    match List.Assoc.find library_modules "Utils" ~equal:String.equal with
    | Some on_disk -> settings_field_initializers (Stdio.In_channel.read_all on_disk)
    | None -> []
  in
  let settings_field_keys field =
    match List.Assoc.find settings_initializers field ~equal:String.equal with
    | Some (_ :: _ as keys) -> keys
    | Some [] | None -> [ field ]
  in
  let module_reaches = Hashtbl.create (module String) in
  let rec external_keys path =
    match settings_predicate_keys path with
    | Some keys when match path with [ "Utils"; _ ] -> true | _ -> false -> keys
    | _ -> (
        match path with
        | wrapper :: (_ :: _ :: _ as inner) when List.mem wrappers wrapper ~equal:String.equal ->
            external_keys inner
        | module_name :: (_ :: _ as value) -> (
            match module_reach module_name with
            | Some reach ->
                Option.value
                  (List.Assoc.find reach (String.concat ~sep:"." value) ~equal:String.equal)
                  ~default:[]
            | None -> [])
        | _ -> [])
  and module_reach module_name =
    match Hashtbl.find module_reaches module_name with
    | Some reach -> reach
    | None ->
        Option.map (List.Assoc.find library_modules module_name ~equal:String.equal)
          ~f:(fun on_disk ->
            Hashtbl.set module_reaches ~key:module_name ~data:(Some []);
            let reach =
              (top_level_key_reach ~external_keys ~settings_field_keys ~opaque:(opaque module_name)
                 (Stdio.In_channel.read_all on_disk))
                .reach
            in
            Hashtbl.set module_reaches ~key:module_name ~data:(Some reach);
            reach)
  in
  { external_keys; settings_field_keys }
