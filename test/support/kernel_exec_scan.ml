(** The tests that compile or execute kernel text they built themselves (gh-ocannl-1685).

    {!Codegen_text_scan} finds the tests that PIN emitted text. A second population breaks on a
    codegen change without pinning anything: the tests whose kernels did not come out of the
    [Assignments] pipeline. gh-ocannl-1338 renamed every GPU node identifier to [t_<name>] and ran
    all 103 text-pinning tests on CUDA and HIP; master still went red on CUDA, because
    [shipped_kernels_alone]'s staged callback writes [sk_staged[0] = 2.0f;] verbatim -- text no
    renderer renames -- and nothing listed it.

    A test reaches such text three ways, and the scan names which:

    - {b a staged callback}: it CONSTRUCTS a [Low_level.Staged_compilation], whose document the
      C-family renderers emit verbatim, invisible to every analysis and renamed by nothing. A
      pattern matching the constructor builds nothing and is not a member. Whether the callback then
      reaches a compiler is not decided here: its text is opaque by design, so the scan does not
      guess from the surrounding syntax, and a test that only renders one costs a run, not a miss.
    - {b a prelowered compile}: it hands [Context.compile] or [Context.compile_outcome] a lowering
      of its own through the [?prelowered] seam (gh-ocannl-562) -- an IR shape the pipeline never
      emits, compiled as written. [Context.link_delta]'s [~prelowered:true] links nothing and is not
      the seam.
    - {b through a harness}: it calls a value of a support module that does either of the above,
      directly or through further values -- [Ll_test.execute] reaches [Ll_test.link]'s seam call,
      and [Bench_harness.time_shipped_kernels] compiles each shipped kernel alone through it.

    The harness values are DERIVED, not listed: {!harness} reads the support modules handed over and
    closes the set under reference, so a new executing helper is on the frontier the day it is
    written. A support module's [include] passes its included module's values on, and a test reaches
    a harness module by a qualifier, an alias of one, an [open] or an [include].

    The reading is syntactic and errs toward listing: a name that resolves to a harness value counts
    whether or not the binding in scope is that value, and an [open] counts file-wide. A spurious
    member costs one run; a missing one is the gh-ocannl-1338 red.

    {1 The tests calling a backend's compiler themselves}

    A third population breaks on a change that is not to the renderers at all -- to a backend's
    builtins prelude, its compiler flags, how its compiler is resolved -- and pins nothing either:
    the tests that hand source of their own to a backend's compiler, the way [test_shared_builtins]
    compiles [Builtins_cc.source] and a probe of its own with [Cc_backend.compiler_command]
    (gh-ocannl-1772). The scan recognises them by a fourth route, {b a compiler call}: a reference
    to one of {!compiler_entries}, the values through which a test reaches a backend's compiler,
    which names the backend. The entries are spellings, as the two seams above are; what keeps them
    from going stale silently is [codegen_text_inventory]'s claim that each is still spelled by some
    source it reads ({!entries_named}). A harness value calling a compiler puts its callers on the
    list as it does for the other routes, and {!base_routes} tells the populations apart. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
module Longident = Ppxlib.Longident

(** A route to self-built kernel text, as the inventory prints it. *)
type route =
  | Staged  (** constructs a [Staged_compilation] *)
  | Prelowered  (** passes [?prelowered] to [compile] / [compile_outcome] *)
  | Compiles of string  (** references a {!compiler_entries} value of this backend *)
  | Through of string  (** references this harness value, as [Module.value] *)

let compare_route : route -> route -> int = Poly.compare

let route_name = function
  | Staged -> "staged callback"
  | Prelowered -> "prelowered compile"
  | Compiles backend -> "calls " ^ backend ^ "'s compiler"
  | Through value -> "through " ^ value

(** Whether a base route is a kernel built outside the pipeline (the gh-ocannl-1685 population). *)
let builds_kernel_text = function Staged | Prelowered -> true | Compiles _ | Through _ -> false

(** The backend whose compiler a base route calls (the gh-ocannl-1772 population). *)
let compiler_of = function
  | Compiles backend -> Some backend
  | Staged | Prelowered | Through _ -> None

type entry = { entry_backend : string; entry_path : string list }
(** A value through which a test reaches [entry_backend]'s compiler with source of its own, spelled
    as the module path suffix that names it: [Cc.compiler_command] under
    [module Cc = Context.Cc_backend] expands to a path ending in [Cc_backend.compiler_command]. *)

(** The compiler entry points. cc's are the backend's own interface: the command its kernels are
    built with and the flags passed after it, which a test pastes into a shell command of its own.
    The GPU backends' are where a test reaches the runtime compiler: [Nvrtc]'s call, HIP's
    production seam assembling hiprtc's options, and Metal's library-from-source. *)
let compiler_entries =
  [
    { entry_backend = "cc"; entry_path = [ "Cc_backend"; "compiler_command" ] };
    { entry_backend = "cc"; entry_path = [ "Cc_backend"; "kernel_compiler_flags" ] };
    { entry_backend = "cc"; entry_path = [ "Cc_backend"; "kernel_link_flags" ] };
    { entry_backend = "cuda"; entry_path = [ "Nvrtc"; "compile_to_ptx" ] };
    { entry_backend = "hip"; entry_path = [ "Hip_backend"; "Impl"; "hip_to_code" ] };
    { entry_backend = "metal"; entry_path = [ "Library"; "on_device" ] };
  ]

let entry_name entry = String.concat ~sep:"." entry.entry_path

(** The spellings the two direct routes are recognized by. The constructor and the label are
    [Low_level.t]'s and [Context]'s own; the callee names keep [Context.link_delta]'s link-free
    [~prelowered:true] out. *)
let staged_constructor = "Staged_compilation"

let seam_label = "prelowered"
let seam_callees = [ "compile"; "compile_outcome" ]

(** A source module's name: [ll_test.ml] is [Ll_test], and a [(select)] arm [x.real.ml] or
    [x.missing.ml] is the module [X] it stands for. *)
let module_name_of_path path =
  let base = match String.rsplit2 path ~on:'/' with Some (_, base) -> base | None -> path in
  let stem = String.chop_suffix_if_exists base ~suffix:".ml" in
  let stem =
    List.fold [ ".real"; ".missing" ] ~init:stem ~f:(fun stem suffix ->
        String.chop_suffix_if_exists stem ~suffix)
  in
  String.capitalize stem

let flatten = Longident.flatten_exn

type scope = {
  aliases : (string * string list) list;
  opened : string list list;
  included : string list list;
  items : (string * expression) list;
}
(** What a parsed source says about the modules it reaches: its aliases ([module H = Bench_harness],
    at any depth), the module paths it opens or includes (file-wide, whatever their scope), the
    paths it includes alone, and its value bindings at structure level, nested modules flattened. *)

let rec module_path me =
  match me.pmod_desc with
  | Pmod_ident { txt; _ } -> Some (flatten txt)
  | Pmod_constraint (me, _) -> module_path me
  | _ -> None

let scope_of structure =
  let aliases = ref [] and opened = ref [] and included = ref [] in
  let alias name me =
    match (name, module_path me) with
    | Some name, Some path -> aliases := (name, path) :: !aliases
    | _ -> ()
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! module_binding mb =
        alias mb.pmb_name.txt mb.pmb_expr;
        super#module_binding mb

      method! expression e =
        (match e.pexp_desc with
        | Pexp_letmodule ({ txt; _ }, me, _) -> alias txt me
        | Pexp_open (od, _) ->
            Option.iter (module_path od.popen_expr) ~f:(fun p -> opened := p :: !opened)
        | _ -> ());
        super#expression e

      method! open_declaration od =
        Option.iter (module_path od.popen_expr) ~f:(fun p -> opened := p :: !opened);
        super#open_declaration od

      method! include_declaration id =
        Option.iter (module_path id.pincl_mod) ~f:(fun p ->
            opened := p :: !opened;
            included := p :: !included);
        super#include_declaration id
    end
  in
  iterator#structure structure;
  let rec items_of structure =
    List.concat_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.filter_map bindings ~f:(fun vb ->
                match vb.pvb_pat.ppat_desc with
                | Ppat_var { txt; _ } -> Some (txt, vb.pvb_expr)
                | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) ->
                    Some (txt, vb.pvb_expr)
                | _ -> None)
        | Pstr_module { pmb_expr; _ } -> items_of_module pmb_expr
        | Pstr_recmodule mbs -> List.concat_map mbs ~f:(fun mb -> items_of_module mb.pmb_expr)
        | _ -> [])
  and items_of_module me =
    match me.pmod_desc with
    | Pmod_structure s -> items_of s
    | Pmod_constraint (me, _) -> items_of_module me
    | _ -> []
  in
  { aliases = !aliases; opened = !opened; included = !included; items = items_of structure }

(** [path] with a leading alias the file binds replaced by what it names, followed a few steps (an
    alias of an alias), never forever. *)
let expand scope path =
  let rec go fuel path =
    match path with
    | first :: rest when fuel > 0 -> (
        match List.Assoc.find scope.aliases first ~equal:String.equal with
        | Some target -> go (fuel - 1) (target @ rest)
        | None -> path)
    | _ -> path
  in
  go 4 path

(** The harness module a module path reaches: its rightmost component that names one, so that
    [Test_utils.Generated] and a bare [Generated] answer alike. *)
let harness_module_of ~known scope path = List.find (List.rev (expand scope path)) ~f:known

(** The compiler entries an identifier [path], spelled in a file with [scope], names: those whose
    path is a suffix of the identifier's with its leading alias expanded, or of the identifier
    prefixed by a module the file opens -- so [compiler_command] under [open Cc_backend] counts, and
    [Impl.hip_to_code] under [open Hip_backend]. *)
let entries_at scope path =
  let candidates =
    expand scope path :: List.map scope.opened ~f:(fun opened -> expand scope opened @ path)
  in
  List.filter compiler_entries ~f:(fun entry ->
      List.exists candidates ~f:(fun candidate ->
          List.is_suffix candidate ~suffix:entry.entry_path ~equal:String.equal))

type harness = {
  modules : Set.M(String).t;  (** every support module read, executing or not *)
  values : route list Map.M(String).t;
      (** every value of them reaching self-built kernel text, as [Module.value], with the routes
          that put it there: the frontier the census was taken with *)
}

(** The routes [path] (a reference inside a file with [scope], itself the module [self] when a
    harness module) resolves to: a harness value it names, or -- a bare name -- one of [self]'s own
    or of a module the file opens. *)
let resolve harness ~self scope path =
  let known m = Set.mem harness.modules m in
  let find m v = Option.map (Map.find harness.values (m ^ "." ^ v)) ~f:(fun _ -> m ^ "." ^ v) in
  match List.rev path with
  | [] -> []
  | [ name ] ->
      let candidates =
        Option.to_list self
        @ List.filter_map scope.opened ~f:(fun p -> harness_module_of ~known scope p)
      in
      List.filter_map candidates ~f:(fun m -> find m name)
  | name :: rev_modules -> (
      match harness_module_of ~known scope (List.rev rev_modules) with
      | Some m -> Option.to_list (find m name)
      | None -> [])

(* The routes found anywhere under a node, each reference resolved by [resolve_ref] and matched
   against the compiler entries by [entries_of]. *)
let collector resolve_ref entries_of =
  object
    inherit [route list] Ast_traverse.fold as super

    method! expression e acc =
      let acc =
        match e.pexp_desc with
        | Pexp_construct ({ txt; _ }, _)
          when String.equal (Longident.last_exn txt) staged_constructor ->
            Staged :: acc
        | Pexp_apply ({ pexp_desc = Pexp_ident { txt; _ }; _ }, args)
          when List.mem seam_callees (Longident.last_exn txt) ~equal:String.equal
               && List.exists args ~f:(fun (label, _) ->
                   match label with
                   | Asttypes.Labelled l | Asttypes.Optional l -> String.equal l seam_label
                   | Asttypes.Nolabel -> false) ->
            Prelowered :: acc
        | Pexp_ident { txt; _ } ->
            let path = flatten txt in
            List.map (entries_of path) ~f:(fun entry -> Compiles entry.entry_backend)
            @ List.map (resolve_ref path) ~f:(fun v -> Through v)
            @ acc
        | _ -> acc
      in
      super#expression e acc
  end

let routes_in harness ~self scope expr =
  (collector (resolve harness ~self scope) (entries_at scope))#expression expr []
  |> List.dedup_and_sort ~compare:compare_route

(** The harness frontier, closed under reference over [modules] -- each a module name with its
    source. Every value's routes are recomputed each round until none changes; they only grow, so
    the closure ends. A module that does not parse raises: a harness the scan cannot read would
    leave its callers off the census silently. *)
let harness modules =
  let parsed =
    List.map modules ~f:(fun (name, contents) ->
        (name, scope_of (Codegen_text_scan.structure_of contents)))
  in
  let names = Set.of_list (module String) (List.map parsed ~f:fst) in
  let step harness =
    List.fold parsed ~init:harness.values ~f:(fun values (m, scope) ->
        let values =
          List.fold scope.items ~init:values ~f:(fun values (v, body) ->
              let key = m ^ "." ^ v in
              let routes =
                List.filter (routes_in harness ~self:(Some m) scope body) ~f:(function
                  | Through other -> not (String.equal other key)
                  | Staged | Prelowered | Compiles _ -> true)
              in
              if List.is_empty routes then values else Map.set values ~key ~data:routes)
        in
        (* An include re-exports: what the included module reaches, this one reaches by the same
           name, unless a binding of its own says more. *)
        List.fold scope.included ~init:values ~f:(fun values path ->
            match harness_module_of ~known:(Set.mem names) scope path with
            | None -> values
            | Some inc ->
                Map.fold harness.values ~init:values ~f:(fun ~key ~data:_ values ->
                    match String.lsplit2 key ~on:'.' with
                    | Some (owner, v) when String.equal owner inc ->
                        Map.update values
                          (m ^ "." ^ v)
                          ~f:(function Some routes -> routes | None -> [ Through key ])
                    | _ -> values)))
  in
  let rec close harness =
    let values = step harness in
    if Map.equal (List.equal Poly.equal) values harness.values then harness
    else close { harness with values }
  in
  close { modules = names; values = Map.empty (module String) }

(** The routes by which a test source reaches self-built kernel text; empty when it reaches none.
    Raises if [contents] does not parse. *)
let classify harness ~contents =
  let structure = Codegen_text_scan.structure_of contents in
  let scope = scope_of structure in
  (collector (resolve harness ~self:None scope) (entries_at scope))#structure structure []
  |> List.dedup_and_sort ~compare:compare_route

(** The routes [routes] reach once every [Through] is followed to the harness values it names:
    [Staged], [Prelowered] and [Compiles], deduplicated. *)
let base_routes harness routes =
  let rec go seen acc = function
    | [] -> acc
    | Through value :: rest ->
        if Set.mem seen value then go seen acc rest
        else
          go (Set.add seen value) acc
            (Option.value (Map.find harness.values value) ~default:[] @ rest)
    | route :: rest -> go seen (route :: acc) rest
  in
  go (Set.empty (module String)) [] routes |> List.dedup_and_sort ~compare:compare_route

(** [routes] restricted to those reaching a base route [keep] holds of: a direct route kept as it
    is, a [Through] kept when the value it names reaches one. How one population is printed without
    the other's routes. *)
let restrict harness ~keep routes =
  List.filter routes ~f:(fun route -> List.exists (base_routes harness [ route ]) ~f:keep)

(** The compiler entries a source spells anywhere, as {!classify} would match them: the evidence
    that an entry is still a name the tree uses, whether or not the file is a member. Raises if
    [contents] does not parse. *)
let entries_named ~contents =
  let structure = Codegen_text_scan.structure_of contents in
  let scope = scope_of structure in
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_ident { txt; _ } -> found := entries_at scope (flatten txt) @ !found
        | _ -> ());
        super#expression e
    end
  in
  iterator#structure structure;
  List.dedup_and_sort !found ~compare:Poly.compare
