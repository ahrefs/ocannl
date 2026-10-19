(** Whose an identifier is: [Verdict]'s, or not -- the one resolver every [Verdict] reader asks
    (gh-ocannl-1361).

    [verdict_ratchet] runs several readers over the tree: the quantified-claim provenance layer
    ({!Verdict_provenance}), the line-break and dialect readers built on
    {!Verdict_scan.claim_calls}, and the gate reader ({!Verdict_scan.gate_census}). Each used to
    decide for itself whether a callee was [Verdict]'s, and they disagreed at the edges: the
    provenance layer read a [module Verdict = Local_printer] as [Verdict] and an unqualified [p]
    past an [open] of another module as certainly [Verdict]'s, the gate reader took any unbound
    [gated_all] for [Verdict]'s, and staging#1150 spent six review rounds fixing the line-break
    reader one edge at a time. So the question is answered once, here, by one walk of the source,
    and the readers look the answer up by the identifier's location.

    {1 What it decides}

    For every value identifier whose last component is a {!members} name (a member of
    [Verdict.Claims]), one {!ownership}, resolved through {!Lexical_scope} at the point the
    identifier is spelled:

    - A QUALIFIED identifier ([Verdict.p], [V.p], [Verdict.Claims.p], [C.p]) is [Verdict]'s when its
      module path denotes [Verdict]: the head [Verdict] with no binding of that name in scope, a
      module the source binds to such a path ([module V = Verdict], [module V = Verdict.Claims]), or
      any submodule of either. A structure of this source ([module C = struct ... end]) exports a
      claim name as [Verdict]'s when it [include]s such a module and binds no value of that name
      after the include. Any other path is not [Verdict]'s: [Other.Verdict.p], a [Verdict] the
      source took over ([module Verdict = Local_printer]), a functor parameter, a module nested in a
      structure ([M.N.p]).
    - An UNQUALIFIED identifier resolves through the scope's frames, innermost first. A binding of
      the source (a [let], a parameter, a pattern, an [external], a class's names, an [open] of a
      structure of the source that binds the name) makes it not [Verdict]'s; an [open] or [include]
      of a module denoting [Verdict] makes it [Verdict]'s; an [open] of a module this resolver
      cannot read is passed over, and makes the name {!Ambiguous} if an open of [Verdict] is reached
      past it. A name nothing binds and no open of [Verdict] brings in is {!Free}.

    {1 Ambiguity}

    What a module this resolver cannot read exports is unknowable: it may define a value named [p],
    or a submodule named [Verdict]. Staging#1150 met both failure directions of that one fact --
    read as "not [Verdict]'s", every claim after an [open Ll_test] went silently unchecked; read as
    "certainly [Verdict]'s", a module exporting its own [p] would be refused with no way out but
    renaming. So a name that reaches [Verdict] only past such an [open] is {!Ambiguous}: every
    reader still reads it, and a refusal of one adds the remedy, to qualify the call if it is the
    other module's. That covers qualified paths too: past an [open] of an unreadable module,
    [Verdict.p] (and an alias of [Verdict]) may name that module's own [Verdict].

    The modules this resolver can read are [Verdict] itself, the structures of the source, and the
    standard libraries ({!standard_libraries}: [Base], [Core], [Stdlib], [Caml], [Stdio]), which
    export neither a claim name nor a module named [Verdict]. Any other module -- [Ocannl], [Ir],
    [Test_utils], a fixture's [Other] -- is unreadable.

    {1 What is not decided here}

    Which of the {!ownership}s a reader reads: the line-break and provenance readers read [Verdicts]
    and [Ambiguous]; the gate reader reads [Free] too, since a [gated_*] name exists nowhere but in
    [Verdict.Claims]. And everything beyond the callee's identity -- labels, populations, values
    bound to a wrapper -- stays each reader's own. *)

open Base
open Ppxlib
module Read = Config_key_scan

(** The [Verdict.Claims] members, the names whose identifiers this resolver decides. Held equal to
    the module in [test/support/verdict.ml] by [verdict_ratchet]. *)
let members =
  [
    "fail";
    "claim";
    "p";
    "pf";
    "claimf";
    "p_all";
    "p_all2";
    "pf_all2";
    "p_none";
    "p_alli";
    "p_empty";
    "p_exists";
    "p_pairwise_distinct";
    "skipped";
    "gated";
    "gated_all";
    "gated_alli";
    "gated_exists";
    "pass_fail";
    "pass_fail_all2";
    "case";
  ]

let member name = List.mem members name ~equal:String.equal

(** Libraries whose exports this resolver knows: no claim name, and no module named [Verdict]. *)
let standard_libraries = [ "Base"; "Core"; "Stdlib"; "Caml"; "Stdio" ]

type ownership =
  | Verdicts  (** Certainly [Verdict]'s. *)
  | Ambiguous
      (** [Verdict]'s only past an [open] of a module this resolver cannot read, which may export a
          name or a [Verdict] module of its own. *)
  | Free  (** Unqualified, and nothing in scope binds it or brings it in from [Verdict]. *)
  | Not_verdicts  (** The source's own, or another module's. *)

type resolution = {
  member : string;  (** The member the identifier names, its last component. *)
  ownership : ownership;
  spelled : string;  (** The identifier as written. *)
}

(* What a claim name denotes in a frame. *)
type value = Source | From_verdict of { ambiguous : bool } | From_elsewhere

(* What a module name denotes. *)
type module_ =
  | Verdict_library of { ambiguous : bool }
  | Structure of value Map.M(String).t
      (** A structure of the source: the claim names it exports, and what each denotes. *)
  | Standard
  | Unknown

(* A module-environment key no source can bind: present once an open of an unreadable module is in
   scope, after which an unbound [Verdict] may be that module's. *)
let opened_elsewhere = "<opened-elsewhere>"

let all_members denotes =
  Map.of_alist_exn (module String) (List.map members ~f:(fun name -> (name, denotes)))

let ambiguate = function
  | Verdict_library _ -> Verdict_library { ambiguous = true }
  | Structure exports ->
      Structure
        (Map.map exports ~f:(function
          | From_verdict _ -> From_verdict { ambiguous = true }
          | other -> other))
  | (Standard | Unknown) as other -> other

let ownership_of_value = function
  | From_verdict { ambiguous } -> if ambiguous then Ambiguous else Verdicts
  | Source | From_elsewhere -> Not_verdicts

type t = {
  values : (int * int, resolution) Hashtbl.t;
  modules : (int * int, ownership) Hashtbl.t;
      (** Every module path spelled as a module expression -- opened, included, bound, a functor's
          argument -- that denotes [Verdict] ([Verdicts] or [Ambiguous]). *)
}

let key (loc : Location.t) = (loc.loc_start.pos_cnum, loc.loc_end.pos_cnum)

let of_structure ast : t =
  let table = Hashtbl.Poly.create () and modules = Hashtbl.Poly.create () in
  let walker =
    object (self)
      inherit [value, module_] Lexical_scope.scoped as super
      method local = Source
      method! shadowed = Some Unknown
      method! attribute _ attr = attr

      method module_path env path =
        let components =
          Option.value ~default:[] (Option.try_with (fun () -> Read.flatten_longident path))
        in
        let head = Lexical_scope.head path in
        Some
          (match Map.find env.Lexical_scope.modules head with
          | Some (Structure _ as structure) when List.length components = 1 -> structure
          | Some (Structure _) -> Unknown
          | Some denotes -> denotes
          | None when String.equal head "Verdict" ->
              Verdict_library { ambiguous = Map.mem env.Lexical_scope.modules opened_elsewhere }
          | None when List.mem standard_libraries head ~equal:String.equal -> Standard
          | None -> Unknown)

      (* A structure of the source, by the claim names it exports: a value it binds is its own, an
         include of [Verdict] exports every claim name as [Verdict]'s, an include of an unreadable
         module as anyone's, each over what came before. Include paths resolve in the scope around
         the structure. *)
      method! module_of env module_expr =
        match module_expr.pmod_desc with
        | Pmod_structure items ->
            let exports =
              List.fold items
                ~init:(Map.empty (module String))
                ~f:(fun exports item ->
                  let own names =
                    List.fold names ~init:exports ~f:(fun exports name ->
                        if member name then Map.set exports ~key:name ~data:Source else exports)
                  in
                  let over included =
                    Map.merge_skewed exports included ~combine:(fun ~key:_ _ later -> later)
                  in
                  match item.pstr_desc with
                  | Pstr_value (_, bindings) ->
                      own (Lexical_scope.pattern_vars (List.map bindings ~f:(fun b -> b.pvb_pat)))
                  | Pstr_primitive { pval_name = { txt; _ }; _ } -> own [ txt ]
                  | Pstr_include { pincl_mod; _ } -> (
                      match self#module_of env pincl_mod with
                      | Some (Verdict_library { ambiguous }) ->
                          over (all_members (From_verdict { ambiguous }))
                      | Some (Structure included) -> over included
                      | Some Standard -> exports
                      | Some Unknown | None -> over (all_members From_elsewhere))
                  | _ -> exports)
            in
            Some (Structure exports)
        | _ -> super#module_of env module_expr

      (* A module path spelled as a module expression, recorded where it denotes [Verdict]: what a
         reader that carries modules through functor applications needs, since inside the functor's
         body the parameter is not [Verdict] to a lexical resolver. *)
      method! module_expr env module_expr =
        (match module_expr.pmod_desc with
        | Pmod_ident { txt; loc } -> (
            match self#module_path env txt with
            | Some (Verdict_library { ambiguous }) ->
                Hashtbl.set modules ~key:(key loc) ~data:(if ambiguous then Ambiguous else Verdicts)
            | _ -> ())
        | _ -> ());
        super#module_expr env module_expr

      method! opened ~top:_ ~include_:_ env denotes =
        match denotes with
        | Some (Verdict_library { ambiguous }) ->
            Lexical_scope.push_frame env (all_members (From_verdict { ambiguous }))
        | Some (Structure exports) -> Lexical_scope.push_frame env exports
        | Some Standard -> env
        | Some Unknown | None ->
            let env = Lexical_scope.push_frame env (all_members From_elsewhere) in
            {
              env with
              Lexical_scope.modules =
                Map.set
                  (Map.map env.Lexical_scope.modules ~f:ambiguate)
                  ~key:opened_elsewhere ~data:Unknown;
            }

      method! ident env ident =
        let record name ownership =
          Hashtbl.set table ~key:(key ident.loc)
            ~data:
              {
                member = name;
                ownership;
                spelled =
                  String.concat ~sep:"."
                    (Option.value ~default:[ name ]
                       (Option.try_with (fun () -> Read.flatten_longident ident.txt)));
              }
        in
        match ident.txt with
        | Longident.Lident name when member name ->
            let rec decide passed = function
              | [] -> Free
              | From_elsewhere :: outer -> decide true outer
              | From_verdict { ambiguous } :: _ ->
                  if passed || ambiguous then Ambiguous else Verdicts
              | Source :: _ -> Not_verdicts
            in
            record name
              (decide false
                 (List.filter_map env.Lexical_scope.frames ~f:(fun frame -> Map.find frame name)))
        | Ldot (prefix, name) when member name ->
            record name
              (match self#module_path env prefix with
              | Some (Verdict_library { ambiguous }) -> if ambiguous then Ambiguous else Verdicts
              | Some (Structure exports) ->
                  Option.value_map (Map.find exports name) ~default:Not_verdicts
                    ~f:ownership_of_value
              | Some (Standard | Unknown) | None -> Not_verdicts)
        | _ -> ()
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  { values = table; modules }

(* A callee under type constraints or coercions is the path it wraps. *)
let rec peel expr =
  match expr.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel inner
  | _ -> expr

(** The resolution of [expr], a claim name's identifier under any constraints, where it is one. *)
let resolve (t : t) expr =
  match (peel expr).pexp_desc with
  | Pexp_ident { loc; _ } -> Hashtbl.find t.values (key loc)
  | _ -> None

(** Whether [module_expr], under any constraints, is a path denoting [Verdict]: [Verdicts],
    [Ambiguous], or [Not_verdicts] for anything else. *)
let rec module_ownership (t : t) module_expr =
  match module_expr.pmod_desc with
  | Pmod_constraint (inner, _) -> module_ownership t inner
  | Pmod_ident { loc; _ } -> Option.value (Hashtbl.find t.modules (key loc)) ~default:Not_verdicts
  | _ -> Not_verdicts
