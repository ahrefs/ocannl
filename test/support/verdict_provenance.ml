(** Scope-and-polarity provenance of Boolean expressions: the analysis layer behind
    [verdict_ratchet]'s quantified-claim rule (gh-ocannl-931).

    The rule (gh-ocannl-729, gh-ocannl-801, gh-ocannl-887): a Boolean that reaches a [Verdict] claim
    must not be able to pass on an empty population. [List.for_all], [for_all2_exn] and [is_empty]
    are true of nothing, and [not (exists …)] likewise, so a claim whose value is one of those, in
    the polarity in which it is vacuous, prints exactly the line a hundred elements would have
    printed. What the reader has to know at a claim site is therefore where the value came from,
    through whatever the source wrote between the quantifier and the claim: a binding, a helper, a
    wrapper parameter, a match, an [if], a local module, a callback.

    Before this module the ratchet answered that with several walkers, one per question -- what a
    binding returns, what it depends on, which names a wrapper's parameters reach, which quantifier
    a call resolves to -- each re-implementing scope and polarity for the syntax it happened to
    handle. Staging PR #633 ran 27 review rounds on the shapes one walker handled and another did
    not. This module models each syntax form ONCE, in {!walk}, as a {!provenance}:

    - two {!view}s, one per polarity -- what the expression's [true] can rest on, and what its
      [false] can. A view lists the {!source}s the value may reduce to (a quantifier written in the
      expression, or a parameter of an enclosing function whose actual argument is not yet known)
      and the {!view.witnesses}: the populations that are certainly non-empty whenever the value has
      that polarity. [not] swaps the two views; [&&] joins them; a conditional merges its branches.
      A source is covered once a witness names its population, and coverage is decided where the
      witness and the source meet, so a guard in a conjunction covers a quantifier written in a
      binding three references away, while a guard in a different scope covers nothing;
    - a constant, when the value is one, so that [if all then true else true] is a constant and
      [if all then true else false] is [all];
    - a {!closure}, when the value is a function: its parameters, the value it returns in terms of
      them, and the claims its body fires in terms of them. Applying a closure substitutes the
      actual arguments -- Boolean parameters by the argument's provenance, population parameters by
      the argument's identity, a destructured parameter by the matching component -- so a wrapper, a
      helper, a partially applied native claim and a [function]-case are the same mechanism.
      [Verdict.p] itself is a closure with a label slot and a value slot.

    Names are resolved through one environment, so every binder -- [let], parameters and their
    optional defaults, match cases, [open], [module] -- shadows and scopes the same way for every
    question. Population identity is lexical: [rows] bound by one [let] and [rows] bound by another
    are two populations, which is what keeps a witness from vouching for a shadowed collection
    without any sealing rule.

    The consumer of all this is {!claims}: every [Verdict] claim the structure fires, with the
    sources its value can rest on and who wrote each. *)

open Base
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
module Longident = Ppxlib.Longident
module Pprintast = Ppxlib.Pprintast
open Ppxlib.Parsetree
module Read = Config_key_scan
module Scan = Verdict_scan

type quantifier_kind = For_all | For_all2 | Is_empty | Not_exists

let quantifier_name = function
  | For_all -> "for_all"
  | For_all2 -> "for_all2_exn"
  | Is_empty -> "is_empty"
  | Not_exists -> "not exists"

let quantifier_arity = function For_all2 -> 2 | For_all | Is_empty | Not_exists -> 1

(* A definition's [position] is its absolute character offset, which no two bindings share; [line]
   and [column] are for saying where it is. Identity has to be the offset: `let refused = … in`
   twice in one expression, or two local scopes written on one line, are two bodies a line number
   cannot tell apart -- and telling two bodies apart is the whole of what an exemption key rests
   on. *)
type site = { line : int; column : int; position : int }

let site_of_location (location : Ppxlib.Location.t) =
  let start = location.loc_start in
  {
    line = start.Stdlib.Lexing.pos_lnum;
    column = start.Stdlib.Lexing.pos_cnum - start.Stdlib.Lexing.pos_bol;
    position = start.Stdlib.Lexing.pos_cnum;
  }

let describe_site site = Printf.sprintf "%d:%d" site.line site.column

type claim_kind = P | Pf | Pass_fail | Claim | Claimf

type binding = {
  name : string;
  site : site;
  span : int * int;
      (** The character range of the expression this binding names. Ownership is what an exemption
          names, so a quantifier goes to the binding a reader would exempt: the innermost one a
          claim reaches through. A binding referenced from outside takes over what was written
          inside its span -- a local of its body is an implementation detail of it -- while a local
          nothing named encloses (in a callback, in a claim argument) stays what there is to name.
      *)
  final : bool;
      (** A parameter, or a parameter default: owned outright, never taken over by the function
          whose span it sits in. *)
  mutable payload : payload;
}

and payload = Pending of (unit -> provenance) | Resolving | Resolved of provenance

and provenance = {
  when_true : view;
  when_false : view;
  constant : bool option;
  closure : closure option;
}

and view = {
  sources : source list;
      (** What the value can rest on at this polarity, minus the sources [witnesses] cover. *)
  witnesses : Set.M(String).t;
      (** Populations certainly non-empty when the value has this polarity. *)
}

and source = {
  origin : origin;
  owner : binding option;
      (** The innermost owning binding whose expression wrote the quantifier. *)
  written : site;
      (** Where the quantifier is written -- or, for a quantifier written directly in a claim's
          argument, that argument, which is the site the exemption of a direct quantifier names. *)
  via : Set.M(String).t;
      (** Optional-parameter labels whose default this source passed through; supplying the label at
          a call removes it. *)
}

and origin =
  | Quantifier of { kind : quantifier_kind; populations : Set.M(String).t }
  | Parameter of { parameter : binding; positive : bool }
  | Steering of { sources : source list; parameter : binding; positive : bool }
      (** [sources] steer to a branch whose value is [parameter] read at [positive]. A condition is
          the value only where the branch returns a constant, and a parameter's constant is not
          known until the closure is applied, so the decision is deferred: once the actual argument
          is that constant, the sources are the value's own; once it is the other, they are gone. *)

and closure =
  | Quantifier_function of quantifier_partial
  | Function of function_closure
  | Alternatives of closure list
      (** A function selected by control flow: applying it applies every alternative. *)

and quantifier_partial = {
  kind : quantifier_kind;
  populations : string option list;
      (** The positional arguments supplied so far, each resolved to its population identity. *)
  predicate : bool;  (** Whether [~f] has been supplied; [is_empty] takes none. *)
}

and function_closure = {
  parameters : parameter list;  (** The parameters not yet supplied, in order. *)
  claims : claim list;  (** Claims the body fires, in terms of the parameters. *)
  body : provenance;  (** What the body returns, in terms of the parameters. *)
  native : bool;
      (** A [Verdict] claim function itself, reached by its path or through an alias, as opposed to
          a local wrapper or a partial application: a direct quantifier reaching a native claim is
          named by the claim's label, one reaching a wrapper by the wrapper. *)
  format : bool;
      (** [Verdict.pf]/[claimf] before their format is supplied: the format literal decides how many
          arguments precede the claimed value. *)
  loose : bool;
      (** A format-taking claim whose format was not a literal: the last unlabelled argument of the
          next application is taken as the claimed value. *)
}

and parameter = {
  slot : binding;  (** The parameter as a whole, what a [function] case matches on. *)
  label : Asttypes.arg_label;
  patterns : pattern list;  (** One pattern, or one per [function] case. *)
  names : binding list;  (** The names the patterns bind, each a parameter of its own. *)
}

and claim = {
  fired : site;  (** The application that fired the claim, outermost. *)
  label : label;
  value : view;  (** The claimed Boolean's [when_true] view. *)
  helper : string option;
      (** The local wrapper the claim fired through, if any; a native claim's direct quantifiers are
          named by the label instead. *)
}

and label = Label_slot of binding | Label_expr of expression

(* ---------------------------------------------------------------------------------------------- *)
(* Views and their algebra. *)

(* [f] applied to every function a closure could be: a quantifier function is left alone, and a
   function selected by control flow is mapped through. *)
let rec map_functions closure ~f =
  match closure with
  | Quantifier_function _ -> closure
  | Function function_closure -> Function (f function_closure)
  | Alternatives closures -> Alternatives (List.map closures ~f:(map_functions ~f))

(* The functions a value selected by control flow could be, flattened, each once, and bounded:
   alternatives multiply through every application, and a bound keeps a long chain of
   function-returning branches from turning the walk exponential. Past the bound the first few stand
   for the rest, which loses precision loudly (an alternative not applied fires no claim) only in
   code shaped like a dispatch table of dispatch tables. *)
let alternatives_bound = 4

let select_closures closures =
  let rec flatten = function
    | Alternatives closures -> List.concat_map closures ~f:flatten
    | closure -> [ closure ]
  in
  let distinct =
    List.concat_map closures ~f:flatten
    |> List.fold ~init:[] ~f:(fun seen closure ->
        if List.exists seen ~f:(phys_equal closure) then seen else closure :: seen)
    |> List.rev
  in
  match List.take distinct alternatives_bound with
  | [] -> None
  | [ closure ] -> Some closure
  | closures -> Some (Alternatives closures)

let no_populations = Set.empty (module String)
let empty_view = { sources = []; witnesses = no_populations }
let nothing = { when_true = empty_view; when_false = empty_view; constant = None; closure = None }
let constant value = { nothing with constant = Some value }
let view_at provenance positive = if positive then provenance.when_true else provenance.when_false

let covered witnesses source =
  match source.origin with
  | Quantifier { populations; _ } ->
      (not (Set.is_empty populations)) && not (Set.is_empty (Set.inter populations witnesses))
  | Parameter _ | Steering _ -> false

let rec source_key source =
  let origin =
    match source.origin with
    | Quantifier { kind; populations } ->
        quantifier_name kind ^ "(" ^ String.concat ~sep:"," (Set.to_list populations) ^ ")"
    | Parameter { parameter; positive } ->
        Printf.sprintf "%s@%d%s" parameter.name parameter.site.position
          (if positive then "+" else "-")
    | Steering { sources; parameter; positive } ->
        Printf.sprintf "[%s]->%s@%d%s"
          (String.concat ~sep:";" (List.map sources ~f:source_key))
          parameter.name parameter.site.position
          (if positive then "+" else "-")
  in
  Printf.sprintf "%s@%d:%s:%s" origin source.written.position
    (Option.value_map source.owner ~default:"" ~f:(fun owner -> Int.to_string owner.site.position))
    (String.concat ~sep:"," (Set.to_list source.via))

(* A view's sources are the ones its witnesses do not cover: coverage is decided wherever a witness
   and a source meet, and never undone, since witnesses only accumulate along a conjunction. *)
let rec settle_sources witnesses sources =
  List.filter_map sources ~f:(fun source ->
      if covered witnesses source then None
      else
        match source.origin with
        | Steering { sources = steered; parameter; positive } -> (
            match settle_sources witnesses steered with
            | [] -> None
            | steered ->
                Some { source with origin = Steering { sources = steered; parameter; positive } })
        | Quantifier _ | Parameter _ -> Some source)
  |> List.dedup_and_sort ~compare:(fun a b -> String.compare (source_key a) (source_key b))

let settle view = { view with sources = settle_sources view.witnesses view.sources }

(* Both views hold: what either can rest on, vouched for by what either establishes. *)
let both a b =
  settle { sources = a.sources @ b.sources; witnesses = Set.union a.witnesses b.witnesses }

(* One of the views holds, which one unknown: what either can rest on, vouched for only by what both
   establish. *)
let either a b =
  settle { sources = a.sources @ b.sources; witnesses = Set.inter a.witnesses b.witnesses }

let merge views = List.reduce views ~f:either |> Option.value ~default:empty_view

let map_sources provenance ~f =
  let map_view view = { view with sources = List.map view.sources ~f } in
  {
    provenance with
    when_true = map_view provenance.when_true;
    when_false = map_view provenance.when_false;
  }

let reachable provenance positive =
  not (Option.equal Bool.equal provenance.constant (Some (not positive)))

let negate provenance =
  {
    when_true = provenance.when_false;
    when_false = provenance.when_true;
    constant = Option.map provenance.constant ~f:not;
    closure = None;
  }

let conjunction a b =
  match (a.constant, b.constant) with
  | Some false, _ | _, Some false -> constant false
  | Some true, _ -> { b with closure = None }
  | _, Some true -> { a with closure = None }
  | None, None ->
      {
        when_true = both a.when_true b.when_true;
        when_false = either a.when_false b.when_false;
        constant = None;
        closure = None;
      }

let disjunction a b = negate (conjunction (negate a) (negate b))

(* Aggregates -- tuples, records -- carry every component's sources at either polarity, since which
   component a later destructuring will read is not known here. *)
let aggregate components =
  let gather positive =
    {
      sources = List.concat_map components ~f:(fun c -> (view_at c positive).sources);
      witnesses = no_populations;
    }
  in
  { when_true = gather true; when_false = gather false; constant = None; closure = None }

(* One alternative of a conditional: the value it yields, and what steers to it -- the condition at
   the polarity that selects the branch, a case's guard. A condition steers between values; it IS
   the value only where the branch returns a constant, so its sources join the branch's only then,
   while its witnesses hold whenever the branch is taken. An alternative whose constant is the other
   polarity is one the value must have avoided, so its steering's other polarity holds on every path
   -- how `match () with () when some -> false | () -> true` rests on `not some`. *)
type alternative = { steering : provenance list; outcome : provenance }

let alternatives cases =
  let at positive =
    (* In order: an alternative the value must have avoided vouches only for the alternatives after
       it -- a later case's guard is never evaluated once an earlier case was taken. *)
    let _, taken =
      List.fold cases ~init:(empty_view, []) ~f:(fun (avoided, taken) { steering; outcome } ->
          if not (reachable outcome positive) then
            let avoided =
              List.fold steering ~init:avoided ~f:(fun acc s -> both acc s.when_false)
            in
            (avoided, taken)
          else
            let outcome_view = view_at outcome positive in
            let selecting = List.concat_map steering ~f:(fun s -> s.when_true.sources) in
            let steering_sources =
              if Option.equal Bool.equal outcome.constant (Some positive) then selecting
              else if List.is_empty selecting then []
              else
                (* The branch returns a parameter: whether the condition is the value waits on the
                   actual argument. *)
                List.filter_map outcome_view.sources ~f:(fun source ->
                    match source.origin with
                    | Parameter { parameter; positive } ->
                        Some
                          {
                            origin = Steering { sources = selecting; parameter; positive };
                            owner = None;
                            written = source.written;
                            via = no_populations;
                          }
                    | Quantifier _ | Steering _ -> None)
            in
            let witnesses =
              List.fold steering ~init:outcome_view.witnesses ~f:(fun acc s ->
                  Set.union acc s.when_true.witnesses)
            in
            let view =
              both (settle { sources = steering_sources @ outcome_view.sources; witnesses }) avoided
            in
            (avoided, view :: taken))
    in
    merge (List.rev taken)
  in
  let agreed =
    match cases with
    | [] -> None
    | first :: rest -> (
        match first.outcome.constant with
        | Some value
          when List.for_all rest ~f:(fun a ->
                   Option.equal Bool.equal a.outcome.constant (Some value)) ->
            Some value
        | _ -> None)
  in
  (* A function chosen by control flow is every function it could be. *)
  let closure =
    select_closures (List.filter_map cases ~f:(fun alternative -> alternative.outcome.closure))
  in
  (* Every alternative returning the same constant is that constant: what steered between them never
     reaches the value. *)
  match agreed with
  | Some value -> constant value
  | None -> { when_true = at true; when_false = at false; constant = None; closure }

(* ---------------------------------------------------------------------------------------------- *)
(* Syntax helpers. *)

let is_name expr name =
  match Read.longident_of expr with
  | Some path -> Option.value_map (List.last path) ~default:false ~f:(String.equal name)
  | None -> false

let path_ends path ~container ~member =
  match List.rev path with
  | found_member :: found_container :: _ ->
      String.equal found_member member && String.equal found_container container
  | _ -> false

let is_collection_path path ~member =
  path_ends path ~container:"Array" ~member || path_ends path ~container:"List" ~member

let is_collection_call expr ~member =
  Option.value_map (Read.longident_of expr) ~default:false ~f:(is_collection_path ~member)

let is_collection_module path =
  match List.last path with Some ("List" | "Array") -> true | _ -> false

let quantifier_of_member = function
  | "for_all" -> Some For_all
  | "for_all2_exn" -> Some For_all2
  | "is_empty" -> Some Is_empty
  | "exists" -> Some Not_exists
  | _ -> None

let quantifier_members = [ "for_all"; "for_all2_exn"; "is_empty"; "exists" ]

let collection_quantifier path =
  match List.rev path with
  | member :: container :: _ when String.equal container "List" || String.equal container "Array" ->
      quantifier_of_member member
  | _ -> None

let claim_kind_of_path = function
  | "Verdict" :: _ as path -> (
      match List.last path with
      | Some "p" -> Some P
      | Some "pf" -> Some Pf
      | Some "pass_fail" -> Some Pass_fail
      | Some "claim" -> Some Claim
      | Some "claimf" -> Some Claimf
      | _ -> None)
  | _ -> None

let claim_kinds =
  [ ("p", P); ("pf", Pf); ("pass_fail", Pass_fail); ("claim", Claim); ("claimf", Claimf) ]

let unlabelled arguments =
  List.filter_map arguments ~f:(function Asttypes.Nolabel, a -> Some a | _ -> None)

let literal_bool expr =
  match expr.pexp_desc with
  | Pexp_construct ({ txt = Longident.Lident "true"; _ }, None) -> Some true
  | Pexp_construct ({ txt = Longident.Lident "false"; _ }, None) -> Some false
  | _ -> None

let int_literal expr =
  match expr.pexp_desc with
  | Pexp_constant (Pconst_integer (value, _)) -> Option.try_with (fun () -> Int.of_string value)
  | _ -> None

let rec bool_pattern_matches pattern value =
  match pattern.ppat_desc with
  | Ppat_construct ({ txt = Longident.Lident found; _ }, None) ->
      String.equal found (Bool.to_string value)
  | Ppat_any | Ppat_var _ -> true
  | Ppat_alias (inner, _) | Ppat_constraint (inner, _) | Ppat_open (_, inner) ->
      bool_pattern_matches inner value
  | Ppat_or (left, right) -> bool_pattern_matches left value || bool_pattern_matches right value
  | _ -> false

let is_boolean_comparison callee =
  is_name callee "=" || is_name callee "<>" || is_name callee "equal" || is_name callee "!="
  || is_name callee ">" || is_name callee ">=" || is_name callee "<" || is_name callee "<="

let is_transparent_boolean_wrapper callee =
  match Read.longident_of callee with
  | Some ([ "Fn"; "id" ] | [ "Fun"; "id" ] | [ "Stdlib"; "Fun"; "id" ]) -> true
  | _ -> false

(* Pipeline and application operators are rewritten to the plain application they stand for, so `xs
   |> List.for_all ~f`, `x |> not`, `x |> Bool.equal true` and `not @@ x` need no case of their own
   anywhere below. *)
let rec normalize ~rebound expr =
  let operator name callee = is_name callee name && not (rebound callee) in
  match expr.pexp_desc with
  | Pexp_apply (pipe, [ (Asttypes.Nolabel, value); (Asttypes.Nolabel, function_) ])
    when operator "|>" pipe ->
      applied ~rebound function_ value ~loc:expr.pexp_loc
  | Pexp_apply (apply, [ (Asttypes.Nolabel, function_); (Asttypes.Nolabel, value) ])
    when operator "@@" apply ->
      applied ~rebound function_ value ~loc:expr.pexp_loc
  | _ -> expr

and applied ~rebound function_ value ~loc =
  let function_ = normalize ~rebound function_ in
  match function_.pexp_desc with
  | Pexp_apply (callee, arguments) ->
      {
        function_ with
        pexp_desc = Pexp_apply (callee, arguments @ [ (Asttypes.Nolabel, value) ]);
        pexp_loc = loc;
      }
  | _ ->
      {
        function_ with
        pexp_desc = Pexp_apply (function_, [ (Asttypes.Nolabel, value) ]);
        pexp_loc = loc;
      }

let pattern_names pattern =
  let names = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with
        | Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ }) ->
            names := (txt, pattern.ppat_loc) :: !names
        | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  List.rev !names
  |> List.dedup_and_sort ~compare:(fun (left, _) (right, _) -> String.compare left right)

type binding_part = {
  part_name : string;
  part_expression : expression;
  part_location : Ppxlib.Location.t;
  exact : bool;
}

let conservative_binding_parts pattern expression =
  pattern_names pattern
  |> List.map ~f:(fun (part_name, part_location) ->
      { part_name; part_expression = expression; part_location; exact = false })

(** A destructuring pattern is not permission to lose the binding. Literal tuples and records give
    each name its exact producer; for a shape that cannot be aligned, every bound name retains the
    whole expression as a conservative producer. *)
let rec binding_parts pattern expression =
  match (pattern.ppat_desc, expression.pexp_desc) with
  | Ppat_var { txt; _ }, _ ->
      [
        {
          part_name = txt;
          part_expression = expression;
          part_location = pattern.ppat_loc;
          exact = true;
        };
      ]
  | Ppat_alias (inner, { txt; _ }), _ ->
      {
        part_name = txt;
        part_expression = expression;
        part_location = pattern.ppat_loc;
        exact = true;
      }
      :: binding_parts inner expression
  | Ppat_constraint (inner, _), _ -> binding_parts inner expression
  | Ppat_variant (pattern_tag, Some inner), Pexp_variant (expression_tag, Some payload)
    when String.equal pattern_tag expression_tag ->
      binding_parts inner payload
  | ( Ppat_construct ({ txt = pattern_constructor; _ }, Some (_, inner)),
      Pexp_construct ({ txt = expression_constructor; _ }, Some payload) )
    when Poly.equal pattern_constructor expression_constructor ->
      binding_parts inner payload
  | Ppat_tuple patterns, Pexp_tuple expressions when List.length patterns = List.length expressions
    ->
      List.map2_exn patterns expressions ~f:binding_parts |> List.concat
  | Ppat_record (patterns, _), Pexp_record (expressions, None) ->
      List.map patterns ~f:(fun (pattern_label, pattern) ->
          List.find_map expressions ~f:(fun (expression_label, expression) ->
              if Poly.equal pattern_label.txt expression_label.txt then
                Some (binding_parts pattern expression)
              else None))
      |> Option.all
      |> Option.value_map ~default:(conservative_binding_parts pattern expression) ~f:List.concat
  | _ -> conservative_binding_parts pattern expression

(* ---------------------------------------------------------------------------------------------- *)
(* The environment. *)

type context = {
  env : binding list;  (** Most recent first; a dotted name is a module member. *)
  pending : claim list ref;
      (** Claims fired inside the function being analysed that still mention a parameter; the
          function's closure collects those that mention its own. *)
  found : claim list ref;  (** Claims whose value no longer mentions any parameter. *)
}

let lookup env name = List.find env ~f:(fun binding -> String.equal binding.name name)

let make_binding ?(span = (0, 0)) ?(final = false) ~name ~site payload =
  { name; site; span; final; payload }

let span_of_location (location : Ppxlib.Location.t) =
  (location.loc_start.Stdlib.Lexing.pos_cnum, location.loc_end.Stdlib.Lexing.pos_cnum)

let resolve_binding binding =
  match binding.payload with
  | Resolved provenance -> provenance
  | Resolving -> nothing
  | Pending compute ->
      binding.payload <- Resolving;
      let provenance = compute () in
      binding.payload <- Resolved provenance;
      provenance

(* Quantifiers nobody owns yet, or owned by a local written inside this binding's expression, become
   this binding's when it is referenced. *)
let rec own binding provenance =
  let start, stop = binding.span in
  let claims source =
    match (source.origin, source.owner) with
    | Quantifier _, None -> true
    | Quantifier _, Some owner ->
        (not owner.final) && start <= owner.site.position && owner.site.position < stop
    | (Parameter _ | Steering _), _ -> false
  in
  let rec own_sources sources =
    List.map sources ~f:(fun source ->
        match source.origin with
        | Steering { sources = steered; parameter; positive } ->
            { source with origin = Steering { sources = own_sources steered; parameter; positive } }
        | Quantifier _ | Parameter _ ->
            if claims source then { source with owner = Some binding } else source)
  in
  let own_view view = { view with sources = own_sources view.sources } in
  {
    when_true = own_view provenance.when_true;
    when_false = own_view provenance.when_false;
    constant = provenance.constant;
    closure =
      Option.map provenance.closure
        ~f:
          (map_functions ~f:(fun closure ->
               {
                 closure with
                 body = own binding closure.body;
                 claims =
                   List.map closure.claims ~f:(fun claim ->
                       { claim with value = own_view claim.value });
               }));
  }

let parameter_provenance binding =
  let source positive =
    {
      origin = Parameter { parameter = binding; positive };
      owner = None;
      written = binding.site;
      via = no_populations;
    }
  in
  {
    when_true = { sources = [ source true ]; witnesses = no_populations };
    when_false = { sources = [ source false ]; witnesses = no_populations };
    constant = None;
    closure = None;
  }

let parameter_binding ~name ~site =
  let binding = make_binding ~final:true ~name ~site (Resolved nothing) in
  binding.payload <- Resolved (parameter_provenance binding);
  binding

let tag_via label provenance =
  map_sources provenance ~f:(fun source -> { source with via = Set.add source.via label })

(* [Verdict.p label value], and the other two-slot claim forms: a closure whose one claim is its
   value slot, labelled by its label slot. *)
let native_claim kind ~site =
  let slot name = parameter_binding ~name ~site in
  let parameter name =
    let slot = slot name in
    { slot; label = Asttypes.Nolabel; patterns = []; names = [] }
  in
  match kind with
  | P | Pass_fail | Claim ->
      let label = parameter "label" and value = parameter "value" in
      let claim =
        {
          fired = site;
          label = Label_slot label.slot;
          value = (parameter_provenance value.slot).when_true;
          helper = None;
        }
      in
      Function
        {
          parameters = [ label; value ];
          claims = [ claim ];
          body = nothing;
          native = true;
          format = false;
          loose = false;
        }
  | Pf | Claimf ->
      Function
        {
          parameters = [ parameter "format" ];
          claims = [];
          body = nothing;
          native = true;
          format = true;
          loose = false;
        }

(* The closure a format literal turns [Verdict.pf fmt] into: one slot per argument the format
   consumes, then the value slot. A `%a` takes two, and a `*` width or precision one more each. *)
let format_claim ~site format_expression =
  let slot name =
    { slot = parameter_binding ~name ~site; label = Asttypes.Nolabel; patterns = []; names = [] }
  in
  let value = slot "value" in
  let claim =
    {
      fired = site;
      label = Label_expr format_expression;
      value = (parameter_provenance value.slot).when_true;
      helper = None;
    }
  in
  match Read.string_literal format_expression with
  | Some format ->
      let consumed =
        Scan.directives format
        |> List.sum
             (module Int)
             ~f:(fun (directive : Scan.directive) ->
               if Scan.consumes_nothing directive.conversion then 0
               else
                 let modifiers =
                   String.sub format ~pos:(directive.start + 1)
                     ~len:(directive.stop - directive.start - 1)
                 in
                 1
                 + String.count modifiers ~f:(Char.equal '*')
                 + if Char.equal directive.conversion 'a' then 1 else 0)
      in
      let arguments = List.init consumed ~f:(fun i -> slot (Printf.sprintf "argument%d" i)) in
      {
        parameters = arguments @ [ value ];
        claims = [ claim ];
        body = nothing;
        native = true;
        format = false;
        loose = false;
      }
  | None ->
      {
        parameters = [ value ];
        claims = [ claim ];
        body = nothing;
        native = true;
        format = false;
        loose = true;
      }

let quantifier_function kind =
  {
    nothing with
    closure = Some (Quantifier_function { kind; populations = []; predicate = false });
  }

let quantifier kind populations ~written =
  let source =
    { origin = Quantifier { kind; populations }; owner = None; written; via = no_populations }
  in
  let vacuous = { sources = [ source ]; witnesses = no_populations } in
  let witnessed = { sources = []; witnesses = populations } in
  match kind with
  | For_all | For_all2 | Is_empty ->
      { when_true = vacuous; when_false = witnessed; constant = None; closure = None }
  | Not_exists -> { when_true = witnessed; when_false = vacuous; constant = None; closure = None }

(* What `open Verdict.Claims`, `open List` and `module L = List` bring into scope: the native
   closures under their unqualified names. *)
let claim_exports ~site =
  List.map claim_kinds ~f:(fun (name, kind) ->
      make_binding ~name ~site (Resolved { nothing with closure = Some (native_claim kind ~site) }))

let quantifier_exports ~site =
  List.filter_map quantifier_members ~f:(fun name ->
      Option.map (quantifier_of_member name) ~f:(fun kind ->
          make_binding ~name ~site (Resolved (quantifier_function kind))))

let prefix_bindings prefix bindings =
  List.map bindings ~f:(fun binding -> { binding with name = prefix ^ "." ^ binding.name })

let module_path module_expr =
  match module_expr.pmod_desc with
  | Pmod_ident { txt; _ } -> Option.try_with (fun () -> Longident.flatten_exn txt)
  | _ -> None

(* A parameter's population is marked, so that a quantifier over it is known to wait for the actual
   argument: [name@P<position>], against [name@<position>] for any other binder. *)
let scope_string binding = (if binding.final then "P" else "") ^ Int.to_string binding.site.position

(* A population key is the expression's text and, behind a separator no printed source contains, the
   scope of every name it mentions: [name=P<position>] for a parameter, [name=<position>] for any
   other binder, [name=0] for a free name. Two spellings are one population only when every name in
   them resolves to the same binder, so a predicate rebound between a witness and a quantifier tells
   the two filtered views apart. *)
let population_separator = '\001'

let population_key ~text entries =
  text ^ String.of_char population_separator ^ String.concat ~sep:";" entries

let population_scopes key =
  match String.rsplit2 key ~on:population_separator with
  | Some (_, scopes) -> String.split scopes ~on:';'
  | None -> []

let population_text key =
  Option.value_map (String.rsplit2 key ~on:population_separator) ~default:key ~f:fst

let scope_entry binding = binding.name ^ "=" ^ scope_string binding
let parameter_scope entry = String.is_substring entry ~substring:"=P"
let parameter_population key = List.exists (population_scopes key) ~f:parameter_scope
let population_of_binding binding = population_key ~text:binding.name [ scope_entry binding ]

(* Population identity: the name a quantifier ranges over, resolved to the binder that introduced
   it. A filtered view is its own population, still anchored to its source's binder: collapsing it
   to the source lets a non-empty [filter rows ~f:p1] guard a distinct, empty [filter rows
   ~f:p2]. *)
let rebound_in env callee =
  Option.exists (Read.longident_of callee) ~f:(fun path ->
      Option.is_some (lookup env (String.concat ~sep:"." path)))

let rec population ctx expr =
  let expr = normalize ~rebound:(rebound_in ctx.env) expr in
  let scope_of name =
    Option.value_map (lookup ctx.env name) ~default:(name ^ "=0") ~f:scope_entry
  in
  match expr.pexp_desc with
  | Pexp_ident { txt = Longident.Lident name; _ } ->
      Some (population_key ~text:name [ scope_of name ])
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> population ctx inner
  | Pexp_apply (callee, _)
    when is_collection_call callee ~member:"filter"
         || is_collection_call callee ~member:"filter_map" ->
      let text = Stdlib.Format.asprintf "%a" Pprintast.expression expr in
      let names = ref [] in
      let iterator =
        object
          inherit Ast_traverse.iter as super

          method! expression child =
            (match child.pexp_desc with
            | Pexp_ident { txt = Longident.Lident name; _ } -> names := name :: !names
            | _ -> ());
            super#expression child
        end
      in
      iterator#expression expr;
      let scopes = List.dedup_and_sort !names ~compare:String.compare |> List.map ~f:scope_of in
      Some (population_key ~text scopes)
  | _ -> None

let length_population ctx expr =
  match (normalize ~rebound:(rebound_in ctx.env) expr).pexp_desc with
  | Pexp_apply (callee, arguments) when is_collection_call callee ~member:"length" ->
      List.hd (unlabelled arguments) |> Option.bind ~f:(population ctx)
  | _ -> None

(* ---------------------------------------------------------------------------------------------- *)
(* Substitution: applying a closure to its arguments. *)

type argument = Supplied of expression | Unknown of expression | Absent

(* Every name a pattern binds, with the component of the actual argument it receives. *)
let projected_arguments parameter (argument : expression) =
  List.concat_map parameter.patterns ~f:(fun pattern -> binding_parts pattern argument)

(* Whether a source waits on a parameter: a parameter's own value, a steering decision on it, or a
   quantifier over a parameter population. [among] restricts the question to given parameters. *)
let waits_on_parameter ?among source =
  let binding_counts parameter =
    Option.value_map among ~default:true ~f:(List.exists ~f:(phys_equal parameter))
  in
  let population_counts key =
    parameter_population key
    && Option.value_map among ~default:true ~f:(fun mine ->
        let scopes = population_scopes key in
        List.exists mine ~f:(fun parameter ->
            List.mem scopes (scope_entry parameter) ~equal:String.equal))
  in
  match source.origin with
  | Parameter { parameter; _ } | Steering { parameter; _ } -> binding_counts parameter
  | Quantifier { populations; _ } -> Set.exists populations ~f:population_counts

let mentions_parameter view = List.exists view.sources ~f:(fun s -> waits_on_parameter s)

let rec substitute_sources ~(replacement : binding -> (site * provenance) option option)
    ~populations sources =
  let from_actual source argument_site actual_sources =
    List.map actual_sources ~f:(fun actual_source ->
        {
          actual_source with
          via = Set.union actual_source.via source.via;
          written =
            (if Option.is_none actual_source.owner then argument_site else actual_source.written);
        })
  in
  List.concat_map sources ~f:(fun source ->
      match source.origin with
      | Quantifier { kind; populations = named } ->
          let named =
            Set.map (module String) named ~f:(fun p -> Option.value (populations p) ~default:p)
          in
          [ { source with origin = Quantifier { kind; populations = named } } ]
      | Parameter { parameter; positive } -> (
          match replacement parameter with
          | None -> [ source ]
          | Some None -> []
          | Some (Some (argument_site, actual)) ->
              from_actual source argument_site (view_at actual positive).sources)
      | Steering { sources = steered; parameter; positive } -> (
          let steered = substitute_sources ~replacement ~populations steered in
          let steering parameter positive =
            { source with origin = Steering { sources = steered; parameter; positive } }
          in
          match replacement parameter with
          | None -> [ steering parameter positive ]
          | Some None -> []
          | Some (Some (argument_site, actual)) -> (
              (* The branch's value is the constant the steering selects exactly when the actual
                 argument is that constant; an actual that is itself a parameter defers again. *)
              match actual.constant with
              | Some constant when Bool.equal constant positive ->
                  from_actual source argument_site steered
              | Some _ -> []
              | None ->
                  List.filter_map (view_at actual positive).sources ~f:(fun actual_source ->
                      match actual_source.origin with
                      | Parameter { parameter; positive } -> Some (steering parameter positive)
                      | Quantifier _ | Steering _ -> None))))

let substitute_view ~replacement ~populations view =
  let sources = substitute_sources ~replacement ~populations view.sources in
  (* A view that is exactly one parameter IS the actual argument, witnesses included: a native
     claim's value slot, a wrapper forwarding a parameter. A parameter beside other sources keeps
     only its own witnesses, since the value may have rested on the others. *)
  let inherited =
    match view.sources with
    | [ { origin = Parameter { parameter; positive }; _ } ] -> (
        match replacement parameter with
        | Some (Some (_, actual)) -> (view_at actual positive).witnesses
        | Some None | None -> no_populations)
    | _ -> no_populations
  in
  let witnesses =
    Set.map
      (module String)
      (Set.union view.witnesses inherited)
      ~f:(fun p -> Option.value (populations p) ~default:p)
  in
  settle { sources; witnesses }

let rec substitute ~replacement ~populations provenance =
  (* A quantifier function that captured a population from a parameter has it substituted too, so
     its later completion ranges over the actual. *)
  let rec closure = function
    | Quantifier_function partial ->
        Quantifier_function
          {
            partial with
            populations =
              List.map partial.populations
                ~f:(Option.map ~f:(fun key -> Option.value (populations key) ~default:key));
          }
    | Function function_closure ->
        Function
          {
            function_closure with
            body = substitute ~replacement ~populations function_closure.body;
            claims =
              List.map function_closure.claims ~f:(substitute_claim ~replacement ~populations);
          }
    | Alternatives closures -> Alternatives (List.map closures ~f:closure)
  in
  {
    when_true = substitute_view ~replacement ~populations provenance.when_true;
    when_false = substitute_view ~replacement ~populations provenance.when_false;
    constant = provenance.constant;
    closure = Option.map provenance.closure ~f:closure;
  }

and substitute_claim ~replacement ~populations claim =
  { claim with value = substitute_view ~replacement ~populations claim.value }

let drop_via labels provenance =
  if Set.is_empty labels then provenance
  else
    let keep source = Set.is_empty (Set.inter source.via labels) in
    let rec drop_sources sources =
      List.filter_map sources ~f:(fun source ->
          if not (keep source) then None
          else
            match source.origin with
            | Steering { sources = steered; parameter; positive } -> (
                match drop_sources steered with
                | [] -> None
                | steered ->
                    Some
                      { source with origin = Steering { sources = steered; parameter; positive } })
            | Quantifier _ | Parameter _ -> Some source)
    in
    let drop_view view = { view with sources = drop_sources view.sources } in
    let rec drop provenance =
      {
        when_true = drop_view provenance.when_true;
        when_false = drop_view provenance.when_false;
        constant = provenance.constant;
        closure =
          Option.map provenance.closure
            ~f:
              (map_functions ~f:(fun closure ->
                   {
                     closure with
                     body = drop closure.body;
                     claims =
                       List.map closure.claims ~f:(fun claim ->
                           { claim with value = drop_view claim.value });
                   }));
      }
    in
    drop provenance

(* ---------------------------------------------------------------------------------------------- *)
(* The walk. *)

let claim_key claim =
  Printf.sprintf "%d:%s:%s" claim.fired.position
    (Option.value claim.helper ~default:"")
    (String.concat ~sep:";" (List.map claim.value.sources ~f:source_key))

let emit ctx ~site ~helper claim =
  let claim = { claim with fired = site; helper = Option.first_some helper claim.helper } in
  if mentions_parameter claim.value then ctx.pending := claim :: !(ctx.pending)
  else ctx.found := claim :: !(ctx.found)

let rec walk ctx expr =
  let expr = normalize ~rebound:(rebound_in ctx.env) expr in
  match expr.pexp_desc with
  | Pexp_ident _ -> resolve ctx expr
  | Pexp_construct ({ txt = Longident.Lident ("true" | "false"); _ }, None) ->
      constant (Option.value_exn (literal_bool expr))
  | Pexp_construct ({ txt = Longident.Lident "Some"; _ }, Some payload) -> walk ctx payload
  | Pexp_variant (_, Some payload) -> walk ctx payload
  | Pexp_construct (_, Some payload) ->
      (* [Ok b], [`Tag b], [Some b]: the payload's provenance, which a match will read out. *)
      walk ctx payload
  | Pexp_construct (_, None) | Pexp_constant _ -> nothing
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> walk ctx inner
  | Pexp_field (record, _) ->
      (* A projection reads one component the aggregate did not keep apart: conservatively the whole
         record's sources, as [fst]/[snd] below and a destructuring of a bound aggregate. *)
      walk ctx record
  | Pexp_let (recursive, bindings, body) -> walk { ctx with env = bind ctx recursive bindings } body
  | Pexp_sequence (setup, result) ->
      discard ctx setup;
      walk ctx result
  | Pexp_open (declaration, body) ->
      walk { ctx with env = module_exports ctx declaration.popen_expr @ ctx.env } body
  | Pexp_letmodule ({ txt = Some name; _ }, module_expr, body) ->
      let exports = module_exports ctx module_expr |> prefix_bindings name in
      walk { ctx with env = exports @ ctx.env } body
  | Pexp_letmodule ({ txt = None; _ }, module_expr, body) ->
      ignore (module_exports ctx module_expr : binding list);
      walk ctx body
  | Pexp_letexception (_, body) -> walk ctx body
  | Pexp_tuple items -> aggregate (List.map items ~f:(walk ctx))
  | Pexp_record (fields, base) ->
      aggregate
        (List.map fields ~f:(fun (_, field) -> walk ctx field)
        @ Option.to_list (Option.map base ~f:(walk ctx)))
  | Pexp_ifthenelse (condition, yes, no) -> (
      let condition = walk ctx condition in
      let yes = walk ctx yes in
      match no with
      | None -> nothing
      | Some no ->
          let no = walk ctx no in
          alternatives
            [
              { steering = [ condition ]; outcome = yes };
              { steering = [ negate condition ]; outcome = no };
            ])
  | Pexp_match (scrutinee_expr, cases) ->
      let scrutinee = walk ctx scrutinee_expr in
      alternatives (case_alternatives ctx ~scrutinee:(Some (scrutinee_expr, scrutinee)) cases)
  | Pexp_try (body, cases) ->
      let body = walk ctx body in
      alternatives ({ steering = []; outcome = body } :: case_alternatives ctx ~scrutinee:None cases)
  | Pexp_function (parameters, _, body) -> function_closure ctx parameters body
  | Pexp_apply (callee, arguments) -> application ctx expr callee arguments
  | _ ->
      fallback ctx expr;
      nothing

and discard ctx expr = ignore (walk ctx expr : provenance)

(* Sub-expressions of a form this reader has no value rule for are still read for the claims they
   fire, in the environment in effect here. *)
and fallback ctx expr =
  let iterator =
    object
      inherit Ast_traverse.iter as super
      method! attribute _ = ()
      method! expression child = discard ctx child
      method! structure items = ignore (scan_structure ctx items : binding list * binding list)
      method children = super#expression
    end
  in
  iterator#children expr

and resolve ctx expr =
  match Read.longident_of expr with
  | None -> nothing
  | Some path -> (
      match lookup ctx.env (String.concat ~sep:"." path) with
      | Some binding -> own binding (resolve_binding binding)
      | None -> (
          match claim_kind_of_path path with
          | Some kind ->
              {
                nothing with
                closure = Some (native_claim kind ~site:(site_of_location expr.pexp_loc));
              }
          | None -> (
              match collection_quantifier path with
              | Some kind -> quantifier_function kind
              | None -> nothing)))

(* The bindings a [let] adds, each walked once, in source order so that claims fire in source order.
   A recursive group is recomputed once per bound name with the previous round's siblings in scope,
   so a dependency can cross the longest possible sibling chain. *)
and new_bindings ctx recursive bindings =
  let parts =
    List.concat_map bindings ~f:(fun binding -> binding_parts binding.pvb_pat binding.pvb_expr)
  in
  (* `let () = …` and `let _ = …` bind no name, and are read for the claims they fire. *)
  List.iter bindings ~f:(fun binding ->
      if List.is_empty (pattern_names binding.pvb_pat) then discard ctx binding.pvb_expr);
  (* [env] is read when a body is walked, not when the binding is made, so a recursive group can put
     its own members in scope before any of them is forced. *)
  let make env =
    List.concat_map bindings ~f:(fun binding ->
        binding_parts binding.pvb_pat binding.pvb_expr
        |> List.map ~f:(fun part ->
            make_binding
              ~span:(span_of_location binding.pvb_expr.pexp_loc)
              ~name:part.part_name
              ~site:(site_of_location part.part_location)
              (Pending (fun () -> walk { ctx with env = env () } part.part_expression))))
  in
  let force bindings =
    List.iter bindings ~f:(fun binding -> ignore (resolve_binding binding : provenance))
  in
  match recursive with
  | Asttypes.Nonrecursive ->
      let added = make (fun () -> ctx.env) in
      force added;
      added
  | Recursive ->
      (* Each round sees the previous round's siblings resolved and its own siblings pending, so a
         forward reference resolves on demand and only a cycle needs the second round. *)
      let rounds = Int.min 2 (Int.max 1 (List.length parts)) in
      let rec close remaining previous =
        if remaining = 0 then previous
        else
          let own = ref [] in
          let added = make (fun () -> List.rev_append !own (List.rev_append previous ctx.env)) in
          own := added;
          force added;
          close (remaining - 1) added
      in
      close rounds []

and bind ctx recursive bindings = List.rev_append (new_bindings ctx recursive bindings) ctx.env

(* A case's pattern binds its names to the scrutinee's components, so `match q with ok -> ok`
   returns [q] and a tuple pattern over a tuple scrutinee returns the matching component. *)
and bind_pattern ctx pattern ~producer =
  let bindings =
    match producer with
    | Some (expr, provenance) ->
        binding_parts pattern expr
        |> List.map ~f:(fun part ->
            let payload =
              if phys_equal part.part_expression expr then Resolved provenance
              else Pending (fun () -> walk ctx part.part_expression)
            in
            make_binding ~name:part.part_name ~site:(site_of_location part.part_location) payload)
    | None ->
        pattern_names pattern
        |> List.map ~f:(fun (name, location) ->
            make_binding ~name ~site:(site_of_location location) (Resolved nothing))
  in
  List.iter bindings ~f:(fun binding -> ignore (resolve_binding binding : provenance));
  List.rev_append bindings ctx.env

and case_alternatives ctx ~scrutinee cases =
  let selections = boolean_selections cases in
  List.map2_exn cases selections ~f:(fun case (on_true, on_false) ->
      let env = bind_pattern ctx case.pc_lhs ~producer:scrutinee in
      let case_ctx = { ctx with env } in
      let guard = Option.map case.pc_guard ~f:(walk case_ctx) in
      let outcome = walk case_ctx case.pc_rhs in
      let selection =
        match scrutinee with
        | Some (_, scrutinee) -> boolean_selection scrutinee (on_true, on_false)
        | None -> []
      in
      { steering = Option.to_list guard @ selection; outcome })

(* Which Boolean values of the scrutinee can reach each case: the values its pattern admits that no
   earlier UNGUARDED case has already taken -- a guarded case takes nothing away, since its guard
   may fail. A case reachable on one value alone is selected by the scrutinee at that polarity,
   whatever its guard adds. *)
and boolean_selections cases =
  let remaining_true = ref true and remaining_false = ref true in
  List.map cases ~f:(fun case ->
      let on_true = !remaining_true && bool_pattern_matches case.pc_lhs true in
      let on_false = !remaining_false && bool_pattern_matches case.pc_lhs false in
      if Option.is_none case.pc_guard then (
        if on_true then remaining_true := false;
        if on_false then remaining_false := false);
      (on_true, on_false))

and boolean_selection scrutinee = function
  | true, false -> [ scrutinee ]
  | false, true -> [ negate scrutinee ]
  | true, true | false, false -> []

(* A function's parameters are bindings whose value is the parameter itself; an optional default is
   walked in the environment before its parameter shadows anything, and joins the parameter's value
   tagged with the label, so that a call supplying the label drops it. The body's claims that
   mention a parameter of this function become the closure's; those mentioning only an enclosing
   function's parameters are handed up to it. *)
and function_closure ctx parameters body =
  let pending = ref [] in
  let env, params =
    List.fold parameters ~init:(ctx.env, []) ~f:(fun (env, params) parameter ->
        match parameter.pparam_desc with
        | Pparam_newtype _ -> (env, params)
        | Pparam_val (label, default, pattern) ->
            let pre_ctx = { ctx with env; pending } in
            let default = Option.map default ~f:(walk pre_ctx) in
            let site = site_of_location pattern.ppat_loc in
            let slot = parameter_binding ~name:"" ~site in
            let names =
              pattern_names pattern
              |> List.map ~f:(fun (name, location) ->
                  parameter_binding ~name ~site:(site_of_location location))
            in
            (match (label, default) with
            | (Asttypes.Optional label_name | Labelled label_name), Some default ->
                List.iter names ~f:(fun binding ->
                    let default = tag_via label_name (own binding default) in
                    let supplied = parameter_provenance binding in
                    binding.payload <-
                      Resolved
                        {
                          when_true = either supplied.when_true default.when_true;
                          when_false = either supplied.when_false default.when_false;
                          constant = None;
                          closure = default.closure;
                        })
            | _ -> ());
            let param = { slot; label; patterns = [ pattern ]; names } in
            (List.rev_append names env, param :: params))
  in
  let params = List.rev params in
  let inner_ctx = { ctx with env; pending } in
  let params, body =
    match body with
    | Pfunction_body body -> (params, walk inner_ctx body)
    | Pfunction_cases (cases, loc, _) ->
        let site = site_of_location loc in
        let slot = parameter_binding ~name:"" ~site in
        let scrutinee = parameter_provenance slot in
        let names = ref [] in
        let alternatives_ =
          let selections = boolean_selections cases in
          List.map2_exn cases selections ~f:(fun case selected ->
              let case_names =
                pattern_names case.pc_lhs
                |> List.map ~f:(fun (name, location) ->
                    parameter_binding ~name ~site:(site_of_location location))
              in
              names := !names @ case_names;
              let case_ctx = { inner_ctx with env = List.rev_append case_names env } in
              let guard = Option.map case.pc_guard ~f:(walk case_ctx) in
              let outcome = walk case_ctx case.pc_rhs in
              let selection = boolean_selection scrutinee selected in
              { steering = Option.to_list guard @ selection; outcome })
        in
        let param =
          {
            slot;
            label = Asttypes.Nolabel;
            patterns = List.map cases ~f:(fun c -> c.pc_lhs);
            names = !names;
          }
        in
        (params @ [ param ], alternatives alternatives_)
  in
  let mine = List.concat_map params ~f:(fun p -> p.slot :: p.names) in
  let mentions_mine claim = List.exists claim.value.sources ~f:(waits_on_parameter ~among:mine) in
  let own_claims, outer = List.partition_tf !pending ~f:mentions_mine in
  ctx.pending := outer @ !(ctx.pending);
  {
    nothing with
    closure =
      Some
        (Function
           {
             parameters = params;
             claims = List.rev own_claims;
             body;
             native = false;
             format = false;
             loose = false;
           });
  }

and application ctx expr callee arguments =
  let site = site_of_location expr.pexp_loc in
  (* A builtin is read as itself only where nothing in scope has rebound its name. *)
  let builtin name = is_name callee name && not (rebound_in ctx.env callee) in
  match arguments with
  | [ (Asttypes.Nolabel, argument) ] when builtin "not" -> negate (walk ctx argument)
  | [ (Asttypes.Nolabel, argument) ] when is_transparent_boolean_wrapper callee && builtin "id" ->
      walk ctx argument
  | [ (Asttypes.Nolabel, argument) ] when builtin "fst" || builtin "snd" -> walk ctx argument
  | [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ] when builtin "&&" ->
      let left = walk ctx left in
      let right = walk ctx right in
      conjunction left right
  | [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ] when builtin "||" ->
      let left = walk ctx left in
      let right = walk ctx right in
      disjunction left right
  | [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ]
    when is_boolean_comparison callee
         && List.exists [ "="; "<>"; "equal"; "!="; ">"; ">="; "<"; "<=" ] ~f:builtin ->
      comparison ctx callee left right
  | _ -> (
      let function_ = walk ctx callee in
      match function_.closure with
      | Some closure ->
          let callee_name =
            Option.bind (Read.longident_of callee) ~f:(fun path ->
                let name = String.concat ~sep:"." path in
                Option.map (lookup ctx.env name) ~f:(fun _ -> name))
          in
          apply ctx ~site ~callee_name closure arguments
      | None ->
          List.iter arguments ~f:(fun (_, argument) -> discard ctx argument);
          nothing)

(* [x = true], [Bool.equal x false], [x <> true]: the other operand, in the polarity the constant
   selects. Otherwise a comparison is a value of its own -- except a literal length bound, which is
   a witness: [List.length xs > 0] and [Array.length got = 4] establish the population. *)
and comparison ctx callee left right =
  let equality = is_name callee "=" || is_name callee "equal" in
  let inequality = is_name callee "<>" || is_name callee "!=" in
  let left_provenance = walk ctx left in
  let right_provenance = walk ctx right in
  let select constant other =
    if equality then if constant then other else negate other
    else if inequality then if constant then negate other else other
    else nothing
  in
  match (left_provenance.constant, right_provenance.constant) with
  | Some constant, None when equality || inequality -> select constant right_provenance
  | None, Some constant when equality || inequality -> select constant left_provenance
  | Some a, Some b when equality || inequality -> constant (Bool.equal (Bool.equal a b) equality)
  | _ -> (
      let witness population =
        {
          nothing with
          when_true = { sources = []; witnesses = Set.singleton (module String) population };
        }
      in
      let gt = is_name callee ">"
      and ge = is_name callee ">="
      and lt = is_name callee "<"
      and le = is_name callee "<=" in
      match
        ( length_population ctx left,
          int_literal right,
          int_literal left,
          length_population ctx right )
      with
      | Some population, Some n, _, _
        when (equality && n > 0) || (inequality && n = 0) || (gt && n >= 0) || (ge && n > 0) ->
          witness population
      | _, _, Some n, Some population
        when (equality && n > 0) || (inequality && n = 0) || (lt && n >= 0) || (le && n > 0) ->
          witness population
      | _ -> nothing)

and apply ctx ~site ~callee_name closure arguments =
  match closure with
  | Alternatives closures ->
      (* Applied to every function it could be; what comes back is any of the results. *)
      let results =
        List.map closures ~f:(fun closure -> apply ctx ~site ~callee_name closure arguments)
      in
      {
        (aggregate results) with
        closure = select_closures (List.filter_map results ~f:(fun result -> result.closure));
      }
  | Quantifier_function partial ->
      (* A quantifier is a value once every population and, but for [is_empty], its predicate have
         arrived; until then it stays the function it is, with what it has received. *)
      List.iter arguments ~f:(fun (_, argument) -> discard ctx argument);
      let populations = partial.populations @ List.map (unlabelled arguments) ~f:(population ctx) in
      let predicate =
        partial.predicate
        || List.exists arguments ~f:(fun (label, _) ->
            match label with Asttypes.Labelled "f" | Optional "f" -> true | _ -> false)
      in
      let complete =
        List.length populations >= quantifier_arity partial.kind
        && (predicate || match partial.kind with Is_empty -> true | _ -> false)
      in
      if complete then
        quantifier partial.kind
          (List.take populations (quantifier_arity partial.kind)
          |> List.filter_opt
          |> Set.of_list (module String))
          ~written:site
      else
        {
          nothing with
          closure = Some (Quantifier_function { partial with populations; predicate });
        }
  | Function closure when closure.format -> (
      (* The format literal, once supplied, says how many arguments precede the value. *)
      match
        List.find arguments ~f:(fun (label, _) ->
            match label with Asttypes.Nolabel -> true | _ -> false)
      with
      | None -> { nothing with closure = Some (Function closure) }
      | Some (_, format) ->
          discard ctx format;
          let expanded = format_claim ~site format in
          let rest =
            List.filter arguments ~f:(fun (label, argument) ->
                not
                  (phys_equal argument format
                  && match label with Asttypes.Nolabel -> true | _ -> false))
          in
          if List.is_empty rest then
            { nothing with closure = Some (Function { expanded with native = false }) }
          else apply ctx ~site ~callee_name (Function expanded) rest)
  | Function closure when closure.loose -> (
      List.iter arguments ~f:(fun (_, argument) -> discard ctx argument);
      match List.last (unlabelled arguments) with
      | None -> { nothing with closure = Some (Function closure) }
      | Some value ->
          let slot = (List.hd_exn closure.parameters).slot in
          let actual = walk ctx value in
          let replacement parameter =
            if phys_equal parameter slot then Some (Some (site_of_location value.pexp_loc, actual))
            else None
          in
          let claims =
            List.map closure.claims ~f:(substitute_claim ~replacement ~populations:(fun _ -> None))
          in
          List.iter claims ~f:(fun claim ->
              emit ctx ~site
                ~helper:(if closure.native then None else callee_name)
                {
                  claim with
                  label = (match claim.label with Label_slot _ -> Label_expr value | l -> l);
                });
          (* How many arguments the format takes is unknown, so the closure stays: a partial
             application's later arguments are claimed the same way. *)
          { nothing with closure = Some (Function closure) })
  | Function closure ->
      let unlabelled_arguments = ref (unlabelled arguments) in
      let assignments =
        List.map closure.parameters ~f:(fun parameter ->
            match parameter.label with
            | Asttypes.Nolabel -> (
                match !unlabelled_arguments with
                | argument :: rest ->
                    unlabelled_arguments := rest;
                    (parameter, Supplied argument)
                | [] -> (parameter, Absent))
            | Labelled name | Optional name ->
                List.find_map arguments ~f:(fun (label, argument) ->
                    match label with
                    | Asttypes.Labelled found when String.equal found name ->
                        Some (Supplied argument)
                    | Optional found when String.equal found name -> (
                        (* `?name:(Some v)` supplies [v]; `?name:None` supplies nothing, as an
                           omission would; any other `?name:e` may be either, so the parameter is
                           both [e]'s payload and its default. *)
                        match argument.pexp_desc with
                        | Pexp_construct ({ txt = Longident.Lident "Some"; _ }, Some payload) ->
                            Some (Supplied payload)
                        | Pexp_construct ({ txt = Longident.Lident "None"; _ }, None) -> Some Absent
                        | _ -> Some (Unknown argument))
                    | _ -> None)
                |> Option.value_map ~default:(parameter, Absent) ~f:(fun a -> (parameter, a)))
      in
      let leftover = !unlabelled_arguments in
      (* An argument no parameter takes is still read for the claims it fires. *)
      let taken (label, argument) =
        match label with
        | Asttypes.Nolabel -> not (List.exists leftover ~f:(phys_equal argument))
        | Labelled name | Optional name ->
            List.exists assignments ~f:(fun (parameter, _) ->
                match parameter.label with
                | Asttypes.Labelled found | Optional found -> String.equal found name
                | Nolabel -> false)
      in
      List.iter arguments ~f:(fun argument ->
          if not (taken argument) then discard ctx (snd argument));
      let fires =
        List.for_all assignments ~f:(fun (parameter, argument) ->
            match (parameter.label, argument) with
            | Asttypes.Optional _, _ -> true
            | _, Absent -> false
            | _, (Supplied _ | Unknown _) -> true)
      in
      let supplied_labels =
        List.filter_map assignments ~f:(fun (parameter, argument) ->
            match (parameter.label, argument) with
            | (Asttypes.Labelled name | Optional name), Supplied _ -> Some name
            | _ -> None)
        |> Set.of_list (module String)
      in
      (* [None] leaves a parameter of an enclosing function alone; [Some None] removes an absent or
         defaulted parameter's references (its default, tagged, remains); [Some (Some _)] is the
         actual argument, or the component of it the name was destructured from. *)
      let table = ref [] in
      let populations = ref [] in
      List.iter assignments ~f:(fun (parameter, argument) ->
          let actual = match argument with Supplied e | Unknown e -> Some e | Absent -> None in
          let resolved_actual =
            Option.map actual ~f:(fun e -> (site_of_location e.pexp_loc, walk ctx e))
          in
          (* An absent parameter is left alone while the application is partial; once the closure
             fires, an absent optional is its default, whose tagged sources the parameter already
             carries. *)
          (match (argument, resolved_actual) with
          | (Supplied _ | Unknown _), Some (argument_site, provenance) ->
              table := (parameter.slot, Some (argument_site, provenance)) :: !table
          | _ -> if fires then table := (parameter.slot, None) :: !table);
          List.iter parameter.names ~f:(fun name ->
              match actual with
              | None -> if fires then table := (name, None) :: !table
              | Some actual -> (
                  let parts =
                    projected_arguments parameter actual
                    |> List.filter ~f:(fun part -> String.equal part.part_name name.name)
                  in
                  match parts with
                  | [] -> table := (name, None) :: !table
                  | parts ->
                      let provenances =
                        List.map parts ~f:(fun part ->
                            if phys_equal part.part_expression actual then
                              Option.value_exn resolved_actual |> snd
                            else walk ctx part.part_expression)
                      in
                      let combined =
                        List.reduce_exn provenances ~f:(fun a b ->
                            {
                              (aggregate [ a; b ]) with
                              constant =
                                (if Option.equal Bool.equal a.constant b.constant then a.constant
                                 else None);
                              closure = a.closure;
                            })
                      in
                      let argument_site =
                        site_of_location (List.hd_exn parts).part_expression.pexp_loc
                      in
                      table := (name, Some (argument_site, combined)) :: !table;
                      (* Only a component the pattern aligned exactly names the formal's population:
                         a bound aggregate hands every formal the whole argument, and one identity
                         for two formals would let one's witness cover the other. *)
                      List.iter parts ~f:(fun part ->
                          if part.exact then
                            Option.iter (population ctx part.part_expression) ~f:(fun key ->
                                populations := (population_of_binding name, key) :: !populations)))));
      let replacement parameter =
        List.find_map !table ~f:(fun (binding, actual) ->
            if phys_equal binding parameter then Some actual else None)
      in
      let population_of key = List.Assoc.find !populations key ~equal:String.equal in
      (* A parameter this application supplied is a parameter no longer: its population, where no
         component of the actual named one, is opaque from here on -- no witness can cover it, and
         no later call will substitute it. *)
      let sealed =
        List.filter_map assignments ~f:(fun (parameter, argument) ->
            match argument with
            | Supplied _ | Unknown _ -> Some parameter
            | Absent -> if fires then Some parameter else None)
        |> List.concat_map ~f:(fun parameter -> parameter.slot :: parameter.names)
        |> List.map ~f:scope_entry
      in
      (* The formal's own name becomes the actual's identity; a view mentioning the formal (a filter
         over it) keeps its text and is sealed, since no caller-side witness spells it. *)
      let population_of key =
        match population_of key with
        | Some mapped -> Some mapped
        | None ->
            let scopes = population_scopes key in
            if List.exists scopes ~f:(List.mem sealed ~equal:String.equal) then
              Some
                (population_key ~text:(population_text key)
                   (List.map scopes ~f:(fun entry ->
                        if List.mem sealed entry ~equal:String.equal then
                          String.substr_replace_first entry ~pattern:"=P" ~with_:"=applied"
                        else entry)))
            else None
      in
      let substitute_all provenance =
        substitute ~replacement ~populations:population_of provenance |> drop_via supplied_labels
      in
      let claims =
        List.map closure.claims ~f:(fun claim ->
            let claim = substitute_claim ~replacement ~populations:population_of claim in
            let claim =
              match claim.label with
              | Label_slot slot -> (
                  match
                    List.find_map assignments ~f:(fun (parameter, argument) ->
                        if phys_equal parameter.slot slot then
                          match argument with Supplied e | Unknown e -> Some e | Absent -> None
                        else None)
                  with
                  | Some label -> { claim with label = Label_expr label }
                  | None -> claim)
              | Label_expr _ -> claim
            in
            {
              claim with
              value = (drop_via supplied_labels { nothing with when_true = claim.value }).when_true;
            })
      in
      let body = substitute_all closure.body in
      if fires then (
        let helper = if closure.native then None else callee_name in
        List.iter claims ~f:(emit ctx ~site ~helper);
        match (leftover, body.closure) with
        | [], _ -> body
        | leftover, Some returned ->
            apply ctx ~site ~callee_name returned
              (List.map leftover ~f:(fun argument -> (Asttypes.Nolabel, argument)))
        | _ :: _, None ->
            List.iter leftover ~f:(discard ctx);
            body)
      else
        let remaining =
          List.filter_map assignments ~f:(fun (parameter, argument) ->
              match argument with Absent -> Some parameter | Supplied _ | Unknown _ -> None)
        in
        {
          nothing with
          closure =
            Some
              (Function
                 {
                   parameters = remaining;
                   claims;
                   body;
                   native = false;
                   format = false;
                   loose = false;
                 });
        }

(* Structure items in order: each binding sees the ones before it. Returns the environment after the
   items and what the structure exports, for a module -- both most recent first, so that a member
   defined twice resolves to its later definition wherever the module is opened. *)
and scan_structure ctx items =
  List.fold items ~init:(ctx.env, []) ~f:(fun (env, exports) item ->
      let ctx = { ctx with env } in
      match item.pstr_desc with
      | Pstr_value (recursive, bindings) ->
          let added = new_bindings ctx recursive bindings in
          (List.rev_append added env, List.rev_append added exports)
      | Pstr_eval (expr, _) ->
          discard ctx expr;
          (env, exports)
      | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
          let nested = module_exports ctx pmb_expr |> prefix_bindings name in
          (nested @ env, nested @ exports)
      | Pstr_module { pmb_expr; _ } ->
          ignore (module_exports ctx pmb_expr : binding list);
          (env, exports)
      | Pstr_recmodule bindings ->
          List.iter bindings ~f:(fun binding ->
              ignore (module_exports ctx binding.pmb_expr : binding list));
          (env, exports)
      | Pstr_open declaration ->
          let opened = module_exports ctx declaration.popen_expr in
          (opened @ env, exports)
      | Pstr_include declaration ->
          let included = module_exports ctx declaration.pincl_mod in
          (included @ env, included @ exports)
      | _ ->
          let iterator =
            object
              inherit Ast_traverse.iter as super
              method! attribute _ = ()
              method! expression expr = discard ctx expr

              method! structure items =
                ignore (scan_structure ctx items : binding list * binding list)

              method! structure_item item = super#structure_item item
            end
          in
          iterator#structure_item item;
          (env, exports))

(* What a module expression brings into scope when opened or aliased: a structure's own bindings; a
   known module's natives ([Verdict.Claims], [List], [Array]); a file-local module's exports. *)
and module_exports ctx module_expr =
  match module_expr.pmod_desc with
  | Pmod_structure items -> snd (scan_structure ctx items)
  | Pmod_constraint (inner, _) -> module_exports ctx inner
  | Pmod_ident _ -> (
      let site = site_of_location module_expr.pmod_loc in
      match module_path module_expr with
      | None -> []
      | Some path ->
          if
            List.equal String.equal path [ "Verdict"; "Claims" ]
            || List.equal String.equal path [ "Verdict" ]
          then claim_exports ~site
          else if is_collection_module path then quantifier_exports ~site
          else
            let prefix = String.concat ~sep:"." path ^ "." in
            List.filter_map ctx.env ~f:(fun binding ->
                String.chop_prefix binding.name ~prefix
                |> Option.map ~f:(fun name -> { binding with name })))
  | _ ->
      let iterator =
        object
          inherit Ast_traverse.iter as super
          method! attribute _ = ()
          method! expression expr = discard ctx expr
          method! structure items = ignore (scan_structure ctx items : binding list * binding list)
          method! module_expr module_expr = super#module_expr module_expr
        end
      in
      iterator#module_expr module_expr;
      []

(** Every [Verdict] claim [structure] fires, in source order, each with the sources its claimed
    Boolean can rest on. *)
let claims structure =
  let ctx = { env = []; pending = ref []; found = ref [] } in
  ignore (scan_structure ctx structure : binding list * binding list);
  List.rev !(ctx.found)
  |> List.dedup_and_sort ~compare:(fun a b ->
      match Int.compare a.fired.position b.fired.position with
      | 0 -> String.compare (claim_key a) (claim_key b)
      | order -> order)

(** How a claim's label reads: the literal, or the expression that computes it. *)
let label_text = function
  | Label_slot slot -> slot.name
  | Label_expr expr -> (
      match Read.string_literal expr with
      | Some literal -> literal
      | None -> Stdlib.Format.asprintf "%a" Pprintast.expression expr)
