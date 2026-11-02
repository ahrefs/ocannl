(** The reader behind [test/operations/debug_name_scan] (gh-ocannl-1606): a test, and the library
    code under [lib/] and [tensor/], reads a tensor node's debug name only to show it, never to find
    the node or to spell a name that outlives the moment it is shown.

    [Tnode.debug_name] is a display name. Once a renderer has minted a code name for a node, the
    debug name IS that code name: a label two nodes of one routine share becomes [n<id>_<label>],
    and with [output_debug_files_in_build_directory] the [.cd]/[.ll] printers mint names the C
    emitter never asks for. [batch_norm_running] found its owner's [running_mean_parallel] by debug
    name and raised [Not_found_s] once debug files were on (gh-ocannl-1552). A node is found by
    [Tnode.label_name], the name its label spells, which no renderer changes. The same holds for a
    name the library stores rather than shows -- a tensor label ([Parallel.shard_along]'s shards), a
    block comment that lands in generated code ([Train.forward]), an ndarray's debug description, a
    shape-constraint origin: those take [label_name] too, so they do not depend on what a renderer
    minted first. This scan refuses a debug name used any other way than the closed list below.

    {1 What it reads}

    The parse tree (ppxlib's) of every [.ml] of the corpus the driver hands it, a quotation's
    payload ([[%expr ...]] in a ppx) included and read as the code it is: a debug-name read a ppx
    generates is held to the rule of one written by hand.

    - {b A reference} is any value identifier whose last component is [debug_name] or [debug_grad],
      under any module path: [Tn.debug_name], [Ir.Tnode.debug_name], [Tensor.debug_name],
      [Tensor.debug_grad], a bare [debug_name] after an open. Shadowing by a [let] is NOT resolved:
      a value a [let] names [debug_name] is read as the library's, the loud direction, which is also
      what makes an alias's callers references. A bare name a parameter or a pattern binds
      ([Shape.of_rows ~debug_name], a punned [{ debug_name }]) is not a reference: it is not the
      library's function, and the value it holds was judged where it was produced.
    - {b An accepted use} is an application of a reference to exactly one positional argument (the
      node), directly or through [f @@ x] / [x |> f], whose value sits in one of these positions:

    + {e printed or raised}: any argument of [printf], [eprintf], [print_endline], [prerr_endline],
      [failwith], [failwithf], [invalid_arg] or [invalid_argf], unqualified or under [Stdio],
      [Printf], [Stdlib], [Stdlib.Printf], [Base.Printf] or [Stdio.Out_channel]; the payload of a
      [Failure] or [Invalid_argument] constructor (unqualified); the message (a tuple's first
      component) of a [Session_error] payload, the framework's session exception, under any module
      path;
    + {e a claim label}: the FIRST positional argument of a [Verdict.Claims] member, as
      {!Verdict_callee} resolves it to [Verdict]. A callee that reaches [Verdict] only past an
      [open] of a module the resolver cannot read ([Ambiguous]) is read as [Verdict]'s, as the other
      [Verdict] readers read it: refusing it would refuse every claim of a test that opens
      [Verdict.Claims] before its DSL modules, and such a module exporting its own [p] whose label
      is a lookup is outside what this scan guards against;
    + {e a needle}: the [~substring] argument of [String.is_substring] (or
      [Base.String.is_substring]) -- a claim that some text names the node;
    + {e marked}: the application carries [[\@rendered_name "<reason>"]], whose payload is one
      string literal of at least two words. It is for a comparison with text a renderer of the same
      process produced under the display name (a census row, a refusal message), which no position
      above can express;
    + {e marked message-only}: the application, or a [[%string]] whose interpolations apply a
      reference, carries [[\@message_only "<reason>"]] (the same payload rule). It is for a debug
      name that reaches a message by a road no position above follows -- a refusal message a
      function returns for its caller to raise, a log line through a local formatter, an autotune
      arm's name on the progress stream -- and that is composed when it is shown, so it names the
      node as the code emitted by then does;
    + {e an alias}: the body of a one-parameter function bound to a reference name
      ([let debug_name t = Tn.debug_name t.value] in [Tensor]), whose callers this scan reads as
      references in turn.

    - {b Transparent forms}, whose value takes the position the form itself sits in: an operand of
      [^]; any argument of [sprintf] (unqualified, or under [Printf], [Stdlib.Printf] or
      [Base.Printf]); any argument of [String.concat] (or [Base.String.concat],
      [Stdlib.String.concat]); [List.map] (or [Base.List.map]) whose [~f] is a reference itself,
      unapplied (a marker may sit on that application); a list literal's elements, when the list
      itself sits in an accepted position; a type constraint or coercion; a local open's body
      ([Tn.( ... )]); a partial application standing for the whole one; and each [%{...}]
      interpolation of a [[%string ...]], which the scan parses out of the literal as ppx_string
      v0.17 splits it (from [%{] to the next [}], the value before the last [#]). Its lines are
      counted in the decoded literal, so within a literal with escapes or line continuations a
      finding's line can be early.
    - {b Trust in the names above}: a sink, transparent function, operator or module head spelled
      unqualified is trusted only when no binding of the source is in scope for it (resolved by
      {!Lexical_scope}: a [let printf = ...], a parameter named [printf], a [module String = ...]
      all make it untrusted, and an untrusted name is an ordinary function). An [open] of a module
      brings no names in, so a sink name an opened module redefines is read as the standard one.
    - {b Refused}: every other reference -- used as an argument of any other function
      ([String.equal], [List.find], [Map.find], a local helper), bound by a [let], returned from a
      function, put in a tuple, list, record or array, compared with [=], referenced without being
      applied ([~f:Tn.debug_name] anywhere but [List.map] in an accepted position), or applied to
      anything but one positional argument. Also refused: a [[\@rendered_name]] marker that is
      malformed or attached to anything but a reference's application (an accounting check over
      every attribute of either name in the file), and a [[%string ...]] payload that mentions a
      reference name in an interpolation that does not parse.

    {1 What it deliberately does not read}

    - Flow beyond the positions above: a message raised through [failwith] and caught, or a needle
      check written as a [List.find] predicate, is accepted. The rule is "the debug name reaches
      display or a claim about rendered text", not a data-flow proof.
    - [Tnode.get_debug_name] (a pure function of an id and a label, no node), a record field named
      [debug_name] ([Shape.t]'s, spelled from the label when the shape is made), a [~debug_name:]
      argument label, string literals and comments.
    - Exception constructors the source redefines ([exception Failure of string]), which are read as
      the standard ones. *)

open Base
open Ppxlib

type report = {
  accepted : (int * string) list;  (** Each accepted use: its line and the position's name. *)
  findings : string list;  (** Each refusal, as [file:line: message]. *)
}

let printed = "printed or raised"
let claim_label = "a claim label"
let needle = "a needle"
let marked = "marked [@rendered_name]"
let message_only = "marked [@message_only]"
let alias = "a debug-name alias"
let logged = "a log payload"
let copied = "copied onto a new node"
let forwarded = "forwarded under its own label"
let label_read = "marked [@label_read]"
let composed = "composed into a message"

type marker = { attr : string; position : string; says : string }
(** A marker attribute: its name, the position it puts its target in, and what its reason must say
    (the tail of the malformed-marker refusal). *)

type rule = {
  references : string list;
      (** Value identifiers, under any module path, that are references -- except a bare one a
          parameter or a pattern binds. *)
  parameter_references : string list;
      (** The subset of [references] that stays a reference where a parameter or a pattern binds it
          bare: a value whose every holder is judged, not only its producer. *)
  qualified_references : string list;
      (** Names that are references only qualified ([Tn.label]), or bare where no binding of the
          source is in scope (an [open] brought them in): bare, they are common local names. *)
  values : string list;
      (** Reference names that may be read unapplied, as the value itself: judged by the position
          the identifier sits in rather than refused for not being applied. *)
  fields : string list;
      (** Record fields, under any type path, whose read ([e.label], [e.Tn.label]) is a reference
          judged like an application's result, and whose destructuring in a pattern is refused. *)
  rule_markers : marker list;
  raised : string list;
      (** Exception constructors, by last component under any path, whose whole payload is printed
          or raised. *)
  copy_slots : bool;
      (** The [~label] and [~namespace] arguments of [Tn.create] / [Tnode.create] (any module path
          ending in one of those, its head trusted even when the source binds it, as
          [module Tn = Tnode] does) are accepted, a list cons there transparent. *)
  composed_is_display : bool;
      (** An operand of [^], of [sprintf] or of a [[%string]] interpolation is accepted outright,
          rather than taking the position of the text it is composed into: that text is a message,
          and the spelling no longer stands alone in it. [String.concat] stays transparent, since a
          label list joined whole is the label. *)
  log_regions : bool;
      (** Every reference inside the payload of a [[%log...]] extension (ppx_minidebug's) is
          accepted: such a payload is evaluated only to be shown. *)
  remedy : string;
  node_argument : string;  (** How a refusal names the argument of an applied reference. *)
  unparsed_mention : string;  (** How a refusal names a reference an unparsed [%string] mentions. *)
  marker_targets : string;
      (** What a marker may be attached to, as a stray marker's refusal says. *)
}
(** What one reading refuses and accepts. [debug_name_rule] is the reading this header describes; a
    second rule ([Label_dispatch_scan]'s, gh-ocannl-1783) reuses the walker with the knobs below,
    each of which adds references or accepted positions and none of which the default turns on. *)

let debug_name_rule =
  {
    references = [ "debug_name"; "debug_grad" ];
    parameter_references = [];
    qualified_references = [];
    values = [];
    fields = [];
    rule_markers =
      [
        {
          attr = "rendered_name";
          position = marked;
          says = "which rendered text the display name is compared with";
        };
        {
          attr = "message_only";
          position = message_only;
          says = "which message the display name reaches";
        };
      ];
    raised = [];
    copy_slots = false;
    composed_is_display = false;
    log_regions = false;
    remedy =
      "a debug name is a display name a renderer can change (gh-ocannl-1552), so find a node by \
       Tnode.label_name; the accepted forms are in test/support/debug_name_scan.ml";
    node_argument = "the node argument of debug_name";
    unparsed_mention = "a debug-name reference";
    marker_targets = "an application of debug_name to a node or a [%string] interpolating one";
  }

(** Every position name a report of [rule] can carry, in a stable order. *)
let positions_of rule =
  [ printed; claim_label; needle ]
  @ List.map rule.rule_markers ~f:(fun m -> m.position)
  @ [ alias ]
  @ (if rule.composed_is_display then [ composed ] else [])
  @ (if rule.log_regions then [ logged ] else [])
  @ (if rule.copy_slots then [ copied ] else [])
  @ if List.is_empty rule.parameter_references then [] else [ forwarded ]

(** Every position name a report can carry, in a stable order. *)
let positions = positions_of debug_name_rule

(* The extension a [[%string ...]] becomes once its interpolations are parsed: one [Pstr_eval] item
   per interpolation, in order. A [[%string]] whose literal does not split and parse keeps its own
   name, and is refused when it mentions a reference name. *)
let interpolations_tag = "string.interpolations"

let parse ~file source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf file;
  Parse.implementation lexbuf

let line (loc : Location.t) = loc.loc_start.pos_lnum
let key (loc : Location.t) = (loc.loc_start.pos_cnum, loc.loc_end.pos_cnum)

let rec components = function
  | Longident.Lident s -> Some [ s ]
  | Ldot (p, s) -> Option.map (components p) ~f:(fun c -> c @ [ s ])
  | Lapply _ -> None

let rec last = function Longident.Lident s | Ldot (_, s) -> s | Lapply (_, p) -> last p

(* The value of each [%{...}] of a [[%string]] literal, with its offset, split as ppx_string v0.17
   splits it: from [%{] to the next [}], the text before the last [#] (a conversion module and pad
   length follow it). *)
let interpolation_values literal =
  let rec go pos acc =
    match String.substr_index literal ~pos ~pattern:"%{" with
    | None -> Some (List.rev acc)
    | Some percent -> (
        match String.substr_index literal ~pos:(percent + 2) ~pattern:"}" with
        | None -> None
        | Some rbrace ->
            let body = String.sub literal ~pos:(percent + 2) ~len:(rbrace - percent - 2) in
            let value =
              match String.rsplit2 body ~on:'#' with Some (value, _) -> value | None -> body
            in
            go (rbrace + 1) ((percent + 2, value) :: acc))
  in
  go 0 []

(* Each interpolation parsed, positioned inside the literal: its line counted in the decoded
   literal, its character offset the literal's start plus its decoded offset -- inside the literal's
   source span, where no other node of the file sits, so location keys stay distinct. *)
let parse_interpolations ~(literal_loc : Location.t) literal =
  Option.bind (interpolation_values literal) ~f:(fun values ->
      Option.try_with (fun () ->
          List.map values ~f:(fun (offset, value) ->
              let start = literal_loc.loc_start in
              let lines = String.count (String.prefix literal offset) ~f:(Char.equal '\n') in
              let cnum = start.pos_cnum + offset in
              let lexbuf = Lexing.from_string value in
              Lexing.set_position lexbuf
                { start with pos_lnum = start.pos_lnum + lines; pos_cnum = cnum; pos_bol = cnum };
              Parse.expression lexbuf)))

(* Every [[%string]] whose literal splits and parses becomes an [interpolations_tag] extension
   holding its parsed interpolations, so the scope resolution and the walk below both read them as
   the expressions they are. *)
let expand_strings =
  object
    inherit Ast_traverse.map as super

    method! expression e =
      let e = super#expression e in
      match e.pexp_desc with
      | Pexp_extension
          ( ({ txt = "string"; _ } as name),
            PStr
              [
                {
                  pstr_desc =
                    Pstr_eval
                      ({ pexp_desc = Pexp_constant (Pconst_string (literal, literal_loc, _)); _ }, _);
                  _;
                };
              ] ) -> (
          match parse_interpolations ~literal_loc literal with
          | Some values ->
              let items =
                List.map values ~f:(fun v ->
                    { pstr_desc = Pstr_eval (v, []); pstr_loc = v.pexp_loc })
              in
              {
                e with
                pexp_desc = Pexp_extension ({ name with txt = interpolations_tag }, PStr items);
              }
          | None -> e)
      | _ -> e
  end

(* Whether an expanded [[%string]]'s interpolations apply a reference anywhere. *)
let interpolates_reference ~is_reference items =
  let found = ref false in
  let finder =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        if is_reference e then found := true;
        super#expression e
    end
  in
  finder#structure items;
  !found

let rec peel e =
  match e.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel inner
  | _ -> e

(* Whether each value identifier's name -- unqualified, or its module head -- is bound by the source
   where it is spelled, keyed by the identifier's location. *)
let bound_by_source structure =
  let table = Hashtbl.Poly.create () in
  let walker =
    object
      inherit [unit, unit] Lexical_scope.scoped
      method local = ()
      method module_path _ _ = Some ()
      method! shadowed = Some ()

      method! ident env { txt; loc } =
        let bound =
          match txt with
          | Lident name -> Option.is_some (Lexical_scope.lookup env name)
          | Ldot _ | Lapply _ -> Map.mem env.modules (Lexical_scope.head txt)
        in
        Hashtbl.set table ~key:(key loc) ~data:bound
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } structure
      : structure);
  table

(* The location keys of the unqualified value identifiers that a parameter or a pattern of the
   source binds where they are spelled. *)
let bound_as_parameter structure =
  let table = Hash_set.Poly.create () in
  let walker =
    object
      inherit [[ `Parameter | `Let ], unit] Lexical_scope.scoped
      method local = `Parameter
      method! let_denotes _ = `Let
      method module_path _ _ = Some ()
      method! shadowed = Some ()

      method! ident env { txt; loc } =
        match txt with
        | Lident name -> (
            match Lexical_scope.lookup env name with
            | Some `Parameter -> Hash_set.add table (key loc)
            | Some `Let | None -> ())
        | Ldot _ | Lapply _ -> ()
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } structure
      : structure);
  table

let sink_names =
  [
    "printf";
    "eprintf";
    "print_endline";
    "prerr_endline";
    "failwith";
    "failwithf";
    "invalid_arg";
    "invalid_argf";
  ]

let sink_modules =
  [
    [];
    [ "Stdio" ];
    [ "Printf" ];
    [ "Stdlib" ];
    [ "Stdlib"; "Printf" ];
    [ "Base"; "Printf" ];
    [ "Stdio"; "Out_channel" ];
  ]

let sprintf_modules = [ []; [ "Printf" ]; [ "Stdlib"; "Printf" ]; [ "Base"; "Printf" ] ]
let string_modules = [ [ "String" ]; [ "Base"; "String" ]; [ "Stdlib"; "String" ] ]
let base_string_modules = [ [ "String" ]; [ "Base"; "String" ] ]
let list_modules = [ [ "List" ]; [ "Base"; "List" ] ]

type head_kind =
  | Reference
  | Sink
  | Transparent
  | Composing
  | List_map
  | Is_substring
  | Claim
  | Create
  | Other of string

type position = Accepted of string | Consumed | Refused of string

let spelled txt = Longident.name txt

let scan ?(rule = debug_name_rule) ~file source =
  let structure = expand_strings#structure (parse ~file source) in
  let bound = bound_by_source structure in
  let parameters = bound_as_parameter structure in
  let mem names name = List.mem names name ~equal:String.equal in
  let is_reference_name name = mem rule.references name || mem rule.qualified_references name in
  let names_reference txt (loc : Location.t) =
    let name = last txt in
    if mem rule.references name then
      mem rule.parameter_references name || not (Hash_set.mem parameters (key loc))
    else if mem rule.qualified_references name then
      match txt with
      | Lident _ -> Option.value (Hashtbl.find bound (key loc)) ~default:true |> not
      | Ldot _ | Lapply _ -> true
    else false
  in
  let is_field txt = mem rule.fields (last txt) in
  let is_reference e =
    match e.pexp_desc with Pexp_ident { txt; loc } -> names_reference txt loc | _ -> false
  in
  let verdicts = Verdict_callee.of_structure structure in
  let trusted (loc : Location.t) =
    not (Option.value (Hashtbl.find bound (key loc)) ~default:true)
  in
  let findings = ref [] and accepted = ref [] in
  let finding (loc : Location.t) message =
    findings := Printf.sprintf "%s:%d: %s" file (line loc) message :: !findings
  in
  let remedy = rule.remedy in
  let log_depth = ref 0 in
  let refuse loc what desc =
    finding loc (Printf.sprintf "%s is used as %s -- %s" what desc remedy)
  in
  let judge loc what position =
    let position = if !log_depth > 0 then Accepted logged else position in
    match position with
    | Accepted form -> accepted := (line loc, form) :: !accepted
    | Refused desc -> refuse loc what desc
    | Consumed -> ()
  in
  let classify_head head =
    match (peel head).pexp_desc with
    | Pexp_ident { txt; loc } when names_reference txt loc -> Reference
    | Pexp_ident { txt; loc } -> (
        let path = components txt in
        let under modules name =
          match path with
          | Some parts -> (
              match List.rev parts with
              | last_part :: rev_modules ->
                  String.equal last_part name
                  && List.mem modules (List.rev rev_modules) ~equal:(List.equal String.equal)
              | [] -> false)
          | None -> false
        in
        let claim = Verdict_callee.resolve verdicts head in
        let creates =
          rule.copy_slots
          &&
          match Option.map path ~f:List.rev with
          | Some ("create" :: ("Tn" | "Tnode") :: _) -> true
          | _ -> false
        in
        match claim with
        | Some { ownership = Verdicts | Ambiguous; _ } -> Claim
        | _ when creates -> Create
        | _ when not (trusted loc) -> Other (spelled txt)
        | _ when List.exists sink_names ~f:(under sink_modules) -> Sink
        | _ when under [ [] ] "^" || under sprintf_modules "sprintf" -> Composing
        | _ when under string_modules "concat" -> Transparent
        | _ when under list_modules "map" -> List_map
        | _ when under base_string_modules "is_substring" -> Is_substring
        | _ -> Other (spelled txt))
    | _ -> Other "a computed function"
  in
  (* (head, every argument) of an application, through partial applications and trusted
     [@@]/[|>]. *)
  let rec flatten e =
    match e.pexp_desc with
    | Pexp_apply (f, args) -> (
        match ((peel f).pexp_desc, args) with
        | Pexp_ident { txt = Lident "@@"; loc }, [ (Nolabel, fn); (Nolabel, x) ] when trusted loc ->
            let head, earlier = flatten_function fn in
            (head, earlier @ [ (Nolabel, x) ])
        | Pexp_ident { txt = Lident "|>"; loc }, [ (Nolabel, x); (Nolabel, fn) ] when trusted loc ->
            let head, earlier = flatten_function fn in
            (head, earlier @ [ (Nolabel, x) ])
        | _ ->
            let head, earlier = flatten_function f in
            (head, earlier @ args))
    | _ -> (e, [])
  and flatten_function f =
    let f = peel f in
    match f.pexp_desc with Pexp_apply _ -> flatten f | _ -> (f, [])
  in
  let marker_of (a : attribute) =
    List.find rule.rule_markers ~f:(fun m -> String.equal m.attr a.attr_name.txt)
  in
  let marker_position a = Option.map (marker_of a) ~f:(fun m -> m.position) in
  let is_marker a = Option.is_some (marker_position a) in
  let consumed_markers = Hash_set.Poly.create () in
  let all_markers = ref [] in
  let check_marker (a : attribute) =
    Hash_set.add consumed_markers (key a.attr_loc);
    let reason =
      match a.attr_payload with
      | PStr
          [
            {
              pstr_desc = Pstr_eval ({ pexp_desc = Pexp_constant (Pconst_string (s, _, _)); _ }, _);
              _;
            };
          ] ->
          Some s
      | _ -> None
    in
    match reason with
    | Some s
      when List.length
             (String.split_on_chars s ~on:[ ' '; '\t'; '\n' ]
             |> List.filter ~f:(Fn.non String.is_empty))
           >= 2 ->
        true
    | _ ->
        finding a.attr_loc
          (Printf.sprintf "a [@%s] marker needs one string literal of at least two words saying %s"
             a.attr_name.txt
             (Option.value_map (marker_of a) ~default:"" ~f:(fun m -> m.says)));
        false
  in
  let is_forwarded_name name = mem rule.parameter_references name in
  let is_forwarded = function
    | Labelled name | Optional name -> is_forwarded_name name
    | Nolabel -> false
  in
  let rec peel_pattern p =
    match p.ppat_desc with Ppat_constraint (inner, _) -> peel_pattern inner | _ -> p
  in
  let pending = ref [] in
  let assign e position = pending := (e, position) :: !pending in
  let take e =
    match List.find !pending ~f:(fun (e', _) -> phys_equal e e') with
    | Some (_, position) ->
        pending := List.filter !pending ~f:(fun (e', _) -> not (phys_equal e e'));
        Some position
    | None -> None
  in
  let enclosing = ref [] in
  let kind_of e =
    match e.pexp_desc with
    | Pexp_let _ -> "a let binding"
    | Pexp_tuple _ -> "a tuple component"
    | Pexp_construct _ | Pexp_variant _ -> "a constructor argument (a list element)"
    | Pexp_record _ -> "a record field"
    | Pexp_array _ -> "an array element"
    | Pexp_function _ -> "a function's body"
    | Pexp_match _ | Pexp_try _ -> "a match"
    | Pexp_ifthenelse _ -> "an if expression"
    | Pexp_sequence _ -> "a sequence"
    | Pexp_field _ | Pexp_setfield _ -> "a field access"
    | _ -> "an expression the scan does not read as display"
  in
  let default_position () =
    Refused (match !enclosing with kind :: _ -> kind | [] -> "a top-level binding")
  in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! attribute a =
        if is_marker a then all_markers := a :: !all_markers;
        super#attribute a

      (* An alias: [let debug_name t = <body>], whose callers are references in turn. *)
      method! value_binding vb =
        (match (vb.pvb_pat.ppat_desc, vb.pvb_expr.pexp_desc) with
        | Ppat_var { txt; _ }, Pexp_function ([ _ ], None, Pfunction_body body)
          when is_reference_name txt ->
            assign body (Accepted alias)
        | _ -> ());
        super#value_binding vb

      method! expression e =
        let position = Option.value_or_thunk (take e) ~default:default_position in
        let marked =
          List.find e.pexp_attributes ~f:is_marker
          |> Option.bind ~f:(fun a ->
              match (e.pexp_desc, flatten e) with
              | Pexp_extension ({ txt; _ }, PStr items), _
                when String.equal txt interpolations_tag
                     && interpolates_reference ~is_reference items ->
                  Some a
              | _, (head, [ (Nolabel, _) ]) when is_reference head -> Some a
              | _, (head, args)
                when (match classify_head head with List_map -> true | _ -> false)
                     && List.exists args ~f:(function
                       | Labelled "f", f -> is_reference (peel f)
                       | _ -> false) ->
                  Some a
              | Pexp_field (_, { txt; _ }), _ when is_field txt -> Some a
              | Pexp_ident { txt; loc }, _
                when names_reference txt loc && mem rule.values (last txt) ->
                  Some a
              | _ -> None)
        in
        let position =
          match marked with
          | Some a ->
              if check_marker a then Accepted (Option.value_exn (marker_position a)) else Consumed
          | None -> position
        in
        (match e.pexp_desc with
        | Pexp_ident { txt; loc } when names_reference txt loc && mem rule.values (last txt) -> (
            match position with Consumed -> () | _ -> judge loc (spelled txt) position)
        | Pexp_ident { txt; loc } when names_reference txt loc -> (
            match position with
            | Consumed -> ()
            | (Accepted _ | Refused _) when !log_depth > 0 -> judge loc (spelled txt) position
            | Accepted _ | Refused _ ->
                let desc = match position with Refused d -> d | _ -> "a value" in
                finding loc
                  (Printf.sprintf "%s is referenced without being applied to a node, as %s -- %s"
                     (spelled txt) desc remedy))
        | Pexp_field (_, { txt; _ }) when is_field txt ->
            judge e.pexp_loc (Printf.sprintf "the %s field" (spelled txt)) position
        | Pexp_construct
            ({ txt = Lident "::"; _ }, Some { pexp_desc = Pexp_tuple [ head; tail ]; _ })
          when match position with Accepted _ -> true | Consumed | Refused _ -> false ->
            assign head position;
            assign tail position
        | Pexp_construct ({ txt; _ }, Some payload) when mem rule.raised (last txt) ->
            assign payload (Accepted printed)
        | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> assign inner position
        | Pexp_open (_, body) -> assign body position
        | Pexp_construct ({ txt = Lident ("Failure" | "Invalid_argument"); _ }, Some payload) ->
            assign payload (Accepted printed)
        | Pexp_construct
            ( { txt = Lident "Session_error" | Ldot (_, "Session_error"); _ },
              Some { pexp_desc = Pexp_tuple (message :: _); _ } ) ->
            assign message (Accepted printed)
        | Pexp_apply (f, args) -> (
            let head, all_args = flatten e in
            let direct = List.map args ~f:snd in
            let own e' = List.exists direct ~f:(phys_equal e') in
            (* The function part: the head itself, or a partial application standing for the whole
               (and, for [@@] / [|>], the operator and the function operand). *)
            let function_part c =
              if phys_equal (peel c) head then assign c Consumed
              else match (peel c).pexp_desc with Pexp_apply _ -> assign c position | _ -> ()
            in
            (match ((peel f).pexp_desc, args) with
            | Pexp_ident { txt = Lident (("@@" | "|>") as operator); loc }, [ (_, a); (_, b) ]
              when trusted loc ->
                assign f Consumed;
                function_part (if String.equal operator "@@" then a else b)
            | _ -> function_part f);
            let positional = ref 0 in
            let for_arg (label, arg) =
              let index =
                match label with
                | Nolabel ->
                    Int.incr positional;
                    !positional
                | _ -> 0
              in
              let argument_position =
                match classify_head head with
                | Reference -> Refused rule.node_argument
                | Sink -> Accepted printed
                | Transparent -> position
                | Composing -> if rule.composed_is_display then Accepted composed else position
                | Is_substring -> (
                    match label with
                    | Labelled "substring" -> Accepted needle
                    | _ -> Refused "an argument of String.is_substring other than ~substring")
                | _ when is_forwarded label -> Accepted forwarded
                | Claim when index = 1 -> Accepted claim_label
                | Claim -> Refused "an argument of a Verdict claim other than its label"
                | Create -> (
                    match label with
                    | Labelled ("label" | "namespace") -> Accepted copied
                    | _ -> Refused "an argument of Tn.create other than ~label or ~namespace")
                | List_map -> (
                    match label with
                    | Labelled "f" when is_reference (peel arg) -> Consumed
                    | _ -> Refused "an argument of List.map")
                | Other name -> Refused ("an argument of " ^ name)
              in
              (* Judged by the node the argument is a direct child of, so a [|>]-spelled map and the
                 partial application inside it judge it once between them. *)
              (match (classify_head head, label) with
              | List_map, Labelled "f" when own arg && is_reference (peel arg) ->
                  judge arg.pexp_loc
                    (Printf.sprintf "List.map ~f:%s's result"
                       (match (peel arg).pexp_desc with
                       | Pexp_ident { txt; _ } -> spelled txt
                       | _ -> ""))
                    position
              | _ -> ());
              if own arg then assign arg argument_position
            in
            List.iter all_args ~f:for_arg;
            match classify_head head with
            | Reference -> (
                let what =
                  match (peel head).pexp_desc with
                  | Pexp_ident { txt; _ } -> spelled txt ^ "'s result"
                  | _ -> "the result"
                in
                match all_args with
                | [ (Nolabel, _) ] -> judge e.pexp_loc what position
                | _ ->
                    finding e.pexp_loc
                      (Printf.sprintf "%s is applied to %d arguments rather than to one node -- %s"
                         what (List.length all_args) remedy))
            | _ -> ())
        | Pexp_extension ({ txt; _ }, PStr items) when String.equal txt interpolations_tag ->
            List.iter items ~f:(function
              | { pstr_desc = Pstr_eval (value, _); _ } ->
                  assign value (if rule.composed_is_display then Accepted composed else position)
              | _ -> ())
        | Pexp_extension ({ txt = "string"; _ }, PStr [ { pstr_desc = Pstr_eval (payload, _); _ } ])
          -> (
            match payload.pexp_desc with
            | Pexp_constant (Pconst_string (s, _, _))
              when List.exists
                     (rule.references @ rule.qualified_references @ rule.fields)
                     ~f:(fun name -> String.is_substring s ~substring:name) ->
                finding e.pexp_loc
                  (Printf.sprintf
                     "a [%%string] payload mentions %s in an interpolation that does not parse, so \
                      this scan cannot read it -- %s"
                     rule.unparsed_mention remedy)
            | _ -> ())
        | _ -> ());
        let region =
          match e.pexp_desc with
          | Pexp_extension ({ txt; _ }, _) -> rule.log_regions && String.is_prefix txt ~prefix:"log"
          | _ -> false
        in
        if region then Int.incr log_depth;
        enclosing := kind_of e :: !enclosing;
        super#expression e;
        enclosing := List.tl !enclosing |> Option.value ~default:[];
        if region then Int.decr log_depth

      (* A destructured reference field is a read this scan cannot follow. *)
      method! pattern p =
        (match p.ppat_desc with
        | Ppat_record (fields, _) ->
            List.iter fields ~f:(fun ({ txt; loc }, field_pattern) ->
                let binds = match field_pattern.ppat_desc with Ppat_any -> false | _ -> true in
                if is_field txt && binds then
                  finding loc
                    (Printf.sprintf
                       "the %s field is destructured in a pattern, a read this scan cannot follow \
                        -- %s"
                       (spelled txt) remedy))
        | _ -> ());
        super#pattern p

      (* A parameter-reference name renamed on entry ([~backend_name:b]) would leave the value under
         a name no reference spells. *)
      method! function_param fp =
        (match fp.pparam_desc with
        | Pparam_val ((Labelled name | Optional name), _, pat) when is_forwarded_name name -> (
            match (peel_pattern pat).ppat_desc with
            | Ppat_var { txt; _ } when String.equal txt name -> ()
            | _ ->
                finding fp.pparam_loc
                  (Printf.sprintf
                     "the ~%s parameter is bound to another name, which no reference spells -- %s"
                     name remedy))
        | _ -> ());
        super#function_param fp
    end
  in
  walker#structure structure;
  List.iter (List.rev !all_markers) ~f:(fun a ->
      if not (Hash_set.mem consumed_markers (key a.attr_loc)) then
        finding a.attr_loc
          (Printf.sprintf
             "a [@%s] marker is attached to something other than %s, so it marks nothing"
             a.attr_name.txt rule.marker_targets));
  { accepted = List.rev !accepted; findings = List.rev !findings }
