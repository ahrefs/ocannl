(** The reader behind [test/operations/debug_name_scan] (gh-ocannl-1606): a test reads a tensor
    node's debug name only to show it, never to find the node.

    [Tnode.debug_name] is a display name. Once a renderer has minted a code name for a node, the
    debug name IS that code name: a label two nodes of one routine share becomes [n<id>_<label>],
    and with [output_debug_files_in_build_directory] the [.cd]/[.ll] printers mint names the C
    emitter never asks for. [batch_norm_running] found its owner's [running_mean_parallel] by debug
    name and raised [Not_found_s] once debug files were on (gh-ocannl-1552). A node is found by
    [Tnode.label_name], the name its label spells, which no renderer changes; this scan refuses a
    debug name used any other way than the closed list below.

    {1 What it reads}

    The parse tree (ppxlib's) of every [.ml] of the corpus the driver hands it.

    - {b A reference} is any value identifier whose last component is [debug_name], under any module
      path: [Tn.debug_name], [Ir.Tnode.debug_name], [Tensor.debug_name], a bare [debug_name] after
      an open. Shadowing is NOT resolved for references: a local value named [debug_name] is read as
      the library's, the loud direction.
    - {b An accepted use} is an application of a reference to exactly one positional argument (the
      node), directly or through [f @@ x] / [x |> f], whose value sits in one of these positions:

    + {e printed or raised}: any argument of [printf], [eprintf], [print_endline], [prerr_endline],
      [failwith], [failwithf], [invalid_arg] or [invalid_argf], unqualified or under [Stdio],
      [Printf], [Stdlib], [Stdlib.Printf], [Base.Printf] or [Stdio.Out_channel]; the payload of a
      [Failure] or [Invalid_argument] constructor (unqualified);
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
      above can express.

    - {b Transparent forms}, whose value takes the position the form itself sits in: an operand of
      [^]; any argument of [sprintf] (unqualified, or under [Printf], [Stdlib.Printf] or
      [Base.Printf]); any argument of [String.concat] (or [Base.String.concat],
      [Stdlib.String.concat]); [List.map] (or [Base.List.map]) whose [~f] is a reference itself,
      unapplied; a type constraint or coercion; a local open's body ([Tn.( ... )]); and a partial
      application standing for the whole one.
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
      every attribute of that name in the file), and a [[%string ...]] payload that mentions
      [debug_name], whose interpolations the parse tree does not expose.

    {1 What it deliberately does not read}

    - Flow beyond the positions above: a message raised through [failwith] and caught, or a needle
      check written as a [List.find] predicate, is accepted. The rule is "the debug name reaches
      display or a claim about rendered text", not a data-flow proof.
    - [Tnode.get_debug_name] (a pure function of an id and a label, no node), a record field named
      [debug_name] ([Shape.t]'s), a [~debug_name:] argument label, string literals and comments.
    - Exception constructors the source redefines ([exception Failure of string]), which are read as
      the standard ones. *)

open Base
open Ppxlib

type report = {
  accepted : (int * string) list;  (** Each accepted use: its line and the position's name. *)
  findings : string list;  (** Each refusal, as [file:line: message]. *)
}

let reference_name = "debug_name"
let marker = "rendered_name"
let printed = "printed or raised"
let claim_label = "a claim label"
let needle = "a needle"
let marked = "marked [@rendered_name]"

(** Every position name a report can carry, in a stable order. *)
let positions = [ printed; claim_label; needle; marked ]

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

let is_reference = function
  | { pexp_desc = Pexp_ident { txt; _ }; _ } -> String.equal (last txt) reference_name
  | _ -> false

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
  | List_map
  | Is_substring
  | Claim
  | Other of string

type position = Accepted of string | Consumed | Refused of string

let spelled txt = Longident.name txt

let scan ~file source =
  let structure = parse ~file source in
  let bound = bound_by_source structure in
  let verdicts = Verdict_callee.of_structure structure in
  let trusted (loc : Location.t) =
    not (Option.value (Hashtbl.find bound (key loc)) ~default:true)
  in
  let findings = ref [] and accepted = ref [] in
  let finding (loc : Location.t) message =
    findings := Printf.sprintf "%s:%d: %s" file (line loc) message :: !findings
  in
  let remedy =
    "a debug name is a display name a renderer can change (gh-ocannl-1552), so find a node by \
     Tnode.label_name; the accepted forms are in test/support/debug_name_scan.ml"
  in
  let refuse loc what desc =
    finding loc (Printf.sprintf "%s is used as %s -- %s" what desc remedy)
  in
  let judge loc what = function
    | Accepted form -> accepted := (line loc, form) :: !accepted
    | Refused desc -> refuse loc what desc
    | Consumed -> ()
  in
  let classify_head head =
    match (peel head).pexp_desc with
    | Pexp_ident { txt; _ } when String.equal (last txt) reference_name -> Reference
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
        match claim with
        | Some { ownership = Verdicts | Ambiguous; _ } -> Claim
        | _ when not (trusted loc) -> Other (spelled txt)
        | _ when List.exists sink_names ~f:(under sink_modules) -> Sink
        | _
          when under [ [] ] "^" || under sprintf_modules "sprintf" || under string_modules "concat"
          ->
            Transparent
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
  let is_marker (a : attribute) = String.equal a.attr_name.txt marker in
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
          "a [@rendered_name] marker needs one string literal of at least two words saying which \
           rendered text the display name is compared with";
        false
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

      method! expression e =
        let position = Option.value_or_thunk (take e) ~default:default_position in
        let marked_reference =
          List.find e.pexp_attributes ~f:is_marker
          |> Option.bind ~f:(fun a ->
              match flatten e with
              | head, [ (Nolabel, _) ] when is_reference head -> Some a
              | _ -> None)
        in
        let position =
          match marked_reference with
          | Some a -> if check_marker a then Accepted marked else Consumed
          | None -> position
        in
        (match e.pexp_desc with
        | Pexp_ident { txt; loc } when String.equal (last txt) reference_name -> (
            match position with
            | Consumed -> ()
            | Accepted _ | Refused _ ->
                let desc = match position with Refused d -> d | _ -> "a value" in
                finding loc
                  (Printf.sprintf "%s is referenced without being applied to a node, as %s -- %s"
                     (spelled txt) desc remedy))
        | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> assign inner position
        | Pexp_open (_, body) -> assign body position
        | Pexp_construct ({ txt = Lident ("Failure" | "Invalid_argument"); _ }, Some payload) ->
            assign payload (Accepted printed)
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
                | Reference -> Refused "the node argument of debug_name"
                | Sink -> Accepted printed
                | Transparent -> position
                | Is_substring -> (
                    match label with
                    | Labelled "substring" -> Accepted needle
                    | _ -> Refused "an argument of String.is_substring other than ~substring")
                | Claim when index = 1 -> Accepted claim_label
                | Claim -> Refused "an argument of a Verdict claim other than its label"
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
        | Pexp_extension ({ txt = "string"; _ }, PStr [ { pstr_desc = Pstr_eval (payload, _); _ } ])
          -> (
            match payload.pexp_desc with
            | Pexp_constant (Pconst_string (s, _, _))
              when String.is_substring s ~substring:reference_name ->
                finding e.pexp_loc
                  (Printf.sprintf
                     "a [%%string] payload mentions %s, and its interpolations are not in the \
                      parse tree for this scan to read -- %s"
                     reference_name remedy)
            | _ -> ())
        | _ -> ());
        enclosing := kind_of e :: !enclosing;
        super#expression e;
        enclosing := List.tl !enclosing |> Option.value ~default:[]
    end
  in
  walker#structure structure;
  List.iter (List.rev !all_markers) ~f:(fun a ->
      if not (Hash_set.mem consumed_markers (key a.attr_loc)) then
        finding a.attr_loc
          "a [@rendered_name] marker is attached to something other than an application of \
           debug_name to a node, so it marks nothing");
  { accepted = List.rev !accepted; findings = List.rev !findings }
