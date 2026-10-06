(** Relating repository-scanner refusal diagnostics to permanent control goldens.

    A scanner refusal is mechanically visible when its source hands a string literal, directly or as
    a [Printf.sprintf] format, to [Verdict.fail], to the local [fail] alias scanners use, or to one
    of Verdict's claim forms: a false claim emits its label as the refusal. The dynamic values a
    helper returns are deliberately outside this reader: there is no diagnostic string constant in
    that scanner source to relate. The live check names those limits in its report rather than
    pretending to infer a value through arbitrary OCaml.

    Formats are reduced to a stable literal fragment. Code-shaped tokens (underscores or qualified
    names) win; otherwise the longest three-word run does. Values substituted by the failing run
    decide nothing. Thus an [(include_subdirs %s)] refusal contributes [include_subdirs], while a
    malformed marker refusal contributes a phrase naming that condition rather than a generic word.
    Coverage uses a marker whose digest is over the complete normalized format, so two diagnostics
    that happen to display the same fragment still need two control entries. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Digest = Stdlib.Digest
module Read = Config_key_scan

type kind = Fail | Claim
type diagnostic = { line : int; fragment : string; format : string; identity : string; kind : kind }

let normalize text =
  String.split_on_chars text ~on:[ ' '; '\t'; '\r'; '\n' ]
  |> List.filter ~f:(Fn.non String.is_empty)
  |> String.concat ~sep:" "

let trim_static_run text =
  String.strip text ~drop:(fun character ->
      Char.is_whitespace character
      || List.mem [ ':'; ';'; ','; '.'; '-'; '`'; '('; ')'; '['; ']' ] character ~equal:Char.equal)

let directive_stop format start =
  let length = String.length format in
  let rec modifiers index =
    if index >= length then index
    else
      match format.[index] with
      | '-' | '0' | '+' | ' ' | '#' | '.' | '*' -> modifiers (index + 1)
      | character when Char.is_digit character -> modifiers (index + 1)
      | _ -> index
  in
  let index = modifiers (start + 1) in
  if index >= length then length
  else
    match format.[index] with
    | ('l' | 'L' | 'n') when index + 1 < length -> index + 2
    | _ -> index + 1

let static_runs format =
  let length = String.length format in
  let buffer = Buffer.create length in
  let found = ref [] in
  let flush () =
    let run = Buffer.contents buffer |> normalize |> trim_static_run in
    Buffer.clear buffer;
    if String.exists run ~f:Char.is_alphanum then found := run :: !found
  in
  let rec loop index =
    if index >= length then flush ()
    else if not (Char.equal format.[index] '%') then (
      Buffer.add_char buffer format.[index];
      loop (index + 1))
    else if index + 1 < length && Char.equal format.[index + 1] '%' then (
      Buffer.add_char buffer '%';
      loop (index + 2))
    else (
      flush ();
      loop (directive_stop format index))
  in
  loop 0;
  List.rev !found

let words text =
  String.split text ~on:' '
  |> List.map
       ~f:
         (String.strip ~drop:(fun c ->
              List.mem [ ':'; ';'; ','; '.'; '-'; '`'; '('; ')'; '['; ']' ] c ~equal:Char.equal))
  |> List.filter ~f:(Fn.non String.is_empty)

let trigrams words =
  let rec loop found = function
    | a :: (b :: c :: _ as tail) -> loop (String.concat ~sep:" " [ a; b; c ] :: found) tail
    | _ -> List.rev found
  in
  loop [] words

let fragment_of_format format =
  let runs = static_runs format in
  let selected =
    Option.first_some
      (List.find runs ~f:(fun run -> String.count run ~f:Char.is_alpha >= 8))
      (List.hd runs)
  in
  Option.bind selected ~f:(fun run ->
      let words = words run in
      let code_tokens =
        List.filter words ~f:(fun token ->
            String.length token >= 4
            && String.exists token ~f:(fun c -> Char.equal c '_' || Char.equal c '.'))
      in
      match
        List.max_elt code_tokens ~compare:(fun a b ->
            Int.compare (String.length a) (String.length b))
      with
      | Some token -> Some token
      | None -> (
          match
            List.max_elt (trigrams words) ~compare:(fun a b ->
                Int.compare (String.length a) (String.length b))
          with
          | Some phrase -> Some phrase
          | None -> Some run))

let last_name expression = Option.bind (Read.longident_of expression) ~f:List.last

(** Multiset difference: [minus xs ys] removes one occurrence per [ys] element, since markers repeat
    when formats do and an argument list can name a source twice. *)
let minus xs ys =
  List.fold ys ~init:xs ~f:(fun remaining y ->
      let before, after = List.split_while remaining ~f:(Fn.non (String.equal y)) in
      before @ Option.value (List.tl after) ~default:[])

(** [Verdict.Claims] owns which claim forms there are; these two lists only judge which of them emit
    their label as a refusal. {!claims_mismatch} holds the two lists together equal to the members
    [Claims] declares, so a combinator [Claims] gains is a red scan until it is judged here, never a
    refusal this reader silently stops seeing. *)
let refusing_claims =
  [
    "fail";
    "p";
    "pf";
    "p_all";
    "p_all2";
    "pf_all2";
    "p_none";
    "p_alli";
    "p_exists";
    "p_empty";
    "p_pairwise_distinct";
    "claim";
    "claimf";
    "pass_fail";
    "pass_fail_all2";
    "gated";
    "gated_all";
    "gated_alli";
    "gated_exists";
  ]

(** [skipped] reports a skip, never a refusal; [case]'s label names a row, and what fails inside it
    reports through the claims it makes or the exception it catches. *)
let non_refusing_claims = [ "skipped"; "case" ]

(** [failwith] is the one refusing callee outside [Verdict.Claims]. *)
let refusal_callees = "failwith" :: refusing_claims

(** The members [module Claims] binds in [verdict_source] (the text of [test/support/verdict.ml]),
    in order; [None] when it declares no such structure. An item other than a plain [let x = ...] is
    named as unreadable, so it can only ever mismatch: a member it hides is never assumed judged. *)
let claims_members verdict_source =
  List.find_map (Read.structure_of verdict_source) ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_module
          {
            pmb_name = { txt = Some "Claims"; _ };
            pmb_expr = { pmod_desc = Pmod_structure items; _ };
            _;
          } ->
          Some
            (List.concat_map items ~f:(fun item ->
                 let unreadable () =
                   [ Printf.sprintf "<unreadable item, line %d>" item.pstr_loc.loc_start.pos_lnum ]
                 in
                 match item.pstr_desc with
                 | Pstr_value (_, bindings) ->
                     List.concat_map bindings ~f:(fun binding ->
                         match binding.pvb_pat.ppat_desc with
                         | Ppat_var { txt; _ } -> [ txt ]
                         | _ -> unreadable ())
                 | _ -> unreadable ()))
      | _ -> None)

type claims_mismatch = {
  unjudged : string list;  (** [Claims] members neither list judges. *)
  not_members : string list;
      (** Judged names [Claims] does not bind, or judged twice (as a multiset difference). *)
}

let claims_mismatch ~members =
  let judged = refusing_claims @ non_refusing_claims in
  { unjudged = minus members judged; not_members = minus judged members }

let is_refusal expression =
  Option.value_map (last_name expression) ~default:false ~f:(fun name ->
      List.mem refusal_callees name ~equal:String.equal)

let rec format_of expression =
  match Read.string_literal expression with
  | Some format -> Some format
  | None -> (
      match expression.pexp_desc with
      | Pexp_apply (operator, [ (Nolabel, _left); (Nolabel, right) ])
        when Option.value_map (last_name operator) ~default:false ~f:(String.equal "@@") ->
          format_of right
      | Pexp_apply (callee, arguments)
        when Option.value_map (Read.longident_of callee) ~default:false ~f:(function
               | [ "Printf"; ("sprintf" | "ksprintf") ] -> true
               | _ -> false) ->
          List.find_map arguments ~f:(fun (label, argument) ->
              match label with
              | Nolabel -> Read.string_literal argument
              | Labelled _ | Optional _ -> None)
      | _ -> None)

let diagnostic_argument expression =
  match expression.pexp_desc with
  | Pexp_apply (callee, arguments) when is_refusal callee ->
      Option.map
        (List.find_map arguments ~f:(fun (label, argument) ->
             match label with Nolabel -> Some argument | Labelled _ | Optional _ -> None))
        ~f:(fun argument -> (Option.value_exn (last_name callee), argument))
  | Pexp_apply (operator, [ (Nolabel, callee); (Nolabel, argument) ])
    when Option.value_map (last_name operator) ~default:false ~f:(String.equal "@@")
         && is_refusal callee ->
      Some (Option.value_exn (last_name callee), argument)
  | _ -> None

let diagnostics content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      (* A documentation comment is an attribute carrying a string. It is prose about a refusal,
         never the application that emits one. *)
      method! attribute _ = ()

      method! expression expression =
        (match diagnostic_argument expression with
        | Some (callee, argument) -> (
            match format_of argument with
            | None -> ()
            | Some format -> (
                match fragment_of_format format with
                | Some fragment ->
                    let identity = Digest.string (normalize format) |> Digest.to_hex in
                    let kind =
                      if List.mem [ "fail"; "failwith" ] callee ~equal:String.equal then Fail
                      else Claim
                    in
                    found :=
                      {
                        line = expression.pexp_loc.loc_start.pos_lnum;
                        fragment;
                        format;
                        identity;
                        kind;
                      }
                      :: !found
                | None -> ()))
        | None -> ());
        super#expression expression
    end
  in
  iterator#structure (Read.structure_of content);
  List.rev !found

let marker diagnostic =
  Printf.sprintf "[scanner-refusal:%s] %s" diagnostic.identity diagnostic.fragment

let format_matches ~format label =
  let runs = static_runs format in
  let label = normalize label in
  let rec consume position = function
    | [] -> true
    | run :: rest -> (
        match String.substr_index label ~pos:position ~pattern:run with
        | None -> false
        | Some found -> consume (found + String.length run) rest)
  in
  (not (List.is_empty runs)) && consume 0 runs

let coverage ~control_text diagnostics =
  let lines = String.split_lines control_text in
  let remaining = Hashtbl.create (module String) in
  List.map diagnostics ~f:(fun diagnostic ->
      let marker = marker diagnostic in
      let available =
        Hashtbl.find_or_add remaining marker ~default:(fun () ->
            List.count lines ~f:(String.is_substring ~substring:marker))
      in
      if available = 0 then false
      else (
        Hashtbl.set remaining ~key:marker ~data:(available - 1);
        true))

let orphans ~control_text diagnostics =
  List.zip_exn diagnostics (coverage ~control_text diagnostics)
  |> List.filter_map ~f:(fun (diagnostic, covered) -> if covered then None else Some diagnostic)
