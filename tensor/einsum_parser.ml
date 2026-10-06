(** Entry point for the einsum parser library.

    This module provides functions to parse einsum notation specifications using a Menhir-based
    parser (instead of Angstrom). *)

open Base

(* Re-export types from Einsum_types *)
include Einsum_types

exception Parse_error of string

let binary_operators_with_generated_specs = [ "+*"; "@^+"; "+++" ]
let unary_operators_with_generated_specs = [ "++"; "@^^" ]
let concat_operator_with_generated_specs = "++^"

let operators_with_generated_specs =
  binary_operators_with_generated_specs @ unary_operators_with_generated_specs
  @ [ concat_operator_with_generated_specs ]

(* Helper to determine if input uses multichar mode *)
let is_multichar = Lexer.is_multichar

(* The name in a [...name..] (optionally [...name,..]) misspelling of the row variable [..name..].
   Such text never parses: after an ellipsis a row continues only with axes, and [..] opens a named
   row variable only in a row that has no ellipsis yet. Identifier characters follow the multichar
   lexer's [alpha alphanum*]. *)
let misspelled_row_variable spec =
  let n = String.length spec in
  let skip_white i =
    let rec go i = if i < n && Char.is_whitespace spec.[i] then go (i + 1) else i in
    go i
  in
  let is_ident_char c = Char.is_alphanum c || Char.equal c '_' in
  let rec ident_end i = if i < n && is_ident_char spec.[i] then ident_end (i + 1) else i in
  let rec find pos =
    match String.substr_index spec ~pos ~pattern:"..." with
    | None -> None
    | Some j ->
        let start = skip_white (j + 3) in
        let stop = if start < n && Char.is_alpha spec.[start] then ident_end start else start in
        let close = skip_white stop in
        let close =
          if close < n && Char.equal spec.[close] ',' then skip_white (close + 1) else close
        in
        if stop > start && String.is_substring_at spec ~pos:close ~substring:".." then
          Some (String.sub spec ~pos:start ~len:(stop - start))
        else find (j + 1)
  in
  find 0

let parse entry spec =
  let multichar = is_multichar spec in
  let lexbuf = Lexing.from_string spec in
  try entry (Lexer.token multichar) lexbuf with
  | Lexer.Syntax_error msg -> raise (Parse_error ("Lexer error: " ^ msg))
  | Parser.Error ->
      let pos = lexbuf.Lexing.lex_curr_p in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      let hint =
        match misspelled_row_variable spec with
        | None -> ""
        | Some name ->
            Printf.sprintf
              "; hint: `...` is the unnamed context ellipsis; a named row variable is written \
               `..%s..`, not `...%s..`"
              name name
      in
      raise
        (Parse_error
           (Printf.sprintf "Parse error at line %d, column %d in spec: %s%s" line col spec hint))

(* Parse axis labels specification *)
let axis_labels_of_spec spec = parse Parser.axis_labels_spec spec

(* Parse einsum specification *)
let einsum_of_spec spec = parse Parser.einsum_spec spec
