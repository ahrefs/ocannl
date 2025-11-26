(** Entry point for the einsum parser library.

    This module provides functions to parse einsum notation specifications
    using a Menhir-based parser (instead of Angstrom).
*)

open Base

(* Re-export types from Einsum_types *)
include Einsum_types

exception Parse_error of string

(* Helper to determine if input uses multichar mode *)
let is_multichar = Lexer.is_multichar

(* Parse axis labels specification *)
let axis_labels_of_spec spec =
  let multichar = is_multichar spec in
  let lexbuf = Lexing.from_string spec in
  try
    Parser.axis_labels_spec (Lexer.token multichar) lexbuf
  with
  | Lexer.Syntax_error msg ->
      raise (Parse_error ("Lexer error: " ^ msg))
  | Parser.Error ->
      let pos = lexbuf.Lexing.lex_curr_p in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      raise (Parse_error (Printf.sprintf "Parse error at line %d, column %d in spec: %s" line col spec))

(* Parse einsum specification *)
let einsum_of_spec spec =
  let multichar = is_multichar spec in
  let lexbuf = Lexing.from_string spec in
  try
    Parser.einsum_spec (Lexer.token multichar) lexbuf
  with
  | Lexer.Syntax_error msg ->
      raise (Parse_error ("Lexer error: " ^ msg))
  | Parser.Error ->
      let pos = lexbuf.Lexing.lex_curr_p in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      raise (Parse_error (Printf.sprintf "Parse error at line %d, column %d in spec: %s" line col spec))
