{
(** Lexer for einsum notation.

    Mode selection:
    - Multichar mode: triggered by presence of ',', '*', '+', '^', '&'
    - Single-char mode: only alphanumeric, whitespace, '.', '|', '->', '=>', ';', '_'

    In single-char mode, COMMA is automatically inserted after each single-char identifier.
*)

open Parser

exception Syntax_error of string

(* State for single-char mode: buffer for next token *)
let buffered_token : token option ref = ref None

}

(* Character classes *)
let white = [' ' '\t' '\r' '\n']
let alpha = ['a'-'z' 'A'-'Z']
let digit = ['0'-'9']
let alphanum = alpha | digit | '_'

(* Main lexer for multicharacter mode *)
rule multichar_token = parse
  | white+           { multichar_token lexbuf }
  | ','              { COMMA }
  | '|'              { PIPE }
  | "->"             { ARROW }
  | "=>"             { DOUBLE_ARROW }
  | ';'              { SEMICOLON }
  | '+'              { PLUS }
  | '*'              { STAR }
  | '^'              { CARET }
  | '&'              { AMPERSAND }
  | '_'              { UNDERSCORE }
  | "..."            { ELLIPSIS }
  | ".."             { DOT_DOT }
  | digit+ as n      { INT (int_of_string n) }
  | alpha alphanum* as id
                     { IDENT id }
  | eof              { EOF }
  | _ as c           { raise (Syntax_error ("Unexpected character in multichar mode: " ^ String.make 1 c)) }

(* Single-character mode lexer - always emits COMMA after each identifier or fixed index *)
and single_char_token = parse
  | white+           { single_char_token lexbuf }
  | '|'              { PIPE }
  | "->"             { ARROW }
  | "=>"             { DOUBLE_ARROW }
  | ';'              { SEMICOLON }
  | "..."            { ELLIPSIS }
  | ".."             { DOT_DOT }
  | digit+ as n      {
      (* Always buffer COMMA after fixed index - parser handles trailing commas *)
      buffered_token := Some COMMA;
      INT (int_of_string n)
    }
  | '_'              {
      (* Always buffer COMMA after identifier - parser handles trailing commas *)
      buffered_token := Some COMMA;
      IDENT "_"
    }
  | alpha as c       {
      (* Always buffer COMMA after identifier - parser handles trailing commas *)
      buffered_token := Some COMMA;
      IDENT (String.make 1 c)
    }
  | eof              { EOF }
  | _ as c           { raise (Syntax_error ("Unexpected character in single-char mode: " ^ String.make 1 c)) }

{
(* Determine if input uses multichar mode *)
let is_multichar spec =
  let has_multichar_trigger = ref false in
  String.iter (fun c ->
    match c with
    | ',' | '*' | '+' | '^' | '&' -> has_multichar_trigger := true
    | _ -> ()
  ) spec;
  !has_multichar_trigger

(* Main entry point that switches between modes and handles buffered tokens *)
let token multichar lexbuf =
  if multichar then
    (* Multichar mode: no buffering, just return tokens directly *)
    multichar_token lexbuf
  else
    (* Single-char mode: check for buffered tokens *)
    match !buffered_token with
    | Some tok ->
        buffered_token := None;
        tok
    | None ->
        single_char_token lexbuf
}
