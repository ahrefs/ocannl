{
(** Lexer for einsum notation.

    Mode selection:
    - Multichar mode: triggered by presence of ',', '*', '+', '^', '&', '<', or a '=' that does
      not begin "=>"; axes are then separated by commas, never by whitespace
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
  | '&'              { raise (Syntax_error "'&' is reserved for future use") }
  | '_'              { UNDERSCORE }
  | "..."            { ELLIPSIS }
  | ".."             { DOT_DOT }
  | '='              { EQUALS }
  | '<'              { LESS_THAN }
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
  | '='              { EQUALS }
  | '<'              { LESS_THAN }
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
(* The first character that puts [spec] in multichar mode, if any: one of [, * + ^ & <], or a
   ['='] that does not begin ["=>"]. *)
let multichar_trigger spec =
  let len = String.length spec in
  let rec go i =
    if i >= len then None
    else
      match spec.[i] with
      | (',' | '*' | '+' | '^' | '&' | '<') as c -> Some c
      | '=' when i + 1 >= len || spec.[i + 1] <> '>' -> Some '='
      | _ -> go (i + 1)
  in
  go 0

(* Determine if input uses multichar mode *)
let is_multichar spec = Option.is_some (multichar_trigger spec)

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
