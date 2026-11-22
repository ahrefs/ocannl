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

(* Single-character mode reads one identifier or a whole fixed index at a time. *)
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
  | digit+ as n      { INT (int_of_string n) }
  | '_'              { IDENT "_" }
  | alpha as c       { IDENT (String.make 1 c) }
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

(* A single-char parser accepts trailing commas, so insert one after every identifier or fixed
   index. Each parse owns its pending separator, including the probes run after a parse error. *)
let token = function
  | true -> multichar_token
  | false ->
      let pending_comma = ref false in
      fun lexbuf ->
        if !pending_comma then (
          pending_comma := false;
          COMMA)
        else
          let next = single_char_token lexbuf in
          (match next with IDENT _ | INT _ -> pending_comma := true | _ -> ());
          next
}
