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

(* In multichar mode axes are separated by commas, and no axis spec can directly follow another: an
   identifier, integer or [_] followed, across whitespace only, by another one never parses there.
   Writing [s a] for two axes is the single-char habit carried into a spec that a [^] (or another
   trigger) switched to multichar mode (gh-ocannl-1454). Returns [spec] with each such whitespace
   gap replaced by [", "], if it has any. *)
let whitespace_separated_axes spec =
  let n = String.length spec in
  let is_axis_char c = Char.is_alphanum c || Char.equal c '_' in
  let buf = Buffer.create (n + 8) in
  let found = ref false in
  let rec go i =
    if i < n then
      if Char.is_whitespace spec.[i] && i > 0 && is_axis_char spec.[i - 1] then (
        let rec gap_end j = if j < n && Char.is_whitespace spec.[j] then gap_end (j + 1) else j in
        let j = gap_end i in
        if j < n && is_axis_char spec.[j] then (
          found := true;
          Buffer.add_string buf ", ")
        else Buffer.add_string buf (String.sub spec ~pos:i ~len:(j - i));
        go j)
      else (
        Buffer.add_char buf spec.[i];
        go (i + 1))
  in
  go 0;
  if !found then Some (Buffer.contents buf) else None

(* The single-char lexer buffers the COMMA it emits after an axis; a parse that failed between the
   two tokens leaves it behind, so every parse starts from an empty buffer. *)
let run entry spec lexbuf =
  Lexer.buffered_token := None;
  entry (Lexer.token (is_multichar spec)) lexbuf

let parse entry spec =
  let multichar = is_multichar spec in
  let lexbuf = Lexing.from_string spec in
  try run entry spec lexbuf with
  | Lexer.Syntax_error msg -> raise (Parse_error ("Lexer error: " ^ msg))
  | Parser.Error ->
      let pos = lexbuf.Lexing.lex_curr_p in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      let row_variable_hint =
        match misspelled_row_variable spec with
        | None -> ""
        | Some name ->
            (* In single-char mode a multi-letter name lexes as one axis per letter, so the fix must
               also switch the spec to multichar mode: the comma in [..name,..] does. *)
            let multichar_note =
              if multichar || String.length name = 1 then ""
              else
                Printf.sprintf
                  "; a multi-letter name needs multichar mode: write `..%s,..` and separate the \
                   other axes with commas"
                  name
            in
            Printf.sprintf
              "; hint: `...` is the unnamed context ellipsis; a named row variable is written \
               `..%s..`, not `...%s..`%s"
              name name multichar_note
      in
      (* The rewrite is offered only when it parses: a gap the scan reads as two axes can sit where
         a comma is no fix either, e.g. inside a row variable's [..batch shard..]. *)
      let parses fixed =
        match run entry fixed (Lexing.from_string fixed) with _ -> true | exception _ -> false
      in
      let separator_hint =
        match (Lexer.multichar_trigger spec, whitespace_separated_axes spec) with
        | Some trigger, Some fixed when parses fixed ->
            Printf.sprintf
              "; hint: the `%c` puts this spec in multichar mode, where axes are separated by \
               commas, not whitespace: write `%s`"
              trigger fixed
        | _ -> ""
      in
      let hint = row_variable_hint ^ separator_hint in
      raise
        (Parse_error
           (Printf.sprintf "Parse error at line %d, column %d in spec: %s%s" line col spec hint))

(* Parse axis labels specification *)
let axis_labels_of_spec spec = parse Parser.axis_labels_spec spec

(* Parse einsum specification *)
let einsum_of_spec spec = parse Parser.einsum_spec spec
