(** The prose of an OCaml source: its comments and its string literals, read by the compiler's own
    lexer (gh-ocannl-1729).

    A scan that reads what a source SAYS, rather than what it computes, needs the comments, and the
    parse tree has none: a plain [(* … *)] is not a node, and only a documentation comment survives
    as an attribute. So this reads the token stream instead, with the lexer the compiler uses, which
    is what keeps the classic approximations out — a quoted string inside a comment, a character
    literal of a double quote opening nothing, a quoted-string literal holding a comment's closing
    delimiter (the reasons {!Config_key_scan}'s header gives for parsing rather than matching text).
    The lexer records every comment it skips, documentation comments included, and hands every
    string literal over as a token carrying its DECODED value.

    Its own library because it reads through [compiler-libs], whose top-level [Location] and
    [Parser] are not ppxlib's: the scans in [test_scan] parse through ppxlib, and keeping the two in
    separate compilation units keeps each file's spelling unambiguous (the reason [emitter_frontier]
    stands apart too). The interface is plain data, so a consumer never sees either. *)

open Base

type kind = Comment | String_literal

type t = {
  kind : kind;
  text : string;
      (** A comment's text between its delimiters, verbatim; a string literal's decoded value. *)
  start : int;  (** Byte offset of the lexeme's first character in the source. *)
  line : int;  (** 1-based line on which the lexeme starts. *)
}

let of_location kind text (loc : Location.t) =
  { kind; text; start = loc.loc_start.pos_cnum; line = loc.loc_start.pos_lnum }

(** [of_source content] is every comment and string literal of [content], in source order. It raises
    the lexer's own error where [content] does not lex, which [Scan_attempt.run] reports as the
    input's fault with its location. Lexing is all it does: a source that lexes but does not parse
    still yields its prose. *)
let of_source content =
  let lexbuf = Lexing.from_string content in
  Lexer.init ();
  let strings = ref [] in
  let rec loop () =
    match Lexer.token lexbuf with
    | Parser.EOF -> ()
    | Parser.STRING (text, loc, _)
    | Parser.QUOTED_STRING_EXPR (_, _, text, loc, _)
    | Parser.QUOTED_STRING_ITEM (_, _, text, loc, _) ->
        strings := of_location String_literal text loc :: !strings;
        loop ()
    | _ -> loop ()
  in
  loop ();
  let comments =
    List.map (Lexer.comments ()) ~f:(fun (text, loc) -> of_location Comment text loc)
  in
  List.sort (comments @ !strings) ~compare:(fun a b -> Int.compare a.start b.start)
