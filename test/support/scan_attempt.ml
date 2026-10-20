(** Telling a source scanner's failures apart: a source that does not parse is the INPUT's fault,
    any other exception is the SCANNER's -- an [Invalid_argument] from a precondition, a [_exn]
    whose invariant a new idiom broke -- and calling that "does not parse" sends the reader to a
    file that is fine. Shared by the repository scans and their case files, whose catch-alls
    otherwise label every exception a parse failure. *)

open Base

(** What became of one source handed to a scan: [Unparsed] carries the parser's own report on one
    line -- where, by line and columns, and what -- since the exception itself renders as nothing a
    reader can use (["Syntaxerr.Error(_)"], gh-ocannl-1264); [Raised] the exception's text. *)
type 'a t = Scanned of 'a | Unparsed of string | Raised of string

(** [f ()], classified. The parser's own errors -- syntax and lexical alike -- are the exceptions
    the compiler registers a located report for, so that registration is the test rather than a list
    of the parser's exception constructors, which ppxlib does not re-export. A raised exception is
    rendered with [Stdlib.Printexc.to_string], which stays on one line (Base's [Exn.to_string] does
    not). *)
let run f =
  match f () with
  | result -> Scanned result
  | exception exn -> (
      match Ppxlib.Location.Error.of_exn exn with
      | Some error ->
          let loc = Ppxlib.Location.Error.get_location error in
          let column (p : Lexing.position) = p.pos_cnum - p.pos_bol in
          (* The compiler's own spelling: a range on one line is that line's characters, one that
             spans lines names both, the end column relative to the end line (review round 7). *)
          let where =
            if loc.loc_start.pos_lnum = loc.loc_end.pos_lnum then
              Printf.sprintf "line %d, characters %d-%d" loc.loc_start.pos_lnum
                (column loc.loc_start) (column loc.loc_end)
            else
              Printf.sprintf "lines %d-%d, characters %d-%d" loc.loc_start.pos_lnum
                loc.loc_end.pos_lnum (column loc.loc_start) (column loc.loc_end)
          in
          Unparsed
            (Printf.sprintf "%s: %s" where
               (String.concat ~sep:" " (String.split_lines (Ppxlib.Location.Error.message error))))
      | None -> Raised (Stdlib.Printexc.to_string exn))

(** For a case file: [f ()], or [default] after a {!Verdict.fail} naming which way the scan failed
    on the case's snippet -- a snippet that does not parse is the case's own defect, an exception
    from one that does is the scanner's, reported with its text. [what] names the family of cases
    and [name] the case. *)
let attempted ~what ~name ~default f =
  match run f with
  | Scanned found -> found
  | Unparsed _ ->
      Verdict.fail (Printf.sprintf "%s -- %s: the snippet does not parse" what name);
      default
  | Raised exn ->
      Verdict.fail
        (Printf.sprintf "%s -- %s: the scan raised %s on a snippet that parses" what name exn);
      default
