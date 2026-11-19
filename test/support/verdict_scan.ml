(** Finding self-decided verdicts that a test prints itself instead of routing through {!Verdict}.

    gh-ocannl-601 established the rule and swept the sites that existed then: a claim the test
    decides — [<label>: <bool>] — must go through [Verdict], so that a failure exits the process
    nonzero. Printed and exited 0, the same line is gated only by the golden diff, and the natural
    next move on a failing diff is [dune promote], which records [<label>: false] as the expected
    output. In a golden that is nothing but verdict lines, a blessed regression and a deliberately
    recorded negative fact are the same text.

    That sweep was one-time and left no mechanical trace, so the population regrew: a test written
    afterwards ([bandwidth_calibration], gh-ocannl-578) came in with four fresh
    [Stdio.printf "…: %b\n"] claims, and nothing failed or warned. This module is the reader behind
    the ratchet that stops that (gh-ocannl-668).

    {1 What shape is recognised}

    A format whose LAST argument-consuming conversion is a bare [%b] at the end, behind a label that
    ends in one of the separators a claim is written with — a colon, an equals sign, or an arrow.
    Two kinds, told apart by whether anything else in the format consumes an argument:

    - {!Literal_label}: the [%b] is the only conversion, so the label is written out.
      ["k-blocks fused: %b\n"], ["round-trip identity = %b\n"], ["hoisted -> %b\n"].
    - {!Computed_label}: the label is built from arguments. ["%s fused: %b\n"],
      ["Epoch %d, loss below threshold=%b\n"].

    Both are recognised now. The literal form was gh-ocannl-668's; the computed one is
    gh-ocannl-624, which the sweep of that issue converted onto {!Verdict.pf} and {!Verdict.claimf}
    — entry points that did not exist when this reader was written, which is why the shape was out
    of scope then and is not now. The separator vocabulary is the other thing that widened: a reader
    that accepted only a colon was blind to the whole ["… = %b"] population, and that is how the
    claims in [data_parallel], [shard_transfer] and [test_buffer_loc] sat outside a check written to
    catch exactly them.

    The escape hatch is narrower as a result, and deliberately so. A descriptive print used to
    escape by carrying a second conversion; a computed label carries one by construction, so that no
    longer distinguishes it. What is left is a named exemption in the test that consumes this — one
    per site, with the reason it is not an assertion. The population that needs one is small (a
    handful of census rows and tables), because a print whose boolean is not a verdict usually does
    not end on the boolean.

    {1 Why it reads the parse tree}

    Same argument as {!Config_key_scan}, whose parse helpers this shares, and the same conclusion:
    this module parses, it does not match text. A text scan reads prose as code — this file's own
    documentation quotes the shape it hunts — and cannot tell a format written over a line
    continuation from one written on a line. On the parse tree a string literal carries its decoded
    value whatever escapes or delimiters produced it.

    Attribute payloads are the one thing deliberately skipped. A documentation comment becomes an
    [@@@ocaml.doc] attribute holding a string, so a comment quoting the shape arrives as a literal
    to flag; that is prose, and prose is not a print. Extension payloads are {e not} skipped — a
    [%cd] or [%expect] payload is code, or is the golden text of one.

    {1 Every claim-shaped literal, not only those applied to a [printf]}

    What the check fails on is the literal, wherever it sits; which function receives it is reported
    but decides nothing. The alternative — flag only what is applied to a printing function — is
    evaded by any wrapper, including the [let p name b = Stdio.printf "%s: %b\n" name b] helpers
    several tests defined before {!Verdict} existed. Those helpers are the exact population this
    ratchet exists to keep from regrowing, so a claim-shaped format string in a test source is
    either an assertion or, by exemption, a documented exception.

    {1 Success trailers (gh-ocannl-1449)}

    The trailer reader reserves a literal line starting with [OK: ] for {!Verdict.success_trailer}.
    It checks decoded string literals passed as the format or text argument of a direct stdout call
    spelled [printf] (bare, [Stdio.printf], or qualified by [Printf] or [Format] with optional
    [Stdlib]), or [print_string] and [print_endline] (bare or qualified by [Stdlib] or [Stdio]). It
    also reads [fprintf] (bare, [Stdio.Out_channel.fprintf], or qualified by [Printf] with optional
    [Stdlib]) and [output_string] (bare, [Stdlib.output_string], or qualified by [Out_channel],
    [Stdlib.Out_channel] or [Stdio.Out_channel]) with a direct [stdout] channel path. The channel
    may be bare or qualified by [Stdlib], [Stdio], or those same [Out_channel] paths. A line starts
    at the literal's beginning or after a decoded newline. The spelling controls the check even if
    shadowed. Channel and printer aliases, computed strings, format-generated line breaks, and
    non-OCaml output are outside the model. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse

(* The parse helpers are Config_key_scan's: one place in this repository decides how an OCaml source
   is read for a scan, and it carries the note on why that reading goes through ppxlib's parse tree
   rather than the compiler's. *)
module Read = Config_key_scan

type directive = { start : int; stop : int; conversion : char }
(** A conversion directive as the format spells it: where the ['%'] is, where the conversion
    character is, and what it is. A bare [%b] has [stop = start + 1]; anything wider carries flags,
    a width or a precision between the two. *)

(** Directives that consume no argument, so a format carrying one still takes however many arguments
    it took: [%%] (a literal per cent), [%!] (flush), [%,] (a separator that prints nothing).
    Everything else counts as consuming — including the nested-format spellings [%\{…%\}] and
    [%(…%)], whose presence therefore takes a format {e out} of the recognised shape rather than
    into it. Erring that way costs a report that was never made, never a false one. *)
let consumes_nothing = function '%' | '!' | ',' -> true | _ -> false

(** The conversion directives of a format string, in order. Scans for ['%'], then flags, width and
    precision, then the conversion character. A trailing ['%'] with nothing after it is not a
    directive and ends the scan. *)
let directives format =
  let length = String.length format in
  let rec skip_modifiers index =
    if index >= length then index
    else
      match format.[index] with
      | '-' | '0' | '+' | ' ' | '#' | '.' | '*' -> skip_modifiers (index + 1)
      | character when Char.is_digit character -> skip_modifiers (index + 1)
      | _ -> index
  in
  let rec scan position found =
    if position >= length then List.rev found
    else if not (Char.equal format.[position] '%') then scan (position + 1) found
    else
      let stop = skip_modifiers (position + 1) in
      if stop >= length then List.rev found
      else scan (stop + 1) ({ start = position; stop; conversion = format.[stop] } :: found)
  in
  scan 0 []

(** What a reader sees printed by a stretch of format text, with the arguments left out: [%%] is a
    per cent sign, every other directive contributes nothing this reader can know, and the rest is
    itself. Applied to the text before and after the [%b].

    It removes each directive by the SPAN {!directives} found, flags and width included, rather than
    by a fixed two characters — otherwise [%-22s] leaves ["22s"] behind and a claim's rendered label
    reads as that debris. The span is also what keeps the tail check honest, since a directive there
    prints nothing whatever its width. *)
let printed_text text =
  let buffer = Buffer.create (String.length text) in
  let cursor = ref 0 in
  List.iter (directives text) ~f:(fun directive ->
      Buffer.add_string buffer (String.sub text ~pos:!cursor ~len:(directive.start - !cursor));
      if Char.equal directive.conversion '%' then Buffer.add_char buffer '%';
      cursor := directive.stop + 1);
  Buffer.add_string buffer (String.subo text ~pos:!cursor);
  Buffer.contents buffer

type kind =
  | Literal_label  (** The [%b] is the format's only conversion: the label is written out. *)
  | Computed_label  (** Something else in the format consumes an argument: the label is built. *)

(** The separators a claim is written with, longest first so that a head ending in ["->"] is not
    read as ending in ["-"] by a shorter match. A colon is what the gh-ocannl-601 sweep normalised
    to; the other two are what the sites it could not convert used, and a reader blind to them is
    blind to most of the population. *)
let separators = [ "->"; ":"; "=" ]

(** {!claim_of} together with the verbatim head, which is what names a computed site. *)
let claim_site format =
  let consuming =
    List.filter (directives format) ~f:(fun d -> not (consumes_nothing d.conversion))
  in
  match List.last consuming with
  | Some { start; stop; conversion = 'b' } when stop = start + 1 ->
      let tail = printed_text (String.subo format ~pos:(stop + 1)) in
      if not (String.for_all tail ~f:Char.is_whitespace) then None
      else
        let verbatim = String.sub format ~pos:0 ~len:start in
        let head = String.rstrip (printed_text verbatim) in
        let kind = if List.length consuming = 1 then Literal_label else Computed_label in
        Option.bind
          (List.find_map separators ~f:(fun sep ->
               Option.map (String.chop_suffix head ~suffix:sep) ~f:(fun label ->
                   (sep, String.strip label))))
          ~f:(fun (separator, label) ->
            if not (String.is_empty label) then Some (label, kind, verbatim)
            else
              match kind with
              (* Nothing before the separator and nothing to build it from: [": %b"] names no
                 fact. *)
              | Literal_label -> None
              (* The label is built ENTIRELY from arguments — [Stdio.printf "%s: %b\n" name b],
                 which is not an edge case but THE shape: it is the wrapper several tests defined
                 before [Verdict] existed, the one gh-ocannl-668 was written to keep from regrowing,
                 and the one this reader would have let straight back in by treating its empty
                 residual as "no label". A residual is empty here because every character of the
                 label was a conversion, which makes the claim more computed, not less. Reported
                 under the verbatim head, which is also what names it in an exemption. *)
              | Computed_label ->
                  let shown = String.rstrip verbatim in
                  let shown =
                    Option.value (String.chop_suffix shown ~suffix:separator) ~default:shown
                  in
                  let shown = String.strip shown in
                  Some ((if String.is_empty shown then "<computed>" else shown), kind, verbatim))
  | _ -> None

(** [claim_of format] is the label of the claim [format] prints and which kind it is, when it prints
    one.

    Three things must hold at once:

    - the LAST argument-consuming conversion is a bare [%b] — a width or a flag means the print is
      laying out a column, which is formatting rather than asserting;
    - what follows it prints as whitespace, or as nothing: a newline, a blank line, a [%!] flush;
    - what precedes it ends in one of {!separators}, with a non-empty label before it.

    So ["k-blocks fused: %b\n"] yields [Some ("k-blocks fused", Literal_label)] and
    ["%s fused: %b\n"] yields [Some ("fused", Computed_label)] — the second's label is what survives
    rendering the head, which drops the conversions it cannot fill in, so it is a report's hint
    rather than the site's identity; an exemption for a computed site is keyed by the head.

    A computed label may render to NOTHING and still be a claim: ["%s: %b\n"] is the wrapper the
    pre-[Verdict] tests defined for themselves, and the whole of its label is the argument. Only a
    LITERAL label has to be non-empty, because there the residual is all there was.
    ["fused: %b (expect false)\n"] and ["fused? %b\n"] yield [None]: neither is the bare claim form,
    and a reader cannot take their boolean at face value. *)
let claim_of format = Option.map (claim_site format) ~f:(fun (label, kind, _) -> (label, kind))

(** {!claim_of} without the kind, for a caller that only wants to know what the line asserts. *)
let claim_label format = Option.map (claim_of format) ~f:fst

type site = {
  label : string;  (** The claim as the format spells it, without the separator. *)
  kind : kind;  (** Whether the label is written out or built from arguments. *)
  format : string;  (** The whole format, so a report can show what is written. *)
  head : string;
      (** The format up to (not including) the [%b], VERBATIM — conversions unrendered. What names a
          computed site: its {!label} has had the conversions it cannot fill in dropped, so two
          different formats can render the same label, and a head cannot. Deliberately not
          claim-shaped itself: it stops before the boolean, so a list of heads written out in a test
          source is not a list of claims, which is what lets the check that consumes them hold
          itself to its own rule instead of exempting its own file. *)
  line : int;  (** 1-based, as the parser located the literal. *)
  column : int;  (** 0-based, retained with {!line} for an actionable report. *)
  position : int;
      (** Absolute character offset. Unlike a line number, this uniquely identifies two literals
          written on the same physical line. *)
  printer : string option;
      (** The function the literal is an argument of, where it is one: ["Stdio.printf"],
          ["Printf.eprintf"]. [None] for a literal bound to a name, handed to a wrapper or built
          into a list — flagged just the same; this only says what the report can show. *)
}

type scan = {
  sites : site list;  (** The claim-shaped literals, in source order. *)
  hand_rolled_trailers : (int * string) list;  (** Literal stdout trailers by line and printer. *)
  stdout_literals : int;  (** Literal text or formats read for a trailer. *)
  literals : int;  (** Every string literal in expression position. *)
  applied_literals : int;  (** How many of those are an argument of a named function. *)
}
(** The counts let the ratchet require that each reader saw input in the real corpus even when it
    found no offenders. *)

let stdout_text path (arguments : (Ppxlib.Asttypes.arg_label * expression) list) =
  match (path, arguments) with
  | ( ( [ "printf" ]
      | [ ("Printf" | "Stdio" | "Format"); "printf" ]
      | [ "Stdlib"; ("Printf" | "Format"); "printf" ]
      | [ ("print_string" | "print_endline") ]
      | [ ("Stdlib" | "Stdio"); ("print_string" | "print_endline") ] ),
      (Nolabel, text) :: _ ) ->
      Some text
  | ( ( [ "fprintf" ]
      | [ "Printf"; "fprintf" ]
      | [ "Stdlib"; "Printf"; "fprintf" ]
      | [ "Stdio"; "Out_channel"; "fprintf" ]
      | [ "output_string" ]
      | [ ("Stdlib" | "Out_channel"); "output_string" ]
      | [ ("Stdlib" | "Stdio"); "Out_channel"; "output_string" ] ),
      (Nolabel, channel) :: (Nolabel, text) :: _ ) -> (
      match Read.longident_of channel with
      | Some
          ( [ "stdout" ]
          | [ ("Stdlib" | "Stdio" | "Out_channel"); "stdout" ]
          | [ ("Stdlib" | "Stdio"); "Out_channel"; "stdout" ] ) ->
          Some text
      | _ -> None)
  | _ -> None

let is_trailer_literal text =
  List.exists (String.split ~on:'\n' text)
    ~f:(String.is_prefix ~prefix:Verdict.success_trailer_prefix)

(** [scan content] reads one source. Raises if [content] does not parse: a reader that cannot read
    its input must say so rather than report an empty census. *)
let scan content =
  let ast = Read.structure_of content in
  (* The function each literal is an argument of, noted as the application is entered — an iterator
     visits a node before its children, so the argument's own visit below finds it. *)
  let printers = Hashtbl.create (module Int) in
  let sites = ref [] and literals = ref 0 and hand_rolled_trailers = ref [] in
  let stdout_literals = ref 0 in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      (* Prose is not a print: see the note on attribute payloads above. *)
      method! attribute _ = ()

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_apply (callee, arguments) -> (
            match Read.longident_of callee with
            | Some path ->
                let name = String.concat ~sep:"." path in
                Option.iter (stdout_text path arguments) ~f:(fun text ->
                    Option.iter (Read.string_literal text) ~f:(fun value ->
                        Int.incr stdout_literals;
                        if is_trailer_literal value then
                          hand_rolled_trailers :=
                            (text.pexp_loc.loc_start.pos_lnum, name) :: !hand_rolled_trailers));
                List.iter arguments ~f:(fun (_, argument) ->
                    if Option.is_some (Read.string_literal argument) then
                      Hashtbl.set printers ~key:argument.pexp_loc.loc_start.pos_cnum ~data:name)
            | None -> ())
        | _ -> ());
        (match Read.string_literal expr with
        | Some value -> (
            Int.incr literals;
            match claim_site value with
            | Some (label, kind, head) ->
                let start = expr.pexp_loc.loc_start in
                sites :=
                  {
                    label;
                    kind;
                    head;
                    format = value;
                    line = start.pos_lnum;
                    column = start.pos_cnum - start.pos_bol;
                    position = start.pos_cnum;
                    printer = Hashtbl.find printers start.pos_cnum;
                  }
                  :: !sites
            | None -> ())
        | None -> ());
        super#expression expr
    end
  in
  iterator#structure ast;
  {
    sites = List.rev !sites;
    hand_rolled_trailers = List.rev !hand_rolled_trailers;
    stdout_literals = !stdout_literals;
    literals = !literals;
    applied_literals = Hashtbl.length printers;
  }

(** {1 A gated claim reported in two dialects (gh-ocannl-997)}

    [Verdict.skipped] prints [<label>: true] — {!Verdict.p}'s line — so that a leg one host cannot
    evaluate leaves the golden exactly as a host that evaluated it does. A claim whose evaluated
    branch reports through [Verdict.pass_fail] instead prints [<label>: PASS] there, and the golden
    promoted on a machine that evaluates it breaks on the first machine that skips: silently, until
    the gate actually fires, on whichever host is least likely to be the one promoting.
    [Verdict.gated] removes the trap by picking the dialect itself; this reader keeps the mixed
    spelling from regrowing.

    What it pairs is a LABEL that reaches both a PASS/FAIL claim and a [skipped] call in one source.
    The dialect includes the [Verdict.Claims] names [pass_fail] and [pass_fail_*]. The label is the
    first positional argument, keyed three ways: a string literal by its value; a name by the
    literal it is bound to, in its lexical scope through [Lexical_scope] (the [let claim = "…"]
    shared by the two branches is the shape both sightings had), otherwise by the name itself;
    anything else by its printed expression, so a label computed the same way on both sides is still
    one label. A literal pairs across the whole source; a name or an expression pairs only within
    one top-level item, since equal names in two functions need not be one label. Named and computed
    keys retain their bindings' identities. The callees are matched by their last path component,
    which is how an opened [Verdict.Claims] and the per-test
    [let skipped = Verdict.skipped ~backend:backend_name] both read.

    Fail-loud boundary: unknown or shadowed labels stay named/computed rather than borrowing a
    literal elsewhere. Unknown opens bring in no values.

    What it cannot see: a skip or a PASS/FAIL claim routed through a wrapper of another name, and
    one label computed by two different expressions. Those are the reason the remedy is an entry
    point rather than this check — [Verdict.gated] has no second dialect to pair. *)

type label_key =
  | Literal of string
  | Named of string * int
  | Computed of string * (string * (string option * int)) list

let label_key_text = function Literal text | Named (text, _) | Computed (text, _) -> text

type dialect_site = { callee : string; key : label_key; item : int; line : int }
(** One claim call: the callee as written, the label it reports, the top-level structure item it
    sits in (identified by its source position), and its line. *)

type dialect_pairing = { pass_fail : dialect_site; skipped : dialect_site }
(** A [pass_fail]-family call and a [skipped] call reporting the same label. *)

type dialect_census = {
  pairings : dialect_pairing list;
  skipped_sites : int;  (** Every [skipped] call whose label was read. *)
  pass_fail_sites : int;  (** Every PASS/FAIL claim whose label was read. *)
}
(** The site counts are what a blind walk cannot produce, as {!scan}'s literal counts are for the
    claim-shape scan: they let the consuming test tell "no pairing" from "nothing read". *)

let pass_fail_callees =
  List.filter Verdict_callee.members ~f:(fun member ->
      String.equal member "pass_fail" || String.is_prefix member ~prefix:"pass_fail_")

let skipped_callees = [ "skipped" ]

(** Whose a claim callee is, as {!Verdict_callee} resolves it: [Verdict]'s, [Ambiguous] -- reaching
    [Verdict] only past an [open] of a module that resolver cannot read, which may export a name or
    a [Verdict] module of its own -- [Free], or not [Verdict]'s. *)
type ownership = Verdict_callee.ownership = Verdicts | Ambiguous | Free | Not_verdicts

(* One application whose callee is a path ending in a name of [callees], with a positional argument:
   its last path component, whether that callee is [Verdict]'s (below), the site its first
   positional argument -- the label -- keys, and the arguments as written. Shared by
   {!dialect_census} and {!line_break_census}, so both key a label one way.

   Whether the callee is [Verdict]'s is {!Verdict_callee}'s answer, the resolver every Verdict
   reader shares; its header lists what it reads as [Verdict]'s and why a name past an [open] of
   another module is {!Ambiguous}. *)
let claim_calls ~callees content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  (* A type constraint or coercion does not change which string a label is: [("…" : string)] is the
     literal it wraps. *)
  let rec peel expr =
    match expr.pexp_desc with
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel inner
    | _ -> expr
  in
  let key_of env argument =
    let argument = peel argument in
    match Read.string_literal argument with
    | Some literal -> Literal literal
    | None -> (
        match argument.pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some (Some literal, _) -> Literal literal
            | Some (None, identity) -> Named (name, identity)
            | None -> Named (name, 0))
        | _ ->
            let references = ref [] in
            let reader =
              object
                inherit [string option * int, bool] Lexical_scope.scoped
                method local = (None, 0)
                method module_path _ _ = None
                method! attribute _ attr = attr

                method! ident env ident =
                  match ident.txt with
                  | Ppxlib.Longident.Lident name -> (
                      match Lexical_scope.lookup env name with
                      | Some (None, 0) ->
                          () (* Bound inside this expression, whose text already identifies it. *)
                      | bound ->
                          let denotation =
                            match bound with
                            | Some (Some literal, _) -> (Some literal, 0)
                            | other -> Option.value other ~default:(None, 0)
                          in
                          references := (name, denotation) :: !references)
                  | _ -> ()
              end
            in
            ignore (reader#expression env argument : expression);
            Computed (Ppxlib.Pprintast.string_of_expression argument, List.rev !references))
  in
  let calls = ref [] and item = ref 0 in
  let iterator =
    object (self)
      inherit [string option * int, bool] Lexical_scope.scoped as super
      val mutable next_binding = 0

      method local =
        next_binding <- next_binding - 1;
        (None, next_binding)

      (* Module paths matter here only to {!Verdict_callee}, which resolves the callee. *)
      method module_path _ _ = None

      (* A name bound to a literal denotes that literal, and so does a name bound to another name
         that does -- [let label = raw] -- through any chain of such renamings, each resolved in the
         scope of its own right-hand side. *)
      method! binding_denotes env binding =
        match Read.pattern_name binding.pvb_pat with
        | Some _ ->
            let rhs = peel binding.pvb_expr in
            let literal =
              match (Read.string_literal rhs, rhs.pexp_desc) with
              | (Some _ as literal), _ -> literal
              | None, Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                  match Lexical_scope.lookup env name with
                  | Some (literal, _) -> literal
                  | None -> None)
              | None, _ -> None
            in
            (literal, binding.pvb_loc.loc_start.pos_cnum + 1)
        | None -> self#local

      method! define ~top env rec_flag bindings ~walk =
        if top then
          Option.iter (List.hd bindings) ~f:(fun b -> item := b.pvb_loc.loc_start.pos_cnum);
        super#define ~top env rec_flag bindings ~walk

      method! attribute _ attr = attr

      method! structure_item env si =
        if List.exists ast ~f:(fun original -> phys_equal original si) then
          item := si.pstr_loc.loc_start.pos_cnum;
        super#structure_item env si

      method! expression env expr =
        (match expr.pexp_desc with
        | Pexp_apply (callee, arguments) -> (
            let label =
              List.find_map arguments ~f:(function
                | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                | _ -> None)
            in
            (* A constrained callee, [(Verdict.p : string -> bool -> unit) label ok], is the path it
               wraps. *)
            let callee = peel callee in
            match (callee.pexp_desc, Read.longident_of callee, label) with
            | Pexp_ident _, Some path, Some label ->
                let name = List.last_exn path in
                if List.mem callees name ~equal:String.equal then
                  let verdicts =
                    Option.value_map (Verdict_callee.resolve owners callee) ~default:Not_verdicts
                      ~f:(fun resolution -> resolution.ownership)
                  in
                  let site =
                    {
                      callee = String.concat ~sep:"." path;
                      key = key_of env label;
                      item = !item;
                      line = expr.pexp_loc.loc_start.pos_lnum;
                    }
                  in
                  calls := (name, verdicts, site, arguments) :: !calls
            | _ -> ())
        | _ -> ());
        super#expression env expr
    end
  in
  ignore
    (iterator#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  List.rev !calls

let dialect_census content =
  let calls = claim_calls ~callees:(pass_fail_callees @ skipped_callees) content in
  let sites_of callees =
    List.filter_map calls ~f:(fun (name, _, site, _) ->
        Option.some_if (List.mem callees name ~equal:String.equal) site)
  in
  let pass_fails = sites_of pass_fail_callees and skips = sites_of skipped_callees in
  (* A literal is the same label anywhere in the source. A name, or an expression over names, is
     only the same label within one top-level item: [label] is the parameter name of half the claim
     helpers in [Verdict] itself, and two functions sharing it share nothing else. *)
  let same_label a b =
    Poly.equal a.key b.key
    && match a.key with Literal _ -> true | Named _ | Computed _ -> a.item = b.item
  in
  let pairings =
    List.concat_map pass_fails ~f:(fun pass_fail ->
        List.filter_map skips ~f:(fun skipped ->
            if same_label pass_fail skipped then Some { pass_fail; skipped } else None))
  in
  { pairings; skipped_sites = List.length skips; pass_fail_sites = List.length pass_fails }

(** {1 A claim label written with a line break (post-wave task of staging#1134)}

    A claim is one row on each channel it prints to: [tools/mutation-run.sh] reads a run's
    [FAIL: …: false] lines off its log and counts its stdout rows against the golden's. A label with
    a newline in it used to split its claim into a [FAIL: ] head that is no false claim and an
    unprefixed tail that is a row the golden lacks. [Verdict] now renders every claim line through
    [Verdict.one_line] (a control character becomes a space), which keeps a COMPUTED label on one
    row; a LITERAL one with a line break can only be a mistake -- a cosmetic blank line before the
    claim, as [test_random_histograms] had, belongs in a [printf] of its own -- so this reader
    refuses it where it is written rather than leaving the sanitizer to turn it into a stray space.

    What it reads is the label argument of every claim form in {!labelled_claims} that is
    [Verdict]'s (see {!claim_calls}): the first positional argument, keyed as {!dialect_census} keys
    it, so a name bound to a literal is that literal. A [pf]/[claimf]/[pf_all2] format is its label
    too, and a line break in the format is one in every rendered label. [Verdict.fail] is not read:
    its argument is a message, multi-line by design, and every line of it is printed with the
    prefix.

    Where a label counts as LITERAL is a deliberate, closed list: a string constant, under any type
    constraints or coercions, or a name bound to one -- directly or through a chain of [let]
    renamings. Every other expression is a computed label, however statically it could be evaluated
    ([if c then "a\nb" else "c"], [String.concat]): that is the sanitizer's case, and [Verdict]
    prints it on one row whatever it holds. This reader is the lint for the label a person writes
    out; the sanitizer is what makes the run correct.

    The CALL is read the same way: a claim applied by its path, under any constraints or coercions.
    A claim applied any other way -- through [@@] or [|>], or passed as a value -- is not read; its
    label, too, is the sanitizer's.

    What it cannot see: a label computed from an expression (the sanitizer's case, by design), a
    claim reached through a wrapper the source binds ([let skipped = Verdict.skipped ~backend]), and
    an unqualified claim name brought in by an [open] this scan does not model (a module that
    re-exports [Verdict.Claims] under another name).

    A claim name that reaches [Verdict] only past an [open] of a module {!Verdict_callee} cannot
    read is {!Ambiguous}: that module may export a function of the name, or a [Verdict] module of
    its own -- [Verdict.p] past an [open Other] is ambiguous as much as an unqualified [p] past one
    (gh-ocannl-1361). Such a site is READ, and a line break in its literal label is refused with the
    remedy of qualifying the call -- loudly, since the alternative, leaving it unread, is how a
    later [open Ll_test] would silently exempt every claim after it. *)

(** The [Verdict.Claims] members whose first positional argument is a claim label (or the format
    rendering one): every member but [fail], whose argument is a multi-line message. *)
let labelled_claims =
  List.filter Verdict_callee.members ~f:(fun member -> not (String.equal member "fail"))

type line_break_site = { site : dialect_site; ambiguous : bool }
(** A claim whose literal label carries a line break; [ambiguous] when its callee reaches
    [Verdict]'s only past an [open] of another module (see {!ownership}). *)

type line_break_census = {
  broken : line_break_site list;
  label_sites : int;  (** Every [Verdict] (or ambiguous) claim whose label was read. *)
}

let has_line_break text = String.exists text ~f:(fun c -> Char.equal c '\n' || Char.equal c '\r')

let line_break_census content =
  let sites =
    claim_calls ~callees:labelled_claims content
    |> List.filter_map ~f:(fun (_, ownership, site, _) ->
        match ownership with
        | Verdicts -> Some { site; ambiguous = false }
        | Ambiguous -> Some { site; ambiguous = true }
        | Free | Not_verdicts -> None)
  in
  let broken =
    List.filter sites ~f:(fun { site; _ } ->
        match site.key with Literal text -> has_line_break text | Named _ | Computed _ -> false)
  in
  { broken; label_sites = List.length sites }

(** {1 A literal floor below one (post-1-task-4 of staging#1578)}

    A quantified claim's [?min] only RAISES its floor: [Verdict.check_floor] refuses a value below
    one with an [Invalid_argument] where the claim is evaluated, since [~min:0] reads as "emptiness
    passes" and no floor can grant that (gh-ocannl-729). That refusal fires only on a run that
    reaches the claim -- a claim in a leg whose gate is closed, or on a branch only another backend
    takes, carries its [~min:0] through every run that does not. A floor written as an integer
    LITERAL needs no run to judge, so this reader refuses it where it is written.

    What it reads is the [~min] argument of every claim of {!floor_claims} that is [Verdict]'s or
    {!Ambiguous} (resolved as {!claim_calls} resolves the callee), when that argument is an integer
    constant, under any type constraints or coercions: [~min:0], [~min:(-1)] (which the parser folds
    into one constant), [~min:0x0], [~min:0_0], each by the value [Int.of_string] gives it. A floor
    COMPUTED from anything ([~min:(List.length xs)], a name bound to a literal) is left to the
    runtime refusal, which sees its value; so is a [?min:(Some 0)] forwarding, and a claim applied
    through [|>] or [@@] or named as a value, which {!claim_calls} does not read. [Verdict.within]
    is no claim and is not read either: its floor is refused at run time, where its claim is. *)

(** The [Verdict.Claims] members whose definition takes a [?min] floor. Held equal by
    [verdict_ratchet] to the members whose definition in [test/support/verdict.ml] takes one. *)
let floor_claims =
  [
    "p_all";
    "p_all2";
    "pf_all2";
    "p_close";
    "pf_close";
    "p_none";
    "p_alli";
    "p_empty";
    "p_exists";
    "gated_all";
    "gated_alli";
    "gated_exists";
    "pass_fail_all2";
    "pass_fail_close";
  ]

(** The top-level values of a source whose definition is a function taking a [?min] -- read from
    [test/support/verdict.ml], what {!floor_claims} is held equal to. *)
let floored_definitions content =
  let rec takes_min expr =
    match expr.pexp_desc with
    | Pexp_function (params, _, body) -> (
        List.exists params ~f:(fun param ->
            match param.pparam_desc with
            | Pparam_val (Optional "min", _, _) -> true
            | Pparam_val _ | Pparam_newtype _ -> false)
        || match body with Pfunction_body inner -> takes_min inner | Pfunction_cases _ -> false)
    | Pexp_constraint (inner, _) | Pexp_newtype (_, inner) -> takes_min inner
    | _ -> false
  in
  List.concat_map (Read.structure_of content) ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.filter_map bindings ~f:(fun binding ->
              match binding.pvb_pat.ppat_desc with
              | Ppat_var { txt; _ } when takes_min binding.pvb_expr -> Some txt
              | _ -> None)
      | _ -> [])

type floor_site = {
  floor_site : dialect_site;  (** The claim, keyed by its label as {!dialect_census} keys one. *)
  floor : int;  (** The literal's value: below one. *)
  floor_ambiguous : bool;  (** The callee reaches [Verdict]'s only past another [open]. *)
}

type floor_census = {
  below_one : floor_site list;
  literal_floors : int;  (** Every [Verdict] (or ambiguous) claim whose [~min] is a literal. *)
}

let floor_census content =
  let rec peel expr =
    match expr.pexp_desc with
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel inner
    | _ -> expr
  in
  let literal_floor arguments =
    List.find_map arguments ~f:(function
      | Ppxlib.Asttypes.Labelled "min", argument -> (
          match (peel argument).pexp_desc with
          | Pexp_constant (Pconst_integer (digits, None)) ->
              Option.try_with (fun () -> Int.of_string digits)
          | _ -> None)
      | _ -> None)
  in
  let floors =
    claim_calls ~callees:floor_claims content
    |> List.filter_map ~f:(fun (_, ownership, site, arguments) ->
        let ambiguous =
          match ownership with
          | Verdicts -> Some false
          | Ambiguous -> Some true
          | Free | Not_verdicts -> None
        in
        Option.both ambiguous (literal_floor arguments)
        |> Option.map ~f:(fun (floor_ambiguous, floor) ->
            { floor_site = site; floor; floor_ambiguous }))
  in
  {
    below_one = List.filter floors ~f:(fun site -> site.floor < 1);
    literal_floors = List.length floors;
  }

(** {1 A float tolerance hand-rolled into a claim (gh-ocannl-1224, gh-ocannl-1470)}

    [Verdict.p_close name got want ~tol] compares two float readbacks under a tolerance and, when a
    cell is out of it, names on stderr how many cells differ, the worst one with both values, and
    the reference's magnitude; [Verdict.within] is the same check as the predicate of an aggregate
    claim. The same comparison written as
    [p_all2 name got want ~f:(fun a b -> Float.(abs (a - b) < 1e-3))] prints a bare [false]:
    gh-ocannl-1217's sweep recorded exactly that for fifteen seeds and it took a reproduction round
    on a remote box to learn that every seed was off by the same five ulps of reference rounding.
    This reader keeps the hand-rolled spelling from regrowing, in two shapes.

    A PARITY PREDICATE: the [~f] of every [Verdict] [p_all2], [pf_all2] and [pass_fail_all2] call
    ({!parity_callees}), resolved as {!claim_calls} resolves the callee, is a hand-rolled tolerance
    when it is a function whose body applies an [abs] (any path ending in it: [Float.abs], or [abs]
    under [Float.( … )]) to a subtraction ([-] or [-.]) -- written in the argument, or a name the
    predicate's lexical scope binds to such a function (directly, through a chain of renamings, or
    partially applied: [~f:(close ~tol:1e-4)]).

    A HELPER REACHING A CLAIM (gh-ocannl-1470): every other argument of every other [Verdict] claim
    ({!helper_claim_callees}: the label aside) is a hand-rolled tolerance when it holds a function
    that BOUNDS an abs-of-difference -- an [abs] of a subtraction on the small side of [<], [<=] (or
    the large side of [>], [>=]) inside a [fun] -- or a name its scope binds to anything holding
    one: a helper ([let approx a b = Float.(abs (a - b) < 1e-3)]) or a value computed through one
    ([let ok = Array.for_all2_exn got want ~f:approx]), through any chain of such bindings. So
    [p "x" (Array.for_all2_exn a b ~f:(fun x y -> Float.(abs (x - y) < 1e-5)))] and
    [p_all "x" runs ~f:(fun (got, want) -> approx got want)] are both read. Only a BOUND difference
    counts here. An unbounded abs-of-difference helper, or a fold containing one, is tracked through
    name bindings and arithmetic until an upper comparison bounds it (gh-ocannl-1837). Merely
    printing a maximum or requiring it to EXCEED a margin does not count. Ordering two error metrics
    is also a scalar comparison, not parity under a tolerance. The reduction reader recognizes calls
    ending in [fold], [foldi], [fold2], [fold2_exn], [fold_left], [fold_right], [max_elt], [reduce]
    and [reduce_exn], applied or piped ([x |> f], [f @@ x]), over an abs-of-difference written in
    the call or a name holding one; it does not prove which reducer the callback implements. A name
    holding an abs-of-difference that is neither a function nor a reduction -- one cell's error, or
    the errors of a collection mapped cell by cell -- is no tolerance when a bound applies to it,
    but a reduction over it is.

    LOOP-CARRIED REFS (gh-ocannl-2122): what the source writes into a [let]-bound name with [x := e]
    is read into that binding, by lexical identity (a shadowed name is another binding), through any
    [let alias = x] renaming. A write inside a [for] or [while] body or a function body the binding
    encloses (a callback of [iter]) runs once per cell: an error it carries makes the ref a
    reduction over the cells ([max_err := Float.max !max_err (Float.abs (a -. b))], or
    [worst := err] for an [err] holding one), and a bound it applies to an abs-of-difference makes
    it a tolerance ([ok := !ok && Float.(abs (a - b) < tol)], or [Float.(err < tol)] for an [err]
    holding one, where the other side holds no error). A write at the binding's own depth is one
    value, read as a binding's right-hand side is. A write is read wherever it sits, before or after
    the claim. The reader runs in passes, each reading what the previous ones found written: first
    the metrics alone, until a pass finds no new one, then the bounds over those settled metrics, so
    an ordering of two refs is never read as a bound while only one is known to hold a metric.

    What it does not read, by design: a scalar comparison written in the claim itself or in a value
    binding outside any function ([p "x" Float.(abs (got - 2.) < 1e-6)]) -- one value against one
    reference, no cell to locate (its [p_close] spelling is still the better one); the argument of
    [in_leg], [in_legf] and [case], whose closures hold claims of their own, read where they are
    written; and the [p_close] family, whose [~tol] answers a bound rather than hand-rolling one.
    What it cannot see: a predicate reaching the call as a function PARAMETER, a tolerance written
    without [abs] ([a -. b < t && b -. a < t]), a lambda parameter shadowing a tolerance name inside
    the claim's argument, and a comparison routed through a helper of another module. Of writes, it
    reads [:=] into a name alone: not a ref a function parameter holds, a record field, an array
    cell ([errs.(i) <- …]), a write a condition controls without carrying the error
    ([if Float.(abs (a - b) > tol) then ok := false]), nor an abs of a name holding a difference
    ([let d = a -. b in Float.abs d]); nor a cell-by-cell mapped error bounded element-wise
    ([List.for_all diffs ~f:(fun d -> Float.(d < tol))]), a [sum] or [min_elt]. These are reader
    boundaries, not evidence that the tree contains no such comparisons; the remedy for a tolerance
    claim using one is [p_close] or [within]. *)

let parity_callees = [ "p_all2"; "pf_all2"; "pass_fail_all2" ]

(** The claims whose non-label arguments are read for a helper hand-rolling a tolerance: every
    [Verdict.Claims] member but the parity forms (read through their [~f] above), the [p_close]
    family, [fail] and [skipped] (no value), and the leg and case wrappers (their closures' claims
    are read where they are written). *)
let helper_claim_callees =
  List.filter Verdict_callee.members ~f:(fun member ->
      not
        (List.mem
           ([
              "p_close";
              "pf_close";
              "pass_fail_close";
              "fail";
              "skipped";
              "in_leg";
              "in_legf";
              "case";
            ]
           @ parity_callees)
           member ~equal:String.equal))

type tolerance_site = {
  tolerance_callee : string;  (** The callee as written. *)
  tolerance_line : int;
  predicate : string;
      (** ["an inline function"], or the name the predicate is bound to, as the source spells it. *)
  tolerance_ambiguous : bool;  (** The callee reaches [Verdict]'s only past another [open]. *)
}

type tolerance_census = {
  hand_rolled : tolerance_site list;
  parity_sites : int;  (** Every [Verdict] (or ambiguous) claim whose arguments were read. *)
}

(* Whether [expr] applies an [abs] to a subtraction anywhere inside it. *)
let applies_abs_to_difference expr =
  let is_subtraction e =
    match e.pexp_desc with
    | Pexp_apply (op, [ (Ppxlib.Asttypes.Nolabel, _); (Ppxlib.Asttypes.Nolabel, _) ]) -> (
        match Read.longident_of op with
        | Some path -> List.mem [ "-"; "-." ] (List.last_exn path) ~equal:String.equal
        | None -> false)
    | _ -> false
  in
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_apply (callee, [ (Ppxlib.Asttypes.Nolabel, argument) ]) -> (
            match Read.longident_of callee with
            | Some path when String.equal (List.last_exn path) "abs" ->
                if is_subtraction argument then found := true
            | _ -> ())
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(* Whether [expr] holds a comparison bounding an abs-of-difference from above: the difference on the
   small side of [<]/[<=], or the large side of [>]/[>=]. *)
let bounds_abs_difference_with ?(bound = fun _ -> true) applies expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_apply (op, [ (Ppxlib.Asttypes.Nolabel, left); (Ppxlib.Asttypes.Nolabel, right) ])
          -> (
            match Option.map (Read.longident_of op) ~f:List.last_exn with
            | Some ("<" | "<=" | "<." | "<=.") when applies left && bound right -> found := true
            | Some (">" | ">=" | ">." | ">=.") when applies right && bound left -> found := true
            | _ -> ())
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(* The reductions whose result is one value over a collection: a fold (Base's and Stdlib's), a
   [reduce] and a [max_elt]. *)
let reducers =
  [
    "fold";
    "foldi";
    "fold2";
    "fold2_exn";
    "fold_left";
    "fold_right";
    "max_elt";
    "reduce";
    "reduce_exn";
  ]

(* Whether [expr] holds a reduction over what [carries] (an application whose callee ends in one of
   {!reducers}, the application and its arguments read whole), reading [x |> f] and [f @@ x] as [f
   x]. *)
let reduces_with carries expr =
  let reducer callee =
    match Read.longident_of callee with
    | Some path -> List.mem reducers (List.last_exn path) ~equal:String.equal
    | None -> false
  in
  let head f = match f.pexp_desc with Pexp_apply (callee, _) -> callee | _ -> f in
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_apply (callee, arguments) -> (
            match (Read.longident_of callee, arguments) with
            | Some [ "|>" ], [ (Ppxlib.Asttypes.Nolabel, x); (Ppxlib.Asttypes.Nolabel, f) ]
            | Some [ "@@" ], [ (Ppxlib.Asttypes.Nolabel, f); (Ppxlib.Asttypes.Nolabel, x) ] ->
                if reducer (head f) && (carries x || carries f) then found := true
            | _ -> if reducer callee && carries e then found := true)
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

let folds_abs_difference = reduces_with applies_abs_to_difference
let bounds_abs_difference = bounds_abs_difference_with applies_abs_to_difference

(* What a name denotes to this reader: a [Bearing] binding holds a function bounding an
   abs-of-difference, or reaches one through another [Bearing] name (a helper, or a value computed
   through one); a [Predicate] one is only a function applying an abs to a difference, which is a
   hand-rolled tolerance as a parity predicate, or when an upper comparison bounds it (a
   maximum-error helper). Derived error metrics retain [Predicate] through name references. A
   [Difference] one holds an abs-of-difference that is neither: one cell's error, or the errors of a
   collection mapped cell by cell, which an upper comparison bounding it does not make a tolerance
   (one value against one reference), but a reduction over it, or a loop carrying it into a [ref],
   does (gh-ocannl-2122). *)
type denotes = Bearing | Predicate | Difference | Neither

let denotes_rank = function Bearing -> 3 | Predicate -> 2 | Difference -> 1 | Neither -> 0
let join_denotes a b = if denotes_rank a >= denotes_rank b then a else b

(* A name in the tolerance reader's scope: what it denotes, and for a [let]-bound name, the [let]
   whose binding a write [x := e] reaches -- its identity and the loop depth it was made at; a
   renaming ([let alias = x]) reaches the binding [x] reaches. *)
type tolerance_name = { denotes : denotes; cell : (int * int) option }

let tolerance_census content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  let rec peel expr =
    match expr.pexp_desc with
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel inner
    | _ -> expr
  in
  let denotes_of env name =
    Option.map (Lexical_scope.lookup env name) ~f:(fun (entry : tolerance_name) -> entry.denotes)
  in
  (* Which facts the current pass derives: first the METRICS ([Predicate], [Difference]) alone, to a
     fixed point, then the BOUNDS ([Bearing]) over those metrics, frozen. A bound is read off the
     metrics -- an ordering of two metrics is none -- so one derived while a metric was still
     unknown could be wrong, and a ref's written facts only grow; with the metrics settled first,
     both stages only grow. *)
  let bounds_stage = ref false in
  (* The first name [e] spells, in reading order, that denotes [wanted] in [env]. *)
  let name_denoting wanted env e =
    let found = ref None in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ }
            when Poly.equal (denotes_of env name) (Some wanted) ->
              if Option.is_none !found then found := Some name
          | _ -> ());
          super#expression e
      end
    in
    iterator#expression e;
    !found
  in
  let predicate_name = name_denoting Predicate in
  (* Whether an expression, in [env], holds a hand-rolled tolerance a claim's value can reach --
     [None] when it does not -- and how a diagnostic names it: the first inline function bounding a
     difference, or the first [Bearing] name, in reading order. *)
  let bearing env expr =
    let found = ref None in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          if Option.is_none !found then (
            if
              bounds_abs_difference_with
                ~bound:(fun side -> Option.is_none (predicate_name env side))
                (fun side -> Option.is_some (predicate_name env side))
                e
            then found := predicate_name env e;
            (match e.pexp_desc with
            | Pexp_function _ when bounds_abs_difference e -> found := Some "an inline function"
            | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                match denotes_of env name with
                | Some Bearing -> found := Some name
                | Some (Predicate | Difference | Neither) | None -> ())
            | _ -> ());
            super#expression e)
      end
    in
    iterator#expression expr;
    !found
  in
  (* Whether an expression, in [env], is a hand-rolled tolerance PARITY predicate -- [None] when it
     is not -- and how a diagnostic names it. *)
  let tolerance env expr =
    let expr = peel expr in
    let bound name =
      match denotes_of env name with
      | Some (Bearing | Predicate) -> Some name
      | Some (Difference | Neither) | None -> None
    in
    match expr.pexp_desc with
    | Pexp_function _ when applies_abs_to_difference expr -> Some "an inline function"
    | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> bound name
    | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Ppxlib.Longident.Lident name; _ }; _ }, _) ->
        bound name
    | _ -> None
  in
  (* An abs-of-difference, written in [e] or through a [Difference] name. *)
  let differs env e =
    applies_abs_to_difference e || Option.is_some (name_denoting Difference env e)
  in
  (* What a value bound by [let], or written into a [ref] where it was bound, denotes. *)
  let classify env expr =
    if !bounds_stage && Option.is_some (bearing env expr) then Bearing
    else if
      Option.is_some (tolerance env expr)
      || Option.is_some (predicate_name env expr)
      || reduces_with (differs env) expr
    then Predicate
    else if differs env expr then Difference
    else Neither
  in
  (* What a value written into a [ref] inside a loop or a function its binding encloses denotes: the
     write runs once per iteration, so an error it carries is a reduction over the cells, and a
     bound it applies to an abs-of-difference -- written in place or through a name holding one, the
     other side holding no error -- is checked cell by cell. *)
  let carried env expr =
    let holds_error side = differs env side || Option.is_some (predicate_name env side) in
    match classify env expr with
    | Bearing -> Bearing
    | _
      when !bounds_stage
           && bounds_abs_difference_with
                ~bound:(fun side -> not (holds_error side))
                (differs env) expr ->
        Bearing
    | Predicate | Difference -> Predicate
    | Neither -> Neither
  in
  (* What the source writes into each [let]-bound [ref], by its binding's identity, read at the
     binding on the next pass: the metric each write carries, joined, and whether one is a bound. *)
  let metrics_written = Hashtbl.create (module Int) in
  let bounds_written = Hash_set.create (module Int) in
  let changed = ref false in
  let sites = ref [] and claims = ref 0 in
  let make_iterator () =
    object (self)
      inherit [tolerance_name, bool] Lexical_scope.scoped as super
      method local = { denotes = Neither; cell = None }
      method module_path _ _ = None
      method! attribute _ attr = attr

      (* The loops and function bodies the walk is inside. *)
      val mutable loop_depth = 0

      method! binding_denotes env binding =
        match Read.pattern_name binding.pvb_pat with
        | Some _ ->
            let own = (binding.pvb_loc.loc_start.pos_cnum + 1, loop_depth) in
            let cell =
              match (peel binding.pvb_expr).pexp_desc with
              | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                  match Lexical_scope.lookup env name with
                  | Some { cell = Some cell; _ } -> cell
                  | Some { cell = None; _ } | None -> own)
              | _ -> own
            in
            let from_writes =
              if Hash_set.mem bounds_written (fst cell) then Bearing
              else Option.value (Hashtbl.find metrics_written (fst cell)) ~default:Neither
            in
            { denotes = join_denotes (classify env binding.pvb_expr) from_writes; cell = Some cell }
        | None -> self#local

      method write env target value =
        match (peel target).pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some { cell = Some (identity, depth); _ } ->
                let value_denotes =
                  if loop_depth > depth then carried env value else classify env value
                in
                if !bounds_stage then (
                  if Poly.equal value_denotes Bearing && not (Hash_set.mem bounds_written identity)
                  then (
                    Hash_set.add bounds_written identity;
                    changed := true))
                else
                  let before =
                    Option.value (Hashtbl.find metrics_written identity) ~default:Neither
                  in
                  let after = join_denotes before value_denotes in
                  if not (Poly.equal before after) then (
                    Hashtbl.set metrics_written ~key:identity ~data:after;
                    changed := true)
            | Some { cell = None; _ } | None -> ())
        | _ -> ()

      method! expression env expr =
        (match expr.pexp_desc with
        | Pexp_apply (op, [ (Ppxlib.Asttypes.Nolabel, target); (Ppxlib.Asttypes.Nolabel, value) ])
          when Option.equal String.equal
                 (Option.map (Read.longident_of op) ~f:List.last_exn)
                 (Some ":=") ->
            self#write env target value
        | _ -> ());
        (match expr.pexp_desc with
        | Pexp_apply (callee, arguments) -> (
            let callee = peel callee in
            match (callee.pexp_desc, Read.longident_of callee) with
            | Pexp_ident _, Some path -> (
                let name = List.last_exn path in
                let parity = List.mem parity_callees name ~equal:String.equal in
                if parity || List.mem helper_claim_callees name ~equal:String.equal then
                  let ownership =
                    Option.value_map (Verdict_callee.resolve owners callee) ~default:Not_verdicts
                      ~f:(fun resolution -> resolution.ownership)
                  in
                  match ownership with
                  | Free | Not_verdicts -> ()
                  | Verdicts | Ambiguous ->
                      Int.incr claims;
                      let record predicate =
                        sites :=
                          {
                            tolerance_callee = String.concat ~sep:"." path;
                            tolerance_line = expr.pexp_loc.loc_start.pos_lnum;
                            predicate;
                            tolerance_ambiguous = Poly.equal ownership Ambiguous;
                          }
                          :: !sites
                      in
                      (* The label is the first positional argument; every other one is read. *)
                      let _label, values =
                        List.fold arguments ~init:(false, []) ~f:(fun (seen, acc) argument ->
                            match argument with
                            | Ppxlib.Asttypes.Nolabel, _ when not seen -> (true, acc)
                            | _ -> (seen, argument :: acc))
                      in
                      List.find_map (List.rev values) ~f:(fun (label, value) ->
                          match label with
                          | Ppxlib.Asttypes.Labelled "f" when parity -> (
                              match tolerance env value with
                              | Some _ as found -> found
                              | None -> bearing env value)
                          | _ -> bearing env value)
                      |> Option.iter ~f:record)
            | _ -> ())
        | _ -> ());
        let loops =
          match expr.pexp_desc with
          | Pexp_for _ | Pexp_while _ | Pexp_function _ -> true
          | _ -> false
        in
        if loops then loop_depth <- loop_depth + 1;
        let result = super#expression env expr in
        if loops then loop_depth <- loop_depth - 1;
        result
    end
  in
  (* Each pass reads the writes the previous ones found; each stage ends with a pass that finds
     nothing new, the metrics' before the bounds' begins. *)
  let rec pass () =
    changed := false;
    sites := [];
    claims := 0;
    ignore
      ((make_iterator ())#structure
         { Lexical_scope.frames = []; modules = Map.empty (module String) }
         ast
        : structure);
    if !changed then pass ()
    else if not !bounds_stage then (
      bounds_stage := true;
      pass ())
  in
  pass ();
  { hand_rolled = List.rev !sites; parity_sites = !claims }

(** {1 A gate that reads the population it quantifies (gh-ocannl-1289)}

    [Verdict.gated_all ~when_ ~on label xs ~f] (and [gated_alli], [gated_exists]) skips its claim
    where [when_] is false. A gate computed from the values under test turns the regression the
    claim exists to catch into a skip: had the population emptied or lost its interesting members,
    the gate closes, the skip prints [p]'s passing line, and the golden stays green. That is the
    gh-ocannl-1115 trap, and it shipped twice in one wave through these very forms — a gate asking
    whether the claim's own [results] were non-empty (staging#1049), and one asking whether the
    [real] seeds [results] was mapped from held a tensorized one (staging#1063).
    [Verdict.gated ~when_ ~on label b] carries the same trap when [b] quantifies: [placement_store]
    gated [(not (List.is_empty stores)) && List.for_all stores ~f:refused] on [stores] being
    non-empty until gh-ocannl-1361. A gate comes from an invariant — the backend, an advertised
    capability, the configuration — never from the population.

    {2 What is read}

    Every application of [gated], [gated_all], [gated_alli] or [gated_exists] that carries its
    [~when_] and both positional arguments, where {!Verdict_callee} -- the resolver every Verdict
    reader shares -- reads the name as [Verdict]'s, [Ambiguous] (reported as such, with the remedy
    of qualifying the call) or [Free]: a [gated] name nothing binds is read too, since it exists
    nowhere but in [Verdict.Claims]. A file's own [gated_all], and another module's, are not
    [Verdict]'s; a name a structure of the file exports is followed to its definition there, so
    [module Checks = struct let gated_all = Verdict.gated_all end] opened is [Verdict]'s. A partial
    application bound by [let] to a name, under any number of further partial applications, is
    completed at each application of that name, and [x |> f] and [f @@ x] are read as [f x]; so the
    [let check = Verdict.gated_all ~when_ ~on in check label xs] wrapper is one site per use. The
    two sides are then compared by LEXICAL identity, a name plus the binding it resolves to through
    {!Lexical_scope}, so a shadowed name is a different value; a name no binding of the source gives
    (a global, or one an unmodelled [open] brought in) is one identity, the same on both sides:

    - the GATE is every value identifier the [~when_] expression spells, and transitively every one
      spelled by the right-hand side of a [let] it resolves to (function bodies included), or by
      what the source writes into that binding (below), in any position; a collection-module
      function {!collection_functions} does not list, applied anywhere the gate reaches, is reported
      as unlisted;
    - the POPULATION of a [gated_*] claim is the identifier the quantified argument spells, and the
      collections it is BUILT from: following a [let]-bound name into its right-hand side, the
      branches (not the condition) of an [if], the cases of a [match] (the scrutinee only through a
      name a case pattern binds and the case returns -- and of that name, only the component it
      takes apart, where the scrutinee is, through [let]-bound names, a literal tuple, record or
      constructor application: see {!parts}), the body of a function in data position (a
      [~default:(fun () -> ...)] thunk), the body and handlers of a [try], the body of a [let],
      [let module], local [open] or sequence, the elements of a list, array or tuple literal or
      constructor payload, the operand of [!] and of [ref] and both operands of [@] (where the
      source does not bind the operator itself), and the arguments that carry the collection into a
      function of a standard collection module ([List], [Array], [Hashtbl], [Map], [Option],
      [String], ... — see {!collection_modules}), reached by name, through a module alias
      ([module Rows = List]) or a value alias ([let map = List.map in map raw ~f], a renaming
      followed through any number of [let]s, though not a partial application), and not under a name
      a local module took over ([module Map = Custom]); an unqualified [List] is [Stdlib]'s until an
      [open Base] in scope rebinds it. Which arguments carry it is the role {!collection_functions}
      gives the function by its exact name ({!builds}): every positional one for a transformation,
      constructor, conversion or quantifier, the collection alone for a [Base]-ordered selector
      ([find], [nth], [take], [remove], ...), the value filled in for [make]/[create] and not the
      count, what the callback returns for [init], and nothing for [range], plus the data-bearing
      labels [~default], [~init], [~data], [~sep], [~with_], [~replacement]. A function of a
      collection module the table does not list is REPORTED ({!unlisted_call}), never read by a
      guess (gh-ocannl-1303);
    - the body of a binding operator ([let+ x = e in body]), where [x] is a part of [e];
    - the POPULATION of a [gated] claim is what its boolean's quantifiers range over, each followed
      as above: the boolean is read through a name into its right-hand side and what the source
      writes into it, [not], [&&] and [||] (bare where the source does not bind them, or qualified
      by a standard library), [!] and [ref] where unbound too, the branches of an [if] (never the
      condition), the cases of a [match], the body and handlers of a [try], the body of a [let],
      [let module], local [open] or sequence, type constraints, pipelines, and a quantifier's own
      arguments, its predicate's body included (a quantifier nested there ranges over a population
      of its own, the predicate's parameters none); a quantifier is a collection-module function
      {!collection_functions} marks [quantifies] ([for_all], [exists], [is_empty], [mem] and their
      indexed and pairwise forms, and the predicates of a collection's elements as a whole:
      [contains], [is_sorted], [is_some], ...), whose collection is its first positional argument in
      [Base]'s order (the first two for a pairwise one) and every positional argument in [Stdlib]'s;
      a collection-module function the table does not list is reported here too;
    - and, for a name it reaches, whatever the source writes into that binding, directly or through
      a [let alias = name] renaming: the right-hand side of [x := e] and of [x.field <- e], and what
      a function of a mutable collection module ({!mutable_modules}) that {!collection_functions}
      marks as writing ({!writes}: [Queue.enqueue x e], [Hashtbl.set x ~key ~data],
      [Hashtbl.filter_keys_inplace x ~f], ...) is handed with [x], where [x] is the first positional
      argument in [Base]'s order and any of them in [Stdlib]'s. A function of a mutable collection
      module the table does not list, handed [x], may write into it: a claim that reaches [x] is
      reported as unlisted. So is a writer handed [x] alone ([Queue.enqueue x]) or whose value a
      [let] binds to a name ([let put = Array.set x 0]): a possible partial application whose later
      call this reader does not follow. A partial writer that escapes otherwise (returned, or handed
      to another function) is not followed. A field write [x.f <- e] reaches [x] whole, and the [f]
      a pattern takes apart from [x]'s literal, not its other fields. [x] is a name the source
      binds; a container only an [open] brought in is not tracked. Only a write that may run before
      the claim counts ({!precedes}): one placed after it, in code that cannot run again before it,
      is not followed.

    The claim is refused when the two share a binding. The asymmetry is the point: a gate may read
    what the population is COMPUTED from (the site, the hardware limits, the backend steering an
    [if]), which is how a capability judgment over the same program reads, and must not read the
    population or a collection it is mapped, filtered or appended from.

    {2 Leg claims (gh-ocannl-1794)}

    A leg claim, [in_leg ~leg label make] or [in_legf ~leg fmt ... make] ({!leg_claims}), is a gated
    claim too. Its GATES are the [~when_] of the qualified [Verdict.leg] or [Verdict.sub_leg]
    application ({!Verdict_callee.leg_constructors}) its [~leg] reaches, read as a gate is above,
    and then of each [sub_leg]'s outer leg; [~leg] is followed through a name into its [let]
    right-hand side, a type constraint, the body of a [let] or a sequence, both branches of an [if]
    and every case of a [match] -- whose condition, scrutinee and guards pick the leg and are gates
    too -- and the body of a local function it applies; an ancestor two branches share is read on
    each path. Its POPULATION is what the claims its [make] function makes in place range over, each
    read by {!leg_population_slots}: a boolean as a [gated] boolean's, collections as a [gated_*]
    claim's. Beside them, every argument the make function hands a helper -- a function of the
    source or another module, not an operator, a {!standard_functions} name, a standard or
    collection module's function, or [Verdict]'s own -- is the population of the claim that helper
    makes, on every path where no claim read in place runs: where each such claim sits in another
    branch of some [if], [match] or [function] than the helper. A helper applied in [make]'s place
    is such a helper. The loud direction. On both sides, and for the [gated_*] forms too, a [lazy e]
    is [e]'s and [Lazy.force x] (or [Base]'s [force]) is [x]'s: a leg's evidence lives in lazies
    only its claims force.

    The same model of what runs bounds the population: a function value or [lazy] inside the make
    function is deferred, and its claims are no population, unless a collection-module function
    calls it back, or the make function itself runs it (gh-ocannl-1893): a function defined inside
    the make function and applied there by its name, or a [lazy] defined there and forced there by
    its name ([Lazy.force], [Base]'s [force]), is walked again where it runs ({!replay}), its claims
    the make function's. A leg claim nested in a make function is a claim that make function makes:
    its population is every enclosing leg's too. Anything else the reader cannot follow is outside
    the model: such a leg is counted as UNCOMPARED (below), never refused.

    {3 A leg or make function passed as a parameter (gh-ocannl-1893)}

    A parameter of a named function of the source -- a [let], top-level or local, recursive or not,
    whose right-hand side is a function; a labelled, optional or positional parameter whose pattern
    is a name, or a record or tuple pattern (below) -- is RESOLVED AT THE APPLICATIONS of that
    function in this source and in its callers (below): the argument each application supplies for
    it, by label or by position among the positional ones, in the scope the application is written
    in, or, where an application omits an optional one, its default. A [~leg] reaching such a
    parameter is followed into each application's argument, through any number of calls (a parameter
    reached again on its own path, a function passing its leg back to itself, adds no gate). Each
    choice of the applications on the way is one RESOLUTION ({!leg_resolution}), and the claim is
    compared once per resolution, under that choice: there a parameter of a chosen function, on
    either side, is the chosen application's argument alone, so one call's gate never meets another
    call's population. Every claim inside a named function -- a leg claim, under each of its
    resolutions, and a [gated_*] claim -- is also compared once per CALL CHAIN reaching that
    function ({!call_chains}: an application of it, then one of the function that application is
    written in, and so on up to code no function encloses, at most 256 chains), so a leg a helper
    builds from its own [~when_] parameter, and
    [let check ~when_ label xs = gated_all ~when_ ~on label xs ~f], are read at their callers, one
    call at a time, through forwarding helpers too. A parameter of a function no resolution or chain
    chose is followed into every application's argument (the loud direction). A make function given
    by name is followed through a [let] to its definition, and through a parameter to each argument
    as a [~leg] is, and the function found is walked again with the claim's collector ({!replay}); a
    helper applied in its place keeps the helper rule above. A local function's application on a
    [~leg]'s way binds its parameters to that application's arguments.

    {3 Applications in other sources (gh-ocannl-2286)}

    The applications are this source's and those of the other sources handed in as [?callers]: the
    ratchet's cross-source index gives each source the sources that spell its module, [M] (the file
    name up to its first dot, capitalized, so each arm of a [(select)] is the module dune copies it
    to). Each caller is walked after this source, as a structure of its own under the one module
    binding [M] -- this source's top-level structure -- with its character positions placed past
    every earlier source's ({!structure_at}), so no binding identity or position-keyed table is
    shared and a caller's [rows] is never this source's. What the walk records there -- call sites,
    parameters, writes, its own helpers and their call chains -- is read as this source's is; the
    claims it makes are its own reader's and are not compared here. An application there is
    [M.f args], [f args] under an [open M] ([M.( ... )], [let open M in]), or one through an alias
    [module X = M] or an [include M] the walk follows; [M.f] is resolved the same way in this source
    for a structure [M] of its own, and spelled other than as an applied callee it escapes as a name
    does. A caller's applications are read only where this source has a claim to compare. Outside
    this: a caller spelling [M] packed, as a functor's argument or under a constraint, or inside
    another module's path ([Lib.M.f], [open Lib.M], a wrapped library's spelling); one re-exporting
    [M] through a structure-level alias or [include] where a third source spells the caller's
    module; and one that does not parse -- each leaves every function of [M], its submodules'
    included, UNCOMPARED, naming the caller ({!module_spelled_otherwise}). A source outside the
    corpus the ratchet globs is not read.

    {3 A leg another source's function returns (gh-ocannl-2336)}

    A [~leg] that is the result of [M.f args] -- [M] a structure of the source, or, in a caller's
    walk, this source itself, or a module name nothing in scope binds -- is [f]'s body read as a
    local function's is: each parameter bound to the argument this application supplies, in the
    scope the application is written in, through the same bounds on choices, with the function's own
    result passing back through it uncompared. Where [M] is no structure in scope, its definitions
    are the other sources the ratchet's index gives as compiled under [M] ([?definitions], asked
    with the source [M] is spelled in -- [None] for this one, else a caller or another definition:
    the source of that name in the speller's own directory where there is one, else every source of
    the name, each an alternative compared in its own right). Each is read once: a caller as it was
    walked, any other source parsed past every source walked before it, so it shares no positions,
    and walked as a caller is, so what it writes into its bindings and its own applications are read
    as a caller's are; its [Verdict] resolutions join this source's. A destructured parameter of [f]
    is the parts of what the application supplies ({!parts}), and [?l:None] for a parameter with a
    default is the default. A module path inside [M] ([M.Sub.f]), an [f] that is no [fun] its body
    follows, an [f] whose body writes into a binding, or whose own gate reads a binding the walk saw
    written anywhere (writes this reader does not time against the claim), an [f] handed [?l:e] for
    a parameter with a default, [e] no literal [None] or [Some _], an [f] exported -- itself or
    through a submodule -- before a later [include] in [M]'s structure, this source's or another's,
    of a module whose exports the reader does not know (one it cannot read, or a [struct ... end]
    holding such an [include]), which may rebind it, a source that does not parse, and a module name
    no source is compiled as each leave the leg UNCOMPARED, with the reason; so does [L.f] under a
    module alias of another source ([module L = Ll_test]), which binds no structure in scope. A
    module name an [open] of a module this reader cannot read rebinds is read as the corpus's source
    of that name, as a caller's [M.f] is.

    {3 A leg taken out of a value (gh-ocannl-2286)}

    A leg read from a record field ([case.leg]), or bound by a record or tuple pattern of a named
    function's parameter ([let run { rows; leg } = ...]), is the part of the value reached that the
    field or pattern names: the reader carries the projections along the leg's way -- through names,
    [let]s, branches and parameters as above -- and takes them at the record or tuple literal it
    reaches, or at a [{ base with ... }]'s base where the field is not given. The parameter of an
    anonymous callback an {!element_iterations} function is handed in place
    ([List.iter cases ~f:(fun case -> run case)]) is each element of the collection where that is a
    list or array literal (through names no write reaches, [let] bodies and sequences): one
    resolution per element, each compared with that element chosen, on both sides, as an application
    is -- so one element's gate never meets another element's population; a claim written inside
    such a callback, or in a function an application inside one reaches, is compared once per
    element of each such callback too. A destructured parameter is likewise, on the gate and
    population sides, the part of each argument its pattern takes apart ({!parts}), with what the
    source writes into that field before the claim; a field a [{ base with ... }] leaves is taken
    from the base where that is a literal too.

    Where this stops, the leg is UNCOMPARED, with the reason: the function's name is spelled other
    than as the callee of an application (handed on as a value, piped), so not every application is
    seen; no application of this source or of a caller supplies the parameter; a caller spells the
    module in a way that may hide one (above); an application omits it, or passes an optional one as
    [?leg:]; the parameter belongs to an anonymous function no {!element_iterations} function is
    handed, or the collection it iterates is no literal; a pattern binds the leg other than through
    record fields and tuple components (a constructor's payload, an or-pattern); a field is taken of
    a value that is no record literal, or of a binding the source writes into; or a make function is
    taken out of a value that way, which the reader does not follow.

    A leg claim applied in part -- without its [~leg], its label and its make function in place, an
    [in_legf] with exactly as many arguments as its literal format takes before the make function
    ({!format_arity}) -- or named as a value is reported unread. A complete one whose [~leg] or make
    function the reader cannot resolve as above -- a pattern, a field, another module's value, an
    unqualified [leg] (an [open Verdict]'s) -- is reported UNCOMPARED: counted, named, and not
    refused.

    {2 What is not read}

    The forms above are a closed list: any other expression form ends the population side there, and
    a population reached only through one is not compared. The exception is a collection-module
    function, which is never outside the list: {!collection_functions} enumerates the ones the
    reader reads, and any other is reported. The ones known to matter:

    The population is not followed through a call to any other function (a file-local
    [gpu_seeds pre] stops at the call), through a record field, through a parameter of an anonymous
    function no {!element_iterations} function is handed over a literal collection, or of a function
    no application read supplies (a named function's parameter is its applications' arguments,
    above), or through an unqualified name under an open of a collection module
    ([List.(permute raw)]) that {!collection_functions} does not list: such a name is read as
    whatever else binds it, and is not reported as unlisted, since without the module's signature it
    cannot be told from a function of the source (a listed name there is the module's,
    {!resolve_callee}); a destructured [let] binds names that are followed no further, and a value
    reached through a module path ([M.rows]) is an identity on neither side. A gate threaded into a
    wrapper as its own parameter is read at the wrapper's applications, in this source and its
    callers, as above; a wrapper no application read applies is read at its definition, where the
    gate is a parameter and reads nothing. A [gated] boolean that quantifies nothing has no
    population: its comparisons and other calls are operands this reader does not read, so
    [gated ~when_:(xs <> []) label (sum xs = want)] passes, and so does a gate folded into a value
    ([p "…" ((not on_cpu) || ok)]), which {!folded_gate_census} reads instead. Each of these is a
    place a population gate can hide from this reader; none is a place it reports one that is not
    there.

    A gated application the reader cannot complete — a partial application no [let] binds, a
    [let]-bound one that no use the reader resolves completes (reached only through a module path,
    say), or a claim (or a [let]-bound wrapper of one) named as a value, handed to [List.iter] say —
    is reported as unread rather than counted as clean, so the boundary is loud there. *)

(** What a module name denotes to this reader. A collection module carries whether its functions
    take arguments in [Stdlib]'s order (the function first, the collection last) rather than
    [Base]'s (the collection first). *)
type gate_module =
  | Collection_module of { stdlib : bool; mutable_ : bool; family : string }
      (** [mutable_]: its values are mutable containers ({!mutable_modules}). [family]: the module's
          own name ([Queue] for [module Q = Queue]), which {!Only_in} reads. *)
  | Standard_library of { stdlib : bool }
      (** [Base] or [Core] ([stdlib = false]), [Stdlib] or [Caml] ([true]): an [open] of one rebinds
          the collection modules to its own. *)
  | Local_structure of {
      values : (string * gate_denotation) list;
      modules : (string * gate_module) list;
      opaque : bool;
          (** It includes a module whose exports this reader does not know -- one it cannot read, or
              an opaque structure -- which may rebind any name exported before the include
              (gh-ocannl-2336). *)
    }
      (** A [struct ... end] of this source, with the value names it exports and what each denotes,
          its definition included (resolved in the structure's own scope), and the modules it
          exports and what each denotes: an [open] of it brings them into scope as this source's
          own, so a [let gated_all = Verdict.gated_all] exported from it is still followed to
          [Verdict]'s, and a [module Rows = List] is still [List] (gh-ocannl-1303). *)
  | Other_module

and gate_denotation = {
  identity : int;
  mutable definition : (expression * (gate_denotation, gate_module) Lexical_scope.env) option;
      (** Mutable only so that a recursive group's definitions can see each other: they are set once
          the group's own scope exists. *)
}

type gate_reading = {
  form : string;  (** The callee as written: [gated_all], [Verdict.gated_exists], a wrapper. *)
  ambiguous : bool;
      (** The claim reaches [Verdict] only past an [open] of a module {!Verdict_callee} cannot read,
          which may export a name or a [Verdict] module of its own. *)
  line : int;
  gate : string;  (** The [~when_] expression, printed. *)
  reads : string list;  (** The population bindings the gate reads, by name, sorted. *)
  applied_in : string list;
      (** The other sources whose applications the offending comparisons were made under, where the
          gate or the population is theirs (gh-ocannl-2286), sorted. *)
}

type unlisted_call = {
  claim : string;  (** The claim's callee as written. *)
  claim_line : int;
  callee : string;
      (** The collection-module function {!collection_functions} does not list, as written. *)
  callee_line : int;
}
(** A collection-module function a compared claim's population, gated boolean or gate reaches, or
    one handed a mutable container they reach, that the table does not list: the reader cannot say
    which of its arguments carry the population, or whether it writes into the container. *)

type unread_leg = {
  leg_form : string;  (** The leg claim's callee as written: [in_leg], [Verdict.in_legf]. *)
  leg_line : int;
  why : string;  (** What the reader could not resolve, as a clause completing "because ...". *)
}
(** A leg claim ({!Verdict.in_leg}, {!Verdict.in_legf}) whose gates or population this reader could
    not resolve (gh-ocannl-1794). *)

type gate_census = {
  readings : gate_reading list;  (** Claims whose gate reads their population, in source order. *)
  gated_sites : int;  (** Every completed [gated_*] claim the reader compared. *)
  unread : (string * int) list;  (** [gated_*] applications it could not complete: callee, line. *)
  unlisted : unlisted_call list;
      (** Compared claims that reach an unlisted collection-module function, in source order. *)
  leg_sites : int;  (** Every leg claim whose gates and population the reader compared. *)
  unread_legs : unread_leg list;
      (** Leg claims applied in part, which the reader cannot complete, in source order. *)
  uncompared_legs : unread_leg list;
      (** Complete leg claims whose leg the reader could not follow to its gates -- a leg handed in
          as a parameter, say -- in source order: the boundary, as a population threaded through a
          parameter is a [gated_*] claim's. *)
}

let empty_gate_census =
  {
    readings = [];
    gated_sites = 0;
    unread = [];
    unlisted = [];
    leg_sites = 0;
    unread_legs = [];
    uncompared_legs = [];
  }

(** The leg claims: [in_leg ~leg label make] and [in_legf ~leg fmt ... make], whose gate is the
    [~when_] of every {!Verdict_callee.leg_constructors} application [~leg] is bound to, through
    each [sub_leg]'s outer leg, and whose population is what the claims [make] makes range over
    ({!leg_population_slots}). *)
let leg_claims = [ "in_leg"; "in_legf" ]

(** Where a claim made inside a leg's [make] carries its population: a boolean, read as a [gated]
    boolean is ({!Verdict_scan.gate_census}'s [quantified]), at a positional argument or the last
    one (after a format's arguments); or collections, read as a [gated_*] claim's is, at positional
    arguments or the last [n] of them, each with a labelled [~over]. *)
type leg_slot =
  | Boolean_at of int
  | Boolean_last
  | Collections_at of int list
  | Collections_last of int

(** Each [Verdict.Claims] member that makes a claim with a value, and where its population is; held
    equal to [Verdict.Claims] together with {!leg_slotless_members} by [verdict_ratchet]. *)
let leg_population_slots =
  [
    ("p", Boolean_at 1);
    ("claim", Boolean_at 1);
    ("p_unless", Boolean_at 1);
    ("pass_fail", Boolean_at 1);
    ("gated", Boolean_at 1);
    ("pf", Boolean_last);
    ("claimf", Boolean_last);
    ("p_all", Collections_at [ 1 ]);
    ("p_alli", Collections_at [ 1 ]);
    ("p_exists", Collections_at [ 1 ]);
    ("p_none", Collections_at [ 1 ]);
    ("p_empty", Collections_at [ 1 ]);
    ("p_pairwise_distinct", Collections_at [ 1 ]);
    ("gated_all", Collections_at [ 1 ]);
    ("gated_alli", Collections_at [ 1 ]);
    ("gated_exists", Collections_at [ 1 ]);
    ("p_all2", Collections_at [ 1; 2 ]);
    ("pass_fail_all2", Collections_at [ 1; 2 ]);
    ("p_close", Collections_at [ 1; 2 ]);
    ("pass_fail_close", Collections_at [ 1; 2 ]);
    ("pf_all2", Collections_last 2);
    ("pf_close", Collections_last 2);
  ]

(** The members that make no claim with a population: a message, a skip, a case boundary, and the
    leg claims themselves. *)
let leg_slotless_members = [ "fail"; "skipped"; "case"; "in_leg"; "in_legf" ]

(** How many arguments a [Printf] format literal takes: one per conversion, two for [%a], one more
    per [*] width or precision; [%%], [%!], [%,] and [%@] take none. [None] for a format this does
    not read ([%(], [%{]): its leg claim is then reported unread. *)
let format_arity format =
  let n = String.length format in
  let rec go i count =
    if i >= n then Some count
    else if Char.equal format.[i] '%' && i + 1 < n then
      let rec spec j stars =
        if j >= n then None
        else
          match format.[j] with
          | '%' | '!' | ',' | '@' -> if j = i + 1 then go (j + 1) count else None
          | '-' | '0' .. '9' | '.' | '+' | ' ' | '#' | '_' -> spec (j + 1) stars
          | '*' -> spec (j + 1) (stars + 1)
          | ('l' | 'n' | 'L') when j + 1 < n && Char.is_alpha format.[j + 1] ->
              go (j + 2) (count + stars + 1)
          | 'a' -> go (j + 1) (count + stars + 2)
          | '(' | '{' -> None
          | c when Char.is_alpha c -> go (j + 1) (count + stars + 1)
          | _ -> None
      in
      spec (i + 1) 0
    else go (i + 1) count
  in
  go 0 0

(** The unqualified functions of the standard libraries a leg's [make] may apply without handing
    them a claim's population: printing, raising, the pair and reference primitives. *)
let standard_functions =
  [
    "not";
    "ignore";
    "fst";
    "snd";
    "raise";
    "failwith";
    "invalid_arg";
    "printf";
    "eprintf";
    "sprintf";
    "print_string";
    "print_endline";
    "prerr_endline";
    "ref";
    "incr";
    "decr";
    "min";
    "max";
    "compare";
    "phys_equal";
  ]

(* A leg claim's population as the walk meets the claims of its [make]: each slot's expression,
   whether it is a boolean, and the scope it was written in. *)
type leg_collector = {
  mutable populations :
    (int * expression * bool * (gate_denotation, gate_module) Lexical_scope.env) list;
      (** Each claim's slot, after the position of the claim. *)
  mutable handed : (int * expression * (gate_denotation, gate_module) Lexical_scope.env) list;
      (** Every argument of every helper application in [make], after the application's position:
          the population on a path where it makes no claim this reader reads in place. *)
  mutable branches : (int * int) list list;
      (** The branch spans of every [if], [match] and [function] in [make]: two points in different
          branches of one never run together. *)
  make_span : int * int;
      (** Where the make function is written: a local function or [lazy] defined there and applied
          or forced there is run code of [make] (gh-ocannl-1893). *)
}

(** Where a parameter of a named function of the source takes its value from at an application of
    the function (gh-ocannl-1893). *)
type parameter_slot =
  | Labelled_slot of string
  | Optional_slot of string * (expression * (gate_denotation, gate_module) Lexical_scope.env) option
      (** With its default, in the scope it is evaluated in, where it has one. *)
  | Positional_slot of int  (** Among the function's positional parameters. *)
  | Element_slot of expression * (gate_denotation, gate_module) Lexical_scope.env
      (** The element parameter of an anonymous callback an {!element_iterations} function is handed
          in place: each element of the collection, in the scope the application is written in
          (gh-ocannl-2286). *)

(** One step taking a value apart: a record's field, or a tuple's component (its index and the
    tuple's arity). *)
type projection = Field of string | Component of int * int

type parameter_of = {
  owner : int;
      (** The function's binding identity; for an {!Element_slot}, an identity of its own, which a
          {!leg_resolution} chooses an element under. *)
  owner_name : string;
  slot : parameter_slot;
  part : (pattern * string) option;
      (** Where the parameter's pattern is not a name: the whole pattern, which takes apart what an
          application supplies, and the name this parameter is of the ones it binds
          (gh-ocannl-2286). *)
}

type leg_resolution = {
  calls : (int * int) list;
      (** A function's binding identity, and the position of the application its parameters took
          their arguments from. *)
  leg_gates : (expression * (gate_denotation, gate_module) Lexical_scope.env) list;
  leg_ambiguous : bool;
}
(** One alternative way a leg's [~leg] reaches its gates: the applications chosen on the way, by
    function, and the gates found under that choice. *)

let gated_forms = [ "gated_all"; "gated_alli"; "gated_exists" ]

(** The collection-module functions that hand the callback they are given each element of ONE
    collection in turn, with the index, among the callback's positional parameters, of the one that
    is the element (gh-ocannl-2286): [List.iter legs ~f:(fun leg -> ...)] binds [leg] to each
    element of [legs], and so does [List.iteri legs ~f:(fun i leg -> ...)]. A closed list, of the
    {!element_families} alone, read where the callback is an anonymous function written in place. *)
let element_iterations =
  [
    ("iter", 0);
    ("map", 0);
    ("concat_map", 0);
    ("filter_map", 0);
    ("iteri", 1);
    ("mapi", 1);
    ("concat_mapi", 1);
    ("filter_mapi", 1);
  ]

let element_families = [ "List"; "Array" ]

(** The projections that take [name] out of a value [pattern] matches, outermost first: through
    record fields, tuple components, aliases and type constraints. [None] where the pattern binds
    [name] some other way -- a constructor's payload, an or-pattern, a lazy one -- or not at all. *)
let rec projection_to name pattern =
  match pattern.ppat_desc with
  | Ppat_var { txt; _ } when String.equal txt name -> Some []
  | Ppat_alias (_, { txt; _ }) when String.equal txt name -> Some []
  | Ppat_alias (inner, _) | Ppat_constraint (inner, _) -> projection_to name inner
  | Ppat_tuple components ->
      let arity = List.length components in
      List.find_mapi components ~f:(fun index component ->
          Option.map (projection_to name component) ~f:(fun rest ->
              Component (index, arity) :: rest))
  | Ppat_record (fields, _) ->
      List.find_map fields ~f:(fun (field, inner) ->
          Option.map (projection_to name inner) ~f:(fun rest ->
              Field (Ppxlib.Longident.last_exn field.txt) :: rest))
  | _ -> None

(** [Verdict.gated ~when_ ~on label b]: its population is what the quantifiers its boolean reduces
    to range over ({!quantifiers}). *)
let gated_boolean = "gated"

(* The arguments a quantifier ranges over: in [Base]'s order the first positional argument, the
   first two for a pairwise one; in [Stdlib]'s, where the predicate comes first ([List.for_all f
   xs]) but a [Base] spelling under no [open Base] reads the same way, every positional one -- the
   loud direction, as for {!collection_arguments}. *)
let quantified_arguments ~stdlib name arguments =
  let positional =
    List.filter_map arguments ~f:(function Ppxlib.Asttypes.Nolabel, a -> Some a | _ -> None)
  in
  (* The collection a predicate is labelled with: [Set.is_subset required ~of_:allowed]. *)
  let of_ =
    List.filter_map arguments ~f:(function
      | Ppxlib.Asttypes.Labelled "of_", a -> Some a
      | _ -> None)
  in
  (if stdlib then positional
   else List.take positional (if String.is_substring name ~substring:"2" then 2 else 1))
  @ of_

(* What a gated claim the walk reached carries: the callee as written, the [Verdict.Claims] member
   it completes, whether that member is {!Verdict_callee.Ambiguous}, and every argument handed to it
   so far, each with the scope it was written in. *)
type gated_reach = {
  reached_form : string;
  member : string;
  reached_ambiguous : bool;
  arguments :
    (Ppxlib.Asttypes.arg_label * expression * (gate_denotation, gate_module) Lexical_scope.env) list;
}

(** The modules whose functions build a collection from their positional arguments, by the module
    component just before the function's name ([List.map], [Base.List.map], [Stdlib.List.map]). *)
let collection_modules =
  [
    (* Base and Core *)
    "List";
    "Array";
    "Sequence";
    "Hashtbl";
    "Hash_set";
    "Hash_queue";
    "Map";
    "Set";
    "Queue";
    "Linked_queue";
    "Stack";
    "Deque";
    "Doubly_linked";
    "Bag";
    "Fqueue";
    "Option";
    "Result";
    "Option_array";
    "Uniform_array";
    "Nonempty_list";
    "String";
    "Bytes";
    (* Stdlib *)
    "ListLabels";
    "ArrayLabels";
    "StringLabels";
    "BytesLabels";
    "Seq";
    "Dynarray";
    (* submodules *)
    "Assoc";
    "Poly";
  ]

(** The collection modules whose values are mutable containers: only a writer of one of these writes
    into its argument; [Set.add] or [Map.update] returns a new collection and leaves its argument as
    it was. *)
let mutable_modules =
  [
    "Array";
    "ArrayLabels";
    "Bytes";
    "BytesLabels";
    "Hashtbl";
    "Hash_set";
    "Hash_queue";
    "Queue";
    "Linked_queue";
    "Stack";
    "Deque";
    "Doubly_linked";
    "Bag";
    "Option_array";
    "Uniform_array";
    "Dynarray";
  ]

(* Whether a claim's arguments so far are all it takes: its gate, and its label and population. *)
let complete arguments =
  List.exists arguments ~f:(function Ppxlib.Asttypes.Labelled "when_", _, _ -> true | _ -> false)
  && List.count arguments ~f:(function Ppxlib.Asttypes.Nolabel, _, _ -> true | _ -> false) >= 2

(** Which arguments of a collection-module function the population is built from, by the role the
    function plays in {!collection_functions}. *)
type builds =
  | Every_argument
      (** Every positional argument, the labeled [~key], what [~f] returns, and the data labels:
          [map], [Map.of_alist_exn (module String) rows], [append], [zip], [Set.add s x], [cons]
          build the population from whatever they are handed, the collection and the elements put
          into it alike. *)
  | Collection
      (** A selector. In [Base]'s order its first positional argument, the collection, and the data
          labels; its later positional arguments are a key, an index or a count, which select the
          population as an [if] condition steers one, so a gate on them is a gate on what was
          selected, not on what was found. In [Stdlib]'s order the collection may come last
          ([List.assoc k l], [List.mem x l]), so there every positional argument, [~key] and [~f] as
          for {!Every_argument} -- the loud direction. *)
  | Collection_and_callback
      (** A selector whose [~f] returns what it puts in or finds ([Map.update m k ~f],
          [List.find_map xs ~f]), unlike a [find] predicate. *)
  | After_count
      (** [init n ~f], [Stdlib.List.init n f]: what its callback returns (labeled [~f], or
          positional after the count) and the data labels, not the count. *)
  | Filled_value
      (** [make n x], [create ~len x]: the positional arguments after the first, the value filled
          in, and the data labels, not the count. *)
  | Nothing  (** [range a b]: bounds build nothing. *)

(** Whether a function writes what it is handed into a container it is handed. *)
type writes =
  | Never
  | On_mutable  (** On a mutable collection module ({!mutable_modules}). *)
  | Only_in of string list
      (** On these modules alone: [Dynarray.append a b] writes into [a], [Array.append] does not. *)

type collection_function = {
  builds : builds;
  quantifies : bool;
      (** True or false of every element, or of none: what a [gated] boolean is read through. *)
  writes : writes;
  consumes : bool;
      (** It runs its callback, or materialises its argument, before it returns even on a lazy
          sequence ([Seq], [Sequence]): [iter], a fold, a quantifier, [find], [to_list],
          [List.of_seq]. A sequence's [map], [filter] or [init] only builds a sequence that calls
          its callback later, when something consumes it. The one owner of which sequence functions
          run what they are handed: the skip-list reader's iterations and the gate reader's
          synchronous callbacks both ask it. *)
}

(** The closed list of collection-module functions this reader knows (the ratchet scans test
    sources, so it enumerates what [Base], [Core] and [Stdlib] give the {!collection_modules}). A
    function is keyed by its name alone: a name plays one role in every collection module that has
    it ([List.map], [Array.map], [Map.map]), and a module's argument order and mutability are
    {!gate_module}'s, not the table's. The one name whose writing depends on the module carries
    {!Only_in}. A collection-module function missing from the table is reported, not guessed: the
    claim it reaches is refused as unlisted, naming the function to add. *)
let collection_functions =
  let group builds ?(quantifies = false) ?(writes = Never) ?(consumes = false) names =
    String.split_on_chars names ~on:[ ' '; '\n' ]
    |> List.filter ~f:(Fn.non String.is_empty)
    |> List.map ~f:(fun name -> (name, { builds; quantifies; writes; consumes }))
  in
  Map.of_alist_exn
    (module String)
    (List.concat
       [
         (* Transformations, constructors and conversions: built from everything they are handed. *)
         group Every_argument
           {|
            map mapi map2 map2_exn rev_map rev_mapi rev_map2_exn rev_map_append filter filteri
            filter_map filter_mapi filter_opt concat concat_map concat_mapi concat_no_order
            concat_array concat_lines rev_append rev sort stable_sort sort_uniq sorted_copy
            dedup_and_sort stable_dedup remove_consecutive_duplicates zip zip_exn unzip unzip3
            combine split partition_tf partitioni_tf partition_map partition_mapi partition
            partition3_map split_while split_on_chars split_on_char split_lines group groupi
            sort_and_group of_list of_list_rev of_list_map of_list_mapi of_array of_char
            of_char_list of_string to_seq to_seq_keys to_seq_values to_sequence to_alist of_alist of_alist_exn of_alist_multi of_alist_reduce of_alist_fold
            of_alist_or_error of_increasing_sequence of_sorted_array of_sorted_array_unchecked
            of_list_with_key of_list_with_key_exn of_list_with_key_multi data keys key_set
            of_key_set elements singleton return cons empty transpose transpose_exn intersperse
            cartesian_product fold_map folding_map folding_mapi hd hd_exn tl tl_exn value value_exn
            value_map value_or_thunk bind join both first_some some some_if merge merge_skewed union
            union_list inter diff symmetric_diff copy map_keys map_keys_exn filter_keys map_error ok
            ok_exn ok_or_failwith error get_ok get_error to_option of_option all all_unit try_with
            call hash to_string escaped
            lowercase uppercase capitalize uncapitalize strip lstrip rstrip trim tr prefix suffix
            chop_prefix chop_prefix_exn chop_prefix_if_exists chop_suffix chop_suffix_exn
            chop_suffix_if_exists drop_prefix drop_suffix substr_index substr_index_exn
            substr_index_all substr_replace_first substr_replace_all lsplit2 lsplit2_exn rsplit2
            rsplit2_exn index index_exn index_from rindex rindex_exn lfindi rfindi assoc assoc_opt
            inverse sexp_of_t t_of_sexp sexp_of_m__t m__t_of_sexp
          |};
         (* The same role, and they consume a lazy sequence: they run their callback, or walk the
            whole sequence, before they return. *)
         group Every_argument ~consumes:true
           {|
            fold foldi fold_left fold_right fold_left2 fold_right2 fold2_exn fold_until fold_result
            reduce reduce_exn reduce_balanced reduce_balanced_exn sum count counti length min_elt
            max_elt min_elt_exn max_elt_exn last last_exn iter iteri iter2 iter2_exn iter_keys
            equal compare find_a_dup find_all_dups to_list to_array of_seq of_sequence
          |};
         (* Quantifiers, and the predicates true or false of a collection's elements as a whole
            (membership, order, presence): a gate closing on the collection hides a false one.
            [equal] and [compare] are not among them: in [Base]'s order their first positional
            argument is the element comparison, not a collection. Nor are a string's pattern tests
            ([is_substring], [is_prefix], ...), which test the text against a pattern argument:
            [foreign_c_stderr] gates on one control marker's presence and claims another's, by
            observation of the same capture, a design this reader leaves to its follow-up. *)
         group Every_argument ~quantifies:true ~consumes:true
           {|
            for_all for_alli for_all2 for_all2_exn exists existsi exists2 exists2_exn is_empty
            contains is_sorted is_sorted_strictly is_subset subset contains_dup is_some is_none
            is_ok is_error is_valid_utf_8
          |};
         group Every_argument {| is_prefix is_suffix is_substring starts_with ends_with |};
         (* Selectors: the finders consume a lazy sequence up to what they find; [take] and [drop]
            build a shorter one. *)
         group Collection ~consumes:true
           {| find find_exn find_opt findi find_index nth nth_exn nth_opt |};
         group Collection
           {|
            find_multi find_and_call find_all get unsafe_get take take_while drop drop_while
            drop_last drop_last_exn sub
            subo slice split_n chunks_of choose choose_exn peek peek_exn
            dequeue dequeue_exn pop pop_exn top top_exn
          |};
         (* Membership: a selector that is true or false of the collection, so a gate closing on the
            collection hides a failed membership claim. *)
         group Collection ~quantifies:true ~consumes:true {| mem mem_assoc |};
         group Collection_and_callback ~consumes:true {| find_map find_mapi find_map_exn |};
         group After_count {| init |};
         group Filled_value {| make create create_float replicate make_matrix |};
         group Nothing {| range range' clear reset |};
         (* Writers into a mutable container: what they are handed joins it. *)
         group Every_argument ~writes:On_mutable
           {|
            set add add_exn add_multi add_last add_seq push enqueue enqueue_all enqueue_front
            enqueue_back replace replace_seq fill blit blito unsafe_blit transfer blit_transfer
            unsafe_set filter_inplace filteri_inplace filter_keys_inplace filter_map_inplace
            filter_mapi_inplace map_inplace mapi_inplace merge_into incr decr strict_add
            strict_add_exn insert_first insert_last insert_before insert_after append_list
            append_array append_seq append_iter set_int32_le set_int64_le
          |};
         group Every_argument ~writes:(Only_in [ "Dynarray" ]) {| append |};
         group Collection ~writes:On_mutable {| find_or_add findi_or_add |};
         (* A removal selects its population, and on a mutable container it writes: what is left
            depends on the key removed. *)
         group Collection ~writes:On_mutable {| remove remove_multi |};
         (* An update builds its result from the key it inserts as well: [Map.update m key ~f]. *)
         group Every_argument ~writes:On_mutable {| update change update_and_return |};
       ])

let collection_function name = Map.find collection_functions name

(* Whether the collection-module function [name] runs what it is handed before it returns on a
   module of [family]: always, except on a lazy sequence, where only a consumer does. *)
let sequence_families = [ "Seq"; "Sequence" ]

let runs_callback ~family name =
  (not (List.mem sequence_families family ~equal:String.equal))
  || Option.exists (collection_function name) ~f:(fun f -> f.consumes)

let collection_arguments ~stdlib function_ arguments =
  let positional =
    List.filter_map arguments ~f:(function
      | Ppxlib.Asttypes.Nolabel, argument -> Some argument
      | _ -> None)
  in
  let labeled names =
    List.filter_map arguments ~f:(function
      | (Ppxlib.Asttypes.Labelled label | Optional label), argument
        when List.mem names label ~equal:String.equal ->
          Some argument
      | _ -> None)
  in
  (* The labeled arguments that carry data rather than steer: [~default] (the fallback
     [Option.value] returns), [~init] (a fold's seed), [~data] (a stored element), [~sep] (what
     [intersperse] and [concat] put between elements), [~with_] (what [substr_replace_all] puts in)
     and [~replacement] (what [tr] maps to). *)
  let data = labeled [ "default"; "init"; "data"; "sep"; "with_"; "replacement" ] in
  match (function_.builds, stdlib) with
  | Nothing, _ -> []
  | After_count, _ ->
      (* {!population} follows a function in data position into its body. *)
      List.drop positional 1 @ labeled [ "f" ] @ data
  | Filled_value, _ ->
      (match positional with _ :: (_ :: _ as values) -> values | values -> values) @ data
  | Collection, false -> List.take positional 1 @ data
  | Collection_and_callback, false -> List.take positional 1 @ labeled [ "f" ] @ data
  | (Every_argument | Collection | Collection_and_callback), _ ->
      (* What a callback returns joins the result: [~f], [~combine] (a merge's value for a shared
         key), [~get_key] (a keyed builder's key), [~finish] ([fold_until]'s result). *)
      positional @ labeled [ "key"; "f"; "combine"; "get_key"; "finish" ] @ data

(* A callee under type annotations, [(List.map : ...) f xs], is the callee. *)
let rec strip_constraints expr =
  match expr.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> strip_constraints inner
  | _ -> expr

(** {2 Which writes may precede a claim (gh-ocannl-1303)}

    A write into a binding the population or the gate reaches is followed only when it may run
    before the claim is evaluated: [Queue.enqueue queue v] written after the claim, in code that
    cannot run again before it, cannot have shaped the claim's population, and a gate reading [v] is
    not refused on its account. Lexical order alone is not the answer -- a write later in a loop
    body precedes the claim's next iteration, and so does a write in a function called again -- so
    the reader places both points in the program's control structure, conservatively: every
    construct whose body may run more than once, or at a time the reader does not know, is a
    {!frame} of each point inside it.

    - A [while] loop, a [for] loop's body (not its bounds), and a function handed directly to a
      function of a collection module in a position that module calls before it returns (a
      [List.iter] callback: {!calls_back}) -- on [Seq] or [Sequence], a function the table marks as
      consuming ({!runs_callback}), not a lazy [map] -- run synchronously, possibly many times: a
      {!Loop}. A function handed in any other position ([List.singleton (fun () -> ...)]) may be
      stored and called later, like any other function value.
    - Any other function may run whenever it is called. It is ONCE when it is the right-hand side of
      a non-recursive local [let] whose name the source spells exactly once, as the callee of a full
      application outside the function's own body: then its body runs exactly where that call runs,
      and a point in it is placed at the call. A top-level function is never once, since another
      module may call it.
    - [lazy], a binding operator's body, an object or class, a functor's body and an extension's
      payload ([let%expect_test] included) run at a time the reader does not model: {!Deferred}.

    A write may precede a claim when, at the outermost level where the two are not inside the same
    call of a once function, (a) both are inside one frame that may repeat (a loop, a function that
    is not once, a deferred construct), (b) the claim is inside such a frame of its own that may run
    at any later time (a function that is not once, or a deferred construct) the write is not in, or
    (c) the write starts before the claim ends, or (d) the two sit in different operands of one
    expression whose operands OCaml evaluates in an unspecified order -- an application's callee and
    arguments ([&&] and [||] apart), a tuple, array or record's components, the bindings of one
    [let ... and ...], a binding operator's operands -- so [consume (claim; ()) (write ())] may run
    the write first. (A write in a function defined before the claim is before it; one in a function
    defined after it cannot run before it unless the claim runs again, which (a) and (b) cover.)
    Inside one call of a once function the same question is asked of the two points within it. A
    claim's own arguments are before its end, so a write there precedes it.

    What this does not model, each erring toward following the write: a function value stored and
    called later is placed where it is defined; a once function's name passed anywhere but a call is
    not once. *)

type frame_kind =
  | Loop
  | Function of {
      binding : int option;
          (** The identity of the non-recursive local [let] whose right-hand side it is. *)
      positional : int;
      labelled : string list;
      erasable : bool;  (** Every optional parameter is followed by a positional one. *)
    }
  | Deferred

type frame = { frame_id : int; shape : frame_kind; frame_start : int; frame_stop : int }

type point = {
  at_start : int;
  at_end : int;
  effect_at : int;
      (** Where its effect takes place: a write into a container, once the writer's arguments are
          evaluated, at its end; anything else at its start. *)
  enclosing : frame list;  (** Innermost first. *)
}

(* The [Stdlib]-ordered functions of a collection module that call the function they are handed
   first -- positionally, after the count for [init] -- before they return. *)
let stdlib_callers =
  Set.of_list
    (module String)
    (String.split_on_chars ~on:[ ' '; '\n' ]
       {|iter iteri iter2 map mapi map2 rev_map rev_map2 filter filteri filter_map concat_map
         fold_left fold_right fold_left2 fold_right2 fold for_all exists for_all2 exists2 find
         find_opt find_index find_map find_mapi find_all partition partition_map sort stable_sort
         fast_sort sort_uniq merge equal compare filter_map_inplace map_inplace mapi_inplace
         filter_inplace|}
    |> List.filter ~f:(Fn.non String.is_empty))

(* Whether the function a collection-module function [name] is handed with [label], as its [index]th
   positional argument if unlabelled, is called before [name] returns: a [~f] callback, or a
   [Stdlib]-ordered callback {!stdlib_callers} names. Any other function handed to it may be stored
   and called later ([List.singleton (fun () -> ...)]), so it is no {!Loop}. *)
let calls_back ~stdlib name label index =
  match label with
  | Ppxlib.Asttypes.Labelled "f" -> true
  | Ppxlib.Asttypes.Nolabel when stdlib ->
      if String.equal name "init" then index = 1 else index = 0 && Set.mem stdlib_callers name
  | _ -> false

(* Whether an application of the collection-module function [name] with these argument labels is
   complete, so that it calls its callback before it returns: [List.map ~f:g] alone is a partial
   application, a function value holding [g] for later. The arity is by name: [Base]-ordered, the
   collection (two for a pairwise [map2], a [Map.update m key], ...) and the [~init] of a fold; in
   [Stdlib]'s order the callback, then the collections and a fold's initial value. *)
let applied_fully ~stdlib name labels =
  let positional = List.count labels ~f:(function Ppxlib.Asttypes.Nolabel -> true | _ -> false) in
  let has label =
    List.exists labels ~f:(function
      | Ppxlib.Asttypes.Labelled given | Ppxlib.Asttypes.Optional given -> String.equal given label
      | Ppxlib.Asttypes.Nolabel -> false)
  in
  let pairwise = String.is_substring name ~substring:"2" in
  if stdlib then
    positional
    >=
    match name with
    | "fold_left2" | "fold_right2" -> 4
    | "fold_left" | "fold_right" | "fold" | "merge" | "equal" | "compare" -> 3
    | _ when pairwise -> 3
    | _ -> 2
  else
    let required_positional, required_labels =
      match name with
      | "merge_into" -> (0, [ "src"; "dst" ])
      | "fold_range_inclusive" -> (1, [ "min"; "max"; "init" ])
      | "fold_until" -> (1, [ "init"; "finish" ])
      | "update" | "change" | "update_and_return" | "merge" | "merge_skewed" | "find_and_call"
      | "findi_and_call" ->
          (2, [])
      | _ when String.is_prefix name ~prefix:"fold" || String.is_prefix name ~prefix:"folding_" ->
          ((if pairwise then 2 else 1), [ "init" ])
      | _ -> ((if pairwise then 2 else 1), [])
    in
    positional >= required_positional && List.for_all required_labels ~f:has

(* The name of the marker an [open] of a collection module leaves in scope, and of that module
   there: no source can spell it. *)
let opened_collection = " opened collection"

(* A function's frame, from its parameters. *)
let function_kind ~binding params body =
  let parameters =
    List.filter_map params ~f:(fun param ->
        match param.pparam_desc with
        | Pparam_val (label, _, _) -> Some label
        | Pparam_newtype _ -> None)
    @ match body with Pfunction_cases _ -> [ Ppxlib.Asttypes.Nolabel ] | Pfunction_body _ -> []
  in
  let rec erasable = function
    | [] -> true
    | Ppxlib.Asttypes.Optional _ :: rest ->
        List.exists rest ~f:(function Ppxlib.Asttypes.Nolabel -> true | _ -> false)
        && erasable rest
    | _ :: rest -> erasable rest
  in
  Function
    {
      binding;
      positional =
        List.count parameters ~f:(function Ppxlib.Asttypes.Nolabel -> true | _ -> false);
      labelled =
        List.filter_map parameters ~f:(function
          | Ppxlib.Asttypes.Labelled label -> Some label
          | _ -> None);
      erasable = erasable parameters;
    }

(* Whether arguments with these labels apply a function of [kind] fully. *)
let fully_applies kind labels =
  match kind with
  | Function { positional; labelled; erasable; _ } ->
      erasable
      && List.count labels ~f:(function Ppxlib.Asttypes.Nolabel -> true | _ -> false)
         >= positional
      && List.for_all labelled ~f:(fun label ->
          List.exists labels ~f:(function
            | Ppxlib.Asttypes.Labelled given -> String.equal given label
            | _ -> false))
  | Loop | Deferred -> false

(* [precedes ~once write claim]: whether [write] may run before [claim] ends, [once] giving the call
   a once function's frame runs at. *)
let precedes ~once ~unordered write claim =
  (* The levels a point runs at, outermost first: each the point (a call site, at every level but
     the innermost) with the frames it is inside there, and the once frame it was entered
     through. *)
  let rec chain depth point =
    let rec split inner = function
      | [] -> (inner, None)
      | frame :: rest -> (
          match once frame with
          | Some call when depth < 64 -> (inner, Some (frame, call))
          | _ -> split (frame :: inner) rest)
    in
    let frames, outer = split [] point.enclosing in
    match outer with
    | None -> [ (None, point, frames) ]
    | Some (frame, call) -> chain (depth + 1) call @ [ (Some frame.frame_id, point, frames) ]
  in
  (* A level's frames are those a point is inside below the once frame it was entered through, so
     every one of them may repeat; all but a loop may also run at any later time. *)
  let deferred frame = match frame.shape with Loop -> false | Function _ | Deferred -> true in
  let inside frames frame = List.exists frames ~f:(fun f -> f.frame_id = frame.frame_id) in
  let rec compare_levels writes claims =
    match (writes, claims) with
    | (_, write, write_frames) :: deeper_writes, (_, claim, claim_frames) :: deeper_claims -> (
        if List.exists write_frames ~f:(inside claim_frames) then true
        else if List.exists claim_frames ~f:(fun f -> deferred f && not (inside write_frames f))
        then true
        else
          match (deeper_writes, deeper_claims) with
          | (Some into_write, _, _) :: _, (Some into_claim, _, _) :: _ when into_write = into_claim
            ->
              compare_levels deeper_writes deeper_claims
          | _ ->
              write.effect_at < claim.at_end
              (* A claim in a callback the writer calls during the call. *)
              || List.exists claim_frames ~f:(fun f ->
                  write.at_start <= f.frame_start && f.frame_stop <= write.at_end)
              || unordered write claim)
    | _ -> true
  in
  compare_levels (chain 0 write) (chain 0 claim)

type caller = {
  source : string;
  text : string;
  spelled_elsewhere : bool;
      (** Some other source spells this one's module, so a module this one re-exports -- an alias or
          an [include] in its structure -- is reachable from there. *)
}
(** A source that may apply another's functions, for {!gate_census} (gh-ocannl-2286). *)

type callers = { module_name : string; sources : caller list }
(** The sources that may apply this one's functions: the module [M] this source is compiled as, and
    each other source that spells [M]. *)

(* [text] parsed as {!Read.structure_of} does, its character positions starting at [offset]: a
   second source walked beside the first shares none of its positions, so none of the identities and
   tables keyed by them. Lines stay the source's own. *)
let structure_at ~offset text =
  let lexbuf = Lexing.from_string text in
  lexbuf.lex_abs_pos <- offset;
  lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_cnum = offset; pos_bol = offset };
  Read.inline_instantiation (Ppxlib.Parse.implementation lexbuf)

(* How [structure] spells the module [module_name] in a way that may hide an application of its
   functions from {!gate_census}, where it does. The walk resolves [M.f] and [M.Sub.f], a name an
   [open M] (or [M.( ... )], [let open M in]) brings into scope, and a module alias of [M] ([module
   X = M], [let module X = M in]) or an [include M], so none of those is a way. A top-level or
   nested alias or [include] re-exports [M]'s functions, though, under the caller's own module,
   which a third source may spell ([Caller.X.f]) where no index entry for [M] sees it: where
   [spelled_elsewhere], that is a way. So is [M] packed, a functor's argument or under a constraint,
   and [M] inside another module's path ([Lib.M.f]): the walk follows none of them. *)
let module_spelled_otherwise ~module_name ~spelled_elsewhere structure =
  let found = ref None in
  let note how = if Option.is_none !found then found := Some how in
  (* Every component, a functor application's included: [F (M).x] spells [M] too. *)
  let rec components = function
    | Ppxlib.Longident.Lident name -> [ name ]
    | Ldot (prefix, name) -> components prefix @ [ name ]
    | Lapply (functor_, argument) -> components functor_ @ components argument
  in
  let spells path = List.mem (components path) module_name ~equal:String.equal in
  (* The one path the walk binds: [M] itself. [open Lib.M] is no such path. *)
  let direct module_expr =
    match module_expr.pmod_desc with
    | Pmod_ident { txt = Ppxlib.Longident.Lident name; _ } -> String.equal name module_name
    | _ -> false
  in
  let reexported what =
    if spelled_elsewhere then
      note
        (Printf.sprintf
           "%s, which re-exports its functions under that source's module, which a third source \
            spells"
           what)
  in
  let scan =
    object
      inherit Ast_traverse.iter as super

      method! open_declaration declaration =
        if not (direct declaration.popen_expr) then super#open_declaration declaration

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module { pmb_expr; pmb_name; _ } when direct pmb_expr ->
            reexported (Printf.sprintf "as the alias `%s`" (Option.value pmb_name.txt ~default:"_"))
        | Pstr_include { pincl_mod; _ } when direct pincl_mod -> reexported "in an `include`"
        | _ -> super#structure_item item

      method! module_expr module_expr =
        (match module_expr.pmod_desc with
        | Pmod_ident { txt; _ } when spells txt ->
            note
              "as a module the reader does not follow (packed, a functor's argument, constrained, \
               or opened, aliased or included through another module's path)"
        | _ -> ());
        super#module_expr module_expr

      method! expression expr =
        match expr.pexp_desc with
        | Pexp_letmodule (_, module_expr, body) when direct module_expr -> super#expression body
        | Pexp_ident { txt; _ }
          when spells txt && not (String.equal (Lexical_scope.head txt) module_name) ->
            note
              (Printf.sprintf "inside another module's path (`%s`)"
                 (String.concat ~sep:"." (components txt)))
        | _ -> super#expression expr
    end
  in
  scan#structure structure;
  !found

(** Every module name [content] spells in a path -- a value's, a module's, a type's, a constructor's
    or a field's, a functor application's components included -- for the ratchet's cross-source
    index of who may apply whose functions (gh-ocannl-2286). Raises where [content] does not parse.
*)
let module_mentions content =
  let found = Hash_set.create (module String) in
  let scan =
    object
      inherit Ast_traverse.iter as super

      method! longident path =
        let rec add = function
          | Ppxlib.Longident.Lident name -> Hash_set.add found name
          | Ldot (prefix, name) ->
              add prefix;
              Hash_set.add found name
          | Lapply (functor_, argument) ->
              add functor_;
              add argument
        in
        add path;
        super#longident path
    end
  in
  scan#structure (Read.structure_of content);
  List.sort (Hash_set.to_list found) ~compare:String.compare

let gate_census ?callers ?(definitions = fun _ _ -> []) content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  let readings = ref [] and gated_sites = ref 0 in
  let leg_sites = ref 0 and unread_legs = ref [] and uncompared_legs = ref [] in
  (* The positions of the leg-claim callees read as applied, which their own visit then passes. *)
  let leg_heads = Hash_set.create (module Int) in
  (* The make function of each leg claim whose comparison is queued, by its position: walked with
     its collector innermost, so that only what [make] does -- not the [~leg] or the label -- is the
     claim's population. *)
  let leg_makes = Hashtbl.create (module Int) in
  (* What is written into a binding after it is made -- [x := e], [Queue.enqueue x e] -- by the
     binding's identity, with the scope it was written in. *)
  let writes = Hashtbl.create (module Int) in
  (* A call of an unlisted function of a mutable collection module that was handed a binding, by the
     binding's identity: it may write into it, which the reader cannot follow. *)
  (* [record.field <- v], by the record binding's identity: the field and what is written. A field
     write leaves the record's other fields their literal components ({!parts}). *)
  let field_writes = Hashtbl.create (module Int) in
  let piped = Hash_set.create (module Int) in
  let unlisted_writes = Hashtbl.create (module Int) in
  let unlisted = ref [] in
  (* For {!precedes}: where each local [let]-bound function's name is spelled, by the binding's
     identity; each application of a name, by the name's position, with its labels and where it
     runs; the functions handed to a collection-module function, by position; and the local
     [let]-bound functions, by position, with their binding's identity. *)
  let references = Hashtbl.create (module Int) in
  let applications = Hashtbl.create (module Int) in
  let synchronous = Hash_set.create (module Int) in
  let function_bindings = Hashtbl.create (module Int) in
  (* Every named function of the source -- a [let], top-level or local, recursive or not, whose
     right-hand side is a function -- by the function's position: its binding's identity and name
     (gh-ocannl-1893). *)
  let named_functions = Hashtbl.create (module Int) in
  (* The anonymous callbacks an {!element_iterations} function is handed in place, by the callback's
     position: the index of its element parameter among its positional ones, the iteration as
     written, and the collection with the scope it is written in (gh-ocannl-2286). *)
  let element_callbacks = Hashtbl.create (module Int) in
  (* Each element callback's owner identity, with its collection and the scope that is written in;
     and the element callbacks enclosing each application of a name, by where the name is spelled
     (gh-ocannl-2286, local review). *)
  let element_owners = Hashtbl.create (module Int) in
  (* Where each caller's positions lie ({!structure_at}), with its name: the callers a choice of
     applications reaches (gh-ocannl-2286, review). *)
  let caller_spans = ref [] in
  let callers_in calls =
    List.filter_map calls ~f:(fun (_, at) ->
        List.find_map !caller_spans ~f:(fun (start, stop, source) ->
            if start <= at && at < stop then Some source else None))
  in
  let site_elements = Hashtbl.create (module Int) in
  (* Where the next source walked or parsed beside this one starts: past every earlier one, so that
     none shares another's positions ({!structure_at}). *)
  let next_offset = ref (String.length content + 1) in
  (* Each source compiled as a module a leg is built by a function of (gh-ocannl-2336), parsed once:
     the structure it is, or why the reader does not follow its functions. *)
  let defining_structures = Hashtbl.create (module String) in
  (* The fresh bindings a structure exports in place of the ones it exported before an [include]
     whose exports this reader does not know, which may rebind them (gh-ocannl-2336, review). *)
  let rebindable = Hash_set.create (module Int) in
  (* Where the sources read as definitions lie: their applications are no call sites, so a write
     there is not timed against a claim (gh-ocannl-2336, review). *)
  let definition_spans = ref [] in
  (* The named functions of those sources that write into a binding, or apply one that does
     (gh-ocannl-2336, review). *)
  let mutating_functions = Hash_set.create (module Int) in
  (* The callers walked, by source: a caller that also defines a module a leg is built by a function
     of is read as it was walked, not walked again. *)
  let walked_callers = Hashtbl.create (module String) in
  let empty_scope = lazy { Lexical_scope.frames = []; modules = Map.empty (module String) } in
  (* The functions of this source whose applications another source may hide, by the function's
     binding identity, with why: that source spells the module other than as [M.f]
     (gh-ocannl-2286). *)
  let foreign_escapes = Hashtbl.create (module Int) in
  (* Every application of a name the source binds, by the binding's identity: where the name is
     spelled, the arguments, and the scope they are written in. *)
  let call_sites = Hashtbl.create (module Int) in
  (* The named function each application of a name is written in, by where the name is spelled: what
     a parameter forwarded to it is resolved through. *)
  let call_callers = Hashtbl.create (module Int) in
  (* Whether two choices of applications agree on every function both choose for. *)
  (* Past 256 resolutions a leg is uncompared, not enumerated further: a tree of forwarding helpers
     could otherwise make it exponential. *)
  let bounded_resolutions resolutions =
    if List.length resolutions > 256 then
      Error "its leg reaches more than 256 choices of the applications it passes through"
    else Ok resolutions
  in
  let calls_compatible a b =
    List.for_all a ~f:(fun (owner, at) ->
        match List.Assoc.find b owner ~equal:Int.equal with
        | Some other -> at = other
        | None -> true)
  in
  (* The parameters of the named functions, by the parameter's identity. *)
  let parameters = Hashtbl.create (module Int) in
  let next_frame = ref 0 in
  let once_calls = Hashtbl.create (module Int) in
  (* For {!precedes}: the operand spans of every expression whose operands are evaluated in an
     unspecified order, and whether two points sit in different operands of one. *)
  let unordered_operands = ref [] in
  let siblings write claim =
    let inside (start, stop) point = start <= point.at_start && point.at_end <= stop in
    List.exists !unordered_operands ~f:(fun operands ->
        List.existsi operands ~f:(fun i operand ->
            inside operand claim
            && List.existsi operands ~f:(fun j other -> i <> j && inside other write)))
  in
  (* The call a once function's frame runs at, if it is once; asked only after the walk. *)
  let once frame =
    match frame.shape with
    | Function { binding = Some identity; _ } ->
        Hashtbl.find_or_add once_calls frame.frame_id ~default:(fun () ->
            match Hashtbl.find_multi references identity with
            | [ spelled ] when spelled < frame.frame_start || spelled >= frame.frame_stop -> (
                match Hashtbl.find applications spelled with
                | Some (labels, call) when fully_applies frame.shape labels -> Some call
                | _ -> None)
            | _ -> None)
    | _ -> None
  in
  (* Claims are compared once the walk is over, when every write is known. *)
  let pending = ref [] in
  let partials = ref [] and bound_partials = ref [] and consumed = Hash_set.create (module Int) in
  let position expr = expr.pexp_loc.loc_start.pos_cnum in
  let span_of expr = (expr.pexp_loc.loc_start.pos_cnum, expr.pexp_loc.loc_end.pos_cnum) in
  (* The nodes a claim's resolution went through, made [consumed] only once the claim is read --
     compared, or reported unread -- so that a binding no read claim reaches stays uncompleted. *)
  let commit marks = Hash_set.iter marks ~f:(Hash_set.add consumed) in
  (* The right-hand side of every [let], under any type constraint: a gated claim bound there,
     partially applied or merely renamed, is read at the uses of the name it is bound to. *)
  let bound = Hash_set.create (module Int) in
  let rec bind expr =
    Hash_set.add bound (position expr);
    match expr.pexp_desc with Pexp_constraint (inner, _) -> bind inner | _ -> ()
  in
  (* The right-hand sides of [let]s that bind a name, under any type constraint: a writer whose
     value is bound to a name ([let put = Array.set rows 0]) may be partially applied. *)
  let named = Hash_set.create (module Int) in
  let rec name_bound expr =
    Hash_set.add named (position expr);
    match expr.pexp_desc with Pexp_constraint (inner, _) -> name_bound inner | _ -> ()
  in
  let binder =
    object
      inherit Ast_traverse.iter as super

      method! value_binding binding =
        bind binding.pvb_expr;
        if Option.is_some (Read.pattern_name binding.pvb_pat) then name_bound binding.pvb_expr;
        (match (strip_constraints binding.pvb_expr, Read.pattern_name binding.pvb_pat) with
        | ({ pexp_desc = Pexp_function _; _ } as function_), Some name ->
            Hashtbl.set named_functions ~key:(position function_)
              ~data:(binding.pvb_loc.loc_start.pos_cnum + 1, name)
        | _ -> ());
        super#value_binding binding

      (* A structure's [let ... and ...] evaluates its right-hand sides in an unspecified order, as
         an expression's does ({!siblings}). *)
      method! structure_item item =
        (match item.pstr_desc with
        | Pstr_value (_, (_ :: _ :: _ as bindings)) ->
            unordered_operands :=
              List.map bindings ~f:(fun binding ->
                  ( binding.pvb_expr.pexp_loc.loc_start.pos_cnum,
                    binding.pvb_expr.pexp_loc.loc_end.pos_cnum ))
              :: !unordered_operands
        | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_let (Nonrecursive, bindings, _) ->
            List.iter bindings ~f:(fun binding ->
                match (strip_constraints binding.pvb_expr, Read.pattern_name binding.pvb_pat) with
                | ({ pexp_desc = Pexp_function _; _ } as function_), Some _ ->
                    Hashtbl.set function_bindings ~key:(position function_)
                      ~data:(binding.pvb_loc.loc_start.pos_cnum + 1)
                | _ -> ())
        | _ -> ());
        super#expression expr
    end
  in
  binder#structure ast;
  let walker =
    object (self)
      inherit [gate_denotation, gate_module] Lexical_scope.scoped as super
      val mutable next_local = 0
      val mutable quiet = 0

      (* Above zero while a function is walked again for the populations of the leg claims its
         claims belong to ({!replay}): nothing else is recorded then. *)
      val mutable replaying = 0

      (* The named function whose parameters are being bound, with the slot of each, by the
         parameter pattern's position. *)
      val mutable binding_function
          : (int
            * string
            * ( int,
                [ `Positional of int
                | `Labelled of string
                | `Optional of string * expression option ] )
              Hashtbl.t)
            option =
        None

      (* The element parameter of the callback being bound, where an {!element_iterations} function
         is handed it in place: the parameter pattern's position, the identity a resolution chooses
         an element under, the iteration as written, and the collection with its scope
         (gh-ocannl-2286). *)
      val mutable element_binding
          : (int * int * string * expression * (gate_denotation, gate_module) Lexical_scope.env)
            option =
        None

      (* While another source's structure is walked for its applications of this one's functions
         (gh-ocannl-2286): what it records is read, and the claims it makes are its own reader's. *)
      val mutable foreign = false

      (* The parameters {!alternatives} is expanding, and those {!parameter_arguments} is finding
         the values of, against one that reaches itself. *)
      val expanding : int Hash_set.t = Hash_set.create (module Int)
      val arguing : int Hash_set.t = Hash_set.create (module Int)

      (* The applications a comparison is made under ({!leg_resolution}): a parameter of one of
         these functions is the argument of that application alone. *)
      val mutable context : (int * int) list = []

      (* The named functions enclosing the node being walked, innermost first. *)
      val mutable enclosing_functions : int list = []

      (* The element callbacks enclosing the node being walked, by their {!Element_slot} owner,
         innermost first (gh-ocannl-2286, local review). *)
      val mutable enclosing_elements : int list = []

      (* The definitions being replayed, against a function that calls itself. *)
      val replayed : int Hash_set.t = Hash_set.create (module Int)

      (* In a replay, the functions handed to a leg claim met there: its make function, whose claims
         are the replayed make function's too, so the collectors do not pause in it. *)
      val replay_live : int Hash_set.t = Hash_set.create (module Int)
      val mutable collecting : (string * gate_denotation) list ref option = None

      (* The unlisted collection-module calls the claim being compared reaches. *)
      val mutable unlisted_sink : (string * int) list ref = ref []

      (* The frames around the node being walked, innermost first, and where the claim being
         compared is evaluated. *)
      val mutable stack : frame list = []
      val mutable claim_point : point option = None

      (* The leg claims whose [make] is being walked, innermost first: a claim met there adds its
         population to the innermost one. *)
      val mutable leg_stack : leg_collector list = []

      (* The bodies of the [for] loops and binding operators being walked, by position: the frame
         each enters. *)
      val body_frames : (int, frame_kind) Hashtbl.t = Hashtbl.create (module Int)

      (* Whether the walk records what it meets: not inside {!identifiers}, nor in a replay. *)
      method recording = quiet = 0 && replaying = 0

      (* [walk ()] with the claims it meets left to their own source's reader ({!foreign}). *)
      method walk_foreign : 'a. (unit -> 'a) -> 'a =
        fun walk ->
          foreign <- true;
          Exn.protect ~f:walk ~finally:(fun () -> foreign <- false)

      method point_of expr =
        {
          at_start = expr.pexp_loc.loc_start.pos_cnum;
          at_end = expr.pexp_loc.loc_end.pos_cnum;
          effect_at = expr.pexp_loc.loc_start.pos_cnum;
          enclosing = stack;
        }

      (* Whether something done at [point] may run before the claim being compared. *)
      method before point =
        match claim_point with
        | None -> true
        | Some claim -> precedes ~once ~unordered:siblings point claim

      method framed : 'a. kind:frame_kind -> start:int -> stop:int -> (unit -> 'a) -> 'a =
        fun ~kind ~start ~stop walk ->
          if quiet > 0 || replaying > 0 then walk ()
          else (
            Int.incr next_frame;
            stack <-
              { frame_id = !next_frame; shape = kind; frame_start = start; frame_stop = stop }
              :: stack;
            Exn.protect ~f:walk ~finally:(fun () -> stack <- List.tl_exn stack))

      method! class_structure env structure =
        self#framed ~kind:Deferred ~start:0 ~stop:0 (fun () -> super#class_structure env structure)

      method! extension env extension =
        let { Ppxlib.loc; _ } = fst extension in
        self#framed ~kind:Deferred ~start:loc.loc_start.pos_cnum ~stop:loc.loc_end.pos_cnum
          (fun () -> super#extension env extension)

      method! module_expr env module_expr =
        match module_expr.pmod_desc with
        | Pmod_functor _ ->
            self#framed ~kind:Deferred ~start:module_expr.pmod_loc.loc_start.pos_cnum
              ~stop:module_expr.pmod_loc.loc_end.pos_cnum (fun () ->
                super#module_expr env module_expr)
        | _ -> super#module_expr env module_expr

      (* While a named function's parameters are bound and its body walked: the slot each parameter
         takes its value from at an application ({!parameters}), and the function as the innermost
         enclosing one (gh-ocannl-1893). *)
      method function_scope : 'a. expression -> (unit -> 'a) -> 'a =
        fun expr walk ->
          let saved_binding = binding_function and saved_enclosing = enclosing_functions in
          let saved_element = element_binding and saved_elements = enclosing_elements in
          element_binding <-
            (match (expr.pexp_desc, Hashtbl.find element_callbacks (position expr)) with
            | Pexp_function (params, _, _), Some (index, iteration, collection, collection_env) ->
                List.filter_map params ~f:(fun param ->
                    match param.pparam_desc with
                    | Pparam_val (Ppxlib.Asttypes.Nolabel, _, pattern) -> Some pattern
                    | _ -> None)
                |> Fn.flip List.nth index
                |> Option.map ~f:(fun pattern ->
                    let owner = self#local.identity in
                    Hashtbl.set element_owners ~key:owner ~data:(collection, collection_env);
                    enclosing_elements <- owner :: enclosing_elements;
                    ( pattern.ppat_loc.loc_start.pos_cnum,
                      owner,
                      iteration,
                      collection,
                      collection_env ))
            | _ -> None);
          (match (expr.pexp_desc, Hashtbl.find named_functions (position expr)) with
          | Pexp_function (params, _, _), Some (owner, name) ->
              let slots = Hashtbl.create (module Int) in
              let positional = ref 0 in
              List.iter params ~f:(fun param ->
                  match param.pparam_desc with
                  | Pparam_val (label, default, pattern) ->
                      let slot =
                        match label with
                        | Ppxlib.Asttypes.Nolabel ->
                            let index = !positional in
                            Int.incr positional;
                            `Positional index
                        | Labelled label -> `Labelled label
                        | Optional label -> `Optional (label, default)
                      in
                      Hashtbl.set slots ~key:pattern.ppat_loc.loc_start.pos_cnum ~data:slot
                  | Pparam_newtype _ -> ());
              binding_function <- Some (owner, name, slots);
              enclosing_functions <- owner :: enclosing_functions
          | _ -> binding_function <- None);
          Exn.protect ~f:walk ~finally:(fun () ->
              binding_function <- saved_binding;
              element_binding <- saved_element;
              enclosing_elements <- saved_elements;
              enclosing_functions <- saved_enclosing)

      method! bind_parameter env label pattern =
        let at = pattern.ppat_loc.loc_start.pos_cnum in
        (* Where the parameter is registered ({!parameters}): a named function's, with its slot, or
           the element parameter of a callback an {!element_iterations} function is handed. *)
        let registered =
          if not self#recording then None
          else
            match (binding_function, element_binding) with
            | Some (owner, owner_name, slots), _ when Hashtbl.mem slots at ->
                let slot =
                  match Hashtbl.find_exn slots at with
                  | `Positional index -> Positional_slot index
                  | `Labelled label -> Labelled_slot label
                  | `Optional (label, default) ->
                      (* The default is evaluated where the parameter is bound. *)
                      Optional_slot (label, Option.map default ~f:(fun d -> (d, env)))
                in
                Some (owner, owner_name, slot)
            | _, Some (element_at, owner, owner_name, collection, collection_env)
              when element_at = at ->
                Some (owner, owner_name, Element_slot (collection, collection_env))
            | _ -> None
        in
        match (registered, Read.pattern_name pattern) with
        | None, _ -> super#bind_parameter env label pattern
        | Some (owner, owner_name, slot), Some name ->
            let inner = super#bind_parameter env label pattern in
            (match Lexical_scope.lookup inner name with
            | Some { identity; _ } ->
                Hashtbl.set parameters ~key:identity ~data:{ owner; owner_name; slot; part = None }
            | None -> ());
            inner
        | Some (owner, owner_name, slot), None ->
            (* A destructured parameter (gh-ocannl-2286): each name a part of what the application
               supplies, taken apart by the whole pattern ({!parts}), so each has a binding of its
               own. *)
            let names = Lexical_scope.pattern_vars [ pattern ] in
            let inner =
              self#forget
                (List.fold names ~init:env ~f:(fun scope name ->
                     let denotation = self#local in
                     Hashtbl.set parameters ~key:denotation.identity
                       ~data:{ owner; owner_name; slot; part = Some (pattern, name) };
                     Lexical_scope.bind_values scope [ name ] denotation))
                (Lexical_scope.pattern_unpacks [ pattern ])
            in
            inner

      method note_unlisted callee =
        let spelled =
          Option.value_map
            (Read.longident_of (strip_constraints callee))
            ~default:"?" ~f:(String.concat ~sep:".")
        in
        unlisted_sink := (spelled, callee.pexp_loc.loc_start.pos_lnum) :: !unlisted_sink

      (* What is known to reach a binding besides its definition: what the source writes into it,
         and any unlisted call it was handed. *)
      method written identity =
        List.iter (Hashtbl.find_multi unlisted_writes identity) ~f:(fun (call, point) ->
            if self#before point then unlisted_sink := call :: !unlisted_sink);
        List.filter_map
          (Hashtbl.find_multi writes identity
          @ List.map (Hashtbl.find_multi field_writes identity) ~f:(fun (_, written) -> written))
          ~f:(fun (written, env, point) -> if self#before point then Some (written, env) else None)

      method local =
        next_local <- next_local - 1;
        { identity = next_local; definition = None }

      (* The binding [prefix.name] denotes, where [prefix] is a structure of this source -- or,
         while another source is walked, this source itself ({!foreign}) -- and [name] one of its
         values with a binding of the source (gh-ocannl-2286). *)
      method structure_value env prefix name =
        match self#module_path env prefix with
        | Some (Local_structure { values; _ }) -> (
            match List.Assoc.find values name ~equal:String.equal with
            | Some { identity; _ } when identity > 0 -> Some identity
            | _ -> None)
        | _ -> None

      (* What a module path denotes to the population side: a collection module, a standard library,
         a structure of this source, or another module. Whether a claim is [Verdict]'s is
         {!Verdict_callee}'s question, not this one. *)
      method module_path env path =
        let components = Read.flatten_longident path in
        let head = Lexical_scope.head path in
        let single = List.length components = 1 in
        Some
          (match Map.find env.Lexical_scope.modules head with
          | Some denotes when single -> denotes
          | Some (Collection_module _ as collection)
            when List.for_all (List.tl_exn components) ~f:(fun c ->
                     List.mem collection_modules c ~equal:String.equal) ->
              (* [List.Assoc] under an [open Base]. *)
              collection
          | Some (Local_structure _ as structure) ->
              (* [Collections.Rows] for [module Collections = struct module Rows = List end]. *)
              let rec descend denotes = function
                | [] -> denotes
                | component :: rest -> (
                    match denotes with
                    | Local_structure { modules; _ } -> (
                        match List.Assoc.find modules component ~equal:String.equal with
                        | Some inner -> descend inner rest
                        | None -> Other_module)
                    | Collection_module _
                      when List.for_all (component :: rest) ~f:(fun c ->
                               List.mem collection_modules c ~equal:String.equal) ->
                        denotes
                    | _ -> Other_module)
              in
              descend structure (List.tl_exn components)
          | Some (Standard_library { stdlib })
            when List.for_all (List.tl_exn components) ~f:(fun c ->
                     List.mem collection_modules c ~equal:String.equal) ->
              (* [B.List] under [module B = Base]. *)
              Collection_module
                {
                  stdlib;
                  mutable_ = List.mem mutable_modules (List.last_exn components) ~equal:String.equal;
                  family = List.last_exn components;
                }
          | Some _ -> Other_module
          | None when single && List.mem [ "Base"; "Core" ] head ~equal:String.equal ->
              Standard_library { stdlib = false }
          | None when single && List.mem [ "Stdlib"; "Caml" ] head ~equal:String.equal ->
              Standard_library { stdlib = true }
          | None ->
              (* [List], [Base.List], [Stdlib.List], [List.Assoc]: a path through the standard
                 libraries' collection modules alone. *)
              let standard = [ "Base"; "Core"; "Stdlib"; "Caml" ] in
              if
                List.for_all components ~f:(fun c ->
                    List.mem collection_modules c ~equal:String.equal
                    || List.mem standard c ~equal:String.equal)
                && List.mem collection_modules (List.last_exn components) ~equal:String.equal
              then
                Collection_module
                  {
                    (* An unqualified [List] is [Stdlib]'s unless an [open Base] in scope rebound
                       it, which {!opened} records. *)
                    stdlib = not (List.mem [ "Base"; "Core" ] head ~equal:String.equal);
                    mutable_ =
                      List.mem mutable_modules (List.last_exn components) ~equal:String.equal;
                    family = List.last_exn components;
                  }
              else Other_module)

      method! shadowed = Some Other_module

      (* A structure's exports, each with what it denotes where it is declared: the scope is built
         up item by item as {!Lexical_scope.scoped} builds it -- bindings, [external]s, modules
         (recursive ones shadowing their names), opens and includes, an included structure's exports
         joining its own -- without walking anything, which the main walk does once. A later [open]
         changes the scope but exports nothing, so an export keeps the denotation its declaration
         gave it; a later declaration of the same name replaces it. *)
      method! module_of env module_expr =
        match module_expr.pmod_desc with
        | Pmod_structure items ->
            let declared env names =
              List.map names ~f:(fun name ->
                  ( name,
                    Option.value_or_thunk (Lexical_scope.lookup env name) ~default:(fun () ->
                        self#local) ))
            in
            let module_exports = ref [] and opaque = ref false in
            let declared_modules env names =
              module_exports :=
                !module_exports
                @ List.map names ~f:(fun name ->
                    ( name,
                      Option.value (Map.find env.Lexical_scope.modules name) ~default:Other_module
                    ))
            in
            let _, exports =
              List.fold items ~init:(env, []) ~f:(fun (env, exports) item ->
                  match item.pstr_desc with
                  | Pstr_value (rec_flag, bindings) ->
                      let env =
                        self#bind_group env bindings (self#denotations env rec_flag bindings)
                      in
                      ( env,
                        exports
                        @ declared env
                            (Lexical_scope.pattern_vars (List.map bindings ~f:(fun b -> b.pvb_pat)))
                      )
                  | Pstr_primitive { pval_name = { txt; _ }; _ } ->
                      let env = Lexical_scope.bind_values env [ txt ] self#local in
                      (env, exports @ declared env [ txt ])
                  | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
                      let env = self#bind_module env name pmb_expr in
                      declared_modules env [ name ];
                      (env, exports)
                  | Pstr_recmodule declarations ->
                      let names = List.filter_map declarations ~f:(fun d -> d.pmb_name.txt) in
                      let env = self#forget env names in
                      declared_modules env names;
                      (env, exports)
                  | Pstr_open { popen_expr; _ } ->
                      ( self#opened ~top:false ~include_:false env (self#module_of env popen_expr),
                        exports )
                  | Pstr_include { pincl_mod; _ } -> (
                      let denotes = self#module_of env pincl_mod in
                      (* An include whose exports are not all known may rebind any name this
                         structure exported so far, its submodules' included: each becomes a binding
                         of its own the reader knows nothing of (gh-ocannl-2336, review), in this
                         structure's exports alone -- the structure it came from keeps its own. *)
                      let unknown exports =
                        opaque := true;
                        let rec forget values modules =
                          ( List.map values ~f:(fun (name, _) ->
                                let unknown = self#local in
                                Hash_set.add rebindable unknown.identity;
                                (name, unknown)),
                            List.map modules ~f:(function
                              | name, Local_structure { values; modules; _ } ->
                                  let values, modules = forget values modules in
                                  (name, Local_structure { values; modules; opaque = true })
                              | other -> other) )
                        in
                        let values, modules = forget exports !module_exports in
                        module_exports := modules;
                        values
                      in
                      let opaque_include =
                        match denotes with
                        | Some (Local_structure { opaque = included_opaque; _ }) -> included_opaque
                        | _ -> true
                      in
                      let exports = if opaque_include then unknown exports else exports in
                      (* What follows the include in this structure sees the same fresh bindings: a
                         later definition's use of an earlier name may be the include's (review). *)
                      let env =
                        if opaque_include then
                          let env =
                            Lexical_scope.push_frame env
                              (Map.of_alist_reduce
                                 (module String)
                                 exports
                                 ~f:(fun _ later -> later))
                          in
                          (* And its submodules as they now are, forgotten (review). *)
                          {
                            env with
                            Lexical_scope.modules =
                              List.fold !module_exports ~init:env.Lexical_scope.modules
                                ~f:(fun modules (name, denotes) ->
                                  Map.set modules ~key:name ~data:denotes);
                          }
                        else env
                      in
                      ( self#opened ~top:false ~include_:true env denotes,
                        match denotes with
                        | Some (Local_structure { values; modules; _ }) ->
                            module_exports := !module_exports @ modules;
                            exports @ values
                        | _ -> exports ))
                  | _ -> (env, exports))
            in
            let latest pairs =
              Map.to_alist (Map.of_alist_reduce (module String) pairs ~f:(fun _ later -> later))
            in
            Some
              (Local_structure
                 { values = latest exports; modules = latest !module_exports; opaque = !opaque })
        | _ -> super#module_of env module_expr

      (* An [open Base] rebinds every collection module to [Base]'s, and an [open Stdlib] back to
         [Stdlib]'s, from there to the end of its scope. *)
      method! opened ~top:_ ~include_:_ env denotes =
        match denotes with
        | Some (Standard_library { stdlib }) ->
            {
              env with
              Lexical_scope.modules =
                List.fold collection_modules ~init:env.Lexical_scope.modules ~f:(fun modules name ->
                    Map.set modules ~key:name
                      ~data:
                        (Collection_module
                           {
                             stdlib;
                             mutable_ = List.mem mutable_modules name ~equal:String.equal;
                             family = name;
                           }));
            }
        | Some (Local_structure { values; modules; _ }) ->
            (* Its names shadow whatever they denoted, a [gated_all] of its own included, and denote
               what the structure defines them as; so do its modules. *)
            let env =
              {
                env with
                Lexical_scope.modules =
                  List.fold modules ~init:env.Lexical_scope.modules ~f:(fun known (name, denotes) ->
                      Map.set known ~key:name ~data:denotes);
              }
            in
            Lexical_scope.push_frame env
              (Map.of_alist_reduce (module String) values ~f:(fun _ later -> later))
        | Some (Collection_module _ as collection) ->
            (* [let open List in map raw ~f], [List.(map raw ~f)]: a marker frame, under which an
               unqualified callee {!collection_functions} lists and nothing inside the open binds is
               the module's ({!resolve_callee}). *)
            Lexical_scope.push_frame env
              (Map.singleton
                 (module String)
                 opened_collection
                 {
                   identity = 0;
                   definition =
                     Some
                       ( Ppxlib.Ast_builder.Default.evar ~loc:Ppxlib.Location.none opened_collection,
                         {
                           env with
                           Lexical_scope.modules =
                             Map.set env.Lexical_scope.modules ~key:opened_collection
                               ~data:collection;
                         } );
                 })
        | Some Other_module | None ->
            (* A barrier for {!resolve_callee}: an unqualified name may be this module's. *)
            Lexical_scope.push_frame env
              (Map.singleton (module String) opened_collection { identity = 0; definition = None })

      method! binding_denotes env binding =
        let identity = binding.pvb_loc.loc_start.pos_cnum + 1 in
        match Read.pattern_name binding.pvb_pat with
        | Some _ -> { identity; definition = Some (binding.pvb_expr, env) }
        | None -> { identity; definition = None }

      (* A recursive group's names see one another's definitions: each definition's scope is the
         group's own, which exists only once the group's denotations do. *)
      method! denotations env rec_flag bindings =
        match rec_flag with
        | Nonrecursive -> super#denotations env rec_flag bindings
        | Recursive ->
            let denotes =
              List.map bindings ~f:(fun binding ->
                  { identity = binding.pvb_loc.loc_start.pos_cnum + 1; definition = None })
            in
            let scope = self#bind_group env bindings denotes in
            List.iter2_exn denotes bindings ~f:(fun denotation binding ->
                if Option.is_some (Read.pattern_name binding.pvb_pat) then
                  denotation.definition <- Some (binding.pvb_expr, scope));
            denotes

      method! attribute _ attr = attr

      method! ident env ident =
        (match (quiet + replaying, ident.txt) with
        | 0, Ppxlib.Longident.Lident name -> (
            match Lexical_scope.lookup env name with
            | Some { identity; _ } when identity > 0 ->
                Hashtbl.add_multi references ~key:identity ~data:ident.loc.loc_start.pos_cnum
            | _ -> ())
        | 0, Ppxlib.Longident.Ldot (prefix, name) -> (
            (* [M.f] spelled anywhere, [M] a structure of this source or this source itself
               (gh-ocannl-2286): a use of [f], which {!escapes} reads as {!call_sites} do. *)
            match self#structure_value env prefix name with
            | Some identity ->
                Hashtbl.add_multi references ~key:identity ~data:ident.loc.loc_start.pos_cnum
            | None -> ())
        | _ -> ());
        match (collecting, ident.txt) with
        | Some found, Ppxlib.Longident.Lident name ->
            (* A name no binding of this source gives, one an unmodelled [open] may have brought in,
               is still a name: identity 0, the same on both sides. *)
            let denotation =
              Option.value (Lexical_scope.lookup env name)
                ~default:{ identity = 0; definition = None }
            in
            found := (name, denotation) :: !found
        | _ -> ()

      (* Every value identifier [expr] spells, resolved in [env], without recording sites. *)
      method identifiers env expr =
        let saved = collecting in
        let found = ref [] in
        collecting <- Some found;
        quiet <- quiet + 1;
        Exn.protect
          ~f:(fun () -> ignore (self#expression env expr : expression))
          ~finally:(fun () ->
            collecting <- saved;
            quiet <- quiet - 1);
        List.rev !found

      (* The gate's side: everything it spells, through every [let] it reaches. *)
      method gate_reads env expr =
        let seen = Hash_set.create (module Int) in
        let reads = Hash_set.Poly.create () in
        let rec visit env expr =
          List.iter (self#identifiers env expr) ~f:(fun (name, denotation) ->
              Hash_set.add reads (name, denotation.identity);
              if denotation.identity <> 0 && not (Hash_set.mem seen denotation.identity) then (
                Hash_set.add seen denotation.identity;
                Option.iter denotation.definition ~f:(fun (rhs, env) -> visit env rhs);
                List.iter (self#parameter_arguments denotation.identity) ~f:(fun (argument, env) ->
                    visit env argument);
                List.iter (self#written denotation.identity) ~f:(fun (written, env) ->
                    visit env written)))
        in
        visit env expr;
        reads

      (* What [expr] is, through the names that bind it and any type constraints: a literal tuple,
         record or constructor application where it is one. A name something writes into is not
         followed: its value is not its definition alone. *)
      method literal env expr =
        let rec go seen env expr =
          let expr = strip_constraints expr in
          match expr.pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
              match Lexical_scope.lookup env name with
              | Some { identity; definition = Some (rhs, rhs_env) }
                when (not (Set.mem seen identity))
                     && (not (Hashtbl.mem writes identity))
                     && not (Hashtbl.mem unlisted_writes identity) ->
                  go (Set.add seen identity) rhs_env rhs
              | Some { identity; definition = None } when not (Set.mem seen identity) -> (
                  (* A parameter is the argument of the one application it is read under
                     (gh-ocannl-1893), so a pattern takes that argument's literal apart. *)
                  match self#parameter_arguments identity with
                  | [ (argument, argument_env) ] -> go (Set.add seen identity) argument_env argument
                  | _ -> (expr, env, Set.to_list seen))
              | _ -> (expr, env, Set.to_list seen))
          | _ -> (expr, env, Set.to_list seen)
        in
        go (Set.empty (module Int)) env expr

      (* The names [pattern] binds, matched against [part], each bound in [env] to what it is a part
         of (gh-ocannl-1303): where [part] is a literal the pattern takes apart -- a tuple of the
         pattern's arity, a record, a constructor or polymorphic variant of the pattern's own name
         -- a name in a component's place is a part of that component alone, as
         [verdict_provenance.ml] reads destructured components; anywhere else, a part of the whole
         of [part]. So [match (on_gpu, raw) with _, xs -> xs] is built from [raw], not [on_gpu]. *)
      method parts ?(extra = []) pattern ((part, part_env) as whole) =
        let all pattern =
          List.map (Lexical_scope.pattern_vars [ pattern ]) ~f:(fun name -> (name, whole, extra))
        in
        let literal () = self#literal part_env part in
        let same_constructor a b =
          String.equal (Ppxlib.Longident.last_exn a) (Ppxlib.Longident.last_exn b)
        in
        match pattern.ppat_desc with
        | Ppat_var { txt; _ } -> [ (txt, whole, extra) ]
        | Ppat_alias (inner, { txt; _ }) -> (txt, whole, extra) :: self#parts ~extra inner whole
        | Ppat_constraint (inner, _) -> self#parts ~extra inner whole
        | Ppat_tuple patterns -> (
            match literal () with
            | { pexp_desc = Pexp_tuple components; _ }, env, _
              when List.length components = List.length patterns ->
                List.concat
                  (List.map2_exn patterns components ~f:(fun pattern component ->
                       self#parts ~extra pattern (component, env)))
            | _ -> all pattern)
        | Ppat_construct (constructor, Some (_, inner)) -> (
            match literal () with
            | { pexp_desc = Pexp_construct (built, Some payload); _ }, env, _
              when same_constructor constructor.txt built.txt ->
                self#parts ~extra inner (payload, env)
            | _ -> all pattern)
        | Ppat_variant (tag, Some inner) -> (
            match literal () with
            | { pexp_desc = Pexp_variant (built, Some payload); _ }, env, _
              when String.equal tag built ->
                self#parts ~extra inner (payload, env)
            | _ -> all pattern)
        | Ppat_record (fields, _) -> (
            match literal () with
            | { pexp_desc = Pexp_record (given, base); _ }, env, names ->
                (* [{ defaults with found = raw }]: a field given is its component; one the update
                   leaves is a part of the whole base. *)
                List.concat_map fields ~f:(fun (field, inner) ->
                    (* What [record.field <- v] wrote into this field of the names on the way. *)
                    let written =
                      List.concat_map names ~f:(fun identity ->
                          List.filter_map (Hashtbl.find_multi field_writes identity)
                            ~f:(fun (written_field, write) ->
                              if String.equal written_field (Ppxlib.Longident.last_exn field.txt)
                              then Some write
                              else None))
                    in
                    match
                      List.find given ~f:(fun (name, _) -> same_constructor field.txt name.txt)
                    with
                    | Some (_, component) ->
                        self#parts ~extra:(extra @ written) inner (component, env)
                    | None -> (
                        match base with
                        | Some base ->
                            (* The field the update leaves is the base's: taken apart where the base
                               is a literal too (gh-ocannl-2286, local review), a part of the whole
                               base otherwise. *)
                            self#parts ~extra:(extra @ written)
                              { pattern with ppat_desc = Ppat_record ([ (field, inner) ], Open) }
                              (base, env)
                        | None -> all inner))
            | _ -> all pattern)
        | _ -> all pattern

      method bind_parts env pattern whole =
        List.fold (self#parts pattern whole) ~init:env ~f:(fun env (name, part, extra) ->
            let denotation = { (self#local) with definition = Some part } in
            List.iter extra ~f:(fun write ->
                Hashtbl.add_multi writes ~key:denotation.identity ~data:write);
            Lexical_scope.bind_values env [ name ] denotation)

      (* The population's side: the collections it is built from. *)
      method population env expr =
        let seen = Hash_set.create (module Int) in
        let found = Hash_set.Poly.create () in
        let rec visit env expr =
          match expr.pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
              match Lexical_scope.lookup env name with
              | Some denotation ->
                  Hash_set.add found (name, denotation.identity);
                  if not (Hash_set.mem seen denotation.identity) then (
                    Hash_set.add seen denotation.identity;
                    Option.iter denotation.definition ~f:(fun (rhs, env) -> visit env rhs);
                    List.iter (self#parameter_arguments denotation.identity)
                      ~f:(fun (argument, env) -> visit env argument);
                    List.iter (self#written denotation.identity) ~f:(fun (written, env) ->
                        visit env written))
              | None -> Hash_set.add found (name, 0))
          | Pexp_constraint (inner, _)
          | Pexp_coerce (inner, _, _)
          | Pexp_sequence (_, inner)
          | Pexp_letexception (_, inner)
          | Pexp_newtype (_, inner)
          | Pexp_lazy inner ->
              (* A [lazy e] is the data thunk of [e] (gh-ocannl-1794: a leg's evidence lives in
                 lazies only its claims force). *)
              visit env inner
          | Pexp_open ({ popen_expr; _ }, inner) ->
              visit
                (self#opened ~top:false ~include_:false env (self#module_of env popen_expr))
                inner
          | Pexp_letmodule ({ txt = Some name; _ }, module_expr, inner) ->
              visit (self#bind_module env name module_expr) inner
          | Pexp_letmodule ({ txt = None; _ }, _, inner) -> visit env inner
          | Pexp_apply (callee, [ (Ppxlib.Asttypes.Nolabel, thunk) ]) when self#forces env callee ->
              (* [Lazy.force rows] is what [rows] holds. *)
              visit env thunk
          | Pexp_letop { let_; ands; body } ->
              (* [let+ x = e in body]: [x] is a part of [e], as a match case's name is of its
                 scrutinee. *)
              visit
                (List.fold (let_ :: ands) ~init:env ~f:(fun scope operand ->
                     Lexical_scope.bind_values scope
                       (Lexical_scope.pattern_vars [ operand.pbop_pat ])
                       { (self#local) with definition = Some (operand.pbop_exp, env) }))
                body
          | Pexp_ifthenelse (_, yes, no) ->
              visit env yes;
              Option.iter no ~f:(visit env)
          | Pexp_match (scrutinee, cases) ->
              (* The scrutinee steers; a name a case pattern binds is a part of it, so a case that
                 returns one is built from the scrutinee -- from its component, where the scrutinee
                 is a literal the pattern takes apart ({!bind_parts}). *)
              List.iter cases ~f:(fun case ->
                  visit (self#bind_parts env case.pc_lhs (scrutinee, env)) case.pc_rhs)
          | Pexp_function (params, _, body) -> (
              (* A thunk or callback in data position ([~default:(fun () -> ...)]): what it returns,
                 in either syntax. *)
              let env =
                List.fold params ~init:env ~f:(fun env param ->
                    match param.pparam_desc with
                    | Pparam_val (_, Some default, pattern) ->
                        (* [fun ?(value = sample) _ -> value]: an optional parameter is a part of
                           its default, which a call without the label passes. *)
                        self#bind_parts env pattern (default, env)
                    | Pparam_val (_, None, pattern) -> self#bind_patterns env [ pattern ]
                    | Pparam_newtype _ -> env)
              in
              match body with
              | Pfunction_body body -> visit env body
              | Pfunction_cases (cases, _, _) ->
                  List.iter cases ~f:(fun case ->
                      visit (self#bind_patterns env [ case.pc_lhs ]) case.pc_rhs))
          | Pexp_try (body, cases) ->
              (* Unlike a scrutinee, a protected body IS the value on success. *)
              visit env body;
              List.iter cases ~f:(fun case ->
                  visit (self#bind_patterns env [ case.pc_lhs ]) case.pc_rhs)
          | Pexp_let (rec_flag, bindings, body) ->
              visit (self#bind_group env bindings (self#denotations env rec_flag bindings)) body
          | Pexp_tuple elements | Pexp_array elements -> List.iter elements ~f:(visit env)
          | Pexp_construct (_, Some payload) | Pexp_variant (_, Some payload) -> visit env payload
          | Pexp_apply (callee, arguments) -> (
              match self#pipeline callee arguments with
              | Some (callee, arguments, _) ->
                  visit env { expr with pexp_desc = Pexp_apply (callee, arguments) }
              | None -> (
                  let positional =
                    List.filter_map arguments ~f:(function
                      | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                      | _ -> None)
                  in
                  let callee, callee_env = self#resolve_callee env callee in
                  match callee.pexp_desc with
                  | Pexp_ident { txt = Ppxlib.Longident.Lident (("!" | "@" | "ref") as name); _ }
                    when Option.is_none (Lexical_scope.lookup callee_env name) ->
                      List.iter positional ~f:(visit env)
                  | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, ("!" | "@" | "ref")); _ }
                    when match self#module_path callee_env prefix with
                         | Some (Standard_library _) -> true
                         | _ -> false ->
                      (* [Stdlib.ref [sample]], [Base.( ! ) rows]: the same primitives,
                         qualified. *)
                      List.iter positional ~f:(visit env)
                  | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ } -> (
                      match self#module_path callee_env prefix with
                      | Some (Collection_module { stdlib; _ }) -> (
                          match collection_function name with
                          | Some function_ ->
                              List.iter
                                (collection_arguments ~stdlib function_ arguments)
                                ~f:(visit env)
                          | None ->
                              (* Reported; its arguments are still followed, so that what the
                                 refusal says is all the reader could not read. *)
                              self#note_unlisted callee;
                              List.iter positional ~f:(visit env))
                      | _ -> ())
                  | _ -> ()))
          | _ -> ()
        in
        visit env expr;
        found

      (* A [gated] boolean's population: the collections the quantifiers it reduces to range over,
         each followed as {!population} follows one. The boolean is read through a name into its
         [let] right-hand side and what the source writes into it, [not], [&&] and [||] (bare or
         qualified by a standard library), [!] and [ref], the branches of an [if] (never the
         condition), the cases of a [match] and the body and handlers of a [try], the body of a
         [let], [let module], local [open] or sequence, type constraints, and a pipeline, as an
         application. A quantifier is a {!quantifiers} function of a collection module, reached as
         {!population} reaches one. *)
      method quantified env expr =
        let seen = Hash_set.create (module Int) in
        let found = Hash_set.Poly.create () in
        let rec visit env expr =
          match expr.pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
              match Lexical_scope.lookup env name with
              | Some denotation when not (Hash_set.mem seen denotation.identity) ->
                  Hash_set.add seen denotation.identity;
                  Option.iter denotation.definition ~f:(fun (rhs, env) -> visit env rhs);
                  List.iter (self#parameter_arguments denotation.identity)
                    ~f:(fun (argument, env) -> visit env argument);
                  List.iter (self#written denotation.identity) ~f:(fun (written, env) ->
                      visit env written)
              | _ -> ())
          | Pexp_constraint (inner, _)
          | Pexp_coerce (inner, _, _)
          | Pexp_sequence (_, inner)
          | Pexp_letexception (_, inner)
          | Pexp_newtype (_, inner)
          | Pexp_lazy inner ->
              visit env inner
          | Pexp_open ({ popen_expr; _ }, inner) ->
              visit
                (self#opened ~top:false ~include_:false env (self#module_of env popen_expr))
                inner
          | Pexp_letmodule ({ txt = Some name; _ }, module_expr, inner) ->
              visit (self#bind_module env name module_expr) inner
          | Pexp_letmodule ({ txt = None; _ }, _, inner) -> visit env inner
          | Pexp_apply (callee, [ (Ppxlib.Asttypes.Nolabel, thunk) ]) when self#forces env callee ->
              visit env thunk
          | Pexp_ifthenelse (_, yes, no) ->
              visit env yes;
              Option.iter no ~f:(visit env)
          | Pexp_match (scrutinee, cases) ->
              List.iter cases ~f:(fun case ->
                  visit (self#bind_parts env case.pc_lhs (scrutinee, env)) case.pc_rhs)
          | Pexp_try (body, cases) ->
              visit env body;
              List.iter cases ~f:(fun case ->
                  visit (self#bind_patterns env [ case.pc_lhs ]) case.pc_rhs)
          | Pexp_let (rec_flag, bindings, body) ->
              visit (self#bind_group env bindings (self#denotations env rec_flag bindings)) body
          | Pexp_function (params, _, body) -> (
              (* A quantifier's predicate: what its body quantifies. *)
              let env =
                List.fold params ~init:env ~f:(fun env param ->
                    match param.pparam_desc with
                    | Pparam_val (_, Some default, pattern) ->
                        (* [fun ?(value = sample) _ -> value]: an optional parameter is a part of
                           its default, which a call without the label passes. *)
                        self#bind_parts env pattern (default, env)
                    | Pparam_val (_, None, pattern) -> self#bind_patterns env [ pattern ]
                    | Pparam_newtype _ -> env)
              in
              match body with
              | Pfunction_body body -> visit env body
              | Pfunction_cases (cases, _, _) ->
                  List.iter cases ~f:(fun case ->
                      visit (self#bind_patterns env [ case.pc_lhs ]) case.pc_rhs))
          | Pexp_apply (callee, arguments) -> (
              match self#pipeline callee arguments with
              | Some (callee, arguments, _) ->
                  visit env { expr with pexp_desc = Pexp_apply (callee, arguments) }
              | None -> (
                  let positional =
                    List.filter_map arguments ~f:(function
                      | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                      | _ -> None)
                  in
                  let connective name = List.mem [ "not"; "&&"; "||" ] name ~equal:String.equal in
                  let callee, callee_env = self#resolve_callee env callee in
                  match callee.pexp_desc with
                  | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ }
                    when (connective name || List.mem [ "!"; "ref" ] name ~equal:String.equal)
                         && Option.is_none (Lexical_scope.lookup callee_env name) ->
                      List.iter positional ~f:(visit env)
                  | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ } -> (
                      match self#module_path callee_env prefix with
                      | Some (Standard_library _)
                        when connective name || List.mem [ "!"; "ref" ] name ~equal:String.equal ->
                          List.iter positional ~f:(visit env)
                      | Some (Collection_module _) when Option.is_none (collection_function name) ->
                          self#note_unlisted callee
                      | Some (Collection_module { stdlib; _ })
                        when Option.exists (collection_function name) ~f:(fun f -> f.quantifies) ->
                          List.iter (quantified_arguments ~stdlib name arguments)
                            ~f:(fun collection ->
                              Hash_set.iter (self#population env collection) ~f:(Hash_set.add found));
                          (* A quantifier nested in the predicate ranges over a population of its
                             own: the arguments are read as booleans too, a function's body
                             included, though the predicate's own parameters are no population. *)
                          List.iter arguments ~f:(fun (_, argument) -> visit env argument)
                      | _ -> ())
                  | _ -> ()))
          | _ -> ()
        in
        visit env expr;
        found

      (* Whether [callee] forces a lazy value: [Lazy.force] or [Lazy.force_val] of the standard
         library (qualified by any path ending in [Lazy] that is not a structure of this source), or
         [Base]'s unqualified [force] where the source binds no [force] of its own. *)
      method forces env callee =
        let callee, callee_env = self#resolve_callee env callee in
        match callee.pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, ("force" | "force_val")); _ } -> (
            match List.last (Read.flatten_longident prefix) with
            | Some "Lazy" -> (
                match Map.find callee_env.Lexical_scope.modules (Lexical_scope.head prefix) with
                | Some (Local_structure _) -> false
                | _ -> true)
            | _ -> false)
        | Pexp_ident { txt = Ppxlib.Longident.Lident "force"; _ } ->
            Option.is_none (Lexical_scope.lookup callee_env "force")
        | _ -> false

      (* A callee through the [let]s that rename it: [let map = List.map in map raw ~f] calls
         [List.map], resolved in the scope its definition was written in (gh-ocannl-1303). Only a
         definition that is itself a name is followed -- a partial application ([let map_raw =
         List.map raw]) is a value, not a renaming. Constraints are stripped on the way. *)
      method resolve_callee env callee =
        let rec go seen env callee =
          let callee = strip_constraints callee in
          (* The opened collection module a name is the function of, if one is opened inside every
             binding of it and lists it. *)
          let opened name =
            if Option.is_none (collection_function name) then None
            else
              Option.join
                (List.find_map env.Lexical_scope.frames ~f:(fun frame ->
                     if Map.mem frame name then Some None
                     else
                       match Map.find frame opened_collection with
                       | Some { definition = Some (_, module_env); _ } -> Some (Some module_env)
                       | Some { definition = None; _ } ->
                           (* An open of a module this reader cannot read: what it brings in is
                              unknown, so an older collection open no longer decides the name. *)
                           Some None
                       | None -> None))
          in
          match callee.pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; loc } when Option.is_some (opened name)
            ->
              ( {
                  callee with
                  pexp_desc =
                    Pexp_ident
                      {
                        txt = Ppxlib.Longident.Ldot (Ppxlib.Longident.Lident opened_collection, name);
                        loc;
                      };
                },
                Option.value_exn (opened name) )
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
              match Lexical_scope.lookup env name with
              | Some { identity; definition = Some (rhs, rhs_env) }
                when (not (Set.mem seen identity))
                     &&
                     match (strip_constraints rhs).pexp_desc with
                     | Pexp_ident _ -> true
                     | _ -> false ->
                  go (Set.add seen identity) rhs_env rhs
              | _ -> (callee, env))
          | _ -> (callee, env)
        in
        go (Set.empty (module Int)) env callee

      (* The binding written into, through any [let alias = rows] renaming on the way: a write
         through an alias is a write into what it names. *)
      method write_target env expr =
        match expr.pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some
                {
                  definition =
                    Some
                      ( ({ pexp_desc = Pexp_ident _ | Pexp_constraint _ | Pexp_coerce _; _ } as
                         aliased),
                        env );
                  _;
                } ->
                self#write_target env aliased
            | Some { identity; _ } -> Some identity
            | None -> None)
        | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> self#write_target env inner
        | _ -> None

      (* A write into a binding: [x := e], or a function {!collection_functions} marks as writing,
         handed [x]. *)
      method record_write env expr callee arguments =
        (* The container changes once the writer's arguments are evaluated: a claim in one of them
           ([Queue.enqueue queue (claim; v)]) runs before the write. *)
        let point = { (self#point_of expr) with effect_at = expr.pexp_loc.loc_end.pos_cnum } in
        let target = self#write_target env in
        let record identity written =
          List.iter written ~f:(fun e ->
              Hashtbl.add_multi writes ~key:identity ~data:(e, env, point))
        in
        let callee, callee_env = self#resolve_callee env callee in
        match (callee.pexp_desc, arguments) with
        | ( Pexp_ident { txt = Ppxlib.Longident.Lident ":="; _ },
            [ (Ppxlib.Asttypes.Nolabel, lhs); (Ppxlib.Asttypes.Nolabel, rhs) ] ) ->
            Option.iter (target lhs) ~f:(fun identity -> record identity [ rhs ])
        | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ }, _ -> (
            match self#module_path callee_env prefix with
            | Some (Collection_module { stdlib; mutable_; family }) ->
                let positional =
                  List.filter_map arguments ~f:(function
                    | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                    | _ -> None)
                in
                let data =
                  List.filter_map arguments ~f:(function
                    | (Ppxlib.Asttypes.Labelled label | Optional label), argument
                      when List.mem
                             [ "data"; "default"; "init"; "key"; "by" ]
                             label ~equal:String.equal ->
                        Some argument
                    | _ -> None)
                in
                (* In [Base]'s order the container comes first. [Stdlib]'s varies by function
                   ([Queue.add x q], [Hashtbl.add t k v]), so there every positional argument is a
                   possible container of all the others -- the loud direction. *)
                (* A labeled destination is a container and a labeled source written into it:
                   [Array.blit ~src ~src_pos ~dst ~dst_pos ~len]. *)
                let labeled name =
                  List.filter_map arguments ~f:(function
                    | (Ppxlib.Asttypes.Labelled label | Optional label), argument
                      when String.equal label name ->
                        Some argument
                    | _ -> None)
                in
                let containers =
                  (if stdlib then positional else List.take positional 1) @ labeled "dst"
                in
                let writes_into =
                  match collection_function name with
                  | Some { writes = On_mutable; _ } -> mutable_
                  | Some { writes = Only_in families; _ } ->
                      List.mem families family ~equal:String.equal
                  | Some { writes = Never; _ } -> false
                  | None ->
                      (* An unlisted function of a mutable module may write into anything it is
                         handed; a claim that reaches one of those bindings is refused. *)
                      if mutable_ then
                        List.iter
                          (positional @ labeled "dst")
                          ~f:(fun container ->
                            Option.iter (target container) ~f:(fun identity ->
                                Hashtbl.add_multi unlisted_writes ~key:identity
                                  ~data:
                                    ( ( String.concat ~sep:"."
                                          (Read.flatten_longident
                                             (Ppxlib.Longident.Ldot (prefix, name))),
                                        callee.pexp_loc.loc_start.pos_lnum ),
                                      point )));
                      false
                in
                if writes_into then
                  List.iter containers ~f:(fun container ->
                      Option.iter (target container) ~f:(fun identity ->
                          match
                            List.filter positional ~f:(fun other ->
                                not (phys_equal other container))
                            @ labeled "src" @ labeled "f" @ data
                          with
                          | written when List.is_empty written || Hash_set.mem named (position expr)
                            ->
                              (* A writer handed its container alone ([Queue.enqueue queue]), or
                                 whose value is bound to a name ([let put = Array.set rows 0]), may
                                 be partially applied: what it writes arrives at a call this reader
                                 does not follow, so a claim reaching the container is refused. *)
                              Hashtbl.add_multi unlisted_writes ~key:identity
                                ~data:
                                  ( ( String.concat ~sep:"."
                                        (Read.flatten_longident
                                           (Ppxlib.Longident.Ldot (prefix, name)))
                                      ^ " (partially applied)",
                                      callee.pexp_loc.loc_start.pos_lnum ),
                                    point )
                          | written -> record identity written))
            | _ -> ())
        | _ -> ()

      (* [x |> f] and [f @@ x], as the application [f x]; with the position of [f] when [f] is
         itself an application, which the rewrite absorbs. *)
      method pipeline callee arguments =
        match (Read.longident_of callee, arguments) with
        | Some [ "|>" ], [ (Ppxlib.Asttypes.Nolabel, x); (Ppxlib.Asttypes.Nolabel, f) ]
        | Some [ "@@" ], [ (Ppxlib.Asttypes.Nolabel, f); (Ppxlib.Asttypes.Nolabel, x) ] -> (
            match f.pexp_desc with
            | Pexp_apply (inner, inner_arguments) ->
                Some (inner, inner_arguments @ [ (Ppxlib.Asttypes.Nolabel, x) ], Some (position f))
            | _ -> Some (f, [ (Ppxlib.Asttypes.Nolabel, x) ], None))
        | _ -> None

      (* The gates of the leg [expr] denotes, each with its scope: the [~when_] of the [Verdict.leg]
         or [Verdict.sub_leg] application it is, then of each [sub_leg]'s outer leg, through a name
         into its [let] right-hand side, a type constraint, the body of a [let] or a sequence, both
         branches of an [if] and every case of a [match] -- whose condition and scrutinee, which
         pick the leg, are gates too -- and the body of a local function it applies
         (gh-ocannl-1794). With whether a constructor was only ambiguously [Verdict]'s; [Error]
         names what it could not follow: a parameter, a field, another module's value. *)
      (* The scope a function's body runs in at one application of it: each parameter whose pattern
         is a name bound to the argument the application (written in [env]) supplies, or to its
         default; the others to nothing the reader follows. *)
      method bind_applied ~definition_env env params arguments =
        let positional = ref 0 in
        List.fold params ~init:definition_env ~f:(fun scope param ->
            match param.pparam_desc with
            | Pparam_val (label, default, pattern) -> (
                let slot =
                  match label with
                  | Ppxlib.Asttypes.Nolabel ->
                      let index = !positional in
                      Int.incr positional;
                      Positional_slot index
                  | Labelled label -> Labelled_slot label
                  | Optional label ->
                      Optional_slot (label, Option.map default ~f:(fun d -> (d, scope)))
                in
                let bound value value_env name =
                  Lexical_scope.bind_values scope [ name ]
                    { (self#local) with definition = Some (value, value_env) }
                in
                (* [?l:e] for a parameter with a default is [e]'s payload or the default
                   (gh-ocannl-2336, local review): a literal [None] or [Some x] says which. Any
                   other [e] a leg's way refuses beforehand ({!unresolved_option}). *)
                let supplied =
                  match (self#argument_for slot arguments, slot) with
                  | `Option e, Optional_slot (_, Some d) -> (
                      match (strip_constraints e).pexp_desc with
                      | Pexp_construct ({ txt = Lident "None"; _ }, None) -> `Default d
                      | Pexp_construct ({ txt = Lident "Some"; _ }, Some payload) -> `Given payload
                      | _ -> `Given e)
                  | argument, _ -> argument
                in
                match (Read.pattern_name pattern, supplied) with
                | Some name, (`Given e | `Option e) -> bound e env name
                | Some name, `Default (d, d_env) -> bound d d_env name
                (* A destructured parameter is the parts of what is supplied (gh-ocannl-2336,
                   review). *)
                | None, (`Given e | `Option e) -> self#bind_parts scope pattern (e, env)
                | None, `Default whole -> self#bind_parts scope pattern whole
                | _ -> self#bind_patterns scope [ pattern ])
            | Pparam_newtype _ -> scope)

      (* The label of an optional parameter with a default among [params] that [arguments] hands
         [?l:e] for, [e] no literal [None] or [Some _]: its value is [e]'s payload or the default,
         which {!bind_applied} cannot say, so a leg built through it is not followed
         (gh-ocannl-2336, review). *)
      method unresolved_option params arguments =
        List.find_map params ~f:(fun param ->
            match param.pparam_desc with
            | Pparam_val (Optional label, Some _, _) ->
                List.find_map arguments ~f:(function
                  | Ppxlib.Asttypes.Optional given, e when String.equal given label -> (
                      match (strip_constraints e).pexp_desc with
                      | Pexp_construct ({ txt = Lident ("None" | "Some"); _ }, _) -> None
                      | _ -> Some label)
                  | _ -> None)
            | _ -> None)

      (* Where the parameter [identity] of a named function takes its value at the application
         [arguments]: an argument, an optional parameter's default, an option passed with [?l:], or
         nothing (an application in part). *)
      method argument_for slot arguments =
        let labelled kind label =
          List.find_map arguments ~f:(fun (given, argument) ->
              match (given, kind) with
              | Ppxlib.Asttypes.Labelled given, `Labelled when String.equal given label ->
                  Some argument
              | Ppxlib.Asttypes.Optional given, `Optional when String.equal given label ->
                  Some argument
              | _ -> None)
        in
        match slot with
        | Labelled_slot label -> (
            match labelled `Labelled label with Some e -> `Given e | None -> `Missing)
        | Optional_slot (label, default) -> (
            match (labelled `Labelled label, labelled `Optional label, default) with
            | Some e, _, _ -> `Given e
            | None, Some e, _ -> `Option e
            | None, None, Some (d, d_env) -> `Default (d, d_env)
            | None, None, None -> `Missing)
        | Positional_slot index -> (
            match
              List.nth
                (List.filter_map arguments ~f:(function
                  | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                  | _ -> None))
                index
            with
            | Some e -> `Given e
            | None -> `Missing)
        | Element_slot _ ->
            (* No application supplies an element: {!parameter_arguments} reads the collection. *)
            `Missing

      (* Whether the source spells the named function [owner] anywhere but as the callee of an
         application: handed on as a value, piped, so that not every application is seen. *)
      method escapes owner =
        let callees =
          List.map (Hashtbl.find_multi call_sites owner) ~f:(fun (at, _, _) -> at)
          |> Set.of_list (module Int)
        in
        List.exists (Hashtbl.find_multi references owner) ~f:(fun at -> not (Set.mem callees at))

      (* Every chain of applications that reaches the named function [owner] -- an application of
         it, then one of the function that application is written in, and so on up to code no
         function of the source encloses -- so that a parameter forwarded through several calls is
         read under one call of each. [[ [] ]] where [owner] is no function, or has no application
         in the source; past 256 chains, too, where every parameter is read through every
         application, the loud direction. *)
      method call_chains owner =
        (* Every chain ends in one terminal case or the other, so counting both bounds the total. *)
        let produced = ref 0 in
        let terminal chain =
          Int.incr produced;
          if !produced > 256 then raise Stdlib.Exit;
          [ chain ]
        in
        let rec go seen owner =
          match Hashtbl.find_multi call_sites owner with
          | [] -> terminal []
          | sites ->
              List.concat_map sites ~f:(fun (at, _, _) ->
                  match Option.join (Hashtbl.find call_callers at) with
                  | Some caller when not (List.mem seen caller ~equal:Int.equal) ->
                      List.map (go (caller :: seen) caller) ~f:(fun chain -> (owner, at) :: chain)
                  | _ -> terminal [ (owner, at) ])
        in
        match owner with
        | None -> [ [] ]
        | Some owner -> ( try go [ owner ] owner with Stdlib.Exit -> [ [] ])

      (* The choices of elements a comparison under [calls] is made under too (gh-ocannl-2286, local
         review): one per element of each element callback enclosing the claim ([here]) or a chosen
         application, where [calls] chooses none of its elements yet -- so a gate and a population
         both read off one callback's parameter meet one element at a time. Past 256 choices, the
         comparison is made under [calls] alone, every element at once: the loud direction. *)
      method element_contexts ~here calls =
        let owners =
          here
          @ List.concat_map calls ~f:(fun (_, at) ->
              Option.value (Hashtbl.find site_elements at) ~default:[])
          |> List.dedup_and_sort ~compare:Int.compare
          |> List.filter ~f:(fun owner -> not (List.Assoc.mem calls owner ~equal:Int.equal))
        in
        try
          List.fold owners ~init:[ calls ] ~f:(fun choices owner ->
              match Hashtbl.find element_owners owner with
              | None -> choices
              | Some (collection, collection_env) -> (
                  match self#elements collection_env collection with
                  | Error _ | Ok [] -> choices
                  | Ok elements ->
                      let extended =
                        List.concat_map choices ~f:(fun calls ->
                            List.map elements ~f:(fun (element, _) ->
                                (owner, position element) :: calls))
                      in
                      if List.length extended > 256 then raise Stdlib.Exit else extended))
        with Stdlib.Exit -> [ calls ]

      (* The applications a parameter of [owner] takes its arguments from: under {!context}, the one
         chosen there; otherwise every one. *)
      method sites owner =
        let sites = Hashtbl.find_multi call_sites owner in
        match List.Assoc.find context owner ~equal:Int.equal with
        | Some chosen -> List.filter sites ~f:(fun (at, _, _) -> at = chosen)
        | None -> sites

      (* What a parameter of a named function is, for the gate and population sides: the argument
         each application it is read under supplies (gh-ocannl-1893). Nothing for any other name. *)
      method parameter_arguments identity =
        match Hashtbl.find parameters identity with
        | None -> []
        | Some _ when Hash_set.mem arguing identity ->
            (* Reached again while its own values are being found, through a pattern taking apart
               what reaches back to it: nothing more. *)
            []
        | Some parameter ->
            Hash_set.add arguing identity;
            Exn.protect
              ~f:(fun () -> self#parameter_values parameter)
              ~finally:(fun () -> Hash_set.remove arguing identity)

      (* What {!parameter_arguments} finds, once per parameter on the path. *)
      method parameter_values { owner; slot; part; _ } =
        let wholes =
          match slot with
          | Element_slot (collection, collection_env) -> (
              (* An element of a literal collection: under {!context}, the one chosen there;
                 otherwise every one (gh-ocannl-2286). *)
              match self#elements collection_env collection with
              | Error _ -> []
              | Ok elements -> (
                  match List.Assoc.find context owner ~equal:Int.equal with
                  | Some chosen ->
                      List.filter elements ~f:(fun (element, _) -> position element = chosen)
                  | None -> elements))
          | _ ->
              List.concat_map (self#sites owner) ~f:(fun (_, arguments, site_env) ->
                  match self#argument_for slot arguments with
                  | `Given e | `Option e -> [ (e, site_env) ]
                  | `Default default -> [ default ]
                  | `Missing -> [])
        in
        match part with
        | None -> wholes
        | Some (pattern, name) ->
            (* A destructured parameter is the part of each value its pattern takes apart that is
               this name's ({!parts}); where the value is no literal, the whole value. A value that
               is itself a parameter -- an element no choice has picked, say -- is each of its own
               values, so a part is taken of every one: the loud direction. *)
            List.concat_map wholes ~f:(fun whole ->
                List.concat_map (self#alternatives whole) ~f:(fun whole ->
                    List.concat_map (self#parts pattern whole) ~f:(fun (part_name, part, written) ->
                        (* With what is written into that field before the claim, as a name's writes
                           are its values too (gh-ocannl-2286, local review). *)
                        if String.equal part_name name then
                          part
                          :: List.filter_map written ~f:(fun (value, value_env, point) ->
                              if self#before point then Some (value, value_env) else None)
                        else [])))

      (* The values [expr] may be, where it names a parameter ({!parameters}): what each application
         or element it is read under supplies, recursively; otherwise [expr] itself. A parameter
         already being expanded on this path is itself, so a function handing its own parameter back
         to itself ends. *)
      method alternatives ((expr, env) as whole) =
        match (strip_constraints expr).pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some { identity; definition = None }
              when Hashtbl.mem parameters identity && not (Hash_set.mem expanding identity) ->
                Hash_set.add expanding identity;
                Exn.protect
                  ~f:(fun () ->
                    match self#parameter_arguments identity with
                    | [] -> [ whole ]
                    | values -> List.concat_map values ~f:self#alternatives)
                  ~finally:(fun () -> Hash_set.remove expanding identity)
            | _ -> [ whole ])
        | _ -> [ whole ]

      (* The elements of the collection [expr] denotes, each with its scope, where it is a list or
         array literal: through names (not written into), type constraints, the bodies of [let]s and
         sequences (gh-ocannl-2286). [Error] says what it is instead. *)
      method elements env expr =
        let rec list_elements env expr =
          match (strip_constraints expr).pexp_desc with
          | Pexp_construct ({ txt = Ppxlib.Longident.Lident "[]"; _ }, None) -> Some []
          | Pexp_construct
              ( { txt = Ppxlib.Longident.Lident "::"; _ },
                Some { pexp_desc = Pexp_tuple [ hd; tl ]; _ } ) ->
              Option.map (list_elements env tl) ~f:(fun rest -> (hd, env) :: rest)
          | _ -> None
        in
        let rec go seen env expr =
          let expr = strip_constraints expr in
          match expr.pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
              match Lexical_scope.lookup env name with
              | Some { identity; definition = Some (rhs, rhs_env) }
                when (not (Set.mem seen identity))
                     && (not (Hashtbl.mem writes identity))
                     && not (Hashtbl.mem unlisted_writes identity) ->
                  go (Set.add seen identity) rhs_env rhs
              | _ ->
                  Error
                    (Printf.sprintf
                       "`%s` is bound to no list or array literal the reader follows (a parameter, \
                        a pattern, a binding the source writes into)"
                       name))
          | Pexp_sequence (_, inner) -> go seen env inner
          | Pexp_let (rec_flag, bindings, body) ->
              go seen (self#bind_group env bindings (self#denotations env rec_flag bindings)) body
          | Pexp_array elements -> Ok (List.map elements ~f:(fun element -> (element, env)))
          | _ -> (
              match list_elements env expr with
              | Some elements -> Ok elements
              | None ->
                  Error
                    (Printf.sprintf "`%s` is no list or array literal"
                       (Ppxlib.Pprintast.string_of_expression expr)))
        in
        go (Set.empty (module Int)) env expr

      (* A [~leg]'s gates, as the alternatives it may be resolved by ({!leg_resolution}): one per
         choice of the applications a parameter on the way takes its argument from. *)
      method leg_gates env expr =
        let single gates ambiguous =
          [ { calls = []; leg_gates = gates; leg_ambiguous = ambiguous } ]
        in
        let compatible = calls_compatible in
        let merge a b =
          {
            calls = List.dedup_and_sort (a.calls @ b.calls) ~compare:Poly.compare;
            leg_gates = a.leg_gates @ b.leg_gates;
            leg_ambiguous = a.leg_ambiguous || b.leg_ambiguous;
          }
        in
        (* Two parts of one leg: every pair of alternatives that agree on their applications. *)
        let both a b =
          Result.bind a ~f:(fun xs ->
              Result.bind b ~f:(fun ys ->
                  bounded_resolutions
                    (List.concat_map xs ~f:(fun x ->
                         List.filter_map ys ~f:(fun y ->
                             if compatible x.calls y.calls then Some (merge x y) else None)))))
        in
        let gated gates = List.map ~f:(fun r -> { r with leg_gates = gates @ r.leg_gates }) in
        (* [seen]: the bindings on this path, so that two branches may share an ancestor leg.
           [path]: the projections still to take of the value reached, outermost first -- the leg is
           that part of it (gh-ocannl-2286): [case.leg], or a name a destructured parameter
           binds. *)
        let rec go seen path env expr =
          let expr = strip_constraints expr in
          let not_taken_apart () =
            Error
              (Printf.sprintf
                 "its leg is %s of `%s`, which is no record or tuple literal the reader takes apart"
                 (match path with
                 | Field field :: _ -> Printf.sprintf "the field `%s`" field
                 | Component (index, arity) :: _ ->
                     Printf.sprintf "component %d of %d" (index + 1) arity
                 | [] -> "a part")
                 (Ppxlib.Pprintast.string_of_expression expr))
          in
          match (expr.pexp_desc, path) with
          | Pexp_field (record, field), _ ->
              go seen (Field (Ppxlib.Longident.last_exn field.txt) :: path) env record
          | Pexp_record (given, base), Field field :: rest -> (
              match
                List.find given ~f:(fun (name, _) ->
                    String.equal (Ppxlib.Longident.last_exn name.txt) field)
              with
              | Some (_, component) -> go seen rest env component
              | None -> (
                  match base with Some base -> go seen path env base | None -> not_taken_apart ()))
          | Pexp_tuple components, Component (index, arity) :: rest
            when List.length components = arity ->
              go seen rest env (List.nth_exn components index)
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ }, _ -> (
              match Lexical_scope.lookup env name with
              | Some { identity; _ }
                when (not (List.is_empty path))
                     && (Hashtbl.mem field_writes identity || Hashtbl.mem writes identity) ->
                  Error
                    (Printf.sprintf
                       "its leg is a part of `%s`, which the source writes into after it is made"
                       name)
              | Some { identity; definition = Some (rhs, rhs_env) } when not (Set.mem seen identity)
                ->
                  go (Set.add seen identity) path rhs_env rhs
              | Some { identity; definition = None } when Hashtbl.mem parameters identity ->
                  if Set.mem seen identity then
                    (* A parameter passed back to its own function: the cycle adds no gate. *)
                    Ok (single [] false)
                  else self#through_parameter (go (Set.add seen identity)) path name identity
              | _ ->
                  Error
                    (Printf.sprintf
                       "its leg reaches `%s`, a name bound to no definition the reader follows (a \
                        parameter of an anonymous function, a pattern, a recursive binding)"
                       name))
          | Pexp_sequence (_, inner), _ -> go seen path env inner
          | Pexp_let (rec_flag, bindings, body), _ ->
              go seen path
                (self#bind_group env bindings (self#denotations env rec_flag bindings))
                body
          | Pexp_ifthenelse (condition, yes, no), _ ->
              both (go seen path env yes)
                (Option.value_map no ~default:(Ok (single [] false)) ~f:(go seen path env))
              |> Result.map ~f:(gated [ (condition, env) ])
          | Pexp_match (scrutinee, cases), _ ->
              List.fold cases
                ~init:(Ok (single [ (scrutinee, env) ] false))
                ~f:(fun acc case ->
                  let case_env = self#bind_parts env case.pc_lhs (scrutinee, env) in
                  (* A guard picks the case's leg as the scrutinee does. *)
                  let guard =
                    Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                        [ (guard, case_env) ])
                  in
                  both (Result.map acc ~f:(gated guard)) (go seen path case_env case.pc_rhs))
          | Pexp_apply (callee, arguments), _ -> (
              match Verdict_callee.resolve_leg owners callee with
              | Some _ when not (List.is_empty path) -> not_taken_apart ()
              | Some { member; ownership = (Verdicts | Ambiguous) as ownership; _ } -> (
                  let ambiguous = Poly.equal ownership Verdict_callee.Ambiguous in
                  let gate =
                    List.find_map arguments ~f:(function
                      | Ppxlib.Asttypes.Labelled "when_", gate -> Some (gate, env)
                      | _ -> None)
                  in
                  let outer =
                    List.find_map arguments ~f:(function
                      | Ppxlib.Asttypes.Nolabel, outer -> Some outer
                      | _ -> None)
                  in
                  match (gate, member, outer) with
                  | None, _, _ ->
                      Error
                        (Printf.sprintf "its leg is a `%s` application without its `~when_`" member)
                  | Some gate, "leg", _ -> Ok (single [ gate ] ambiguous)
                  | Some gate, _, Some outer ->
                      Result.map (go seen [] env outer) ~f:(fun resolutions ->
                          List.map resolutions ~f:(fun r ->
                              {
                                r with
                                leg_gates = r.leg_gates @ [ gate ];
                                leg_ambiguous = r.leg_ambiguous || ambiguous;
                              }))
                  | Some _, _, None ->
                      Error
                        (Printf.sprintf "its leg is a `%s` application without its outer leg" member)
                  )
              | Some _ ->
                  Error "its leg is built by a `leg` or `sub_leg` that is not certainly `Verdict`'s"
              | None -> (
                  (* A local function's application: what its body returns, each parameter bound to
                     the argument this application supplies (or its default), the others to nothing
                     the reader follows. *)
                  match (strip_constraints callee).pexp_desc with
                  | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                      match Lexical_scope.lookup env name with
                      | Some
                          {
                            identity;
                            definition =
                              Some
                                ( { pexp_desc = Pexp_function (params, _, Pfunction_body body); _ },
                                  definition_env );
                          }
                        when not (Set.mem seen identity) -> (
                          match self#unresolved_option params arguments with
                          | Some label ->
                              Error
                                (Printf.sprintf
                                   "its leg is the result of `%s`, handed `?%s:` an option the \
                                    reader does not resolve against the parameter's default"
                                   name label)
                          | None ->
                              let seen = Set.add seen identity in
                              go seen path
                                (self#bind_applied ~definition_env env params arguments)
                                body)
                      | _ ->
                          Error
                            (Printf.sprintf
                               "its leg is the result of `%s`, which is not a local function the \
                                reader follows"
                               name))
                  | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ } ->
                      (* A function of a structure, or of another source (gh-ocannl-2336): read as a
                         local function is, at this application, once per definition. *)
                      Result.bind
                        (self#defined_function env ~at:(position callee) prefix name)
                        ~f:(fun definitions ->
                          List.fold definitions ~init:(Ok []) ~f:(fun acc definition ->
                              Result.bind acc ~f:(fun alternatives ->
                                  let identity, params, body, definition_env, (start, stop) =
                                    definition
                                  in
                                  let spelled = Ppxlib.Longident.name (Ldot (prefix, name)) in
                                  (* A gate the body writes builds on something written elsewhere --
                                     [prepare xs] setting the reference the gate reads -- at a time
                                     this reader does not relate to the claim (review): such a leg
                                     is not followed. *)
                                  let written_gate r =
                                    List.find_map r.leg_gates ~f:(fun (gate, gate_env) ->
                                        let at = position gate in
                                        (* In the body, or anywhere in a source read as a
                                           definition: its helpers' gates too (review). *)
                                        if
                                          (start <= at && at < stop)
                                          || List.exists !definition_spans ~f:(fun (from, until) ->
                                              from <= at && at < until)
                                        then
                                          Hash_set.find (self#gate_reads gate_env gate)
                                            ~f:(fun (_, read) ->
                                              read <> 0
                                              && (Hashtbl.mem writes read
                                                || Hashtbl.mem field_writes read
                                                || Hashtbl.mem unlisted_writes read))
                                        else None)
                                  in
                                  let mutates =
                                    Hash_set.exists mutating_functions ~f:(fun mutating ->
                                        List.exists (Hashtbl.find_multi references mutating)
                                          ~f:(fun at -> start <= at && at < stop))
                                  in
                                  if mutates then
                                    Error
                                      (Printf.sprintf
                                         "its leg is the result of `%s`, whose body applies a \
                                          function that writes into a binding, an effect the \
                                          reader does not time against the claim"
                                         spelled)
                                  else if Option.is_some (self#unresolved_option params arguments)
                                  then
                                    Error
                                      (Printf.sprintf
                                         "its leg is the result of `%s`, handed an optional \
                                          argument the reader does not resolve against its default"
                                         spelled)
                                  else if Set.mem seen identity then
                                    Error
                                      (Printf.sprintf
                                         "its leg is the result of `%s`, whose own result passes \
                                          back through it"
                                         spelled)
                                  else
                                    Result.bind
                                      (go (Set.add seen identity) path
                                         (self#bind_applied ~definition_env env params arguments)
                                         body)
                                      ~f:(fun more ->
                                        match List.find_map more ~f:written_gate with
                                        | Some (read, _) ->
                                            Error
                                              (Printf.sprintf
                                                 "its leg is the result of `%s`, whose gate reads \
                                                  `%s`, which the source writes into at a time the \
                                                  reader does not relate to the claim"
                                                 spelled read)
                                        | None -> bounded_resolutions (alternatives @ more)))))
                  | _ ->
                      Error
                        (Printf.sprintf "its leg is the result of `%s`, another module's function"
                           (Ppxlib.Pprintast.string_of_expression callee))))
          | (Pexp_record _ | Pexp_tuple _), _ :: _ -> not_taken_apart ()
          | _ ->
              Error
                (Printf.sprintf "its leg is `%s`, an expression the reader does not follow"
                   (Ppxlib.Pprintast.string_of_expression expr))
        in
        match go (Set.empty (module Int)) [] env expr with
        | Ok [] ->
            Error "no choice of the applications its leg passes through agrees on all of them"
        | result -> result

      (* Whether the walk recorded a write into any binding -- [x := e], a field set, a mutable
         container's writer, an unlisted call handed one -- at a point inside [start, stop): a
         function body whose writes this reader does not time against a claim reached through it
         (gh-ocannl-2336, review). *)
      method writes_within (start, stop) =
        let inside { at_start; _ } = start <= at_start && at_start < stop in
        Hashtbl.exists writes ~f:(List.exists ~f:(fun (_, _, point) -> inside point))
        || Hashtbl.exists field_writes ~f:(List.exists ~f:(fun (_, (_, _, point)) -> inside point))
        || Hashtbl.exists unlisted_writes ~f:(List.exists ~f:(fun (_, point) -> inside point))

      (* The function [prefix.name] a leg is the result of (gh-ocannl-2336): a value of a structure
         [prefix] denotes -- one of this source's, or, while a caller is walked, this source itself
         -- or, where [prefix] is a module name nothing in scope binds, of each source
         {!definitions} gives as compiled under it. Each definition comes with its parameters, its
         body and the scope its body runs in; [Error] says why the reader does not follow one. *)
      method defined_function env ~at prefix name =
        let spelled = Ppxlib.Longident.name (Ldot (prefix, name)) in
        let defined values ~where =
          match List.Assoc.find values name ~equal:String.equal with
          | Some { identity; _ } when Hash_set.mem rebindable identity ->
              Error
                (Printf.sprintf
                   "its leg is the result of `%s`, which a later `include` in %s of a module the \
                    reader does not read may rebind"
                   spelled where)
          | Some
              {
                identity;
                definition =
                  Some
                    ( ({ pexp_desc = Pexp_function (params, _, Pfunction_body body); _ } as fn),
                      definition_env );
              }
            when identity > 0 ->
              if self#writes_within (span_of fn) then
                Error
                  (Printf.sprintf
                     "its leg is the result of `%s`, whose body writes into a binding, which the \
                      reader does not time against the claim"
                     spelled)
              else Ok (identity, params, body, definition_env, span_of fn)
          | _ ->
              Error
                (Printf.sprintf
                   "its leg is the result of `%s`, which %s defines as no function the reader \
                    follows"
                   spelled where)
        in
        match (prefix, self#module_path env prefix) with
        | _, Some (Local_structure { values; _ }) ->
            Result.map (defined values ~where:"its structure") ~f:List.return
        | Lident head, Some Other_module when not (Map.mem env.Lexical_scope.modules head) -> (
            (* Read from the source the name is spelled in: this one, a caller or a definition. *)
            let spelled_in =
              List.find_map !caller_spans ~f:(fun (start, stop, source) ->
                  if start <= at && at < stop then Some source else None)
            in
            match definitions spelled_in head with
            | [] ->
                Error
                  (Printf.sprintf
                     "its leg is the result of `%s`, another module's function, and no source \
                      handed to the reader is compiled as `%s`"
                     spelled head)
            | sources ->
                Result.all
                  (List.map sources ~f:(fun (source, text) ->
                       Result.bind (self#defining_structure ~head source text) ~f:(fun values ->
                           defined values ~where:(Printf.sprintf "`%s`" source)))))
        | _ ->
            Error
              (Printf.sprintf "its leg is the result of `%s`, another module's function" spelled)

      (* The values of [source], compiled as the module [head]. A source walked as a caller is read
         as it was walked; any other is parsed once past every source walked or parsed before it,
         and walked as a caller is (gh-ocannl-2336, local review): what it writes into its own
         bindings, and its own applications, are then read as a caller's are. Its [Verdict]
         resolutions join this source's, keyed by positions no other source shares. *)
      method defining_structure ~head source text =
        Hashtbl.find_or_add defining_structures source ~default:(fun () ->
            let parsed =
              match Hashtbl.find walked_callers source with
              | Some (structure, scope) ->
                  List.iter !caller_spans ~f:(fun (from, until, caller) ->
                      if String.equal caller source then
                        definition_spans := (from, until) :: !definition_spans);
                  Scan_attempt.Scanned (structure, scope)
              | None -> (
                  let at = !next_offset in
                  next_offset := at + String.length text + 1;
                  match Scan_attempt.run (fun () -> structure_at ~offset:at text) with
                  | Scanned structure ->
                      caller_spans := (at, at + String.length text + 1, source) :: !caller_spans;
                      definition_spans := (at, at + String.length text + 1) :: !definition_spans;
                      Verdict_callee.absorb ~into:owners (Verdict_callee.of_structure structure);
                      binder#structure structure;
                      self#walk_foreign (fun () ->
                          ignore (self#structure (Lazy.force empty_scope) structure : structure));
                      Scanned (structure, Lazy.force empty_scope)
                  | Unparsed report -> Unparsed report
                  | Raised exn -> Raised exn)
            in
            match parsed with
            | Scanned (structure, scope) ->
                (* In the scope it was walked in: a caller's binds this source's module. *)
                let values =
                  match
                    self#module_of scope
                      (Ppxlib.Ast_builder.Default.pmod_structure ~loc:Ppxlib.Location.none structure)
                  with
                  | Some (Local_structure { values; _ }) -> values
                  | _ -> []
                in
                (* Its named functions that write, or apply one that does, through any number of
                   calls: an application of one is an effect -- into a reference handed to it, say
                   -- no call site of this reader times against a claim (review). *)
                let functions = ref [] in
                let finder =
                  object
                    inherit Ast_traverse.iter as super

                    method! value_binding binding =
                      (match (strip_constraints binding.pvb_expr).pexp_desc with
                      | Pexp_function _ ->
                          functions :=
                            (binding.pvb_loc.loc_start.pos_cnum + 1, span_of binding.pvb_expr)
                            :: !functions
                      | _ -> ());
                      super#value_binding binding
                  end
                in
                finder#structure structure;
                let applies (start, stop) =
                  Hash_set.exists mutating_functions ~f:(fun mutating ->
                      List.exists (Hashtbl.find_multi references mutating) ~f:(fun at ->
                          start <= at && at < stop))
                in
                List.iter !functions ~f:(fun (identity, span) ->
                    if self#writes_within span then Hash_set.add mutating_functions identity);
                let rec close () =
                  let grown =
                    List.filter !functions ~f:(fun (identity, span) ->
                        (not (Hash_set.mem mutating_functions identity)) && applies span)
                  in
                  if not (List.is_empty grown) then (
                    List.iter grown ~f:(fun (identity, _) ->
                        Hash_set.add mutating_functions identity);
                    close ())
                in
                close ();
                Ok values
            | Unparsed report ->
                Error
                  (Printf.sprintf
                     "its leg is the result of a function of `%s`, and `%s`, which is compiled as \
                      `%s`, does not parse (%s)"
                     head source head report)
            | Raised exn ->
                failwith
                  (Printf.sprintf "%s (while parsing %s, which defines a leg's function)" exn source))

      (* A parameter on a leg's way, resolved at the applications of its function in this source and
         in the sources handed to the reader (gh-ocannl-1893, gh-ocannl-2286), or, for an
         {!Element_slot}, at each element of the collection: one alternative per application or
         element, each with it chosen. A destructured parameter is the part of each value its
         pattern takes apart that is [name]'s: [path] grows by the projections to it. *)
      method through_parameter go path name identity =
        let { owner; owner_name; slot; part } = Hashtbl.find_exn parameters identity in
        let path =
          match part with
          | None -> Ok path
          | Some (pattern, _) -> (
              match projection_to name pattern with
              | Some projections -> Ok (projections @ path)
              | None ->
                  Error
                    (Printf.sprintf
                       "its leg reaches `%s`, bound by a pattern of `%s` the reader does not take \
                        apart (a record or tuple pattern is taken apart)"
                       name owner_name))
        in
        let chosen at resolutions =
          List.filter_map resolutions ~f:(fun r ->
              match List.Assoc.find r.calls owner ~equal:Int.equal with
              | Some other when other <> at -> None
              | Some _ -> Some r
              | None -> Some { r with calls = (owner, at) :: r.calls })
        in
        let alternatives values =
          List.fold values ~init:(Ok []) ~f:(fun acc (at, value) ->
              Result.bind acc ~f:(fun acc ->
                  match value with
                  | `Value (e, e_env) ->
                      Result.bind path ~f:(fun path ->
                          Result.map (go path e_env e) ~f:(fun rs -> acc @ chosen at rs))
                  | `Unseen why -> Error why))
          |> fun resolutions -> Result.bind resolutions ~f:bounded_resolutions
        in
        match slot with
        | Element_slot (collection, collection_env) -> (
            match self#elements collection_env collection with
            | Error why ->
                Error
                  (Printf.sprintf
                     "its leg reaches `%s`, an element of the collection `%s` hands its callback: \
                      %s"
                     name owner_name why)
            | Ok elements ->
                alternatives
                  (List.map elements ~f:(fun (element, element_env) ->
                       (position element, `Value (element, element_env)))))
        | Labelled_slot _ | Optional_slot _ | Positional_slot _ -> (
            match Hashtbl.find foreign_escapes owner with
            | Some why ->
                Error
                  (Printf.sprintf "its leg reaches `%s`, a parameter of `%s`, %s" name owner_name
                     why)
            | None -> (
                if self#escapes owner then
                  Error
                    (Printf.sprintf
                       "its leg reaches `%s`, a parameter of `%s`, whose name the source also uses \
                        other than as the function it applies, so not every application is seen"
                       name owner_name)
                else
                  match Hashtbl.find_multi call_sites owner with
                  | [] ->
                      Error
                        (Printf.sprintf
                           "its leg reaches `%s`, a parameter of `%s`, which no application in \
                            this source or in a source spelling its module supplies"
                           name owner_name)
                  | sites ->
                      alternatives
                        (List.rev_map sites ~f:(fun (at, arguments, site_env) ->
                             ( at,
                               match self#argument_for slot arguments with
                               | `Given e -> `Value (e, site_env)
                               | `Default (d, d_env) -> `Value (d, d_env)
                               | `Option _ ->
                                   `Unseen
                                     (Printf.sprintf
                                        "its leg reaches `%s`, a parameter of `%s`, which an \
                                         application passes as an option (`?%s:`)"
                                        name owner_name name)
                               | `Missing ->
                                   `Unseen
                                     (Printf.sprintf
                                        "its leg reaches `%s`, a parameter of `%s`, which an \
                                         application of it does not supply"
                                        name owner_name) )))))

      method helper env callee =
        let callee, callee_env = self#resolve_callee env callee in
        match callee.pexp_desc with
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup callee_env name with
            | Some _ -> true
            | None ->
                Char.is_alpha name.[0] && not (List.mem standard_functions name ~equal:String.equal)
            )
        | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, _); _ } -> (
            match self#module_path callee_env prefix with
            | Some (Collection_module _ | Standard_library _) -> false
            | _ -> true)
        | _ -> true

      (* While a leg's [make] is walked: a claim adds its population to the innermost leg's, and an
         application of a {!helper} hands its arguments to the leg's fallback population. *)
      method collect_leg_population env expr callee arguments =
        let at = position expr in
        match leg_stack with
        | [] -> ()
        | collectors -> (
            (* [rows |> p_all label ~f] and [p_all label ~f @@ rows] are the application they
               spell. *)
            let callee, arguments =
              match self#pipeline callee arguments with
              | Some (callee, arguments, _) -> (callee, arguments)
              | None -> (callee, arguments)
            in
            (* A [Verdict] claim or leg constructor is no helper: its arguments are a claim's own
               slots, or a skip's label and backend. *)
            let verdicts =
              match
                (Verdict_callee.resolve owners callee, Verdict_callee.resolve_leg owners callee)
              with
              | Some { ownership = Verdicts | Ambiguous | Free; _ }, _
              | _, Some { ownership = Verdicts | Ambiguous; _ } ->
                  true
              | _ -> false
            in
            (* Every enclosing leg's: a leg claim nested in a make function is a claim that make
               function makes. *)
            if (not verdicts) && self#helper env callee then
              List.iter collectors ~f:(fun collector ->
                  collector.handed <-
                    List.map arguments ~f:(fun (_, argument) -> (at, argument, env))
                    @ collector.handed);
            match Verdict_callee.resolve owners callee with
            | Some { member; ownership = Verdicts | Ambiguous | Free; _ } -> (
                let positional =
                  List.filter_map arguments ~f:(function
                    | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                    | _ -> None)
                in
                let over =
                  List.filter_map arguments ~f:(function
                    | Ppxlib.Asttypes.Labelled "over", argument -> Some argument
                    | _ -> None)
                in
                let add boolean expressions =
                  List.iter collectors ~f:(fun collector ->
                      collector.populations <-
                        List.map expressions ~f:(fun e -> (at, e, boolean, env))
                        @ collector.populations)
                in
                let last n = List.drop positional (List.length positional - n) in
                match List.Assoc.find leg_population_slots member ~equal:String.equal with
                | Some (Boolean_at i) -> add true (Option.to_list (List.nth positional i))
                | Some Boolean_last -> add true (Option.to_list (List.last positional))
                | Some (Collections_at indices) ->
                    add false (List.filter_map indices ~f:(List.nth positional) @ over)
                | Some (Collections_last n) -> add false (last n @ over)
                | None -> ())
            | _ -> ())

      (* A leg claim [expr] applies, with its [~leg] and its make function in place: its comparison
         is queued and its collector registered, for the walk of [make] to fill. Applied in part, it
         is reported unread. *)
      (* The make functions a leg claim's [make] names, under {!context} (gh-ocannl-1893): a
         function written in place, a helper applied to its arguments, or a name followed through a
         [let] to its definition and through a parameter to each argument its applications supply. *)
      method make_functions env make =
        let rec go seen env make =
          let make = strip_constraints make in
          match make.pexp_desc with
          | Pexp_function _ -> Ok [ `Function (make, env) ]
          | Pexp_apply (_, arguments) -> Ok [ `Helper (make, arguments, env) ]
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
              match Lexical_scope.lookup env name with
              | Some { identity; definition = Some (rhs, rhs_env) } when not (Set.mem seen identity)
                ->
                  go (Set.add seen identity) rhs_env rhs
              | Some { identity; definition = None }
                when Hashtbl.mem parameters identity && not (Set.mem seen identity) -> (
                  let { owner; owner_name; slot; part } = Hashtbl.find_exn parameters identity in
                  let unseen why =
                    Error
                      (Printf.sprintf "its make function is `%s`, a parameter of `%s`, %s" name
                         owner_name why)
                  in
                  match (slot, part, Hashtbl.find foreign_escapes owner) with
                  | Element_slot _, _, _ | _, Some _, _ ->
                      (* A make function taken from a collection's element or out of a destructured
                         parameter: outside the model (gh-ocannl-2286), named uncompared. *)
                      unseen
                        "which is an element of a collection or a part of a destructured \
                         parameter, which the reader does not follow to a make function"
                  | _, _, Some why -> unseen why
                  | _ -> (
                      if self#escapes owner then
                        unseen
                          "whose name the source also uses other than as the function it applies, \
                           so not every application is seen"
                      else
                        match self#sites owner with
                        | [] ->
                            unseen
                              "which no application in this source or in a source spelling its \
                               module supplies"
                        | sites ->
                            List.fold sites ~init:(Ok []) ~f:(fun acc (_, arguments, site_env) ->
                                Result.bind acc ~f:(fun acc ->
                                    match self#argument_for slot arguments with
                                    | `Given e ->
                                        Result.map
                                          (go (Set.add seen identity) site_env e)
                                          ~f:(List.append acc)
                                    | `Default (d, d_env) ->
                                        Result.map
                                          (go (Set.add seen identity) d_env d)
                                          ~f:(List.append acc)
                                    | `Option _ | `Missing ->
                                        unseen "which an application of it does not supply"))))
              | _ ->
                  Error
                    (Printf.sprintf
                       "its make function is `%s`, a value given by name that the reader does not \
                        follow to a function"
                       name))
          | _ ->
              Error
                (Printf.sprintf
                   "its make function is `%s`, an expression the reader does not follow"
                   (Ppxlib.Pprintast.string_of_expression make))
        in
        go (Set.empty (module Int)) env make

      (* Walk [expr] again, in [scope], for the populations its claims add to [stack]'s collectors:
         a make function given by name, or a local function or [lazy] a make function runs. Nothing
         else is recorded ({!recording}). *)
      method replay stack scope expr =
        let saved = leg_stack in
        leg_stack <- stack;
        replaying <- replaying + 1;
        Exn.protect
          ~f:(fun () -> ignore (self#visit scope expr : expression))
          ~finally:(fun () ->
            leg_stack <- saved;
            replaying <- replaying - 1)

      (* A deferred value a make function runs (gh-ocannl-1893): a local function defined inside it
         and applied there, or a [lazy] defined inside it and forced there, is replayed with the
         collectors live -- what it claims is a claim [make] makes. *)
      method run_deferred env callee arguments =
        match leg_stack with
        | [] -> ()
        | collector :: _ -> (
            let start, stop = collector.make_span in
            let inside (e, _) = start <= position e && position e < stop in
            let definition name =
              match Lexical_scope.lookup env name with
              | Some { identity; definition = Some ((rhs, _) as definition) }
                when inside definition && not (Hash_set.mem replayed identity) ->
                  Some (identity, strip_constraints rhs, snd definition)
              | _ -> None
            in
            let run identity scope expr =
              Hash_set.add replayed identity;
              Exn.protect
                ~f:(fun () -> self#replay leg_stack scope expr)
                ~finally:(fun () -> Hash_set.remove replayed identity)
            in
            match ((strip_constraints callee).pexp_desc, arguments) with
            | _, [ (Ppxlib.Asttypes.Nolabel, thunk) ] when self#forces env callee -> (
                match (strip_constraints thunk).pexp_desc with
                | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                    match definition name with
                    | Some (identity, { pexp_desc = Pexp_lazy inner; _ }, scope) ->
                        run identity scope inner
                    | _ -> ())
                | _ -> ())
            | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ }, _ -> (
                (* Its body, run where a complete application supplies every positional and required
                   labelled parameter, each bound to what this application hands it. *)
                match definition name with
                | Some
                    ( identity,
                      { pexp_desc = Pexp_function (params, _, Pfunction_body body); _ },
                      scope )
                  when fully_applies
                         (function_kind ~binding:None params (Pfunction_body body))
                         (List.map arguments ~f:fst) ->
                    run identity (self#bind_applied ~definition_env:scope env params arguments) body
                | Some
                    ( identity,
                      {
                        pexp_desc =
                          Pexp_function (params, _, (Pfunction_cases (case_list, _, _) as cases));
                        _;
                      },
                      scope )
                  when fully_applies
                         (function_kind ~binding:None params cases)
                         (List.map arguments ~f:fst) -> (
                    (* A [function]'s cases, each a branch, its pattern taking apart the implicit
                       argument this application supplies: the positional one after the explicit
                       parameters'. *)
                    let explicit =
                      List.count params ~f:(fun param ->
                          match param.pparam_desc with
                          | Pparam_val (Ppxlib.Asttypes.Nolabel, _, _) -> true
                          | _ -> false)
                    in
                    let body_env = self#bind_applied ~definition_env:scope env params arguments in
                    match
                      List.nth
                        (List.filter_map arguments ~f:(function
                          | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                          | _ -> None))
                        explicit
                    with
                    | Some implicit ->
                        List.iter case_list ~f:(fun case ->
                            let case_env = self#bind_parts body_env case.pc_lhs (implicit, env) in
                            Option.iter case.pc_guard ~f:(run identity case_env);
                            run identity case_env case.pc_rhs)
                    | None -> ())
                | _ -> ())
            | _ -> ())

      (* A leg claim [expr] applies, with its [~leg] and its make function in place: its comparison
         is queued and its collector registered, for the walk of [make] to fill. Applied in part, it
         is reported unread. *)
      method leg_claim env expr ~member ~spelled ~ambiguous arguments =
        let line = expr.pexp_loc.loc_start.pos_lnum in
        let positional =
          List.filter_map arguments ~f:(function
            | Ppxlib.Asttypes.Nolabel, argument -> Some argument
            | _ -> None)
        in
        let leg =
          List.find_map arguments ~f:(function
            | Ppxlib.Asttypes.Labelled "leg", leg -> Some leg
            | _ -> None)
        in
        (* The make function: [in_leg]'s second positional argument, [in_legf]'s the one after its
           literal format's arguments -- complete only with exactly that many. *)
        let make =
          if String.equal member "in_leg" then
            if List.length positional = 2 then List.nth positional 1 else None
          else
            match
              Option.bind (List.hd positional) ~f:(fun format ->
                  Option.bind (Read.string_literal (strip_constraints format)) ~f:format_arity)
            with
            | Some arity when List.length positional = arity + 2 -> List.last positional
            | _ -> None
        in
        match (leg, make) with
        | Some leg, Some make -> (
            (* The make function written in place is walked with this collector innermost; one given
               by name is replayed into a collector of its own, under each resolution. *)
            let in_place =
              { populations = []; handed = []; branches = []; make_span = span_of make }
            in
            let point = self#point_of expr in
            let enclosing = List.hd enclosing_functions in
            let elements_here = enclosing_elements in
            (* The collectors [make] fills under the current {!context}: what each function it names
               claims, and what each helper it names is handed. *)
            let collectors () =
              Result.map (self#make_functions env make) ~f:(fun made ->
                  List.map made ~f:(function
                    | `Function (function_, _) when phys_equal function_ (strip_constraints make) ->
                        in_place
                    | `Function (function_, scope) ->
                        let collector =
                          {
                            populations = [];
                            handed = [];
                            branches = [];
                            make_span = span_of function_;
                          }
                        in
                        self#replay [ collector ] scope function_;
                        collector
                    | `Helper (helper, _, _) when phys_equal helper (strip_constraints make) ->
                        in_place
                    | `Helper (helper, helper_arguments, scope) ->
                        {
                          populations = [];
                          handed =
                            List.map helper_arguments ~f:(fun (_, argument) ->
                                (position helper, argument, scope));
                          branches = [];
                          make_span = span_of helper;
                        }))
            in
            (* The population bindings one collector's claims are built from. *)
            let built_by collector =
              let built = Hash_set.Poly.create () in
              (* A helper's arguments are its claim's population on a path where no claim read in
                 place runs: where every such claim sits in another branch of some conditional than
                 the helper. *)
              let disjoint a b =
                List.exists collector.branches ~f:(fun spans ->
                    let branch_of point =
                      List.findi spans ~f:(fun _ (start, stop) -> start <= point && point < stop)
                      |> Option.map ~f:fst
                    in
                    match (branch_of a, branch_of b) with Some i, Some j -> i <> j | _ -> false)
              in
              let populations =
                List.map collector.populations ~f:(fun (_, e, boolean, e_env) ->
                    (e, boolean, e_env))
                @ List.filter_map collector.handed ~f:(fun (helper, e, e_env) ->
                    if
                      List.for_all collector.populations ~f:(fun (claim, _, _, _) ->
                          disjoint claim helper)
                    then Some (e, false, e_env)
                    else None)
              in
              List.iter populations ~f:(fun (population, boolean, population_env) ->
                  Hash_set.iter
                    (if boolean then self#quantified population_env population
                     else self#population population_env population)
                    ~f:(Hash_set.add built));
              built
            in
            pending :=
              (fun () ->
                unlisted_sink := [];
                claim_point <- Some point;
                let uncompared why =
                  uncompared_legs :=
                    { leg_form = spelled; leg_line = line; why } :: !uncompared_legs
                in
                match self#leg_gates env leg with
                | Error why -> uncompared why
                | Ok resolutions -> (
                    (* Each resolution is compared under its own applications, so that one call's
                       gate never meets another call's population. *)
                    (* Under each chain of applications reaching the claim's enclosing function,
                       too, so that a leg the helper builds from its own parameters is compared per
                       call (gh-ocannl-1893, local review). *)
                    let chains = self#call_chains enclosing in
                    let resolutions =
                      List.concat_map resolutions ~f:(fun resolution ->
                          match
                            List.filter_map chains ~f:(fun chain ->
                                if calls_compatible resolution.calls chain then
                                  Some
                                    {
                                      resolution with
                                      calls =
                                        List.dedup_and_sort (resolution.calls @ chain)
                                          ~compare:Poly.compare;
                                    }
                                else None)
                          with
                          | [] -> [ resolution ]
                          | chosen -> chosen)
                      (* And under each choice of the elements its enclosing callbacks iterate. *)
                      |> List.concat_map ~f:(fun resolution ->
                          List.map (self#element_contexts ~here:elements_here resolution.calls)
                            ~f:(fun calls -> { resolution with calls }))
                    in
                    let compared =
                      List.map resolutions ~f:(fun resolution ->
                          context <- resolution.calls;
                          Exn.protect
                            ~finally:(fun () -> context <- [])
                            ~f:(fun () ->
                              Result.map (collectors ()) ~f:(fun collectors ->
                                  let reads = Hash_set.Poly.create () in
                                  List.iter resolution.leg_gates ~f:(fun (gate, gate_env) ->
                                      Hash_set.iter (self#gate_reads gate_env gate)
                                        ~f:(Hash_set.add reads));
                                  let shared =
                                    List.concat_map collectors ~f:(fun collector ->
                                        Hash_set.filter (built_by collector) ~f:(Hash_set.mem reads)
                                        |> Hash_set.to_list |> List.map ~f:fst)
                                  in
                                  (resolution, shared))))
                    in
                    match Result.all compared with
                    | Error why -> uncompared why
                    | Ok compared ->
                        Int.incr leg_sites;
                        List.dedup_and_sort !unlisted_sink ~compare:Poly.compare
                        |> List.iter ~f:(fun (callee, callee_line) ->
                            unlisted :=
                              { claim = spelled; claim_line = line; callee; callee_line }
                              :: !unlisted);
                        let offending =
                          List.filter compared ~f:(fun (_, shared) -> not (List.is_empty shared))
                        in
                        if not (List.is_empty offending) then
                          readings :=
                            {
                              form = spelled;
                              ambiguous =
                                ambiguous
                                || List.exists offending ~f:(fun (r, _) -> r.leg_ambiguous);
                              line;
                              gate =
                                List.concat_map offending ~f:(fun (r, _) ->
                                    List.map r.leg_gates ~f:(fun (gate, _) ->
                                        Ppxlib.Pprintast.string_of_expression gate))
                                |> List.dedup_and_sort ~compare:String.compare
                                |> String.concat ~sep:" && ";
                              reads =
                                List.concat_map offending ~f:snd
                                |> List.dedup_and_sort ~compare:String.compare;
                              applied_in =
                                List.concat_map offending ~f:(fun (r, _) -> callers_in r.calls)
                                |> List.dedup_and_sort ~compare:String.compare;
                            }
                            :: !readings))
              :: !pending;
            (* Not a name: what it names is replayed where the comparison resolves it. *)
            match (strip_constraints make).pexp_desc with
            | Pexp_ident _ -> ()
            | _ -> Hashtbl.set leg_makes ~key:(position make) ~data:in_place)
        | _ ->
            unread_legs :=
              {
                leg_form = spelled;
                leg_line = line;
                why =
                  "it is applied in part: its `~leg`, its label and its make function are not all \
                   handed to it in place";
              }
              :: !unread_legs

      (* The gated claim an expression reaches, with every argument it has been handed so far, each
         with the scope it was written in. The nodes it absorbs -- the callee of an application, an
         application a pipeline rewrites -- are marked [consumed]: read here, they are not read
         again on their own. Whether a name is [Verdict]'s is {!Verdict_callee}'s answer: what it
         reads as [Verdict]'s or [Ambiguous], or as [Free] -- a [gated] name exists nowhere but in
         [Verdict.Claims], so one nothing binds is read too. *)
      method gated env expr ~through ~marks =
        let verdicts () =
          match Verdict_callee.resolve owners expr with
          | Some { member; ownership = (Verdicts | Ambiguous | Free) as ownership; spelled }
            when List.mem (gated_boolean :: gated_forms) member ~equal:String.equal ->
              Some
                {
                  reached_form = spelled;
                  member;
                  reached_ambiguous = Poly.equal ownership Verdict_callee.Ambiguous;
                  arguments = [];
                }
          | _ -> None
        in
        match expr.pexp_desc with
        | Pexp_constraint (inner, _) -> self#gated env inner ~through ~marks
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            (* A local binding of the name comes first: a file's own [gated_all] is not [Verdict]'s,
               and a [let check = Verdict.gated_all ...] is followed. *)
            match Lexical_scope.lookup env name with
            | Some { identity; definition = Some (rhs, rhs_env) }
              when not (Hash_set.mem through identity) ->
                Hash_set.add through identity;
                (* The binding this use completes, through every renaming on the way. *)
                Hash_set.add marks (position rhs);
                Option.map (self#gated rhs_env rhs ~through ~marks) ~f:(fun reach ->
                    { reach with reached_form = name })
            | Some _ -> None
            | None -> verdicts ())
        | Pexp_ident _ -> verdicts ()
        | Pexp_apply (callee, arguments) -> (
            let callee, arguments, absorbed =
              match self#pipeline callee arguments with
              | Some rewritten -> rewritten
              | None -> (callee, arguments, None)
            in
            match self#gated env callee ~through ~marks with
            | Some reach ->
                Option.iter absorbed ~f:(Hash_set.add marks);
                Hash_set.add marks (position callee);
                Some
                  {
                    reach with
                    arguments =
                      reach.arguments
                      @ List.map arguments ~f:(fun (label, argument) -> (label, argument, env));
                  }
            | None -> None)
        | _ -> None

      (* A [for] loop's bounds and a binding operator's operands are evaluated once, where the
         expression is: only the body repeats, or is deferred, and it enters its frame before
         anything in it -- a claim or a write that is the body itself -- is placed. *)
      method! expression env expr =
        let dispatch () =
          match Hashtbl.find_and_remove body_frames (position expr) with
          | Some kind ->
              self#framed ~kind ~start:expr.pexp_loc.loc_start.pos_cnum
                ~stop:expr.pexp_loc.loc_end.pos_cnum (fun () -> self#visit env expr)
          | None -> self#visit env expr
        in
        let with_stack stack =
          let saved = leg_stack in
          leg_stack <- stack;
          Exn.protect ~f:dispatch ~finally:(fun () -> leg_stack <- saved)
        in
        (* A leg claim's make function: walked with its collector innermost. Inside it, a function
           value no collection function calls back, or a lazy, is deferred: what it claims is no
           claim [make] makes, so the collectors pause there. *)
        match if quiet > 0 then None else Hashtbl.find_and_remove leg_makes (position expr) with
        | Some collector -> with_stack (collector :: leg_stack)
        | None -> (
            match expr.pexp_desc with
            | (Pexp_function _ | Pexp_lazy _)
              when quiet = 0
                   && (not (List.is_empty leg_stack))
                   && (not (Hash_set.mem synchronous (position expr)))
                   && not (replaying > 0 && Hash_set.mem replay_live (position expr)) ->
                with_stack []
            | _ -> dispatch ())

      method visit env expr =
        (if self#recording then
           let span e = (e.pexp_loc.loc_start.pos_cnum, e.pexp_loc.loc_end.pos_cnum) in
           let operands =
             match expr.pexp_desc with
             | Pexp_apply (callee, arguments) -> (
                 match (strip_constraints callee).pexp_desc with
                 | Pexp_ident
                     { txt = Ppxlib.Longident.Lident (("&&" | "||" | "&" | "or") as name); _ }
                   when Option.is_none (Lexical_scope.lookup env name) ->
                     (* The built-in short-circuit operators; a rebound one is an application. *)
                     []
                 | _ -> callee :: List.map arguments ~f:snd)
             | Pexp_tuple elements | Pexp_array elements -> elements
             | Pexp_record (fields, base) -> Option.to_list base @ List.map fields ~f:snd
             | Pexp_setfield (record, _, value) -> [ record; value ]
             | Pexp_let (_, (_ :: _ :: _ as bindings), _) ->
                 List.map bindings ~f:(fun binding -> binding.pvb_expr)
             | Pexp_letop { let_; ands = _ :: _ as ands; _ } ->
                 List.map (let_ :: ands) ~f:(fun operand -> operand.pbop_exp)
             | _ -> []
           in
           if List.length operands >= 2 then
             unordered_operands := List.map operands ~f:span :: !unordered_operands);
        (* The gate's side ({!identifiers}): a collection-module function the table does not list,
           which may compute anything from what it is handed, is reported as the population side's
           is. *)
        (match (collecting, expr.pexp_desc) with
        | Some _, Pexp_apply (callee, arguments) -> (
            let callee =
              match self#pipeline callee arguments with
              | Some (callee, _, _) -> callee
              | None -> callee
            in
            match self#resolve_callee env callee with
            | ( ({ pexp_desc = Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ }; _ } as
                 resolved),
                callee_env ) -> (
                match self#module_path callee_env prefix with
                | Some (Collection_module _) when Option.is_none (collection_function name) ->
                    self#note_unlisted resolved
                | _ -> ())
            | _ -> ())
        | _ -> ());
        (if self#recording then
           match expr.pexp_desc with
           | Pexp_apply (callee, arguments) -> (
               (* [sample |> Queue.enqueue queue] is [Queue.enqueue queue sample]. *)
               (* What runs where: the application of a name, for {!once}; the functions a
                  collection-module function is handed, which it calls synchronously. *)
               (match (strip_constraints callee).pexp_desc with
               | Pexp_ident { txt = Ppxlib.Longident.Lident name; loc } -> (
                   Hashtbl.set applications ~key:loc.loc_start.pos_cnum
                     ~data:(List.map arguments ~f:fst, self#point_of expr);
                   (* Where a parameter of a named function takes its value (gh-ocannl-1893). *)
                   match Lexical_scope.lookup env name with
                   | Some { identity; _ } when identity > 0 ->
                       Hashtbl.add_multi call_sites ~key:identity
                         ~data:(loc.loc_start.pos_cnum, arguments, env);
                       Hashtbl.set site_elements ~key:loc.loc_start.pos_cnum
                         ~data:enclosing_elements;
                       Hashtbl.set call_callers ~key:loc.loc_start.pos_cnum
                         ~data:(List.hd enclosing_functions)
                   | _ -> ())
               | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); loc } -> (
                   (* [M.f args], [M] a structure of this source or, in another source's walk, this
                      source itself (gh-ocannl-2286): an application of [f]'s binding. *)
                   match self#structure_value env prefix name with
                   | Some identity ->
                       Hashtbl.add_multi call_sites ~key:identity
                         ~data:(loc.loc_start.pos_cnum, arguments, env);
                       Hashtbl.set site_elements ~key:loc.loc_start.pos_cnum
                         ~data:enclosing_elements;
                       Hashtbl.set call_callers ~key:loc.loc_start.pos_cnum
                         ~data:(List.hd enclosing_functions)
                   | None -> ())
               | _ -> ());
               (* A pipeline is read as the application it spells, so that [rows |> List.iter ~f] is
                  complete; only a complete application calls its callback. *)
               let applied, applied_arguments =
                 match self#pipeline callee arguments with
                 | Some (callee, arguments, _) -> (callee, arguments)
                 | None -> (callee, arguments)
               in
               (match self#resolve_callee env applied with
               | ( { pexp_desc = Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ }; _ },
                   callee_env ) -> (
                   (* The module is the one the alias's definition named, in its own scope. *)
                   let arguments = applied_arguments in
                   match self#module_path callee_env prefix with
                   | Some (Collection_module { stdlib; family; _ })
                     when runs_callback ~family name
                          && applied_fully ~stdlib name (List.map arguments ~f:fst) ->
                       let positional = ref 0 in
                       List.iter arguments ~f:(fun (label, argument) ->
                           let index = !positional in
                           (match label with
                           | Ppxlib.Asttypes.Nolabel -> Int.incr positional
                           | _ -> ());
                           match (strip_constraints argument).pexp_desc with
                           | Pexp_function _ when calls_back ~stdlib name label index ->
                               Hash_set.add synchronous (position (strip_constraints argument))
                           | _ -> ())
                   | _ -> ())
               | _ -> ());
               (* A callback an {!element_iterations} function is handed in place: its element
                  parameter is each element of the collection (gh-ocannl-2286). The callback is the
                  [~f] argument with the collection the one positional argument, or, with no [~f],
                  the first of exactly two positional ones, the collection the second. *)
               (match self#resolve_callee env applied with
               | ( { pexp_desc = Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ }; _ },
                   callee_env ) -> (
                   match
                     ( self#module_path callee_env prefix,
                       List.Assoc.find element_iterations name ~equal:String.equal )
                   with
                   | Some (Collection_module { family; _ }), Some index
                     when List.mem element_families family ~equal:String.equal -> (
                       let positional =
                         List.filter_map applied_arguments ~f:(function
                           | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                           | _ -> None)
                       in
                       let labelled_f =
                         List.find_map applied_arguments ~f:(function
                           | Ppxlib.Asttypes.Labelled "f", argument -> Some argument
                           | _ -> None)
                       in
                       let others_labelled =
                         List.exists applied_arguments ~f:(function
                           | Ppxlib.Asttypes.Nolabel, _ | Ppxlib.Asttypes.Labelled "f", _ -> false
                           | _ -> true)
                       in
                       match (labelled_f, positional) with
                       | (Some callback, [ collection ] | None, [ callback; collection ])
                         when not others_labelled -> (
                           match (strip_constraints callback).pexp_desc with
                           | Pexp_function _ ->
                               Hashtbl.set element_callbacks
                                 ~key:(position (strip_constraints callback))
                                 ~data:
                                   ( index,
                                     String.concat ~sep:"."
                                       (Read.flatten_longident prefix @ [ name ]),
                                     collection,
                                     env )
                           | _ -> ())
                       | _ -> ())
                   | _ -> ())
               | _ -> ());
               (* The application a pipeline absorbs ([Queue.enqueue queue] in [sample |>
                  Queue.enqueue queue]) is written through here, not again on its own, where it
                  would read as partially applied. *)
               match self#pipeline callee arguments with
               | Some (callee, arguments, absorbed) ->
                   Option.iter absorbed ~f:(Hash_set.add piped);
                   self#record_write env expr callee arguments
               | None ->
                   if not (Hash_set.mem piped (position expr)) then
                     self#record_write env expr callee arguments)
           | Pexp_setfield (record, field, value) ->
               (* [probe.found <- v] writes [v] into [probe]'s [found]: following [probe] whole
                  reaches it, and so does taking [found] apart ({!parts}). *)
               let point =
                 { (self#point_of expr) with effect_at = expr.pexp_loc.loc_end.pos_cnum }
               in
               Option.iter (self#write_target env record) ~f:(fun identity ->
                   Hashtbl.add_multi field_writes ~key:identity
                     ~data:(Ppxlib.Longident.last_exn field.txt, (value, env, point)))
           | _ -> ());
        (* A leg claim (gh-ocannl-1794): its [make] is registered to be walked with its collector
           innermost (see [expression]), so the claims made there add their populations; one named
           as a value, or applied in part, is reported unread. *)
        (match (quiet, leg_stack) with
        | 0, collector :: _ -> (
            let span e = (e.pexp_loc.loc_start.pos_cnum, e.pexp_loc.loc_end.pos_cnum) in
            let record spans = collector.branches <- spans :: collector.branches in
            match expr.pexp_desc with
            | Pexp_ifthenelse (_, yes, no) ->
                record (span yes :: Option.to_list (Option.map no ~f:span))
            | Pexp_match (_, cases) | Pexp_function (_, _, Pfunction_cases (cases, _, _)) ->
                record (List.map cases ~f:(fun case -> span case.pc_rhs))
            | _ -> ())
        | _ -> ());
        (if quiet = 0 then
           let leg_member callee =
             match Verdict_callee.resolve owners callee with
             | Some { member; ownership = (Verdicts | Ambiguous | Free) as ownership; spelled }
               when List.mem leg_claims member ~equal:String.equal ->
                 Some (member, spelled, Poly.equal ownership Verdict_callee.Ambiguous)
             | _ -> None
           in
           match expr.pexp_desc with
           | Pexp_apply (callee, arguments) -> (
               self#collect_leg_population env expr callee arguments;
               self#run_deferred env callee arguments;
               match leg_member callee with
               | Some _ when foreign -> ()
               | Some (member, spelled, ambiguous) when self#recording ->
                   Hash_set.add leg_heads (position (strip_constraints callee));
                   self#leg_claim env expr ~member ~spelled ~ambiguous arguments
               | Some _ ->
                   (* A leg claim nested in a replayed function: its make function's claims are the
                      enclosing make function's, as in the walk itself. *)
                   List.iter arguments ~f:(fun (_, argument) ->
                       match (strip_constraints argument).pexp_desc with
                       | Pexp_function _ ->
                           Hash_set.add replay_live (position argument);
                           Hash_set.add replay_live (position (strip_constraints argument))
                       | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                           (* One given by name: its definition, replayed with the collectors. *)
                           match Lexical_scope.lookup env name with
                           | Some
                               {
                                 identity;
                                 definition =
                                   Some (({ pexp_desc = Pexp_function _; _ } as function_), scope);
                               }
                             when not (Hash_set.mem replayed identity) ->
                               Hash_set.add replayed identity;
                               Exn.protect
                                 ~f:(fun () -> self#replay leg_stack scope function_)
                                 ~finally:(fun () -> Hash_set.remove replayed identity)
                           | _ -> ())
                       | _ -> ())
               | None -> ())
           | Pexp_ident _
             when self#recording && (not foreign) && not (Hash_set.mem leg_heads (position expr))
             -> (
               match leg_member expr with
               | Some (_, spelled, _) ->
                   unread_legs :=
                     {
                       leg_form = spelled;
                       leg_line = expr.pexp_loc.loc_start.pos_lnum;
                       why = "it is named as a value rather than applied";
                     }
                     :: !unread_legs
               | None -> ())
           | _ -> ());
        (if self#recording && (not foreign) && not (Hash_set.mem consumed (position expr)) then
           match expr.pexp_desc with
           | Pexp_ident _ -> (
               (* A claim named as a value rather than applied -- handed to [List.iter], say -- is
                  one this reader cannot complete. A [let] binding it to another name is not a use;
                  that name's own uses are. *)
               let marks = Hash_set.create (module Int) in
               match self#gated env expr ~through:(Hash_set.create (module Int)) ~marks with
               | Some { reached_form; arguments; _ }
                 when (not (complete arguments)) && not (Hash_set.mem bound (position expr)) ->
                   commit marks;
                   partials := (reached_form, expr.pexp_loc.loc_start.pos_lnum) :: !partials
               | _ -> ())
           | Pexp_apply _ -> (
               let marks = Hash_set.create (module Int) in
               match self#gated env expr ~through:(Hash_set.create (module Int)) ~marks with
               | None -> ()
               | Some { reached_form = form; member; reached_ambiguous; arguments } -> (
                   let gate =
                     List.find_map arguments ~f:(function
                       | Ppxlib.Asttypes.Labelled "when_", argument, env -> Some (argument, env)
                       | _ -> None)
                   in
                   let positional =
                     List.filter_map arguments ~f:(function
                       | Ppxlib.Asttypes.Nolabel, argument, env -> Some (argument, env)
                       | _ -> None)
                   in
                   let line = expr.pexp_loc.loc_start.pos_lnum in
                   match (gate, List.nth positional 1) with
                   | Some (gate, gate_env), Some (population, population_env)
                     when complete arguments ->
                       commit marks;
                       Int.incr gated_sites;
                       let point = self#point_of expr in
                       let enclosing = List.hd enclosing_functions in
                       let elements_here = enclosing_elements in
                       pending :=
                         (fun () ->
                           unlisted_sink := [];
                           claim_point <- Some point;
                           (* Inside a named function its applications supply, compared once per
                              application, so that one call's gate never meets another call's
                              population (gh-ocannl-1893). *)
                           let contexts =
                             List.concat_map (self#call_chains enclosing)
                               ~f:(self#element_contexts ~here:elements_here)
                           in
                           let per_context =
                             List.map contexts ~f:(fun calls ->
                                 context <- calls;
                                 Exn.protect
                                   ~finally:(fun () -> context <- [])
                                   ~f:(fun () ->
                                     let reads = self#gate_reads gate_env gate in
                                     let built =
                                       if String.equal member gated_boolean then
                                         self#quantified population_env population
                                       else self#population population_env population
                                     in
                                     ( calls,
                                       Hash_set.filter built ~f:(Hash_set.mem reads)
                                       |> Hash_set.to_list |> List.map ~f:fst )))
                           in
                           let shared =
                             List.concat_map per_context ~f:snd
                             |> List.dedup_and_sort ~compare:String.compare
                           in
                           let applied_in =
                             List.concat_map per_context ~f:(fun (calls, shared) ->
                                 if List.is_empty shared then [] else callers_in calls)
                             |> List.dedup_and_sort ~compare:String.compare
                           in
                           List.dedup_and_sort !unlisted_sink ~compare:Poly.compare
                           |> List.iter ~f:(fun (callee, callee_line) ->
                               unlisted :=
                                 { claim = form; claim_line = line; callee; callee_line }
                                 :: !unlisted);
                           if not (List.is_empty shared) then
                             readings :=
                               {
                                 form;
                                 ambiguous = reached_ambiguous;
                                 line;
                                 gate = Ppxlib.Pprintast.string_of_expression gate;
                                 reads = shared;
                                 applied_in;
                               }
                               :: !readings)
                         :: !pending
                   | _ ->
                       (* A partial application bound by [let] is completed at that name's uses,
                          which are read on their own -- unless none is, below. *)
                       if Hash_set.mem bound (position expr) then
                         bound_partials := (position expr, form, line) :: !bound_partials
                       else (
                         commit marks;
                         partials := (form, line) :: !partials)))
           | _ -> ());
        let start = expr.pexp_loc.loc_start.pos_cnum and stop = expr.pexp_loc.loc_end.pos_cnum in
        let walk () = super#expression env expr in
        match expr.pexp_desc with
        | Pexp_for (_, _, _, _, body) ->
            Hashtbl.set body_frames ~key:(position body) ~data:Loop;
            walk ()
        | Pexp_letop { body; _ } ->
            Hashtbl.set body_frames ~key:(position body) ~data:Deferred;
            walk ()
        | Pexp_while _ -> self#framed ~kind:Loop ~start ~stop walk
        | Pexp_function _ when Hash_set.mem synchronous (position expr) ->
            self#framed ~kind:Loop ~start ~stop (fun () -> self#function_scope expr walk)
        | Pexp_function (params, _, body) ->
            self#framed
              ~kind:
                (function_kind
                   ~binding:(Hashtbl.find function_bindings (position expr))
                   params body)
              ~start ~stop
              (fun () -> self#function_scope expr walk)
        | Pexp_lazy _ | Pexp_object _ -> self#framed ~kind:Deferred ~start ~stop walk
        | _ -> walk ()
    end
  in
  let top = { Lexical_scope.frames = []; modules = Map.empty (module String) } in
  ignore (walker#structure top ast : structure);
  (* The applications other sources make of this one's functions (gh-ocannl-2286): each caller is
     walked as code of its own, its positions past every earlier source's so that no position or
     binding identity is shared, under one module binding -- [M], this source's structure -- and
     what it records is read: its [M.f] applications are call sites of [f]. Only where this source
     has a claim to compare: otherwise none of it could be read. *)
  (match callers with
  | Some { module_name; sources } when not (List.is_empty !pending) -> (
      match
        walker#module_of top
          (Ppxlib.Ast_builder.Default.pmod_structure ~loc:Ppxlib.Location.none ast)
      with
      | Some (Local_structure { values; modules; _ } as structure) ->
          (* Every function of [M], its submodules' included, so [M.Sub.f] too. *)
          let escape why =
            let rec mark values modules =
              List.iter values ~f:(fun (_, { identity; _ }) ->
                  if identity > 0 && not (Hashtbl.mem foreign_escapes identity) then
                    Hashtbl.set foreign_escapes ~key:identity ~data:why);
              List.iter modules ~f:(function
                | _, Local_structure { values; modules; _ } -> mark values modules
                | _ -> ())
            in
            mark values modules
          in
          List.iter sources ~f:(fun { source; text; spelled_elsewhere } ->
              let at = !next_offset in
              next_offset := at + String.length text + 1;
              caller_spans := (at, at + String.length text + 1, source) :: !caller_spans;
              match Scan_attempt.run (fun () -> structure_at ~offset:at text) with
              | Scanned caller ->
                  Hashtbl.set walked_callers ~key:source
                    ~data:
                      ( caller,
                        {
                          top with
                          Lexical_scope.modules =
                            Map.singleton (module String) module_name structure;
                        } );
                  Verdict_callee.absorb ~into:owners (Verdict_callee.of_structure caller);
                  binder#structure caller;
                  (match module_spelled_otherwise ~module_name ~spelled_elsewhere caller with
                  | Some how ->
                      escape
                        (Printf.sprintf
                           "whose module `%s` another source, `%s`, spells %s, so not every \
                            application is seen"
                           module_name source how)
                  | None -> ());
                  walker#walk_foreign (fun () ->
                      ignore
                        (walker#structure
                           {
                             top with
                             Lexical_scope.modules =
                               Map.singleton (module String) module_name structure;
                           }
                           caller
                          : structure))
              | Unparsed report ->
                  escape
                    (Printf.sprintf
                       "whose module `%s` another source, `%s`, spells, and that source does not \
                        parse (%s), so not every application is seen"
                       module_name source report)
              | Raised exn ->
                  failwith (Printf.sprintf "%s (while parsing the caller %s)" exn source))
      | _ -> ())
  | _ -> ());
  List.iter (List.rev !pending) ~f:(fun compare -> compare ());
  (* A bound partial application no resolved use completed -- reached only through a module path,
     say, or from another source -- is one this reader never compared. *)
  let uncompleted =
    List.filter_map !bound_partials ~f:(fun (at, form, line) ->
        if Hash_set.mem consumed at then None else Some (form, line))
  in
  let unread =
    List.sort (!partials @ uncompleted) ~compare:(fun (_, a) (_, b) -> Int.compare a b)
  in
  {
    readings = List.rev !readings;
    gated_sites = !gated_sites;
    unread;
    unlisted = List.rev !unlisted;
    leg_sites = !leg_sites;
    unread_legs = List.sort !unread_legs ~compare:(fun a b -> Int.compare a.leg_line b.leg_line);
    uncompared_legs =
      List.sort !uncompared_legs ~compare:(fun a b -> Int.compare a.leg_line b.leg_line);
  }

(** {1 A hand-written skip list (gh-ocannl-1794)}

    A claim sequence only some backends can evaluate is written ONCE, as a leg ({!Verdict.leg},
    {!Verdict.in_leg}): where the leg is closed each claim skips under its own label, in the order
    the same code reaches it. The two-branch form the legs replaced kept the off-gate side in a
    second copy -- the evaluating branch's claims, then [List.iter claims ~f:skipped] over a
    hand-written list, or a run of literal [skipped] calls -- whose order nothing tied to the first:
    [grid_launch_floor] appended two claims to its list while the CPU branch made them mid-sequence,
    and Metal, CUDA and HIP went red (gh-ocannl-1668). gh-ocannl-1703 migrated about thirty such
    tests; this reader keeps the form from regrowing.

    {2 The model (the boundary, stated once)}

    This reader decides skip lists under a BOUNDED, static model of what runs. It reads exactly the
    constructs listed below; anything else is outside the model and is read CONSERVATIVELY: as no
    skip list, never refused. A construct outside the model can hide a skip list from this reader;
    it is never a reason to report one. Widening the model is a change to this header first.

    - A SKIP is a [skipped] {!Verdict_callee} reads as [Verdict]'s ([Verdict.skipped], an opened
      one), or a name of any spelling a [let] of the source binds to a skip or its application (the
      per-test [let skipped = Verdict.skipped ~backend:backend_name], or [let skip = ...]), through
      [open]s of the standard libraries; never a parameter or other binding of the name, nor another
      module's. A skip RUNS where it is applied to its label (a positional argument); a partial skip
      is a callback, read only where a collection function is handed it.
    - An EVALUATING claim is an application of any other {!Verdict_callee.members} name but [fail]
      and [case] that the resolver reads as [Verdict]'s.
    - What RUNS: the code written in place. A function value and a [lazy] are DEFERRED -- their
      bodies are not read as run -- except a function a collection-module function calls back, and a
      DEFERRED SKIP VALUE where it is run (gh-ocannl-1893).
    - A DEFERRED SKIP VALUE is a function or a [lazy] a named [let] of the source binds (or a [let]
      renaming one), resolved lexically, so a shadowing binding is not it. Its run code is its body
      read as above, down to the branches nested in it, which are judged where they are written. It
      RUNS where a complete application of its name supplies every positional and required labelled
      parameter of the function, or where its name is forced ([Lazy.force], [Lazy.force_val],
      [Base]'s unqualified [force]) -- a [lazy] once per branch, being memoised. There it runs its
      run code: a SKIP WRAPPER (one whose body evaluates no claim, its nested branches included) its
      skips, through wrappers of wrappers (a cycle runs nothing); any other its evaluating claims,
      so a branch calling a function that evaluates a claim evaluates one. A skip wrapper's name, or
      its partial application, handed where a collection function calls it back is a skip handed
      there.
    - A STORED ITERATION is a name a [let] binds to a partial application of a collection iteration
      (one not {!applied_fully}), or to a renaming of one: an application of the name completes it,
      the stored arguments followed by the new ones, and is read as that iteration.
    - A COLLECTION ITERATION is a complete ({!applied_fully}) application of a function of
      {!collection_modules} -- the module resolved through the source's own module aliases and
      structures, [Option] and [Result] excluded (at most one run), [Seq]/[Sequence] only through
      the functions {!collection_functions} marks as consuming ({!runs_callback}) -- written
      directly or through [|>]/[@@]; its argument is CALLED BACK where {!calls_back} says so, in
      either library's order. A complete application of a consuming function also calls back the
      callbacks of a lazy sequence handed to it positionally: a [Seq]/[Sequence] function that does
      not consume ([Seq.map skipped s]) carries its callback, through the names it is bound to and
      the sequences it is built from, to the consumer that runs it.
    - A BRANCH is one branch of an [if], or one case of a [match] or [function].

    Two shapes are refused:

    - an ITERATED skip: a called-back argument of a collection iteration that is a skip itself,
      partially applied or not ([List.iter labels ~f:skipped], [~f:(Verdict.skipped ~backend)]), or
      a function written there whose run code makes a skip and no evaluating claim
      ([~f:(fun l -> skipped (tag ^ l))]); and the body of a [for] or [while] loop whose run code
      makes a skip and no evaluating claim. A loop whose callback evaluates the claim on one side of
      its gate and skips it on the other is one claim written once, and passes;
    - a SKIP RUN: a branch whose run code applies two or more skips, read down to the branches
      nested in it, which are judged on their own. A run is counted up to 64 skips, as each deferred
      skip value's summary keeps at most that many, so a chain of wrappers that each run the
      previous one twice stays linear.

    {2 Outside the model}

    Known examples, each read as no skip list: labels iterated by a function of the source rather
    than a collection module's; a run of single skips spread across nested branches; a deferred
    value reached other than by its name -- stored in a container or a field, handed as an argument
    and run through the parameter, or another module's; a stored iteration completed through a
    parameter or a container; and a lazy sequence consumed by a function of the source. *)

type skip_list_kind =
  | Iterated  (** A skip handed to an iteration, or a loop body that only skips. *)
  | Skip_run of int  (** A branch applying this many skips. *)

type skip_list = {
  skip_kind : skip_list_kind;
  skip_line : int;  (** Where the iteration, loop or branch starts. *)
  first_label : string;
      (** The first skip's label: its literal, or its expression printed; the skip itself printed
          where it is not applied. What an exemption is keyed by, after the source. *)
}

type skip_list_census = {
  skip_lists : skip_list list;  (** In source order. *)
  skips_read : int;  (** Every skip the reader met. *)
}

let skip_name = "skipped"
let skip_list_unread_members = [ "fail"; "case"; skip_name ]

(* What a name of the source denotes to the skip-list reader: whether it is a skip, and the named
   [let] binding it resolves to, if one does -- what a deferred skip value ({!skip_list_census}) is
   read from. *)
type skip_denotation = { is_skip : bool; bound : value_binding option }

(* The positions of the identifiers that are skips -- a [skipped] {!Verdict_callee} reads as
   [Verdict]'s (qualified, or unqualified where nothing of the source binds it), and a name of any
   spelling a [let] binds to a skip or a skip's application (the per-test [let skipped =
   Verdict.skipped ~backend:backend_name], or [let skip = Verdict.skipped ~backend]), but not a
   parameter, a pattern, a [let] of anything else, or another module's -- and, for every unqualified
   name, the named [let] binding it resolves to, by the name's position. *)
let skip_positions ~owners ast =
  let positions = Hash_set.create (module Int) in
  let uses = Hashtbl.create (module Int) in
  let bound = Hash_set.create (module Int) in
  let verdicts loc =
    match Verdict_callee.resolve_at owners loc with
    | Some { ownership = Verdicts | Ambiguous | Free; _ } -> true
    | _ -> false
  in
  let resolved env name loc =
    match Lexical_scope.lookup env name with
    | Some (Some { is_skip; _ }) -> is_skip
    | Some None | None -> String.equal name skip_name && verdicts loc
  in
  let walker =
    object
      inherit [skip_denotation option, bool] Lexical_scope.scoped

      (* [Some d]: the source binds the name, to a skip or not; [None]: an [open] or [include] may
         have brought a [skipped] in, which the resolver decides. *)
      method local = Some { is_skip = false; bound = None }

      (* A module's denotation: whether it may export a [skipped] -- a standard library does not
         ({!Verdict_callee.standard_libraries}); any other module, or [Verdict] itself, may. *)
      method module_path _ path =
        Some
          (not
             (List.mem Verdict_callee.standard_libraries (Lexical_scope.head path)
                ~equal:String.equal))

      method! attribute _ attr = attr

      method! opened ~top:_ ~include_:_ env denotes =
        match denotes with
        | Some false -> env
        | Some true | None ->
            Lexical_scope.push_frame env (Map.singleton (module String) skip_name None)

      method! binding_denotes env binding =
        let rec skip_head expr =
          match (strip_constraints expr).pexp_desc with
          | Pexp_ident { txt = Ppxlib.Longident.Lident name; loc } -> resolved env name loc
          | Pexp_ident { txt = Ppxlib.Longident.Ldot (_, name); loc } ->
              String.equal name skip_name && verdicts loc
          | Pexp_apply (callee, _) -> skip_head callee
          | _ -> false
        in
        let named = Option.is_some (Read.pattern_name binding.pvb_pat) in
        Some
          {
            is_skip = named && skip_head binding.pvb_expr;
            bound = (if named then Some binding else None);
          }

      method! ident env ident =
        let skip =
          match ident.txt with
          | Ppxlib.Longident.Lident name -> resolved env name ident.loc
          | Ldot (_, name) -> String.equal name skip_name && verdicts ident.loc
          | _ -> false
        in
        if skip then Hash_set.add positions ident.loc.loc_start.pos_cnum;
        match ident.txt with
        | Lident name -> (
            match Lexical_scope.lookup env name with
            | Some (Some { bound = Some binding; _ }) ->
                Hash_set.add bound ident.loc.loc_start.pos_cnum;
                Hashtbl.set uses ~key:ident.loc.loc_start.pos_cnum ~data:binding
            | Some (Some _) -> Hash_set.add bound ident.loc.loc_start.pos_cnum
            | _ -> ())
        | _ -> ()
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  (positions, uses, bound)

(* What running a deferred skip value does: the skips its run code applies, outside the branches
   nested in it, in source order, and how many evaluating claims it makes anywhere. *)
type skip_effect = { run_skips : (expression * int option) list; run_evaluating : int }
(** Each skip with the [lazy] it runs through, by its binding, where it does: a lazy forced again in
    the same branch -- directly or through a wrapper -- runs nothing the second time. *)

(* A named binding of the source whose value runs claims later ({!skip_list_census}). *)
type deferred_skips =
  | Skip_function of {
      positional : int;
      labelled : string list;
      runs : skip_effect;
      callback_skips : expression list;
      defaults : (string * skip_effect) list;
          (** What each optional parameter's default runs, by its label: run before the body by an
              application omitting the label. *)
    }
      (** A function: its positional and required labelled parameters (a [function]'s implicit one
          among them), which an application must supply to run it; what an application runs ([runs],
          its nested branches judged where they are written); and the skips it runs as a collection
          function's callback -- its whole body, branches included, as an anonymous callback's is
          read. *)
  | Skip_lazy of { key : int; runs : skip_effect }
      (** A [lazy], by its binding: memoised, so forcing it twice runs it once. *)

let skip_list_census content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  let skips, uses, source_bound = skip_positions ~owners ast in
  let is_skip_head expr =
    match (strip_constraints expr).pexp_desc with
    | Pexp_ident { loc; _ } -> Hash_set.mem skips loc.loc_start.pos_cnum
    | _ -> false
  in
  let is_evaluating_head expr =
    match Verdict_callee.resolve owners expr with
    | Some { member; ownership = Verdicts | Ambiguous | Free; _ } ->
        not (List.mem skip_list_unread_members member ~equal:String.equal)
    | _ -> false
  in
  let label_of expr =
    match expr.pexp_desc with
    | Pexp_apply (_, arguments) -> (
        match
          List.find_map arguments ~f:(function
            | Ppxlib.Asttypes.Nolabel, argument -> Some argument
            | _ -> None)
        with
        | Some label -> (
            match Read.string_literal (strip_constraints label) with
            | Some literal -> literal
            | None -> Ppxlib.Pprintast.string_of_expression label)
        | None -> Ppxlib.Pprintast.string_of_expression expr)
    | _ -> Ppxlib.Pprintast.string_of_expression expr
  in
  (* The named [let] binding an unqualified name resolves to, lexically. *)
  let binding_of expr =
    match (strip_constraints expr).pexp_desc with
    | Pexp_ident { txt = Ppxlib.Longident.Lident _; loc } ->
        Hashtbl.find uses loc.loc_start.pos_cnum
    | _ -> None
  in
  let binding_key binding = binding.pvb_loc.loc_start.pos_cnum in
  (* [labels |> List.iter ~f] and [List.iter ~f @@ labels] are the application they spell. *)
  let normalize callee arguments =
    match (Read.longident_of callee, arguments) with
    | Some [ "|>" ], [ (Ppxlib.Asttypes.Nolabel, x); (Ppxlib.Asttypes.Nolabel, f) ]
    | Some [ "@@" ], [ (Ppxlib.Asttypes.Nolabel, f); (Ppxlib.Asttypes.Nolabel, x) ] -> (
        match (strip_constraints f).pexp_desc with
        | Pexp_apply (inner, inner_arguments) ->
            (inner, inner_arguments @ [ (Ppxlib.Asttypes.Nolabel, x) ])
        | _ -> (f, [ (Ppxlib.Asttypes.Nolabel, x) ]))
    | _ -> (callee, arguments)
  in
  (* [Lazy.force], [Lazy.force_val], or [Base]'s unqualified [force] where the source binds none. *)
  (* The modules the source binds, by name: an alias to a path ([module L = List]) by the path's
     last component, a structure of its own as [None]. By name across the source, not by scope: a
     name bound both ways reads as the alias, the loud direction. *)
  let module_bindings = Hashtbl.create (module String) in
  let bind_module name (module_expr : module_expr) =
    let rec peel me =
      match me.pmod_desc with Pmod_constraint (inner, _) -> peel inner | _ -> me
    in
    match ((peel module_expr).pmod_desc, Hashtbl.find module_bindings name) with
    | Pmod_ident { txt; _ }, _ ->
        Hashtbl.set module_bindings ~key:name ~data:(Some (Ppxlib.Longident.last_exn txt))
    | _, Some (Some _) -> ()
    | _, _ -> Hashtbl.set module_bindings ~key:name ~data:None
  in
  let module_binder =
    object
      inherit Ast_traverse.iter as super

      method! module_binding binding =
        Option.iter binding.pmb_name.txt ~f:(fun name -> bind_module name binding.pmb_expr);
        super#module_binding binding

      method! expression e =
        (match e.pexp_desc with
        | Pexp_letmodule ({ txt = Some name; _ }, module_expr, _) -> bind_module name module_expr
        | _ -> ());
        super#expression e
    end
  in
  module_binder#structure ast;
  let rec resolve_module seen name =
    match Hashtbl.find module_bindings name with
    | Some (Some target) when not (List.mem seen name ~equal:String.equal) ->
        resolve_module (name :: seen) target
    | Some (Some _) -> Some name
    | Some None -> None
    | None -> Some name
  in
  let forces callee =
    match Read.longident_of (strip_constraints callee) with
    | Some [ "force" ] -> (
        (* Unless the source binds a [force] of its own: a [let], a parameter, a pattern. *)
        match (strip_constraints callee).pexp_desc with
        | Pexp_ident { loc; _ } -> not (Hash_set.mem source_bound loc.loc_start.pos_cnum)
        | _ -> false)
    | Some path -> (
        (* [Lazy.force], through the source's module aliases; not a [Lazy] of the source's own. *)
        match List.rev path with
        | ("force" | "force_val") :: module_ :: _ ->
            Option.equal String.equal (resolve_module [] module_) (Some "Lazy")
        | _ -> false)
    | None -> false
  in
  (* Whether an argument supplies the optional parameter [label], so that its default does not run:
     [~label:v], or [?label:e] -- except a literal [None], which runs the default. *)
  let supplies_default label (given, value) =
    match given with
    | Ppxlib.Asttypes.Labelled given -> String.equal given label
    | Optional given -> (
        String.equal given label
        &&
        match (strip_constraints value).pexp_desc with
        | Pexp_construct ({ txt = Ppxlib.Longident.Lident "None"; _ }, None) -> false
        | _ -> true)
    | Nolabel -> false
  in
  let skip_cap = 64 in
  let memo = Hashtbl.create (module Int) and in_progress = Hash_set.create (module Int) in
  (* The skips and evaluating claims [expr] applies, in source order; [~branches:false] stops at the
     branches nested in it and the functions written in it. A complete application of a deferred
     skip function, or a force of a deferred skip lazy, runs what its run code runs: a skip wrapper
     (one that evaluates nothing) its skips, any other its evaluating claims. *)
  let rec claims ~branches expr =
    let root = expr in
    let found_skips = ref [] and evaluating = ref 0 in
    let forced = Hash_set.create (module Int) in
    let run ?through runs =
      if runs.run_evaluating > 0 then evaluating := !evaluating + runs.run_evaluating
      else
        let fresh =
          List.filter_map runs.run_skips ~f:(fun (skip, key) ->
              let key = Option.first_some key through in
              match key with Some key when Hash_set.mem forced key -> None | _ -> Some (skip, key))
        in
        List.iter fresh ~f:(fun (_, key) -> Option.iter key ~f:(Hash_set.add forced));
        (* At most [skip_cap] skips are kept: a branch needs two to be refused, and a chain of
           wrappers each running the previous one twice would otherwise grow exponentially. *)
        let room = skip_cap - List.length !found_skips in
        if room > 0 then found_skips := List.rev_append (List.take fresh room) !found_skips
    in
    let reader =
      object (self)
        inherit Ast_traverse.iter as super

        method! expression e =
          (* Applied skips only: a skip handed on as a value is an iteration's argument, which the
             walk reads where it is handed. *)
          (match e.pexp_desc with
          | Pexp_apply (callee, arguments)
            when is_skip_head callee
                 && List.exists arguments ~f:(fun (label, _) ->
                     Poly.equal label Ppxlib.Asttypes.Nolabel) ->
              (* A skip handed its label; a partial one is a callback, read where it is handed. *)
              found_skips := (e, None) :: !found_skips
          | Pexp_apply (callee, _) when is_evaluating_head callee -> Int.incr evaluating
          | Pexp_apply (callee, arguments) -> (
              match run_of callee arguments with
              | Some (Skip_function { runs; defaults; _ }) ->
                  List.iter defaults ~f:(fun (label, default) ->
                      if
                        not
                          (List.exists arguments ~f:(fun (given, value) ->
                               supplies_default label (given, value)))
                      then run default);
                  run runs
              | Some (Skip_lazy { key; runs }) -> run ~through:key runs
              | None -> ())
          | _ -> ());
          match e.pexp_desc with
          | Pexp_apply (callee, arguments) when is_skip_head callee ->
              (* The skip's own head is not a second skip. *)
              List.iter arguments ~f:(fun (_, argument) -> self#expression argument)
          | (Pexp_function _ | Pexp_lazy _) when branches && not (phys_equal e root) ->
              (* A function value or lazy inside the callback or loop body is deferred: what it
                 skips is not skipped here. *)
              ()
          | (Pexp_ifthenelse _ | Pexp_match _ | Pexp_function _ | Pexp_lazy _) when not branches
            -> (
              match e.pexp_desc with
              | Pexp_ifthenelse (condition, _, _) -> self#expression condition
              | Pexp_match (scrutinee, _) -> self#expression scrutinee
              | _ -> ())
          | _ -> super#expression e
      end
    in
    reader#expression expr;
    (List.rev !found_skips, !evaluating)
  (* The deferred skip value a named binding holds -- a function or a [lazy], or a renaming of one
     -- read once; a cycle reads as nothing. *)
  and deferred binding =
    let key = binding_key binding in
    match Hashtbl.find memo key with
    | Some known -> known
    | None when Hash_set.mem in_progress key -> None
    | None ->
        Hash_set.add in_progress key;
        let rhs = strip_constraints binding.pvb_expr in
        let result =
          match rhs.pexp_desc with
          | Pexp_function (params, _, body) ->
              let positional =
                List.count params ~f:(fun param ->
                    match param.pparam_desc with
                    | Pparam_val (Ppxlib.Asttypes.Nolabel, _, _) -> true
                    | _ -> false)
                + match body with Pfunction_cases _ -> 1 | Pfunction_body _ -> 0
              in
              let labelled =
                List.filter_map params ~f:(fun param ->
                    match param.pparam_desc with
                    | Pparam_val (Ppxlib.Asttypes.Labelled label, _, _) -> Some label
                    | _ -> None)
              in
              (* A lazy the function allocates itself is a new cell at each application: its
                 identity does not outlive one, unlike a lazy the function captures. *)
              let start, stop = (rhs.pexp_loc.loc_start.pos_cnum, rhs.pexp_loc.loc_end.pos_cnum) in
              let run_skips =
                (match body with
                  | Pfunction_body body -> fst (claims ~branches:false body)
                  | Pfunction_cases _ -> [])
                |> List.map ~f:(fun (skip, key) ->
                    match key with
                    | Some key when start <= key && key < stop -> (skip, None)
                    | _ -> (skip, key))
              in
              (* What the body runs, branches and a [function]'s cases included, apart from the
                 defaults, which run only where an application omits their label. *)
              let whole, run_evaluating =
                let bodies =
                  match body with
                  | Pfunction_body body -> [ body ]
                  | Pfunction_cases (cases, _, _) ->
                      List.concat_map cases ~f:(fun case ->
                          Option.to_list case.pc_guard @ [ case.pc_rhs ])
                in
                let read = List.map bodies ~f:(claims ~branches:true) in
                (List.concat_map read ~f:fst, List.sum (module Int) read ~f:snd)
              in
              Some
                (Skip_function
                   {
                     positional;
                     labelled;
                     runs = { run_skips; run_evaluating };
                     callback_skips = List.map whole ~f:fst;
                     defaults =
                       List.filter_map params ~f:(fun param ->
                           match param.pparam_desc with
                           | Pparam_val (Ppxlib.Asttypes.Optional label, Some default, _) ->
                               let run_skips, run_evaluating = claims ~branches:false default in
                               Some (label, { run_skips; run_evaluating })
                           | _ -> None);
                   })
          | Pexp_lazy inner ->
              let run_skips = fst (claims ~branches:false inner) in
              let run_evaluating = snd (claims ~branches:true rhs) in
              Some (Skip_lazy { key; runs = { run_skips; run_evaluating } })
          | Pexp_ident _ -> Option.bind (binding_of rhs) ~f:deferred
          | _ -> None
        in
        Hash_set.remove in_progress key;
        Hashtbl.set memo ~key ~data:result;
        result
  (* What an application runs of a deferred skip value: a complete application of a function's name,
     or a force of a lazy's. *)
  and run_of callee arguments =
    if forces callee then
      match arguments with
      | [ (Ppxlib.Asttypes.Nolabel, thunk) ] -> (
          match Option.bind (binding_of thunk) ~f:deferred with
          | Some (Skip_lazy _) as lazy_ -> lazy_
          | _ -> None)
      | _ -> None
    else
      match Option.bind (binding_of callee) ~f:deferred with
      | Some (Skip_function { positional; labelled; _ }) as function_
        when List.count arguments ~f:(fun (label, _) -> Poly.equal label Ppxlib.Asttypes.Nolabel)
             >= positional
             && List.for_all labelled ~f:(fun label ->
                 List.exists arguments ~f:(fun (given, _) ->
                     Poly.equal given (Ppxlib.Asttypes.Labelled label))) ->
          function_
      | _ -> None
  in
  let found = ref [] and skips_read = ref 0 in
  let report kind (at : expression) skips =
    match skips with
    | first :: _ ->
        found :=
          {
            skip_kind = kind;
            skip_line = at.pexp_loc.loc_start.pos_lnum;
            first_label = label_of first;
          }
          :: !found
    | [] -> ()
  in
  let skip_only expr =
    let skips, evaluating = claims ~branches:true expr in
    if evaluating = 0 then List.map skips ~f:fst else []
  in
  (* How many positional arguments a collection function [name] hands its callback at the least: the
     element, and an index ([iteri], [mapi], ...), an accumulator ([fold]) or a second collection's
     element (a pairwise [iter2]) besides. *)
  let callback_arguments name =
    (* A comparator takes two elements. *)
    (if
       List.mem
         [ "sort"; "stable_sort"; "fast_sort"; "sort_uniq"; "merge"; "equal"; "compare" ]
         name ~equal:String.equal
     then 2
     else 1)
    + (if String.is_suffix name ~suffix:"i" then 1 else 0)
    + (if String.is_prefix name ~prefix:"fold" then 1 else 0)
    + if String.is_substring name ~substring:"2" then 1 else 0
  in
  (* The skips a skip wrapper named by [callee] runs as the callback of the collection function
     [name], where it is one -- a function of the source whose run code skips and evaluates nothing
     -- and the call saturates it: [stored], the arguments a partial application of it already
     holds, and what [name] hands it supply every positional and required labelled parameter. *)
  let wrapper_skips name callee stored =
    match Option.bind (binding_of callee) ~f:deferred with
    | Some
        (Skip_function
           { positional; labelled; runs = { run_evaluating = 0; _ }; callback_skips; defaults })
      when List.count stored ~f:(fun (label, _) -> Poly.equal label Ppxlib.Asttypes.Nolabel)
           + callback_arguments name
           >= positional
           && List.for_all labelled ~f:(fun label ->
               List.exists stored ~f:(fun (given, _) ->
                   Poly.equal given (Ppxlib.Asttypes.Labelled label))) ->
        (* A default runs where the stored arguments omit its label; one that evaluates a claim then
           makes the callback no skip wrapper. *)
        let omitted =
          List.filter_map defaults ~f:(fun (label, default) ->
              if List.exists stored ~f:(supplies_default label) then None else Some default)
        in
        if List.exists omitted ~f:(fun default -> default.run_evaluating > 0) then []
        else
          List.concat_map omitted ~f:(fun default -> List.map default.run_skips ~f:fst)
          @ callback_skips
    | _ -> []
  in
  let iterated_argument name argument =
    let argument = strip_constraints argument in
    match argument.pexp_desc with
    | Pexp_ident _ when is_skip_head argument -> [ argument ]
    | Pexp_apply (callee, _) when is_skip_head callee -> [ argument ]
    | Pexp_ident _ -> wrapper_skips name argument []
    | Pexp_apply (callee, stored) -> wrapper_skips name callee stored
    | Pexp_function _ -> skip_only argument
    | _ -> []
  in
  (* The module and function name of a collection-module function [callee] spells, where it is one:
     its module component resolved through the source's module bindings ([resolve_module]). *)
  let collection_callee callee =
    match Read.longident_of (strip_constraints callee) with
    | Some (_ :: _ :: _ as path) -> (
        match resolve_module [] (List.nth_exn path (List.length path - 2)) with
        | None -> None
        | Some module_ ->
            if
              List.mem collection_modules module_ ~equal:String.equal
              && not (List.mem [ "Option"; "Result" ] module_ ~equal:String.equal)
            then Some (module_, List.last_exn path)
            else None)
    | _ -> None
  in
  (* The function name of a collection iteration [callee] spells: a lazy sequence's only where it
     consumes the sequence ({!runs_callback}). *)
  let collection_iteration callee =
    match collection_callee callee with
    | Some (family, name) when runs_callback ~family name -> Some name
    | _ -> None
  in
  let fully name arguments =
    let labels = List.map arguments ~f:fst in
    applied_fully ~stdlib:false name labels || applied_fully ~stdlib:true name labels
  in
  (* An application of a name bound to a partial application -- [let skip_all = List.iter
     ~f:skipped] ... [skip_all labels] -- is the application it completes: the stored arguments,
     then the new ones, through every renaming on the way. *)
  let rec completed seen callee arguments =
    match binding_of callee with
    | Some binding when not (List.mem seen (binding_key binding) ~equal:Int.equal) -> (
        let seen = binding_key binding :: seen in
        let rhs = strip_constraints binding.pvb_expr in
        match rhs.pexp_desc with
        | Pexp_ident _ -> completed seen rhs arguments
        | Pexp_apply (inner, stored) -> (
            let inner, stored = normalize inner stored in
            (* A value already, where complete in both libraries' orders; complete in one only
               ([Stdlib.List.iter f] meets [Base]'s one-positional arity), it may still await its
               collection. *)
            let labels = List.map stored ~f:fst in
            match collection_iteration inner with
            | Some name
              when applied_fully ~stdlib:false name labels && applied_fully ~stdlib:true name labels
              ->
                (callee, arguments)
            | _ -> completed seen inner (stored @ arguments))
        | _ -> (callee, arguments))
    | _ -> (callee, arguments)
  in
  (* The skip callbacks a lazy sequence carries until something consumes it: [Seq.map skipped s],
     through the names it is bound to and the sequences it is built from. *)
  (* The positional arguments of the collection function [name] that may be the collection it
     consumes: the first in [Base]'s order, the last in [Stdlib]'s (two each for a pairwise one) --
     never a fold's initial accumulator or another value between them. *)
  let collection_positionals name arguments =
    let positional =
      List.filter_map arguments ~f:(function
        | Ppxlib.Asttypes.Nolabel, argument -> Some argument
        | _ -> None)
    in
    let n = List.length positional in
    let pairwise = String.is_substring name ~substring:"2" in
    List.filteri positional ~f:(fun i _ -> i = 0 || i = n - 1 || (pairwise && (i = 1 || i = n - 2)))
  in
  let sequence_element_constructors =
    [ "return"; "singleton"; "repeat"; "forever"; "make"; "init"; "unfold"; "iterate"; "once" ]
  in
  let rec producer_skips seen expr =
    let expr = strip_constraints expr in
    match expr.pexp_desc with
    | Pexp_ident _ -> (
        match binding_of expr with
        | Some binding when not (List.mem seen (binding_key binding) ~equal:Int.equal) ->
            producer_skips (binding_key binding :: seen) binding.pvb_expr
        | _ -> [])
    | Pexp_apply (callee, arguments) -> (
        let callee, arguments = normalize callee arguments in
        match collection_callee callee with
        | Some (family, name)
          when List.mem sequence_families family ~equal:String.equal
               && not (runs_callback ~family name) ->
            let positional = ref 0 in
            (* The sequences it is built from -- not the element a constructor stores ([Seq.return
               s] holds [s] unconsumed, [Seq.cons x s] holds [x]). *)
            let collections =
              if List.mem sequence_element_constructors name ~equal:String.equal then []
              else if String.equal name "cons" then
                List.drop (collection_positionals name arguments) 1
              else collection_positionals name arguments
            in
            List.concat_map arguments ~f:(fun (label, argument) ->
                let index = !positional in
                (match label with Ppxlib.Asttypes.Nolabel -> Int.incr positional | _ -> ());
                let callback =
                  if
                    calls_back ~stdlib:false name label index
                    || calls_back ~stdlib:true name label index
                  then [ iterated_argument name argument ]
                  else []
                in
                let built =
                  if List.exists collections ~f:(phys_equal argument) then
                    producer_skips seen argument
                  else []
                in
                List.filter (callback @ built) ~f:(Fn.non List.is_empty))
        | _ -> [])
    | _ -> []
  in
  let branch body =
    let skips = List.map (fst (claims ~branches:false body)) ~f:fst in
    if List.length skips >= 2 then report (Skip_run (List.length skips)) body skips
  in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_ident _ when is_skip_head expr -> Int.incr skips_read
        | Pexp_apply (callee, arguments) ->
            let callee, arguments = normalize callee arguments in
            let callee, arguments =
              let completed_callee, completed_arguments = completed [] callee arguments in
              if Option.is_some (collection_iteration completed_callee) then
                (completed_callee, completed_arguments)
              else (callee, arguments)
            in
            Option.iter (collection_iteration callee) ~f:(fun name ->
                (* Only an argument the function calls back is iterated over -- [List.iter]'s [~f],
                   not what [List.singleton] stores -- in either library's order, the loud direction
                   ({!calls_back}). *)
                let positional = ref 0 in
                List.iter arguments ~f:(fun (label, argument) ->
                    let index = !positional in
                    (match label with Ppxlib.Asttypes.Nolabel -> Int.incr positional | _ -> ());
                    let labels = List.map arguments ~f:fst in
                    (* A partial application stores its callback; only a complete one runs it. *)
                    if
                      calls_back ~stdlib:false name label index
                      && applied_fully ~stdlib:false name labels
                      || calls_back ~stdlib:true name label index
                         && applied_fully ~stdlib:true name labels
                    then report Iterated expr (iterated_argument name argument));
                (* A consumer runs the callbacks of the lazy sequence it is handed. *)
                if
                  Option.exists (collection_function name) ~f:(fun f -> f.consumes)
                  && fully name arguments
                then
                  List.iter (collection_positionals name arguments) ~f:(fun argument ->
                      List.iter (producer_skips [] argument) ~f:(report Iterated expr)))
        | Pexp_for (_, _, _, _, body) | Pexp_while (_, body) ->
            report Iterated expr (skip_only body)
        | Pexp_ifthenelse (_, yes, no) ->
            branch yes;
            Option.iter no ~f:branch
        | Pexp_match (_, cases) | Pexp_function (_, _, Pfunction_cases (cases, _, _)) ->
            List.iter cases ~f:(fun case -> branch case.pc_rhs)
        | _ -> ());
        super#expression expr
    end
  in
  walker#structure ast;
  {
    skip_lists = List.sort !found ~compare:(fun a b -> Int.compare a.skip_line b.skip_line);
    skips_read = !skips_read;
  }

(** {1 A gate folded into the claim's value (gh-ocannl-1310)}

    [p "…" ((not on_cpu) || ok)] prints [true] on a GPU run: a pass, byte-identical to a CPU run
    that checked [ok], where the honest report is a skip. The run's own transcript then says the
    claim held on hardware that never evaluated it -- the same invisibility {!Verdict.skipped}
    exists to end, reached by folding the gate into the boolean instead of spelling the skip.
    [Verdict.gated ~when_:on_cpu ~on:backend_name "…" ok] (or a quantified [gated_*] form) is the
    spelling that reports it, and this reader refuses the folded one.

    {2 What is read}

    The VALUE of every [Verdict] claim this source applies directly by its path (under any type
    constraints), whose callee {!Verdict_callee} reads as [Verdict]'s or ambiguous:

    - the boolean of [p], [claim], [pass_fail] and [gated] (the second positional argument), and of
      [pf] and [claimf] (the last positional argument, after the format's);
    - the predicate [~f] of a quantified claim -- [p_all], [p_alli], [p_exists], [p_all2],
      [pf_all2], [pass_fail_all2] and the [gated_*] forms, whose element passes where it returns
      [true], and [p_none], whose element passes where it returns [false] -- when it is a function
      written there or a name bound to one: its body, under its parameters.

    {!folded_slots} is that table; {!folded_unread_members} the members with no boolean to read
    ([fail], [skipped], [case], [p_empty], [p_pairwise_distinct]), and [verdict_ratchet] holds the
    two equal to [Verdict.Claims] together.

    The value is split into the DISJUNCTS whose truth alone passes the claim: through [||] (and,
    under a [not], [&&]), [not] (which swaps the two), a name into its [let] right-hand side, type
    constraints, a [let] body and a sequence's last expression; and through an [if] one of whose
    branches is the passing constant -- [if on_cpu then ok else true] is [(not on_cpu) || ok]. The
    claim is refused when one disjunct is RUN-CONSTANT and another is computed by the test: the gate
    decides the verdict wherever it is closed. For a [gated] claim, a disjunct that makes its own
    [~when_] false -- a conjunct of the gate, negated -- is dropped first: the claim is never
    evaluated where it holds. A disjunct that is a constant -- a literal, a library value -- is
    neither, so a claim over the configuration and constants alone is no fold.

    {2 Run-constant}

    A value fixed by the run's configuration, not computed by the test: what a configuration reader
    returns ({!configuration_readers}: [Utils.get_global_arg], [Utils.get_global_flag],
    [Sys.getenv]/[getenv_opt], [Context.backend_name], [Context.hardware_limits],
    [Context.backend_facts] -- an applied path ending in one of these names, or an unqualified one
    the source does not bind), a value of [Sys] ([Sys.win32], [Sys.os_type], [Sys.big_endian],
    [Sys.argv], ...), the settings record [Utils.settings] (its fields being the configuration), and
    what is computed from those and constants alone: an application of a function from outside the
    source to run-constant and constant arguments, at least one of them run-constant
    ([Context.backend_facts_of_name backend_name], [String.equal backend_name "metal"],
    [not on_gpu]); an [if], [match] or [try] over such parts; a field, a [lazy] or a constructor of
    one, a field of a record the source writes out being that field's value; a name bound to one by
    [let], anywhere in the source, a qualified path into a module of the source ([Gates.on_cpu])
    being that module's binding of the name -- or, for a cell the source assigns into with [:=], to
    a run-constant value in its initializer or in any write, whatever their order. A constant is a
    literal, a constructor of constants, or a library value.

    So [on_cpu], [on_gpu], [is_metal], [seeds_here = Autotune.seeded_default_backend facts] and a
    [lazy] hardware probe are gates; a value computed from the program under test -- a census, a
    readback, a query's answer, a seed list -- is not, though it read the backend along the way,
    because it also reads something that is not run-constant (a parameter, a pattern variable, a
    function value, a call over constants alone such as [Context.auto ()]). The distinction is the
    one [Verdict]'s gates draw: the backend, an advertised capability, the configuration.

    {2 What is not read}

    A claim reached through a wrapper the source binds ([let check label ok = p label ok]), applied
    through [|>] or [@@], or partially applied; a function's result where the function is the
    source's own ([has_mma ()] is not run-constant, whatever its body reads); a name a destructuring
    [let] or a [match] case binds; and a gate folded below a conjunction
    ([census_ok && ((not on_metal) || dialect_ok)]), which the claim checks partly everywhere -- the
    label then promises both halves, and only the reader of the label can tell that the second is
    vacuous off Metal. Each is a place a folded gate can hide; none is one this reader reports where
    the fold is not there.

    The other direction has three known places, each loud -- a refusal of a claim that does not
    fold: a cell's writes are not ordered against its reads (above); a configuration reader is
    run-constant whatever it is handed ([Context.hardware_limits ctx] of an element's own [ctx]);
    and [Sys], [Utils.settings] and the reader names are recognized by spelling, so a module the
    source binds under one of those names is read as the standard one, unless the path names a
    binding of that module of the source. *)

type run_constancy =
  | Constant  (** A literal, a constructor of constants, a library value. *)
  | Run_constant  (** Fixed by the run's configuration (see above). *)
  | Varying  (** Computed by the test. *)

type folded_gate = {
  folded_form : string;  (** The claim's callee, as written. *)
  folded_ambiguous : bool;
      (** The claim reaches [Verdict] only past an [open] of a module {!Verdict_callee} cannot read.
      *)
  folded_line : int;
  when_ : string;
      (** The gate a [~when_] would carry: the negation of the run-constant disjunct, printed. *)
}

type folded_hatch = {
  hatch_form : string;  (** The claim's callee, as written. *)
  hatch_ambiguous : bool;
  hatch_line : int;
  hatch : string;  (** The hatch disjunct, printed: what [~unless] would name. *)
}
(** A run-time hatch folded into a claim's value (gh-ocannl-996): see {!hatch_fields}. *)

type folded_census = {
  folded : folded_gate list;  (** In source order. *)
  folded_hatches : folded_hatch list;  (** In source order. *)
  folded_claim_sites : int;  (** Every claim whose value was read. *)
}

(** {2 A run-time hatch folded into the value (gh-ocannl-996)}

    The same split finds a second fold. [p "…" (reading.contended || within_envelope)] passes
    wherever the instrument reported contention, printing the row a check prints and recording an
    evaluation: a claim bypassed seven runs in eight reads as verified. [Verdict.p_unless ~unless]
    (or a [gated] form given [~unless]) takes the hatch as its own argument and records the bypass.

    What is refused, as a closed list: a claim whose value has a disjunct, at the claim's passing
    polarity, that is a field projection or a bare name spelled with one of {!hatch_fields} -- the
    run-time verdicts the timing instrument reports about its own window ([Autotune.timing_result]'s
    [contended], [dispersed] and [unbatched]) -- beside another disjunct the test computed. What is
    not: any other spelling of a hatch (a [match] on a status, a helper returning one, a hatch below
    a conjunction), a hatch that is the claim's whole value (a claim ABOUT contention, which checks
    it), and a negated one ([(not r.contended) || b] checks [b] exactly when the window was
    contended, which is no bypass). A new hatch vocabulary joins the list in the PR that introduces
    it. *)
let hatch_fields = [ "contended"; "dispersed"; "unbatched" ]

let configuration_readers =
  [
    "get_global_arg";
    "get_global_flag";
    "getenv";
    "getenv_opt";
    (* What [Context] reports of the device and backend a context runs on. *)
    "backend_name";
    "hardware_limits";
    "codegen_capabilities";
    "backend_facts";
    "static_properties";
    "timing_identity";
  ]

type value_slot =
  | Positional of int  (** The [n]th positional argument, from zero. *)
  | Last_positional  (** After a format's arguments. *)
  | Predicate  (** The [~f] function's body. *)

(** Each [Verdict.Claims] member whose value this reader reads, where it reads it, and the polarity
    in which its value passes: [true] for every member but [p_none], whose element passes where its
    predicate is [false]. *)
let folded_slots =
  [
    ("p", Positional 1, true);
    ("claim", Positional 1, true);
    ("p_unless", Positional 1, true);
    ("pass_fail", Positional 1, true);
    ("gated", Positional 1, true);
    ("pf", Last_positional, true);
    ("claimf", Last_positional, true);
    ("p_all", Predicate, true);
    ("p_alli", Predicate, true);
    ("p_exists", Predicate, true);
    ("p_all2", Predicate, true);
    ("pf_all2", Predicate, true);
    ("pass_fail_all2", Predicate, true);
    ("gated_all", Predicate, true);
    ("gated_alli", Predicate, true);
    ("gated_exists", Predicate, true);
    ("p_none", Predicate, false);
  ]

(** The members with no boolean of the claim's own to read: a message, a skip, a row boundary, an
    emptiness census, a distinctness check, the float-parity forms, whose [~tol] answers a distance
    rather than a verdict, and the leg claims, whose value is the claim their [make] function makes
    -- an ordinary claim this reader reads where it is written. *)
let folded_unread_members =
  [
    "fail";
    "skipped";
    "case";
    "p_empty";
    "p_pairwise_distinct";
    "p_close";
    "pf_close";
    "pass_fail_close";
    "in_leg";
    "in_legf";
  ]

let folded_gate_census content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  let folded = ref [] and hatches = ref [] and sites = ref 0 in
  let hatch_leaf leaf =
    match (strip_constraints leaf).pexp_desc with
    | Pexp_field (_, { txt = Ppxlib.Longident.Lident name | Ldot (_, name); _ })
    | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } ->
        List.mem hatch_fields name ~equal:String.equal
    | _ -> false
  in
  let constancy = Hashtbl.create (module Int) in
  (* What the source assigns into a binding with [:=], by the binding's identity: a ref's value is
     its initializer's and every write's. Claims are judged once the walk is over, when every write
     is known. *)
  let writes = Hashtbl.create (module Int) in
  let pending = ref [] in
  (* The value bindings of the source's own modules, by qualified name ([Gates.on_cpu]), the last
     binding of a name winning: a qualified path into a module of this source is that binding. By
     name, not by module identity -- an alias of the module ([module G = Gates]) is not followed. *)
  let members = Hashtbl.create (module String) in
  let modules = ref [] in
  let path_of expr =
    match (strip_constraints expr).pexp_desc with
    | Pexp_ident { txt; _ } -> Option.try_with (fun () -> Read.flatten_longident txt)
    | _ -> None
  in
  let is_function expr =
    match (strip_constraints expr).pexp_desc with Pexp_function _ -> true | _ -> false
  in
  let rec irrefutable pattern =
    match pattern.ppat_desc with
    | Ppat_any | Ppat_var _ -> true
    | Ppat_alias (inner, _) | Ppat_constraint (inner, _) -> irrefutable inner
    | Ppat_tuple parts -> List.for_all parts ~f:irrefutable
    | _ -> false
  in
  let is_bool expr value =
    match (strip_constraints expr).pexp_desc with
    | Pexp_construct ({ txt = Ppxlib.Longident.Lident name; _ }, None) ->
        String.equal name (if value then "true" else "false")
    | _ -> false
  in
  let walker =
    object (self)
      inherit [gate_denotation, gate_module] Lexical_scope.scoped as super
      val mutable next_local = 0

      method local =
        next_local <- next_local - 1;
        { identity = next_local; definition = None }

      (* Modules matter here only to {!Verdict_callee}, which resolves the claim. *)
      method module_path _ _ = None

      method! binding_denotes env binding =
        let identity = binding.pvb_loc.loc_start.pos_cnum + 1 in
        match Read.pattern_name binding.pvb_pat with
        | Some _ -> { identity; definition = Some (binding.pvb_expr, env) }
        | None -> { identity; definition = None }

      method! attribute _ attr = attr

      method! module_binding env binding =
        match binding.pmb_name.txt with
        | Some name ->
            modules := name :: !modules;
            Exn.protect
              ~f:(fun () -> super#module_binding env binding)
              ~finally:(fun () -> modules := List.tl_exn !modules)
        | None -> super#module_binding env binding

      method! define ~top env rec_flag bindings ~walk =
        let denotes = super#define ~top env rec_flag bindings ~walk in
        (if not (List.is_empty !modules) then
           let prefix = String.concat ~sep:"." (List.rev !modules) in
           List.iter2_exn bindings denotes ~f:(fun binding denotation ->
               Option.iter (Read.pattern_name binding.pvb_pat) ~f:(fun name ->
                   Hashtbl.set members ~key:(prefix ^ "." ^ name) ~data:denotation)));
        denotes

      (* The binding a qualified path names in a module of this source. *)
      method source_member expr =
        Option.bind (path_of expr) ~f:(fun path ->
            Hashtbl.find members (String.concat ~sep:"." path))

      (* The path of a callee from outside this source: qualified, or a name nothing here binds. *)
      method external_path env callee =
        match path_of callee with
        | Some [ name ] when Option.is_some (Lexical_scope.lookup env name) -> None
        | path -> path

      (* A standard connective the source does not rebind: bare, or qualified by a standard
         library. *)
      method connective env callee names =
        match self#external_path env callee with
        | Some [ name ] -> List.mem names name ~equal:String.equal
        | Some [ library; name ] ->
            List.mem Verdict_callee.standard_libraries library ~equal:String.equal
            && List.mem names name ~equal:String.equal
        | _ -> false

      (* A binding's constancy, by its identity, wherever it is reached from: its right-hand side's,
         and for a cell the source assigns into, its writes' too. *)
      method bound_constancy identity rhs rhs_env =
        match Hashtbl.find constancy identity with
        | Some known -> known
        | None ->
            (* A cycle is the test's own computation. *)
            Hashtbl.set constancy ~key:identity ~data:Varying;
            let known =
              match Hashtbl.find_multi writes identity with
              | [] -> self#constancy rhs_env rhs
              | written ->
                  (* A cell the source assigns into holds, somewhere, each value it is given: one
                     run-constant value makes it a gate wherever it is read -- the order of reads
                     and writes is not followed, in the loud direction. *)
                  let held =
                    self#constancy rhs_env rhs
                    :: List.map written ~f:(fun (value, env) -> self#constancy env value)
                  in
                  if List.mem held Run_constant ~equal:Poly.equal then Run_constant
                  else
                    List.fold held ~init:Constant ~f:(fun acc c ->
                        match (acc, c) with
                        | Varying, _ | _, Varying -> Varying
                        | Run_constant, _ | _, Run_constant -> Run_constant
                        | Constant, Constant -> Constant)
            in
            Hashtbl.set constancy ~key:identity ~data:known;
            known

      method constancy env expr =
        let join parts =
          if List.mem parts Varying ~equal:Poly.equal then Varying
          else if List.mem parts Run_constant ~equal:Poly.equal then Run_constant
          else Constant
        in
        match expr.pexp_desc with
        | Pexp_constant _ | Pexp_construct (_, None) | Pexp_variant (_, None) -> Constant
        | Pexp_construct (_, Some inner)
        | Pexp_variant (_, Some inner)
        | Pexp_lazy inner
        | Pexp_constraint (inner, _)
        | Pexp_coerce (inner, _, _)
        | Pexp_sequence (_, inner)
        | Pexp_open (_, inner)
        | Pexp_newtype (_, inner) ->
            self#constancy env inner
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some { identity; definition = Some (rhs, rhs_env) } ->
                self#bound_constancy identity rhs rhs_env
            | Some _ -> Varying
            | None -> Constant)
        | Pexp_field (record, { txt = field; _ }) -> (
            (* A field of a record the source writes out is that field's value, not the whole
               record's. *)
            let field = Ppxlib.Longident.last_exn field in
            let rec literal env expr =
              match (strip_constraints expr).pexp_desc with
              | Pexp_record (fields, _) -> Some (fields, env)
              | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                  match Lexical_scope.lookup env name with
                  | Some { identity; definition = Some (rhs, rhs_env) }
                    when List.is_empty (Hashtbl.find_multi writes identity) ->
                      literal rhs_env rhs
                  | _ -> None)
              | _ -> None
            in
            match literal env record with
            | Some (fields, fields_env) -> (
                match
                  List.find fields ~f:(fun ({ txt; _ }, _) ->
                      String.equal (Ppxlib.Longident.last_exn txt) field)
                with
                | Some (_, value) -> self#constancy fields_env value
                | None -> self#constancy env record)
            | None -> self#constancy env record)
        | Pexp_try (body, cases) ->
            join
              (self#constancy env body
              :: List.map cases ~f:(fun case ->
                  self#constancy (self#bind_patterns env [ case.pc_lhs ]) case.pc_rhs))
        | Pexp_ident _ when Option.is_some (self#source_member expr) -> (
            match self#source_member expr with
            | Some { identity; definition = Some (rhs, rhs_env) } ->
                self#bound_constancy identity rhs rhs_env
            | _ -> Varying)
        | Pexp_ident _ -> (
            match Option.map (path_of expr) ~f:List.rev with
            | Some (_ :: "Sys" :: _) -> Run_constant
            | Some ("settings" :: "Utils" :: _) -> Run_constant
            | _ -> Constant)
        | Pexp_apply (callee, arguments) -> (
            match self#external_path env callee with
            | Some path when List.mem configuration_readers (List.last_exn path) ~equal:String.equal
              ->
                Run_constant
            | Some _ -> (
                (* A call over constants alone is a computation -- [Context.auto ()] -- not a value
                   the configuration fixes. *)
                match join (List.map arguments ~f:(fun (_, a) -> self#constancy env a)) with
                | Constant -> Varying
                | known -> known)
            | None -> Varying)
        | Pexp_ifthenelse (condition, yes, no) ->
            join (List.map (condition :: yes :: Option.to_list no) ~f:(self#constancy env))
        | Pexp_match (scrutinee, cases) ->
            (* Cases whose patterns all match anything select by their guards alone. *)
            let selects = not (List.for_all cases ~f:(fun case -> irrefutable case.pc_lhs)) in
            join
              ((if selects then self#constancy env scrutinee else Constant)
              :: List.map cases ~f:(fun case ->
                  let env = self#bind_patterns env [ case.pc_lhs ] in
                  join
                    (List.map (case.pc_rhs :: Option.to_list case.pc_guard) ~f:(self#constancy env)))
              )
        | Pexp_tuple parts | Pexp_array parts -> join (List.map parts ~f:(self#constancy env))
        | Pexp_record (fields, base) ->
            join (List.map (List.map fields ~f:snd @ Option.to_list base) ~f:(self#constancy env))
        | Pexp_let (rec_flag, bindings, body) ->
            self#constancy
              (self#bind_group env bindings (self#denotations env rec_flag bindings))
              body
        | _ -> Varying

      (* The disjuncts whose truth alone makes [expr] -- or, at [polarity = false], its negation --
         true: each a leaf, its scope, and whether the disjunct is the leaf itself or its
         negation. *)
      method disjuncts env polarity expr ~seen =
        let again env polarity expr = self#disjuncts env polarity expr ~seen in
        let leaf = [ (expr, env, polarity) ] in
        match expr.pexp_desc with
        | Pexp_constraint (inner, _)
        | Pexp_coerce (inner, _, _)
        | Pexp_sequence (_, inner)
        | Pexp_open (_, inner)
        | Pexp_newtype (_, inner) ->
            again env polarity inner
        | Pexp_let (rec_flag, bindings, body) ->
            again
              (self#bind_group env bindings (self#denotations env rec_flag bindings))
              polarity body
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some { identity; definition = Some (rhs, rhs_env) }
              when not (Hash_set.mem seen identity) -> (
                Hash_set.add seen identity;
                (* A name that is one disjunct, or a gate as a whole, is reported by its name, which
                   is what a remedy's [~when_] would spell. *)
                if Poly.equal (self#constancy env expr) Run_constant then leaf
                else
                  match again rhs_env polarity rhs with
                  | [ (_, _, same) ] when Bool.equal same polarity -> leaf
                  | parts -> parts)
            | _ -> leaf)
        | Pexp_apply (callee, [ (Nolabel, a); (Nolabel, b) ])
          when (polarity && self#connective env callee [ "||"; "or" ])
               || ((not polarity) && self#connective env callee [ "&&"; "&" ]) ->
            again env polarity a @ again env polarity b
        | Pexp_apply (callee, [ (Nolabel, a) ]) when self#connective env callee [ "not" ] ->
            again env (not polarity) a
        | Pexp_ifthenelse (condition, yes, Some no) when is_bool no polarity ->
            (* [if c then x else true]: [not c], or [x]. *)
            (condition, env, false) :: again env polarity yes
        | Pexp_ifthenelse (condition, yes, Some no) when is_bool yes polarity ->
            (condition, env, true) :: again env polarity no
        | Pexp_try (body, cases) ->
            (* The protected value, or a handler's: either can be what passes. *)
            again env polarity body
            @ List.concat_map cases ~f:(fun case ->
                again (self#bind_patterns env [ case.pc_lhs ]) polarity case.pc_rhs)
        | Pexp_match (scrutinee, cases)
          when List.exists cases ~f:(fun case -> is_bool case.pc_rhs polarity) ->
            (* [match s with A -> x | _ -> true]: [s] reaching a passing case, or a computed branch.
               The first disjunct is the match with each case returning whether it passes, which is
               as run-constant as [s] and the guards are. *)
            let passes case =
              let value = is_bool case.pc_rhs polarity in
              {
                case with
                pc_rhs =
                  {
                    case.pc_rhs with
                    pexp_desc =
                      Pexp_construct
                        ( {
                            txt = Ppxlib.Longident.Lident (if value then "true" else "false");
                            loc = case.pc_rhs.pexp_loc;
                          },
                          None );
                  };
              }
            in
            ({ expr with pexp_desc = Pexp_match (scrutinee, List.map cases ~f:passes) }, env, true)
            :: List.concat_map cases ~f:(fun case ->
                if is_bool case.pc_rhs polarity then []
                else again (self#bind_patterns env [ case.pc_lhs ]) polarity case.pc_rhs)
        | _ -> leaf

      (* The body a predicate returns, under its parameters: a function written in place, or one a
         name is bound to. *)
      method bodies env expr ~seen =
        match (strip_constraints expr).pexp_desc with
        | Pexp_function (params, _, body) -> (
            let env =
              List.fold params ~init:env ~f:(fun env param ->
                  match param.pparam_desc with
                  | Pparam_val (label, _, pattern) -> self#bind_parameter env label pattern
                  | Pparam_newtype _ -> env)
            in
            match body with
            | Pfunction_body body when is_function body -> self#bodies env body ~seen
            | Pfunction_body body -> [ (body, env) ]
            | Pfunction_cases (cases, _, _) ->
                (* [function p when g -> x | ...] is a match over the element, guards and all. *)
                let element = "element" in
                [
                  ( {
                      expr with
                      pexp_desc =
                        Pexp_match
                          ( {
                              expr with
                              pexp_desc =
                                Pexp_ident
                                  { txt = Ppxlib.Longident.Lident element; loc = expr.pexp_loc };
                            },
                            cases );
                    },
                    Lexical_scope.bind_values env [ element ] self#local );
                ])
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some { identity; definition = Some (rhs, rhs_env) }
              when not (Hash_set.mem seen identity) ->
                Hash_set.add seen identity;
                self#bodies rhs_env rhs ~seen
            | _ -> [])
        | _ -> []

      method claim_values env slot arguments =
        let positional =
          List.filter_map arguments ~f:(function
            | Ppxlib.Asttypes.Nolabel, argument -> Some argument
            | _ -> None)
        in
        match slot with
        | Positional n -> List.map (Option.to_list (List.nth positional n)) ~f:(fun v -> (v, env))
        | Last_positional ->
            if List.length positional >= 2 then [ (List.last_exn positional, env) ] else []
        | Predicate ->
            List.concat_map arguments ~f:(function
              | (Ppxlib.Asttypes.Labelled "f" | Optional "f"), predicate ->
                  self#bodies env predicate ~seen:(Hash_set.create (module Int))
              | _ -> [])

      (* Whether two disjunct leaves are the same expression: one binding, where both are names. *)
      method same_leaf (a, a_env) (b, b_env) =
        let print e = Ppxlib.Pprintast.string_of_expression e in
        match (a.pexp_desc, b.pexp_desc) with
        | ( Pexp_ident { txt = Ppxlib.Longident.Lident x; _ },
            Pexp_ident { txt = Ppxlib.Longident.Lident y; _ } ) -> (
            String.equal x y
            &&
            match (Lexical_scope.lookup a_env x, Lexical_scope.lookup b_env y) with
            | Some { identity = i; _ }, Some { identity = j; _ } -> i = j
            | None, None -> true
            | _ -> false)
        | _ -> String.equal (print a) (print b)

      method! expression env expr =
        (match expr.pexp_desc with
        | Pexp_apply
            ( callee,
              [
                (Nolabel, { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }); (Nolabel, written);
              ] )
          when self#connective env callee [ ":=" ] -> (
            (* Through any [let alias = cell] renaming: a write through an alias is a write into the
               cell. *)
            let rec cell env name =
              match Lexical_scope.lookup env name with
              | Some
                  {
                    definition =
                      Some
                        ( { pexp_desc = Pexp_ident { txt = Ppxlib.Longident.Lident aliased; _ }; _ },
                          env );
                    _;
                  } ->
                  cell env aliased
              | Some { identity; _ } -> Some identity
              | None -> None
            in
            match cell env name with
            | Some identity -> Hashtbl.add_multi writes ~key:identity ~data:(written, env)
            | None -> ())
        | _ -> ());
        (match expr.pexp_desc with
        | Pexp_apply (callee, arguments) -> (
            match Verdict_callee.resolve owners callee with
            | Some { member; ownership = (Verdicts | Ambiguous) as ownership; spelled } -> (
                match List.find folded_slots ~f:(fun (name, _, _) -> String.equal name member) with
                | Some (_, slot, polarity) ->
                    Int.incr sites;
                    (* A gated claim is evaluated only where its [~when_] holds, so a disjunct that
                       makes the gate false -- a copy of the gate's own negation -- is never the one
                       that passes it. *)
                    let unreachable () =
                      List.concat_map arguments ~f:(function
                        | Ppxlib.Asttypes.Labelled "when_", gate ->
                            self#disjuncts env false gate ~seen:(Hash_set.create (module Int))
                        | _ -> [])
                    in
                    let values = self#claim_values env slot arguments in
                    pending :=
                      (fun () ->
                        let unreachable = unreachable () in
                        List.iter values ~f:(fun (value, env) ->
                            let parts =
                              List.filter_map
                                (self#disjuncts env polarity value
                                   ~seen:(Hash_set.create (module Int)))
                                ~f:(fun (leaf, env, positive) ->
                                  if
                                    List.exists unreachable
                                      ~f:(fun (gate, gate_env, gate_positive) ->
                                        Bool.equal positive gate_positive
                                        && self#same_leaf (leaf, env) (gate, gate_env))
                                  then None
                                  else Some (leaf, positive, self#constancy env leaf))
                            in
                            let printed leaf =
                              Ppxlib.Pprintast.string_of_expression leaf
                              |> String.split_on_chars ~on:[ '\n'; ' ' ]
                              |> List.filter ~f:(fun word -> not (String.is_empty word))
                              |> String.concat ~sep:" "
                            in
                            let is_hatch (leaf, positive, _) = positive && hatch_leaf leaf in
                            (match List.find parts ~f:is_hatch with
                            | Some (leaf, _, _)
                              when List.exists parts ~f:(fun ((_, _, c) as part) ->
                                       Poly.equal c Varying && not (is_hatch part)) ->
                                hatches :=
                                  {
                                    hatch_form = spelled;
                                    hatch_ambiguous = Poly.equal ownership Ambiguous;
                                    hatch_line = expr.pexp_loc.loc_start.pos_lnum;
                                    hatch = printed leaf;
                                  }
                                  :: !hatches
                            | _ -> ());
                            let gate =
                              List.find parts ~f:(fun (_, _, c) -> Poly.equal c Run_constant)
                            in
                            match gate with
                            | Some (leaf, positive, _)
                              when List.exists parts ~f:(fun (_, _, c) -> Poly.equal c Varying) ->
                                let printed = printed leaf in
                                folded :=
                                  {
                                    folded_form = spelled;
                                    folded_ambiguous = Poly.equal ownership Ambiguous;
                                    folded_line = expr.pexp_loc.loc_start.pos_lnum;
                                    when_ = (if positive then "not (" ^ printed ^ ")" else printed);
                                  }
                                  :: !folded
                            | _ -> ()))
                      :: !pending
                | None -> ())
            | _ -> ())
        | _ -> ());
        super#expression env expr
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  List.iter (List.rev !pending) ~f:(fun judge -> judge ());
  { folded = List.rev !folded; folded_hatches = List.rev !hatches; folded_claim_sites = !sites }
