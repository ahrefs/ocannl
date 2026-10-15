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
    either an assertion or, by exemption, a documented exception. *)

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
  literals : int;  (** Every string literal in expression position. *)
  applied_literals : int;  (** How many of those are an argument of a named function. *)
}
(** [literals] and [applied_literals] are not verdicts; they are what a blind walk cannot produce. A
    walk whose expression hook stopped firing reports zero of both over a corpus where thousands are
    expected, so the consuming test can say that its empty offender list came from a scan that read
    something — rather than leaving "no offenders" and "no reading" the same result. *)

(** [scan content] reads one source. Raises if [content] does not parse: a reader that cannot read
    its input must say so rather than report an empty census. *)
let scan content =
  let ast = Read.structure_of content in
  (* The function each literal is an argument of, noted as the application is entered — an iterator
     visits a node before its children, so the argument's own visit below finds it. *)
  let printers = Hashtbl.create (module Int) in
  let sites = ref [] and literals = ref 0 in
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
  { sites = List.rev !sites; literals = !literals; applied_literals = Hashtbl.length printers }

(** {1 A gated claim reported in two dialects (gh-ocannl-997)}

    [Verdict.skipped] prints [<label>: true] — {!Verdict.p}'s line — so that a leg one host cannot
    evaluate leaves the golden exactly as a host that evaluated it does. A claim whose evaluated
    branch reports through [Verdict.pass_fail] instead prints [<label>: PASS] there, and the golden
    promoted on a machine that evaluates it breaks on the first machine that skips: silently, until
    the gate actually fires, on whichever host is least likely to be the one promoting.
    [Verdict.gated] removes the trap by picking the dialect itself; this reader keeps the mixed
    spelling from regrowing.

    What it pairs is a LABEL that reaches both a [pass_fail]/[pass_fail_all2] call and a [skipped]
    call in one source. The label is the first positional argument, keyed three ways: a string
    literal by its value; a name by the literal it is bound to, in its lexical scope through
    [Lexical_scope] (the [let claim = "…"] shared by the two branches is the shape both sightings
    had), otherwise by the name itself; anything else by its printed expression, so a label computed
    the same way on both sides is still one label. A literal pairs across the whole source; a name
    or an expression pairs only within one top-level item, since equal names in two functions need
    not be one label. Named and computed keys retain their bindings' identities. The callees are
    matched by their last path component, which is how an opened [Verdict.Claims] and the per-test
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
  pass_fail_sites : int;  (** Every [pass_fail]/[pass_fail_all2] call whose label was read. *)
}
(** The site counts are what a blind walk cannot produce, as {!scan}'s literal counts are for the
    claim-shape scan: they let the consuming test tell "no pairing" from "nothing read". *)

let pass_fail_callees = [ "pass_fail"; "pass_fail_all2" ]
let skipped_callees = [ "skipped" ]

(* What an [open] binds each claim name to: an identity no [let] and no local can carry ([let]s
   count up from 1, locals down from -1) -- one for a module denoting [Verdict], one for any
   other. *)
let opened_from_verdict = Int.min_value
let opened_elsewhere = Int.min_value + 1

(** Whose a claim callee is: [Verdict]'s, not [Verdict]'s, or [Ambiguous] -- an unqualified name
    that reaches an [open Verdict.Claims] only past a later [open] of another module, which may or
    may not export a function of that name: this scan cannot read what an unmodelled module exports.
*)
type ownership = Verdicts | Ambiguous | Not_verdicts

(* One application whose callee is a path ending in a name of [callees], with a positional argument:
   its last path component, whether that callee is [Verdict]'s (below), and the site its first
   positional argument -- the label -- keys. Shared by {!dialect_census} and {!line_break_census},
   so both key a label one way.

   Whether the callee is [Verdict]'s is resolved through {!Lexical_scope}, never by spelling. A
   qualified callee is when its module path denotes [Verdict] in scope: [Verdict] unless a [module
   Verdict = ...] took the name over, an alias of it ([module V = Verdict]), or a submodule of
   either ([Verdict.Claims]); [Other.Verdict] is not. An unqualified one is when the name it reaches
   was brought in by an [open] (or [include]) of such a module, as [open Verdict.Claims] brings the
   claim names in; a name the source binds itself ([let p = Stdio.printf], [let skipped =
   Verdict.skipped ~backend]), or one no modelled open brings in, is not; and one that reaches such
   an open only past a later [open] of another module is {!Ambiguous}. *)
let claim_calls ~callees content =
  let ast = Read.structure_of content in
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

      (* A module path denotes [true] when its head is [Verdict] with no binding of that name in
         scope, or a module name bound to something denoting [Verdict]; a name a binding the scan
         cannot see into took over ({!shadowed}) denotes [false]. *)
      method module_path env path =
        let head = Lexical_scope.head path in
        match Map.find env.Lexical_scope.modules head with
        | Some denotes -> Some denotes
        | None -> Some (String.equal head "Verdict")

      method! shadowed = Some false

      (* Every open pushes a frame for the claim names: Verdict's for a module denoting [Verdict],
         and for any other module a marker that the name MAY have been rebound there, since this
         scan cannot read what an unmodelled module exports. {!ownership_of} reads past the
         markers. *)
      method! opened ~top:_ ~include_:_ env denotes =
        let identity =
          if Option.value denotes ~default:false then opened_from_verdict else opened_elsewhere
        in
        Lexical_scope.push_frame env
          (Map.of_alist_reduce
             (module String)
             (List.map callees ~f:(fun name -> (name, (None, identity))))
             ~f:(fun first _ -> first))

      (* An unqualified claim name's owner, innermost frame first: past any number of opens of other
         modules, the first binding decides -- an open of [Verdict] makes it [Verdict]'s, or
         [Ambiguous] when such an open was passed on the way; anything the source binds, or no
         binding at all, makes it not [Verdict]'s. *)
      method ownership_of env name =
        let rec decide passed_open = function
          | [] -> Not_verdicts
          | (None, identity) :: outer when identity = opened_elsewhere -> decide true outer
          | (None, identity) :: _ when identity = opened_from_verdict ->
              if passed_open then Ambiguous else Verdicts
          | _ :: _ -> Not_verdicts
        in
        decide false
          (List.filter_map env.Lexical_scope.frames ~f:(fun frame -> Map.find frame name))

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
            | Pexp_ident { txt = ident; _ }, Some path, Some label ->
                let name = List.last_exn path in
                if List.mem callees name ~equal:String.equal then
                  let verdicts =
                    match ident with
                    | Ppxlib.Longident.Ldot (prefix, _) ->
                        if Option.value (self#module_path env prefix) ~default:false then Verdicts
                        else Not_verdicts
                    | _ -> self#ownership_of env name
                  in
                  let site =
                    {
                      callee = String.concat ~sep:"." path;
                      key = key_of env label;
                      item = !item;
                      line = expr.pexp_loc.loc_start.pos_lnum;
                    }
                  in
                  calls := (name, verdicts, site) :: !calls
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
    List.filter_map calls ~f:(fun (name, _, site) ->
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

    An unqualified claim name that reaches [open Verdict.Claims] only past a later [open] of another
    module is {!Ambiguous}: that module may export a function of the name, and this scan cannot read
    its exports. Such a site is READ, and a line break in its literal label is refused with the
    remedy of qualifying the call -- loudly, since the alternative, leaving it unread, is how a
    later [open Ll_test] would silently exempt every claim after it. *)

(** The [Verdict.Claims] members whose first positional argument is a claim label (or the format
    rendering one): every member but [fail], whose argument is a multi-line message. *)
let labelled_claims =
  [
    "claim";
    "p";
    "pf";
    "claimf";
    "p_all";
    "p_all2";
    "pf_all2";
    "p_none";
    "p_alli";
    "p_empty";
    "p_exists";
    "p_pairwise_distinct";
    "skipped";
    "gated";
    "gated_all";
    "gated_alli";
    "gated_exists";
    "pass_fail";
    "pass_fail_all2";
    "case";
  ]

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
    |> List.filter_map ~f:(fun (_, ownership, site) ->
        match ownership with
        | Verdicts -> Some { site; ambiguous = false }
        | Ambiguous -> Some { site; ambiguous = true }
        | Not_verdicts -> None)
  in
  let broken =
    List.filter sites ~f:(fun { site; _ } ->
        match site.key with Literal text -> has_line_break text | Named _ | Computed _ -> false)
  in
  { broken; label_sites = List.length sites }

(** {1 A gate that reads the population it quantifies (gh-ocannl-1289)}

    [Verdict.gated_all ~when_ ~on label xs ~f] (and [gated_alli], [gated_exists]) skips its claim
    where [when_] is false. A gate computed from the values under test turns the regression the
    claim exists to catch into a skip: had the population emptied or lost its interesting members,
    the gate closes, the skip prints [p]'s passing line, and the golden stays green. That is the
    gh-ocannl-1115 trap, and it shipped twice in one wave through these very forms — a gate asking
    whether the claim's own [results] were non-empty (staging#1049), and one asking whether the
    [real] seeds [results] was mapped from held a tensorized one (staging#1063). A gate comes from
    an invariant — the backend, an advertised capability, the configuration — never from the
    population.

    {2 What is read}

    Every application of [gated_all], [gated_alli] or [gated_exists] that carries its [~when_] and
    both positional arguments: unqualified where no binding of the source gives the name (an open of
    [Verdict.Claims] is how it arrives), or qualified by a module that denotes [Verdict] -- the name
    itself, unless a [module Verdict = ...] took it over, or an alias of it ([module V = Verdict]).
    A file's own [gated_all] is its own. A partial application bound by [let] to a name, under any
    number of further partial applications, is completed at each application of that name, and
    [x |> f] and [f @@ x] are read as [f x]; so the
    [let check = Verdict.gated_all ~when_ ~on in check label xs] wrapper is one site per use. The
    two sides are then compared by LEXICAL identity, a name plus the binding it resolves to through
    {!Lexical_scope}, so a shadowed name is a different value; a name no binding of the source gives
    (a global, or one an unmodelled [open] brought in) is one identity, the same on both sides:

    - the GATE is every value identifier the [~when_] expression spells, and transitively every one
      spelled by the right-hand side of a [let] it resolves to (function bodies included), or by
      what the source writes into that binding (below), in any position;
    - the POPULATION is the identifier the quantified argument spells, and the collections it is
      BUILT from: following a [let]-bound name into its right-hand side, the branches (not the
      condition) of an [if], the cases of a [match] (the scrutinee only through a name a case
      pattern binds and the case returns), the body of a function in data position (a
      [~default:(fun () -> ...)] thunk), the body and handlers of a [try], the body of a [let],
      [let module], local [open] or sequence, the elements of a list, array or tuple literal or
      constructor payload, the operand of [!] and of [ref] and both operands of [@], and the
      arguments that carry the collection into a function of a standard collection module ([List],
      [Array], [Hashtbl], [Map], [Option], [String], ... — see {!collection_modules}), reached by
      name, through an alias ([module Rows = List]), and not under a name a local module took over
      ([module Map = Custom]); an unqualified [List] is [Stdlib]'s until an [open Base] in scope
      rebinds it. Which arguments carry it is {!collection_arguments}': every positional one, except
      that a [Base]-ordered selector ([find], [nth], [take], [remove], ...) contributes its
      collection alone, a [make]/[create] the value it fills in and not its count, an [init] what
      its callback returns, and a [range] nothing, plus the data-bearing labels [~default], [~init],
      [~data];
    - the body of a binding operator ([let+ x = e in body]), where [x] is a part of [e];
    - and, for a name it reaches, whatever the source writes into that binding, directly or through
      a [let alias = name] renaming: the right-hand side of [x := e], and what a writer of a mutable
      collection module ({!mutable_modules}: [Queue.enqueue x e], [Hashtbl.set x ~key ~data], ... —
      see {!writers}) is handed with [x], where [x] is the first positional argument in [Base]'s
      order and any of them in [Stdlib]'s. [x] is a name the source binds; a container only an
      [open] brought in is not tracked.

    The claim is refused when the two share a binding. The asymmetry is the point: a gate may read
    what the population is COMPUTED from (the site, the hardware limits, the backend steering an
    [if]), which is how a capability judgment over the same program reads, and must not read the
    population or a collection it is mapped, filtered or appended from.

    {2 What is not read}

    The forms above are a closed list: any other expression form ends the population side there, and
    a population reached only through one is not compared. The ones known to matter:

    The population is not followed through a call to any other function (a file-local
    [gpu_seeds pre] stops at the call), through a record field, through a function parameter (the
    caller's argument is not substituted), or through an [open List]-style unqualified collection
    call; a destructured [let] binds names that are followed no further, and a value reached through
    a module path ([M.rows]) is an identity on neither side. A gate threaded into a wrapper as its
    own parameter ([let check ~when_ label xs = gated_all ~when_ ...]) is read at the wrapper's
    definition, where it is a parameter, and not at its callers. [Verdict.gated]'s boolean is not a
    population and is out of scope. Each of these is a place a population gate can hide from this
    reader; none is a place it reports one that is not there.

    A [gated_*] application the reader cannot complete — a partial application no [let] binds, a
    [let]-bound one that no use the reader resolves completes (reached only through a module path,
    say), or a claim (or a [let]-bound wrapper of one) named as a value, handed to [List.iter] say —
    is reported as unread rather than counted as clean, so the boundary is loud there. *)

(** What a module name denotes to this reader. A collection module carries whether its functions
    take arguments in [Stdlib]'s order (the function first, the collection last) rather than
    [Base]'s (the collection first). *)
type gate_module =
  | Verdict_module
  | Collection_module of { stdlib : bool; mutable_ : bool }
      (** [mutable_]: its values are mutable containers ({!mutable_modules}). *)
  | Standard_library of { stdlib : bool }
      (** [Base] or [Core] ([stdlib = false]), [Stdlib] or [Caml] ([true]): an [open] of one rebinds
          the collection modules to its own. *)
  | Local_structure of string list
      (** A [struct ... end] of this source, with the value names it binds: an [open] of it brings
          them into scope as this source's own. *)
  | Other_module

type gate_denotation = {
  identity : int;
  mutable definition : (expression * (gate_denotation, gate_module) Lexical_scope.env) option;
      (** Mutable only so that a recursive group's definitions can see each other: they are set once
          the group's own scope exists. *)
}

type gate_reading = {
  form : string;  (** The callee as written: [gated_all], [Verdict.gated_exists], a wrapper. *)
  line : int;
  gate : string;  (** The [~when_] expression, printed. *)
  reads : string list;  (** The population bindings the gate reads, by name, sorted. *)
}

type gate_census = {
  readings : gate_reading list;  (** Claims whose gate reads their population, in source order. *)
  gated_sites : int;  (** Every completed [gated_*] claim the reader compared. *)
  unread : (string * int) list;  (** [gated_*] applications it could not complete: callee, line. *)
}

let gated_forms = [ "gated_all"; "gated_alli"; "gated_exists" ]

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

(** Which arguments of a collection-module function the population is built from, by the function's
    name and the library's argument order.

    Every positional argument by default: [map], [Map.of_alist_exn (module String) rows], [append],
    [zip], [Set.add s x], [cons] all build the population from whatever they are handed, the
    collection and the elements put into it alike. The exception is a selector in [Base]'s order
    ({!selectors}: [find], [nth], [take], [remove], [update], ...), whose first positional argument
    is the collection and whose later ones are a key, an index or a count: those select the
    population, as an [if] condition steers one, and a gate on them is a gate on what was selected,
    not on what was found. In [Stdlib]'s order the collection may come last ([List.assoc k l],
    [List.mem x l]), so a [Stdlib]-ordered selector takes every positional argument too -- the loud
    direction. A generator's count builds nothing: [make] and [create] build from the value they
    fill in, [init] from what its callback returns (labeled [~f], or positional after the count),
    and [range] from nothing. An inserting function's labeled [~key] joins the population
    ([Map.set m ~key ~data]), and so does what a transforming function's [~f] returns
    ([List.map xs ~f]), or an update-like selector's ([Map.update m k ~f], [find_map]); a selector's
    key and a [find] predicate do not. A writer's [~f] is written into its container.

    The labeled arguments that carry data rather than steer are followed too: [~default] (the
    fallback [Option.value] returns), [~init] (a fold's seed) and [~data] (a stored element). *)
let selectors =
  [
    "find";
    "nth";
    "get";
    "take";
    "drop";
    "sub";
    "slice";
    "split_n";
    "chunks_of";
    "mem";
    "remove";
    "update";
    "change";
  ]

let collection_arguments ~stdlib name arguments =
  let starts prefix = String.is_prefix name ~prefix in
  let positional =
    List.filter_map arguments ~f:(function
      | Ppxlib.Asttypes.Nolabel, argument -> Some argument
      | _ -> None)
  in
  let data =
    List.filter_map arguments ~f:(function
      | (Ppxlib.Asttypes.Labelled label | Optional label), argument
        when List.mem [ "default"; "init"; "data" ] label ~equal:String.equal ->
          Some argument
      | _ -> None)
  in
  if starts "range" then (* [List.range start stop]: bounds build nothing. *) []
  else if starts "init" then
    (* [Array.init n ~f], [Stdlib.Array.init n f]: the elements are what the callback returns
       ({!population} follows a function in data position into its body); the count, the first
       positional argument, builds nothing. *)
    List.drop positional 1
    @ List.filter_map arguments ~f:(function
      | (Ppxlib.Asttypes.Labelled "f" | Optional "f"), callback -> Some callback
      | _ -> None)
    @ data
  else if List.exists [ "make"; "create"; "replicate" ] ~f:starts then
    (* [Array.make n x], [Array.create ~len x]: the value filled in, not the count before it. *)
    (match positional with _ :: (_ :: _ as values) -> values | values -> values) @ data
  else if (not stdlib) && List.exists selectors ~f:starts then
    List.take positional 1
    (* [Map.update m k ~f], [List.find_map xs ~f]: the callback's result is the population's, a
       [find] predicate's is not. *)
    @ (if List.exists [ "update"; "change"; "find_map" ] ~f:starts then
         List.filter_map arguments ~f:(function
           | (Ppxlib.Asttypes.Labelled "f" | Optional "f"), callback -> Some callback
           | _ -> None)
       else [])
    @ data
  else
    (* An inserting function's labeled key joins the population ([Map.set m ~key ~data]); a
       selector's selects it, above. *)
    positional
    @ List.filter_map arguments ~f:(function
      | (Ppxlib.Asttypes.Labelled "key" | Optional "key"), key -> Some key
      (* [List.map xs ~f]: what the callback returns is the population's elements ({!population}
         follows a function in data position into its body); a selector's [~f] is a predicate. *)
      | (Ppxlib.Asttypes.Labelled "f" | Optional "f"), callback -> Some callback
      | _ -> None)
    @ data

(** The collection-module functions that write into their first positional argument, a mutable
    container: what they are handed joins it. *)
let writers =
  [
    "set";
    "add";
    "push";
    "enqueue";
    "replace";
    "fill";
    "blit";
    "update";
    "change";
    "transfer";
    "find_or_add";
  ]

(* A callee under type annotations, [(List.map : ...) f xs], is the callee. *)
let rec strip_constraints expr =
  match expr.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> strip_constraints inner
  | _ -> expr

let gate_census content =
  let ast = Read.structure_of content in
  let readings = ref [] and gated_sites = ref 0 in
  (* What is written into a binding after it is made -- [x := e], [Queue.enqueue x e] -- by the
     binding's identity, with the scope it was written in. *)
  let writes = Hashtbl.create (module Int) in
  (* Claims are compared once the walk is over, when every write is known. *)
  let pending = ref [] in
  let partials = ref [] and bound_partials = ref [] and consumed = Hash_set.create (module Int) in
  let position expr = expr.pexp_loc.loc_start.pos_cnum in
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
  let binder =
    object
      inherit Ast_traverse.iter as super

      method! value_binding binding =
        bind binding.pvb_expr;
        super#value_binding binding
    end
  in
  binder#structure ast;
  let walker =
    object (self)
      inherit [gate_denotation, gate_module] Lexical_scope.scoped as super
      val mutable next_local = 0
      val mutable quiet = 0
      val mutable collecting : (string * gate_denotation) list ref option = None

      method local =
        next_local <- next_local - 1;
        { identity = next_local; definition = None }

      (* A module denotes [true] when it is [Verdict] or reaches into it ([Verdict.Claims], a
         [module V = Verdict] alias), [false] when it is anything else; a name some binding the scan
         cannot see into takes over denotes something else. *)
      method module_path env path =
        let components = Read.flatten_longident path in
        let head = Lexical_scope.head path in
        let single = List.length components = 1 in
        Some
          (match Map.find env.Lexical_scope.modules head with
          | Some denotes when single -> denotes
          | Some Verdict_module -> Verdict_module
          | Some (Collection_module _ as collection)
            when List.for_all (List.tl_exn components) ~f:(fun c ->
                     List.mem collection_modules c ~equal:String.equal) ->
              (* [List.Assoc] under an [open Base]. *)
              collection
          | Some (Standard_library { stdlib })
            when List.for_all (List.tl_exn components) ~f:(fun c ->
                     List.mem collection_modules c ~equal:String.equal) ->
              (* [B.List] under [module B = Base]. *)
              Collection_module
                {
                  stdlib;
                  mutable_ = List.mem mutable_modules (List.last_exn components) ~equal:String.equal;
                }
          | Some _ -> Other_module
          | None when String.equal head "Verdict" -> Verdict_module
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
                  }
              else Other_module)

      method! shadowed = Some Other_module

      method! module_of env module_expr =
        match module_expr.pmod_desc with
        | Pmod_structure items ->
            Some
              (Local_structure
                 (List.concat_map items ~f:(fun item ->
                      match item.pstr_desc with
                      | Pstr_value (_, bindings) ->
                          Lexical_scope.pattern_vars (List.map bindings ~f:(fun b -> b.pvb_pat))
                      | Pstr_primitive { pval_name = { txt; _ }; _ } -> [ txt ]
                      | _ -> [])))
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
                           { stdlib; mutable_ = List.mem mutable_modules name ~equal:String.equal }));
            }
        | Some (Local_structure names) ->
            (* Its names shadow whatever they denoted, a [gated_all] of its own included. *)
            Lexical_scope.bind_values env names self#local
        | _ -> env

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
                List.iter (Hashtbl.find_multi writes denotation.identity) ~f:(fun (written, env) ->
                    visit env written)))
        in
        visit env expr;
        reads

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
                    List.iter (Hashtbl.find_multi writes denotation.identity)
                      ~f:(fun (written, env) -> visit env written))
              | None -> Hash_set.add found (name, 0))
          | Pexp_constraint (inner, _)
          | Pexp_coerce (inner, _, _)
          | Pexp_sequence (_, inner)
          | Pexp_letexception (_, inner)
          | Pexp_newtype (_, inner) ->
              visit env inner
          | Pexp_open ({ popen_expr; _ }, inner) ->
              visit
                (self#opened ~top:false ~include_:false env (self#module_of env popen_expr))
                inner
          | Pexp_letmodule ({ txt = Some name; _ }, module_expr, inner) ->
              visit (self#bind_module env name module_expr) inner
          | Pexp_letmodule ({ txt = None; _ }, _, inner) -> visit env inner
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
                 returns one is built from the scrutinee. *)
              List.iter cases ~f:(fun case ->
                  let part = { (self#local) with definition = Some (scrutinee, env) } in
                  visit
                    (Lexical_scope.bind_values env
                       (Lexical_scope.pattern_vars [ case.pc_lhs ])
                       part)
                    case.pc_rhs)
          | Pexp_function (params, _, body) -> (
              (* A thunk or callback in data position ([~default:(fun () -> ...)]): what it returns,
                 in either syntax. *)
              let env =
                List.fold params ~init:env ~f:(fun env param ->
                    match param.pparam_desc with
                    | Pparam_val (_, _, pattern) -> self#bind_patterns env [ pattern ]
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
                  match (strip_constraints callee).pexp_desc with
                  | Pexp_ident { txt = Ppxlib.Longident.Lident ("!" | "@" | "ref"); _ } ->
                      List.iter positional ~f:(visit env)
                  | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ } -> (
                      match self#module_path env prefix with
                      | Some (Collection_module { stdlib; _ }) ->
                          List.iter (collection_arguments ~stdlib name arguments) ~f:(visit env)
                      | _ -> ())
                  | _ -> ()))
          | _ -> ()
        in
        visit env expr;
        found

      (* A write into a binding: [x := e], or a writer of a collection module ({!writers}) handed
         [x] first. *)
      method record_write env callee arguments =
        (* The binding written into, through any [let alias = rows] renaming on the way: a write
           through an alias is a write into what it names. *)
        let rec target env expr =
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
                  target env aliased
              | Some { identity; _ } -> Some identity
              | None -> None)
          | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> target env inner
          | _ -> None
        in
        let target = target env in
        let record identity written =
          List.iter written ~f:(fun e -> Hashtbl.add_multi writes ~key:identity ~data:(e, env))
        in
        match ((strip_constraints callee).pexp_desc, arguments) with
        | ( Pexp_ident { txt = Ppxlib.Longident.Lident ":="; _ },
            [ (Ppxlib.Asttypes.Nolabel, lhs); (Ppxlib.Asttypes.Nolabel, rhs) ] ) ->
            Option.iter (target lhs) ~f:(fun identity -> record identity [ rhs ])
        | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ }, _
          when List.exists writers ~f:(fun w ->
                   String.is_prefix (String.chop_prefix_if_exists name ~prefix:"unsafe_") ~prefix:w)
          -> (
            match self#module_path env prefix with
            | Some (Collection_module { stdlib; mutable_ = true }) ->
                let positional =
                  List.filter_map arguments ~f:(function
                    | Ppxlib.Asttypes.Nolabel, argument -> Some argument
                    | _ -> None)
                in
                let data =
                  List.filter_map arguments ~f:(function
                    | (Ppxlib.Asttypes.Labelled label | Optional label), argument
                      when List.mem [ "data"; "default"; "init"; "key" ] label ~equal:String.equal
                      ->
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
                List.iter containers ~f:(fun container ->
                    Option.iter (target container) ~f:(fun identity ->
                        record identity
                          (List.filter positional ~f:(fun other -> not (phys_equal other container))
                          @ labeled "src" @ labeled "f" @ data)))
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

      (* The [gated_*] claim an expression reaches, with every argument it has been handed so far,
         each with the scope it was written in. The nodes it absorbs -- the callee of an
         application, an application a pipeline rewrites -- are marked [consumed]: read here, they
         are not read again on their own. *)
      method gated env expr ~through ~marks =
        match expr.pexp_desc with
        | Pexp_constraint (inner, _) -> self#gated env inner ~through ~marks
        | Pexp_ident { txt; _ } -> (
            let form () = Option.map (Read.longident_of expr) ~f:(String.concat ~sep:".") in
            match txt with
            | Ppxlib.Longident.Lident name -> (
                (* A local binding of the name comes first: a file's own [gated_all] is not
                   [Verdict]'s, and a [let check = Verdict.gated_all ...] is followed. *)
                match Lexical_scope.lookup env name with
                | Some { identity; definition = Some (rhs, rhs_env) }
                  when not (Hash_set.mem through identity) ->
                    Hash_set.add through identity;
                    (* The binding this use completes, through every renaming on the way. *)
                    Hash_set.add marks (position rhs);
                    Option.map (self#gated rhs_env rhs ~through ~marks) ~f:(fun (_, arguments) ->
                        (name, arguments))
                | Some _ -> None
                | None ->
                    if List.mem gated_forms name ~equal:String.equal then
                      Option.map (form ()) ~f:(fun form -> (form, []))
                    else None)
            | Ppxlib.Longident.Ldot (prefix, name)
              when List.mem gated_forms name ~equal:String.equal
                   && Poly.equal (self#module_path env prefix) (Some Verdict_module) ->
                Option.map (form ()) ~f:(fun form -> (form, []))
            | _ -> None)
        | Pexp_apply (callee, arguments) -> (
            let callee, arguments, absorbed =
              match self#pipeline callee arguments with
              | Some rewritten -> rewritten
              | None -> (callee, arguments, None)
            in
            match self#gated env callee ~through ~marks with
            | Some (form, prior) ->
                Option.iter absorbed ~f:(Hash_set.add marks);
                Hash_set.add marks (position callee);
                Some
                  ( form,
                    prior @ List.map arguments ~f:(fun (label, argument) -> (label, argument, env))
                  )
            | None -> None)
        | _ -> None

      method! expression env expr =
        (if quiet = 0 then
           match expr.pexp_desc with
           | Pexp_apply (callee, arguments) -> (
               (* [sample |> Queue.enqueue queue] is [Queue.enqueue queue sample]. *)
               match self#pipeline callee arguments with
               | Some (callee, arguments, _) -> self#record_write env callee arguments
               | None -> self#record_write env callee arguments)
           | _ -> ());
        (if quiet = 0 && not (Hash_set.mem consumed (position expr)) then
           match expr.pexp_desc with
           | Pexp_ident _ -> (
               (* A claim named as a value rather than applied -- handed to [List.iter], say -- is
                  one this reader cannot complete. A [let] binding it to another name is not a use;
                  that name's own uses are. *)
               let marks = Hash_set.create (module Int) in
               match self#gated env expr ~through:(Hash_set.create (module Int)) ~marks with
               | Some (form, arguments)
                 when (not (complete arguments)) && not (Hash_set.mem bound (position expr)) ->
                   commit marks;
                   partials := (form, expr.pexp_loc.loc_start.pos_lnum) :: !partials
               | _ -> ())
           | Pexp_apply _ -> (
               let marks = Hash_set.create (module Int) in
               match self#gated env expr ~through:(Hash_set.create (module Int)) ~marks with
               | None -> ()
               | Some (form, arguments) -> (
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
                       pending :=
                         (fun () ->
                           let reads = self#gate_reads gate_env gate in
                           let shared =
                             Hash_set.filter
                               (self#population population_env population)
                               ~f:(Hash_set.mem reads)
                             |> Hash_set.to_list |> List.map ~f:fst
                             |> List.dedup_and_sort ~compare:String.compare
                           in
                           if not (List.is_empty shared) then
                             readings :=
                               {
                                 form;
                                 line;
                                 gate = Ppxlib.Pprintast.string_of_expression gate;
                                 reads = shared;
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
        super#expression env expr
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
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
  { readings = List.rev !readings; gated_sites = !gated_sites; unread }
