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

(** Whose a claim callee is, as {!Verdict_callee} resolves it: [Verdict]'s, [Ambiguous] -- reaching
    [Verdict] only past an [open] of a module that resolver cannot read, which may export a name or
    a [Verdict] module of its own -- [Free], or not [Verdict]'s. *)
type ownership = Verdict_callee.ownership = Verdicts | Ambiguous | Free | Not_verdicts

(* One application whose callee is a path ending in a name of [callees], with a positional argument:
   its last path component, whether that callee is [Verdict]'s (below), and the site its first
   positional argument -- the label -- keys. Shared by {!dialect_census} and {!line_break_census},
   so both key a label one way.

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
    |> List.filter_map ~f:(fun (_, ownership, site) ->
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
    [Verdict]'s. A partial application bound by [let] to a name, under any number of further partial
    applications, is completed at each application of that name, and [x |> f] and [f @@ x] are read
    as [f x]; so the [let check = Verdict.gated_all ~when_ ~on in check label xs] wrapper is one
    site per use. The two sides are then compared by LEXICAL identity, a name plus the binding it
    resolves to through {!Lexical_scope}, so a shadowed name is a different value; a name no binding
    of the source gives (a global, or one an unmodelled [open] brought in) is one identity, the same
    on both sides:

    - the GATE is every value identifier the [~when_] expression spells, and transitively every one
      spelled by the right-hand side of a [let] it resolves to (function bodies included), or by
      what the source writes into that binding (below), in any position;
    - the POPULATION of a [gated_*] claim is the identifier the quantified argument spells, and the
      collections it is BUILT from: following a [let]-bound name into its right-hand side, the
      branches (not the condition) of an [if], the cases of a [match] (the scrutinee only through a
      name a case pattern binds and the case returns), the body of a function in data position (a
      [~default:(fun () -> ...)] thunk), the body and handlers of a [try], the body of a [let],
      [let module], local [open] or sequence, the elements of a list, array or tuple literal or
      constructor payload, the operand of [!] and of [ref] and both operands of [@] (where the
      source does not bind the operator itself), and the arguments that carry the collection into a
      function of a standard collection module ([List], [Array], [Hashtbl], [Map], [Option],
      [String], ... — see {!collection_modules}), reached by name, through an alias
      ([module Rows = List]), and not under a name a local module took over ([module Map = Custom]);
      an unqualified [List] is [Stdlib]'s until an [open Base] in scope rebinds it. Which arguments
      carry it is {!collection_arguments}': every positional one, except that a [Base]-ordered
      selector ([find], [nth], [take], [remove], ...) contributes its collection alone, a
      [make]/[create] the value it fills in and not its count, an [init] what its callback returns,
      and a [range] nothing, plus the data-bearing labels [~default], [~init], [~data];
    - the body of a binding operator ([let+ x = e in body]), where [x] is a part of [e];
    - the POPULATION of a [gated] claim is what its boolean's quantifiers range over, each followed
      as above: the boolean is read through a name into its right-hand side and what the source
      writes into it, [not], [&&] and [||] (bare where the source does not bind them, or qualified
      by a standard library), [!] and [ref] where unbound too, the branches of an [if] (never the
      condition), the cases of a [match], the body and handlers of a [try], the body of a [let],
      [let module], local [open] or sequence, type constraints, pipelines, and a quantifier's own
      arguments, its predicate's body included (a quantifier nested there ranges over a population
      of its own, the predicate's parameters none); a quantifier is a {!quantifiers} function of a
      collection module, whose collection is its first positional argument in [Base]'s order (the
      first two for a pairwise one) and every positional argument in [Stdlib]'s;
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
    definition, where it is a parameter, and not at its callers. A [gated] boolean that quantifies
    nothing has no population: its comparisons and other calls are operands this reader does not
    read, so [gated ~when_:(xs <> []) label (sum xs = want)] passes, and so does a gate folded into
    a value ([p "…" ((not on_cpu) || ok)]), which {!folded_gate_census} reads instead. Each of these
    is a place a population gate can hide from this reader; none is a place it reports one that is
    not there.

    A gated application the reader cannot complete — a partial application no [let] binds, a
    [let]-bound one that no use the reader resolves completes (reached only through a module path,
    say), or a claim (or a [let]-bound wrapper of one) named as a value, handed to [List.iter] say —
    is reported as unread rather than counted as clean, so the boundary is loud there. *)

(** What a module name denotes to this reader. A collection module carries whether its functions
    take arguments in [Stdlib]'s order (the function first, the collection last) rather than
    [Base]'s (the collection first). *)
type gate_module =
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
  ambiguous : bool;
      (** The claim reaches [Verdict] only past an [open] of a module {!Verdict_callee} cannot read,
          which may export a name or a [Verdict] module of its own. *)
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

(** [Verdict.gated ~when_ ~on label b]: its population is what the quantifiers its boolean reduces
    to range over ({!quantifiers}). *)
let gated_boolean = "gated"

(** The collection-module functions that quantify over their collection, which a [gated] boolean is
    read through: true or false of every element, or of none. *)
let quantifiers =
  [
    "for_all";
    "for_alli";
    "for_all2";
    "for_all2_exn";
    "exists";
    "existsi";
    "exists2";
    "exists2_exn";
    "is_empty";
  ]

(* The arguments a quantifier ranges over: in [Base]'s order the first positional argument, the
   first two for a pairwise one; in [Stdlib]'s, where the predicate comes first ([List.for_all f
   xs]) but a [Base] spelling under no [open Base] reads the same way, every positional one -- the
   loud direction, as for {!collection_arguments}. *)
let quantified_arguments ~stdlib name arguments =
  let positional =
    List.filter_map arguments ~f:(function Ppxlib.Asttypes.Nolabel, a -> Some a | _ -> None)
  in
  if stdlib then positional
  else List.take positional (if String.is_substring name ~substring:"2" then 2 else 1)

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
  let owners = Verdict_callee.of_structure ast in
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
                  | Pexp_ident { txt = Ppxlib.Longident.Lident (("!" | "@" | "ref") as name); _ }
                    when Option.is_none (Lexical_scope.lookup env name) ->
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
                  List.iter (Hashtbl.find_multi writes denotation.identity)
                    ~f:(fun (written, env) -> visit env written)
              | _ -> ())
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
          | Pexp_ifthenelse (_, yes, no) ->
              visit env yes;
              Option.iter no ~f:(visit env)
          | Pexp_match (scrutinee, cases) ->
              List.iter cases ~f:(fun case ->
                  let part = { (self#local) with definition = Some (scrutinee, env) } in
                  visit
                    (Lexical_scope.bind_values env
                       (Lexical_scope.pattern_vars [ case.pc_lhs ])
                       part)
                    case.pc_rhs)
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
                    | Pparam_val (_, _, pattern) -> self#bind_patterns env [ pattern ]
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
                  match (strip_constraints callee).pexp_desc with
                  | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ }
                    when (connective name || List.mem [ "!"; "ref" ] name ~equal:String.equal)
                         && Option.is_none (Lexical_scope.lookup env name) ->
                      List.iter positional ~f:(visit env)
                  | Pexp_ident { txt = Ppxlib.Longident.Ldot (prefix, name); _ } -> (
                      match self#module_path env prefix with
                      | Some (Standard_library _) when connective name ->
                          List.iter positional ~f:(visit env)
                      | Some (Collection_module { stdlib; _ })
                        when List.mem quantifiers name ~equal:String.equal ->
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
                       pending :=
                         (fun () ->
                           let reads = self#gate_reads gate_env gate in
                           let built =
                             if String.equal member gated_boolean then
                               self#quantified population_env population
                             else self#population population_env population
                           in
                           let shared =
                             Hash_set.filter built ~f:(Hash_set.mem reads)
                             |> Hash_set.to_list |> List.map ~f:fst
                             |> List.dedup_and_sort ~compare:String.compare
                           in
                           if not (List.is_empty shared) then
                             readings :=
                               {
                                 form;
                                 ambiguous = reached_ambiguous;
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
    [Sys.getenv]/[getenv_opt], [Context.backend_name], [Context.hardware_limits] -- an applied path
    ending in one of these names, or an unqualified one the source does not bind), a value of [Sys]
    ([Sys.win32], [Sys.os_type], [Sys.big_endian], [Sys.argv], ...), the settings record
    [Utils.settings] (its fields being the configuration), and what is computed from those and
    constants alone: an application of a function from outside the source to run-constant and
    constant arguments, at least one of them run-constant ([backend_is_cpu backend_name],
    [String.equal backend_name "metal"], [not on_gpu]); an [if] or [match] over such parts; a field,
    a [lazy] or a constructor of one; a name bound to one by [let], anywhere in the source -- or,
    for a cell the source assigns into with [:=], to a run-constant value in its initializer or in
    any write, whatever their order. A constant is a literal, a constructor of constants, or a
    library value.

    So [on_cpu], [on_gpu], [is_metal], [seeds_here = Autotune.seeded_default_backend backend_name]
    and a [lazy] hardware probe are gates; a value computed from the program under test -- a census,
    a readback, a query's answer, a seed list -- is not, though it read the backend along the way,
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
    source binds under one of those names is read as the standard one. *)

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

type folded_census = {
  folded : folded_gate list;  (** In source order. *)
  folded_claim_sites : int;  (** Every claim whose value was read. *)
}

let configuration_readers =
  [ "get_global_arg"; "get_global_flag"; "getenv"; "getenv_opt"; "backend_name"; "hardware_limits" ]

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
    emptiness census, a distinctness check. *)
let folded_unread_members = [ "fail"; "skipped"; "case"; "p_empty"; "p_pairwise_distinct" ]

let folded_gate_census content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  let folded = ref [] and sites = ref 0 in
  let constancy = Hashtbl.create (module Int) in
  (* What the source assigns into a binding with [:=], by the binding's identity: a ref's value is
     its initializer's and every write's. Claims are judged once the walk is over, when every write
     is known. *)
  let writes = Hashtbl.create (module Int) in
  let pending = ref [] in
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
        | Pexp_field (inner, _)
        | Pexp_sequence (_, inner)
        | Pexp_open (_, inner)
        | Pexp_newtype (_, inner) ->
            self#constancy env inner
        | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
            match Lexical_scope.lookup env name with
            | Some { identity; definition = Some (rhs, rhs_env) } -> (
                match Hashtbl.find constancy identity with
                | Some known -> known
                | None ->
                    (* A cycle is the test's own computation. *)
                    Hashtbl.set constancy ~key:identity ~data:Varying;
                    let known =
                      match Hashtbl.find_multi writes identity with
                      | [] -> self#constancy rhs_env rhs
                      | written ->
                          (* A cell the source assigns into holds, somewhere, each value it is
                             given: one run-constant value makes it a gate wherever it is read --
                             the order of reads and writes is not followed, in the loud
                             direction. *)
                          let held =
                            self#constancy rhs_env rhs
                            :: List.map written ~f:(fun (value, env) -> self#constancy env value)
                          in
                          if List.mem held Run_constant ~equal:Poly.equal then Run_constant
                          else join held
                    in
                    Hashtbl.set constancy ~key:identity ~data:known;
                    known)
            | Some _ -> Varying
            | None -> Constant)
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
                            let gate =
                              List.find parts ~f:(fun (_, _, c) -> Poly.equal c Run_constant)
                            in
                            match gate with
                            | Some (leaf, positive, _)
                              when List.exists parts ~f:(fun (_, _, c) -> Poly.equal c Varying) ->
                                let printed =
                                  Ppxlib.Pprintast.string_of_expression leaf
                                  |> String.split_on_chars ~on:[ '\n'; ' ' ]
                                  |> List.filter ~f:(fun word -> not (String.is_empty word))
                                  |> String.concat ~sep:" "
                                in
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
  { folded = List.rev !folded; folded_claim_sites = !sites }
