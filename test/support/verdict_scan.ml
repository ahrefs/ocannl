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

let dialect_census content =
  let ast = Read.structure_of content in
  let key_of env argument =
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
                inherit [string option * int, unit] Lexical_scope.scoped
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
  let pass_fails = ref [] and skips = ref [] and item = ref 0 in
  let iterator =
    object (self)
      inherit [string option * int, unit] Lexical_scope.scoped as super
      val mutable next_binding = 0

      method local =
        next_binding <- next_binding - 1;
        (None, next_binding)

      method module_path _ _ = None

      method! let_denotes binding =
        match Read.pattern_name binding.pvb_pat with
        | Some _ -> (Read.string_literal binding.pvb_expr, binding.pvb_loc.loc_start.pos_cnum + 1)
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
            match (Read.longident_of callee, label) with
            | Some path, Some label ->
                let name = List.last_exn path in
                let site () =
                  {
                    callee = String.concat ~sep:"." path;
                    key = key_of env label;
                    item = !item;
                    line = expr.pexp_loc.loc_start.pos_lnum;
                  }
                in
                if List.mem pass_fail_callees name ~equal:String.equal then
                  pass_fails := site () :: !pass_fails
                else if List.mem skipped_callees name ~equal:String.equal then
                  skips := site () :: !skips
            | _ -> ())
        | _ -> ());
        super#expression env expr
    end
  in
  ignore
    (iterator#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  let pass_fails = List.rev !pass_fails and skips = List.rev !skips in
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

    Every application of a callee whose last path component is [gated_all], [gated_alli] or
    [gated_exists] (qualified, or opened through [Verdict.Claims]) that carries its [~when_] and
    both positional arguments. A partial application bound by [let] to a name, under any number of
    further partial applications, is completed at each application of that name, and [x |> f] and
    [f @@ x] are read as [f x]; so the [let check = Verdict.gated_all ~when_ ~on in check label xs]
    wrapper is one site per use. The two sides are then compared by LEXICAL identity, a name plus
    the binding it resolves to through {!Lexical_scope}, so a shadowed name is a different value:

    - the GATE is every value identifier the [~when_] expression spells, and transitively every one
      spelled by the right-hand side of a [let] it resolves to (function bodies included), in any
      position;
    - the POPULATION is the identifier the quantified argument spells, and the collections it is
      BUILT from: following a [let]-bound name into its right-hand side, the branches (not the
      condition) of an [if] and the cases (not the scrutinee) of a [match], the body of a [let] or
      sequence, the elements of a list, array or tuple literal or constructor payload, the operand
      of [!] and both operands of [@], and the positional arguments of a function qualified by a
      collection module ([List], [Array], [Hashtbl], [Map], [Option], [String], ... — see
      {!collection_modules}).

    The claim is refused when the two share a binding. The asymmetry is the point: a gate may read
    what the population is COMPUTED from (the site, the hardware limits, the backend steering an
    [if]), which is how a capability judgment over the same program reads, and must not read the
    population or a collection it is mapped, filtered or appended from.

    {2 What is not read}

    The population is not followed through a call to any other function (a file-local
    [gpu_seeds pre] stops at the call), through a record field, through a function parameter (the
    caller's argument is not substituted), or through an [open List]-style unqualified collection
    call; a destructured [let] binds names that are followed no further, and a value reached through
    a module path ([M.rows]) is an identity on neither side. A gate threaded into a wrapper as its
    own parameter ([let check ~when_ label xs = gated_all ~when_ ...]) is read at the wrapper's
    definition, where it is a parameter, and not at its callers. [Verdict.gated]'s boolean is not a
    population and is out of scope. Each of these is a place a population gate can hide from this
    reader; none is a place it reports one that is not there.

    A [gated_*] application the reader cannot complete — passed un-applied to another function, say
    — is reported as unread rather than counted as clean, so the boundary is loud there. *)

type gate_denotation = {
  identity : int;
  definition : (expression * (gate_denotation, unit) Lexical_scope.env) option;
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
    "List";
    "ListLabels";
    "Array";
    "ArrayLabels";
    "Seq";
    "Sequence";
    "Hashtbl";
    "Map";
    "Set";
    "Queue";
    "Stack";
    "Option";
    "String";
  ]

let gate_census content =
  let ast = Read.structure_of content in
  let readings = ref [] and gated_sites = ref 0 in
  let partials = ref [] and consumed = Hash_set.create (module Int) in
  let position expr = expr.pexp_loc.loc_start.pos_cnum in
  let walker =
    object (self)
      inherit [gate_denotation, unit] Lexical_scope.scoped as super
      val mutable next_local = 0
      val mutable quiet = 0
      val mutable collecting : (string * gate_denotation) list ref option = None

      method local =
        next_local <- next_local - 1;
        { identity = next_local; definition = None }

      method module_path _ _ = None

      method! binding_denotes env binding =
        let identity = binding.pvb_loc.loc_start.pos_cnum + 1 in
        match Read.pattern_name binding.pvb_pat with
        | Some _ -> { identity; definition = Some (binding.pvb_expr, env) }
        | None -> { identity; definition = None }

      method! attribute _ attr = attr

      method! ident env ident =
        match (collecting, ident.txt) with
        | Some found, Ppxlib.Longident.Lident name -> (
            match Lexical_scope.lookup env name with
            | Some denotation -> found := (name, denotation) :: !found
            | None -> ())
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
              if not (Hash_set.mem seen denotation.identity) then (
                Hash_set.add seen denotation.identity;
                Option.iter denotation.definition ~f:(fun (rhs, env) -> visit env rhs)))
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
                    Option.iter denotation.definition ~f:(fun (rhs, env) -> visit env rhs))
              | None -> ())
          | Pexp_constraint (inner, _)
          | Pexp_coerce (inner, _, _)
          | Pexp_sequence (_, inner)
          | Pexp_open (_, inner) ->
              visit env inner
          | Pexp_ifthenelse (_, yes, no) ->
              visit env yes;
              Option.iter no ~f:(visit env)
          | Pexp_match (_, cases) | Pexp_try (_, cases) ->
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
                  match Read.longident_of callee with
                  | Some [ ("!" | "@") ] -> List.iter positional ~f:(visit env)
                  | Some path
                    when List.length path >= 2
                         && List.mem collection_modules
                              (List.nth_exn path (List.length path - 2))
                              ~equal:String.equal ->
                      List.iter positional ~f:(visit env)
                  | _ -> ()))
          | _ -> ()
        in
        visit env expr;
        found

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

      (* The [gated_*] claim an application reaches, with every argument it has been handed so far,
         each with the scope it was written in; the applications it went through are marked
         consumed. *)
      method gated env expr ~through =
        match expr.pexp_desc with
        | Pexp_constraint (inner, _) -> self#gated env inner ~through
        | Pexp_ident { txt; _ } -> (
            match (Read.longident_of expr, txt) with
            | Some path, _ when List.mem gated_forms (List.last_exn path) ~equal:String.equal ->
                Some (String.concat ~sep:"." path, [])
            | _, Ppxlib.Longident.Lident name -> (
                match Lexical_scope.lookup env name with
                | Some { identity; definition = Some (rhs, rhs_env) }
                  when not (Hash_set.mem through identity) ->
                    Hash_set.add through identity;
                    Option.map (self#gated rhs_env rhs ~through) ~f:(fun (_, arguments) ->
                        (name, arguments))
                | _ -> None)
            | _ -> None)
        | Pexp_apply (callee, arguments) -> (
            let callee, arguments, absorbed =
              match self#pipeline callee arguments with
              | Some rewritten -> rewritten
              | None -> (callee, arguments, None)
            in
            match self#gated env callee ~through with
            | Some (form, prior) ->
                (* The partial applications this one completes are read here, not on their own. *)
                Option.iter absorbed ~f:(Hash_set.add consumed);
                (match callee.pexp_desc with
                | Pexp_apply _ -> Hash_set.add consumed (position callee)
                | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> (
                    match Lexical_scope.lookup env name with
                    | Some { definition = Some (rhs, _); _ } -> Hash_set.add consumed (position rhs)
                    | _ -> ())
                | _ -> ());
                Some
                  ( form,
                    prior @ List.map arguments ~f:(fun (label, argument) -> (label, argument, env))
                  )
            | None -> None)
        | _ -> None

      method! expression env expr =
        (if quiet = 0 && not (Hash_set.mem consumed (position expr)) then
           match expr.pexp_desc with
           | Pexp_apply _ -> (
               match self#gated env expr ~through:(Hash_set.create (module Int)) with
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
                   | Some (gate, gate_env), Some (population, population_env) ->
                       Int.incr gated_sites;
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
                           :: !readings
                   | _ -> partials := (position expr, form, line) :: !partials))
           | _ -> ());
        super#expression env expr
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  let unread =
    List.rev !partials
    |> List.filter_map ~f:(fun (at, form, line) ->
        if Hash_set.mem consumed at then None else Some (form, line))
  in
  { readings = List.rev !readings; gated_sites = !gated_sites; unread }
