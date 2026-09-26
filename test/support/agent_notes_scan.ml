(** Reading [docs/agent-notes.md] and [docs/agent-notes/] as structure rather than as prose.

    The agent notes are the project's cross-session memory: an index table of one row per subsystem
    file, and files that are a title, a two-line abstract, a backlink, and then a flat list of
    bullets. Nothing checks that shape, so it corrupts silently — and a corrupted note is worse than
    a missing one, because every later session inherits it as fact. Three of the six review findings
    on the split that created these files (lukstafi/ocannl-staging#406) were exactly that, and each
    was decidable from the text alone:

    - a merge resolution inserted an incoming bullet after the FIRST line of its hunk context
      instead of the last, cutting a bullet's closing sentence in half and stranding the tail 39
      lines later;
    - the index's [fast math] hook sat on a row whose file contains no fast-math guidance, so
      following the index led away from the trap it names — and two further hooks had drifted out of
      their file's wording entirely ([identifier blacklist] against [ident_blacklist]);
    - an edit wrapped one index row across two physical lines, which ends a Markdown table: the row
      was truncated and the five rows below it rendered as pipe-delimited prose.

    This module is the reader behind the checks for those, plus the two the wave added: that no file
    is unreachable from the index, and that no bullet is repeated across files. It is pure over
    strings — a file is a [(name, contents)] pair, nothing is opened here — so the negative controls
    in [test/operations/agent_notes_scan_cases.ml] exercise the same functions the live-tree scan in
    [test/operations/agent_notes_structure.ml] runs over the repository.

    {1 The seven rules}

    Each is stated as the thing that must be TRUE, and each finding names the rule that failed.

    - {b bullet-integrity}: the list structure parses under one unambiguous reading, and no bullet's
      text is cut. Markers are ["- "] at an even indentation; a continuation line sits at exactly
      the innermost open bullet's indentation plus two; a bullet's text — its start line and its
      continuation lines joined by single spaces — ends in sentence-terminating punctuation. The
      last clause is the one that catches a splice: a bullet a merge cut in half ends mid-word, and
      the tail it stranded lands inside some other bullet. See {!bullet_text_is_terminated} for the
      exact predicate and its escape.
    - {b index-agreement}: every index row is
      [| [<basename>](agent-notes/<basename>) | prose with `hooks` |], the target is a file this
      scan was handed, the link text is that file's basename, an anchor (if the link carries one)
      names a heading the file has, and every backticked hook in the second cell occurs verbatim in
      the target. The index is what a lookup greps before opening anything, so a hook absent from
      its target is a dead end even when the file is right.
    - {b table-shape}: every table block — a maximal run of lines whose trimmed form starts with a
      pipe — is a table: at least a header, a delimiter row and one data row; every line closes with
      an unescaped pipe; all lines carry the same number of cells; and the line after the block, if
      non-blank, has no pipe outside inline code, which is what a wrapped row's tail looks like.
      This is checked BEFORE index-agreement consumes the rows, because a wrapped row otherwise
      drops silently out of the set being checked — which is how the wrap survived the round that
      made it.
    - {b reachability}: every file handed to this scan is the target of exactly one index row, and
      every file carries the backlink to the index. An orphan is precisely the "the hook names a
      file carrying none of it" failure seen from the other end.
    - {b no-repetition}: no two bullets in the notes share their whitespace-normalized text, and no
      two share their first {!near_duplicate_prefix} characters case-insensitively. A fact promoted
      twice is a fact that will be updated once.
    - {b qualified-citations}: every numeric GitHub reference names its repository. A bare [#NNN]
      silently resolves against whichever repository renders the note; [staging#NNN],
      [gh-ocannl-NNN] and [ahrefs/ocannl#NNN] do not. Inline code, fenced blocks, comments and
      identifier-attached hashes are inert to this rule.
    - {b guide-anchors}: every [<note>.md#<anchor>] pointer in the agent guide ([AGENTS.md]) names a
      notes file and a heading that file has. The guide keeps a rule and points at its mechanism; a
      note long enough to need sections is useless to a pointer naming only the file, and a heading
      renamed under an anchored pointer strands every rule pointing at it (gh-ocannl-1044). A bare
      basename is a notes file; a path through [docs/agent-notes/] is resolved the same way, and a
      path anywhere else is not a pointer into the notes and is not read.

    {1 What it deliberately does not read}

    Prose. Whether a bullet is TRUE of the code is not decidable here and is not attempted; the
    claim is only that what someone wrote is intact and reachable. Nor is Markdown implemented: this
    reads the dialect the notes are written in — flat ["- "] lists, one nesting level, one table in
    the index — and reports anything outside it rather than guessing. A note that wants a fenced
    code block or a setext heading has to teach this module about it first, which is the intended
    cost.

    {1 Inline code spans}

    Pipes and backticks matter to two rules, and the notes are full of both inside code spans
    ([`=:||`], [`none|cc|…`], [`[%expect {|…|}]`]). {!pipes_outside_code} and {!code_spans} are the
    one place that distinction is decided: a backtick run opens a span that the next run of the SAME
    length closes, per CommonMark, so ["``a|b``"] is one span and the pipe inside it is not a cell
    separator. An unterminated run is not a span and its content stays outside. *)

open Base

type finding = {
  rule : string;
  file : string;  (** The notes file, apart from the line, so a consumer never re-parses [where]. *)
  line : int option;  (** The line, numerically, so findings order by document position. *)
  where : string;
  message : string;
  subject : string option;
      (** The thing the finding is ABOUT, kept apart from the prose that describes it: for a bullet,
          its opening. An exemption names a bullet, and a message is free to be reworded, so
          matching an exemption against the message would break the moment the wording improved --
          and did: the key format documented one thing while the message carried the bullet's TAIL
          (Codex P2, round 1). Consumers compare against [where]'s file and this. *)
}
(** A structural defect: which rule it broke, where, and what is wrong. [where] is
    ["<file>:<line>"], or ["<file>"] for a whole-file finding. *)

type bullet = { file : string; line : int; indent : int; text : string }
(** One bullet, joined: [text] is the start line's content and every continuation line, separated by
    single spaces, with the ["- "] marker removed. [line] is the start line, 1-based. *)

let rule_bullet_integrity = "bullet-integrity"
let rule_index_agreement = "index-agreement"
let rule_table_shape = "table-shape"
let rule_reachability = "reachability"
let rule_no_repetition = "no-repetition"
let rule_qualified_citations = "qualified-citations"
let rule_guide_anchors = "guide-anchors"

(** All seven, in the order the live-tree scan reports them.

    This list stands in for a protocol -- "the rules this scan has" -- that nothing else states, so
    the two ways it can part from the rules are both checked rather than assumed (gh-ocannl-706).
    Below, {!unnamed_rule_findings} answers whether a finding carries a rule the list does not name;
    the live-tree scan holds the rules it reports equal to this list, and the case file holds the
    rules it exercises equal to it too. *)
let rules =
  [
    rule_bullet_integrity;
    rule_index_agreement;
    rule_table_shape;
    rule_reachability;
    rule_no_repetition;
    rule_qualified_citations;
    rule_guide_anchors;
  ]

let names_a_rule r = List.mem rules r ~equal:String.equal

let finding ?subject ~file ~line ~rule message =
  { rule; file; line = Some line; where = Printf.sprintf "%s:%d" file line; message; subject }

let file_finding ?subject ~file ~rule message =
  { rule; file; line = None; where = file; message; subject }

(** How an exemption or a report names the bullet a finding is about: the file it is in and the
    bullet's opening, which is what a person can copy out of the message and paste into a list. *)
let subject_key ~file ~subject = Printf.sprintf "%s: %s" file subject

(** The exemption key a finding would be silenced by, for the findings that can be exempted at all.
    Built from the finding's own structured fields, so the key a message tells you to paste is
    exactly the key that matches — the two cannot drift apart. *)
let exemption_key f = Option.map f.subject ~f:(fun subject -> subject_key ~file:f.file ~subject)

(* ------------------------------------------------------------------ *)
(* Lines, indentation, inline code *)
(* ------------------------------------------------------------------ *)

(** Physical lines, 1-based, with a trailing CR dropped: a golden promoted on Windows and a file
    edited there both arrive CRLF, and an indentation or terminator rule that sees the CR as content
    would fail on the same text that passes elsewhere. *)
let lines contents =
  String.split ~on:'\n' contents
  |> List.map ~f:(fun l -> match String.chop_suffix l ~suffix:"\r" with Some l -> l | None -> l)
  |> List.mapi ~f:(fun i l -> (i + 1, l))

let indent_of line =
  let n = String.length line in
  let rec go i = if i < n && Char.equal line.[i] ' ' then go (i + 1) else i in
  go 0

let has_leading_tab line =
  match String.lfindi line ~f:(fun _ c -> not (Char.equal c ' ')) with
  | Some i -> Char.equal line.[i] '\t'
  | None -> false

let is_blank line = String.is_empty (String.strip line)

(** Half-open [(start, stop)] character ranges of the inline code spans of a line, CommonMark's
    rule: a run of N backticks opens a span that the next run of exactly N closes, and a run with no
    partner opens nothing. The ranges include the delimiters. *)

(** {2 Inert regions: one lexer, both states}

    Two things in these notes hold text that is NOT content — inline code spans and HTML comments —
    and every rule above depends on knowing where they are: a pipe inside one is not a cell
    separator, a link inside one is not navigable.

    Both cross line boundaries, and both were first written as per-line functions. That cost two
    review rounds in the same place, each finding the same shape from a different side: state
    recomputed per line is state discarded at every newline, so a construct spanning two lines hides
    whatever is on the second one. Splitting the job between two independent per-line strippers also
    let them disagree — a backtick inside a comment, a [<!--] inside a code span.

    So there is one left-to-right pass over the file, holding one state, and every caller reads its
    answer. A blank line ends an unterminated code span, because CommonMark does not let one cross a
    paragraph break; an HTML comment survives blank lines, because it is a block that ends only at
    [-->]. Either construct left open at the end of the file is reported rather than assumed closed.
*)

(** Markdown allows a block's line at most THREE leading spaces; at four it is an indented code
    block instead. Stripping all of it read a table indented into code as a table, so an index whose
    rows drifted right rendered as a code sample with no navigable link in it while every rule
    stayed green (Codex P2, round 2). Beyond three the line falls through to the continuation rules,
    which is right inside a bullet and reported outside one. *)
let max_block_indent = 3

(** Whether the character at [i] is escaped: by the PARITY of the run of backslashes before it, not
    by the single character before it. Two backslashes are an escaped backslash, which leaves the
    pipe after them live — so ["| a \\\\| b | c |"] is three cells to a renderer and was two to this
    scan, quietly dropping a cell out of the width check (Codex P2, round 3). *)
let escaped_at line i =
  let rec count j acc =
    if j >= 0 && Char.equal line.[j] '\\' then count (j - 1) (acc + 1) else acc
  in
  count (i - 1) 0 % 2 = 1

type inert_state = In_text | In_code of int | In_comment | In_fence of char * int

type inert_scan = {
  ranges : (int * (int * int) list) list;
  comment_ranges : (int * (int * int) list) list;
      (** The subset of {!ranges} that is HTML comment. The distinction is not cosmetic: text inside
          a code span RENDERS -- it is part of what a reader sees and part of what a bullet says --
          while text inside a comment does not. Treating both as equally absent made two bullets
          differing only on a code line read as the same bullet, and the repetition rule reported a
          duplicate that a reader can plainly tell apart (Codex P2, round 8). *)
  fence_ranges : (int * (int * int) list) list;
      (** The lines of a fenced block, delimiters included. A fence is a LEAF BLOCK: it ends the
          paragraph above it, so its lines belong to no bullet and contribute no text -- unlike a
          code span's lines, which are part of the sentence they sit in. *)
  unclosed : inert_state;
  fences : (int * string) list;  (** line, and the marker that opened it *)
  comments : int list;  (** line on which each HTML comment opened *)
}
(** What one pass over a file learned: where the inert text is, what was left open at the end, and
    where each construct that HIDES text from the reader began. *)

let at line i pattern =
  let n = String.length line and m = String.length pattern in
  i + m <= n && String.equal (String.sub line ~pos:i ~len:m) pattern

let run_length line i c =
  let n = String.length line in
  let j = ref i in
  while !j < n && Char.equal line.[!j] c do
    Int.incr j
  done;
  !j - i

(** A fence is a BLOCK: three or more backticks or tildes at the start of a line, after at most
    three spaces. It is not a code span, however much a run of three backticks looks like one — and
    reading it as one made a whole fenced block inert and swallowed the report of the fence itself
    (Codex P2, round 5, by way of a fixture that stopped failing). Deciding it here, in the one pass
    that owns inert text, is what stops this scan from having two disagreeing notions of a fence. *)
let fence_at line =
  let indent = indent_of line in
  if indent > max_block_indent then None
  else
    let c = if indent < String.length line then line.[indent] else ' ' in
    if Char.equal c '`' || Char.equal c '~' then
      let len = run_length line indent c in
      (* A backtick fence's info string may not contain a backtick, which is precisely what keeps an
         inline ```foo``` span from opening a block. Looking only at the leading run reported a
         perfectly ordinary prose line as an unclosed fenced block (Codex P2, round 6). Tildes carry
         no such restriction, because a tilde run is never a code span. *)
      let closed_inline =
        Char.equal c '`' && String.contains (String.drop_prefix line (indent + len)) '`'
      in
      if len >= 3 && not closed_inline then Some (c, len) else None
    else None

(** Per line, the half-open ranges that are inert; ranges include their delimiters. *)
let inert_by_line contents =
  let state = ref In_text in
  let fences = ref [] in
  let comments = ref [] in
  let ranges_of = ref [] in
  (* Committed ranges, by line. A code span's ranges are held back until it CLOSES: an unmatched
     backtick run renders literally, so the text after it is ordinary prose and any link in it is
     navigable. Marking it inert as the run opened, then dropping the state at the paragraph break
     without dropping the ranges, suppressed a real backlink and reported the file unreachable
     (Codex P2, round 6) -- a false failure. Pending ranges are discarded at a paragraph break and
     at the end of the file, which is exactly what a renderer does with them. *)
  let committed : (int, (int * int) list) Hashtbl.t = Hashtbl.create (module Int) in
  let comment_tbl : (int, (int * int) list) Hashtbl.t = Hashtbl.create (module Int) in
  let fence_tbl : (int, (int * int) list) Hashtbl.t = Hashtbl.create (module Int) in
  let pending = ref [] in
  let add tbl lineno range =
    Hashtbl.update tbl lineno ~f:(function None -> [ range ] | Some rs -> range :: rs)
  in
  let commit () =
    List.iter !pending ~f:(fun (lineno, range) -> add committed lineno range);
    pending := []
  in
  let discard () = pending := [] in
  List.iter (lines contents) ~f:(fun (lineno, line) ->
      (* A paragraph break ends an unterminated span, and its text was never code. A comment and a
         fence are blocks and survive one. *)
      (if is_blank line then
         match !state with
         | In_code _ ->
             discard ();
             state := In_text
         | _ -> ());
      let n = String.length line in
      let start = ref 0 in
      let i = ref 0 in
      let fence_line =
        match !state with
        | In_fence (c, len) -> (
            match fence_at line with
            | Some (c', len') when Char.equal c c' && len' >= len ->
                state := In_text;
                true
            | _ -> true)
        | In_text -> (
            match fence_at line with
            | Some (c, len) ->
                fences := (lineno, String.make len c) :: !fences;
                state := In_fence (c, len);
                true
            | None -> false)
        | _ -> false
      in
      if fence_line then
        if n > 0 then (
          add committed lineno (0, n);
          add fence_tbl lineno (0, n))
        else add fence_tbl lineno (0, 0)
      else
        while !i < n do
          match !state with
          | In_fence _ -> i := n
          | In_text ->
              (* An ESCAPED backtick renders literally and opens nothing. Inside a span the rule
                 does not apply: CommonMark gives backslashes no meaning there, so a backtick always
                 closes. *)
              if Char.equal line.[!i] '`' && not (escaped_at line !i) then (
                let len = run_length line !i '`' in
                start := !i;
                state := In_code len;
                i := !i + len)
              else if at line !i "<!--" && not (escaped_at line !i) then (
                comments := lineno :: !comments;
                start := !i;
                state := In_comment;
                i := !i + 4)
              else Int.incr i
          | In_code len ->
              if Char.equal line.[!i] '`' then (
                let run = run_length line !i '`' in
                if run = len then (
                  pending := (lineno, (!start, !i + run)) :: !pending;
                  commit ();
                  state := In_text;
                  start := 0);
                i := !i + run)
              else Int.incr i
          | In_comment ->
              if at line !i "-->" then (
                add committed lineno (!start, !i + 3);
                add comment_tbl lineno (!start, !i + 3);
                state := In_text;
                start := 0;
                i := !i + 3)
              else Int.incr i
        done;
      (* Still open at the end of the line: the remainder carries. A code span's share is pending
         until it closes; a comment's is committed, since a comment hides text whether or not it is
         ever closed. An empty range says nothing. *)
      (match !state with
      | In_text -> ()
      | In_code _ -> if n > !start then pending := (lineno, (!start, n)) :: !pending
      | In_comment ->
          if n > !start then (
            add committed lineno (!start, n);
            add comment_tbl lineno (!start, n))
      | In_fence _ -> ());
      ranges_of := lineno :: !ranges_of);
  (* An unmatched run at the end of the file is literal text too. *)
  (match !state with
  | In_code _ -> discard ()
  | _ -> ());
  {
    ranges =
      List.rev_map !ranges_of ~f:(fun lineno ->
          (lineno, List.rev (Option.value (Hashtbl.find committed lineno) ~default:[])));
    comment_ranges =
      List.rev_map !ranges_of ~f:(fun lineno ->
          (lineno, List.rev (Option.value (Hashtbl.find comment_tbl lineno) ~default:[])));
    fence_ranges =
      List.rev_map !ranges_of ~f:(fun lineno ->
          (lineno, List.rev (Option.value (Hashtbl.find fence_tbl lineno) ~default:[])));
    unclosed = !state;
    fences = List.rev !fences;
    comments = List.rev !comments;
  }

(** The inert ranges of a single line, for a caller with no file context — a table cell, a fixture.
*)
let inert_of_line line =
  match (inert_by_line line).ranges with (_, ranges) :: _ -> ranges | [] -> []

(** Kept under its old name: what this answers for one line is still "where is the inline code", and
    every caller that has a whole file passes the file's answer instead. *)
let code_spans = inert_of_line

let in_any_span spans i = List.exists spans ~f:(fun (start, stop) -> start <= i && i < stop)
let spans_at map lineno = Option.value (List.Assoc.find map lineno ~equal:Int.equal) ~default:[]

(** Whether a line is WHOLLY inside an inert region — every character of it that is not a space sits
    in a code span or an HTML comment. Such a line is somebody's example: a ["- "] in it is not a
    bullet, a pipe is not a table, a ['#'] is not a heading, and classifying it produces findings
    about text that no reader ever sees. Every structural consumer skips these, and skips only
    these.

    Wholly, not partly, and that boundary is load-bearing. The line that OPENS a multiline span
    still carries prose before the backtick, and the line that CLOSES one still carries prose after
    it — [`  <that target>\` afterwards exits 0 having produced nothing`] is a real continuation
    line in these notes, inert for its first seventeen characters and ordinary text for the rest.
    Skipping those would lose real content; classifying the middle lines invents failures. *)
let line_is_inert ~spans line =
  (not (List.is_empty spans))
  && String.foldi line ~init:true ~f:(fun i acc c ->
      acc && (Char.equal c ' ' || Char.equal c '\t' || in_any_span spans i))

(** Whether a line's FIRST VISIBLE COLUMN is real text — the one question every marker test has to
    ask before it classifies a line. A ['>'], ['#'], ['|'], ['<'] or ['-'] sitting inside a code
    span is not a marker, and the line carrying it is ordinary text: it renders, and it belongs to
    whatever paragraph or bullet encloses it.

    Distinct from {!line_is_inert}, and the difference is where every miss has come from: a line
    that OPENS or CLOSES a multiline span is not wholly inert, so the inert test passes it through
    and whatever raw column it happens to start with then gets classified. Three readers asked the
    question separately, two of them by the wrong test, so it is asked once here (Codex P2,
    gh-ocannl-714 round 2).

    Gating the marker rather than masking the inert text, because masking moves the indentation and
    the indentation is load-bearing: the notes contain a continuation line whose first seventeen
    characters close a code span and whose remainder is prose, and it is a continuation at depth two
    however much of its left edge is code. *)
let marker_is_text ~spans line = not (in_any_span spans (indent_of line))

(** Markdown's whitespace after a list marker: a tab is as good as a space, so ["1.\tFact"] is an
    ordered item and ["*\tFact"] a bulleted one. Requiring a literal space let both fall through as
    prose, which is the omission the foreign-marker check exists to prevent (Codex P2, round 5). *)
let md_space c = Char.equal c ' ' || Char.equal c '\t'

(** Positions of the ['|'] characters that separate table cells: outside inline code, and not
    backslash-escaped. [?spans] passes the paragraph-aware reading when the caller has one. *)
let pipes_outside_code ?spans line =
  let spans = match spans with Some s -> s | None -> code_spans line in
  String.foldi line ~init:[] ~f:(fun i acc c ->
      if Char.equal c '|' && (not (in_any_span spans i)) && not (escaped_at line i) then i :: acc
      else acc)
  |> List.rev

(** The cells of a table row: the text between the separating pipes, trimmed. A well-formed row
    starts and ends with one, so the empty pieces outside them are dropped. Returns [None] for a
    line that does not both start and end with a separating pipe — the shape a wrapped row takes. *)
let row_cells ?spans line =
  let dropped = String.length line - String.length (String.lstrip line) in
  let spans = Option.map spans ~f:(List.map ~f:(fun (a, b) -> (a - dropped, b - dropped))) in
  let line = String.strip line in
  match pipes_outside_code ?spans line with
  | [] -> None
  | first :: _ as pipes ->
      let last = List.last_exn pipes in
      if first <> 0 || last <> String.length line - 1 || List.length pipes < 2 then None
      else
        let rec cut = function
          | a :: (b :: _ as rest) ->
              String.strip (String.sub line ~pos:(a + 1) ~len:(b - a - 1)) :: cut rest
          | _ -> []
        in
        Some (cut pipes)

let is_table_line line =
  indent_of line <= max_block_indent && String.is_prefix (String.strip line) ~prefix:"|"

(** GitHub wants at least THREE hyphens in a delimiter cell. A cell shortened to ["-"] or ["--"]
    stops the block rendering as a table while every other property of it still holds, so accepting
    "nonempty and all dashes" would leave the rule green over a table that is no longer one (Codex
    P2, round 1). *)
let delimiter_min_hyphens = 3

let is_delimiter_row cells =
  (not (List.is_empty cells))
  && List.for_all cells ~f:(fun c ->
      let c = String.strip c in
      let c = Option.value (String.chop_prefix c ~prefix:":") ~default:c in
      let c = Option.value (String.chop_suffix c ~suffix:":") ~default:c in
      String.length c >= delimiter_min_hyphens && String.for_all c ~f:(Char.equal '-'))

(* ------------------------------------------------------------------ *)
(* Rule 1: bullet integrity *)
(* ------------------------------------------------------------------ *)

(** Trailing characters stripped before looking for the terminator, so that ["…done.**"],
    ["…see `x`."] and ["…(gh-ocannl-665)."] all read as terminated. *)
let closing_markup = [ '`'; ')'; ']'; '}'; '"'; '\''; '*'; '_' ]

let terminators = [ '.'; '!'; '?'; ':'; ';' ]

(** Whether a bullet's joined text ends a sentence. This is the whole of the "not cut in half" rule,
    and the corpus it was written against ends 175 of 177 bullets with a period, one with a colon
    and one with [".)"], so the predicate costs nothing to satisfy on purpose and is broken only by
    accident.

    The escape hatch for a bullet that genuinely ends without punctuation is to give it some — that
    is the cheap and usually right fix. The other, for a bullet whose ending is load-bearing (a
    trailing identifier, a table, a deliberate ellipsis), is the named exemption list in
    [agent_notes_structure.ml], where an entry has to name the bullet and say why. *)
let bullet_text_is_terminated text =
  let text = String.rstrip text in
  (* Whitespace comes off with the markup, not only before it: ["a fact ends. `"] -- period, space,
     a literal backtick -- is a finished sentence, and stripping the backtick to leave a trailing
     space read it as unfinished. *)
  let rec strip s =
    let s = String.rstrip s in
    match String.length s with
    | 0 -> s
    | n ->
        if List.mem closing_markup s.[n - 1] ~equal:Char.equal then strip (String.prefix s (n - 1))
        else s
  in
  let s = strip text in
  (not (String.is_empty s)) && List.mem terminators s.[String.length s - 1] ~equal:Char.equal

(** Whitespace collapsed to single spaces: the one normal form the repetition rule compares in and
    the one a bullet is named by, so a bullet re-wrapped across different lines is the same bullet
    to both. *)
let normalize text =
  String.split_on_chars text ~on:[ ' '; '\t' ]
  |> List.filter ~f:(fun s -> not (String.is_empty s))
  |> String.concat ~sep:" "

(** How much of a bullet is SHOWN when a finding talks about it. Display only — the identity below
    is exact, because a prefix is not an identity: two bullets agreeing for 48 characters and
    diverging afterwards got the same exemption key, so exempting one deliberate ending silenced the
    other's accidental truncation as well, and the staleness check still saw a match (Codex P2,
    round 3). *)
let subject_display_length = 48

(** How a bullet is NAMED, in a finding and in an exemption: its WHOLE text, whitespace-normalized.
    Whole, so that no two bullets can share a key. Normalized, so that re-wrapping a bullet across
    different lines does not invalidate an exemption written against it — while editing its WORDS
    does, which is right: an exemption is a claim about a particular sentence's particular ending,
    and a changed sentence deserves a fresh claim rather than inherited cover. *)
let bullet_subject (b : bullet) = normalize b.text

(** The ATX heading a line is, if it is one. CommonMark wants one to six ['#'] followed by
    whitespace or the end of the line: ["##ident_blacklist"] renders as PROSE, and recording it as a
    heading let the index's [#ident_blacklist] anchor pass while pointing at nothing (Codex P2,
    round 3). The marker also has to be a real ATX run — seven hashes is not a heading either. *)
let atx_heading line =
  if indent_of line > max_block_indent then None
  else
    (* Four spaces make it an indented code block, not a heading -- so an anchor naming it is dead,
       the same failure the missing-space case had (Codex P2, round 6). *)
    let s = String.strip line in
    let hashes =
      match String.lfindi s ~f:(fun _ c -> not (Char.equal c '#')) with
      | Some i -> i
      | None -> String.length s
    in
    if hashes = 0 || hashes > 6 then None
    else
      let rest = String.drop_prefix s hashes in
      if String.is_empty rest || Char.equal rest.[0] ' ' then
        (* A closing run of hashes is decoration, not content. *)
        Some (String.strip (String.rstrip (String.strip rest) ~drop:(Char.equal '#')))
      else None

(** Whether a line opens with something that WANTS to be a heading, so that a malformed one is
    reported rather than read as prose.

    A hash against a COMPLETE NUMBER wants nothing of the kind: ["#598"] is an issue or a pull
    request, and the notes cite those constantly, so a citation that wrapped onto a continuation's
    first column was read as a broken heading and intact prose was reported -- twice over, since
    closing the list there made the next continuation an orphan too (lukstafi/ocannl-staging#598).

    The number has to END for it to be a citation, which is what keeps the exemption off a title:
    ["#3D convolution"] is a level-one heading missing its space, and exempting it on its leading
    digit alone would have retired the malformed-heading rule for every title that starts with a
    number (Codex P2, round 1). So the digit run is read whole and the character after it must not
    continue a word -- a boundary the possessive ["#598's"] and the trailing comma both clear.

    No ATX heading is lost to the exception either way: a real one has a space or the line's end
    after its hash run, never a digit. Every other hash-prefixed line that is not a heading stays a
    finding, ["##598"] among them: two hashes against a number is a heading nobody writes as a
    citation. *)
let looks_like_heading line =
  let s = String.strip line in
  let is_citation =
    String.length s > 1
    && Char.is_digit s.[1]
    &&
    let after =
      match String.lfindi s ~f:(fun i c -> i > 0 && not (Char.is_digit c)) with
      | Some i -> Some s.[i]
      | None -> None
    in
    match after with None -> true | Some c -> not (Char.is_alphanum c || Char.equal c '_')
  in
  String.is_prefix s ~prefix:"#" && not is_citation

(** The headings a reader actually sees: a heading-looking line inside a multiline code span or an
    HTML comment is an example, and an index anchor naming its slug points at nothing (Codex P2,
    round 5). *)
let headings contents =
  let map = (inert_by_line contents).ranges in
  List.filter_map (lines contents) ~f:(fun (lineno, line) ->
      if marker_is_text ~spans:(spans_at map lineno) line then atx_heading line else None)

(** {2 The closed dialect}

    The module's claim is that it reads the notes' dialect and REPORTS anything outside it rather
    than guessing. Round 1 of review found four separate places where the code did the opposite and
    silently skipped instead — an ordered-list item, a fenced block's contents, an HTML block, a
    thematic break — each of which drops real text out of every rule while the golden stays green.
    They are one defect, not four: an unrecognised line fell through to "prose", and prose is not
    checked. So recognition is a closed vocabulary now, and the fallthrough is a finding.

    Adding a construct to the notes therefore means teaching this function about it first, which is
    the intended cost: a note that wants a fenced example has to say what the bullet rules mean
    inside one. *)

(** A list marker at any indentation that is not this scan's ["- "]. An ordered item is the one that
    bit: at column zero it read as prose (so its text got no termination or repetition check at
    all), and indented it folded into its parent's continuation (Codex P2, round 1). *)
let foreign_list_marker stripped =
  let n = String.length stripped in
  let ordered =
    match String.lfindi stripped ~f:(fun _ c -> not (Char.is_digit c)) with
    (* CommonMark caps the numeric part at nine digits, so a longer run is not a marker at all --
       and reporting one made ordinary prose opening with a long identifier a spurious failure
       (Codex P2, round 7). *)
    | Some i when i > 0 && i <= 9 && i + 1 < n ->
        let sep = stripped.[i] and after = stripped.[i + 1] in
        if (Char.equal sep '.' || Char.equal sep ')') && md_space after then
          Some (String.prefix stripped (i + 1))
        else None
    | _ -> None
  in
  match ordered with
  | Some m -> Some m
  | None ->
      (* A marker of any other flavour, or this repository's own dash written with a tab: the
         accepted form is "- " exactly, and everything else is reported rather than read as
         prose. *)
      List.find_map [ '*'; '+'; '-' ] ~f:(fun c ->
          if n >= 2 && Char.equal stripped.[0] c && md_space stripped.[1] then
            if Char.equal c '-' && Char.equal stripped.[1] ' ' then None
            else Some (String.of_char c ^ if Char.equal stripped.[1] '\t' then "\\t" else " ")
          else None)

(** A fence opens a region whose contents are not the notes' dialect at all: a ["- "] line inside
    one is example text, and reading it as a bullet invents integrity and repetition failures out of
    someone's code sample. Reported, and its contents skipped, so one unsupported construct is one
    finding. *)
let fence_marker stripped =
  List.find [ "```"; "~~~" ] ~f:(fun m -> String.is_prefix stripped ~prefix:m)

(** A block quote marker, which Markdown honours at EVERY depth: nested under a bullet, [`  > …`] is
    a quote inside the list item, not part of the bullet's prose. Wiring this to column zero only
    let a nested quote fold into its parent's text unchecked (Codex P2, round 2).

    The space after [>] is OPTIONAL, and nothing further along the line changes that: block
    structure is settled before any of it is read as arithmetic, so a line whose first visible
    column is [">= 8"] is a quote whatever the operand's type. Excluding [">="] under-read the
    marker, which is the one thing this dialect promises not to do; and so did the first attempt at
    fixing it, which excused a [">="] followed by a NUMBER and would have folded a quote whose text
    happens to open with a digit into the bullet above it (Codex P2, gh-ocannl-714).

    These notes do compare numbers in prose ([`  <= 8 and Retype-…`] is a real continuation line),
    so the requirement lands on the prose rather than on the marker: a comparison keeps its operator
    off the line's first visible column, or inside a code span, where this test never sees it. The
    finding says both. *)
let block_quote_marker stripped =
  if not (String.is_prefix stripped ~prefix:">") then None
  else if String.is_prefix stripped ~prefix:">=" then
    Some
      "a block quote: Markdown's quote marker does not need a space after it, so a line whose \
       first visible column is \">=\" renders as a quote and not as the comparison it reads like \
       -- rewrap so the operator is not first on the line, or write the comparison inside a code \
       span"
  else Some "a block quote, whose text belongs to no bullet and which this scan does not read"

(** A thematic break or a setext heading underline: a line of nothing but one repeated marker. It is
    a block at every depth, which is why it is factored out here rather than living inside the
    column-zero test. *)
let thematic_break stripped =
  let all_of c =
    String.length stripped >= 3
    && String.for_all stripped ~f:(fun d -> Char.equal d ' ' || Char.equal d c)
  in
  if all_of '-' || all_of '*' || all_of '_' || all_of '=' then
    Some "a thematic break or a setext heading underline, neither of which the notes use"
  else None

(** Whether a line opens a raw HTML block, as opposed to merely starting with ['<']. Every
    column-zero ['<'] used to count, which failed two kinds of perfectly ordinary prose: an autolink
    ([<https://example.com>]) and a comparison ([<= 8], which these notes write) (Codex P2, round
    6). An HTML block needs a tag-ish opener — a letter, ['/'], ['!'] or ['?'] — and an autolink is
    excluded by what it is: a bracketed run with no whitespace, carrying a scheme or an address. *)
let html_block_opener stripped =
  String.length stripped >= 2
  && Char.equal stripped.[0] '<'
  && (Char.is_alpha stripped.[1] || List.mem [ '/'; '!'; '?' ] stripped.[1] ~equal:Char.equal)
  && not
       (match String.index stripped '>' with
       | Some close ->
           let inside = String.sub stripped ~pos:1 ~len:(close - 1) in
           (not (String.exists inside ~f:(fun c -> Char.equal c ' ' || Char.equal c '\t')))
           && (String.contains inside ':' || String.contains inside '@')
       | None -> false)

(** A setext heading underline: a line of nothing but one repeated ['-'] or ['='], directly under a
    paragraph, which makes the line ABOVE it a heading. One marker is enough. The three-marker floor
    belongs to the THEMATIC BREAK and applying it here missed [--] and [==] entirely
    (gh-ocannl-714), so a heading written that way carried an anchor {!headings} does not know: an
    index row naming it would be reported dead while the link navigates.

    [under_paragraph] is the whole of what separates a heading from ordinary text here, and it is
    not a formality: the same [--] under a blank line, a heading, a table or a fence is a paragraph
    of two hyphens, and only a paragraph above it turns it into an underline. A run of three or more
    is reported either way, by {!thematic_break} when nothing above it is a paragraph. *)
let setext_underline ~under_paragraph stripped =
  let all_of c = (not (String.is_empty stripped)) && String.for_all stripped ~f:(Char.equal c) in
  if under_paragraph && (all_of '-' || all_of '=') then
    Some
      "a setext heading underline, which makes the line above it a heading: headings here are ATX \
       (\"## Title\"), and one written this way carries an anchor that this scan cannot read, so \
       an index row naming it reads as dead while the link navigates"
  else None

(** The lines that can be the last line of a paragraph — the one thing {!setext_underline} needs
    above it. Computed over the whole file rather than threaded through {!parse_file}'s dispatch, so
    that every branch of it reads the same answer off the same test.

    A blank line, a fence and its contents, a comment and any line that OPENS a block of its own are
    NOT paragraph content: an underline under any of them underlines nothing. A line inside a code
    span is, because it renders as part of the paragraph carrying it.

    What decides "opens a block" is the same gate {!parse_file} puts on every other marker test: a
    marker whose first visible column is inert is not a marker. Classifying the raw column instead
    dropped a bullet's closing code-span line — one beginning [`> example.`], [`| a | b |`] or
    [`## Section`] inside the span — out of the paragraph set, and the underline below it went
    unreported while a renderer made the line a heading (Codex P2, round 2). *)
let paragraph_lines ~inert ~fences_at ~comments_at contents =
  List.filter_map (lines contents) ~f:(fun (lineno, line) ->
      let stripped = String.strip line in
      let opens_a_block =
        marker_is_text ~spans:(spans_at inert lineno) line
        && (looks_like_heading line || is_table_line line || html_block_opener stripped
           || Option.is_some (block_quote_marker stripped)
           || Option.is_some (thematic_break stripped))
      in
      let in_fence = not (List.is_empty (spans_at fences_at lineno)) in
      let in_comment = line_is_inert ~spans:(spans_at comments_at lineno) line in
      if (not (is_blank line)) && (not in_fence) && (not in_comment) && not opens_a_block then
        Some lineno
      else None)
  |> Set.of_list (module Int)

(** Block constructs Markdown recognises only at column zero, and that the notes do not use. The
    column-zero condition is load-bearing for these two rather than incidental: [`  <= 8 and …`] and
    [`  <that target>`] are real continuation lines, the second of them continuing a code span
    opened on the line above, so an HTML test at depth would fail a correct file. *)
let unsupported_block ~under_paragraph stripped =
  match block_quote_marker stripped with
  | Some what -> Some what
  | None -> (
      if html_block_opener stripped then Some "an HTML block, whose text no rule here can see"
      else
        match thematic_break stripped with
        | Some what -> Some what
        | None -> setext_underline ~under_paragraph stripped)

(** One nesting level is what the notes are written in, and what {!parse_file} documents. A third
    level was being accepted anyway — [expected] simply advanced by two more spaces — so the
    discipline held only by convention (Codex P2, round 1). *)
let max_nesting_indent = 2

type parse = { bullets : bullet list; structure : finding list }

(** The one pass that both collects bullets and reports where the structure stopped parsing.

    The open-bullet stack is closed by a blank line, a heading, a table line and any line at column
    zero that is not a bullet — the notes' abstract and backlink paragraphs are exactly that. A
    continuation line must sit at the innermost open bullet's indentation plus two, with no popping:
    an indented line at the PARENT's depth while a nested bullet is open is a lazy continuation of
    the nested one to Markdown and of the parent to a naive reader, and reporting that ambiguity is
    cheaper than picking a side of it. *)
let parse_file ~file contents =
  let findings = ref [] in
  let report line rule msg = findings := finding ~file ~line ~rule msg :: !findings in
  let bad line msg = report line rule_bullet_integrity msg in
  let bullets = ref [] in
  (* Innermost FIRST: each entry is a bullet being accumulated, with its text lines reversed.
     Flushing always goes outermost first, so [bullets] comes out in document order -- which is what
     the repetition rule reports against (the SECOND occurrence is the finding) and what orders the
     findings of a file. *)
  let stack : (bullet * string list ref) list ref = ref [] in
  (* A blank line inside a list item does NOT end the list: what follows, indented to the item's
     continuation depth, is a second PARAGRAPH of the same bullet. Closing the list there reported
     nine findings against a correctly written note the moment one arrived on master (gh-ocannl-691
     review, round 9 CI) -- a false failure on somebody else's intact work, which is the failure
     mode this whole scan is supposed to avoid causing. So a blank line is remembered, not acted on;
     the next line decides whether the list ended. *)
  let blank_seen = ref false in
  let close_all () =
    List.iter (List.rev !stack) ~f:(fun (b, texts) ->
        let text = String.concat ~sep:" " (List.rev !texts) in
        bullets := { b with text } :: !bullets);
    stack := []
  in
  (* Set while inside a fenced region, holding the fence that opened it: its contents are somebody's
     example, not the notes' dialect, and reading a `- ` line in one as a bullet invents
     failures. *)
  let scan = inert_by_line contents in
  let inert = scan.ranges in
  let comments_at = scan.comment_ranges in
  let fences_at = scan.fence_ranges in
  let paragraphs = paragraph_lines ~inert ~fences_at ~comments_at contents in
  List.iter (lines contents) ~f:(fun (lineno, line) ->
      let stripped = String.strip line in
      (* A line wholly inside a code span or an HTML comment is somebody's example. Parsing it as
         structure both invents findings about text no reader sees and lets the closing delimiter be
         reported as an illegal lazy continuation (Codex P2, round 5). Such a line is transparent
         here: it neither opens, continues nor closes anything. *)
      let spans = spans_at inert lineno in
      (* Whether this line's FIRST VISIBLE COLUMN is real text. A marker sitting inside a code span
         is not a marker: "- sample`" on the line that closes a span is code, and reading it as a
         bullet invented one out of somebody's example (Codex P2, round 7). Why it is a gate rather
         than a mask is at [marker_is_text]. *)
      let marker_is_text = marker_is_text ~spans line in
      (* Read once and cleared here, so every branch below sees the same answer and no branch has to
         remember to reset it. *)
      let after_blank = !blank_seen in
      (* What a setext underline needs above it, and nothing else does: whether the PREVIOUS line is
         paragraph content. Decided by [paragraph_lines] over the whole file, so this reads the same
         in every branch below. *)
      let under_paragraph = Set.mem paragraphs (lineno - 1) in
      if not (is_blank line) then blank_seen := false;
      if line_is_inert ~spans line then
        (* Wholly inert, so it carries no structure -- but it may still carry TEXT. A line inside a
           code span renders, and is part of what its bullet says; a line inside an HTML comment
           does not, and is not. Collapsing the two made bullets differing only on a code line
           identical, and the repetition rule called them duplicates (Codex P2, round 8). *)
        if not (List.is_empty (spans_at fences_at lineno)) then
          (* A fenced block is a leaf block: it ends the list above it and its lines are nobody's
             prose. The fence itself is reported by [hiding_constructs]. *)
          close_all ()
        else if line_is_inert ~spans:(spans_at comments_at lineno) line then ()
        else match !stack with [] -> () | (_, texts) :: _ -> texts := stripped :: !texts
      else if is_blank line then blank_seen := true
      else if not marker_is_text then (
        if
          (* Text, with no structural marker of its own: it continues an open bullet, or it is prose
             and closes the list. At column zero under an open bullet it is a LAZY CONTINUATION, the
             same construct the marker path reports -- and this path was closing the stack silently,
             so the rule went inert exactly where an inert prefix hid the marker (Codex P2, round
             9). The first line of such a bullet reads as terminated once the unmatched delimiter is
             stripped, so nothing else would have caught it. *)
          indent_of line = 0
        then (
          if (not after_blank) && not (List.is_empty !stack) then
            bad lineno
              "prose at column zero directly under a bullet, which Markdown reads as a lazy \
               continuation of it: separate it with a blank line, or indent it two spaces to make \
               it the bullet's own";
          close_all ())
        else
          match !stack with
          | [] -> ()
          | (b, texts) :: _ -> if indent_of line = b.indent + 2 then texts := stripped :: !texts)
      else if has_leading_tab line then (
        bad lineno "a tab in the indentation: indentation here is spaces, two per nesting level";
        close_all ())
      else
        let indent = indent_of line in
        if looks_like_heading line then (
          if Option.is_none (atx_heading line) then
            bad lineno
              "a line opening with # that is not a heading: an ATX marker is one to six hashes, \
               followed by a space, indented at most three -- without all three of those it \
               renders as prose or as code, so an anchor naming it points at nothing";
          close_all ())
        else if is_table_line line then close_all ()
        else
          match foreign_list_marker stripped with
          | Some m ->
              bad lineno
                (Printf.sprintf
                   "the list marker %S: bullets here are written \"- \", and an item this scan \
                    does not recognise is an item no rule checks"
                   m);
              close_all ()
          | None -> (
              if String.equal stripped "-" then (
                (* A lone "-" under a paragraph is that paragraph's setext underline rather than an
                   empty list item -- CommonMark resolves the ambiguity in the heading's favour, and
                   the message follows it. Either way it is a finding; which one it is decides what
                   the author is told to fix. *)
                bad lineno
                  (match setext_underline ~under_paragraph stripped with
                  | Some what -> what
                  | None -> "an empty bullet");
                close_all ())
              else if String.is_prefix stripped ~prefix:"- " then (
                (* A bullet start closes every open bullet at or inside its own depth. *)
                let rec pop acc =
                  match !stack with
                  | (b, texts) :: rest when b.indent >= indent ->
                      stack := rest;
                      pop ((b, texts) :: acc)
                  | _ -> acc
                in
                (* [pop] prepends as it unwinds inwards-out, so [popped] is outermost first. *)
                List.iter (pop []) ~f:(fun (b, texts) ->
                    let text = String.concat ~sep:" " (List.rev !texts) in
                    bullets := { b with text } :: !bullets);
                let expected = match !stack with [] -> 0 | (b, _) :: _ -> b.indent + 2 in
                if indent > max_nesting_indent then
                  bad lineno
                    (Printf.sprintf
                       "a bullet nested %d deep: the notes go one level down and this scan reads \
                        no further, so anything below %d is unchecked"
                       ((indent / 2) + 1)
                       max_nesting_indent)
                else if indent <> expected then
                  bad lineno
                    (Printf.sprintf
                       "a bullet indented %d, where the open list puts the next one at %d" indent
                       expected);
                let b = { file; line = lineno; indent; text = "" } in
                stack := (b, ref [ String.drop_prefix stripped 2 ]) :: !stack)
              else if indent = 0 then (
                (match unsupported_block ~under_paragraph stripped with
                | Some what -> bad lineno what
                | None ->
                    (* Markdown reads column-zero prose directly under an open bullet as a LAZY
                       CONTINUATION of that item, so this text belongs to the bullet to a renderer
                       while this scan was closing the list and dropping it from every rule. A
                       bullet whose first line already ends in punctuation hid the whole transition
                       (Codex P2, round 3). Nothing legitimate needs it: the notes separate a
                       paragraph from the list above with a blank line, which is what closes the
                       list here. *)
                    if (not after_blank) && not (List.is_empty !stack) then
                      bad lineno
                        "prose at column zero directly under a bullet, which Markdown reads as a \
                         lazy continuation of it: separate it with a blank line, or indent it two \
                         spaces to make it the bullet's own");
                close_all ())
              else
                match
                  match block_quote_marker stripped with
                  | Some what -> Some what
                  | None -> (
                      (* A hyphen line under a bullet is a separate block to Markdown, not part of
                         the bullet's prose -- and folding it in left every rule green, since the
                         joined text still ended in a period (Codex P2, round 6). A SHORT run of
                         them is a block too, when a paragraph sits above it: it underlines that
                         paragraph. *)
                      match thematic_break stripped with
                      | Some what -> Some what
                      | None -> setext_underline ~under_paragraph stripped)
                with
                | Some what ->
                    (* Honoured at depth too: nested under a bullet this is a quote inside the list
                       item, and folding it into the parent's prose would leave its text checked by
                       nothing. *)
                    bad lineno what;
                    close_all ()
                | None -> (
                    match !stack with
                    | [] ->
                        bad lineno
                          (if String.is_prefix stripped ~prefix:"|" then
                             "a table row indented four spaces or more, which Markdown renders as \
                              an indented code block rather than a table"
                           else
                             "an indented line continuing no bullet: nothing above it is an open \
                              list item")
                    | (b, texts) :: _ ->
                        if indent <> b.indent + 2 then
                          bad lineno
                            (Printf.sprintf
                               "an indented line at %d continuing the bullet at line %d, whose \
                                continuations sit at %d"
                               indent b.line (b.indent + 2))
                        else texts := stripped :: !texts)));
  close_all ();
  let bullets = List.rev !bullets in
  let terminator_findings =
    List.filter_map bullets ~f:(fun b ->
        if bullet_text_is_terminated b.text then None
        else
          let subject = bullet_subject b in
          Some
            (finding ~subject ~file ~line:b.line ~rule:rule_bullet_integrity
               (Printf.sprintf
                  "a bullet that does not end a sentence, so its tail may be elsewhere: %S ends \
                   \"…%s\" -- exempt it as %S if that ending is deliberate"
                  (String.prefix subject subject_display_length)
                  (String.suffix b.text 40) (subject_key ~file ~subject))))
  in
  { bullets; structure = List.rev !findings @ terminator_findings }

(** The bullets of a file, for callers that want only those. *)
let bullets ~file contents = (parse_file ~file contents).bullets

(** A code span or an HTML comment the file never closes. Left open, it makes every line below it
    inert -- which silences the rules over the rest of the file rather than failing them, so it is
    reported instead of assumed closed. *)
let unclosed_inert ~file contents =
  match (inert_by_line contents).unclosed with
  | In_text -> []
  (* An unmatched backtick run is literal text to a renderer, and the lexer now treats it that way
     -- nothing is hidden, so there is nothing to report. *)
  | In_code _ -> []
  | In_comment ->
      [
        finding ~file
          ~line:(List.length (lines contents))
          ~rule:rule_bullet_integrity
          "an HTML comment opened and never closed, so everything below it is commented out and no \
           rule can see it";
      ]
  | In_fence (c, n) ->
      [
        finding ~file
          ~line:(List.length (lines contents))
          ~rule:rule_bullet_integrity
          (Printf.sprintf
             "a %s fence that is never closed, so the rest of the file is inside it and unread"
             (String.make n c));
      ]

(** The constructs that HIDE text from a reader: a fenced block, and an HTML comment. Neither
    appears anywhere in the notes, and both are reported rather than silently skipped — the whole
    argument for a closed dialect is that a note wanting one has to teach this module what its lines
    mean first. They are read off the lexer's own record, so the report and the inert ranges can
    never disagree about where a construct began. *)
let hiding_constructs ~file contents =
  let scan = inert_by_line contents in
  List.map scan.fences ~f:(fun (line, marker) ->
      finding ~file ~line ~rule:rule_bullet_integrity
        (Printf.sprintf
           "a %s fenced block: the notes have no fenced code, and the lines inside one are not \
            bullets, table rows or prose -- teach this scan what they are before writing one"
           marker))
  @ List.map scan.comments ~f:(fun line ->
      finding ~file ~line ~rule:rule_bullet_integrity
        "an HTML comment: its text is invisible to a reader, so anything promoted inside one is \
         promoted nowhere")

(** Rule 1 over one file. *)
let check_structure ~file contents =
  let structural = (parse_file ~file contents).structure in
  let extra = hiding_constructs ~file contents @ unclosed_inert ~file contents in
  (* Ordered by document position, which is numeric: a string sort over ["f.md:10"] and ["f.md:3"]
     puts line 10 first. *)
  List.sort (structural @ extra) ~compare:(fun a b ->
      match String.compare a.file b.file with
      | 0 -> Option.compare Int.compare a.line b.line
      | c -> c)

(* ------------------------------------------------------------------ *)
(* Rule 3: table shape *)
(* ------------------------------------------------------------------ *)

type table = { start_line : int; rows : (int * string) list }

(** The table blocks of a file: maximal runs of lines whose trimmed form starts with a pipe. *)
let tables contents =
  let map = (inert_by_line contents).ranges in
  let rec go acc current = function
    | [] -> List.rev (match current with None -> acc | Some t -> t :: acc)
    | (lineno, line) :: rest ->
        (* A pipe-led line inside a code span or a comment is an example of a table, not one.
           Reading it as a block started one whose cells then parsed as inert, so a correct note
           carrying a table example FAILED table-shape (Codex P2, round 5) -- a false failure, the
           direction that gets a check switched off. *)
        if not (marker_is_text ~spans:(spans_at map lineno) line) then
          let acc = match current with None -> acc | Some t -> t :: acc in
          go acc None rest
        else if is_table_line line then
          let current =
            match current with
            | None -> Some { start_line = lineno; rows = [ (lineno, line) ] }
            | Some t -> Some { t with rows = t.rows @ [ (lineno, line) ] }
          in
          go acc current rest
        else
          let acc = match current with None -> acc | Some t -> t :: acc in
          go acc None rest
  in
  go [] None (lines contents)

(** Rule 3 over one file. *)
let check_tables ~file contents =
  let map = (inert_by_line contents).ranges in
  let cells_of line lineno = row_cells ~spans:(spans_at map lineno) line in
  let all = lines contents in
  let line_at n = List.Assoc.find all n ~equal:Int.equal in
  List.concat_map (tables contents) ~f:(fun t ->
      let report line msg = finding ~file ~line ~rule:rule_table_shape msg in
      let closed =
        List.filter_map t.rows ~f:(fun (lineno, line) ->
            match cells_of line lineno with
            | None ->
                Some
                  (report lineno
                     "a table row that does not close with a cell separator: a row cannot span \
                      physical lines, so this ends the table and truncates the row")
            | Some _ -> None)
      in
      if not (List.is_empty closed) then closed
      else
        let cells =
          List.map t.rows ~f:(fun (lineno, line) ->
              (lineno, Option.value_exn (cells_of line lineno)))
        in
        let shape =
          match cells with
          | [] | [ _ ] | [ _; _ ] ->
              [
                report t.start_line
                  (Printf.sprintf
                     "a table of %d line(s): a table is a header, a delimiter row and at least one \
                      data row"
                     (List.length cells));
              ]
          | (_, header) :: (delim_line, delim) :: _ ->
              let width = List.length header in
              let delim_finding =
                if is_delimiter_row delim then []
                else
                  [
                    report delim_line
                      "the line below a table's header is not a delimiter row, so the block above \
                       it is not a table";
                  ]
              in
              let width_findings =
                List.filter_map cells ~f:(fun (lineno, c) ->
                    if List.length c = width then None
                    else
                      Some
                        (report lineno
                           (Printf.sprintf "a row of %d cells in a table whose header has %d"
                              (List.length c) width)))
              in
              delim_finding @ width_findings
        in
        let after =
          let next = t.start_line + List.length t.rows in
          match line_at next with
          | Some l
            when (not (is_blank l))
                 && not (List.is_empty (pipes_outside_code ~spans:(spans_at map next) l)) ->
              [
                report next
                  "a cell separator on the line below a table: this reads as the tail of a row \
                   that was wrapped, which ends the table above it";
              ]
          | _ -> []
        in
        shape @ after)

(* ------------------------------------------------------------------ *)
(* Rules 2 and 4: the index against the files *)
(* ------------------------------------------------------------------ *)

type index_row = {
  row_line : int;
  link_text : string;
  target : string;
  anchor : string option;
  hooks : string list;
}
(** One parsed index row. [target] is as written, relative to the index's own directory; [anchor] is
    the fragment after ['#'], if any. *)

(** A code span's RENDERED content: the delimiter runs removed, and then one padding space from each
    side. The padding is how a span carries a literal backtick — [``` `` `foo` `` ```] renders as
    [`foo`] — and returning it as part of the hook made [index-agreement] reject a file that
    contains the hook exactly as rendered (Codex P2, round 9). A span of nothing but spaces keeps
    them, per the same rule. *)
let code_span_content s =
  let s = String.strip s ~drop:(Char.equal '`') in
  let n = String.length s in
  if
    n >= 2
    && Char.equal s.[0] ' '
    && Char.equal s.[n - 1] ' '
    && not (String.for_all s ~f:(Char.equal ' '))
  then String.sub s ~pos:1 ~len:(n - 2)
  else s

let backticked cell =
  List.filter_map (code_spans cell) ~f:(fun (start, stop) ->
      let s = code_span_content (String.sub cell ~pos:start ~len:(stop - start)) in
      if String.is_empty (String.strip s) then None else Some s)

(** GitHub's heading slug: lowercased, spaces to hyphens, other punctuation dropped — and
    UNDERSCORES KEPT, which is the half that matters here. The headings a note would anchor are
    identifiers (`ident_blacklist`, `promote_prec`), and GitHub anchors those as `#ident_blacklist`;
    rewriting the underscore rejected the correct anchor and accepted the wrong one (Codex P2, round
    1). Hyphens are likewise kept as themselves rather than re-derived. *)
let slug heading =
  String.lowercase heading |> String.to_list
  |> List.filter_map ~f:(fun c ->
      if Char.is_alphanum c || Char.equal c '_' || Char.equal c '-' then Some c
      else if Char.equal c ' ' then Some '-'
      else None)
  |> String.of_list

(** The anchor ids GitHub gives a file's headings, in document order — github-slugger's allocation:
    a heading takes its {!slug} unless an EARLIER id already holds it, and then the first
    [<slug>-<k>] no earlier id holds, [k] counting up from where that slug's last suffix left off.
    Comparing an anchor to each heading's bare slug rejected a valid pointer at the later of two
    same-titled sections (Codex P2, round 1 on lukstafi/ocannl-staging#811); counting only the bare
    slug's repeats then collided with a heading whose own title is [Foo-1] (round 2), so every
    candidate is checked against every id already emitted. *)
let heading_ids contents =
  let taken = Hashtbl.create (module String) in
  let last_suffix = Hashtbl.create (module String) in
  List.map (headings contents) ~f:(fun h ->
      let base = slug h in
      let rec allocate candidate =
        if not (Hashtbl.mem taken candidate) then candidate
        else
          let k = Option.value (Hashtbl.find last_suffix base) ~default:0 + 1 in
          Hashtbl.set last_suffix ~key:base ~data:k;
          allocate (Printf.sprintf "%s-%d" base k)
      in
      let id = allocate base in
      Hashtbl.set taken ~key:id ~data:();
      id)

(** A link's DESTINATION, separated from its optional title. [(../agent-notes.md "Agent notes")] is
    a perfectly ordinary link, and comparing the whole parenthesised text against the index filename
    reported a note unreachable over a link that navigates there (Codex P2, round 9). A destination
    containing spaces has to be written in angle brackets, which is the other form handled here. *)
let link_destination inside =
  let inside = String.strip inside in
  match (String.chop_prefix inside ~prefix:"<", String.index inside '>') with
  | Some _, Some close -> String.sub inside ~pos:1 ~len:(close - 1)
  | _ -> (
      match String.lfindi inside ~f:(fun _ c -> Char.equal c ' ' || Char.equal c '\t') with
      | Some i -> String.prefix inside i
      | None -> inside)

(** The navigable Markdown links of a line, outside inline code and HTML comments and excluding
    images. A substring test used to call a file reachable when backlink-like bytes sat in a code
    span, in a comment, or behind an image (Codex P2, round 1) — all three of which a reader cannot
    follow. *)
let markdown_links ?spans line =
  let spans = match spans with Some s -> s | None -> inert_of_line line in
  let n = String.length line in
  let rec go i acc =
    if i >= n then List.rev acc
    else if Char.equal line.[i] '[' && (not (in_any_span spans i)) && not (escaped_at line i) then
      (* An escaped bracket renders literally, so it opens no link. Same parity rule as pipes -- it
         was applied to one and not the other (Codex P2, round 5). *)
      let image = i > 0 && Char.equal line.[i - 1] '!' && not (escaped_at line (i - 1)) in
      (* The first UNESCAPED bracket: "[index\](...)" renders no link, and taking the escaped one
         counted a target that a reader cannot follow (Codex P2, round 7). Third site of the same
         parity rule, after pipes and opening brackets. *)
      let rec unescaped_close from =
        match String.index_from line from ']' with
        | Some j when escaped_at line j -> unescaped_close (j + 1)
        | other -> other
      in
      match unescaped_close i with
      | Some close
        when close + 1 < n
             && Char.equal line.[close + 1] '('
             && (not (in_any_span spans close))
             && not (in_any_span spans (close + 1)) -> (
          (* The first UNESCAPED parenthesis, for the same reason as the bracket above:
             "[index](../agent-notes.md#draft\)" finishes no link, and accepting the escaped one
             resolved to the index anyway once the fragment was dropped (Codex P2, round 8). Fourth
             site of the parity rule. *)
          let rec unescaped_rparen from =
            match String.index_from line from ')' with
            | Some j when escaped_at line j -> unescaped_rparen (j + 1)
            | other -> other
          in
          match unescaped_rparen (close + 1) with
          | Some rparen when not (in_any_span spans rparen) ->
              let text = String.sub line ~pos:(i + 1) ~len:(close - i - 1) in
              let target =
                link_destination (String.sub line ~pos:(close + 2) ~len:(rparen - close - 2))
              in
              let acc = if image then acc else (text, target) :: acc in
              go (rparen + 1) acc
          | _ -> go (i + 1) acc)
      | _ -> go (i + 1) acc
    else go (i + 1) acc
  in
  go 0 []

(** Every navigable link of a whole file, read with the paragraph-aware span map so that a link on
    the middle line of a multiline code span is not one. *)
let links_of contents =
  let map = (inert_by_line contents).ranges in
  List.concat_map (lines contents) ~f:(fun (lineno, l) ->
      markdown_links ~spans:(spans_at map lineno) l)

(** Where a relative link written IN [from_file] actually points, as a path in the same space as the
    scan's file keys. [from_file] is a file, so the link resolves against its DIRECTORY; [".."] pops
    a segment and ["."] is dropped, exactly as a browser or GitHub would.

    This exists because a target's correct spelling depends on how deep the file is, and hard-coding
    the one-level spelling made the backlink rule demand [`../agent-notes.md`] of a note in a
    subdirectory — which from there resolves to [docs/agent-notes/agent-notes.md], a file that does
    not exist. So nested notes could only pass by carrying an unusable link (Codex P2, round 2). *)
let resolve_link ~from_file target : string option =
  let dir =
    match List.rev (String.split from_file ~on:'/') with _ :: rest -> List.rev rest | [] -> []
  in
  let target = List.hd_exn (String.split target ~on:'#') in
  let step acc segment =
    match acc with
    | None -> None
    | Some acc ->
        if String.equal segment "." || String.is_empty segment then Some acc
        else if String.equal segment ".." then
          (* Above the root is a DISTINCT answer, not the root. Clamping made agent-notes/a.md's
             "../../agent-notes.md" resolve to the index, so a link pointing at the repository root
             passed as a backlink (Codex P2, round 3). *)
          match acc with
          | _ :: rest -> Some rest
          | [] -> None
        else Some (segment :: acc)
  in
  List.fold (String.split target ~on:'/') ~init:(Some (List.rev dir)) ~f:step
  |> Option.map ~f:(fun acc -> List.rev acc |> String.concat ~sep:"/")

(** [[text](target)] filling the whole cell, and nothing else. *)
let parse_link cell =
  let cell = String.strip cell in
  match (String.chop_prefix cell ~prefix:"[", String.chop_suffix cell ~suffix:")") with
  | Some _, Some _ -> (
      let body = String.sub cell ~pos:1 ~len:(String.length cell - 2) in
      match String.substr_index body ~pattern:"](" with
      | None -> None
      | Some i ->
          let text = String.prefix body i in
          let target = String.drop_prefix body (i + 2) in
          if String.contains text ']' || String.contains target ')' then None
          else Some (text, target))
  | _ -> None

(** Whether a table block is one: header, delimiter, at least one data row, every line closing with
    a separator, and one width throughout. The same questions {!check_tables} answers with findings,
    asked as a yes or no -- so the two cannot disagree about what parses. *)
let table_parses ~contents t =
  let map = (inert_by_line contents).ranges in
  let cells =
    List.map t.rows ~f:(fun (lineno, line) -> row_cells ~spans:(spans_at map lineno) line)
  in
  match cells with
  | Some header :: Some delim :: (_ :: _ as rest) ->
      is_delimiter_row delim
      && List.for_all (Some header :: Some delim :: rest) ~f:(function
        | Some c -> List.length c = List.length header
        | None -> false)
  | _ -> false

(** The index's rows, or the one finding that says its table did not parse at all.

    The distinction is the point. A wrapped row ends the table, which leaves the index looking like
    TWO tables with most of its rows outside both -- and asking the hook and reachability rules
    about that produces one spurious "unreachable" line per notes file, burying the single finding
    that says what actually happened. So a refusal comes back as [Error] and stops those rules,
    which then report that they could not be evaluated rather than reporting twelve falsehoods. It
    cannot hide a real defect: the refusal is itself a failure, and the table rule has already named
    the line. *)
let index_rows ~file contents =
  match tables contents with
  | [] -> Error (file_finding ~file ~rule:rule_index_agreement "no table: the index is a table")
  | _ :: _ :: _ ->
      Error
        (file_finding ~file ~rule:rule_index_agreement
           "more than one table: the index is one table, so a row that ends it puts most of the \
            index outside both")
  | [ t ] when not (table_parses ~contents t) ->
      (* One block, but not a table: a row that does not close, a ragged width, a missing delimiter.
         Extracting rows anyway dropped the bad one and then reported EVERY notes file as an orphan,
         burying the one actionable finding under a dozen spurious ones -- the same cascade the
         two-table refusal above exists to prevent, through the door it left open (Codex P2, round
         6). *)
      Error
        (file_finding ~file ~rule:rule_index_agreement
           "a table that does not parse, so its rows cannot be read: fix what table-shape reports \
            about it first")
  | [ t ] ->
      let map = (inert_by_line contents).ranges in
      let data = match t.rows with _ :: _ :: rest -> rest | _ -> [] in
      let rows, findings =
        List.partition_map data ~f:(fun (lineno, line) ->
            match row_cells ~spans:(spans_at map lineno) line with
            (* Exactly two: the index's schema is "| file link | coverage prose |", and folding any
               further cells into the hooks accepted a wider table silently (Codex P2, round 5). *)
            | Some [ link; hooks ] -> (
                let hooks = hooks in
                match parse_link link with
                | None ->
                    Either.Second
                      (finding ~file ~line:lineno ~rule:rule_index_agreement
                         (Printf.sprintf "a row whose first cell is not a link to a notes file: %S"
                            link))
                | Some (link_text, target) ->
                    let target, anchor =
                      match String.lsplit2 target ~on:'#' with
                      | Some (t, a) -> (t, Some a)
                      | None -> (target, None)
                    in
                    Either.First
                      { row_line = lineno; link_text; target; anchor; hooks = backticked hooks })
            | Some cells ->
                Either.Second
                  (finding ~file ~line:lineno ~rule:rule_index_agreement
                     (Printf.sprintf
                        "a row of %d cells: the index's schema is \"| file link | coverage prose \
                         |\""
                        (List.length cells)))
            | None ->
                Either.Second
                  (finding ~file ~line:lineno ~rule:rule_index_agreement "a row with no cells"))
      in
      Ok (rows, findings)

(** Rules 2 and 4. [files] is every notes file, keyed by its path relative to the index's directory
    — ["agent-notes/build-and-test.md"] — which is what an index link spells. [index_file] is the
    index's own path relative to the same place, as it appears in the files' backlinks. *)
let check_index ~index_file ~index_contents ~(files : (string * string) list) =
  match index_rows ~file:index_file index_contents with
  | Error refusal ->
      [
        refusal;
        file_finding ~file:index_file ~rule:rule_reachability
          "not evaluated: the index's table does not parse, so which files it reaches is unknown \
           -- fix the table first";
      ]
  | Ok (rows, row_findings) ->
      let per_row =
        List.concat_map rows ~f:(fun row ->
            let report msg =
              finding ~file:index_file ~line:row.row_line ~rule:rule_index_agreement msg
            in
            let basename = List.last_exn (String.split row.target ~on:'/') in
            let text_finding =
              if String.equal row.link_text basename then []
              else
                [
                  report
                    (Printf.sprintf
                       "link text %S naming the file %S: the index reads as a file list"
                       row.link_text basename);
                ]
            in
            match List.Assoc.find files row.target ~equal:String.equal with
            | None ->
                text_finding
                @ [ report (Printf.sprintf "a link to %S, which is not a notes file" row.target) ]
            | Some contents ->
                let anchor_finding =
                  match row.anchor with
                  | None -> []
                  | Some a ->
                      if List.mem (heading_ids contents) a ~equal:String.equal then []
                      else
                        [
                          report
                            (Printf.sprintf "an anchor #%s that %s has no heading for" a row.target);
                        ]
                in
                let hook_findings =
                  List.filter_map row.hooks ~f:(fun hook ->
                      if String.is_substring contents ~substring:hook then None
                      else
                        Some
                          (report
                             (Printf.sprintf
                                "the hook `%s`, which does not occur in %s: the index is what a \
                                 lookup greps first, so a hook absent from its target is a dead \
                                 end"
                                hook row.target)))
                in
                text_finding @ anchor_finding @ hook_findings)
      in
      let targets = List.map rows ~f:(fun r -> r.target) in
      let duplicates =
        List.filter_map rows ~f:(fun r ->
            let earlier =
              List.filter rows ~f:(fun o ->
                  String.equal o.target r.target && o.row_line < r.row_line)
            in
            if List.is_empty earlier then None
            else
              Some
                (finding ~file:index_file ~line:r.row_line ~rule:rule_reachability
                   (Printf.sprintf
                      "a second row for %s: one row per file, or a lookup has two answers" r.target)))
      in
      let orphans =
        List.filter_map files ~f:(fun (name, _) ->
            if List.mem targets name ~equal:String.equal then None
            else
              Some
                (file_finding ~file:name ~rule:rule_reachability
                   "a notes file no index row links to: unreachable from the one place a lookup \
                    starts"))
      in
      let backlinks =
        List.filter_map files ~f:(fun (name, contents) ->
            (* A link, not a byte sequence -- and one that RESOLVES to the index from where this
               file sits, so a note one directory deeper is asked for the spelling that works from
               there rather than for the spelling that works one level up. *)
            if
              List.exists (links_of contents) ~f:(fun (_, t) ->
                  Option.value_map (resolve_link ~from_file:name t) ~default:false
                    ~f:(String.equal index_file))
            then None
            else
              Some
                (file_finding ~file:name ~rule:rule_reachability
                   (Printf.sprintf
                      "no navigable link back to the index: a link resolving to %S from here \
                       (outside code spans, HTML comments and images), which from %S is written %S"
                      index_file name
                      (String.concat
                         (List.init
                            (List.length (String.split name ~on:'/') - 1)
                            ~f:(fun _ -> "../"))
                      ^ index_file))))
      in
      row_findings @ per_row @ duplicates @ orphans @ backlinks

(* ------------------------------------------------------------------ *)
(* Rule 5: no bullet repeated *)
(* ------------------------------------------------------------------ *)

(** Two bullets whose normalized texts agree on this many leading characters are reported as near
    duplicates. Long enough that no two of the corpus's 177 bullets collide, short enough that a
    fact re-promoted with its tail reworded is still caught. *)
let near_duplicate_prefix = 60

(** Rule 5 over every bullet of the notes, the index's own included. Exact repetition is reported
    first and suppresses the near-duplicate report for the same pair, so one fact promoted twice is
    one finding. *)
let check_repetition (bullets : bullet list) =
  let exact = Hashtbl.create (module String) in
  let prefix = Hashtbl.create (module String) in
  List.concat_map bullets ~f:(fun b ->
      let key = normalize b.text in
      let report rule msg = finding ~file:b.file ~line:b.line ~rule msg in
      let where (o : bullet) = Printf.sprintf "%s:%d" o.file o.line in
      match Hashtbl.find exact key with
      | Some (o : bullet) ->
          [
            report rule_no_repetition
              (Printf.sprintf "a bullet repeating %s verbatim: \"%s…\"" (where o)
                 (String.prefix key 60));
          ]
      | None -> (
          Hashtbl.set exact ~key ~data:b;
          let pkey = String.lowercase (String.prefix key near_duplicate_prefix) in
          if String.length pkey < near_duplicate_prefix then []
          else
            match Hashtbl.find prefix pkey with
            | Some (o : bullet) ->
                [
                  report rule_no_repetition
                    (Printf.sprintf
                       "a bullet opening exactly as %s does: \"%s…\" — one fact promoted twice is \
                        a fact that will be updated once"
                       (where o)
                       (String.prefix key near_duplicate_prefix));
                ]
            | None ->
                Hashtbl.set prefix ~key:pkey ~data:b;
                []))

(* ------------------------------------------------------------------ *)
(* Rule 6: numeric citations name their repository *)
(* ------------------------------------------------------------------ *)

(** Whether [#] at [i] opens an unqualified numeric citation. A preceding identifier is code or a
    repository qualifier ([node#12], [staging#12], [ahrefs/ocannl#12]), except for the ambiguous
    work labels [PR#12] and [issue#12]. A second hash makes a heading marker. The digit run must end
    at an identifier boundary, so [#12abc] is not mistaken for a citation. *)
let bare_citation_at line i =
  let n = String.length line in
  if i + 1 >= n || (not (Char.is_digit line.[i + 1])) || (i > 0 && Char.equal line.[i - 1] '#') then
    None
  else
    let stop = ref (i + 1) in
    while !stop < n && Char.is_digit line.[!stop] do
      Int.incr stop
    done;
    if !stop < n && (Char.is_alphanum line.[!stop] || Char.equal line.[!stop] '_') then None
    else
      let identifier_char c = Char.is_alphanum c || Char.equal c '_' in
      let start = ref i in
      while !start > 0 && identifier_char line.[!start - 1] do
        Int.decr start
      done;
      if !start = i then Some !stop
      else
        let prefix = String.sub line ~pos:!start ~len:(i - !start) |> String.lowercase in
        if List.mem [ "pr"; "issue" ] prefix ~equal:String.equal then Some !stop else None

(** Bare [#NNN] citations outside the inert regions of the notes. The same paragraph-aware lexer
    used by the structural rules owns code spans, fenced blocks and comments, so this rule cannot
    acquire a second, drifting interpretation of Markdown. *)
let check_citations ~file contents =
  let inert = (inert_by_line contents).ranges in
  List.concat_map (lines contents) ~f:(fun (lineno, line) ->
      let spans = spans_at inert lineno in
      let rec find i acc =
        if i >= String.length line then List.rev acc
        else if Char.equal line.[i] '#' && not (in_any_span spans i) then
          match bare_citation_at line i with
          | Some stop ->
              let citation = String.sub line ~pos:i ~len:(stop - i) in
              let f =
                finding ~file ~line:lineno ~rule:rule_qualified_citations
                  (Printf.sprintf
                     "an unqualified GitHub citation %s: write staging#NNN for a staging PR, \
                      gh-ocannl-NNN for a codebase-fact issue, or ahrefs/ocannl#NNN for another \
                      upstream issue mention"
                     citation)
              in
              find stop (f :: acc)
          | None -> find (i + 1) acc
        else find (i + 1) acc
      in
      find 0 [])

(* ------------------------------------------------------------------ *)
(* Rule 7: the agent guide's anchored pointers resolve *)
(* ------------------------------------------------------------------ *)

type pointer = { pointer_line : int; path : string; anchor : string }
(** One [<path>.md#<anchor>] occurrence in the guide, as written: [path] includes the [.md]. *)

let path_char c = Char.is_alphanum c || List.mem [ '_'; '-'; '.'; '/' ] c ~equal:Char.equal
let anchor_char c = Char.is_alphanum c || Char.equal c '_' || Char.equal c '-'

(** A line as it RENDERS, for the purpose of finding pointers in it: a backslash escaping ASCII
    punctuation outside a code span is dropped, since a pointer whose hash or underscore is escaped
    with a backslash displays as the pointer it spells (Codex P2, round 2 on
    lukstafi/ocannl-staging#811); inside a code span a backslash is literal and stays. Each rendered
    character keeps its position in the source line, so a caller can still ask the lexer about it.
*)
let rendered_line ~code line =
  let n = String.length line in
  let rec go i acc =
    if i >= n then List.rev acc
    else if
      Char.equal line.[i] '\\'
      && i + 1 < n
      && Char.is_print line.[i + 1]
      && (not (Char.is_alphanum line.[i + 1]))
      && (not (Char.equal line.[i + 1] ' '))
      && not (in_any_span code i)
    then go (i + 2) ((line.[i + 1], i + 1) :: acc)
    else go (i + 1) ((line.[i], i) :: acc)
  in
  go 0 []

(** Every [<path>.md#<anchor>] in [contents] that a reader sees, code spans and fenced blocks
    included: a pointer set in backticks is still a pointer. One inside an HTML comment renders
    nowhere, so it is not read -- neither checked nor counted toward the live scan's floor (Codex
    P2, round 1 on lukstafi/ocannl-staging#811). The text searched is {!rendered_line}'s, so an
    escaped spelling is found as the pointer it displays. The path is the maximal run of path
    characters before [.md#] and the anchor the maximal run of slug characters after it; a
    placeholder such as [<note>.md#<slug>] has an empty one and is not a pointer. *)
let guide_pointers contents =
  let scan = inert_by_line contents in
  List.concat_map (lines contents) ~f:(fun (lineno, line) ->
      let hidden = spans_at scan.comment_ranges lineno in
      let code =
        List.filter (spans_at scan.ranges lineno) ~f:(fun r ->
            not (List.mem hidden r ~equal:(fun (a, b) (c, d) -> a = c && b = d)))
      in
      let chars = rendered_line ~code line in
      let text = String.of_list (List.map chars ~f:fst) in
      let source = Array.of_list (List.map chars ~f:snd) in
      let n = String.length text in
      String.substr_index_all text ~may_overlap:false ~pattern:".md#"
      |> List.filter ~f:(fun i -> not (in_any_span hidden source.(i)))
      |> List.filter_map ~f:(fun i ->
          let start = ref i in
          while !start > 0 && path_char text.[!start - 1] do
            Int.decr start
          done;
          let stop = ref (i + 4) in
          while !stop < n && anchor_char text.[!stop] do
            Int.incr stop
          done;
          let path = String.sub text ~pos:!start ~len:(i + 3 - !start) in
          let anchor = String.sub text ~pos:(i + 4) ~len:(!stop - i - 4) in
          if String.equal path ".md" || String.is_empty anchor then None
          else Some { pointer_line = lineno; path; anchor }))

(** The notes file a guide pointer names, keyed as {!check_index} keys [files], or [None] when the
    path points outside the notes. A bare basename is a note, which is how the guide spells them; a
    path is read relative to the repository root, where [docs/agent-notes/] holds the notes and
    [docs/agent-notes.md] is the index. A leading [./] makes a path of a bare name, which is how a
    root file such as [./CHANGES.md#…] is spelled. *)
let pointer_target path =
  let from_root path =
    match String.chop_prefix path ~prefix:"docs/" with
    | Some rest when String.is_prefix rest ~prefix:"agent-notes/" -> Some rest
    | Some rest when String.equal rest "agent-notes.md" -> Some rest
    | _ -> None
  in
  match String.chop_prefix path ~prefix:"./" with
  | Some rest -> from_root rest
  | None -> if String.mem path '/' then from_root path else Some ("agent-notes/" ^ path)

(** Rule 7 over the agent guide. [files] is keyed as {!check_index} describes; the index is looked
    up beside them, so a pointer at [docs/agent-notes.md#…] is checked against the index's headings.
*)
let check_guide ~guide_file ~guide_contents ~index_file ~index_contents
    ~(files : (string * string) list) =
  let known = (index_file, index_contents) :: files in
  List.filter_map (guide_pointers guide_contents) ~f:(fun p ->
      let report msg =
        Some (finding ~file:guide_file ~line:p.pointer_line ~rule:rule_guide_anchors msg)
      in
      match pointer_target p.path with
      | None -> None
      | Some target -> (
          match List.Assoc.find known target ~equal:String.equal with
          | None ->
              report
                (Printf.sprintf
                   "%s#%s names %s, which is not a notes file: a bare basename is read as a note \
                    under docs/agent-notes/, so spell any other file with its directory, ./ for \
                    one at the root"
                   p.path p.anchor ("docs/" ^ target))
          | Some contents ->
              if List.mem (heading_ids contents) p.anchor ~equal:String.equal then None
              else
                report
                  (Printf.sprintf
                     "%s#%s names a heading %s does not have: the rule pointing here has lost its \
                      mechanism -- restore the heading or re-point the rule (grep -n '^#' %s lists \
                      them)"
                     p.path p.anchor ("docs/" ^ target) ("docs/" ^ target))))

(* ------------------------------------------------------------------ *)
(* The whole scan *)
(* ------------------------------------------------------------------ *)

(** The findings whose [rule] is not one of {!rules}: a rule added to the scan and not named there.

    Grouping the report by {!rules} used to be a filter, which DROPPED such a finding -- the new
    rule fired, and its findings went nowhere. The order is still {!rules}, but nothing is lost:
    {!in_rule_order} puts an unnamed rule's findings last, and both consumers claim this census is
    empty, so the omission fails a run instead of silently costing coverage. *)
let unnamed_rule_findings found = List.filter found ~f:(fun f -> not (names_a_rule f.rule))

(** Findings grouped by rule in {!rules} order, and within a rule in the order they were found;
    findings carrying a rule {!rules} does not name keep their order, at the end. *)
let in_rule_order found =
  let named, unnamed = List.partition_tf found ~f:(fun f -> names_a_rule f.rule) in
  List.concat_map rules ~f:(fun r -> List.filter named ~f:(fun f -> String.equal f.rule r))
  @ unnamed

(** Every rule, over an index and the files it indexes, and over the agent guide's pointers into
    them when [guide] — [(name, contents)] — is given. Findings come back in {!in_rule_order} --
    grouped by rule in {!rules} order, and within a rule in file and line order, with any finding
    carrying an unnamed rule last. [files] is keyed as {!check_index} describes. *)
let check_all ?guide ~index_file ~index_contents ~(files : (string * string) list) () =
  let all = (index_file, index_contents) :: files in
  let structure = List.concat_map all ~f:(fun (file, c) -> check_structure ~file c) in
  let table = List.concat_map all ~f:(fun (file, c) -> check_tables ~file c) in
  let index = check_index ~index_file ~index_contents ~files in
  let bullets = List.concat_map all ~f:(fun (file, c) -> bullets ~file c) in
  let repetition = check_repetition bullets in
  let citations = List.concat_map all ~f:(fun (file, c) -> check_citations ~file c) in
  let guide =
    Option.value_map guide ~default:[] ~f:(fun (guide_file, guide_contents) ->
        check_guide ~guide_file ~guide_contents ~index_file ~index_contents ~files)
  in
  let found = structure @ table @ index @ repetition @ citations @ guide in
  (bullets, in_rule_order found)
