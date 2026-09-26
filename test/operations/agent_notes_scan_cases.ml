(** What {!Test_utils.Agent_notes_scan} calls a defect, on input built to break it.

    The live-tree scan next door ([agent_notes_structure]) is green whenever the notes are intact,
    which is most days — and a check that is green because it sees nothing looks exactly like a
    check that is green because everything holds. So each of the rules is exercised here on
    synthetic notes: a violation the rule must flag, and beside it the nearest legitimate text it
    must NOT, since a rule that fires on ordinary prose gets turned off rather than obeyed.

    The fixtures are written against the shapes the real defects took (lukstafi/ocannl-staging#406):
    a bullet cut mid-sentence by a merge with its tail stranded inside a later bullet, an index hook
    naming a file that does not carry it, a table row wrapped across two physical lines. Each case
    names the findings the scan should produce, as ["<rule> @ <where>"], in order. *)

open Base
open Stdio
module Notes = Test_utils.Agent_notes_scan

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt
let render (f : Notes.finding) = Printf.sprintf "%s @ %s" f.Notes.rule f.Notes.where

(* ------------------------------------------------------------------ *)
(* Rule 1: bullet integrity, over a single file *)
(* ------------------------------------------------------------------ *)

(* Each case is a file body and the findings expected from [check_structure] alone. Line 1 of every
   body is the heading, so the reported line numbers read off the fixture directly. *)
let structure_cases =
  [
    ( "a flat list of finished bullets",
      "# Title\n\n\
       - One fact, stated.\n\
       - Another, with a continuation line that\n\
      \  wraps and then ends.\n",
      [] );
    (* The shape a merge leaves behind: an incoming bullet inserted after the FIRST line of its hunk
       context instead of the last, so the bullet above it stops mid-word. The tail lands inside a
       later bullet, where nothing marks it -- the truncation above is the decidable half. *)
    ( "a bullet a merge cut in half",
      "# Title\n\n\
       - A fact whose closing sentence stops mid-senten\n\
       - An unrelated incoming bullet, complete in itself.\n\
      \  ce and the rest of it, stranded here.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ("a bullet ending in a colon", "# Title\n\n- The three of them are:\n", []);
    ( "a bullet ending in a parenthesis after the period",
      "# Title\n\n- A fact with its issue named (gh-ocannl-406.)\n",
      [] );
    ("a bullet ending in bold", "# Title\n\n- A fact that ends **emphatically.**\n", []);
    ("a bullet ending in markup behind a space", "# Title\n\n- A fact ends. `\n", []);
    ("a bullet ending in a code span", "# Title\n\n- A fact naming `Ops.promote_prec`.\n", []);
    ( "a bullet ending in an identifier and no punctuation",
      "# Title\n\n- A fact that trails off into `Ops.promote_prec`\n",
      [ "bullet-integrity @ f.md:3" ] );
    (* One nesting level is what the notes use; its continuations sit two deeper again. *)
    ( "a nested bullet with its own continuations",
      "# Title\n\n\
       - A fact with parts, as follows:\n\
      \  - the first part, which\n\
      \    wraps across a line.\n\
      \  - the second part.\n\
       - The next fact.\n",
      [] );
    (* Findings come out in document order, parent before child, however the list nests. *)
    ( "a nested list where parent and child are both cut short",
      "# Title\n\n\
       - A parent that stops mid-senten\n\
      \  - a child that stops too\n\
       - A finished fact.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4" ] );
    ( "an indented line at the parent's depth while a nested bullet is open",
      "# Title\n\n- A fact with parts:\n  - the first part.\n  more of the first fact.\n",
      [ "bullet-integrity @ f.md:5" ] );
    (* A blank line inside a list item does NOT end the list: what follows, indented to the item's
       continuation depth, is a second PARAGRAPH of the same bullet. This is the shape a note on
       master arrived in while this PR was in review, and closing the list there reported nine
       findings against somebody else's correctly written work. *)
    ( "a second paragraph of a bullet, after a blank line",
      "# Title\n\n- A finished fact.\n\n  A second paragraph of the same bullet.\n",
      [] );
    ( "a second paragraph that is itself cut short",
      "# Title\n\n- A finished fact.\n\n  A second paragraph that stops mid-senten\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an indented line continuing nothing",
      "# Title\n\nPreamble prose.\n\n  a line indented under no bullet at all.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a continuation indented past its bullet",
      "# Title\n\n- A fact that\n    continues four deep.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4" ] );
    ( "a nested bullet under no bullet at all",
      "# Title\n\nPreamble prose.\n\n  - a bullet indented under nothing.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a bullet written with a star",
      "# Title\n\n* A fact written with the other marker.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ("an empty bullet", "# Title\n\n-\n- A real fact.\n", [ "bullet-integrity @ f.md:3" ]);
    ( "a tab in the indentation",
      "# Title\n\n- A fact that\n\tcontinues after a tab.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4" ] );
    (* Prose paragraphs at column zero are the abstract and the backlink, and they close the list
       above them rather than continuing it. *)
    (* Round 1's family: four constructs that fell through to "prose", which is not checked -- so
       the text inside them got no termination, no repetition and no table rule at all, and the
       golden stayed green over it. Recognition is closed now, and the fallthrough is a finding. *)
    ( "an ordered item at column zero",
      "# Title\n\n1. A fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an ordered item written with a parenthesis",
      "# Title\n\n1) A fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    (* A line wholly inside a code span or a comment is an example. Parsing it invents findings
       about text nobody sees -- and can report the closing delimiter as an illegal lazy
       continuation. *)
    ( "a bullet-looking line inside a multiline code span",
      "# Title\n\n- A fact showing the shape `\n  - not a bullet\n  ` in passing.\n",
      [] );
    ( "a bullet-looking line inside an HTML comment",
      "# Title\n\n<!-- draft:\n- not a bullet yet\n-->\n\n- A fact.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a table-looking line inside a multiline code span",
      "# Title\n\n- A fact showing the shape `\n  | not | a table |\n  ` in passing.\n",
      [] );
    (* A backtick fence's info string may not contain a backtick, which is what keeps an inline
       triple-backtick span from opening a block. *)
    (* A marker inside the code that CLOSES a span is not a marker. The line is not wholly inert --
       prose follows the closer -- so the wholly-inert skip does not reach it. *)
    ( "a bullet marker inside the code closing a span",
      "# Title\n\nProse `\n- sample` afterward.\n",
      [] );
    ( "a marker inside a closing span, under an open bullet",
      "# Title\n\n- A fact showing `\n  - sample` in passing.\n",
      [] );
    ( "a long digit run is prose, not an ordered item",
      "# Title\n\n- A fact.\n\n1234567890. That is an identifier, not a list.\n",
      [] );
    ( "nine digits is still an ordered item",
      "# Title\n\n123456789. A fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a prose line that is a complete triple-backtick span",
      "# Title\n\n```foo``` is how it is written.\n",
      [] );
    ("a real fence is still a fence", "# Title\n\n```\ncode\n```\n", [ "bullet-integrity @ f.md:3" ]);
    ( "an autolink at column zero is not an HTML block",
      "# Title\n\n<https://example.com> is the reference.\n",
      [] );
    ( "a comparison at column zero is not an HTML block either",
      "# Title\n\n<= 8 lanes is the threshold.\n",
      [] );
    ( "a real HTML block still is one",
      "# Title\n\n<div class=\"x\">\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a thematic break nested under a bullet",
      "# Title\n\n- A fact:\n  ---\n  More.\n",
      [ "bullet-integrity @ f.md:4"; "bullet-integrity @ f.md:5" ] );
    ( "an ordered item written with a tab",
      "# Title\n\n1.\tA fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a star marker written with a tab",
      "# Title\n\n*\tA fact written with the other marker.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a dash marker written with a tab",
      "# Title\n\n-\tA fact written with a tab.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an ordered item indented under a bullet",
      "# Title\n\n- A fact with parts:\n  1. the first part.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a number that merely opens a sentence",
      "# Title\n\n- 296 stanzas were placed against a floor of 295.\n",
      [] );
    ( "a fenced block, whose contents are not bullets",
      "# Title\n\n\
       - A fact.\n\n\
       ```\n\
       - not a bullet, an example\n\
       | not | a table |\n\
       ```\n\n\
       -        Another fact.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a fence that is never closed",
      "# Title\n\n- A fact.\n\n~~~\n- not a bullet\n",
      [ "bullet-integrity @ f.md:5"; "bullet-integrity @ f.md:7" ] );
    ("a block quote", "# Title\n\n> Quoted prose.\n", [ "bullet-integrity @ f.md:3" ]);
    ( "an unmatched backtick run is literal text, not an unclosed span",
      "# Title\n\n- A fact naming `Ops.promote_prec.",
      [] );
    ( "an HTML comment the file never closes",
      "# Title\n\nProse <!-- and then nothing.\n\n- A fact.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:6" ] );
    (* Markdown honours a quote marker at every depth, so nested under a bullet this is a quote
       inside the list item -- not part of the bullet's prose, and checked by nothing if folded
       in. *)
    ( "a block quote nested under a bullet",
      "# Title\n\n- A fact:\n  > Quoted guidance.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "an index whose rows drifted four spaces right",
      "# Title\n\n    | File | Covers |\n    | --- | --- |\n    | a | b |\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4"; "bullet-integrity @ f.md:5" ] );
    ( "prose at column zero directly under a bullet",
      "# Title\n\n- A fact follows:\nthe omitted continuation.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a lazy continuation whose first column is inert",
      "# Title\n\n- A fact ends. `\ncode` continues.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a blank line makes the same prose a paragraph",
      "# Title\n\n- A fact follows:\n\nAn ordinary paragraph.\n",
      [] );
    ( "a heading written without a space after the hashes",
      "# Title\n\n##ident_blacklist\n\n- A fact.\n",
      [ "bullet-integrity @ f.md:3" ] );
    (* A hash on a wrapped line is an issue or a pull request far more often than a botched heading,
       and the notes cite those constantly. Reading one as a heading closed the list under it too,
       so the line below the citation was reported as an orphan continuation as well -- two findings
       on prose that renders exactly as written (lukstafi/ocannl-staging#598). The control beside it
       keeps the exception to the digit: a hash against a WORD is still the shape that carries a
       dead anchor. *)
    ( "a pull request cited at a continuation's first column",
      "# Title\n\n\
       - A fact about the review loop that\n\
      \  #598's rounds settled, and about what\n\
      \  they left behind.\n",
      [] );
    ( "a heading written against a word at a continuation's first column",
      "# Title\n\n- A fact, stated.\n  #Title is not a heading here.\n",
      [ "bullet-integrity @ f.md:4" ] );
    (* The exemption is to a citation, not to a leading digit: a title that starts with a number is
       exactly the malformed heading the rule exists for, and reading the digit alone would have
       retired the rule for every such title (Codex P2, round 1). What separates them is where the
       number ends -- against a letter it is a word, and the line is a heading missing its space. *)
    ( "a numeric title written without its space is still a heading",
      "# Title\n\n- A fact, stated.\n  #3D convolution, as written.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a citation ending against a letter is a word, not a citation",
      "# Title\n\n- A fact, stated.\n  #598abc is no reference.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "two hashes against a number is not a citation either",
      "# Title\n\n- A fact, stated.\n  ##598 as a heading nobody writes.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a citation ending at a comma is still a citation",
      "# Title\n\n- A fact about the loop that\n  #598, and its successors, settled.\n",
      [] );
    (* The block-quote marker's space is OPTIONAL, and block structure is settled before any of the
       line reads as arithmetic -- so a comparison wrapped onto a line's first visible column is a
       quote whatever follows the operator. The requirement therefore lands on the PROSE, and what
       the controls pin is that rule: reported wherever the operator opens the line, silent wherever
       the note keeps it off the first visible column (gh-ocannl-714). *)
    ( "a comparison wrapped onto a continuation's first column is a quote",
      "# Title\n\n- A fact about widths:\n  >= 8, and about `dune build`.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a comparison written without its own space is a quote too",
      "# Title\n\n- A fact about dune:\n  >=3.20, which generates the alias.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a quote written without the space after its marker",
      "# Title\n\n>=Quoted prose.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a quote written without its space, nested under a bullet",
      "# Title\n\n- A fact:\n  >= Quoted guidance.\n",
      [ "bullet-integrity @ f.md:4" ] );
    (* The nearest legitimate text: the same comparison written the two ways that keep the operator
       out of the marker position, which is what the finding above asks its author to do. *)
    ( "a comparison inside the line is not a quote",
      "# Title\n\n- A fact about widths that are >= 8 lanes.\n",
      [] );
    ( "a comparison rewrapped to keep its operator off the first column",
      "# Title\n\n- A fact about widths that\n  are >= 8 lanes, and about `dune build`.\n",
      [] );
    ( "a comparison inside a code span at the first column is not a quote",
      "# Title\n\n- A fact about widths\n  `>= 8` lanes, stated in passing.\n",
      [] );
    ( "an HTML block at column zero",
      "# Title\n\n<details><summary>x</summary>\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an angle bracket inside a continuation is ordinary text",
      "# Title\n\n- A fact about widths\n  <= 8, and about `dune build <that target>`.\n",
      [] );
    ( "a thematic break",
      "# Title\n\n- A fact.\n\n---\n\n- Another fact.\n",
      [ "bullet-integrity @ f.md:5" ] );
    (* A setext underline needs ONE marker, not the three a thematic break needs, so "--" and "=="
       used to fall through to prose while a renderer made the line above them a heading -- an
       anchor the heading rules never see (gh-ocannl-714). What makes it a heading is the paragraph
       ABOVE it, and each of the four things that is not a paragraph gets its own control below. *)
    ( "a setext underline at column zero",
      "# Title\n\nAn abstract paragraph.\n--\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a setext underline written with equals signs",
      "# Title\n\nAn abstract paragraph.\n==\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a setext underline of a single marker",
      "# Title\n\nAn abstract paragraph.\n=\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a setext underline nested under a bullet",
      "# Title\n\n- A fact:\n  --\n  More.\n",
      [ "bullet-integrity @ f.md:4"; "bullet-integrity @ f.md:5" ] );
    (* The nearest legitimate text, and the one the notes actually contain: a wrapped line opening
       with the ASCII dash this project writes for an em-dash. It is not a line of markers, so it is
       not an underline. *)
    ( "a dash opening a continuation line is not an underline",
      "# Title\n\n- A fact\n  -- and its aside -- carries on here.\n",
      [] );
    ( "a flag opening a continuation line inside a code span is not an underline",
      "# Title\n\n- A fact about `git fetch\n  --prune origin`, which cleans up.\n",
      [] );
    ( "two hyphens under a blank line are a paragraph, not an underline",
      "# Title\n\n- A fact.\n\n--\n\n- Another fact.\n",
      [] );
    ("two hyphens under a heading are not an underline", "# Title\n\n## Section\n--\n", []);
    ( "two hyphens under a table row are not an underline",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a | b |\n--\n",
      [] );
    ( "two hyphens under a closing fence are not an underline",
      "# Title\n\n- A fact.\n\n```\ncode\n```\n--\n",
      [ "bullet-integrity @ f.md:5" ] );
    (* A marker whose first visible column is inert is not a marker, and the line carrying it is
       ordinary paragraph text -- it renders. Classifying the raw column instead dropped such a line
       out of the paragraph set, and the underline below it went unreported (Codex P2, round 2). *)
    ( "an underline below a line whose inert first column looks like a quote",
      "# Title\n\n- A fact showing `\n  > example.` in passing:\n  --\n  More.\n",
      [ "bullet-integrity @ f.md:5"; "bullet-integrity @ f.md:6" ] );
    ( "an underline below a line whose inert first column looks like a table",
      "# Title\n\n- A fact showing `\n  | a | b |` in passing:\n  --\n  More.\n",
      [ "bullet-integrity @ f.md:5"; "bullet-integrity @ f.md:6" ] );
    ( "an underline below a line whose inert first column looks like a heading",
      "# Title\n\n- A fact showing `\n  ## Section` in passing:\n  --\n  More.\n",
      [ "bullet-integrity @ f.md:5"; "bullet-integrity @ f.md:6" ] );
    ( "a third nesting level",
      "# Title\n\n\
       - A fact with parts:\n\
      \  - the first part, itself with parts:\n\
      \    - a third        level.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "prose at column zero between lists",
      "# Title\n\n- A fact.\n\nA paragraph of prose.\n\n- Another fact.\n",
      [] );
    ( "a heading closes the list above it",
      "# Title\n\n- A fact.\n\n## Section\n\n- Another fact.\n",
      [] );
  ]

(* ------------------------------------------------------------------ *)
(* Rule 3: table shape *)
(* ------------------------------------------------------------------ *)

let table_cases =
  [
    ( "a well-formed table",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | b |\n| c | d |\n",
      [] );
    (* PR #406's third finding: an edit wrapped a row, which ends the table -- the row is truncated
       and every row below it renders as pipe-delimited prose. Both halves are visible: the row does
       not close, and the line below the block carries the rest of it. *)
    ( "a row wrapped across two physical lines",
      "# Title\n\n\
       | File | What it covers |\n\
       | --- | --- |\n\
       | a | a description long enough to\n\
      \  have been wrapped |\n\
       | c | d |\n",
      [ "table-shape @ f.md:5"; "table-shape @ f.md:7" ] );
    ( "the wrapped tail below a table that itself closes",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | b |\nthe rest of it |\n",
      [ "table-shape @ f.md:6" ] );
    ( "a row with a cell too few",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a |\n",
      [ "table-shape @ f.md:5" ] );
    ( "a row with a cell too many",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | b | c |\n",
      [ "table-shape @ f.md:5" ] );
    ( "a header with no delimiter row",
      "# Title\n\n| File | What it covers |\n| a | b |\n| c | d |\n",
      [ "table-shape @ f.md:4" ] );
    ( "a header and a delimiter and no rows",
      "# Title\n\n| File | Covers |\n| --- | --- |\n",
      [ "table-shape @ f.md:3" ] );
    (* The notes quote pipes constantly (`=:||`, `none|cc|metal`), and a quoted one is not a cell
       separator -- a rule that thought otherwise would fire on a dozen real bullets. *)
    ( "a pipe inside a code span is not a cell separator",
      "# Title\n\n- The vocabulary is closed (`none|cc|metal`).\n",
      [] );
    (* The same gate one level up: a pipe-led line whose leading pipe is INERT is not a table line,
       and reading the raw column opened a one-row table out of a bullet's closing code-span line --
       a false failure on correct text, which is the direction that gets a check switched off. *)
    ( "a pipe-led line whose leading pipe is inert opens no table",
      "# Title\n\n- A fact showing `\n  | a | b |` in passing.\n",
      [] );
    ( "a pipe inside a code span below a table",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a | b |\n\n- The spelling is `a|b`.\n",
      [] );
    ( "an escaped pipe is not a cell separator either",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a \\| b | c |\n",
      [] );
    (* Four spaces make an indented code block, so an index whose rows drifted right renders as a
       code sample with no navigable link in it. Three is Markdown's limit and still a table. *)
    ( "a table indented three spaces is still a table",
      "# Title\n\n   | File | Covers |\n   | --- | --- |\n   | a | b |\n",
      [] );
    ( "a table indented four spaces is a code block",
      "# Title\n\n    | File | Covers |\n    | --- | --- |\n    | a | b |\n",
      [] );
    ( "an escaped backslash leaves the pipe separating",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a \\\\| b | c |\n",
      [ "table-shape @ f.md:5" ] );
    ( "a table example inside a code span is not a table",
      "# Title\n\n- A fact showing `\n  | File | Covers |\n  | --- | --- |\n  ` in passing.\n",
      [] );
    ( "a delimiter cell of one hyphen",
      "# Title\n\n| File | Covers |\n| - | --- |\n| a | b |\n",
      [ "table-shape @ f.md:4" ] );
    ( "a delimiter cell of two hyphens",
      "# Title\n\n| File | Covers |\n| -- | -- |\n| a | b |\n",
      [ "table-shape @ f.md:4" ] );
    ( "alignment colons are still a delimiter row",
      "# Title\n\n| File | Covers |\n| :--- | ---: |\n| a | b |\n",
      [] );
  ]

(* ------------------------------------------------------------------ *)
(* Rules 2, 4 and 5: the index against the files it indexes *)
(* ------------------------------------------------------------------ *)

(* The backlink a file at [depth] directories below docs/ has to carry. Written as a function of the
   depth rather than as the one-level constant it used to be: the constant made the nested-file
   fixture bless a target that resolves to nothing (Codex P2, round 2), which is the sharpest way a
   test can go wrong -- it locked the defect in rather than catching it. *)
let backlink_at depth =
  let up = String.concat (List.init depth ~f:(fun _ -> "../")) in
  Printf.sprintf "Part of the agent notes; the [index](%sagent-notes.md) carries the rest.\n" up

let file_at depth body = "# A file\n\n" ^ backlink_at depth ^ "\n" ^ body

(* Every unnested fixture sits at agent-notes/<name>, one directory below the index. *)
let file body = file_at 1 body

let index rows =
  "# Agent notes\n\nAn index.\n\n| File | What it covers |\n| --- | --- |\n"
  ^ String.concat ~sep:"" (List.map rows ~f:(fun r -> r ^ "\n"))

let row name hooks = Printf.sprintf "| [%s](agent-notes/%s) | %s |" name name hooks

(* Each case is an index body, the files it should be checked against, and the expected findings. *)
let index_cases =
  [
    ( "an index whose rows all resolve",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [
        ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
      ],
      [] );
    (* PR #406's second finding, from both ends: the hook sits on the wrong row, so following the
       index leads away from the trap it names. *)
    ( "a hook the file it points at does not carry",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "`fast math` and the `Gadget` seam" ],
      [
        ("agent-notes/a.md", file "- A fact about `Widget`, and about fast math.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
      ],
      [ "index-agreement @ agent-notes.md:8" ] );
    ( "a hook whose wording drifted from the file's",
      index [ row "a.md" "the `identifier blacklist`" ],
      [ ("agent-notes/a.md", file "- The `ident_blacklist` covers generated names.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    ( "a link to a file that is not there",
      index [ row "a.md" "the `Widget` seam"; row "gone.md" "something" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:8" ] );
    ( "link text that does not name its file",
      index [ "| [shape inference](agent-notes/a.md) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    ( "a first cell that is not a link",
      index [ "| a.md | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7"; "reachability @ agent-notes/a.md" ] );
    ( "an anchor the file has a heading for",
      index [ "| [a.md](agent-notes/a.md#the-widget-seam) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "## The Widget seam\n\n- A fact about `Widget`.\n") ],
      [] );
    ( "an anchor on the second of two same-titled headings",
      index [ "| [a.md](agent-notes/a.md#the-widget-seam-1) | the `Widget` seam |" ],
      [
        ( "agent-notes/a.md",
          file "## The Widget seam\n\n- A fact about `Widget`.\n\n## The Widget seam\n\n- More.\n"
        );
      ],
      [] );
    (* github-slugger allocates against every id already taken: the third heading's [-1] is held by
       the second heading's own title, so it becomes [-2]. *)
    ( "an anchor past a heading whose own title holds the suffix",
      index [ "| [a.md](agent-notes/a.md#foo-2) | the `Widget` seam |" ],
      [
        ( "agent-notes/a.md",
          file "## Foo\n\n- The `Widget` seam.\n\n## Foo-1\n\n- One.\n\n## Foo\n\n- Two.\n" );
      ],
      [] );
    ( "an anchor past the last suffix allocated",
      index [ "| [a.md](agent-notes/a.md#foo-3) | the `Widget` seam |" ],
      [
        ( "agent-notes/a.md",
          file "## Foo\n\n- The `Widget` seam.\n\n## Foo-1\n\n- One.\n\n## Foo\n\n- Two.\n" );
      ],
      [ "index-agreement @ agent-notes.md:7" ] );
    ( "an anchor the file has no heading for",
      index [ "| [a.md](agent-notes/a.md#the-gadget-seam) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "## The Widget seam\n\n- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    (* Reachability, which is the "hook names a file carrying none of it" failure seen from the
       other end: a file nothing links is a file no lookup will reach. *)
    (* PR #406's third finding, end to end: the wrap ends the table, so most of the index is outside
       it. Asking the hook and reachability rules about that would produce one "unreachable" line per
       file and bury the one finding that says what happened -- so they decline, loudly. *)
    ( "a wrapped row takes the rest of the index out of the table",
      "# Agent notes\n\n\
       An index.\n\n\
       | File | What it covers |\n\
       | --- | --- |\n\
       | [a.md](agent-notes/a.md) | the\n\
       `Widget` seam |\n\
       | [b.md](agent-notes/b.md) | the `Gadget` seam |\n",
      [
        ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
      ],
      [
        "index-agreement @ agent-notes.md";
        "table-shape @ agent-notes.md:7";
        "table-shape @ agent-notes.md:9";
        "reachability @ agent-notes.md";
      ] );
    ( "an anchor on an identifier heading, which GitHub keeps underscored",
      index [ "| [a.md](agent-notes/a.md#ident_blacklist) | the `ident_blacklist` |" ],
      [ ("agent-notes/a.md", file "## ident_blacklist\n\n- A fact about `ident_blacklist`.\n") ],
      [] );
    ( "the same anchor with the underscore hyphenated",
      index [ "| [a.md](agent-notes/a.md#ident-blacklist) | the `ident_blacklist` |" ],
      [ ("agent-notes/a.md", file "## ident_blacklist\n\n- A fact about `ident_blacklist`.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    (* A backlink has to be one a reader can FOLLOW. All three of these carry the bytes and none of
       them carries a link. *)
    ( "a backlink that is only a code span",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           The index lives at `[index](../agent-notes.md)`.\n\n\
           - A fact about            `Widget`.\n" );
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink commented out",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n<!-- [index](../agent-notes.md) -->\n\n- A fact about `Widget`.\n" );
      ],
      [ "bullet-integrity @ agent-notes/a.md:3"; "reachability @ agent-notes/a.md" ] );
    ( "a backlink that is an image",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ("agent-notes/a.md", "# A file\n\n![index](../agent-notes.md)\n\n- A fact about `Widget`.\n");
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink hidden in the middle of a multiline code span",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Part of an example: `\n\
           [index](../agent-notes.md)\n\
           `.\n\n\
           - A fact about            `Widget`.\n" );
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink after a code span that closed on the line above",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Part of an example: `\n\
           example`\n\
           [index](../agent-notes.md)\n\n\
           - A fact            about `Widget`.\n" );
      ],
      [] );
    ( "a backlink inside a multiline HTML comment",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Prose <!--\n\
           [index](../agent-notes.md)\n\
           -->\n\n\
           - A fact about            `Widget`.\n" );
      ],
      [ "bullet-integrity @ agent-notes/a.md:3"; "reachability @ agent-notes/a.md" ] );
    ( "a backlink whose bracket is escaped",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Write it \\[index](../agent-notes.md) to show the syntax.\n\n\
           - A fact            about `Widget`.\n" );
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "an anchor naming a heading that only appears inside a comment",
      index [ "| [a.md](agent-notes/a.md#draft-heading) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "<!--\n## Draft heading\n-->\n\n- A fact about `Widget`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:5"; "index-agreement @ agent-notes.md:7" ] );
    ( "an index row with a third cell",
      "# Agent notes\n\n\
       An index.\n\n\
       | File | Covers | Owner |\n\
       | --- | --- | --- |\n\
       |        [a.md](agent-notes/a.md) | the `Widget` seam | me |\n",
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7"; "reachability @ agent-notes/a.md" ] );
    (* Text inside a code span RENDERS, so it is part of what a bullet says; text inside a comment
       does not. Two bullets differing only on a code line are two bullets. *)
    ( "two bullets differing only inside a multiline code span",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Widget` seam" ],
      [
        ("agent-notes/a.md", file "- A fact about `Widget`, showing `\n  foo\n  ` in passing.\n");
        ("agent-notes/b.md", file "- A fact about `Widget`, showing `\n  bar\n  ` in passing.\n");
      ],
      [] );
    ( "two bullets differing only inside an HTML comment are one bullet",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          file "- A fact about `Widget`, showing <!--\n  foo\n  --> in passing.\n" );
        ( "agent-notes/b.md",
          file "- A fact about `Widget`, showing <!--\n  bar\n  --> in passing.\n" );
      ],
      [
        "bullet-integrity @ agent-notes/a.md:5";
        "bullet-integrity @ agent-notes/b.md:5";
        "no-repetition @ agent-notes/b.md:5";
      ] );
    ( "a backlink carrying a link title",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           See the [index](../agent-notes.md \"Agent notes\").\n\n\
           - A fact about `Widget`.\n" );
      ],
      [] );
    ( "a backlink whose destination is angle-bracketed",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\nSee the [index](<../agent-notes.md>).\n\n- A fact about `Widget`.\n" );
      ],
      [] );
    ( "an escaped comment opener starts no comment",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\nProse \\<!--[index](../agent-notes.md)-->\n\n- A fact about `Widget`.\n" );
      ],
      [] );
    ( "a hook padded to carry a literal backtick",
      "# Agent notes\n\n\
       An index.\n\n\
       | File | Covers |\n\
       | --- | --- |\n\
       | [a.md](agent-notes/a.md) | the `` `Widget` `` seam |\n",
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [] );
    ( "a backlink whose closing parenthesis is escaped",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Write it [index](../agent-notes.md#draft\\) to show the syntax.\n\n\
           - A fact about `Widget`.\n" );
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink whose closing bracket is escaped",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Write it [index\\](../agent-notes.md) to show the syntax.\n\n\
           - A fact about `Widget`.\n" );
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink between escaped backticks is navigable",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Written \\`[index](../agent-notes.md)\\` in the source.\n\n\
           - A fact about `Widget`.\n" );
      ],
      [] );
    ( "an anchor naming a heading indented into code",
      index [ "| [a.md](agent-notes/a.md#draft) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "    # Draft\n\n- A fact about `Widget`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:5"; "index-agreement @ agent-notes.md:7" ] );
    (* One block, but not a table: the row extraction must decline rather than report every file as
       an orphan. *)
    ( "an index whose single table has a row that does not close",
      "# Agent notes\n\n\
       An index.\n\n\
       | File | Covers |\n\
       | --- | --- |\n\
       | [a.md](agent-notes/a.md) | the `Widget` seam\n\
       | [b.md](agent-notes/b.md) | the `Gadget` seam |\n",
      [
        ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
      ],
      [
        "index-agreement @ agent-notes.md";
        "table-shape @ agent-notes.md:7";
        "reachability @ agent-notes.md";
      ] );
    ( "a backlink one directory too far up",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\nSee the [index](../../agent-notes.md).\n\n- A fact about `Widget`.\n" );
      ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "an anchor on a heading written without a space",
      index [ "| [a.md](agent-notes/a.md#ident_blacklist) | the `ident_blacklist` |" ],
      [ ("agent-notes/a.md", file "##ident_blacklist\n\n- A fact about `ident_blacklist`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:5"; "index-agreement @ agent-notes.md:7" ] );
    ( "a backlink carrying an anchor is still a backlink",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\n\
           Part of the [index](../agent-notes.md#index).\n\n\
           - A fact about            `Widget`.\n" );
      ],
      [] );
    (* The directory is flat today; a note in a subdirectory of it is judged like any other, which
       is what makes recursing over the tree safe rather than merely more thorough. *)
    ( "a note in a subdirectory, linked and backlinked from its own depth",
      index [ "| [a.md](agent-notes/sub/a.md) | the `Widget` seam |" ],
      [ ("agent-notes/sub/a.md", file_at 2 "- A fact about `Widget`.\n") ],
      [] );
    ( "a nested note carrying the one-level backlink, which resolves to nothing",
      index [ "| [a.md](agent-notes/sub/a.md) | the `Widget` seam |" ],
      [ ("agent-notes/sub/a.md", file_at 1 "- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/sub/a.md" ] );
    ( "a note in a subdirectory, unlinked",
      index [ row "b.md" "the `Gadget` seam" ],
      [
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
        ("agent-notes/sub/a.md", file_at 2 "- A fact about `Widget`.\n");
      ],
      [ "reachability @ agent-notes/sub/a.md" ] );
    ( "a backlink written with a redundant ./ still resolves",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ( "agent-notes/a.md",
          "# A file\n\nSee the [index](./../agent-notes.md).\n\n- A fact about `Widget`.\n" );
      ],
      [] );
    ( "a file no row links",
      index [ row "a.md" "the `Widget` seam" ],
      [
        ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
      ],
      [ "reachability @ agent-notes/b.md" ] );
    ( "two rows for one file",
      index [ row "a.md" "the `Widget` seam"; row "a.md" "the `Widget` seam again" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes.md:8" ] );
    ( "a file that does not link back to the index",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md", "# A file\n\n- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    (* End-to-end wiring for the sixth rule: exercising [check_citations] directly would stay green
       if [check_all] silently stopped invoking it. *)
    ( "a bare citation reaches the whole scan",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget` first established in #12.\n") ],
      [ "qualified-citations @ agent-notes/a.md:5" ] );
    (* Rule 5. A fact promoted into two files is a fact that will be corrected in one of them. *)
    ( "the same bullet in two files",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [
        ("agent-notes/a.md", file "- One fact, promoted twice, about `Widget` and `Gadget`.\n");
        ("agent-notes/b.md", file "- One fact, promoted twice, about `Widget` and `Gadget`.\n");
      ],
      [ "no-repetition @ agent-notes/b.md:5" ] );
    ( "the same bullet re-wrapped",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [
        ("agent-notes/a.md", file "- One fact, promoted twice, about `Widget` and `Gadget`.\n");
        ("agent-notes/b.md", file "- One fact, promoted twice, about `Widget`\n  and `Gadget`.\n");
      ],
      [ "no-repetition @ agent-notes/b.md:5" ] );
    ( "two bullets opening alike and diverging",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [
        ( "agent-notes/a.md",
          file "- The analysis pass establishes what the specializer may assume of `Widget`.\n" );
        ( "agent-notes/b.md",
          file "- The analysis pass establishes what the specializer may assume of `Gadget`.\n" );
      ],
      [ "no-repetition @ agent-notes/b.md:5" ] );
    ( "two bullets that merely start with the same few words",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [
        ( "agent-notes/a.md",
          file "- The `Widget` seam is where storage meets compute, and it is narrow.\n" );
        ( "agent-notes/b.md",
          file "- The `Gadget` seam is where the pool meets the stream, and it is wide.\n" );
      ],
      [] );
  ]

(* Rule 7: the agent guide's anchored pointers into the notes (gh-ocannl-1044). Every case runs the
   whole scan over one index, two clean notes and a guide, so a finding from any other rule would
   show up too. The heading the pointers aim at lives in [a.md] only, and [b.md] has one of its own:
   a reader that accepted a heading from ANY note would pass the "wrong note" case. *)
let guide_notes =
  [
    ("agent-notes/a.md", file "## The Widget seam\n\n- A fact about `Widget`.\n");
    ("agent-notes/b.md", file "## The Gadget seam\n\n- A fact about `Gadget`.\n");
    ( "agent-notes/c.md",
      file
        "## The Widget seam\n\n\
         - A first fact about `Sprocket`.\n\n\
         ## The Widget seam\n\n\
         - A second fact about `Sprocket`.\n" );
  ]

let guide_index =
  index
    [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam"; row "c.md" "the `Sprocket`" ]

let guide line = "# OCANNL Agent Guide\n\n" ^ line ^ "\n"

let guide_cases =
  [
    ( "a pointer to a heading its note has",
      guide "- A rule; the mechanism: a.md#the-widget-seam.",
      [] );
    ( "a pointer to a heading its note lacks",
      guide "- A rule; the mechanism: a.md#the-sprocket-seam.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    ( "a pointer to a heading only another note has",
      guide "- A rule; the mechanism: a.md#the-gadget-seam.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    ( "a pointer spelled through docs/agent-notes/",
      guide "- A rule (docs/agent-notes/b.md#the-gadget-seam).",
      [] );
    ("a pointer into the index itself", guide "- A rule (docs/agent-notes.md#agent-notes).", []);
    ( "a bare basename that is no note",
      guide "- A rule; the mechanism: d.md#the-widget-seam.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    ( "a path outside the notes is not a pointer into them",
      guide "- A rule; see docs/proposals/x.md#anything and ./CHANGES.md#unreleased.",
      [] );
    ( "a pointer set in a code span is still a pointer",
      guide "- A rule; the mechanism: `a.md#the-sprocket-seam`.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    ("a placeholder names no anchor", guide "- Pointers read `<note>.md#<anchor>`.", []);
    ( "two pointers on one line are each checked",
      guide "- Both a.md#the-widget-seam and b.md#the-widget-seam.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    (* A comment renders nowhere, so a stale pointer inside one is not a pointer; the same pointer
       beside the comment on the next line still is. *)
    ( "a pointer inside an HTML comment is not read",
      guide "- A rule. <!-- was a.md#the-sprocket-seam -->\n- Another; a.md#the-sprocket-seam.",
      [ "guide-anchors @ AGENTS.md:4" ] );
    (* GitHub suffixes a repeated heading's anchor: the second "The Widget seam" in [c.md] is
       [#the-widget-seam-1], and there is no third. *)
    ( "a pointer at the second of two same-titled headings",
      guide "- A rule; the mechanism: c.md#the-widget-seam-1.",
      [] );
    ( "a pointer at a third that does not exist",
      guide "- A rule; the mechanism: c.md#the-widget-seam-2.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    (* Escapes render away, so an escaped pointer is the pointer it displays -- and is checked. *)
    ( "an escaped hash is still a pointer",
      guide "- A rule; the mechanism: a.md\\#the-sprocket-seam.",
      [ "guide-anchors @ AGENTS.md:3" ] );
    ( "an escaped hash naming a real heading",
      guide "- A rule; the mechanism: a.md\\#the-widget\\-seam.",
      [] );
  ]

(* The LEXICAL layer, tested directly rather than only through the rules above it.

   Round 3 was six findings and three of them were here -- code-span pairing, backslash parity, ATX
   spacing -- each an approximation of a CommonMark rule, each feeding every rule in the scan, and
   each invisible from the rule level because a rule reads "no pipe there" and "no heading there"
   the same whether the primitive was right or lazy. A layer that everything depends on and nothing
   tests directly is where this class regenerates, so it gets its own fixtures. *)
let primitive_cases =
  [
    (* A run of N backticks is closed by the next run of exactly N. *)
    ("one span", "a `b` c", [ "2-5" ]);
    ("two spans", "`a` and `b`", [ "0-3"; "8-11" ]);
    ("a double run holds a single", "``a`b`` c", [ "0-7" ]);
    (* CommonMark renders an unmatched run literally, so it opens nothing and the text after it is
       ordinary prose -- which is what makes a link there navigable. *)
    ("an unpaired run opens nothing", "a `b c", []);
    ("an escaped backtick opens nothing either", "a \\`b` c", []);
    ("no backticks at all", "plain text", []);
    ("a comment is inert too", "a <!-- b --> c", [ "2-12" ]);
    ("a backtick inside a comment opens nothing", "a <!-- ` --> c", [ "2-12" ]);
    ("a comment opener inside code opens nothing", "a `<!--` b", [ "2-8" ]);
  ]

(* The lexer's state across line boundaries, which is where rounds 3 and 4 both landed. Each case is
   a whole file and the lines whose text is inert, as "<line>:<start>-<stop>". *)
let carry_cases =
  [
    ("a span closed on the next line", "a `b\nc` d\n", [ "1:2-4"; "2:0-2" ]);
    (* Round 4's finding: the closing run being LAST on its line must clear the carry, and did not
       -- the whole rest of the file went on reading as code. *)
    ( "a span closed by the last run on its line",
      "a `b\nexample`\n[index](../agent-notes.md)\n",
      [ "1:2-4"; "2:0-8" ] );
    ("a paragraph break discards an unterminated span's ranges", "a `b\n\nc | d\n", []);
    ( "a comment spanning three lines",
      "Prose <!--\n[index](../agent-notes.md)\n--> after\n",
      [ "1:6-10"; "2:0-26"; "3:0-3" ] );
    ("a comment survives a blank line", "Prose <!--\n\n--> after\n", [ "1:6-10"; "3:0-3" ]);
  ]

(* Escaping is decided by the PARITY of the backslash run, not by the character before the pipe. *)
let pipe_cases =
  [
    ("a plain row", "| a | b |", [ 0; 4; 8 ]);
    ("an escaped pipe is not a separator", "| a \\| b |", [ 0; 9 ]);
    ("an escaped backslash leaves the pipe live", "| a \\\\| b |", [ 0; 6; 10 ]);
    ("three backslashes escape it again", "| a \\\\\\| b |", [ 0; 11 ]);
    ("a pipe inside code is not a separator", "| `a|b` | c |", [ 0; 8; 12 ]);
  ]

(* An ATX marker is one to six hashes followed by a space or the end of the line. *)
let heading_cases =
  [
    ("a heading", "## Title", Some "Title");
    ("a heading with no space", "##Title", None);
    ("a bare marker", "#", Some "");
    ("closing hashes are decoration", "## Title ##", Some "Title");
    ("seven hashes is not a heading", "####### Title", None);
    ("an identifier heading", "## ident_blacklist", Some "ident_blacklist");
  ]

(* A relative link resolves against its file's DIRECTORY, and above the root is a distinct
   answer. *)
let resolve_cases =
  [
    ("one level up", "agent-notes/a.md", "../agent-notes.md", Some "agent-notes.md");
    ( "two levels up from a nested note",
      "agent-notes/sub/a.md",
      "../../agent-notes.md",
      Some "agent-notes.md" );
    ("one too many is above the root", "agent-notes/a.md", "../../agent-notes.md", None);
    ("a redundant ./ is dropped", "agent-notes/a.md", "./../agent-notes.md", Some "agent-notes.md");
    ("a sibling needs no ..", "agent-notes/a.md", "b.md", Some "agent-notes/b.md");
    ( "an anchor is not part of the path",
      "agent-notes/a.md",
      "../agent-notes.md#x",
      Some "agent-notes.md" );
  ]

(* The escape hatch is only a hatch if the key a message tells you to paste is the key that matches.
   Round 1 found it could match nothing at all: the documented format named the bullet's opening
   while the message carried its tail, and the comparison ran over the message text. Both halves are
   pinned here -- the key's shape, and that the message hands it to you verbatim. *)
let exemption_cases =
  [
    ( "the key names the file and the bullet's opening",
      "agent-notes/a.md",
      "# A file\n\n- A trailing identifier is the ending here: `Ops.promote_prec`\n",
      [ "agent-notes/a.md: A trailing identifier is the ending here: `Ops.promote_prec`" ] );
    ( "a re-wrapped bullet has the same key",
      "agent-notes/a.md",
      "# A file\n\n- A trailing identifier is the ending\n  here: `Ops.promote_prec`\n",
      [ "agent-notes/a.md: A trailing identifier is the ending here: `Ops.promote_prec`" ] );
    (* The display length is 48; these two agree for well past that and diverge only at the end, so
       a prefix key would have silenced both from one exemption. *)
    ( "two bullets agreeing past the display length keep distinct keys",
      "agent-notes/a.md",
      "# A file\n\n\
       - The analysis pass establishes what the specializer may assume of `Widget`\n\
       -        The analysis pass establishes what the specializer may assume of `Gadget`\n",
      [
        "agent-notes/a.md: The analysis pass establishes what the specializer may assume of \
         `Widget`";
        "agent-notes/a.md: The analysis pass establishes what the specializer may assume of \
         `Gadget`";
      ] );
    ( "a finished bullet has no finding to key",
      "agent-notes/a.md",
      "# A file\n\n- A fact that ends properly.\n",
      [] );
  ]

(* A bare numeric reference is ambiguous between the staging PR repository and the upstream issue
   tracker. The citation rule shares the Markdown lexer with the structural rules, so its nearest
   legitimate cases include both canonical qualifiers and text that only resembles prose inside an
   inert region. *)
let citation_cases =
  [
    ( "a bare numeric citation",
      "A regression first appeared in #12.\n",
      [ "qualified-citations @ f.md:1" ] );
    ( "compact PR and issue labels are still unqualified",
      "Regressions: PR#12 and issue#13.\n",
      [ "qualified-citations @ f.md:1"; "qualified-citations @ f.md:1" ] );
    ("the canonical issue and PR forms", "Facts: gh-ocannl-12; staging#12; ahrefs/ocannl#12.\n", []);
    ("a hash inside a code span", "The literal `#12` is example text.\n", []);
    ("a hash inside a fenced block", "```text\n#12 is fixture output\n```\n", []);
    ("a hash attached to a code identifier", "The generated name is node#12.\n", []);
  ]

let () =
  let check name expected found =
    if List.equal String.equal found expected then printf "ok: %s\n" name
    else
      fail "%s -- expected [%s], found [%s]" name
        (String.concat ~sep:"; " expected)
        (String.concat ~sep:"; " found)
  in
  List.iter structure_cases ~f:(fun (name, body, expected) ->
      check ("bullets -- " ^ name) expected
        (List.map (Notes.check_structure ~file:"f.md" body) ~f:render));
  List.iter table_cases ~f:(fun (name, body, expected) ->
      check ("tables -- " ^ name) expected
        (List.map (Notes.check_tables ~file:"f.md" body) ~f:render));
  List.iter primitive_cases ~f:(fun (name, line, expected) ->
      let found = List.map (Notes.code_spans line) ~f:(fun (a, b) -> Printf.sprintf "%d-%d" a b) in
      check ("code spans -- " ^ name) expected found);
  List.iter carry_cases ~f:(fun (name, contents, expected) ->
      let found =
        List.concat_map (Notes.inert_by_line contents).Notes.ranges ~f:(fun (lineno, ranges) ->
            List.map ranges ~f:(fun (a, b) -> Printf.sprintf "%d:%d-%d" lineno a b))
      in
      check ("inert carry -- " ^ name) expected found);
  List.iter pipe_cases ~f:(fun (name, line, expected) ->
      let found = List.map (Notes.pipes_outside_code line) ~f:Int.to_string in
      check ("cell separators -- " ^ name) (List.map expected ~f:Int.to_string) found);
  List.iter heading_cases ~f:(fun (name, line, expected) ->
      check ("atx headings -- " ^ name) (Option.to_list expected)
        (Option.to_list (Notes.atx_heading line)));
  List.iter resolve_cases ~f:(fun (name, from_file, target, expected) ->
      check ("link resolution -- " ^ name) (Option.to_list expected)
        (Option.to_list (Notes.resolve_link ~from_file target)));
  List.iter exemption_cases ~f:(fun (name, file, body, expected) ->
      let keys = List.filter_map (Notes.check_structure ~file body) ~f:Notes.exemption_key in
      let message_offers_key =
        List.for_all (Notes.check_structure ~file body) ~f:(fun f ->
            match Notes.exemption_key f with
            | None -> true
            | Some k -> String.is_substring f.Notes.message ~substring:k)
      in
      check ("exemption -- " ^ name) expected keys;
      if not message_offers_key then
        fail "exemption -- %s: the finding's message does not print the key that would silence it"
          name);
  List.iter citation_cases ~f:(fun (name, body, expected) ->
      check ("citations -- " ^ name) expected
        (List.map (Notes.check_citations ~file:"f.md" body) ~f:render));
  List.iter index_cases ~f:(fun (name, index_contents, files, expected) ->
      let _, found = Notes.check_all ~index_file:"agent-notes.md" ~index_contents ~files () in
      check ("index -- " ^ name) expected (List.map found ~f:render));
  List.iter guide_cases ~f:(fun (name, guide_contents, expected) ->
      let _, found =
        Notes.check_all ~guide:("AGENTS.md", guide_contents) ~index_file:"agent-notes.md"
          ~index_contents:guide_index ~files:guide_notes ()
      in
      check ("guide -- " ^ name) expected (List.map found ~f:render));
  (* gh-ocannl-706. A finding whose rule [Notes.rules] does not name -- a sixth rule written and not
     added to the list -- used to be dropped where the report is grouped by that list: the rule
     fired and nothing showed it. Put to the rule synthetically, since no fixture here can produce
     one: the scan's own findings are tagged from the five constants. *)
  let named = Notes.finding ~file:"agent-notes/a.md" ~line:1 ~rule:Notes.rule_table_shape "named" in
  let unnamed = Notes.finding ~file:"agent-notes/a.md" ~line:2 ~rule:"invented-rule" "unnamed" in
  check "unnamed rule -- a finding the list does not name survives the report, last"
    [ "table-shape @ agent-notes/a.md:1"; "invented-rule @ agent-notes/a.md:2" ]
    (List.map (Notes.in_rule_order [ unnamed; named ]) ~f:render);
  check "unnamed rule -- the census reports it, and not the named one beside it"
    [ "invented-rule @ agent-notes/a.md:2" ]
    (List.map (Notes.unnamed_rule_findings [ unnamed; named ]) ~f:render);
  check "unnamed rule -- a named finding leaves the census empty" []
    (List.map (Notes.unnamed_rule_findings [ named ]) ~f:render);
  (* This file's own relationship to the scan: the point of it is that each rule has a violation it
     must flag beside the legitimate text it must not, and a sixth rule shipping without one would
     leave the live-tree scan as its only reader -- green on days the notes are intact, which is
     most days, which is the shape this file exists to refuse. The rules exercised are read off the
     expectations rather than restated, and compared as sorted lists so a duplicate in [Notes.rules]
     is a mismatch too. The claim is a bare boolean: the golden must not become the list again. *)
  let rule_of_expectation e = List.hd_exn (String.split e ~on:' ') in
  let exercised =
    List.concat_map structure_cases ~f:(fun (_, _, e) -> e)
    @ List.concat_map table_cases ~f:(fun (_, _, e) -> e)
    @ List.concat_map index_cases ~f:(fun (_, _, _, e) -> e)
    @ List.concat_map citation_cases ~f:(fun (_, _, e) -> e)
    @ List.concat_map guide_cases ~f:(fun (_, _, e) -> e)
    |> List.map ~f:rule_of_expectation
    |> List.dedup_and_sort ~compare:String.compare
  in
  let named_rules = List.sort Notes.rules ~compare:String.compare in
  let covered = List.equal String.equal exercised named_rules in
  if not covered then
    eprintf "the cases here flag [%s]; the scan names [%s]\n"
      (String.concat ~sep:"; " exercised)
      (String.concat ~sep:"; " named_rules);
  Verdict.p "every rule the scan names is exercised by a case here" covered
