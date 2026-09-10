(** gh-ocannl-807: the changelog's `## [Unreleased]` section obeys the editorial convention
    mechanically.

    `CHANGES.md` is written in editorial passes rather than in feature PRs, and the two rules a pass
    has to hold are stated in [docs/agent-notes/conventions.md]: a top-level bullet is at most three
    lines, and it cites the record the reader can follow — a `gh-ocannl-NNN` issue or a
    `lukstafi/ocannl-staging` PR (`PR #NNN`). Both were enforced by hand on the 2026-09-02 batch
    catch-up (lukstafi/ocannl-staging PR #601, review findings 3915031304 and 3915183641), in rounds
    three and four of eight. A reviewer re-deriving a decidable rule from prose costs a round per
    rule; this file costs none.

    Only the Unreleased section is read. Released sections are history — they record what a past
    release said, under whatever conventions held then, and rewriting them to satisfy today's rules
    would falsify the record rather than fix anything.

    {1 A canonical section, and everything else refused}

    The section is held to a CANONICAL FORM, and every line outside it is refused rather than
    interpreted. A line is one of four shapes:

    - blank;
    - a subheading — a heading of level three or deeper at column zero ([### Added]);
    - a bullet — [- ] at column zero, one space, then text;
    - a continuation — exactly two spaces, then text.

    That is what the changelog has always been, and it is the whole grammar this scan decides. The
    alternative was matching CommonMark's block structure for arbitrary input, which is how this
    file spent eight review rounds: lazy continuations, content columns measured in display columns,
    thematic breaks that look like markers, ordered markers that may or may not interrupt a
    paragraph, tab stops, blocks that end at a blank line and blocks that end at a tag. Each was a
    real defect and each fix was correct, but the supply is CommonMark-sized, and every one of them
    could fail SILENTLY — an extent read a line short hides half a bullet, a marker unrecognized
    drops an entry out of the population, and the golden stays green either way.

    A whitelist fails the other direction. Any shape this grammar does not name — an [* ] or [+ ]
    marker, a marker followed by two spaces or a tab, a marker alone on its line, an ordered item,
    an indented marker, a thematic break, a block quote, an unindented lazy continuation, a fenced
    block, an HTML comment, raw HTML, an indented heading, a four-space indent — fails the gate,
    loudly and by name. So the scan's imprecision, wherever it remains, produces a failure a person
    reads rather than a bullet nobody checked. The day the changelog wants one of those shapes is
    the day somebody decides what these rules mean for it; until then, saying so costs nothing.

    {1 The anchor, and history that must not fail}

    The one place CommonMark still has to be tracked is the search for the anchor, because it runs
    over the WHOLE file, released history included, and history is not editable to satisfy this
    scan. A released entry quoting [## [Unreleased]] inside a fence, a comment, or a raw HTML block
    is inert, and counting it would fail the scan over the past. So fences (by delimiter character
    and run length), HTML comments (outside code spans) and raw HTML blocks (to their own closing
    tag, or to a blank line) are tracked across the file, and only lines Markdown renders as
    structure can be an anchor or a section boundary. The anchor itself is a level-2 heading at
    column zero: the decidable form of "the document-level section everyone means", which excludes a
    nested child heading and an indented code copy without any list-container state.

    {1 What the golden holds}

    The claims, and no tally: the number of bullets in Unreleased moves on every editorial pass and
    drops to zero at every release, so a count in the golden would make every correct change a
    promote indistinguishable from blessing a regression. The exact counts go to stderr. What a
    count was there for — the assurance that a scan reporting nothing read something — is kept as
    the anchor and gate claims, which fail when the section cannot be found or is not what this
    grammar describes, plus the synthetic controls, which hold every rule against text built to
    break it so a checker that stopped deciding cannot pass here by finding nothing. *)

open Base
open Stdio
open Verdict.Claims

type bullet = { first_line : string; lines : string list }
(** A top-level bullet: the physical lines of one item, first line first, blank separators included
    where they sit between two blocks of the same item. *)

let is_blank line = String.is_empty (String.strip line)

(** The count of leading spaces, and where the line's content starts. Markdown allows up to three
    before a block-level construct and reads a fourth as an indented code block, so the same
    allowance decides headings and fence delimiters alike. *)
let indent_of line =
  let n = String.length line in
  let rec run i = if i < n && Char.equal line.[i] ' ' then run (i + 1) else i in
  run 0

let block_indent = 3

(** The level of an ATX heading — up to three spaces, then the run of [#], at most six of them,
    closed by whitespace or the end of the line — and [None] for anything else. A prefix test cannot
    answer this: [### Added] and [#### Security] are both subsections of the entry they sit under,
    while [## [1.0.1]] ends it. Seven or more hashes is prose, and reading it as a heading would cut
    an open bullet short at a line that renders as part of it. The indent allowance is what keeps
    this recognizer and the anchor's agreeing: an indented release heading has to close the section
    it would close in a renderer, and a four-space-indented anchor is code, not an anchor. *)
let heading_level line =
  let start = indent_of line in
  if start > block_indent then None
  else
    let hashes =
      let rec run i =
        if i < String.length line && Char.equal line.[i] '#' then run (i + 1) else i
      in
      run start
    in
    let count = hashes - start in
    if count > 0 && count <= 6 && (hashes = String.length line || Char.is_whitespace line.[hashes])
    then Some count
    else None

let unreleased_heading = "## [Unreleased]"

(** The anchor: a level-2 heading at COLUMN ZERO whose text is exactly [[Unreleased]].

    Indentation is where "is this the shared anchor" and "is this a heading" part company. A heading
    may carry up to three spaces and still be a heading — which is why the section BOUNDARY keeps
    that allowance — but an indented one is not the document-level section every reader and every
    editorial pass means by the anchor: two spaces inside a released list item makes it that item's
    child heading, and four makes it code. Column zero is the decidable form of "document-level",
    and it costs nothing: an anchor that ever picks up indentation fails the uniqueness claim
    loudly, rather than a nested copy being read as a second one. *)
let is_unreleased_heading line =
  match heading_level line with
  | Some 2 -> indent_of line = 0 && String.equal (String.strip line) unreleased_heading
  | _ -> false

(** The section ends at the next document-level heading — level 1 or 2, at COLUMN ZERO, by the same
    rule that decides the anchor. An INDENTED level-2 heading is not a document section: two spaces
    make it a child heading of a list item, and ending the section there silently drops every bullet
    after it. Indented headings inside the section are refused by the canonical-form gate below
    rather than interpreted, so neither reading has to be guessed. *)
let ends_section line =
  indent_of line = 0 && match heading_level line with Some level -> level <= 2 | None -> false

(** A setext underline: a line of only [=] or only [-], which turns the NON-BLANK line above it into
    a document-level heading. `Release 1.0` over `-----------` is a level-2 heading as much as `##
    Release 1.0` is, and a section that does not end there reads released history as its own. *)
let is_setext_underline line =
  indent_of line = 0
  &&
  let body = String.strip line in
  (not (String.is_empty body))
  && (String.count body ~f:(Fn.non (Char.equal '=')) = 0
     || String.count body ~f:(Fn.non (Char.equal '-')) = 0)

(** A fence delimiter: up to three spaces, then at least three [`] or [~], as (character, run
    length, whatever follows). A fence closes only on the same character, a run at least as long,
    and nothing after it — so a three-backtick line inside a four-backtick fence is content, and
    released history that quotes one cannot reopen the file's structure. *)
let fence_at line =
  let n = String.length line in
  let start = indent_of line in
  if start > block_indent || start >= n then None
  else
    let c = line.[start] in
    if not (Char.equal c '`' || Char.equal c '~') then None
    else
      let rec run j = if j < n && Char.equal line.[j] c then run (j + 1) else j in
      let stop = run start in
      let info = String.strip (String.drop_prefix line stop) in
      (* A backtick fence's info string cannot itself contain a backtick, so such a line opens no
         block. Accepting it would swallow the structure that follows until some later delimiter. *)
      if stop - start >= 3 && not (Char.equal c '`' && String.contains info '`') then
        Some (c, stop - start, info)
      else None

let opens_fence line = Option.is_some (fence_at line)

(** A comment that opens an HTML BLOCK: one that begins the line. A `<!--` in the middle of a
    sentence is inline raw HTML — it hides its own contents from the reader, which is
    {!visible_text}'s business, but it starts no block. Reading one as a block opener was worse than
    a missed comment: an entry ending in an unclosed code span before a `<!--` swallowed the rest of
    the file, so the scan read released history as the Unreleased section and reported hundreds of
    failures against history it must not touch. *)
let opens_comment_block line =
  indent_of line <= block_indent && String.is_prefix (String.strip line) ~prefix:"<!--"

(** A raw HTML block, in the two shapes that can hold a line reading exactly like a heading: a
    [<pre>], [<script>], [<style>] or [<textarea>] block, which runs to its closing tag, and any
    other block-level tag, which runs to the next blank line. Released history is searched for the
    anchor like everything else, and a [## [Unreleased]] inside such a block renders as HTML content
    — counting it would fail the scan over history that must stay untouched. *)
let raw_html_tags = [ "pre"; "script"; "style"; "textarea" ]

(** Only the opener's OWN closing tag ends its block: a literal [</script>] inside a [<pre>] leaves
    the [<pre>] open, and closing on it would expose what follows as structure. *)
let closes_raw_html_tag ~tag line =
  String.is_substring (String.lowercase line) ~substring:("</" ^ tag ^ ">")

let opens_raw_html line =
  (* Tag NAMES are case-insensitive; `<![CDATA[` is not, so the raw text decides that one. *)
  let text = String.lowercase (String.strip line) in
  if indent_of line > block_indent then None
  else
    (* The tag NAME has to end there: `<prelude>` is not a `<pre>` block but a generic one, and
       reading it as `<pre>` leaves the state open until a `</pre>` that never comes -- hiding every
       heading after it, the anchor included. *)
    let names tag =
      String.is_prefix text ~prefix:("<" ^ tag)
      &&
      let k = 1 + String.length tag in
      k >= String.length text
      ||
      let next = text.[k] in
      Char.is_whitespace next || Char.equal next '>' || Char.equal next '/'
    in
    match List.find raw_html_tags ~f:names with
    (* A block that closes on its own opening line -- [<pre>example</pre>] -- opens no state at all;
       keeping one would hide every following line until some later closer. *)
    | Some tag -> if closes_raw_html_tag ~tag line then None else Some (`Until_close tag)
    | None -> (
        (* A generic HTML block opens with a TAG: `<`, an optional `/`, a name, then a boundary.
           `<https://example.com>` is an autolink -- inline Markdown -- and reading it as a block
           would hide every heading up to the next blank line, the anchor included. *)
        (* A complete tag ALONE on its line, which is the only generic form that opens a block
           (CommonMark type 7). `<span>text</span>` is inline HTML and opens nothing, where reading
           it as a block would hide every heading up to the next blank line. *)
        let one_complete_tag =
          let n = String.length text in
          let after_slash = if String.is_prefix text ~prefix:"</" then 2 else 1 in
          n > after_slash
          && Char.is_alpha text.[after_slash]
          && Char.equal text.[n - 1] '>'
          && String.count text ~f:(Char.equal '<') = 1
          && String.count text ~f:(Char.equal '>') = 1
          &&
          (* ... and the NAME has to end at a tag boundary: `<https://example.com>` is an autolink,
             whose "name" runs into a colon. *)
          let rec name i =
            if i < n && (Char.is_alphanum text.[i] || Char.equal text.[i] '-') then name (i + 1)
            else i
          in
          let stop = name after_slash in
          let next = text.[stop] in
          Char.is_whitespace next || Char.equal next '>' || Char.equal next '/'
        in
        (* Declarations, processing instructions and CDATA run to their own terminators. Released
           history quoting an XML example is inert content, not a second anchor. *)
        (* A block whose terminator is already on its opening line opens no state at all -- the
           same reason a `<pre>example</pre>` does not. *)
        let declaration ?(raw = false) prefix marker =
          let subject = if raw then String.strip line else text in
          if not (String.is_prefix subject ~prefix) then None
          else if
            String.is_substring
              (String.drop_prefix subject (String.length prefix))
              ~substring:marker
          then Some None
          else Some (Some (`Until_marker marker))
        in
        let doctype () =
          (* CommonMark's declaration opener takes an UPPERCASE letter, so the raw text decides this
             one too: a lowercase `<!note` opens no block. *)
          let raw = String.strip line in
          if
            String.is_prefix raw ~prefix:"<!"
            && (not (String.is_prefix raw ~prefix:"<!--"))
            && String.length raw > 2
            && Char.is_uppercase raw.[2]
          then declaration ~raw:true "<!" ">"
          else None
        in
        match declaration "<?" "?>" with
        | Some state -> state
        | None -> (
            match declaration ~raw:true "<![CDATA[" "]]>" with
            | Some state -> state
            | None -> (
                match doctype () with
                | Some state -> state
                | None ->
                    if String.is_prefix text ~prefix:"<" && one_complete_tag then Some `Until_blank
                    else None)))

(** Each line paired with whether Markdown reads it as STRUCTURE — outside fenced code blocks and
    HTML comments. The whole file is searched for the anchor, released history included, and a
    released entry that quotes `## [Unreleased]` inside a fence or leaves one in a comment is not a
    second anchor: reading it as one would fail the scan over history the convention says to leave
    untouched, and take the live section down with it. *)
let structural_lines lines =
  let fence = ref None and comment = ref false and html = ref None in
  List.map lines ~f:(fun line ->
      let structural = (not !comment) && Option.is_none !fence && Option.is_none !html in
      (if !comment then (if String.is_substring line ~substring:"-->" then comment := false)
       else
         match !html with
         | Some (`Until_close tag) -> if closes_raw_html_tag ~tag line then html := None
         | Some (`Until_marker marker) ->
             if String.is_substring line ~substring:marker then html := None
         | Some `Until_blank -> if is_blank line then html := None
         | None -> (
             match (!fence, fence_at line) with
             | Some (opened, length), Some (c, run, info) ->
                 if Char.equal opened c && run >= length && String.is_empty info then fence := None
             | Some _, None -> ()
             | None, Some (c, run, _) -> fence := Some (c, run)
             | None, None ->
                 if opens_comment_block line && not (String.is_substring line ~substring:"-->") then
                   comment := true
                 else html := opens_raw_html line));
      (line, structural))

let unreleased_headings lines =
  List.filter_map (structural_lines lines) ~f:(fun (line, structural) ->
      if structural && is_unreleased_heading line then Some line else None)

(** The lines under `## [Unreleased]`, to the next heading of level 1 or 2, or the end of file.
    Subheadings (`### Added`, `#### Security`) stay in: they separate bullets, they never continue
    one, and one nested deeper than the pass happened to use is no boundary. Returns [None] unless
    there is exactly ONE Unreleased anchor — with two, a first-match reader silently stops looking
    before the second one's bullets. *)
let unreleased_section lines =
  match unreleased_headings lines with
  | [ _ ] ->
      let rec find = function
        | [] -> None
        | (line, structural) :: rest when structural && is_unreleased_heading line ->
            (* The boundary is the heading's TEXT line, which a setext heading only reveals on the
               line after it -- hence the lookahead rather than a plain predicate. *)
            let rec upto = function
              | [] -> []
              | (line, structural) :: _ when structural && ends_section line -> []
              | (line, structural) :: ((next, next_structural) :: _ as tail)
                when structural && next_structural
                     && (not (is_blank line))
                     && is_setext_underline next ->
                  ignore tail;
                  []
              | (line, _) :: tail -> line :: upto tail
            in
            Some (upto rest)
        | _ :: rest -> find rest
      in
      find (structural_lines lines)
  | _ -> None

(** {1 The canonical section} *)

let bullet_marker = "- "
let continuation_indent = "  "

(** Text that starts no block of its own: not a list marker, a heading, a quote, a fence or a
    thematic break. It is what a bullet's content and a continuation line both have to be, so that
    the extent needs no interpretation — anything else is refused by name. *)
let is_plain_text text =
  (not (is_blank text))
  &&
  let c = text.[0] in
  let run_of ch =
    let rec run i = if i < String.length text && Char.equal text.[i] ch then run (i + 1) else i in
    run 0
  in
  let is_one_of a b d = Char.equal c a || Char.equal c b || Char.equal c d in
  let body = String.filter text ~f:(Fn.non Char.is_whitespace) in
  (* A thematic break: three or more of the same character, whitespace aside. `* * *` presents
     exactly the marker-then-space shape a bullet does. *)
  let thematic =
    is_one_of '-' '*' '_'
    && String.length body >= 3
    && String.count body ~f:(Fn.non (Char.equal c)) = 0
  in
  (* Only an actual block opener is refused. An inline code span is ordinary prose -- most entries
     in this changelog open with one -- so it takes a fence's three backticks to matter here. *)
  let fence = (Char.equal c '`' || Char.equal c '~') && run_of c >= 3 in
  let marker = is_one_of '-' '*' '+' && String.length text > 1 && Char.is_whitespace text.[1] in
  (* An ordered marker opens a list too, and a nested one under a bullet would otherwise ride in on
     that bullet's citation as a continuation line. *)
  let ordered =
    Char.is_digit c
    &&
    let n = String.length text in
    let rec run i = if i < n && Char.is_digit text.[i] then run (i + 1) else i in
    let stop = run 0 in
    (* One to nine digits, per CommonMark: a longer run is prose -- an identifier, a year, a phone
       number -- and refusing it would fail the gate over an ordinary entry. *)
    stop <= 9 && stop < n
    && (Char.equal text.[stop] '.' || Char.equal text.[stop] ')')
    && (stop + 1 = n || Char.is_whitespace text.[stop + 1])
  in
  (* A link-reference definition renders as NOTHING: `[record]: gh-ocannl-807` is invisible in the
     changelog, so a citation inside one is a citation no reader can follow -- exactly what these
     rules exist to guarantee. *)
  (* A COMPLETE link-reference definition, and only that: `[label]: destination`, with at most a
     title behind it. It renders as nothing, so a citation inside one is invisible. But
     `- [API]: behavior changed (gh-ocannl-807).` is ordinary paragraph text -- the words after the
     destination are no title -- and refusing it would fail the gate over a legitimate entry. *)
  let link_reference =
    Char.equal c '['
    &&
    match String.index text ']' with
    | Some close when close + 1 < String.length text && Char.equal text.[close + 1] ':' -> (
        let rest = String.drop_prefix text (close + 2) in
        match String.split rest ~on:' ' |> List.filter ~f:(Fn.non String.is_empty) with
        | [ _destination ] -> true
        | _destination :: _ :: _ ->
            (* A title, and NOTHING after it: `[API]: behavior "changed" for users` has words beyond
               the quotes, which makes the whole line ordinary paragraph text. *)
            let after_destination =
              let trimmed = String.lstrip rest in
              match String.index trimmed ' ' with
              | Some space -> String.lstrip (String.drop_prefix trimmed space)
              | None -> ""
            in
            let closes opening close =
              String.length after_destination >= 2
              && Char.equal after_destination.[0] opening
              && Char.equal after_destination.[String.length after_destination - 1] close
              && String.count (String.drop_prefix after_destination 1) ~f:(Char.equal close) = 1
            in
            closes '"' '"' || closes '\'' '\'' || closes '(' ')'
        | [] -> false)
    | _ -> false
  in
  (* A setext underline turns the line above it into a heading; `===` under a bullet's first line is
     block syntax the canonical form does not have. The `-` form is covered by the break check. *)
  let setext = Char.equal c '=' && String.count body ~f:(Fn.non (Char.equal '=')) = 0 in
  (not thematic) && (not fence) && (not marker) && (not ordered) && (not link_reference)
  && (not setext)
  (* Nor may it start with whitespace: a marker followed by two spaces, or by a tab, indents its
     content past the one column this grammar names. *)
  && (not (Char.is_whitespace c))
  (* A leading `#` or `<` is refused only where it actually opens a block, which the recognizers
     already decide: `#include` is no heading and `<5 ms` no HTML block, and refusing either would
     fail the gate over ordinary prose. A block quote has no such lookalike -- `>` opening a line's
     text always opens one. *)
  (* A leading `|` is a GitHub table row: block syntax the canonical form does not have, and one
     whose cells would otherwise supply a citation from inside a table nobody declared. *)
  && (not (Char.equal c '|'))
  && (not (Option.is_some (heading_level text)))
  && (not (Option.is_some (opens_raw_html text)))
  && not (Char.equal c '>')

(** A bullet: [- ] at column zero, ONE space, then text that opens no block of its own. *)
let is_bullet line =
  String.is_prefix line ~prefix:bullet_marker
  && is_plain_text (String.drop_prefix line (String.length bullet_marker))

(** A continuation line: exactly two spaces, then text that opens no block of its own. Three spaces
    is not a continuation and four is a code block; a marker or a heading at two is a nested block
    whose extent this grammar does not describe. All are refused rather than guessed at, which is
    what keeps the extent exact. *)
let is_continuation line =
  String.is_prefix line ~prefix:continuation_indent
  && (not (String.is_prefix line ~prefix:(continuation_indent ^ " ")))
  && is_plain_text (String.drop_prefix line (String.length continuation_indent))

(** A subheading: [### Added] and deeper, at column zero. It separates bullets and never continues
    one. *)
let is_subheading line =
  indent_of line = 0 && match heading_level line with Some level -> level >= 3 | None -> false

let is_canonical line =
  is_blank line || is_subheading line || is_bullet line || is_continuation line

(** The bullets of a canonical section. Under the gate the extent is exact rather than inferred: an
    item runs to the first line that is neither a continuation nor a blank run followed by one. *)
let parse_section lines =
  let rec item acc rest =
    match rest with
    | line :: tail when is_continuation line -> item (line :: acc) tail
    | line :: _ when is_blank line -> (
        let blanks, after = List.split_while rest ~f:is_blank in
        match after with
        | next :: _ when is_continuation next -> item (List.rev_append blanks acc) after
        | _ -> (List.rev acc, rest))
    | _ -> (List.rev acc, rest)
  in
  let rec loop bullets others = function
    | [] -> (List.rev bullets, List.rev others)
    | line :: rest when is_bullet line ->
        let continuation, rest = item [] rest in
        loop ({ first_line = line; lines = line :: continuation } :: bullets) others rest
    | line :: rest -> loop bullets (line :: others) rest
  in
  loop [] [] lines

let bullets_of lines = fst (parse_section lines)

(** Continuation lines no bullet took: a two-space line under a subheading, or after a blank run
    that ended the item above it. It passes the gate line by line and then belongs to nothing —
    neither rule ever reads it — so the gate has to ask the parse, not just the lines. *)
let orphan_continuations lines = List.filter (snd (parse_section lines)) ~f:is_continuation

let max_lines = 3
let within_line_budget bullet = List.count bullet.lines ~f:(Fn.non is_blank) <= max_lines
let development_repo = "lukstafi/ocannl-staging"

(** Where a number ENDS matters as much as that one starts: [gh-ocannl-807oops] is a typo, not a
    record anyone can follow, and a predicate satisfied by the first digit accepts it. A number is a
    full digit run closed by a token boundary — anything but an alphanumeric, a dash or an
    underscore, the end of the text included. *)
let token_boundary text k =
  k >= String.length text
  ||
  let c = text.[k] in
  not (Char.is_alphanum c || Char.equal c '-' || Char.equal c '_')

(** A record number: a digit run closed by a token boundary, and not zero — GitHub numbers issues
    and pull requests from one, so `gh-ocannl-0` identifies nothing a reader can open. *)
let number_at text k =
  let tlen = String.length text in
  let rec digits j = if j < tlen && Char.is_digit text.[j] then digits (j + 1) else j in
  let stop = digits k in
  stop > k && token_boundary text stop
  && String.count (String.sub text ~pos:k ~len:(stop - k)) ~f:(Fn.non (Char.equal '0')) > 0

(** Whether [text] carries [prefix] followed by a well-formed number. Bare "gh-ocannl-" prose with
    no number is not a citation, which is why the number is part of the question. *)
let starts_token text i = i = 0 || token_boundary text (i - 1)

let cites_number text ~prefix =
  let plen = String.length prefix and tlen = String.length text in
  let rec at i =
    if i + plen >= tlen then false
    else if
      starts_token text i
      && String.equal (String.sub text ~pos:i ~len:plen) prefix
      && number_at text (i + plen)
    then true
    else at (i + 1)
  in
  at 0

(** Whether [text] cites a pull request of the development repository AS ONE CONSTRUCT: the
    repository name immediately qualifying the number, across an optional closing backtick and
    whitespace ([`lukstafi/ocannl-staging` PR #601]) or directly ([lukstafi/ocannl-staging#601]).
    Two independent substring hits would let a sentence naming the repository qualify any other
    project's `PR #NNN`. *)
let pr_prefix = "PR #"

let cites_staging_pr text =
  let repo = development_repo in
  let rlen = String.length repo and tlen = String.length text in
  let qualifies j =
    if j < tlen && Char.equal text.[j] '#' then number_at text (j + 1)
    else
      let j = if j < tlen && Char.equal text.[j] '`' then j + 1 else j in
      let rec skip k = if k < tlen && Char.is_whitespace text.[k] then skip (k + 1) else k in
      let k = skip j in
      (* At least one space: `lukstafi/ocannl-stagingPR #601` is neither documented form, and
         letting the skip consume nothing turned that typo into a citation (Codex P2, round 5). *)
      k > j
      && k + String.length pr_prefix < tlen
      && String.equal (String.sub text ~pos:k ~len:(String.length pr_prefix)) pr_prefix
      && number_at text (k + String.length pr_prefix)
  in
  let rec at i =
    if i + rlen > tlen then false
    else if
      starts_token text i
      && String.equal (String.sub text ~pos:i ~len:rlen) repo
      && qualifies (i + rlen)
    then true
    else at (i + 1)
  in
  at 0

(** A bullet's VISIBLE text: its lines joined, with HTML comments removed. A comment renders as
    nothing, so `- An uncited change <!-- gh-ocannl-807 -->` reads as uncited to every reader, and a
    citation found only inside one is not a citation. Comments are stripped rather than refused
    (round 11 refused them): a code span can wrap across a bullet's lines, [- Explain `<!--] then
    [  syntax` (gh-ocannl-807).], and a per-line refusal failed that legitimate entry. Code spans
    are honoured here, so backticked comment syntax is ordinary text and the citation beside it
    counts. *)
let visible_text bullet =
  let joined = String.concat ~sep:" " bullet.lines in
  let n = String.length joined in
  let buffer = Buffer.create n in
  let rec ticks i c = if i < n && Char.equal joined.[i] c then ticks (i + 1) c else i in
  let rec loop i =
    if i >= n then ()
    else if Char.equal joined.[i] '`' then (
      (* Inside a code span nothing opens a comment; an unclosed run opens no span. *)
      let opening = ticks i '`' in
      let width = opening - i in
      let rec find j =
        if j >= n then None
        else if Char.equal joined.[j] '`' then
          let closing = ticks j '`' in
          if closing - j = width then Some closing else find closing
        else find (j + 1)
      in
      match find opening with
      | Some closing ->
          Buffer.add_string buffer (String.sub joined ~pos:i ~len:(closing - i));
          loop closing
      | None ->
          Buffer.add_string buffer (String.drop_prefix joined i);
          ())
    else if i + 4 <= n && String.equal (String.sub joined ~pos:i ~len:4) "<!--" then
      match String.substr_index joined ~pos:i ~pattern:"-->" with
      | Some close -> loop (close + 3)
      | None -> ()
    else if
      (* Inline tag markup renders as nothing a reader can follow either: a citation in `<span
         data-issue="gh-ocannl-807">` is an attribute, not text. A `<` that opens no tag -- `<5 ms`
         -- is ordinary prose and stays. *)
      Char.equal joined.[i] '<'
      && i + 1 < n
      && (Char.is_alpha joined.[i + 1]
         || (Char.equal joined.[i + 1] '/' || Char.equal joined.[i + 1] '!')
            && i + 2 < n
            && Char.is_alpha joined.[i + 2])
    then (
      (* The tag ends at a `>` OUTSIDE its attribute quotes: `<span title="> gh-ocannl-807">` ends
         at the second one, and stopping at the first would leave the attribute's text visible. *)
      let rec close j quote =
        if j >= n then None
        else
          match quote with
          | Some q -> close (j + 1) (if Char.equal joined.[j] q then None else quote)
          | None ->
              if Char.equal joined.[j] '>' then Some j
              else if Char.equal joined.[j] '"' || Char.equal joined.[j] '\'' then
                close (j + 1) (Some joined.[j])
              else close (j + 1) None
      in
      match close i None with
      | Some stop -> loop (stop + 1)
      | None ->
          Buffer.add_char buffer joined.[i];
          loop (i + 1))
    else if
      (* A link DESTINATION renders as a target, not as text: `[details](…/gh-ocannl-807)` shows the
         reader "details", and a record named only in the URL is one they cannot see. The link text
         itself is kept -- it is what the entry says. *)
      Char.equal joined.[i] '('
      && Buffer.length buffer > 0
      && Char.equal (Buffer.nth buffer (Buffer.length buffer - 1)) ']'
    then (
      match String.index_from joined i ')' with
      | Some close -> loop (close + 1)
      | None ->
          Buffer.add_char buffer joined.[i];
          loop (i + 1))
    else (
      Buffer.add_char buffer joined.[i];
      loop (i + 1))
  in
  loop 0;
  Buffer.contents buffer

let cites_record bullet =
  let text = visible_text bullet in
  cites_number text ~prefix:"gh-ocannl-" || cites_staging_pr text

let mentions bullet substring = String.is_substring (String.concat ~sep:" " bullet.lines) ~substring

let opening bullet =
  let text = String.strip bullet.first_line in
  if String.length text <= 78 then text else String.prefix text 78 ^ "..."

let report label offenders =
  List.iter offenders ~f:(fun bullet -> eprintf "  %s: %s\n" label (opening bullet))

(** {1 Synthetic controls}

    Text built to break each rule, so a checker that stopped deciding cannot pass by finding
    nothing. The shapes here are the ones eight rounds of review named on
    lukstafi/ocannl-staging#603; under the gate most of them are no longer parser branches but
    entries in [non_canonical], which is the point. *)

let control_section =
  [
    "";
    "### Added";
    "";
    "- A compliant bullet, one line, citing (gh-ocannl-807).";
    "- A bullet citing the development repository rather than an issue";
    "  (`lukstafi/ocannl-staging` PR #601).";
    "- A bullet whose continuation lines carry it past the budget, line two of the four";
    "  it is going to take, line three of the four it is going to take, and line four";
    "  which is where it exceeds what the convention allows for one entry, and it does";
    "  cite (gh-ocannl-807) so only the length rule can flag it.";
    "- A bullet of three lines, the second of them here, and the third of them here,";
    "  which then continues after a blank line into a second paragraph";
    "";
    "  that carries the fourth and fifth lines past the budget, and where the whole";
    "  citation (gh-ocannl-807) sits, so a reader that stopped at the blank line sees";
    "  neither the length nor the citation.";
    "- A bullet with no record behind it at all, and no OCANNL record anywhere.";
    "- A bullet saying development happens in `lukstafi/ocannl-staging`, and separately";
    "  mentioning that dependency Foo PR #123 changed something.";
    "- A bullet naming somebody else's pull request, Foo PR #123, and nothing of ours.";
    "- A bullet whose number runs into a typo, gh-ocannl-807oops, and cites nothing else.";
    "- A bullet citing `lukstafi/ocannl-staging` PR #601oops, and nothing else.";
    "- A bullet citing lukstafi/ocannl-stagingPR #601, with the separator missing.";
    "- A bullet citing notgh-ocannl-807, where the prefix is embedded in another token.";
    "- A bullet whose only citation hides in a comment <!-- gh-ocannl-807 -->";
    "- A bullet explaining `<!--";
    "  syntax` across a wrapped code span (gh-ocannl-807).";
    "- A bullet whose citation hides in <span data-issue=\"gh-ocannl-807\">an attribute</span>.";
    "- A bullet citing gh-ocannl-0, a number no record has.";
    "- A bullet citing `lukstafi/ocannl-staging` PR #0, which is no pull request either.";
    "- A bullet whose record hides in a link [destination](https://example.invalid/gh-ocannl-807).";
    "- A bullet hiding one in <span title=\"> gh-ocannl-807\">a quoted attribute</span>.";
  ]

(** Every shape the gate refuses, each one a defect an earlier round found in the parser this file
    no longer has. Refusing them is what makes the remaining imprecision fail loudly. *)
let non_canonical =
  [
    ("an asterisk marker", "* An entry marked with an asterisk.");
    ("a plus marker", "+ An entry marked with a plus.");
    ("a double-spaced marker", "-  An entry whose marker carries two spaces.");
    ("a tab-separated marker", "-\tAn entry whose marker carries a tab.");
    ("a marker alone on its line", "-");
    ("an ordered item", "1. An ordered entry.");
    ("an ordered item with a paren", "2) Another ordered entry.");
    ("an indented marker", "  - An indented marker, top-level or nested?");
    ("a thematic break", "* * *");
    ("a thematic break of dashes", "- - -");
    ("a block quote", "> A quote carrying gh-ocannl-807.");
    ("a lazy continuation", "an unindented line that is not a bullet");
    ("a fenced code block", "```markdown");
    ("an HTML comment", "<!-- a comment -->");
    ("a raw HTML block", "<pre>");
    ("a three-space indent", "   not quite a continuation");
    ("a four-space indent", "    an indented code block");
    ("an indented heading", "  ## Details");
    ("a nested ordered item", "  1. A nested ordered entry.");
    ("a nested ordered item with a paren", "  2) Another nested one.");
    ("a table row", "  | Change | behavior (gh-ocannl-807) |");
    ("a table row opening a bullet", "- | Change |");
    ("a link reference definition", "  [record]: gh-ocannl-807");
    ("a link reference definition with a title", "  [record]: gh-ocannl-807 \"the record\"");
    ("a setext underline", "  ===");
  ]

(* A continuation line with no bullet above it: canonical line by line, and read by neither rule. *)
let control_orphan_continuation = [ "### Added"; ""; "  Added uncited behaviour." ]

let canonical_shapes =
  [
    "";
    "### Added";
    "#### Security";
    "- An entry (gh-ocannl-807).";
    "  a continuation line";
    (* Prose that only looks structural: a ten-digit number is no ordered marker (CommonMark allows
       nine), and comment syntax inside a code span is an ordinary description. *)
    "- 1234567890. is the external identifier for it (gh-ocannl-807).";
    "- The parser now handles `<!--` in prose (gh-ocannl-807).";
    (* Prose that opens with a bracketed label is not a link-reference definition: the words after
       the destination are no title, so it renders, and refusing it would fail a real entry. *)
    "- [API]: behavior changed (gh-ocannl-807).";
    "- [API]: behavior \"changed\" for users (gh-ocannl-807).";
    (* Prose whose first character only looks structural: `#include` is no heading, `<5 ms` no HTML
       block. The recognizers decide, so neither fails the gate. *)
    "- `#include` ordering is stable now (gh-ocannl-807).";
    "- <5 ms per step on the tuned path (gh-ocannl-807).";
  ]

(* Anchors, and history that must not fail the scan. *)
let control_duplicate_anchors =
  [ "# Changelog"; ""; "## [Unreleased]"; ""; "- one (gh-ocannl-807)."; ""; "## [Unreleased]"; "" ]

let control_inexact_anchors =
  [ "## [Unreleased] (draft)"; ""; "- one (gh-ocannl-807)."; ""; "## [Unreleased] old"; "" ]

let control_indented_anchor = [ "  ## [Unreleased]"; ""; "- one (gh-ocannl-807)." ]

let control_code_indented_anchor =
  [ "## [Unreleased]"; ""; "- one (gh-ocannl-807)."; ""; "    ## [Unreleased]"; "" ]

let released_history quoted =
  [ "## [Unreleased]"; ""; "- one (gh-ocannl-807)."; ""; "## [1.0.1] -- 2026-08-26"; "" ] @ quoted

let control_inert_anchors =
  released_history [ "```markdown"; "## [Unreleased]"; "```"; ""; "<!--"; "## [Unreleased]"; "-->" ]

let control_long_fence = released_history [ "````markdown"; "```"; "## [Unreleased]"; "````" ]
let control_raw_html = released_history [ "<pre>"; "## [Unreleased]"; "</pre>" ]

let control_mismatched_html_close =
  released_history [ "<pre>"; "</script>"; "## [Unreleased]"; "</pre>" ]

let control_one_line_html = released_history [ "<pre>example</pre>"; "## [Unreleased]" ]

(* A generic HTML block whose tag merely STARTS with a raw-HTML tag name: it ends at the blank line,
   so the anchor after it is visible. And an autolink, which is inline Markdown and opens no block
   at all -- the heading right after it is still structure. *)
let control_generic_html = released_history [ "<prelude>"; ""; "## [Unreleased]" ]
let control_autolink = released_history [ "<https://example.com>"; "## [Unreleased]" ]
let control_inline_html = released_history [ "<span>text</span>"; "## [Unreleased]" ]
let control_declaration = released_history [ "<?xml version=\"1.0\"?>"; "## [Unreleased]" ]

let control_cdata =
  released_history [ "<![CDATA["; "## [Unreleased]"; "]]>"; ""; "- still released." ]

(* `<![CDATA[` is case-SENSITIVE: a lowercase one opens no block, so the anchor behind it is
   seen. *)
let control_lowercase_cdata = released_history [ "<![cdata["; "## [Unreleased]" ]
let control_lowercase_declaration = released_history [ "<!note"; "## [Unreleased]" ]

(* A released section headed the setext way: the section above it has to end there too, or its
   bullets are read as Unreleased entries and judged by rules they never lived under. *)
let control_setext_boundary =
  [
    "## [Unreleased]";
    "";
    "- one (gh-ocannl-807).";
    "";
    "Release 1.0";
    "-----------";
    "";
    "- released, uncited, and none of this scan's business.";
  ]

let control_deep_subheading =
  [
    "## [Unreleased]";
    "";
    "### Added";
    "";
    "- one (gh-ocannl-807).";
    "";
    "#### Security";
    "";
    "- two, uncited and inside Unreleased.";
    "";
    "## [1.0.1] -- 2026-08-26";
    "";
    "- released, uncited, and none of this scan's business.";
  ]

let control_code_span_comment =
  [ ""; "### Added"; ""; "- The parser now handles `<!--` in prose (gh-ocannl-807)." ]

let () =
  let path =
    match Array.to_list (Sys.get_argv ()) with
    | _ :: path :: _ -> path
    | _ ->
        (* Not a claim: the run never started, so there is no verdict to record -- the scan was
           invoked wrongly, which the dune rule cannot do. *)
        eprintf "FAILED: expected the path to CHANGES.md as the one argument\n";
        Stdlib.exit 1
  in
  let contents = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all in
  let lines = String.split_lines contents in
  print_endline
    "The `## [Unreleased]` section of CHANGES.md against the editorial convention stated in\n\
     docs/agent-notes/conventions.md (gh-ocannl-807). Released sections are history and are not\n\
     read; the counts scanned go to stderr, since a tally in a golden moves on every editorial\n\
     pass (gh-ocannl-665).\n";
  let anchors = List.length (unreleased_headings lines) in
  if anchors <> 1 then eprintf "  CHANGES.md carries %d `%s` headings\n" anchors unreleased_heading;
  p "CHANGES.md carries exactly one `## [Unreleased]` heading" (anchors = 1);
  let section = Option.value (unreleased_section lines) ~default:[] in
  let bullets = bullets_of section in
  eprintf "Read %d lines of `## [Unreleased]`, %d top-level bullets.\n" (List.length section)
    (List.length bullets);
  (* The gate. Emptiness is the passing case: what it reports is a line this grammar does not name,
     and the section is expected to contain none. Every shape refused here was a silent defect while
     the scan tried to parse it instead. *)
  let uncanonical = List.filter section ~f:(Fn.non is_canonical) in
  List.iter uncanonical ~f:(fun line -> eprintf "  not a canonical line: %s\n" line);
  p_empty "every line in Unreleased is blank, a subheading, a bullet, or a two-space continuation"
    ~over:lines uncanonical;
  let orphans = orphan_continuations section in
  List.iter orphans ~f:(fun line -> eprintf "  a continuation with no bullet above it: %s\n" line);
  p_empty "every continuation line in Unreleased belongs to a bullet" ~over:lines orphans;
  (* Guarded over the LINES of CHANGES.md rather than over the bullets, because this is the one site
     in the file where an empty bullet population is a PASSING case: an editorial pass at release
     prep moves every bullet into the new released section, and the Unreleased section that remains
     is legitimately empty until the next merge. A guard over the bullets would fail the
     release-prep build for having nothing to complain about. What the guard is for -- a scan that
     reports nothing because it read nothing -- is what the file's lines witness, alongside the
     anchor and gate claims above and the synthetic controls below, whose population is fixed and
     non-empty. *)
  let over_budget = List.filter bullets ~f:(Fn.non within_line_budget) in
  report "over three lines" over_budget;
  p_empty "every Unreleased bullet is at most three lines" ~over:lines over_budget;
  let uncited = List.filter bullets ~f:(Fn.non cites_record) in
  report "no gh-ocannl-NNN or `lukstafi/ocannl-staging` PR #NNN citation" uncited;
  p_empty "every Unreleased bullet cites gh-ocannl-NNN or a staging PR #NNN" ~over:lines uncited;
  (* The controls. *)
  let control = bullets_of control_section in
  p "the section reader finds the eighteen synthetic control bullets" (List.length control = 18);
  p_exists "the length rule flags a synthetic four-line bullet" control
    ~f:(Fn.non within_line_budget);
  p_exists "the length rule flags a bullet whose fourth line follows a blank one" control
    ~f:(fun bullet -> (not (within_line_budget bullet)) && List.exists bullet.lines ~f:is_blank);
  p_exists "the citation rule flags a synthetic uncited bullet" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "no OCANNL record anywhere");
  p_exists "the citation rule flags an unqualified `PR #NNN`" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "somebody else's pull request");
  p_exists "the citation rule flags a repository name not qualifying the PR number" control
    ~f:(fun bullet ->
      (not (cites_record bullet))
      && mentions bullet development_repo && mentions bullet "Foo PR #123");
  p_exists "the citation rule flags an issue number running into a typo" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "gh-ocannl-807oops");
  p_exists "the citation rule flags a PR number running into a typo" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "PR #601oops");
  p_exists "the citation rule flags a repository run into `PR #` with no separator" control
    ~f:(fun bullet -> (not (cites_record bullet)) && mentions bullet "stagingPR #601");
  p_exists "the citation rule flags an issue prefix embedded in another token" control
    ~f:(fun bullet -> (not (cites_record bullet)) && mentions bullet "notgh-ocannl-807");
  p_exists "the citation rule flags a citation that hides in an HTML comment" control
    ~f:(fun bullet -> (not (cites_record bullet)) && mentions bullet "hides in a comment");
  p_exists "the citation rule flags a citation that hides in an HTML attribute" control
    ~f:(fun bullet -> (not (cites_record bullet)) && mentions bullet "data-issue");
  p_exists "the citation rule flags a record named only in a link destination" control
    ~f:(fun bullet -> (not (cites_record bullet)) && mentions bullet "example.invalid");
  p_exists "the citation rule flags a record inside a quoted tag attribute" control
    ~f:(fun bullet -> (not (cites_record bullet)) && mentions bullet "a quoted attribute");
  p_exists "the citation rule flags issue number zero" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "gh-ocannl-0,");
  p_exists "the citation rule flags PR number zero" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "PR #0");
  p_exists "a code span wrapped across a bullet's lines is canonical, and its citation counts"
    control ~f:(fun bullet ->
      cites_record bullet
      && mentions bullet "across a wrapped code span"
      && List.for_all bullet.lines ~f:is_canonical);
  (* One positive control per ACCEPTED form, each excluding the other: a single "some bullet passes"
     claim is satisfied by whichever form still works, so breaking one acceptance path outright
     would leave the golden green on the strength of the other's fixture. *)
  p_exists "both rules pass a bullet citing a staging PR" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet && mentions bullet development_repo);
  p_exists "both rules pass a bullet citing a gh-ocannl issue" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet && mentions bullet "(gh-ocannl-807)"
      && not (mentions bullet development_repo));
  (* The gate, both directions. *)
  List.iter non_canonical ~f:(fun (what, line) ->
      if is_canonical line then eprintf "  the gate accepted %s: %s\n" what line);
  p_all "the gate refuses every shape this grammar does not name" non_canonical ~f:(fun (_, line) ->
      not (is_canonical line));
  p_all "the gate accepts every shape it names" canonical_shapes ~f:is_canonical;
  (* No control line opens a bullet -- every bullet's first line is one of the input's lines -- and
     the one continuation is reported as an orphan. *)
  let orphan_control_bullets = bullets_of control_orphan_continuation in
  p_all "the gate refuses a continuation with no bullet above it" control_orphan_continuation
    ~f:(fun line ->
      (not
         (List.exists orphan_control_bullets ~f:(fun bullet -> String.equal bullet.first_line line)))
      && List.length (orphan_continuations control_orphan_continuation) = 1);
  (* The anchor, and released history that must not fail the scan. *)
  let one_anchor_one_bullet control =
    List.length (unreleased_headings control) = 1
    && List.length (bullets_of (Option.value (unreleased_section control) ~default:[])) = 1
  in
  p "the section reader refuses a changelog with two Unreleased anchors"
    (Option.is_none (unreleased_section control_duplicate_anchors)
    && List.length (unreleased_headings control_duplicate_anchors) = 2);
  (* Every anchor found is one of the control's lines, so "no control line is a found anchor" is "no
     anchor was found", quantified over the lines the reader was given. *)
  let inexact_anchors = unreleased_headings control_inexact_anchors in
  p_all "only the exact `## [Unreleased]` line counts as the anchor" control_inexact_anchors
    ~f:(fun line ->
      (not (List.mem inexact_anchors line ~equal:String.equal))
      && Option.is_none (unreleased_section control_inexact_anchors));
  let indented_anchors = unreleased_headings control_indented_anchor in
  p_all "an indented copy of the anchor's text is not the anchor" control_indented_anchor
    ~f:(fun line ->
      (not (List.mem indented_anchors line ~equal:String.equal))
      && Option.is_some (heading_level "  ## [Unreleased]"));
  p "a four-space-indented copy of the anchor is code, not a second anchor"
    (one_anchor_one_bullet control_code_indented_anchor);
  p "an anchor quoted in a fence or a comment is not a second anchor"
    (one_anchor_one_bullet control_inert_anchors);
  p "an anchor quoted inside a longer fence stays inert" (one_anchor_one_bullet control_long_fence);
  p "a backtick fence's info string may not contain a backtick"
    ((not (opens_fence "```lang`option"))
    && Option.is_none (fence_at "```lang`option")
    && Option.is_some (fence_at "```lang")
    && Option.is_some (fence_at "~~~lang~option"));
  p "an anchor inside released raw HTML is not a second anchor"
    (one_anchor_one_bullet control_raw_html);
  p "an unrelated closing tag leaves a raw HTML block open"
    (one_anchor_one_bullet control_mismatched_html_close);
  p "a raw HTML block that closes on its opening line hides nothing after it"
    (List.length (unreleased_headings control_one_line_html) = 2);
  p "a tag merely starting with a raw-HTML name is a generic block, ending at the blank line"
    (List.length (unreleased_headings control_generic_html) = 2);
  p "an autolink opens no HTML block, so the heading after it is still structure"
    (List.length (unreleased_headings control_autolink) = 2);
  p "inline HTML with content after the tag opens no block"
    (List.length (unreleased_headings control_inline_html) = 2);
  p "a declaration runs to its own terminator, and hides the anchor inside it"
    (List.length (unreleased_headings control_declaration) = 2
    && one_anchor_one_bullet control_cdata);
  p "a lowercase cdata opener is not a CDATA block, so the anchor behind it is seen"
    (List.length (unreleased_headings control_lowercase_cdata) = 2);
  p "a lowercase declaration opener is no declaration, so the anchor behind it is seen"
    (List.length (unreleased_headings control_lowercase_declaration) = 2);
  p "a setext heading ends the section, like the ATX form"
    (match bullets_of (Option.value (unreleased_section control_setext_boundary) ~default:[]) with
    | [ bullet ] -> cites_record bullet
    | _ -> false);
  let code_span_bullets = bullets_of control_code_span_comment in
  p_all "a comment opener inside prose or a code span opens no HTML block" control_code_span_comment
    ~f:(fun line ->
      (not (opens_comment_block line))
      && (not (opens_comment_block "- An uncited change <!-- gh-ocannl-807 -->"))
      && match code_span_bullets with [ bullet ] -> cites_record bullet | _ -> false);
  p "a level-4 subheading stays inside Unreleased, and the released section does not"
    (let deep =
       bullets_of (Option.value (unreleased_section control_deep_subheading) ~default:[])
     in
     List.length deep = 2
     && List.exists deep ~f:(fun bullet -> mentions bullet "inside Unreleased")
     && not (List.exists deep ~f:(fun bullet -> mentions bullet "released, uncited")));
  Test_utils.Refusal_control_manifest.print "changelog_unreleased_scan.ml"
