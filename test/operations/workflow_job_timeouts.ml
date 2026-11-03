(* gh-ocannl-605 follow-up: every GitHub Actions job bounds its own runtime.

   A job with no [timeout-minutes] runs to GitHub's 6-hour default, so one hung step spends six
   hours of runner time before anything notices -- a sibling repository lost a month of Actions
   minutes to a single such job. Every job in this repository's workflows was given a ceiling by
   hand; nothing held the next job someone adds to the same rule, and the failure is invisible until
   the bill or the queue shows it.

   This scan reads the workflow files as text, the way [config_usage_scan] does: no YAML library is
   a dependency of this repository, and a scan is not a reason to add one. The grammar it decides is
   the one these files are written in -- block mappings indented with spaces -- and it is
   deliberately narrow: the top-level [jobs:] key, the job keys one level under it, and the keys one
   level under each job. Block scalars ([run: |]) are skipped wholesale, so a shell script that
   happens to contain a colon decides nothing.

   The value is not interpreted. [timeout-minutes: 45] and [timeout-minutes: ${{ matrix.os ==
   'windows-latest' && 240 || 45 }}] are both a ceiling; which ceiling is right for a job is a
   judgment the job's author makes, and this scan only holds that one was made -- before [steps:],
   where the reader of the job looks for it.

   A CALLER JOB is the exception, and a two-sided one: a job whose body is a job-level [uses:]
   calling a reusable workflow may not carry [timeout-minutes] at all -- GitHub rejects the workflow
   if it does -- and the jobs it runs are bounded in the workflow it calls. So a caller job is
   required NOT to declare one, rather than exempted from the question.

   {1 The dispatch bisection floor (gh-ocannl-1853)}

   A [ci.yml] dispatch with [expected_sha] runs the workflow as the dispatched ref spells it, over a
   checkout of the named commit, and a STEP that [uses:] a local action ([./...]) loads that action
   from the checkout -- so a commit older than the action fails the step. The oldest commit every
   such step can load is the bisection floor; docs/agent-notes/build-and-test.md records it with the
   local actions behind it (section "How far back a dispatch reaches"), and this scan holds the
   record to the workflows. Its grammar is a closed allowlist, and what falls outside it is refused
   by name rather than read generously: - A workflow is DISPATCHED when a line of its structure
   (comments and block scalars dropped) contains [expected_sha]. That is decided per FILE, not per
   job: every job of such a workflow counts, whatever its checkout says, so a checkout spelled some
   way this reader does not parse cannot take a job out of the rule. A job that checks out the
   dispatched head instead is then listed needlessly, which is loud and costs a bullet. - A
   step-level local action is a [uses:] key that is not one of its job's own keys, whose value
   (quotes and a trailing comment removed) starts with [./]. A job-level [uses:] calls a reusable
   workflow, which GitHub loads from the dispatched ref and not from the checkout, so it bounds
   nothing. In a dispatched workflow every structural line spelling [uses:] must be read as one of
   those two keys, or the file is refused: a flow mapping ([- { uses: ./x }]) is not read here, so
   it may not pass unread either. - The record is the top-level bullets of that section that begin
   with [`./] (an action entry) or with [Floor:] (the floor entry); every other bullet is prose. An
   action entry is exactly [- `<path>`: YYYY-MM-DD, lukstafi/ocannl-staging#N.] and the floor entry
   exactly [- Floor: YYYY-MM-DD, lukstafi/ocannl-staging#N.]. An entry-shaped bullet in any other
   form is refused, as are a missing or repeated heading, a missing or repeated floor entry, a
   record with no action entry, and a run that was not handed the note at all. What is held: the
   dispatched workflows' step-level local actions, deduplicated, equal the record's action entries,
   compared as sorted lists so a repeated entry is a mismatch; the entries' dates never decrease;
   and the floor entry equals the last action entry. So a new local [uses:] fails until it is listed
   last and the floor moves to it. What is not: that an entry's date and merge are true of the
   history (no git is read here), and an interface change to a listed action, which moves its entry
   by hand. A step's [run:] scripts bound a dispatch further, per job; the note says how to read
   that, and nothing here reads scripts.

   {1 What the reader refuses}

   The two levels this scan reads -- the job keys under [jobs:], and the keys under each job -- must
   present a shape it can DECIDE. Anything else is refused by name rather than read generously: a
   file with no [jobs:] key, an inline [jobs: {}] value, a [jobs:] block with no entries, a job
   whose body is inline, a job with no keys, a line at either level this reader cannot parse as a
   key, and a file indented with tabs. The alternative -- accepting what it cannot parse -- makes a
   file that was never examined indistinguishable from one that passed, which is the failure mode
   this whole scan exists to prevent.

   The live tree is valid, which is exactly why its refusals need synthetic controls: green because
   intact and green because blind are the same output. [control] builds workflow trees that differ
   from a legitimate one in one respect each, runs this same executable over them as a CHILD, and
   claims both the exit status and the diagnostic text -- for every refusal, and for the nearest
   accepted counterpart of each. Capturing the children's streams keeps their [FAIL:] markers out of
   a green suite log, following [ocamlformat_ignore_scan] and [generated_provenance]. *)

open Base
open Stdio
open Verdict.Claims

type line = { number : int; indent : int; text : string }

type job = {
  file : string;
  name : string;
  timeout : (int * string) option;  (** line number and rendered value *)
  steps_line : int option;
  calls_reusable : bool;
  dispatched : bool;  (** its workflow names [expected_sha] *)
  step_uses : (int * string) list;
      (** line number and value of each step-level [uses:], quotes and comment removed *)
}

let strip_cr text = Option.value (String.chop_suffix text ~suffix:"\r") ~default:text

(* A UTF-8 BOM is permitted at the start of a workflow, and left in place it would attach itself to
   the first key's name -- so the file's own `jobs:` would not be one. *)
let strip_bom text = Option.value (String.chop_prefix text ~prefix:"\xef\xbb\xbf") ~default:text

let indent_of text =
  let rec count index =
    if index < String.length text && Char.equal text.[index] ' ' then count (index + 1) else index
  in
  count 0

(* Content lines only: blanks and whole-line comments carry no structure, and a document marker
   ([---]) is not a mapping key. *)
let content_lines text =
  String.split_lines text
  |> List.mapi ~f:(fun index raw ->
      let raw = strip_cr (if index = 0 then strip_bom raw else raw) in
      { number = index + 1; indent = indent_of raw; text = String.rstrip raw })
  |> List.filter ~f:(fun { text; _ } ->
      let stripped = String.strip text in
      (not (String.is_empty stripped))
      && (not (String.is_prefix stripped ~prefix:"#"))
      (* A `%YAML`/`%TAG` directive and a document marker are not content, and a directive left in
         would drag the document's base indentation down to its own column zero. *)
      && (not (String.is_prefix stripped ~prefix:"%"))
      && (not (String.is_prefix stripped ~prefix:"---"))
      && not (String.is_prefix stripped ~prefix:"..."))

(* A key line is [<name>:] or [<name>: <value>]. The name is unquoted in every workflow this
   repository has, and an unrecognized line is simply not a key -- it cannot become one by being
   read more generously. *)
(* A key line is [<name>:] or [<name>: <value>], with the name plain or quoted. Quoting is decided
   HERE and nowhere else, so every level that asks what a key is -- the root mapping, the job keys,
   the keys inside a job, the header that opens a block scalar -- reads the same grammar; a quoted
   spelling handled at one of those sites and not the others is how a valid workflow gets refused
   in one place while passing in another. An unrecognized line is simply not a key: it cannot
   become one by being read more generously, and its caller refuses it by name. *)
let plain_key_char c = Char.is_alphanum c || List.mem [ '_'; '-'; '.' ] c ~equal:Char.equal

let quoted_key stripped =
  let quote = stripped.[0] in
  if not (Char.equal quote '"' || Char.equal quote '\'') then None
  else
    match String.index_from stripped 1 quote with
    | None -> None
    | Some closing ->
        let name = String.sub stripped ~pos:1 ~len:(closing - 1) in
        let rest = String.drop_prefix stripped (closing + 1) in
        if String.is_prefix rest ~prefix:":" then
          Some (name, String.strip (String.drop_prefix rest 1))
        else None

let key_of { text; _ } =
  let stripped = String.strip text in
  if String.is_empty stripped then None
  else
    match quoted_key stripped with
    | Some key -> Some key
    | None -> (
        match String.lsplit2 stripped ~on:':' with
        (* YAML permits separation space before the colon ([jobs :]), so the name is right-trimmed
           before it is judged. *)
        | Some (name, rest)
          when let name = String.rstrip name in
               (not (String.is_empty name)) && String.for_all name ~f:plain_key_char ->
            Some (String.rstrip name, String.strip rest)
        | _ -> None)

let strip_trailing_comment value =
  if String.is_prefix (String.strip value) ~prefix:"#" then ""
  else
    match String.substr_index value ~pattern:" #" with
    | Some position -> String.strip (String.prefix value position)
    | None -> value

(* [run: |] and its relatives introduce a literal block: everything indented under such a key is
   text, not structure. Dropping those lines is what keeps a step's shell script from contributing
   keys of its own. *)
(* [|] and [>] may each carry YAML's two header indicators -- an explicit indentation digit and a
   chomping [+]/[-] -- in either order ([|2-] and [|-2] are both legal). What matters here is only
   that the line OPENS a block, so the indicators are accepted and not interpreted. *)
let is_block_scalar value =
  let value = strip_trailing_comment value in
  let after_style =
    Option.first_some (String.chop_prefix value ~prefix:"|") (String.chop_prefix value ~prefix:">")
  in
  Option.value_map after_style ~default:false
    ~f:
      (String.for_all ~f:(fun c ->
           Char.equal c '-' || Char.equal c '+' || (Char.is_digit c && not (Char.equal c '0'))))

(* Whether a line leaves a quoted scalar open at its end. A quote only QUOTES where a scalar BEGINS
   -- as the first character after a [:] or a sequence [-], or at the start of the line's content.
   Once a plain scalar is under way its quotes are ordinary characters, which is how YAML reads
   [name: Publish the 'next branch] and [name: Don't stop]. Inside a quoted scalar, a backslash
   escapes in the double-quoted form and a doubled quote in the single-quoted one; outside one, a
   [#] preceded by space starts a comment and ends the question. *)
let opens_unterminated_quote text =
  let text = String.strip text in
  let length = String.length text in
  let quote_character c = Char.equal c '"' || Char.equal c '\'' in
  (* [at_scalar_start] is true only for the first character of a scalar: it is set by an indicator
     ([:] or [-]) and survives the whitespace that follows it, and nothing else turns it back on. *)
  let indicator index character =
    (* A [:] indicates only when a space or the line's end follows it, and a [-] only as the line's
       leading sequence marker. Anywhere else both are ordinary plain-scalar text -- which is what
       [name: Publish - 'next branch] is, apostrophe included. *)
    (Char.equal character ':' && (index + 1 >= length || Char.is_whitespace text.[index + 1]))
    || Char.equal character '-' && index = 0
       && index + 1 < length
       && Char.is_whitespace text.[index + 1]
  in
  let rec outside index ~at_scalar_start =
    if index >= length then false
    else
      let character = text.[index] in
      if quote_character character && at_scalar_start then inside (index + 1) character
      else if Char.equal character '#' && index > 0 && Char.is_whitespace text.[index - 1] then
        false
      else
        let at_scalar_start =
          indicator index character || (at_scalar_start && Char.is_whitespace character)
        in
        outside (index + 1) ~at_scalar_start
  and inside index opening =
    if index >= length then true
    else if Char.equal text.[index] '\\' && Char.equal opening '"' then inside (index + 2) opening
    else if Char.equal text.[index] opening then
      if Char.equal opening '\'' && index + 1 < length && Char.equal text.[index + 1] '\'' then
        inside (index + 2) opening
      else outside (index + 1) ~at_scalar_start:false
    else inside (index + 1) opening
  in
  outside 0 ~at_scalar_start:true

(* A block scalar is introduced by a KEY, and that key may itself sit in a sequence entry -- [- run:
   |] is how every step in this repository spells it. Strip the entry markers before asking whether
   the line opens a block, or the scripts under `steps:` are never dropped and their contents are
   read as workflow structure. *)
let rec strip_markers text =
  let stripped = String.lstrip text in
  match String.chop_prefix stripped ~prefix:"-" with
  | Some rest when String.is_prefix rest ~prefix:" " || String.is_empty rest -> strip_markers rest
  | _ -> stripped

let block_scalar_opener line =
  match key_of { line with text = strip_markers line.text } with
  | Some (_, value) -> is_block_scalar value
  | None -> false

(* A [uses:] value as GitHub resolves it: a trailing comment is not part of it, and a quoted
   reference is the same reference -- so quoting cannot take a local action out of the floor
   rule. *)
let uses_value value =
  let value = strip_trailing_comment value in
  let unquoted quote =
    Option.bind (String.chop_prefix value ~prefix:quote) ~f:(String.chop_suffix ~suffix:quote)
  in
  Option.value (Option.first_some (unquoted "\"") (unquoted "'")) ~default:value

let is_local_reference value = String.is_prefix value ~prefix:"./"

let drop_block_scalars lines =
  let skip_below = ref None in
  List.filter lines ~f:(fun line ->
      let inside =
        match !skip_below with
        | Some indent when line.indent > indent -> true
        | Some _ ->
            skip_below := None;
            false
        | None -> false
      in
      if inside then false
      else (
        if block_scalar_opener line then skip_below := Some line.indent;
        true))

(* The lines strictly under [head], i.e. up to the next line at [head]'s indentation or
   shallower. *)
let body_of ~head rest = List.take_while rest ~f:(fun line -> line.indent > head.indent)

(* Why a file cannot be judged, in the words its refusal will use. Keeping the reasons in one type
   is what makes "the reader refuses what it cannot decide" checkable: a new shape adds a
   constructor here and an arm to the controls, rather than a silent acceptance nobody sees. *)
type unreadable =
  | Tab_indentation
  | No_jobs_key
  | Inline_jobs_value
  | Empty_jobs_block
  | Unreadable_key of int
  | Inline_job_body of string
  | Job_without_keys of string
  | Multiline_quoted_scalar of int
  | Unread_uses of int

let explain = function
  | Tab_indentation -> "a line indented with tabs, which this reader measures in spaces"
  | No_jobs_key -> "no top-level `jobs:` key"
  | Inline_jobs_value -> "an inline `jobs:` value rather than a block of job entries"
  | Empty_jobs_block -> "a `jobs:` key with no job entries under it"
  | Unreadable_key line -> Printf.sprintf "line %d, which is not a key this reader can parse" line
  | Inline_job_body name -> Printf.sprintf "an inline body for job `%s`" name
  | Job_without_keys name -> Printf.sprintf "no keys at all under job `%s`" name
  | Multiline_quoted_scalar line ->
      Printf.sprintf "line %d, which opens a quoted scalar this reader will not follow across lines"
        line
  | Unread_uses line ->
      Printf.sprintf
        "line %d, which spells `uses:` in a workflow naming expected_sha but is no `uses:` key \
         this reader can read"
        line

type reading = Jobs of job list | Unreadable of unreadable

let leading_tab text =
  let rec scan index =
    index < String.length text
    && (Char.equal text.[index] '\t' || (Char.equal text.[index] ' ' && scan (index + 1)))
  in
  scan 0

let minimum_indent lines =
  List.min_elt (List.map lines ~f:(fun { indent; _ } -> indent)) ~compare:Int.compare

(* The root mapping's own indentation, which YAML does not require to be column zero: a document
   whose every root key carries two spaces is still a root mapping. Reading "top level" as literal
   zero would refuse such a workflow, and a scan that refuses valid files is one people switch
   off. *)
let jobs_head lines =
  let base = Option.value (minimum_indent lines) ~default:0 in
  List.findi lines ~f:(fun _ line ->
      line.indent = base && match key_of line with Some ("jobs", _) -> true | _ -> false)

(* Reads the two levels this scan decides, and refuses -- by name -- every shape it cannot. The
   refusal is raised rather than returned so that no arm can degrade into an empty job list, which
   is the shape that would make an unexamined file look like a passing one. *)
exception Refuse of unreadable

let jobs_of ~file text =
  let refuse reason = raise (Refuse reason) in
  let required option ~reason = match option with Some value -> value | None -> refuse reason in
  let read () =
    (* Block scalars first: a tab is legal CONTENT (a Makefile recipe in a `run: |` body), and only
       a tab this reader would have to measure as indentation is a refusal. *)
    let lines = drop_block_scalars (content_lines text) in
    if List.exists lines ~f:(fun { text; _ } -> leading_tab text) then refuse Tab_indentation;
    (* A quoted scalar may legally run across lines, and a reader that does not follow it would read
       its continuation as structure -- which is how a decoy `jobs:` inside a string could stand in
       for the real one and let a whole workflow go unexamined. Rather than follow it, this refuses:
       no workflow here writes one, and a loud refusal is what the contract owes a shape the reader
       cannot decide. *)
    (match List.find lines ~f:(fun line -> opens_unterminated_quote line.text) with
    | Some line -> refuse (Multiline_quoted_scalar line.number)
    | None -> ());
    let dispatched =
      List.exists lines ~f:(fun { text; _ } -> String.is_substring text ~substring:"expected_sha")
    in
    (* The line numbers this reader took as a [uses:] key, job-level or step-level: in a dispatched
       workflow, every other structural line spelling [uses:] is refused below. *)
    let read_uses = ref [] in
    let position, head = required (jobs_head lines) ~reason:No_jobs_key in
    (match key_of head with
    | Some (_, value) when not (String.is_empty (strip_trailing_comment value)) ->
        refuse Inline_jobs_value
    | _ -> ());
    let block = body_of ~head (List.drop lines (position + 1)) in
    let job_indent = required (minimum_indent block) ~reason:Empty_jobs_block in
    let rec collect found = function
      | [] -> List.rev found
      | head :: rest when head.indent = job_indent ->
          let body = body_of ~head rest in
          let name, inline = required (key_of head) ~reason:(Unreadable_key head.number) in
          if not (String.is_empty (strip_trailing_comment inline)) then
            refuse (Inline_job_body name);
          let key_indent = required (minimum_indent body) ~reason:(Job_without_keys name) in
          let job_keys =
            (* A block sequence may be written at its key's own indentation ([steps:] then [- name:
               …] both at four), and those entries are the key's VALUE, not further keys of the job.
               `.github/workflows/ci.yml` is written that way. *)
            List.filter body ~f:(fun line ->
                line.indent = key_indent
                && not (String.is_prefix (String.strip line.text) ~prefix:"-"))
            |> List.map ~f:(fun line ->
                let key, value = required (key_of line) ~reason:(Unreadable_key line.number) in
                (line.number, key, value))
          in
          let timeout =
            List.find_map job_keys ~f:(fun (number, key, value) ->
                Option.some_if
                  (String.equal key "timeout-minutes")
                  (number, strip_trailing_comment value))
          in
          let steps_line =
            List.find_map job_keys ~f:(fun (number, key, _) ->
                Option.some_if (String.equal key "steps") number)
          in
          let calls_reusable =
            List.exists job_keys ~f:(fun (_, key, _) -> String.equal key "uses")
          in
          List.iter job_keys ~f:(fun (number, key, _) ->
              if String.equal key "uses" then read_uses := number :: !read_uses);
          (* Everything under the job that is not one of its own keys belongs to a step or to a
             key's value, and a [uses:] there is a step's action. *)
          let step_uses =
            List.filter_map body ~f:(fun line ->
                if
                  line.indent = key_indent
                  && not (String.is_prefix (String.strip line.text) ~prefix:"-")
                then None
                else
                  match key_of { line with text = strip_markers line.text } with
                  | Some ("uses", value) ->
                      read_uses := line.number :: !read_uses;
                      Some (line.number, uses_value value)
                  | _ -> None)
          in
          collect
            ({ file; name; timeout; steps_line; calls_reusable; dispatched; step_uses } :: found)
            rest
      | line :: rest ->
          if line.indent < job_indent then refuse (Unreadable_key line.number)
          else collect found rest
    in
    match collect [] block with
    | [] -> refuse Empty_jobs_block
    | jobs ->
        (if dispatched then
           match
             List.find lines ~f:(fun line ->
                 String.is_substring
                   (strip_trailing_comment (String.strip line.text))
                   ~substring:"uses:"
                 && not (List.mem !read_uses line.number ~equal:Int.equal))
           with
           | Some line -> refuse (Unread_uses line.number)
           | None -> ());
        jobs
  in
  match read () with jobs -> Jobs jobs | exception Refuse reason -> Unreadable reason

let declared_timeout job =
  match job.timeout with
  | None -> None
  | Some (_, value) -> Option.some_if (not (String.is_empty value)) value

(* Two-sided: an ordinary job must declare a ceiling, and a caller job must not. GitHub rejects
   `timeout-minutes` on the job-level `uses:` form, so requiring it there would demand a workflow
   that cannot run -- and for a caller it is the KEY that GitHub rejects, whatever value follows it,
   which is why this side asks about [job.timeout] rather than about a value that survived. *)
let bounded job =
  if job.calls_reusable then Option.is_none job.timeout else Option.is_some (declared_timeout job)

let ahead_of_steps job =
  match (job.timeout, job.steps_line) with
  | Some (number, _), Some steps -> number < steps
  | Some _, None -> true
  | None, _ -> job.calls_reusable

let describe job =
  if job.calls_reusable then
    match job.timeout with
    | None -> "calls a reusable workflow, which carries its own ceilings"
    | Some (_, value) ->
        "calls a reusable workflow AND declares timeout-minutes: "
        ^ if String.is_empty value then "(no value)" else value
  else
    match declared_timeout job with
    | Some value -> "timeout-minutes: " ^ value
    | None -> "NO timeout-minutes"

(* GitHub discovers workflows directly inside `.github/workflows`; a YAML file in a subdirectory
   under it is a fixture or a supporting file that no job ever runs. The globs are recursive because
   that is what makes this a repository-wide scan in dune's eyes, so the depth is decided here. *)
let is_workflow relative =
  match String.chop_prefix relative ~prefix:".github/workflows/" with
  | None -> false
  | Some basename ->
      (not (String.mem basename '/'))
      && (String.is_suffix basename ~suffix:".yml" || String.is_suffix basename ~suffix:".yaml")

(* {1 The bisection-floor record} *)

let floor_note = "docs/agent-notes/build-and-test.md"
let floor_heading = "### How far back a dispatch reaches"
let floor_pointer = "build-and-test.md#how-far-back-a-dispatch-reaches"

(* An entry's date and the master merge it names: [2026-10-23, lukstafi/ocannl-staging#1359]. *)
type mark = { date : string; merge : string }
type record = { actions : (string * mark) list; floor : mark }
type record_reading = Record of record | No_record of string

let is_date text =
  String.length text = 10
  && String.for_alli text ~f:(fun index c ->
      if index = 4 || index = 7 then Char.equal c '-' else Char.is_digit c)
  &&
  let month = Int.of_string (String.sub text ~pos:5 ~len:2)
  and day = Int.of_string (String.sub text ~pos:8 ~len:2) in
  month >= 1 && month <= 12 && day >= 1 && day <= 31

let is_merge text =
  match String.chop_prefix text ~prefix:"lukstafi/ocannl-staging#" with
  | Some number ->
      (not (String.is_empty number))
      && String.for_all number ~f:Char.is_digit
      && not (Char.equal number.[0] '0')
  | None -> false

(* [<date>, <merge>.] and nothing else: a second clause, a missing stop, or a merge in another
   spelling is an entry its author got wrong, and it is refused rather than repaired. *)
let mark_of text =
  match String.chop_suffix text ~suffix:"." with
  | None -> None
  | Some body -> (
      match String.lsplit2 body ~on:',' with
      | Some (date, rest) when is_date date -> (
          match String.chop_prefix rest ~prefix:" " with
          | Some merge when is_merge merge -> Some { date; merge }
          | _ -> None)
      | _ -> None)

let action_entry text =
  match String.chop_prefix text ~prefix:"`" with
  | None -> None
  | Some rest -> (
      match String.lsplit2 rest ~on:'`' with
      | Some (path, after)
        when is_local_reference path && not (String.exists path ~f:Char.is_whitespace) ->
          Option.map
            (Option.bind (String.chop_prefix after ~prefix:": ") ~f:mark_of)
            ~f:(fun mark -> (path, mark))
      | _ -> None)

let floor_entry text = Option.bind (String.chop_prefix text ~prefix:"Floor: ") ~f:mark_of
let entry_spelling = "`- `<path>`: YYYY-MM-DD, lukstafi/ocannl-staging#N.`"
let floor_spelling = "`- Floor: YYYY-MM-DD, lukstafi/ocannl-staging#N.`"

(* The section runs from its heading to the next heading of any level. Its top-level bullets are
   read with their continuation lines joined, the way the agent-notes reader joins them, so an entry
   wrapped across lines is the same entry. *)
let read_record note =
  let lines = List.map (String.split_lines note) ~f:strip_cr in
  let headings =
    List.filter lines ~f:(fun line -> String.equal (String.rstrip line) floor_heading)
  in
  match headings with
  | [] -> No_record (Printf.sprintf "no `%s` heading" floor_heading)
  | _ :: _ :: _ -> No_record (Printf.sprintf "the `%s` heading more than once" floor_heading)
  | [ _ ] -> (
      let numbered = List.mapi lines ~f:(fun index line -> (index + 1, line)) in
      let section =
        List.drop_while numbered ~f:(fun (_, line) ->
            not (String.equal (String.rstrip line) floor_heading))
        |> List.tl_exn
        |> List.take_while ~f:(fun (_, line) -> not (String.is_prefix line ~prefix:"#"))
      in
      let rec bullets found = function
        | [] -> List.rev found
        | (number, line) :: rest when String.is_prefix line ~prefix:"- " ->
            let continuation, rest =
              List.split_while rest ~f:(fun (_, l) ->
                  String.is_prefix l ~prefix:" " && not (String.is_empty (String.strip l)))
            in
            let text =
              String.concat ~sep:" "
                (String.rstrip (String.drop_prefix line 2)
                :: List.map continuation ~f:(fun (_, l) -> String.strip l))
            in
            bullets ((number, text) :: found) rest
        | _ :: rest -> bullets found rest
      in
      let read (actions, floors) (number, text) =
        if String.is_prefix text ~prefix:"`./" then
          match action_entry text with
          | Some entry -> Ok (entry :: actions, floors)
          | None ->
              Error
                (Printf.sprintf "line %d, an action entry not of the form %s" number entry_spelling)
        else if String.is_prefix text ~prefix:"Floor:" then
          match floor_entry text with
          | Some mark -> Ok (actions, mark :: floors)
          | None ->
              Error
                (Printf.sprintf "line %d, a floor entry not of the form %s" number floor_spelling)
        else Ok (actions, floors)
      in
      match List.fold_result (bullets [] section) ~init:([], []) ~f:read with
      | Error reason -> No_record reason
      | Ok (_, []) -> No_record (Printf.sprintf "no floor entry %s" floor_spelling)
      | Ok (_, _ :: _ :: _) -> No_record "more than one floor entry"
      | Ok ([], [ _ ]) ->
          No_record (Printf.sprintf "no action entry %s under the floor" entry_spelling)
      | Ok (actions, [ floor ]) -> Record { actions = List.rev actions; floor })

let describe_mark { date; merge } = date ^ ", " ^ merge

let scan workspace_root paths =
  let base = Test_utils.Dune_stanza_scan.base_dir workspace_root in
  let files =
    List.filter paths ~f:(fun path -> not (Stdlib.Sys.is_directory path))
    |> List.filter_map ~f:(fun path ->
        let relative = Test_utils.Dune_stanza_scan.repo_relative base path in
        Option.some_if (is_workflow relative) (relative, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let scanned =
    List.map files ~f:(fun (relative, path) ->
        (relative, jobs_of ~file:relative (In_channel.read_all path)))
  in
  let jobs =
    List.concat_map scanned ~f:(fun (_, reading) ->
        match reading with Jobs jobs -> jobs | Unreadable _ -> [])
  in
  (* The inventory goes to stderr, where the other repo-wide scans put their censuses: a bounded job
     renamed, or a ceiling tuned from 30 to 45, changes none of the claims below, and a golden that
     moved with it would make every workflow edit promote this file -- a conflict point for branches
     that have nothing to do with the rule. Stdout keeps what is stably true. *)
  eprintf
    "GitHub workflow jobs and the runtime ceiling each one declares (not part of the golden):\n";
  List.iter scanned ~f:(fun (relative, reading) ->
      match reading with
      | Unreadable reason -> eprintf "  %s: UNREADABLE -- %s\n" relative (explain reason)
      | Jobs jobs ->
          List.iter jobs ~f:(fun job -> eprintf "  %s %s: %s\n" relative job.name (describe job)));
  List.iter scanned ~f:(fun (relative, reading) ->
      match reading with
      | Jobs _ -> ()
      | Unreadable reason ->
          eprintf "%s: this scan cannot answer for its jobs -- it found %s\n" relative
            (explain reason));
  List.iter jobs ~f:(fun job ->
      if not (bounded job) then
        if job.calls_reusable then
          eprintf
            "%s: job `%s` calls a reusable workflow and declares `timeout-minutes`, which GitHub \
             rejects on that form\n"
            job.file job.name
        else
          eprintf
            "%s: job `%s` declares no job-level `timeout-minutes`; it would run to GitHub's 6-hour \
             default\n"
            job.file job.name
      else if not (ahead_of_steps job) then
        eprintf "%s: job `%s` declares `timeout-minutes` after its `steps:`\n" job.file job.name);
  p_all ~min:4 "every checked-in GitHub workflow file presents jobs this scan can read" scanned
    ~f:(fun (_, reading) -> match reading with Jobs _ -> true | Unreadable _ -> false);
  p_all
    "every workflow job declares a job-level runtime ceiling, unless it calls a reusable workflow"
    jobs ~f:bounded;
  p_all "every workflow job states that ceiling ahead of the steps it bounds" jobs ~f:ahead_of_steps;
  (* The bisection floor (gh-ocannl-1853): the record against the dispatched workflows' local
     actions. The note is found among the handed paths by its repository path, so a run not handed
     it is refused here rather than reading some other file of that name. *)
  let record =
    match
      List.find paths ~f:(fun path ->
          String.equal (Test_utils.Dune_stanza_scan.repo_relative base path) floor_note)
    with
    | None -> No_record "nothing: the note was not handed to this scan"
    | Some path -> read_record (In_channel.read_all path)
  in
  let used =
    List.concat_map jobs ~f:(fun job ->
        if not job.dispatched then []
        else
          List.filter_map job.step_uses ~f:(fun (number, value) ->
              Option.some_if (is_local_reference value) (value, (job, number))))
  in
  let listed =
    match record with Record { actions; _ } -> List.map actions ~f:fst | No_record _ -> []
  in
  eprintf "Local actions of expected_sha-dispatched workflows (not part of the golden):\n";
  List.iter used ~f:(fun (value, (job, number)) ->
      eprintf "  %s:%d job %s: %s\n" job.file number job.name value);
  (match record with
  | No_record reason ->
      eprintf "%s: no bisection-floor record this scan can read -- it found %s (%s)\n" floor_note
        reason floor_pointer
  | Record { actions; floor } ->
      List.iter actions ~f:(fun (path, mark) ->
          eprintf "  recorded: %s since %s\n" path (describe_mark mark));
      eprintf "  recorded floor: %s\n" (describe_mark floor));
  p "the bisection-floor record in docs/agent-notes/build-and-test.md is one this scan can read"
    (match record with Record _ -> true | No_record _ -> false);
  let identities = List.dedup_and_sort ~compare:String.compare (List.map used ~f:fst @ listed) in
  p_all
    "every local action a step of an expected_sha-dispatched workflow uses is listed once in the \
     bisection-floor record, and every one listed is used"
    identities ~f:(fun path ->
      let times = List.count listed ~f:(String.equal path) in
      (match List.find used ~f:(fun (value, _) -> String.equal value path) with
      | Some (_, (job, number)) when times = 0 ->
          eprintf
            "%s:%d: job `%s` uses the local action `%s`, which the bisection-floor record does not \
             list: an expected_sha dispatch of a commit older than it fails that step -- list it \
             last in %s and move the floor to it (%s)\n"
            job.file number job.name path floor_note floor_pointer
      | Some _ -> ()
      | None ->
          eprintf "%s lists `%s`, which no step of an expected_sha-dispatched workflow uses\n"
            floor_note path);
      if times > 1 then eprintf "%s lists `%s` %d times\n" floor_note path times;
      times = 1 && List.exists used ~f:(fun (value, _) -> String.equal value path));
  p "the bisection-floor record lists its actions oldest first, and its floor is the newest one's"
    (match record with
    | No_record _ -> false
    | Record { actions; floor } ->
        let rec ordered = function
          | (above, earlier) :: ((path, later) :: _ as rest) ->
              if String.( < ) later.date earlier.date then (
                eprintf
                  "%s: the bisection-floor entry for `%s` (%s) is older than the one above it, \
                   `%s` (%s); entries go oldest first\n"
                  floor_note path (describe_mark later) above (describe_mark earlier);
                ignore (ordered rest : bool);
                false)
              else ordered rest
          | _ -> true
        in
        let in_order = ordered actions in
        let newest_path, newest = List.last_exn actions in
        let at_newest =
          String.equal floor.date newest.date && String.equal floor.merge newest.merge
        in
        if not at_newest then
          eprintf
            "%s: the bisection floor (%s) is not the newest entry's (`%s`: %s) -- the floor moves \
             with the entry that raised it\n"
            floor_note (describe_mark floor) newest_path (describe_mark newest);
        in_order && at_newest);
  eprintf "Scanned %d workflow files and %d jobs.\n" (List.length files) (List.length jobs)

(* {1 Synthetic controls} *)

let write_file path data =
  let rec mkdirs dir =
    if not (String.equal dir Stdlib.Filename.current_dir_name || Stdlib.Sys.file_exists dir) then (
      mkdirs (Stdlib.Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdirs (Stdlib.Filename.dirname path);
  Out_channel.write_all path ~data

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let sound_job ~name =
  Printf.sprintf
    "  %s:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - run: echo hi\n"
    name

let workflow jobs = "name: fixture\non: workflow_dispatch\njobs:\n" ^ String.concat jobs

(* A workflow taking [expected_sha] the way [ci.yml] does: the input, and a checkout naming it. *)
let dispatched_workflow jobs =
  "name: dispatched fixture\n\
   on:\n\
  \  workflow_dispatch:\n\
  \    inputs:\n\
  \      expected_sha:\n\
  \        required: false\n\
   jobs:\n" ^ String.concat jobs

let dispatched_job ?(checkout = true) ~name steps =
  Printf.sprintf "  %s:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n%s%s" name
    (if checkout then
       "    - uses: actions/checkout@v4\n      with:\n        ref: ${{ inputs.expected_sha }}\n"
     else "")
    (String.concat steps)

let dispatched_filler =
  dispatched_workflow
    [ dispatched_job ~name:"build" [ "    - uses: ./.github/actions/filler-action\n" ] ]

(* A note whose record lists the filler's action, then [entries] (bullet lines, without the floor),
   then the floor -- by default the filler's own mark. Prose bullets on either side, one of them
   naming an action in passing, and an entry-shaped bullet under the NEXT heading, which the
   section's end must keep out of the record. *)
let floor_note_text ?(floor = "- Floor: 2026-01-01, lukstafi/ocannl-staging#1.\n") entries =
  "# Build and test mechanics\n\n\
   ### How far back a dispatch reaches\n\n\
   - Prose naming `./.github/actions/prose-only` in passing, which no entry is.\n\
   - `./.github/actions/filler-action`: 2026-01-01, lukstafi/ocannl-staging#1.\n"
  ^ String.concat entries ^ floor
  ^ "- Prose after the record.\n\n\
     ## The next section\n\n\
     - `./.github/actions/outside-the-section`: 2026-01-01, lukstafi/ocannl-staging#9.\n"

(* The child is run FROM the fixture root with the workflow paths spelled relative to it, which is
   the shape dune's own action has: [%{workspace_root}] and paths that resolve against it. Handing
   absolute paths instead would make [repo_relative] answer with a filesystem path that starts at
   the temporary directory, and every fixture would look like a tree containing no workflows at
   all. *)
(* Relative workflow paths must resolve against the fixture root. *)
let run_child ~exe ~root ~files = Fresh_process.run ~cwd:root ~exe ("--scan-only" :: "." :: files)

(* Every fixture tree carries the same three sound workflows beside the one under test, so the
   file-population floor the live run makes is the floor the controls run against too: a control
   that quietly ran a weaker claim than the shipping one would prove nothing about the shipping
   one. *)
let control () =
  let exe = Fresh_process.executable () in
  let fixture = Stdlib.Filename.temp_dir "workflow timeouts control " "" in
  let case_index = ref 0 in
  (* One filler is a `.yaml`, so the accepted arms exercise both discovered spellings; a refusing
     arm names its subject `.yaml` too, which is what makes discovery of that spelling load-bearing
     rather than incidental. *)
  (* The fourth filler is a dispatched workflow whose one local action the default note records, so
     every tree meets the floor claims' population floor the way the live one does; a floor arm
     varies the subject or the note, never this pair. *)
  let run ?(extra = []) ?(subject_name = "subject.yml") ?(note = Some (floor_note_text [])) subject
      =
    Int.incr case_index;
    let root = Stdlib.Filename.concat fixture (Printf.sprintf "case%d" !case_index) in
    let workflows = Stdlib.Filename.concat root ".github/workflows" in
    let fillers = [ "filler1.yml"; "filler2.yml"; "filler3.yaml" ] in
    List.iter fillers ~f:(fun name ->
        write_file (Stdlib.Filename.concat workflows name) (workflow [ sound_job ~name:"build" ]));
    write_file (Stdlib.Filename.concat workflows "dispatched.yml") dispatched_filler;
    write_file (Stdlib.Filename.concat workflows subject_name) subject;
    List.iter extra ~f:(fun (path, content) ->
        write_file (Stdlib.Filename.concat root path) content);
    Option.iter note ~f:(fun text -> write_file (Stdlib.Filename.concat root floor_note) text);
    let files =
      (subject_name :: "dispatched.yml" :: fillers
      |> List.map ~f:(Stdlib.Filename.concat ".github/workflows"))
      @ List.map extra ~f:fst
      @ Option.value_map note ~default:[] ~f:(fun _ -> [ floor_note ])
    in
    run_child ~exe ~root ~files
  in
  let report label child = Fresh_process.report ~label child in
  let passed label ((status, _, _) as child) =
    let ok = match status with Unix.WEXITED 0 -> true | _ -> false in
    if not ok then report label child;
    ok
  in
  (* Split so the emptiness guard sits in the boolean this returns: a refusal nobody named is not
     one this control observed, so an empty message list fails rather than satisfying [for_all]
     vacuously. *)
  let matched_refusal ~messages child =
    (not (List.is_empty messages)) && Fresh_process.matches ~exit:1 ~contains:messages child
  in
  let refused label ~messages outcome =
    let ok = matched_refusal ~messages outcome in
    if not ok then report label outcome;
    ok
  in
  let missing_timeout ?(file = "subject.yml") name =
    Printf.sprintf ".github/workflows/%s: job `%s` declares no job-level `timeout-minutes`" file
      name
  in
  let unreadable reason =
    Printf.sprintf "this scan cannot answer for its jobs -- it found %s" reason
  in
  let unbounded_job name =
    Printf.sprintf "  %s:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n" name
  in
  let legitimate =
    run
      (workflow
         [
           "  matrixed:\n\
           \    runs-on: ${{ matrix.os }}\n\
           \    timeout-minutes: ${{ matrix.os == 'windows-latest' && 240 || 45 }}\n\
           \    steps:\n\
           \      - run: |\n\
           \          echo 'a block scalar mentioning timeout-minutes: 1'\n";
           "  caller:\n    uses: ./.github/workflows/filler1.yml\n";
           (* The shape `ci.yml` is written in: the sequence under `steps:` starts at the key's own
              indentation rather than deeper. *)
           "  flush_sequence:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 15\n\
           \    steps:\n\
           \    - name: one\n\
           \      run: echo hi\n";
           sound_job ~name:"plain";
         ])
  in
  let no_timeout =
    run
      (workflow [ "  unbounded:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n" ])
  in
  let after_steps =
    run
      (workflow
         [
           "  late:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \      - run: echo hi\n\
           \    timeout-minutes: 9\n";
         ])
  in
  let block_scalar_decoy =
    run
      (workflow
         [
           "  decoy:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \      - run: |\n\
           \          echo 'timeout-minutes: 30'\n";
         ])
  in
  let step_level_decoy =
    run
      (workflow
         [
           "  stepwise:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \      - name: bounded step\n\
           \        timeout-minutes: 5\n\
           \        run: echo hi\n";
         ])
  in
  let caller_with_timeout =
    run
      (workflow
         [ "  caller:\n    uses: ./.github/workflows/filler1.yml\n    timeout-minutes: 30\n" ])
  in
  (* The key present with nothing behind it: a declaration that names no ceiling, against its
     nearest counterpart, a real ceiling wearing a trailing comment. *)
  let empty_value =
    run
      (workflow
         [
           "  hollow:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes:\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let comment_only_value =
    run
      (workflow
         [
           "  deferred:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: # decide later\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let commented_ceiling =
    run
      (workflow
         [
           "  annotated:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 20 # ~2min normally; a ceiling only a hang can reach\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let caller_with_empty_timeout =
    run
      (workflow [ "  caller:\n    uses: ./.github/workflows/filler1.yml\n    timeout-minutes:\n" ])
  in
  let yaml_subject = run ~subject_name:"subject.yaml" (workflow [ unbounded_job "unbounded" ]) in
  let flush_sequence_unbounded =
    run
      (workflow
         [
           "  flush_sequence:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \    - name: one\n\
           \      run: echo hi\n";
         ])
  in
  let no_jobs = run "name: fixture\non: workflow_dispatch\n" in
  (* The shapes the reader refuses rather than reads generously. Each is a file that would otherwise
     be examined by nobody while every claim stayed green. *)
  let inline_jobs = run "name: fixture\non: workflow_dispatch\njobs: {}\n" in
  let empty_jobs_block = run "name: fixture\non: workflow_dispatch\njobs:\n" in
  let inline_job_body =
    run "name: fixture\non: workflow_dispatch\njobs:\n  build: { runs-on: ubuntu-latest }\n"
  in
  let job_without_keys = run "name: fixture\non: workflow_dispatch\njobs:\n  build:\n" in
  let quoted_job_key =
    run
      ("name: fixture\non: workflow_dispatch\njobs:\n  \"quoted job\":\n"
     ^ "    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - run: echo hi\n")
  in
  let quoted_job_key_unbounded =
    run
      ("name: fixture\non: workflow_dispatch\njobs:\n  \"quoted job\":\n"
     ^ "    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n")
  in
  let unreadable_job_key =
    run "name: fixture\non: workflow_dispatch\njobs:\n  - a sequence entry where a job belongs\n"
  in
  (* A line inside an otherwise valid job -- ceiling and all -- that is no key in any spelling. A
     reader that skipped it would report that job as fully examined. The counterpart is the same job
     with a real key in its place. *)
  let unreadable_key_in_job =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    a line that is no key at all\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let readable_key_in_job =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    continue-on-error: false\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let quoted_block_scalar_key =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    steps:\n\
           \      - \"run\": |\n\
           \          all:\n\
           \          \techo hi\n";
         ])
  in
  let directive_header =
    run
      ("%YAML 1.2\n---\n  name: fixture\n  on: workflow_dispatch\n  jobs:\n"
     ^ "    build:\n\
       \      runs-on: ubuntu-latest\n\
       \      timeout-minutes: 10\n\
       \      steps:\n\
       \        - run: echo hi\n")
  in
  (* A decoy `jobs:` inside a multiline quoted scalar is the one shape that could have made a real
     workflow go UNEXAMINED rather than merely refused, so it is refused by name; the counterpart is
     the same field on one line. The BOM and document-end arms are the other direction: valid files
     this reader must not turn away. *)
  let multiline_quoted_scalar =
    run
      ("name: \"a description that runs on\n  jobs:\n    decoy:\n      timeout-minutes: 1\"\n"
     ^ "on: workflow_dispatch\njobs:\n" ^ unbounded_job "unbounded")
  in
  let single_line_quoted_scalar =
    run
      ("name: \"a description with a colon: and quotes\"\non: workflow_dispatch\njobs:\n"
     ^ sound_job ~name:"build")
  in
  let apostrophe_in_plain_scalar =
    run ("name: Don't stop the build\non: workflow_dispatch\njobs:\n" ^ sound_job ~name:"build")
  in
  let quote_mid_plain_scalar =
    run ("name: Publish the 'next branch\non: workflow_dispatch\njobs:\n" ^ sound_job ~name:"build")
  in
  let hyphen_in_plain_scalar =
    run ("name: Publish - 'next branch\non: workflow_dispatch\njobs:\n" ^ sound_job ~name:"build")
  in
  let spaced_keys =
    run
      ("name : fixture\non : workflow_dispatch\njobs :\n"
     ^ "  build :\n\
       \    runs-on : ubuntu-latest\n\
       \    timeout-minutes : 10\n\
       \    steps :\n\
       \      - run : echo hi\n")
  in
  let byte_order_mark = run ("\xef\xbb\xbf" ^ workflow [ sound_job ~name:"build" ]) in
  let document_end_marker =
    run
      ("---\n  name: fixture\n  on: workflow_dispatch\n  jobs:\n"
     ^ "    build:\n\
       \      runs-on: ubuntu-latest\n\
       \      timeout-minutes: 10\n\
       \      steps:\n\
       \        - run: echo hi\n\
        ...\n")
  in
  let tab_inside_block_scalar =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    steps:\n\
           \      - run: |\n\
           \          cat > Makefile <<'EOF'\n\
           \          all:\n\
           \          \techo hi\n\
           \          EOF\n";
         ])
  in
  (* A block header carrying YAML's indentation and chomping indicators, and a root mapping the
     document indents uniformly. Neither is how these workflows are written, and both are valid YAML
     that GitHub accepts -- a scan that refused them would be a scan people switch off. *)
  let indicator_block_scalar =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    steps:\n\
           \      - run: |2-\n\
           \          all:\n\
           \          \techo hi\n";
         ])
  in
  let indented_root_mapping =
    run
      ("  name: fixture\n  on: workflow_dispatch\n  jobs:\n"
     ^ "    build:\n\
       \      runs-on: ubuntu-latest\n\
       \      timeout-minutes: 10\n\
       \      steps:\n\
       \        - run: echo hi\n")
  in
  let tab_indented =
    run "name: fixture\non: workflow_dispatch\njobs:\n\tbuild:\n\t\truns-on: ubuntu-latest\n"
  in
  (* A YAML file in a subdirectory of `.github/workflows` is not a workflow: GitHub discovers only
     the files sitting directly there. The recursive glob hands this one over anyway, which is what
     makes the depth check load-bearing rather than decorative. *)
  let nested_fixture =
    run
      ~extra:
        [
          ( ".github/workflows/fixtures/not_a_workflow.yml",
            workflow
              [ "  unbounded:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n" ] );
        ]
      (workflow [ sound_job ~name:"plain" ])
  in
  (* The bisection floor (gh-ocannl-1853). The planted negative control is a dispatched job that
     starts using a local action the record does not list; its accepted counterpart lists it last
     and moves the floor, and the arms between take one of those two steps without the other. *)
  let planted_step = "    - uses: ./.github/actions/planted\n" in
  let planted = dispatched_workflow [ dispatched_job ~name:"bisectable" [ planted_step ] ] in
  let planted_entry date =
    Printf.sprintf "- `./.github/actions/planted`: %s, lukstafi/ocannl-staging#2.\n" date
  in
  let moved_floor = "- Floor: 2026-02-01, lukstafi/ocannl-staging#2.\n" in
  let unlisted job =
    Printf.sprintf
      "job `%s` uses the local action `./.github/actions/planted`, which the bisection-floor \
       record does not list"
      job
  in
  let no_record reason =
    Printf.sprintf "%s: no bisection-floor record this scan can read -- it found %s" floor_note
      reason
  in
  let floor_planted = run planted in
  let floor_listed_unmoved =
    run ~note:(Some (floor_note_text [ planted_entry "2026-02-01" ])) planted
  in
  let floor_moved =
    run ~note:(Some (floor_note_text ~floor:moved_floor [ planted_entry "2026-02-01" ])) planted
  in
  let floor_wrapped_entry =
    run
      ~note:
        (Some
           (floor_note_text ~floor:moved_floor
              [ "- `./.github/actions/planted`: 2026-02-01,\n  lukstafi/ocannl-staging#2.\n" ]))
      planted
  in
  let floor_out_of_order =
    run
      ~note:
        (Some
           (floor_note_text ~floor:"- Floor: 2025-06-01, lukstafi/ocannl-staging#2.\n"
              [ planted_entry "2025-06-01" ]))
      planted
  in
  let floor_decoys =
    run
      (dispatched_workflow
         [
           dispatched_job ~name:"scripted"
             [
               "    - run: |\n        echo 'uses: ./.github/actions/planted'\n";
               "    # - uses: ./.github/actions/planted\n";
               "    - run: echo hi # uses: ./.github/actions/planted\n";
             ];
         ])
  in
  let floor_reusable_caller =
    run
      (dispatched_workflow
         [
           dispatched_job ~name:"plain" []; "  caller:\n    uses: ./.github/workflows/filler1.yml\n";
         ])
  in
  let floor_undispatched =
    run
      (workflow
         [
           "  plain:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    steps:\n\
           \      - uses: ./.github/actions/planted\n";
         ])
  in
  let floor_job_without_checkout =
    run (dispatched_workflow [ dispatched_job ~checkout:false ~name:"headless" [ planted_step ] ])
  in
  let floor_quoted =
    run
      (dispatched_workflow
         [
           dispatched_job ~name:"quoted"
             [ "    - uses: \"./.github/actions/planted\" # a comment\n" ];
         ])
  in
  let floor_flow_mapping =
    run
      (dispatched_workflow
         [ dispatched_job ~name:"flow" [ "    - { uses: ./.github/actions/planted }\n" ] ])
  in
  let floor_stale_entry =
    run
      ~note:
        (Some
           (floor_note_text ~floor:moved_floor
              [ "- `./.github/actions/retired`: 2026-02-01, lukstafi/ocannl-staging#2.\n" ]))
      (workflow [ sound_job ~name:"plain" ])
  in
  let floor_duplicate_entry =
    run
      ~note:
        (Some
           (floor_note_text
              [ "- `./.github/actions/filler-action`: 2026-01-01, lukstafi/ocannl-staging#1.\n" ]))
      (workflow [ sound_job ~name:"plain" ])
  in
  let floor_malformed_entry =
    run
      ~note:
        (Some
           (floor_note_text ~floor:moved_floor
              [ "- `./.github/actions/planted` since 2026-02-01, lukstafi/ocannl-staging#2.\n" ]))
      planted
  in
  let floor_impossible_date =
    run ~note:(Some (floor_note_text ~floor:moved_floor [ planted_entry "2026-13-01" ])) planted
  in
  let floor_malformed_floor =
    run
      ~note:
        (Some
           (floor_note_text ~floor:"- Floor: 2026-02-01 (lukstafi/ocannl-staging#2).\n"
              [ planted_entry "2026-02-01" ]))
      planted
  in
  let floor_two_floors =
    run
      ~note:
        (Some
           (floor_note_text
              ~floor:
                "- Floor: 2026-01-01, lukstafi/ocannl-staging#1.\n\
                 - Floor: 2026-01-01, lukstafi/ocannl-staging#1.\n"
              []))
      (workflow [ sound_job ~name:"plain" ])
  in
  let floor_no_heading =
    run
      ~note:
        (Some
           "# Build and test mechanics\n\n\
            - `./.github/actions/filler-action`: 2026-01-01, lukstafi/ocannl-staging#1.\n\
            - Floor: 2026-01-01, lukstafi/ocannl-staging#1.\n")
      (workflow [ sound_job ~name:"plain" ])
  in
  let floor_note_not_handed = run ~note:None (workflow [ sound_job ~name:"plain" ]) in
  printf
    "\n\
     Synthetic controls run this same scanner over workflow trees differing from a legitimate one\n\
     in one respect each; refusal output is captured and matched below.\n\n";
  p
    "a tree whose jobs carry an expression ceiling, a plain one, and a reusable-workflow call \
     passes"
    (passed "legitimate" legitimate);
  p "a job with no ceiling at all is refused, named, with GitHub's 6-hour default spelled out"
    (refused "unbounded job" ~messages:[ missing_timeout "unbounded" ] no_timeout);
  p "a ceiling declared after the steps it would bound is refused as out of order"
    (refused "late ceiling"
       ~messages:[ "job `late` declares `timeout-minutes` after its `steps:`" ]
       after_steps);
  p "a ceiling that exists only inside a run block scalar leaves the job unbounded"
    (refused "block scalar decoy" ~messages:[ missing_timeout "decoy" ] block_scalar_decoy);
  p "a per-step ceiling is not the job's own, and the job is still refused"
    (refused "step-level decoy" ~messages:[ missing_timeout "stepwise" ] step_level_decoy);
  p "a reusable-workflow call that declares a ceiling GitHub would reject is refused too"
    (refused "caller with a ceiling"
       ~messages:
         [
           "job `caller` calls a reusable workflow and declares `timeout-minutes`, which GitHub \
            rejects on that form";
         ]
       caller_with_timeout);
  p "a timeout-minutes key with nothing behind it declares no ceiling and is refused"
    (refused "empty ceiling value" ~messages:[ missing_timeout "hollow" ] empty_value);
  p "a timeout-minutes key whose whole value is a comment is refused the same way"
    (refused "comment-only ceiling value"
       ~messages:[ missing_timeout "deferred" ]
       comment_only_value);
  p "a real ceiling wearing a trailing comment is a ceiling, and its job passes"
    (passed "commented ceiling" commented_ceiling);
  p "a reusable-workflow call carrying the key with no value is refused on the key, not the value"
    (refused "caller with an empty ceiling key"
       ~messages:
         [
           "job `caller` calls a reusable workflow and declares `timeout-minutes`, which GitHub \
            rejects on that form";
         ]
       caller_with_empty_timeout);
  p "a .yaml workflow is discovered and judged like a .yml one"
    (refused "yaml-spelled subject"
       ~messages:[ missing_timeout ~file:"subject.yaml" "unbounded" ]
       yaml_subject);
  p
    "a job whose steps sequence starts at its own key indentation is still judged, and refused \
     unbounded"
    (refused "flush sequence, no ceiling"
       ~messages:[ missing_timeout "flush_sequence" ]
       flush_sequence_unbounded);
  p "an inline jobs value is refused as a shape this reader cannot decide"
    (refused "inline jobs value"
       ~messages:[ unreadable "an inline `jobs:` value rather than a block of job entries" ]
       inline_jobs);
  p "a jobs key with no entries under it is refused rather than read as a file of no jobs"
    (refused "empty jobs block"
       ~messages:[ unreadable "a `jobs:` key with no job entries under it" ]
       empty_jobs_block);
  p "an inline job body is refused, its keys being unreadable to this reader"
    (refused "inline job body"
       ~messages:[ unreadable "an inline body for job `build`" ]
       inline_job_body);
  p "a job with no keys under it is refused rather than read as a job without a ceiling"
    (refused "job without keys"
       ~messages:[ unreadable "no keys at all under job `build`" ]
       job_without_keys);
  p "an unreadable line inside an otherwise valid job is refused, ceiling and all"
    (refused "unreadable key inside a job"
       ~messages:[ "which is not a key this reader can parse" ]
       unreadable_key_in_job);
  p "the same job with that key spelled plainly is read and passes"
    (passed "readable key inside a job" readable_key_in_job);
  p "a quoted job key is a job like any other, and its bounded job passes"
    (passed "quoted job key" quoted_job_key);
  p "the same quoted job without a ceiling is refused, so quoting hides nothing"
    (refused "quoted job key, no ceiling"
       ~messages:[ missing_timeout "quoted job" ]
       quoted_job_key_unbounded);
  p "a sequence entry where a job mapping belongs is refused, not skipped"
    (refused "unreadable job key"
       ~messages:[ "which is not a key this reader can parse" ]
       unreadable_job_key);
  p "a quoted key opens a block scalar too, its body staying content"
    (passed "quoted block-scalar key" quoted_block_scalar_key);
  p "a document with a YAML directive and an indented root mapping is read"
    (passed "directive header" directive_header);
  p "a decoy jobs mapping inside a multiline quoted scalar is refused, never read as the real one"
    (refused "multiline quoted scalar"
       ~messages:[ "opens a quoted scalar this reader will not follow across lines" ]
       multiline_quoted_scalar);
  p "the same field written on one line, colon and quotes and all, is read normally"
    (passed "single-line quoted scalar" single_line_quoted_scalar);
  p "an apostrophe inside a plain scalar is text, not a quote this reader must follow"
    (passed "apostrophe in a plain scalar" apostrophe_in_plain_scalar);
  p "a quote partway through a plain scalar is text too, whitespace before it notwithstanding"
    (passed "quote mid plain scalar" quote_mid_plain_scalar);
  p "a hyphen partway through a plain scalar indicates nothing, quotes after it included"
    (passed "hyphen in a plain scalar" hyphen_in_plain_scalar);
  p "keys carrying YAML's separation space before the colon are read as the keys they are"
    (passed "spaced keys" spaced_keys);
  p "a leading byte-order mark does not hide the first key"
    (passed "byte-order mark" byte_order_mark);
  p "a document-end marker leaves an indented root mapping readable"
    (passed "document-end marker" document_end_marker);
  p "a tab inside a run block scalar is content, and its workflow passes"
    (passed "tab inside a block scalar" tab_inside_block_scalar);
  p "a block header carrying YAML's indentation and chomping indicators still opens a block"
    (passed "indicator block scalar" indicator_block_scalar);
  p
    "a root mapping the document indents uniformly is read, `jobs:` not having to sit at column \
     zero"
    (passed "indented root mapping" indented_root_mapping);
  p "a tab-indented workflow is refused, this reader measuring indentation in spaces"
    (refused "tab indentation"
       ~messages:[ unreadable "a line indented with tabs, which this reader measures in spaces" ]
       tab_indented);
  p "a workflow file with no jobs mapping is refused rather than read as having no jobs"
    (refused "no jobs mapping" ~messages:[ unreadable "no top-level `jobs:` key" ] no_jobs);
  p "a YAML file nested under .github/workflows is no workflow, and its unbounded job is not judged"
    (passed "nested fixture" nested_fixture);
  p
    "a dispatched job that starts using a local action the bisection-floor record does not list is \
     refused, named"
    (refused "planted local action" ~messages:[ unlisted "bisectable" ] floor_planted);
  p "listing the new action without moving the floor to it is refused"
    (refused "floor left behind"
       ~messages:
         [
           "the bisection floor (2026-01-01, lukstafi/ocannl-staging#1) is not the newest entry's \
            (`./.github/actions/planted`: 2026-02-01, lukstafi/ocannl-staging#2)";
         ]
       floor_listed_unmoved);
  p "listing the new action last and moving the floor to it passes"
    (passed "floor moved" floor_moved);
  p "an entry wrapped across two lines is the same entry, and its moved floor passes"
    (passed "wrapped entry" floor_wrapped_entry);
  p
    "an entry dated before the one above it is refused, the floor having moved to it \
     notwithstanding"
    (refused "entries out of order"
       ~messages:
         [
           "the bisection-floor entry for `./.github/actions/planted` (2025-06-01, \
            lukstafi/ocannl-staging#2) is older than the one above it";
         ]
       floor_out_of_order);
  p "a local action named only in a run block scalar or a comment is no step's, and passes unlisted"
    (passed "block scalar and comment decoys" floor_decoys);
  p "a job-level reusable-workflow call is loaded from the dispatched ref, and passes unlisted"
    (passed "reusable caller" floor_reusable_caller);
  p "a local action in a workflow that never names expected_sha bounds no dispatch, and passes"
    (passed "undispatched workflow" floor_undispatched);
  p "a dispatched workflow's job counts whether or not its own checkout names the input"
    (refused "job without a checkout" ~messages:[ unlisted "headless" ] floor_job_without_checkout);
  p "a quoted local action with a trailing comment is the same action, refused unlisted"
    (refused "quoted local action" ~messages:[ unlisted "quoted" ] floor_quoted);
  p "a flow-mapping step spelling uses: in a dispatched workflow is refused as unread"
    (refused "flow-mapping step"
       ~messages:
         [
           unreadable
             "line 15, which spells `uses:` in a workflow naming expected_sha but is no `uses:` \
              key this reader can read";
         ]
       floor_flow_mapping);
  p "a recorded action no dispatched step uses is refused as stale"
    (refused "stale entry"
       ~messages:
         [
           "lists `./.github/actions/retired`, which no step of an expected_sha-dispatched \
            workflow uses";
         ]
       floor_stale_entry);
  p "an action recorded twice is refused, the lists being compared with their repetitions"
    (refused "duplicate entry"
       ~messages:[ "lists `./.github/actions/filler-action` 2 times" ]
       floor_duplicate_entry);
  p "an action entry in another form is refused, never read as some entry"
    (refused "malformed entry"
       ~messages:[ no_record "line 7, an action entry not of the form" ]
       floor_malformed_entry);
  p "an entry dated in a thirteenth month is refused the same way"
    (refused "impossible date"
       ~messages:[ no_record "line 7, an action entry not of the form" ]
       floor_impossible_date);
  p "a floor entry in another form is refused"
    (refused "malformed floor"
       ~messages:[ no_record "line 8, a floor entry not of the form" ]
       floor_malformed_floor);
  p "two floor entries are refused rather than one of them read"
    (refused "two floors" ~messages:[ no_record "more than one floor entry" ] floor_two_floors);
  p "a note without the record's heading is refused, its entry-shaped bullets unread"
    (refused "no heading"
       ~messages:[ no_record "no `### How far back a dispatch reaches` heading" ]
       floor_no_heading);
  p "a run not handed the note is refused rather than read as an empty record"
    (refused "note not handed"
       ~messages:[ no_record "nothing: the note was not handed to this scan" ]
       floor_note_not_handed);
  Test_utils.Refusal_control_manifest.print "workflow_job_timeouts.ml";
  remove_tree fixture

(* Every file this scan reads arrives as a `@<path>` response file, because the list a repository-
   wide glob produces may be longer than a Windows command line; see [Test_utils.Scan_argv]. The
   synthetic controls pass their handful of paths directly, which reaches the same argv. *)
let () =
  match Array.to_list (Test_utils.Scan_argv.expand Stdlib.Sys.argv) with
  | _ :: "--scan-only" :: workspace_root :: paths -> scan workspace_root paths
  | _ :: workspace_root :: paths ->
      scan workspace_root paths;
      control ()
  | _ ->
      eprintf "Usage: %s [--scan-only] <workspace root> <workflow files...>\n" Stdlib.Sys.argv.(0);
      Stdlib.exit 2
