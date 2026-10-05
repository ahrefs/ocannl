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
let block_scalar_opener line =
  let rec strip_markers text =
    let stripped = String.lstrip text in
    match String.chop_prefix stripped ~prefix:"-" with
    | Some rest when String.is_prefix rest ~prefix:" " || String.is_empty rest -> strip_markers rest
    | _ -> stripped
  in
  match key_of { line with text = strip_markers line.text } with
  | Some (_, value) -> is_block_scalar value
  | None -> false

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
          collect ({ file; name; timeout; steps_line; calls_reusable } :: found) rest
      | line :: rest ->
          if line.indent < job_indent then refuse (Unreadable_key line.number)
          else collect found rest
    in
    match collect [] block with [] -> refuse Empty_jobs_block | jobs -> jobs
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
  let run ?(extra = []) ?(subject_name = "subject.yml") subject =
    Int.incr case_index;
    let root = Stdlib.Filename.concat fixture (Printf.sprintf "case%d" !case_index) in
    let workflows = Stdlib.Filename.concat root ".github/workflows" in
    let fillers = [ "filler1.yml"; "filler2.yml"; "filler3.yaml" ] in
    List.iter fillers ~f:(fun name ->
        write_file (Stdlib.Filename.concat workflows name) (workflow [ sound_job ~name:"build" ]));
    write_file (Stdlib.Filename.concat workflows subject_name) subject;
    List.iter extra ~f:(fun (path, content) ->
        write_file (Stdlib.Filename.concat root path) content);
    let files =
      (subject_name :: fillers |> List.map ~f:(Stdlib.Filename.concat ".github/workflows"))
      @ List.map extra ~f:fst
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
