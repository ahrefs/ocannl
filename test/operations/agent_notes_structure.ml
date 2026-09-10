(** gh-ocannl-691: the agent notes are structurally intact and wholly reachable from their index.

    [docs/agent-notes.md] and the files under [docs/agent-notes/] are the project's cross-session
    memory, and nothing read them until this. They are prose, so they were assumed to be beyond
    mechanical checking — but three of the six review findings on the split that created them
    (lukstafi/ocannl-staging#406) were structural and every one was decidable from the text alone: a
    bullet a merge resolution cut in half, an index hook naming a file that carries none of it, and
    a row wrapped across two physical lines, which ends a Markdown table and drops the five rows
    below it out of the index. Each was caught by a human reading carefully. A note that corrupts
    silently is worse than one that is missing, because every later session inherits it as fact.

    The six rules, and what each is for, are stated in {!Test_utils.Agent_notes_scan}, which is
    where they are decided — pure functions over strings, so the negative controls in
    [agent_notes_scan_cases.ml] exercise the same code this runs over the repository. This file is
    the live-tree half: it opens what dune hands it and reports one verdict per rule.

    {1 What the golden holds}

    The six claims, and nothing that counts. A tally ("12 notes files", "177 bullets") moves on
    every correct addition, so every contributor promotes a file they did not touch and the promote
    is indistinguishable from blessing a regression (gh-ocannl-665, and the notes' own entry on it).
    The exact numbers go to stderr. What the counts were there for — the assurance that a scan
    reporting nothing scanned something — is kept as floors asserted through {!Verdict.claim}, which
    prints nothing while they hold and fails the run when they do not: a glob that matched no files,
    or notes that lost most of their bullets, cannot pass here vacuously. *)

open Base
open Stdio
module Notes = Test_utils.Agent_notes_scan

(** Bullets that legitimately end without sentence-terminating punctuation, keyed by
    [Agent_notes_scan.subject_key] form — ["<file>: <the bullet's opening, whitespace-normalized>"],
    which the finding's own message prints ready to paste — and each carrying its reason. The cheap
    fix for a flagged bullet is punctuation; this list is for one whose ending is load-bearing.
    Every entry has to earn its place on every run — an exemption that no longer matches a flagged
    bullet is reported, so a stale one cannot sit here granting cover to whatever text replaced it.
*)
let unterminated_bullets : (string * string) list = []

let read path = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all

(** The path a notes file is known by, relative to the index's own directory: dune hands over
    [../../docs/agent-notes/build-and-test.md] and an index row spells
    [agent-notes/build-and-test.md]. *)
let docs_relative path =
  let path = String.substr_replace_all path ~pattern:"\\" ~with_:"/" in
  match String.substr_index path ~pattern:"docs/" with
  | Some i -> String.drop_prefix path (i + String.length "docs/")
  | None -> path

let () =
  let args = Array.to_list Stdlib.Sys.argv |> List.tl_exn in
  let md = List.filter args ~f:(String.is_suffix ~suffix:".md") in
  let index_path, file_paths =
    List.partition_tf md ~f:(fun p -> String.equal (docs_relative p) "agent-notes.md")
  in
  let index_path =
    match index_path with
    | [ p ] -> p
    | ps ->
        eprintf "FAILED: expected exactly one index among the arguments, got %d\n" (List.length ps);
        Stdlib.exit 1
  in
  let index_file = docs_relative index_path in
  let index_contents = read index_path in
  let files =
    List.map file_paths ~f:(fun p -> (docs_relative p, read p))
    |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let bullets, findings = Notes.check_all ~index_file ~index_contents ~files in
  eprintf "Scanned %d notes files plus the index, %d bullets, %d findings.\n" (List.length files)
    (List.length bullets) (List.length findings);
  (* The floors. Neither moves when a note is edited, a bullet added or a file split; both fail the
     moment the scan is handed nothing, which is the one way its silence would be a lie. *)
  Verdict.claim "the scan was handed the notes files" (List.length files >= 10);
  Verdict.claim "the scan read the notes' bullets" (List.length bullets >= 100);
  print_endline
    "Structure of docs/agent-notes.md and docs/agent-notes/, over the live tree. The rules are\n\
     stated in test/support/agent_notes_scan.ml; the counts scanned go to stderr, since a tally in\n\
     a golden moves on every correct addition anywhere (gh-ocannl-665).\n";
  (* An exemption suppresses the finding it names, and only that one. The comparison is against the
     finding's own structured identity -- file plus the bullet's OPENING, which is the key its
     message tells you to paste -- never against the message text, which is prose and gets reworded.
     Keyed on the message it could not match at all (Codex P2, round 1). *)
  let exempted (f : Notes.finding) =
    String.equal f.Notes.rule Notes.rule_bullet_integrity
    &&
    match Notes.exemption_key f with
    | None -> false
    | Some key -> List.exists unterminated_bullets ~f:(fun (k, _) -> String.equal k key)
  in
  let of_rule r =
    List.filter findings ~f:(fun f -> String.equal f.Notes.rule r && not (exempted f))
  in
  (* Findings go to stderr only. A run that fails exits nonzero from Verdict's teardown, and dune
     never writes the redirected stdout of such a process -- stderr is the channel on which the
     message survives to be read. Keeping them off stdout also keeps them out of the golden, so a
     real defect cannot be `dune promote`d into the expected output. *)
  let reported = ref [] in
  let report rule claim =
    reported := rule :: !reported;
    let found = of_rule rule in
    List.iter found ~f:(fun f ->
        eprintf "  %s: %s: %s\n" f.Notes.rule f.Notes.where f.Notes.message);
    Verdict.p_empty claim ~over:files found
  in
  report Notes.rule_bullet_integrity "every bullet is whole and every list parses as one reading";
  report Notes.rule_index_agreement "every index row names a file that carries what it claims";
  report Notes.rule_table_shape "every table is a table, row by row";
  report Notes.rule_reachability "every notes file is reachable from the index, and links back";
  report Notes.rule_no_repetition "no bullet is repeated across the notes";
  report Notes.rule_qualified_citations "no numeric GitHub citation uses an unqualified bare hash";
  (* The relationship the six calls above rest on, and nothing used to state (gh-ocannl-706): a rule
     this file does not report is a rule whose findings the live tree never shows, and the omission
     is silent -- the scan computes them, [of_rule] is never asked for them, and the golden is six
     green lines either way. Sorted lists rather than sets, so a rule reported twice (two verdicts
     over one set of findings, one of them dead) is a mismatch as well; a bare boolean, so the
     golden stays fixed as rules come and go and only the stderr line moves. *)
  let sorted l = List.sort l ~compare:String.compare in
  let reported = sorted !reported and named = sorted Notes.rules in
  let all_reported = List.equal String.equal reported named in
  if not all_reported then
    eprintf "  this file reports [%s]; the scan names [%s]\n"
      (String.concat ~sep:"; " reported)
      (String.concat ~sep:"; " named);
  Verdict.p "every rule the scan names is reported over the live tree" all_reported;
  (* The other direction: a finding tagged with a rule [Notes.rules] does not name. Such a finding
     is kept, at the end of the report, precisely so that this can fail. *)
  let unnamed = Notes.unnamed_rule_findings findings in
  List.iter unnamed ~f:(fun f ->
      eprintf "  %s: %s: a finding tagged with a rule the scan does not name\n" f.Notes.rule
        f.Notes.where);
  Verdict.p_empty "every finding carries one of the rules the scan names" ~over:files unnamed;
  (* An exemption is a claim about a specific bullet; one that matches nothing has stopped being
     one, and left behind a hole the next edit falls into. *)
  (* An exemption key is the whole normalized bullet, so two bullets cannot share one. That is a
     property of the key rather than of this list, and a property worth failing loudly if it ever
     stops holding -- an exemption silencing a second, accidental truncation is exactly the harm the
     hatch must not do. *)
  let bullet_findings =
    List.filter_map findings ~f:(fun f ->
        if String.equal f.Notes.rule Notes.rule_bullet_integrity then Notes.exemption_key f
        else None)
  in
  let colliding =
    List.find_a_dup (List.sort bullet_findings ~compare:String.compare) ~compare:String.compare
  in
  (match colliding with
  | Some key -> eprintf "  two flagged bullets share the exemption key %s\n" key
  | None -> ());
  Verdict.p "each flagged bullet has an exemption key of its own" (Option.is_none colliding);
  let stale =
    List.filter unterminated_bullets ~f:(fun (key, _) ->
        not
          (List.exists findings ~f:(fun f ->
               String.equal f.Notes.rule Notes.rule_bullet_integrity
               && Option.value_map (Notes.exemption_key f) ~default:false ~f:(String.equal key))))
  in
  List.iter stale ~f:(fun (key, reason) ->
      eprintf "  stale exemption for %s (%s): no bullet is flagged for it any more\n" key reason);
  Verdict.p "every exemption still names a bullet that needs one" (List.is_empty stale);
  Test_utils.Refusal_control_manifest.print "agent_notes_structure.ml"
