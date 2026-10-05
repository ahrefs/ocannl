(** No census total in a repository-scan golden (gh-ocannl-1056).

    A count printed on stdout by a scan -- "170 tests in this directory", "41 checks did not hold"
    -- moves on every correct addition anywhere, so every unrelated contributor promotes a file they
    did not touch; two branches that each add an item conflict on the line, or, worse, merge the
    items right and the total wrong. It was hand-fixed three times (gh-ocannl-665, gh-ocannl-701,
    gh-ocannl-1046) and each sweep for other instances was a round of ad-hoc greps. This scan makes
    the fourth instance a refusal on the PR that adds it: every number in the goldens of the
    [@scans] family must be named by an entry of [allowed] -- a floor constant, a shape fact, an
    algorithm output, text quoted from a source -- each with its reason. The totals themselves
    belong on stderr, which a golden does not diff (docs/agent-notes/build-and-test.md).

    {1 Boundary}

    This is a LINE-SHAPED scan over text, and these four rules are all it reads.

    - WHICH GOLDENS. Every checked-in [dune] file is parsed ([Dune_stanza_scan.stanzas]), its
      [(subdir …)] forms applied. In every directory that defines an [(alias (name scans) …)], the
      aliases that alias reaches -- its [deps], then the [deps] of each stanza attached to one of
      those, to a fixed point ([Dune_stanza_scan.aliases_reached_from]) -- are the family. A golden
      is the first operand of each [(diff …)] or [(diff? …)] in the action of a rule attached to a
      family alias, resolved in the directory any enclosing [(chdir …)] moves it to, and
      [<name>.expected], where it is checked in, for a [(test)] or [(tests)] stanza whose per-test
      alias [runtest-<name>] is in the family. Not read: stderr, [.actual] outputs, inline [%expect]
      blocks, goldens of tests outside the family, an alias dependency naming another directory
      ([(alias dir/name)], [(alias_rec …)]), and dune fragments pulled in by [(include …)] (no dune
      file uses either today). A diffed golden named through a pform, or under a [chdir] to one, is
      not skipped: it is no checked-in file, so the first claim refuses it.
    - WHAT IS A NUMBER. A maximal run of two or more ASCII digits that is not immediately preceded
      by an ASCII letter or an underscore: a run continuing a name ([bf16], [m16n8k16],
      [gh514_cells]) is part of the name. A run FOLLOWED by a letter is still a number, so a total
      cannot hide behind a unit. A single digit is not a number here, nor is a number spelled in
      words or one broken across lines.
    - WHAT ALLOWS ONE. An entry of [allowed] applies to one golden, or to every golden, and holds a
      [Str] pattern; a number is allowed when some match of the pattern, starting anywhere in the
      number's line at or before it, ends at or after it -- so a match must span the WHOLE run, and
      a total sharing a line with an allowed citation is still refused. An entry that allows nothing
      is stale and fails, which is also this scan's signal that it did not go blind: a scan that
      read no goldens leaves every entry stale.
    - THE TEARDOWN TOTAL. A whole line [FAILED: <n> check(s) did not hold.] -- the line
      [Verdict.report_failures] writes -- is refused whatever its digit count, so the one-digit
      count the number rule does not read cannot carry it in: a negative control whose failures are
      its golden ends through [Verdict.exit_negative_control] instead. *)

open Base
open Stdio
open Verdict.Claims
module Dune = Test_utils.Dune_stanza_scan
module Inventory = Test_utils.Source_inventory

let printf = Test_utils.Refusal_control_manifest.printf

type golden = Every_golden | Golden of string
type entry = { golden : golden; pattern : string; reason : string }

let citation =
  {
    golden = Every_golden;
    pattern = {|\(gh-\(ocannl-\)?\|PR #\|staging#\)[0-9]+|};
    reason = "an issue or pull-request citation names a record, not a quantity";
  }

(** The judgment list. Each entry is as narrow as the form it allows, and says why the number is not
    a total: a floor or cap is a constant of its scan and moves only with it, a quoted source text
    moves only with that source, a fixture location moves only with the fixture beside it. *)
let allowed =
  [
    citation;
    {
      golden = Every_golden;
      pattern = {|\[scanner-refusal:[0-9a-f]+\]|};
      reason =
        "a refusal marker's digest names one diagnostic format (Refusal_control_manifest) and \
         moves only with that format's text";
    };
    {
      golden = Golden "test/operations/agents_md_size.expected";
      pattern = {|32 KiB|};
      reason = "the cap Claude Code applies to an imported instructions file, the scan's constant";
    };
    {
      golden = Golden "test/operations/codegen_text_inventory.expected";
      pattern = {|^    ".*"$|};
      reason = "a string literal quoted from the source that pins it; its digits are that text";
    };
    {
      golden = Golden "test/operations/config_usage_scan_control.expected";
      pattern = {|^FAIL: [a-z_]+\.fixture:[0-9]+:|};
      reason = "the line of the synthetic fixture a refusal names, moving only with that fixture";
    };
    {
      golden = Golden "test/operations/env_var_deps.expected";
      pattern = {|the floor of [0-9]+ callers|};
      reason = "the artifact-caller floor, a hand-written constant of the scan";
    };
    {
      golden = Golden "test/operations/env_var_deps.expected";
      pattern = {|diagnostics: at least [0-9]+ (count and details on stderr)|};
      reason = "the extracted-diagnostic floor, a hand-written constant of the scan";
    };
    {
      golden = Golden "test/operations/ll_test_ratchet.expected";
      pattern = {|^Source floor: [^ ]+ >= [0-9]+$\| -- records<=[0-9]+ traversals<=[0-9]+: |};
      reason =
        "a per-root source floor or a per-file exemption cap, hand-written constants of the scan";
    };
    {
      golden = Golden "test/operations/ll_test_ratchet.expected";
      pattern = {|^\(test\|arrayjit/test\)/[^ ]+\.ml -- records=[0-9]+ traversals=[0-9]+$|};
      reason =
        "per-source residual IR counts deliberately remain review-visible after adoption \
         (gh-ocannl-1090), not an aggregate over the source corpus";
    };
    {
      golden = Golden "test/operations/provenance_tag_inventory.expected";
      pattern = {|^  [0-9]+:[a-z0-9-]+\( ([A-Z][A-Za-z0-9_']*)\)?$|};
      reason = "a provenance tag as a source mints it; its digits name the tag";
    };
    {
      golden = Golden "test/operations/provenance_tag_inventory.expected";
      pattern = {|^  [0-9]+:[a-z0-9-]+\( [0-9]+:[a-z0-9-]+\)+ -- |};
      reason = "the tags of a pinned number collision, each an identifier rather than a quantity";
    };
    {
      golden = Golden "test/operations/provenance_tag_inventory.expected";
      pattern = {|^[^ ]+ --\( [0-9]+\)+$|};
      reason = "the tag numbers a listed file cites, each an identifier rather than a quantity";
    };
    {
      golden = Golden "test/operations/operand_key_ratchet.expected";
      pattern = {|^Source floor: [^ ]+ >= [0-9]+$|};
      reason = "a per-root source floor, a hand-written constant of the scan";
    };
    {
      golden = Golden "test/operations/operand_key_ratchet.expected";
      pattern = {|^[^ ]+ -- `[^`]*`: |};
      reason = "an exempted site's source text, quoted between its path and its reason";
    };
    {
      golden = Golden "test/operations/operand_key_scan_cases.expected";
      pattern = {|the blind [0-9]+x[0-9]+ fixture|};
      reason = "the shape of the synthetic fixture that opened gh-ocannl-1018";
    };
    {
      golden = Golden "test/operations/shell_scripts_parse.expected";
      pattern = {|at least the [0-9]+ shell scripts|};
      reason = "the script floor, a hand-written constant of the scan";
    };
    {
      golden = Golden "test/operations/verdict_ratchet.expected";
      pattern = {|^  [a-z_/]+\.ml:\([^ ]\|  *[^ -]\|  *-[^- ]\|  *--[^ ]\)*|};
      reason =
        "an exempted claim format quoted from the file it sits in, up to the ` -- ` before its \
         reason";
    };
  ]

(** A goldens population this small means the family went unread; there are over thirty. *)
let golden_floor = 20

(** A number population this small means the goldens went unread; the refusal markers alone run to
    several hundred. *)
let number_floor = 100

(** {1 The family's goldens} *)

(* A [(chdir <dir> …)] moves where the operands of the diffs inside it resolve; a [<dir>] holding a
   pform resolves to no checked-in file, which the first claim refuses. *)
let rec diffed_goldens ?(cwd = "") = function
  | Sexp.List [ Sexp.Atom ("diff" | "diff?"); Sexp.Atom golden; _ ] -> [ Dune.in_subdir cwd golden ]
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: body) ->
      List.concat_map body ~f:(diffed_goldens ~cwd:(Dune.in_subdir cwd dir))
  | Sexp.List l -> List.concat_map l ~f:(diffed_goldens ~cwd)
  | Sexp.Atom _ -> []

(** The goldens of every [scans] family among [dune_files] (path, content), repository-relative,
    with [checked_in] deciding whether a test's optional [<name>.expected] exists. A diffed golden
    is kept whether or not it exists: a missing one is the caller's finding. *)
let family_goldens ~dune_files ~checked_in =
  List.concat_map dune_files ~f:(fun (path, content) ->
      Dune.walk (Stdlib.Filename.dirname path) (Dune.stanzas content) ~f:(fun dir stanza ->
          [ (Dune.normalize_path dir, stanza) ]))
  |> Map.of_alist_multi (module String)
  |> Map.to_alist
  |> List.concat_map ~f:(fun (dir, stanzas) ->
      if
        not
          (List.exists stanzas ~f:(fun stanza ->
               Option.equal String.equal (Dune.alias_stanza_name stanza) (Some "scans")))
      then []
      else
        let family = Dune.aliases_reached_from stanzas "scans" in
        let in_dir name = Dune.normalize_path (dir ^ "/" ^ name) in
        List.concat_map stanzas ~f:(fun stanza ->
            match stanza with
            | Sexp.List (Sexp.Atom ("test" | "tests") :: _) ->
                Dune.names_of stanza
                |> List.filter ~f:(fun name -> Set.mem family ("runtest-" ^ name))
                |> List.map ~f:(fun name -> in_dir (name ^ ".expected"))
                |> List.filter ~f:checked_in
            | Sexp.List (Sexp.Atom "rule" :: _)
              when List.exists (Dune.aliases_of stanza) ~f:(Set.mem family) ->
                Option.value_map (Dune.field stanza "action") ~default:[] ~f:(fun action ->
                    List.concat_map action ~f:(diffed_goldens ~cwd:"") |> List.map ~f:in_dir)
            | _ -> []))
  |> List.dedup_and_sort ~compare:String.compare

(** {1 Numbers and the entries that allow them} *)

type number = { in_golden : string; line_no : int; line : string; start : int; stop : int }

let continues_name c = Char.is_alpha c || Char.equal c '_'

let numbers_of ~golden content =
  String.split_lines content
  |> List.concat_mapi ~f:(fun index line ->
      let length = String.length line in
      let rec runs position acc =
        if position >= length then List.rev acc
        else if not (Char.is_digit line.[position]) then runs (position + 1) acc
        else
          let stop =
            Option.value ~default:length
              (List.find (List.range position length) ~f:(fun i -> not (Char.is_digit line.[i])))
          in
          let named = position > 0 && continues_name line.[position - 1] in
          let acc =
            if stop - position >= 2 && not named then
              { in_golden = golden; line_no = index + 1; line; start = position; stop } :: acc
            else acc
          in
          runs stop acc
      in
      runs 0 [])

let applies entry golden =
  match entry.golden with Every_golden -> true | Golden path -> String.equal path golden

let allows entry =
  let re = Str.regexp entry.pattern in
  fun number ->
    applies entry number.in_golden
    && List.exists (List.range 0 number.start ~stop:`inclusive) ~f:(fun position ->
        Str.string_match re number.line position && Str.match_end () >= number.stop)

let refused ~entries numbers =
  let checks = List.map entries ~f:allows in
  List.filter numbers ~f:(fun number -> not (List.exists checks ~f:(fun allows -> allows number)))

let stale ~entries numbers =
  List.filter entries ~f:(fun entry -> not (List.exists numbers ~f:(allows entry)))

(** Verdict's teardown line ([Verdict.report_failures]) totals the failure lines above it, so it is
    a census total whatever its digit count -- and a one-digit count is below the number boundary
    above. It is matched as a whole line instead. *)
let teardown_total = Str.regexp {|^FAILED: [0-9]+ checks? did not hold\.$|}

let teardown_totals ~golden content =
  String.split_lines content
  |> List.filter_mapi ~f:(fun index line ->
      if Str.string_match teardown_total line 0 then Some (golden, index + 1, line) else None)

let render number =
  Printf.sprintf "%s:%d: %s in: %s" number.in_golden number.line_no
    (String.sub number.line ~pos:number.start ~len:(number.stop - number.start))
    number.line

(** {1 The live scan} *)

let scan root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let dune_files =
    Inventory.select inventory ~f:(fun path -> String.equal (Stdlib.Filename.basename path) "dune")
    |> List.map ~f:(fun (file : Inventory.file) -> (file.path, In_channel.read_all file.on_disk))
  in
  let goldens = family_goldens ~dune_files ~checked_in:(Inventory.mem inventory) in
  eprintf "census_total_ratchet: %d goldens of the scans family (not part of the golden):\n"
    (List.length goldens);
  List.iter goldens ~f:(eprintf "  %s\n");
  printf
    "Numbers in the goldens of the @scans family, each allowed by a named entry with its reason\n\
     (gh-ocannl-1056). The goldens read and every refused number go to stderr.\n\n";
  p_all ~min:golden_floor "every golden the scans family diffs is a checked-in file this scan read"
    goldens ~f:(Inventory.mem inventory);
  let contents =
    List.concat_map goldens ~f:(fun golden ->
        List.find inventory ~f:(fun (file : Inventory.file) -> String.equal file.path golden)
        |> Option.value_map ~default:[] ~f:(fun (file : Inventory.file) ->
            [ (golden, In_channel.read_all file.on_disk) ]))
  in
  let numbers = List.concat_map contents ~f:(fun (golden, content) -> numbers_of ~golden content) in
  eprintf "census_total_ratchet: %d numbers read (not part of the golden)\n" (List.length numbers);
  let refusals = refused ~entries:allowed numbers in
  List.iter refusals ~f:(fun number ->
      eprintf "census_total_ratchet: no entry allows %s\n" (render number));
  p_empty ~min:number_floor
    "no golden of the scans family carries a number the allow-list does not name" ~over:numbers
    refusals;
  let teardowns =
    List.concat_map contents ~f:(fun (golden, content) -> teardown_totals ~golden content)
  in
  List.iter teardowns ~f:(fun (golden, line_no, line) ->
      eprintf
        "census_total_ratchet: %s:%d: Verdict's teardown total %S; end the negative control \
         through Verdict.exit_negative_control, which exits 1 without it\n"
        golden line_no line);
  p_empty "no golden of the scans family carries Verdict's FAILED teardown line, whatever its count"
    ~over:contents teardowns;
  let stale_entries = stale ~entries:allowed numbers in
  List.iter stale_entries ~f:(fun entry ->
      eprintf "census_total_ratchet: entry %S allows no number in its golden; delete it\n"
        entry.pattern);
  p_empty "every allow-list entry still allows a number in its golden" ~over:allowed stale_entries;
  p_all "every allow-list entry gives its reason in words" allowed ~f:(fun entry ->
      List.length (String.split entry.reason ~on:' ') > 2)

(** {1 Synthetic controls}

    The live goldens are clean on a good day, so a clean result cannot tell a rule from a blind
    reader: each decision is put to text the repository does not contain. *)

let controls () =
  printf "\nSynthetic controls:\n";
  let line text = numbers_of ~golden:"x/a.expected" text in
  let refused_in ?(entries = []) text = refused ~entries (line text) in
  let starts numbers = List.map numbers ~f:(fun number -> number.start) in
  let at positions numbers = List.equal Int.equal (starts numbers) positions in
  p "a census total on a synthetic golden line is refused"
    (List.length (refused_in "Scanned 170 tests in this directory.") = 1);
  p_empty "a number the whole of one match spans is allowed" ~over:(line "(gh-ocannl-665)")
    (refused_in ~entries:[ citation ] "(gh-ocannl-665)");
  p "a match spanning only part of a number does not allow it"
    (List.length
       (refused_in ~entries:[ { citation with pattern = "17"; golden = Every_golden } ] "170 tests")
    = 1);
  p "a total sharing its line with an allowed citation is still refused"
    (at [ 15 ] (refused_in ~entries:[ citation ] "gh-ocannl-665: 170 tests"));
  p "an entry for another golden does not reach this one"
    (List.length
       (refused_in
          ~entries:[ { citation with golden = Golden "x/b.expected"; pattern = "[0-9]+" } ]
          "170 tests")
    = 1);
  p "a total followed by its unit is still a number" (List.length (line "29876B read") = 1);
  p "a digit run continuing a name is not a number"
    (at [ 41 ] (line "bf16_arithmetic m16n8k16 gh514_cells.sh: 170 lines"));
  p "a single digit is outside the boundary" (at [ 11 ] (line "7 tests of 170"));
  (* Only the two-digit count is a number the boundary reads; the teardown rule refuses both whole
     lines, and not the line that merely begins like one. The lines are Verdict's own, so a reworded
     teardown fails this control rather than leaving the matcher blind. *)
  let teardown_golden =
    String.concat_lines
      [
        "FAIL: a";
        Verdict.teardown_line 3;
        Verdict.teardown_line 12;
        Verdict.teardown_line 1 ^ " (not a teardown)";
      ]
  in
  p "a one-digit teardown total is refused as a line, though no number the boundary reads"
    (List.equal Int.equal
       (List.map (teardown_totals ~golden:"x/a.expected" teardown_golden) ~f:(fun (_, n, _) -> n))
       [ 2; 3 ]
    && List.equal Int.equal
         (List.map (line teardown_golden) ~f:(fun number -> number.line_no))
         [ 3 ]);
  p "an entry that allows nothing is stale"
    (List.length (stale ~entries:[ citation ] (line "Scanned 170 tests.")) = 1);
  let dune =
    {|(alias (name scans) (deps (alias runtest-a) (alias runtest-b) (alias runtest-c)))
(tests (names a g))
(rule (alias runtest-b) (action (diff b.expected b.actual)))
(rule (alias runtest-c) (deps (alias runtest-d)) (action (run c.exe)))
(rule (alias runtest-d) (action (progn (diff? d.expected d.actual))))
(rule (alias runtest-e) (action (diff e.expected e.actual)))
(subdir sub
 (alias (name scans) (deps (alias runtest-f)))
 (rule (alias runtest-f) (action (diff f.expected f.actual))))
|}
  in
  p "the family is a test's golden, a rule's diff, one behind a member's own deps, and a subdir's"
    (List.equal String.equal
       (family_goldens ~dune_files:[ ("x/dune", dune) ] ~checked_in:(fun _ -> true))
       [ "x/a.expected"; "x/b.expected"; "x/d.expected"; "x/sub/f.expected" ]);
  p "a diff under a chdir resolves in the directory the action runs in"
    (List.equal String.equal
       (family_goldens
          ~dune_files:
            [
              ( "x/dune",
                {|(alias (name scans) (deps (alias runtest-h)))
(rule (alias runtest-h) (action (chdir fixtures (progn (diff h.expected h.actual)))))|}
              );
            ]
          ~checked_in:(fun _ -> true))
       [ "x/fixtures/h.expected" ]);
  (* The live allow-list as a whole, so a wider entry for the same golden cannot hide behind the one
     this is about. *)
  let refused_by golden text = refused ~entries:allowed (numbers_of ~golden text) in
  let tag_golden = "test/operations/provenance_tag_inventory.expected" in
  let tag_line = "  1131:" ^ "fixture-tag" in
  p_all "provenance tags allow constructor suffixes with digits and apostrophes"
    [ ""; " (Visit_cap)"; " (Visit_cap2)"; " (Visit_cap')"; " (V2_')" ] ~f:(fun suffix ->
      let text = tag_line ^ suffix in
      let numbers = numbers_of ~golden:tag_golden text in
      (not (List.is_empty numbers)) && List.is_empty (refused_by tag_golden text));
  p_all "provenance constructor allowances refuse invalid suffixes and trailing totals"
    [
      " (visit_cap)";
      " (_Visit_cap)";
      " (2Visit_cap)";
      " (Visit-cap)";
      " (Visit cap)";
      " (Visit_cap2) total=170";
    ] ~f:(fun suffix -> not (List.is_empty (refused_by tag_golden (tag_line ^ suffix))));
  p "provenance constructor allowances stay in their own golden"
    (not (List.is_empty (refused_by "x/a.expected" (tag_line ^ " (Visit_cap2)"))));
  let adopted_row = "test/operations/new.ml -- records=19 traversals=12" in
  p_empty "per-source adopted IR counts remain review-visible in their own golden"
    ~over:(numbers_of ~golden:"test/operations/ll_test_ratchet.expected" adopted_row)
    (refused_by "test/operations/ll_test_ratchet.expected" adopted_row);
  p "the adopted-count allowance cannot hide aggregate or trailing totals"
    (List.length
       (refused_by "test/operations/ll_test_ratchet.expected" "Total -- records=19 traversals=12")
     = 2
    && List.length
         (refused_by "test/operations/ll_test_ratchet.expected" (adopted_row ^ " total=170"))
       = 3
    && List.length (refused_by "x/a.expected" adopted_row) = 2);
  p "a quoted-source entry spans the quotation and not the reason written after it"
    (at [ 50 ]
       (refused_by "test/operations/verdict_ratchet.expected"
          "  test/operations/foo.ml:row (K=64): -- the other 170 rows")
    && at [ 46 ]
         (refused_by "test/operations/operand_key_ratchet.expected"
            "test/operations/foo.ml -- `(x % 97)`: one of `170` sites"));
  p "a test outside the family, or with no golden checked in, contributes nothing"
    (List.equal String.equal
       (family_goldens
          ~dune_files:
            [
              ( "x/dune",
                {|(alias (name scans) (deps (alias runtest-a) (alias runtest-b)))
(test (name a))
(test (name c))
(rule (alias runtest-b) (action (diff b.expected b.actual)))|}
              );
            ]
          ~checked_in:(fun path -> not (String.equal path "x/a.expected")))
       [ "x/b.expected" ])

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: root :: generated ->
      scan root generated;
      controls ();
      Test_utils.Refusal_control_manifest.print "census_total_ratchet.ml"
  | _ -> Stdlib.exit 2
