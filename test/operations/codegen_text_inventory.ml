(** The inventory of everything in this tree that pins the TEXT of generated code (gh-ocannl-712).

    A codegen change -- how a float constant is spelled, how a loop is opened, which intrinsic a
    reduction renders as -- has to re-run every golden that snapshots emitted source and every test
    that asserts on it from a string literal. Before this, working that out was a search, and the
    search was incomplete twice over on gh-ocannl-623: [arrayjit/test/] carries codegen goldens that
    no [test/*.expected] glob sees, and a substring assertion inside a [.ml] is invisible to any
    [.expected] scan however thorough. The second miss is the expensive one -- those assertions are
    {!Verdict} claims, so they exit nonzero and fail a plain [dune build], not merely
    [dune runtest].

    So the population is enumerated here, and the enumeration is a golden. Adding a pin promotes a
    line; a codegen change reads the list and knows what it must re-run, before pushing rather than
    after a GPU box reports red days later.

    The rules are decided in {!Test_scan.Codegen_text_scan} as pure functions over a path and its
    contents, so [codegen_text_scan_cases] exercises the same code on input built to break it.

    {1 What the golden holds}

    The inventory itself: every member, and under each source site the fragments it pins, preceded
    by the emitter frontier the census was taken with. That is a list that moves when someone adds a
    pin -- which is the point, and is not the tally gh-ocannl-665 warns about, because the line that
    appears names the pin that appeared. The COUNTS go to stderr, and what the counts were there for
    -- the assurance that a scan reporting nothing scanned something -- is kept as floors, which
    fail the run rather than printing (gh-ocannl-701).

    {1 The emitter frontier}

    A test can reach generated text in memory, by calling a renderer and never touching
    [build_files/]. Which values those are used to be a list of five names inside the scan, and it
    was the one hand-maintained frontier left in it: three of the four review rounds on
    gh-ocannl-712 found a member it did not name, each time silently -- a route the list misses does
    not shrink the inventory visibly, it just leaves files off it.

    So the set is derived, from the compiler libraries' COMPILED interfaces (gh-ocannl-748): a value
    whose result is a [PPrint.document], or which takes a [Buffer.t] to write into, is an emitter
    whatever it is called. The types are read rather than the sources because the flagship member
    has neither an [.mli] nor a return annotation -- [C_syntax.compile_proc]'s document exists only
    as an inferred type, and a source scan would have to be told about it, which is the frontier
    again.

    The derivation's own silent failure is being handed nothing: a glob that stops matching leaves a
    scan finding no emitters and reporting a smaller census cheerfully. What closes that is the
    relationship this test pins -- the modules a library's wrapper interface DECLARES against the
    ones whose interfaces were actually read. Neither side is written down here; they are compared,
    and a difference fails the run with both lists on stderr.

    {1 The aliases that re-run them}

    A file is not yet a command: the kernel snapshots beside [top_down_prec] are diffed by
    [runtest-top_down_prec-extension], not by the test's own alias, and which of them it diffs
    depends on the backend the build resolves (gh-ocannl-1533). So each member is printed with the
    aliases that re-run it -- read from the dune files of the same roots by
    {!Test_scan.Rerun_aliases}, with the pform substitution a golden's rule makes -- and the golden
    closes on their union, ready to hand to [tools/test-run.sh run build]. A member no alias can be
    found for fails the run: the list is only worth reading if it is the whole of what to run.

    {1 The tests that build their own kernel text}

    A second population breaks on a codegen change without pinning anything: tests compiling or
    executing kernel text that did not come out of the [Assignments] pipeline -- a staged callback's
    verbatim text, a hand-built lowering through the [?prelowered] seam, a shipped kernel compiled
    alone. gh-ocannl-1338 ran every text pin on CUDA and HIP and master still went red on one of
    these (gh-ocannl-1685). They are listed in a section of their own, with the support values that
    reach such text DERIVED from the sources ({!Test_scan.Kernel_exec_scan}) and their own closing
    list of aliases, which a codegen change runs on every GPU backend.

    {1 The tests calling a backend's compiler themselves}

    A third population breaks on a change to a backend's builtins prelude, its compiler flags or how
    its compiler is resolved, and pins nothing either: [test_shared_builtins] compiles
    [Builtins_cc.source] and a probe of its own with [Cc_backend.compiler_command] (gh-ocannl-1772).
    They are listed by the backend whose compiler they call, through the same harness derivation,
    with a closing list per backend. The entries they are recognised by are spellings
    ({!Test_scan.Kernel_exec_scan.compiler_entries}); a spelling no source handed over still uses
    fails the run, since an entry renamed away would leave the population empty and the list
    cheerful.

    {1 Pins one file away}

    A test can pin emitted text through a [test/support] helper that tests the text it is handed:
    gh-ocannl-1736 moved the residency markers into [Ll_test], and after that
    [masked_persistent_fragment] -- pinning them through [Ll_test.residency_holds] -- read here as a
    member pinning nothing, complete. The helpers are derived from the support modules
    ({!Test_scan.Codegen_text_scan.helper_predicates}), each call is judged by its caller's
    provenance, and every helper some test hands text to is listed with the fragments it spells and
    the tests calling it, whose aliases are the ones above. *)

open Base
open Stdio

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_scan.Scan_argv]. *)
let argv = Test_scan.Scan_argv.expand Stdlib.Sys.argv

module Scan = Test_scan.Codegen_text_scan
module Floors = Test_scan.Scan_floors
module Attempt = Test_scan.Scan_attempt
module Rerun = Test_scan.Rerun_aliases
module Exec = Test_scan.Kernel_exec_scan

(** Lower bounds per scanned root, well below the census of the day they were written. See
    {!Test_scan.Scan_floors}: a glob that breaks goes to zero, a member added moves nothing. *)
let golden_floors = [ ("test", 12); ("arrayjit/test", 2) ]

(** [arrayjit/test] has its own floor, and the reason it once looked as though it could not have
    sites is worth keeping: [test_utils] is a [neural_nets_lib] library, so the arrayjit tests
    cannot link {!Test_utils.Generated} at all. They reach generated text the third way instead --
    calling [C_syntax.compile_proc] and rendering the document -- which is precisely the route the
    first version of this scan did not model, and which made a whole root look empty (Codex P2,
    round 2). *)
let source_floors = [ ("test", 20); ("arrayjit/test", 2) ]

(** The tests building their own kernel text, likewise floored: [test/operations] alone held 47 on
    the day gh-ocannl-1685 derived them. *)
let exec_floors = [ ("test", 20) ]

(** The tests calling a backend's compiler with source of their own: on the day gh-ocannl-1772
    derived them, [test/operations] held a dozen and [arrayjit/test] four. *)
let compile_floors = [ ("test", 5); ("arrayjit/test", 2) ]

(** The support helpers pinning emitted text for their callers: [Ll_test]'s residency predicates on
    the day gh-ocannl-1772 derived them. *)
let helper_floors = [ ("test", 1) ]

(** The sources handed over only so that the harness frontier can be derived from them: the runners'
    support libraries, which tests link ([Bench_harness]) but which are no test themselves, so no
    census of members reads them. Their [dune] file comes too, since what makes a module a harness
    is the [(library)] stanza owning it. *)
let harness_only_root = "benchmarks/"

(** Files the globs hand over that are not members of either population, each for a reason that is
    about this scan rather than about them.

    Both are checked to be present below: an exclusion naming a file the globs no longer produce has
    stopped excluding anything and is hiding the next file that takes its name. *)
let excluded =
  [
    ( "test/support/generated.ml",
      "the freshness-checked reader itself -- it opens build_files/ because that is its job, and a \
       codegen change does not re-run it" );
    ( "test/operations/codegen_text_inventory.expected",
      "this scan's own output, which quotes the fragments the sources pin: including it would make \
       the scan's result depend on its own golden, so a promote would take two rounds to converge"
    );
    ( "test/operations/generated_provenance.ml",
      "the test OF the freshness-checked reader: it writes its artifact contents by hand and \
       checks that a stale or overwritten one is refused, so the strings under it are fixtures no \
       backend emits and a codegen change does not re-run it" );
  ]

(** Members allowed to read a kernel's statements as the text after its [Main logic] banner, each
    for a reason that is about the reading rather than about the kernel. Checked to be still doing
    it below: an exemption whose file stopped reading that way is hiding the next one that does. *)
let after_marker_exempt =
  [
    ( "test/operations/cc_kernel_outline.ml",
      "the outlining test's negative control: it reads after an outlined kernel's banner to show \
       that reading holds none of the kernel's statements" );
  ]

let read path = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all

module Dune = Test_scan.Dune_stanza_scan

let stale_exclusions ~excluded ~handed_over =
  List.filter excluded ~f:(fun (path, _) -> not (List.mem handed_over path ~equal:String.equal))

let refuse_stale_exclusions ~fail stale =
  List.iter stale ~f:(fun (path, reason) ->
      fail
        (Printf.sprintf
           "the exclusion for %s (%s) names a file the globs no longer hand over -- drop it, or \
            fix the path it was meant to name"
           path reason))

let stale_entries ~entries ~named =
  List.filter entries ~f:(fun entry -> not (List.mem named entry ~equal:Poly.equal))

let refuse_stale_entries ~fail stale =
  List.iter stale ~f:(fun entry ->
      fail
        (Printf.sprintf
           "the compiler entry %s (%s) is spelled by no source handed over, so the tests calling \
            that compiler are found by nothing -- rename it to the spelling the tree uses now, or \
            drop it (gh-ocannl-1772)"
           (Exec.entry_name entry) entry.Exec.entry_backend))

let refuse_raised ~fail raised =
  List.iter raised ~f:(fun (path, exn) ->
      fail
        (Printf.sprintf
           "the scan raised %s on %s -- a defect in the scanner rather than in the file, which \
            parses, so the census is short by whatever the file pins"
           exn path))

(* gh-ocannl-1533: a member nothing re-runs by name is a hole in the list of what to run. *)
let refuse_unplaced ~fail unplaced =
  List.iter unplaced ~f:(fun path ->
      fail
        (Printf.sprintf
           "%s pins generated text or builds kernel text of its own, and no stanza of the dune \
            files handed over re-runs it by an alias -- no rule diffs it and no rule or test runs \
            its module -- so a codegen change cannot name what to run for it; give it a per-test \
            alias (gh-ocannl-1533)"
           path))

(* gh-ocannl-1581: a route re-running a golden only under a condition no backend's build meets -- a
   value the configuration reader writes for no backend, or a pform reading no file of the reader's
   -- names no box to run it on, and the first kind is a golden no run compares at all. *)
let unselected routes =
  List.concat_map routes ~f:(fun (path, routes) ->
      List.filter_map routes ~f:(fun route ->
          match Rerun.selection route with
          | Rerun.Every_backend | Rerun.Only (_ :: _) -> None
          | Rerun.Only [] | Rerun.Unnamed _ -> Some (path, Rerun.render route)))

let refuse_unselected ~fail unselected =
  List.iter unselected ~f:(fun (path, route) ->
      fail
        (Printf.sprintf
           "%s is re-run by %s, a condition no backend's build is named by: either the \
            configuration reader (test/config/config_reads.ml) writes that value for no backend, \
            so no run compares the golden, or the pform reads no file the reader writes, so which \
            box compares it cannot be said -- rename the golden to a backend's value, or teach \
            Config_reads the file (gh-ocannl-1581)"
           path route))

(* gh-ocannl-1294: a test that opens build_files/ itself reads whatever is there -- a previous run's
   artifact, a same-named routine's -- and, where nothing is, passes over zero files. *)
let refuse_direct_reads ~fail sites =
  List.iter sites ~f:(fun site ->
      if site.Scan.direct then
        fail
          (Printf.sprintf
             "%s reads build_files/ itself (Utils.build_files_dir or Utils.build_file), so nothing \
              ties what it reads to this run's compile: a stale artifact, another routine's of the \
              same name, or no file at all reads the same. Read generated code through \
              Test_utils.Generated -- init before the first compile, read or assert_emits after it \
              (gh-ocannl-1294)"
             site.Scan.site_path))

(* gh-ocannl-1708: the reading gh-ocannl-1678 retired. Under outlining the text after a kernel's
   banner holds only calls to its parts, so a positive claim over it goes red and a negative one
   passes without checking anything -- and no regular run forces outlining, so a new reader of that
   shape would pass every run until a kernel grew large enough to outline. *)
let refuse_after_marker_reads ~fail ~exempt sites =
  List.iter sites ~f:(fun site ->
      if
        site.Scan.after_marker
        && not (List.Assoc.mem exempt site.Scan.site_path ~equal:String.equal)
      then
        fail
          (Printf.sprintf
             "%s reads a kernel's statements as the text after its Main logic banner (a \
              substr_index on it), which under outlining holds only calls to the kernel's parts: a \
              positive claim over it goes red and a negative one checks nothing. Read the \
              statements through Test_utils.Generated.main_logic or main_logic_lines, which splice \
              each part in at its call (gh-ocannl-1708)"
             site.Scan.site_path))

let stale_after_marker_exemptions ~exempt sites =
  List.filter exempt ~f:(fun (path, _) ->
      not
        (List.exists sites ~f:(fun site ->
             site.Scan.after_marker && String.equal site.Scan.site_path path)))

let refuse_stale_after_marker_exemptions ~fail stale =
  List.iter stale ~f:(fun (path, reason) ->
      fail
        (Printf.sprintf
           "the after-marker exemption for %s (%s) names a file that no longer reads after the \
            Main logic banner -- drop it"
           path reason))

let refusal_control ~wrapper =
  let source = "test/operations/codegen_text_inventory.ml" in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "the exclusion for %s (%s) names a file the globs no longer hand over -- drop it, or fix \
         the path it was meant to name"
  in
  let stale = stale_exclusions ~excluded:[ ("gone.expected", "fixture") ] ~handed_over:[] in
  refuse_stale_exclusions ~fail stale;
  Verdict.p "an exclusion absent from the hand-over reaches the stale-exclusion refusal"
    (!refused && List.length stale = 1);
  (* gh-ocannl-1772: an entry no source spells, judged by the reader the census uses -- a fixture
     spelling only one of two entries, the other in a comment. *)
  let entry_refused = ref false in
  let fail _message =
    entry_refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "the compiler entry %s (%s) is spelled by no source handed over, so the tests calling that \
         compiler are found by nothing -- rename it to the spelling the tree uses now, or drop it \
         (gh-ocannl-1772)"
  in
  let named =
    Exec.entries_named
      ~contents:
        "module Cc = Context.Cc_backend\n\
         (* Nvrtc.compile_to_ptx *)\n\
         let cmd = Cc.compiler_command ()\n"
  in
  let entries =
    List.filter Exec.compiler_entries ~f:(fun e ->
        List.mem
          [ "compiler_command"; "compile_to_ptx" ]
          (List.last_exn e.Exec.entry_path) ~equal:String.equal)
  in
  let stale = stale_entries ~entries ~named in
  refuse_stale_entries ~fail stale;
  Verdict.p "a compiler entry only a comment spells reaches the stale-entry refusal"
    (!entry_refused
    && List.equal String.equal (List.map stale ~f:Exec.entry_name) [ "Nvrtc.compile_to_ptx" ]);
  (* The non-parse path: a scanner defect on a file that parses is reported as the exception it is,
     not as an unparsable file. The provocation is [boundaries_of]'s own precondition, broken the
     one way the scan's constructors make unreachable from source. *)
  let raised = ref false in
  let fail _message =
    raised := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "the scan raised %s on %s -- a defect in the scanner rather than in the file, which \
         parses, so the census is short by whatever the file pins"
  in
  let outcome =
    Attempt.run (fun () ->
        Scan.boundaries_of { Scan.no_provenance with Scan.uncertainty = Scan.Unresolved })
  in
  (match outcome with
  | Attempt.Raised exn -> refuse_raised ~fail [ ("test/parses.ml", exn) ]
  | Attempt.Scanned _ | Attempt.Unparsed _ -> ());
  Verdict.p
    "a scanner exception on a parsing file reaches the scanner-defect refusal, not the parse one"
    (!raised
    &&
    match outcome with
    | Attempt.Raised exn -> String.is_substring exn ~substring:"Invalid_argument"
    | Attempt.Scanned _ | Attempt.Unparsed _ -> false);
  (* gh-ocannl-1207: the parse refusal, executed: this scan as a child, handed a source that does
     not parse. *)
  Test_scan.Refusal_control_manifest.with_tree
    [ ("test/unparsable.ml", "let =\n") ]
    (fun root ->
      Verdict.p "a scan handed a source that does not parse refuses it, exiting 1"
        (Test_scan.Refusal_control_manifest.refuses ~source
           ~claim:"a scan handed a source that does not parse refuses it, exiting 1"
           ~exe:(Fresh_process.executable ())
           [ root; Stdlib.Filename.concat root "test/unparsable.ml" ]));
  (* gh-ocannl-1233: the declared-but-unread refusal, executed the same way: this scan handed a
     library's wrapper interface alone -- the [emitter_fixture] library's, copied into a tree of its
     own, so that neither the hand-over nor the wrapper's directory holds the members it
     declares. *)
  Test_scan.Refusal_control_manifest.with_tree
    [ ("lib/emitter_fixture.cmi", read wrapper) ]
    (fun root ->
      Verdict.p
        "a scan handed a wrapper interface without its members' interfaces refuses it, exiting 1"
        (Test_scan.Refusal_control_manifest.refuses ~source
           ~claim:
             "a scan handed a wrapper interface without its members' interfaces refuses it, \
              exiting 1"
           ~exe:(Fresh_process.executable ())
           [ root; Stdlib.Filename.concat root "lib/emitter_fixture.cmi" ]));
  (* gh-ocannl-1294: the direct-read refusal, on a test that lists build_files/ itself and pins a
     fragment of what it finds there -- classified by the scan's own reader, so the control fails if
     that reader stops marking such a test. *)
  let direct_fixture =
    "let sources =\n\
    \  Stdlib.Sys.readdir (Utils.build_files_dir ())\n\
    \  |> Array.to_list\n\
    \  |> List.map ~f:Stdio.In_channel.read_all\n\
     let () =\n\
    \  let has substring = List.exists sources ~f:(String.is_substring ~substring) in\n\
    \  Verdict.p \"guard\" (has \": (float)(0.0))\")\n"
  in
  let direct_refused = ref false in
  let fail _message =
    direct_refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "%s reads build_files/ itself (Utils.build_files_dir or Utils.build_file), so nothing ties \
         what it reads to this run's compile: a stale artifact, another routine's of the same \
         name, or no file at all reads the same. Read generated code through Test_utils.Generated \
         -- init before the first compile, read or assert_emits after it (gh-ocannl-1294)"
  in
  refuse_direct_reads ~fail
    (Option.to_list
       (Scan.classify_source ~emitters:[] ~helpers:[] ~path:"test/direct.ml"
          ~contents:direct_fixture));
  Verdict.p "a test reading build_files/ itself reaches the direct-read refusal" !direct_refused;
  (* gh-ocannl-1708: the after-marker refusal, on the reader test_uniform_virtual_lane had before
     gh-ocannl-1678 -- a substr_index on the banner, then a slice -- classified by the scan's own
     reader, beside the same file with that reading exempted. *)
  let after_marker_fixture =
    "let has sub s =\n\
    \  let body = match String.substr_index s ~pattern:\"Main logic\" with\n\
    \    | Some i -> String.subo s ~pos:i\n\
    \    | None -> s\n\
    \  in\n\
    \  String.is_substring body ~substring:sub\n\
     let () = Verdict.p \"lane\" (has \"_uniform_lane(\" (Generated.read \"r\"))\n"
  in
  let after_marker_sites =
    Option.to_list
      (Scan.classify_source ~emitters:[] ~helpers:[] ~path:"test/after_marker.ml"
         ~contents:after_marker_fixture)
  in
  let after_refused = ref false in
  let fail _message =
    after_refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "%s reads a kernel's statements as the text after its Main logic banner (a substr_index on \
         it), which under outlining holds only calls to the kernel's parts: a positive claim over \
         it goes red and a negative one checks nothing. Read the statements through \
         Test_utils.Generated.main_logic or main_logic_lines, which splice each part in at its \
         call (gh-ocannl-1708)"
  in
  refuse_after_marker_reads ~fail ~exempt:[] after_marker_sites;
  let refused_unexempted = !after_refused in
  after_refused := false;
  refuse_after_marker_reads ~fail ~exempt:[ ("test/after_marker.ml", "fixture") ] after_marker_sites;
  Verdict.p
    "a test reading after the Main logic banner by hand reaches the after-marker refusal, unless \
     exempted"
    (refused_unexempted && not !after_refused);
  let stale_refused = ref false in
  let fail _message =
    stale_refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "the after-marker exemption for %s (%s) names a file that no longer reads after the Main \
         logic banner -- drop it"
  in
  let stale =
    stale_after_marker_exemptions
      ~exempt:[ ("test/direct.ml", "fixture"); ("test/after_marker.ml", "fixture") ]
      (Option.to_list
         (Scan.classify_source ~emitters:[] ~helpers:[] ~path:"test/direct.ml"
            ~contents:direct_fixture)
      @ after_marker_sites)
  in
  refuse_stale_after_marker_exemptions ~fail stale;
  Verdict.p
    "an after-marker exemption of a file reading no banner reaches the stale-exemption refusal"
    (!stale_refused && List.equal String.equal (List.map stale ~f:fst) [ "test/direct.ml" ]);
  (* gh-ocannl-1533: a golden beside a dune file that diffs something else is a member nothing
     re-runs, through the same reader the census uses. *)
  let unplaced_refused = ref false in
  let fail _message =
    unplaced_refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "%s pins generated text or builds kernel text of its own, and no stanza of the dune files \
         handed over re-runs it by an alias -- no rule diffs it and no rule or test runs its \
         module -- so a codegen change cannot name what to run for it; give it a per-test alias \
         (gh-ocannl-1533)"
  in
  let stanzas =
    Rerun.stanzas_of
      [
        ( "t",
          "(test (name kept))\n\
           (rule (alias runtest-kept-c) (action (diff kept.c.expected kept.c.actual)))\n" );
      ]
  in
  let path = "t/lost.c.expected" in
  refuse_unplaced ~fail (if List.is_empty (Rerun.for_golden stanzas ~path) then [ path ] else []);
  Verdict.p "a golden no rule of its directory diffs reaches the unplaced-member refusal"
    !unplaced_refused;
  (* gh-ocannl-1581: a snapshot under an extension no backend emits, diffed by the same reader's
     route as a real one -- so the refusal is reached through [Rerun.selection], and the control
     fails if the reader starts naming a backend for it. *)
  let unselected_refused = ref false in
  let fail _message =
    unselected_refused := true;
    Test_scan.Refusal_control_manifest.observe_failure ~source
      ~format:
        "%s is re-run by %s, a condition no backend's build is named by: either the configuration \
         reader (test/config/config_reads.ml) writes that value for no backend, so no run compares \
         the golden, or the pform reads no file the reader writes, so which box compares it cannot \
         be said -- rename the golden to a backend's value, or teach Config_reads the file \
         (gh-ocannl-1581)"
  in
  let stanzas =
    Rerun.stanzas_of
      [
        ( "t",
          "(rule (alias runtest-kept-extension) (action (diff \
           \"kept.%{read:../config/ocannl_backend_extension.txt}.expected\" kept.actual)))\n" );
      ]
  in
  let path = "t/kept.opencl.expected" in
  let routes = Rerun.for_golden stanzas ~path in
  refuse_unselected ~fail (unselected [ (path, routes) ]);
  Verdict.p
    "a snapshot under an extension no backend emits reaches the unselected-route refusal, through \
     the route that diffs it"
    (!unselected_refused && not (List.is_empty routes));
  Test_scan.Refusal_control_manifest.print source

let () =
  if Array.length argv = 3 && String.equal argv.(1) "--refusal-control" then (
    refusal_control ~wrapper:argv.(2);
    Stdlib.exit 0);
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <file...>\n" argv.(0);
    Stdlib.exit 1);
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. The translation is [Dune_stanza_scan]'s, shared with the
     other scans so that two of them cannot disagree about what a path names. *)
  let base = Dune.base_dir argv.(1) in
  let arguments =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let is_excluded path = List.Assoc.mem excluded path ~equal:String.equal in
  let of_suffix suffix =
    List.filter arguments ~f:(fun (name, _) -> String.is_suffix name ~suffix)
  in
  let golden_files = of_suffix ".expected" in
  let in_harness_only_root (name, _) = String.is_prefix name ~prefix:harness_only_root in
  let harness_only_ml, all_ml = List.partition_tf (of_suffix ".ml") ~f:in_harness_only_root in
  (* The dune files of the scanned roots, whose rules say which alias re-runs each member. *)
  let stanzas =
    List.filter arguments ~f:(fun (name, _) ->
        String.equal name "dune" || String.is_suffix name ~suffix:"/dune")
    |> List.map ~f:(fun (name, on_disk) ->
        (Option.value (String.chop_suffix name ~suffix:"/dune") ~default:"", read on_disk))
    |> Rerun.stanzas_of
  in
  (* The emitter frontier, derived from the interfaces dune handed over rather than listed here. See
     the header: the derivation is what keeps a renderer added to a library from silently widening
     the blind spot, and the module-coverage claim below is what keeps the derivation from failing
     silently in its turn. *)
  let frontier = Emitter_frontier.derive (List.map (of_suffix ".cmi") ~f:snd) in
  let emitters =
    List.map frontier.Emitter_frontier.emitters ~f:(fun e ->
        {
          Scan.emitter_name = e.Emitter_frontier.name;
          Scan.origins = e.Emitter_frontier.origins;
          Scan.destinations =
            List.map e.Emitter_frontier.destinations ~f:(function
              | Emitter_frontier.At_label label -> Scan.At_label label
              | Emitter_frontier.At_position position -> Scan.At_position position);
        })
  in
  let present name = List.exists all_ml ~f:(fun (n, _) -> String.equal n name) in
  let source_files =
    (* Two kinds of argument are a second copy of a source already in the list, and both would make
       the inventory differ between machines rather than between trees.

       dune's preprocessed twin [x.pp.ml] is the ppx expansion of [x.ml], and exists only where the
       library that owns it is built. And a [(select)] target [x.ml] is a COPY of whichever of
       [x.real.ml] / [x.missing.ml] this machine's toolchain selected -- so on a box with Metal it
       holds the real test and on a box without it holds the stub, and the inventory would move with
       the hardware. Both are dropped only when the source they copy is present, so a file genuinely
       named that way is not lost; the [.real.ml] and [.missing.ml] originals are inventoried on
       every machine alike. *)
    List.filter all_ml ~f:(fun (name, _) ->
        match String.chop_suffix name ~suffix:".pp.ml" with
        | Some stem -> not (present (stem ^ ".ml"))
        | None -> (
            match String.chop_suffix name ~suffix:".ml" with
            | Some stem -> not (present (stem ^ ".real.ml") || present (stem ^ ".missing.ml"))
            | None -> true))
  in
  let handed_over = List.map (golden_files @ source_files) ~f:fst in
  let stale = stale_exclusions ~excluded ~handed_over in
  refuse_stale_exclusions ~fail:Verdict.fail stale;
  let by_itself =
    List.filter_map golden_files ~f:(fun (name, on_disk) ->
        if is_excluded name then None else Scan.classify_golden ~path:name ~contents:(read on_disk))
  in
  let unparsed = ref [] in
  let raised = ref [] in
  let rejected = ref [] in
  (* gh-ocannl-1772: the support helpers testing a text they are handed, derived from every support
     module -- a module some [(library)] stanza owns, or a runner library's -- before the census,
     which judges each call of one by its caller's provenance. A module that does not parse is
     reported by the census (a test-root source) or by the harness derivation below. *)
  let helpers =
    List.filter (harness_only_ml @ source_files) ~f:(fun (name, _) ->
        Rerun.owned_by_library stanzas ~path:name && not (is_excluded name))
    |> List.concat_map ~f:(fun (name, on_disk) ->
        match
          Attempt.run (fun () ->
              Scan.helper_predicates ~emitters ~module_name:(Exec.module_name_of_path name)
                ~path:name ~contents:(read on_disk))
        with
        | Attempt.Scanned helpers -> helpers
        | Attempt.Unparsed _ -> []
        | Attempt.Raised exn ->
            raised := (name, exn) :: !raised;
            [])
  in
  let sites =
    List.filter_map source_files ~f:(fun (name, on_disk) ->
        if is_excluded name then None
        else
          let contents = read on_disk in
          match
            Attempt.run (fun () ->
                ( Scan.rejections ~emitters ~path:name ~contents,
                  Scan.classify_source ~emitters ~helpers ~path:name ~contents ))
          with
          | Attempt.Scanned (rejections, site) ->
              rejected := rejections @ !rejected;
              site
          | Attempt.Unparsed _ ->
              unparsed := name :: !unparsed;
              None
          | Attempt.Raised exn ->
              raised := (name, exn) :: !raised;
              None)
  in
  (* Goldens that nothing about the file itself made members, paired with the test beside them. See
     Codegen_text_scan.classify_associated: the markers describe whole dumps, and a golden can hold
     emitted text in fragments instead. *)
  let stems =
    List.map sites ~f:(fun s -> (Scan.source_stem s.Scan.site_path, s.Scan.site_path))
    |> Map.of_alist_reduce (module String) ~f:(fun first _ -> first)
  in
  let already = Set.of_list (module String) (List.map by_itself ~f:(fun g -> g.Scan.path)) in
  let associated =
    List.filter_map golden_files ~f:(fun (name, on_disk) ->
        if is_excluded name || Set.mem already name then None
        else
          match String.chop_suffix name ~suffix:".expected" with
          | None -> None
          | Some stem ->
              Option.bind (Map.find stems stem) ~f:(fun source ->
                  Scan.classify_associated ~path:name ~contents:(read on_disk) ~source))
  in
  let goldens = by_itself @ associated in
  (* gh-ocannl-1685: the tests building their own kernel text. The harness is every support module
     -- a module some [(library)] stanza owns, or a runner library's -- read for the values reaching
     such text; the members are the test sources reaching them, or reaching such text directly. *)
  let parses (name, on_disk) =
    let contents = read on_disk in
    match Attempt.run (fun () -> ignore (Scan.structure_of contents)) with
    | Attempt.Scanned () -> Some (Exec.module_name_of_path name, contents)
    | Attempt.Unparsed _ | Attempt.Raised _ ->
        (* A test-root source that fails here failed the census above already. *)
        if in_harness_only_root (name, on_disk) then unparsed := name :: !unparsed;
        None
  in
  let unparsed_by_census = !unparsed in
  let harness =
    Exec.harness
      (List.filter_map
         (List.filter (harness_only_ml @ source_files) ~f:(fun (name, _) ->
              Rerun.owned_by_library stanzas ~path:name))
         ~f:parses)
  in
  let executing =
    List.filter_map source_files ~f:(fun (name, on_disk) ->
        if List.mem unparsed_by_census name ~equal:String.equal then None
        else
          match Attempt.run (fun () -> Exec.classify harness ~contents:(read on_disk)) with
          | Attempt.Scanned [] -> None
          | Attempt.Scanned routes -> Some (name, routes)
          | Attempt.Unparsed _ ->
              unparsed := name :: !unparsed;
              None
          | Attempt.Raised exn ->
              raised := (name, exn) :: !raised;
              None)
    (* A support module reaching such text is the harness its callers are found through, not a test
       a codegen change runs by name -- unless something does run it, as an inline-test module. *)
    |> List.filter_map ~f:(fun (name, routes) ->
        match Rerun.for_source stanzas ~path:name with
        | [] when Rerun.owned_by_library stanzas ~path:name -> None
        | runs -> Some (name, routes, runs))
  in
  let exec_unplaced =
    List.filter_map executing ~f:(fun (name, _, runs) ->
        if List.is_empty runs then Some name else None)
  in
  (* The two populations the harness routes reach, each printed with only the routes reaching it:
     kernels built outside the pipeline (gh-ocannl-1685), and compilers called directly
     (gh-ocannl-1772). A test can be in both. *)
  let population ~keep =
    ( List.filter_map executing ~f:(fun (name, routes, runs) ->
          match Exec.restrict harness ~keep routes with
          | [] -> None
          | routes -> Some (name, routes, runs)),
      Map.filter_map harness.Exec.values ~f:(fun routes ->
          match Exec.restrict harness ~keep routes with [] -> None | routes -> Some routes) )
  in
  let kernel_exec, kernel_values = population ~keep:Exec.builds_kernel_text in
  let compiling, compiling_values =
    population ~keep:(fun route -> Option.is_some (Exec.compiler_of route))
  in
  let backends_called routes =
    Exec.base_routes harness routes |> List.filter_map ~f:Exec.compiler_of
  in
  (* The entries' own relationship: each still a spelling some source handed over uses. *)
  let named_entries =
    List.concat_map (source_files @ harness_only_ml) ~f:(fun (name, on_disk) ->
        if is_excluded name then []
        else
          match Attempt.run (fun () -> Exec.entries_named ~contents:(read on_disk)) with
          | Attempt.Scanned entries -> entries
          | Attempt.Unparsed _ | Attempt.Raised _ -> [])
  in
  let stale_entry_list = stale_entries ~entries:Exec.compiler_entries ~named:named_entries in
  refuse_stale_entries ~fail:Verdict.fail stale_entry_list;
  (* The helpers some test hands text to, each with its callers. *)
  let callers_of key =
    List.filter_map sites ~f:(fun s ->
        List.find_map s.Scan.through ~f:(fun (k, validated) ->
            Option.some_if (String.equal k key) (s.Scan.site_path, validated)))
  in
  let called_helpers =
    List.filter_map helpers ~f:(fun h ->
        match callers_of h.Scan.helper_key with [] -> None | callers -> Some (h, callers))
    |> List.sort ~compare:(fun (a, _) (b, _) ->
        String.compare
          (a.Scan.helper_path ^ " " ^ a.Scan.helper_key)
          (b.Scan.helper_path ^ " " ^ b.Scan.helper_key))
  in
  List.iter !unparsed ~f:(fun path ->
      Verdict.fail
        (Printf.sprintf
           "%s does not parse as OCaml, so the scan cannot say whether it pins emitted text" path));
  refuse_raised ~fail:Verdict.fail (List.rev !raised);
  List.iter (List.sort !rejected ~compare:String.compare) ~f:Verdict.fail;
  refuse_direct_reads ~fail:Verdict.fail sites;
  refuse_after_marker_reads ~fail:Verdict.fail ~exempt:after_marker_exempt sites;
  let stale_after_marker = stale_after_marker_exemptions ~exempt:after_marker_exempt sites in
  refuse_stale_after_marker_exemptions ~fail:Verdict.fail stale_after_marker;
  let golden_paths = List.map goldens ~f:(fun g -> g.Scan.path) in
  let golden_routes =
    Map.of_alist_exn
      (module String)
      (List.map golden_paths ~f:(fun path -> (path, Rerun.for_golden stanzas ~path)))
  in
  let site_routes =
    Map.of_alist_exn
      (module String)
      (List.map sites ~f:(fun s ->
           (s.Scan.site_path, Rerun.for_source stanzas ~path:s.Scan.site_path)))
  in
  let routes_of routes path = Option.value (Map.find routes path) ~default:[] in
  let unplaced =
    List.filter_map
      (Map.to_alist golden_routes @ Map.to_alist site_routes)
      ~f:(fun (path, routes) -> if List.is_empty routes then Some path else None)
  in
  refuse_unplaced ~fail:Verdict.fail (unplaced @ exec_unplaced);
  let unselected = unselected (Map.to_alist golden_routes) in
  refuse_unselected ~fail:Verdict.fail unselected;
  let site_paths = List.map sites ~f:(fun s -> s.Scan.site_path) in
  let exec_paths = List.map kernel_exec ~f:(fun (name, _, _) -> name) in
  let compile_paths = List.map compiling ~f:(fun (name, _, _) -> name) in
  let helper_paths =
    List.dedup_and_sort ~compare:String.compare
      (List.map called_helpers ~f:(fun (h, _) -> h.Scan.helper_path))
  in
  Floors.report ~floors:golden_floors ~noun:"golden" ~what:"Goldens holding emitted text"
    golden_paths;
  Floors.report ~floors:source_floors ~noun:"site" ~what:"Sources pinning emitted text" site_paths;
  Floors.report ~floors:exec_floors ~noun:"test" ~what:"Tests building their own kernel text"
    exec_paths;
  Floors.report ~floors:compile_floors ~noun:"test"
    ~what:"Tests calling a backend's compiler themselves" compile_paths;
  Floors.report ~floors:helper_floors ~noun:"helper"
    ~what:"Support modules pinning emitted text for their callers" helper_paths;
  eprintf "Files handed over: %d goldens, %d sources.\n" (List.length golden_files)
    (List.length source_files);
  print_endline
    "What in this tree pins the TEXT of generated code (gh-ocannl-712). A codegen change re-runs\n\
     every golden listed here, on the backend whose family it names, and every test source under\n\
     it. The rules are stated in test/support/codegen_text_scan.ml; the counts scanned go to\n\
     stderr, since a tally in a golden moves on every correct addition anywhere (gh-ocannl-665).\n";
  printf "== emitters the census was taken with ==\n";
  printf "interfaces read: %s\n"
    (String.concat ~sep:", "
       (List.map frontier.Emitter_frontier.interfaces ~f:(fun i -> i.Emitter_frontier.library)));
  List.iter frontier.Emitter_frontier.emitters ~f:(fun e ->
      printf "%s%s [%s]\n" e.Emitter_frontier.name
        (match e.Emitter_frontier.destinations with
        | [] -> ""
        | destinations ->
            " writes into "
            ^ String.concat ~sep:" " (List.map destinations ~f:Emitter_frontier.render_destination))
        (String.concat ~sep:" " e.Emitter_frontier.origins));
  (* The other half of the derivation, printed rather than dropped. These produce a document out of
     strings, numbers and other documents -- nothing the libraries define -- so they render no
     program, and matching such a name behind any qualifier would make a member of every test that
     calls `Bench_args.int`. A renderer that lands here is a miss, and listing them is what makes
     that miss a line in a diff rather than an absence (gh-ocannl-748). *)
  printf
    "\ndocument combinators, given nothing of the libraries to render, so not on the frontier:\n";
  List.iter frontier.Emitter_frontier.combinators ~f:(fun c ->
      printf "%s [%s]\n" c.Emitter_frontier.name (String.concat ~sep:" " c.Emitter_frontier.origins));
  printf "\n== goldens holding emitted kernel or IR text ==\n";
  printf "roots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:golden_floors golden_paths));
  List.iter
    (List.sort goldens ~compare:(fun a b -> String.compare a.Scan.path b.Scan.path))
    ~f:(fun g ->
      let origin =
        (match g.Scan.by_extension with Some ext -> [ "extension " ^ ext ] | None -> [])
        @ (match g.Scan.tags with [] -> [] | tags -> [ "markers " ^ String.concat ~sep:" " tags ])
        @ match g.Scan.beside with Some source -> [ "beside " ^ source ] | None -> []
      in
      let rerun =
        match routes_of golden_routes g.Scan.path with
        | [] -> []
        | routes -> [ "re-run by " ^ String.concat ~sep:", " (List.map routes ~f:Rerun.render) ]
      in
      printf "%s [%s] %s\n" g.Scan.path
        (String.concat ~sep:" " g.Scan.families)
        (String.concat ~sep:"; " (origin @ rerun)));
  printf "\n== test sources pinning emitted text ==\n";
  printf "roots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:source_floors site_paths));
  List.iter
    (List.sort sites ~compare:(fun a b -> String.compare a.Scan.site_path b.Scan.site_path))
    ~f:(fun s ->
      let flags =
        (if s.Scan.direct then [ "reads build_files/ directly" ] else [])
        @ (if s.Scan.after_marker then [ "reads after the Main logic banner, exempted" ] else [])
        @ (if s.Scan.rendered then [ "renders generated text in memory" ] else [])
        @ List.map s.Scan.through ~f:(fun (helper, validated) ->
            "pins through " ^ helper ^ if validated then "" else ", haystack unvalidated")
        @ (match routes_of site_routes s.Scan.site_path with
          | [] -> []
          | routes -> [ "run by " ^ String.concat ~sep:", " (List.map routes ~f:Rerun.render) ])
        @
        match s.Scan.partial with
        | [] -> []
        | boundaries ->
            [
              "also pins text this scan cannot name: "
              ^ String.concat ~sep:", " (List.map boundaries ~f:Scan.boundary_name);
            ]
      in
      printf "%s%s\n" s.Scan.site_path
        (match flags with [] -> "" | flags -> " [" ^ String.concat ~sep:"; " flags ^ "]");
      List.iter s.Scan.pins ~f:(fun pin -> printf "    %s\n" pin));
  printf
    "\n\
     == support helpers pinning emitted text for their callers ==\n\
     A test/support value testing the text a test hands it (gh-ocannl-1772), with the fragments it\n\
     spells itself -- a field of a record it is handed is itemised as every fragment its module\n\
     builds that field with -- and the tests calling it, each run by its aliases above. A caller\n\
     whose haystack the scan could not show to be generated text is listed all the same.\n";
  printf "roots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:helper_floors helper_paths));
  List.iter called_helpers ~f:(fun (h, callers) ->
      let flags =
        [
          "called by "
          ^ String.concat ~sep:", "
              (List.map callers ~f:(fun (path, validated) ->
                   if validated then path else path ^ " (haystack unvalidated)"));
        ]
        @
        match h.Scan.helper_partial with
        | [] -> []
        | boundaries ->
            [
              "also pins text this scan cannot name: "
              ^ String.concat ~sep:", " (List.map boundaries ~f:Scan.boundary_name);
            ]
      in
      printf "%s: %s [%s]\n" h.Scan.helper_path h.Scan.helper_key (String.concat ~sep:"; " flags);
      List.iter h.Scan.helper_pins ~f:(fun pin -> printf "    %s\n" pin));
  printf
    "\n\
     == the aliases a codegen change runs ==\n\
     Every alias named above, once. Hand them to `tools/test-run.sh run build`; an alias re-running\n\
     a golden only when a pform reads some value covers that golden on the backend that makes it\n\
     so, listed next, and the goldens' families say which backends a change has to be run on.\n";
  Map.data golden_routes @ Map.data site_routes
  |> List.concat_map ~f:(List.map ~f:(fun (r : Rerun.route) -> r.Rerun.alias))
  |> List.dedup_and_sort ~compare:String.compare
  |> List.iter ~f:print_endline;
  (* gh-ocannl-1581: the conditional ones, split by the backend each condition's value means -- as
     the reader writing the file decides it, by running its own code over every backend. *)
  printf
    "\n\
     == of those, the aliases re-running a golden only on one backend ==\n\
     Each re-runs the goldens above whose condition that backend's build meets, and only there: run\n\
     a backend's list on a box with OCANNL_BACKEND set to it. Which backend a value means is what\n\
     test/config/config_reads.ml -- the code the reader writing the file runs -- writes for it.\n";
  let conditional =
    List.concat_map (Map.data golden_routes) ~f:(fun routes ->
        List.concat_map routes ~f:(fun (r : Rerun.route) ->
            match Rerun.selection r with
            | Rerun.Only meant -> List.map meant ~f:(fun backend -> (backend, r.Rerun.alias))
            | Rerun.Every_backend | Rerun.Unnamed _ -> []))
  in
  List.iter Rerun.backends ~f:(fun backend ->
      printf "%s:\n" backend;
      match
        List.filter_map conditional ~f:(fun (b, alias) ->
            if String.equal b backend then Some alias else None)
        |> List.dedup_and_sort ~compare:String.compare
      with
      | [] -> printf "  (none)\n"
      | aliases -> List.iter aliases ~f:(printf "  %s\n"));
  printf
    "\n\
     == tests compiling or executing kernel text they built themselves ==\n\
     They pin nothing, and a codegen change breaks them all the same (gh-ocannl-1685): a staged\n\
     callback's text is emitted verbatim and renamed by nothing, a [?prelowered] lowering is an IR\n\
     shape the pipeline never emits, a shipped kernel compiled alone loses the batch around it. Run\n\
     them on every GPU backend. The rules are stated in test/support/kernel_exec_scan.ml.\n";
  printf "support values reaching such text, derived from the sources:\n";
  Map.iteri kernel_values ~f:(fun ~key ~data ->
      printf "%s [%s]\n" key (String.concat ~sep:"; " (List.map data ~f:Exec.route_name)));
  printf "\nroots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:exec_floors exec_paths));
  List.iter kernel_exec ~f:(fun (name, routes, runs) ->
      printf "%s [%s]\n" name
        (String.concat ~sep:"; "
           (List.map routes ~f:Exec.route_name
           @ [ "run by " ^ String.concat ~sep:", " (List.map runs ~f:Rerun.render) ])));
  printf
    "\n\
     == the aliases a codegen change runs on every GPU backend ==\n\
     # This section lists the kernel-execution routes only, not the whole acceptance set. On each\n\
     # GPU backend, also run the applicable golden routes from the earlier sections \"the aliases a\n\
     # codegen change runs\" and \"of those, the aliases re-running a golden only on one backend\".\n\
     # For example, @test/operations/runtest-micrograd_demo_logging is required by those earlier\n\
     # sections but is absent from this kernel-execution list.\n";
  List.concat_map kernel_exec ~f:(fun (_, _, runs) -> List.map runs ~f:(fun r -> r.Rerun.alias))
  |> List.dedup_and_sort ~compare:String.compare
  |> List.iter ~f:print_endline;
  printf
    "\n\
     == tests calling a backend's compiler with source of their own ==\n\
     They pin nothing and build no kernel through the pipeline, and a change to a backend's\n\
     builtins prelude, its compiler flags or how its compiler is resolved breaks them all the same\n\
     (gh-ocannl-1772): test_shared_builtins compiles Builtins_cc.source and a probe of its own with\n\
     Cc_backend.compiler_command. Run a backend's list, at the end, on a box with OCANNL_BACKEND set\n\
     to it. The rules are stated in test/support/kernel_exec_scan.ml.\n";
  printf "compiler entries, each still spelled by a source handed over:\n";
  List.iter Exec.compiler_entries ~f:(fun e ->
      printf "%s [%s]\n" (Exec.entry_name e) e.Exec.entry_backend);
  printf "\nsupport values reaching them, derived from the sources:\n";
  Map.iteri compiling_values ~f:(fun ~key ~data ->
      printf "%s [%s]\n" key (String.concat ~sep:"; " (List.map data ~f:Exec.route_name)));
  printf "\nroots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:compile_floors compile_paths));
  List.iter compiling ~f:(fun (name, routes, runs) ->
      printf "%s [%s]\n" name
        (String.concat ~sep:"; "
           (List.map routes ~f:Exec.route_name
           @ [ "run by " ^ String.concat ~sep:", " (List.map runs ~f:Rerun.render) ])));
  printf "\n== of those, the aliases by the backend whose compiler they call ==\n";
  List.iter
    (List.dedup_and_sort ~compare:String.compare
       (List.map Exec.compiler_entries ~f:(fun e -> e.Exec.entry_backend)))
    ~f:(fun backend ->
      printf "%s:\n" backend;
      match
        List.concat_map compiling ~f:(fun (_, routes, runs) ->
            if List.mem (backends_called routes) backend ~equal:String.equal then
              List.map runs ~f:(fun r -> r.Rerun.alias)
            else [])
        |> List.dedup_and_sort ~compare:String.compare
      with
      | [] -> printf "  (none)\n"
      | aliases -> List.iter aliases ~f:(printf "  %s\n"));
  printf "\n";
  let exec_violations =
    Floors.violations ~floors:exec_floors ~noun:"test"
      ~floors_name:"codegen_text_inventory.exec_floors" exec_paths
  in
  let compile_violations =
    Floors.violations ~floors:compile_floors ~noun:"test"
      ~floors_name:"codegen_text_inventory.compile_floors" compile_paths
  in
  let helper_violations =
    Floors.violations ~floors:helper_floors ~noun:"helper"
      ~floors_name:"codegen_text_inventory.helper_floors" helper_paths
  in
  let golden_violations =
    Floors.violations ~floors:golden_floors ~noun:"golden"
      ~floors_name:"codegen_text_inventory.golden_floors" golden_paths
  in
  let source_violations =
    Floors.violations ~floors:source_floors ~noun:"site"
      ~floors_name:"codegen_text_inventory.source_floors" site_paths
  in
  (* The derivation's own tripwire. A scan handed no interfaces derives no emitters, drops every
     in-memory site from the census and says so cheerfully -- the silent direction again. So the
     relationship is pinned instead of the result: a library's wrapper interface DECLARES its
     modules (it is a list of aliases, one per module), and every declared module must be one whose
     own interface this run actually read. Neither list is written here; a difference fails, with
     both on stderr. *)
  let declared =
    List.concat_map frontier.Emitter_frontier.interfaces ~f:(fun i ->
        List.map i.Emitter_frontier.declared ~f:(fun m -> i.Emitter_frontier.library ^ ": " ^ m))
    |> List.sort ~compare:String.compare
  in
  let read_interfaces =
    List.concat_map frontier.Emitter_frontier.interfaces ~f:(fun i ->
        List.map i.Emitter_frontier.read ~f:(fun m -> i.Emitter_frontier.library ^ ": " ^ m))
    |> List.sort ~compare:String.compare
  in
  eprintf "Modules declared by the wrapper interfaces: %s\n" (String.concat ~sep:" " declared);
  eprintf "Modules whose interface was read: %s\n" (String.concat ~sep:" " read_interfaces);
  List.iter frontier.Emitter_frontier.interfaces ~f:(fun i ->
      List.iter i.Emitter_frontier.missing ~f:(fun m ->
          Verdict.fail
            (Printf.sprintf
               "%s declares module %s, whose interface was not handed over and is not beside its \
                wrapper -- the emitter frontier is derived from these, so a census taken without \
                it is short by however many renderers it exports"
               i.Emitter_frontier.library m)));
  List.iter
    (golden_violations @ source_violations @ exec_violations @ compile_violations
   @ helper_violations)
    ~f:Verdict.fail;
  Verdict.p_empty "every scanned root meets its golden floor" ~over:golden_paths golden_violations;
  Verdict.p_empty "every scanned root meets its source-site floor" ~over:site_paths
    source_violations;
  Verdict.p_empty "every scanned root meets its floor of tests building their own kernel text"
    ~over:exec_paths exec_violations;
  Verdict.p_empty "every scanned root meets its floor of tests calling a backend's compiler"
    ~over:compile_paths compile_violations;
  Verdict.p_empty "every scanned root meets its floor of support helpers pinning emitted text"
    ~over:helper_paths helper_violations;
  Verdict.p_empty "every compiler entry is still spelled by some source handed over"
    ~over:(List.map Exec.compiler_entries ~f:Exec.entry_name)
    (List.map stale_entry_list ~f:Exec.entry_name);
  Verdict.p_all "every compiler entry names a backend the configuration reader writes"
    Exec.compiler_entries ~f:(fun e ->
      List.mem Rerun.backends e.Exec.entry_backend ~equal:String.equal);
  Verdict.p_empty "every source handed over parsed as OCaml" ~over:source_files !unparsed;
  Verdict.p_empty "every condition a golden is re-run under names the backend whose build meets it"
    ~over:(Map.data golden_routes |> List.concat)
    unselected;
  Verdict.p_empty "every member has an alias that re-runs it"
    ~over:
      (golden_paths
      @ List.map sites ~f:(fun s -> s.Scan.site_path)
      @ List.map executing ~f:(fun (name, _, _) -> name))
    (unplaced @ exec_unplaced);
  Verdict.p_empty "the scan raised on no source handed over" ~over:source_files !raised;
  Verdict.p_empty "every exclusion still names a file the globs hand over" ~over:handed_over stale;
  Verdict.p "every module the scanned library interfaces declare was read"
    (List.equal String.equal declared read_interfaces);
  Verdict.p_all "every scanned library declares modules" frontier.Emitter_frontier.interfaces
    ~f:(fun i -> not (List.is_empty i.Emitter_frontier.declared));
  Verdict.p_empty "no source hides a route to generated text behind an open" ~over:source_files
    !rejected;
  Verdict.p_none
    "no test source reads build_files/ itself: generated code is read through Test_utils.Generated"
    sites ~f:(fun site -> site.Scan.direct);
  Verdict.p_none
    "no test source reads a kernel's statements after its Main logic banner unexempted: they are \
     read through Test_utils.Generated.main_logic"
    sites ~f:(fun site ->
      site.Scan.after_marker
      && not (List.Assoc.mem after_marker_exempt site.Scan.site_path ~equal:String.equal));
  Verdict.p_empty "every after-marker exemption still names a file reading after the banner"
    ~over:(List.map after_marker_exempt ~f:fst)
    (List.map stale_after_marker ~f:fst);
  Test_scan.Refusal_control_manifest.print "codegen_text_inventory.ml"
