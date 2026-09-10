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

    The rules are decided in {!Test_utils.Codegen_text_scan} as pure functions over a path and its
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
    and a difference fails the run with both lists on stderr. *)

open Base
open Stdio

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv
let printf = Test_utils.Refusal_control_manifest.printf

module Scan = Test_utils.Codegen_text_scan
module Floors = Test_utils.Scan_floors

(** Lower bounds per scanned root, well below the census of the day they were written. See
    {!Test_utils.Scan_floors}: a glob that breaks goes to zero, a member added moves nothing. *)
let golden_floors = [ ("test", 12); ("arrayjit/test", 2) ]

(** [arrayjit/test] has its own floor, and the reason it once looked as though it could not have
    sites is worth keeping: [test_utils] is a [neural_nets_lib] library, so the arrayjit tests
    cannot link {!Test_utils.Generated} at all. They reach generated text the third way instead --
    calling [C_syntax.compile_proc] and rendering the document -- which is precisely the route the
    first version of this scan did not model, and which made a whole root look empty (Codex P2,
    round 2). *)
let source_floors = [ ("test", 20); ("arrayjit/test", 2) ]

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

let read path = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all

module Dune = Test_utils.Dune_stanza_scan

let stale_exclusions ~excluded ~handed_over =
  List.filter excluded ~f:(fun (path, _) -> not (List.mem handed_over path ~equal:String.equal))

let refuse_stale_exclusions ~fail stale =
  List.iter stale ~f:(fun (path, reason) ->
      fail
        (Printf.sprintf
           "the exclusion for %s (%s) names a file the globs no longer hand over -- drop it, or \
            fix the path it was meant to name"
           path reason))

let refusal_control () =
  let source = "test/operations/codegen_text_inventory.ml" in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source
      ~format:
        "the exclusion for %s (%s) names a file the globs no longer hand over -- drop it, or fix \
         the path it was meant to name"
  in
  let stale = stale_exclusions ~excluded:[ ("gone.expected", "fixture") ] ~handed_over:[] in
  refuse_stale_exclusions ~fail stale;
  Verdict.p "an exclusion absent from the hand-over reaches the stale-exclusion refusal"
    (!refused && List.length stale = 1);
  Test_utils.Refusal_control_manifest.print source

let () =
  if Array.length argv = 2 && String.equal argv.(1) "--refusal-control" then (
    refusal_control ();
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
  let all_ml = of_suffix ".ml" in
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
  let rejected = ref [] in
  let sites =
    List.filter_map source_files ~f:(fun (name, on_disk) ->
        if is_excluded name then None
        else
          let contents = read on_disk in
          try
            rejected := Scan.rejections ~emitters ~path:name ~contents @ !rejected;
            Scan.classify_source ~emitters ~path:name ~contents
          with _ ->
            unparsed := name :: !unparsed;
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
  List.iter !unparsed ~f:(fun path ->
      Verdict.fail
        (Printf.sprintf
           "%s does not parse as OCaml, so the scan cannot say whether it pins emitted text" path));
  List.iter (List.sort !rejected ~compare:String.compare) ~f:Verdict.fail;
  let golden_paths = List.map goldens ~f:(fun g -> g.Scan.path) in
  let site_paths = List.map sites ~f:(fun s -> s.Scan.site_path) in
  Floors.report ~floors:golden_floors ~noun:"golden" ~what:"Goldens holding emitted text"
    golden_paths;
  Floors.report ~floors:source_floors ~noun:"site" ~what:"Sources pinning emitted text" site_paths;
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
      printf "%s [%s] %s\n" g.Scan.path
        (String.concat ~sep:" " g.Scan.families)
        (String.concat ~sep:"; " origin));
  printf "\n== test sources pinning emitted text ==\n";
  printf "roots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:source_floors site_paths));
  List.iter
    (List.sort sites ~compare:(fun a b -> String.compare a.Scan.site_path b.Scan.site_path))
    ~f:(fun s ->
      let flags =
        (if s.Scan.direct then [ "reads build_files/ directly" ] else [])
        @ (if s.Scan.rendered then [ "renders generated text in memory" ] else [])
        @ if s.Scan.partial then [ "also pins text this scan cannot name" ] else []
      in
      printf "%s%s\n" s.Scan.site_path
        (match flags with [] -> "" | flags -> " [" ^ String.concat ~sep:"; " flags ^ "]");
      List.iter s.Scan.pins ~f:(fun pin -> printf "    %s\n" pin));
  printf "\n";
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
  List.iter (golden_violations @ source_violations) ~f:Verdict.fail;
  Verdict.p_empty "every scanned root meets its golden floor" ~over:golden_paths golden_violations;
  Verdict.p_empty "every scanned root meets its source-site floor" ~over:site_paths
    source_violations;
  Verdict.p_empty "every source handed over parsed as OCaml" ~over:source_files !unparsed;
  Verdict.p_empty "every exclusion still names a file the globs hand over" ~over:handed_over stale;
  Verdict.p "every module the scanned library interfaces declare was read"
    (List.equal String.equal declared read_interfaces);
  Verdict.p_all "every scanned library declares modules" frontier.Emitter_frontier.interfaces
    ~f:(fun i -> not (List.is_empty i.Emitter_frontier.declared));
  Verdict.p_empty "no source hides a route to generated text behind an open" ~over:source_files
    !rejected;
  Test_utils.Refusal_control_manifest.print "codegen_text_inventory.ml"
