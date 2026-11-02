(** How the claim-shape reader behind [verdict_ratchet] reads a format and a source, exercised on
    input built to break it rather than on whatever the test directories happen to contain today.

    [Verdict_scan] decides what counts as a self-decided claim, and both of its mistakes are quiet.
    A shape it stops recognising leaves an empty offender list, which reads exactly like a clean
    tree -- the failure mode gh-ocannl-668 exists to close, arriving in the check meant to close it.
    A shape it recognises too eagerly turns a descriptive print into a demand for an exemption, and
    an exemption list long enough stops being read.

    So the two halves are pinned separately: which formats are the claim shape, and which literals
    of a source the walk reaches. *)

open Base
module Scan = Test_utils.Verdict_scan
module Attempt = Test_utils.Scan_attempt

let printf = Stdio.printf

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601) -- the rule this whole check is
   about, and not one to be exempt from. *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

(* Planted for [verdict_ratchet], which names both in its canary list and fails if its scan of the
   test directories does not find them here.

   They are ordinary fixture inputs -- the two spellings below feed the shape cases -- and they are
   also the only thing standing between a walk that went blind and an empty offender list read as a
   clean tree. The second is written over a line continuation on purpose: its decoded value is the
   claim shape while no single line of this file contains it, so a reader that matched text would
   find the first and miss this one. *)
let planted_plain = "planted canary: %b\n"
let planted_continued = "planted canary over a continuation: %b\n"

(* Formats that ARE the claim shape, with the label and kind each yields. *)
let claim_cases =
  [
    ("the plain form", "fused: %b\n", "fused");
    (* The separator vocabulary. Reading only a colon left the whole `= %b` population outside a
       check written for it (gh-ocannl-624). *)
    ("an equals separator", "round-trip identity = %b\n", "round-trip identity");
    ("an equals with no space", "batch-free=%b\n", "batch-free");
    ("an arrow separator", "hoisted -> %b\n", "hoisted");
    ("no trailing newline", "fused: %b", "fused");
    ("a blank line either side", "\nfused: %b\n\n", "fused");
    ("a flush directive consumes no argument", "fused: %b\n%!", "fused");
    ("nor does a separator", "fused: %b\n%,", "fused");
    ("a per cent sign in the label", "100%% covered: %b\n", "100% covered");
    ("a colon inside the label", "kernel k: fused: %b\n", "kernel k: fused");
    ("no space after the colon", "fused:%b\n", "fused");
    ("a tab before the boolean", "fused:\t%b\n", "fused");
    (* The two literals the ratchet looks for, which are fixture inputs first. The continuation is
       what says the reader works on decoded values: its source line ends mid-label. *)
    ("the planted canary", planted_plain, "planted canary");
    ( "the planted canary over a continuation",
      planted_continued,
      "planted canary over a continuation" );
  ]

(* Formats whose label is COMPUTED: the claim shape too, since gh-ocannl-624 gave them an entry
   point ([Verdict.pf] / [Verdict.claimf]). The label is what survives rendering a head this reader
   cannot fill in, which is why an exemption for one of these is keyed by the whole format. *)
let computed_cases =
  [
    ("a leading string argument", "%s fused: %b\n", "fused");
    ( "an interpolated measurement",
      "Epoch %d, loss below threshold=%b\n",
      "Epoch , loss below threshold" );
    ("arguments on both sides of the label", "%s %s parallelizable: %b\n", "parallelizable");
    (* The wrapper the pre-`Verdict` tests defined for themselves, and the exact shape this ratchet
       exists to keep from regrowing: the whole label is the argument, so the rendered residual is
       empty and the verbatim head has to speak for it. Reading that empty residual as "no label"
       would have let the one form that matters straight back in (Codex P2, round 1). *)
    ("a label built entirely from arguments", "%s: %b\n", "%s");
    ("the same with a width", "%-22s: %b\n", "%-22s");
    ("a numeric label", "%d: %b\n", "%d");
    (* A census row of several booleans is this shape too, and that is the intent: it is the row
       shape that needs an exemption, precisely because a reader cannot tell it from a verdict. *)
    ("a second boolean before the last", "fused: %b %b\n", "fused");
  ]

(* Formats that are NOT, each for its own reason. What they have in common is that the line does not
   END on a bare boolean behind a separator, which is what a verdict looks like; a print that
   describes rather than decides almost always fails one of these. *)
let non_claim_cases =
  [
    ("an interpolated value after the boolean", "fused: %b (%d blocks)\n");
    ("an annotation after the boolean", "fused: %b (expect false)\n");
    ("a width means the print is laying out a column", "fused: %6b\n");
    ("so does a left-justifying flag", "fused: %-6b\n");
    ("no colon, so no label", "fused? %b\n");
    (* Empty residual AND nothing to build one from: a literal label has to be non-empty, which is
       what keeps this out while `"%s: %b\n"` is in. *)
    ("no label before the colon, and no argument to make one", ": %b\n");
    ("a bare boolean", "%b\n");
    ("nothing before it but a blank line", "\n%b\n");
    ("not a boolean at all", "fused: %d\n");
    ("a boolean that is not the last conversion", "fused: %b, blocks: %d\n");
    ("prose after the boolean", "fused: %b -- and the reason\n");
    (* A per cent with nothing after it is not a directive, and must not run the scan off the end of
       the format either. *)
    ("a trailing per cent neither crashes the scan nor vanishes", "fused: %b\n%");
    (* Nested formats are counted as consuming, which takes them out of the shape rather than into
       it: erring that way costs a report that was never made, never a false one. *)
    ("a nested format is not read as a bare boolean", "fused: %b %{%d%}\n");
  ]

(* Whole sources, with the labels the walk must report and where it says each one sits. A source
   case exists where the answer depends on the SOURCE and not on the format: where the literal sits,
   how it is spelled, whether it is prose. *)
let source_cases =
  [
    ( "a claim applied to a printing function names it",
      {ocaml|let () = Stdio.printf "fused: %b\n" true|ocaml},
      [ ("fused", Some "Stdio.printf") ] );
    ( "the receiver is reported as written",
      {ocaml|open Stdio
let () = eprintf "fused: %b\n" true|ocaml},
      [ ("fused", Some "eprintf") ] );
    (* The wrapper is why the check is about the literal and not about the call: a `let p name b =
       printf "%s: %b\n" name b` helper is what these sites looked like before Verdict existed, and
       pointing the scan at printing functions alone would miss every claim routed through one. *)
    ( "a claim handed to a wrapper is still a claim",
      {ocaml|let () = report "fused: %b\n" true|ocaml},
      [ ("fused", Some "report") ] );
    ( "a claim bound to a name, applied to nothing",
      {ocaml|let message = "fused: %b\n"|ocaml},
      [ ("fused", None) ] );
    ( "a claim inside a list is reached",
      {ocaml|let messages = [ "fused: %b\n"; "tiled: %b\n" ]|ocaml},
      [ ("fused", None); ("tiled", None) ] );
    (* Delimiters and escapes are the parser's business, not the scan's: what is compared is the
       decoded value. *)
    ( "a quoted-string literal decodes the same way",
      {ocaml|let () = printf {xx|fused: %b|xx} true|ocaml},
      [ ("fused", Some "printf") ] );
    ( "a line continuation joins the label",
      "let () = printf \"fused over a \\\n   continuation: %b\\n\" true",
      [ ("fused over a continuation", Some "printf") ] );
    (* Prose is not a print. An ordinary comment the parser discards outright; a documentation
       comment survives as an attribute holding a string, which is why attribute payloads are
       skipped rather than merely hoped about. *)
    ("an ordinary comment quoting the shape", {ocaml|(* fused: %b *) let x = 1|ocaml}, []);
    ("a documentation comment quoting the shape", {ocaml|(** fused: %b *) let x = 1|ocaml}, []);
    ( "an attached documentation comment quoting the shape",
      {ocaml|let x = 1 (** fused: %b *)|ocaml},
      [] );
    (* An extension payload is code, or the golden text of some, so it is NOT skipped. *)
    ( "a claim inside an extension payload is reached",
      {ocaml|let () = [%probe let () = printf "fused: %b\n" true]|ocaml},
      [ ("fused", Some "printf") ] );
    ( "a claim quoted inside a string is not a claim",
      {ocaml|let doc = "write printf \"fused: %b\\n\" here"|ocaml},
      [] );
  ]

(* gh-ocannl-1207: verdict_ratchet's own refusals, executed rather than vouched for by the cases
   below, which exercise the reader it rests on in process. The ratchet runs as a child over a
   corpus built to trip them -- a source printing a claim it decided itself, and one that does not
   parse -- in which every exemption, every exempted fixture source and every planted canary is
   missing too, and once over no source at all. What each child refused is what
   `raw_direct_evidence` attributes to the two claims. *)
let refusal_corpus =
  [
    ("test/self_decided.ml", "let () = Stdio.printf \"decided: %b\\n\" true\n");
    ("test/unparsable.ml", "let =\n");
  ]

let scanner_refusal_controls exe =
  let source = "test/operations/verdict_ratchet.ml" in
  let refuses = Test_utils.Refusal_control_manifest.refuses ~source ~exe in
  Test_utils.Refusal_control_manifest.with_tree refusal_corpus (fun root ->
      Verdict.p "verdict_ratchet refuses a corpus built to trip its refusals, exiting 1"
        (refuses ~claim:"verdict_ratchet refuses a corpus built to trip its refusals, exiting 1"
           (root :: List.map refusal_corpus ~f:(fun (path, _) -> Stdlib.Filename.concat root path)));
      Verdict.p "verdict_ratchet refuses a run handed no source, exiting 1"
        (refuses ~claim:"verdict_ratchet refuses a run handed no source, exiting 1" [ root ]))

let show_printer = Option.value ~default:"<unapplied>"

let () =
  let check_kind ~what ~expected (name, format, label) =
    match Scan.claim_of format with
    | Some (found, kind) when String.equal found label ->
        if Poly.equal kind expected then printf "ok: %s -- %s\n" what name
        else fail "%s -- %s: read the right label with the wrong kind" what name
    | Some (found, _) -> fail "%s -- %s: expected label %S, found %S" what name label found
    | None -> fail "%s -- %s: expected label %S, found no claim" what name label
  in
  List.iter claim_cases ~f:(check_kind ~what:"claim shape" ~expected:Scan.Literal_label);
  List.iter computed_cases
    ~f:(check_kind ~what:"computed claim shape" ~expected:Scan.Computed_label);
  (* The head is what an exemption is keyed by, so it has to be the format VERBATIM up to the
     boolean -- conversions unrendered, and stopping before the `%b` so that writing one out is not
     itself a claim. *)
  List.iter
    [
      ("conversions are left unrendered", "%s fused: %b\n", "%s fused: ");
      ("a width is kept", "%-22s option: %b\n", "%-22s option: ");
      ("an earlier boolean is kept", "fused: %b tiled: %b\n", "fused: %b tiled: ");
      ("a wholly computed label keeps its conversion", "%s: %b\n", "%s: ");
    ]
    ~f:(fun (name, format, expected) ->
      match Scan.claim_site format with
      | Some (_, _, head) when String.equal head expected -> printf "ok: claim head -- %s\n" name
      | Some (_, _, head) -> fail "claim head -- %s: expected %S, found %S" name expected head
      | None -> fail "claim head -- %s: expected %S, found no claim" name expected);
  List.iter non_claim_cases ~f:(fun (name, format) ->
      match Scan.claim_label format with
      | None -> printf "ok: not a claim -- %s\n" name
      | Some found -> fail "not a claim -- %s: read it as the claim %S" name found);
  List.iter source_cases ~f:(fun (name, source, expected) ->
      let found =
        Attempt.attempted ~what:"source" ~name ~default:[] (fun () ->
            List.map (Scan.scan source).Scan.sites ~f:(fun site ->
                (site.Scan.label, site.Scan.printer)))
      in
      let render sites =
        List.map sites ~f:(fun (label, printer) -> label ^ " (" ^ show_printer printer ^ ")")
        |> String.concat ~sep:"; "
      in
      if
        List.equal
          (fun (a, b) (c, d) -> String.equal a c && Option.equal String.equal b d)
          found expected
      then printf "ok: source -- %s\n" name
      else fail "source -- %s: expected [%s], found [%s]" name (render expected) (render found));
  List.iter
    [
      ( "a parameter does not borrow an outer label",
        {ocaml|let label = "same"
let f label = pass_fail label ok
let () = skipped "same"|ocaml},
        0 );
      ( "same-named local literals pair with their own text",
        {ocaml|let f () = let label = "first" in pass_fail label ok
let g () = let label = "second" in skipped label
let () = skipped "first"|ocaml},
        1 );
      ( "a later literal does not resolve an earlier use",
        {ocaml|let f () = pass_fail label ok
let label = "later"
let () = skipped "later"|ocaml},
        0 );
      ( "a case binder shadows a literal",
        {ocaml|let label = "outer"
let f x = match x with label -> pass_fail label ok
let () = skipped "outer"|ocaml},
        0 );
      ( "same-named parameters in one item remain separate",
        {ocaml|let pair = ((fun label -> pass_fail label ok), (fun label -> skipped label))|ocaml},
        0 );
      ( "computed labels preserve their binding identity",
        {ocaml|let pair = ((fun label -> pass_fail (label ^ " holds") ok), (fun label -> skipped (label ^ " holds")))|ocaml},
        0 );
      ( "computed labels pair when shadowing inputs resolve to the same literal",
        {ocaml|let f () = let suffix = "same" in pass_fail ("x" ^ suffix) ok;
let suffix = "same" in skipped ("x" ^ suffix)|ocaml},
        1 );
      ( "computed labels do not pair when literal inputs differ",
        {ocaml|let f () = let suffix = "first" in pass_fail ("x" ^ suffix) ok;
let suffix = "second" in skipped ("x" ^ suffix)|ocaml},
        0 );
      ( "bindings inside a computed expression hide outer names",
        {ocaml|let f label =
pass_fail (let label = "same" in label) ok;
let label = other in skipped (let label = "same" in label)|ocaml},
        1 );
      ( "the same literal still pairs across scopes",
        {ocaml|let label = "same"
let f () = pass_fail label ok
let () = skipped "same"|ocaml},
        1 );
    ]
    ~f:(fun (name, source, expected) ->
      let found = List.length (Scan.dialect_census source).Scan.pairings in
      if found = expected then printf "ok: dialect scope -- %s\n" name
      else fail "dialect scope -- %s: expected %d pairings, found %d" name expected found);
  (* gh-ocannl-1794: a hand-written skip list -- a skip handed to an iteration, a callback or loop
     that only skips, a branch running two skips -- is found; the leg form and a gated claim in a
     loop are not. Each finding prints as its kind and its first skip's label. *)
  List.iter
    [
      ( "an iteration handed a skip",
        {ocaml|let () = if on_gpu then check () else List.iter labels ~f:skipped|ocaml},
        [ "iterated skipped" ] );
      ( "an iteration handed a partially applied Verdict skip",
        {ocaml|let () = List.iter labels ~f:(Verdict.skipped ~backend:backend_name)|ocaml},
        [ "iterated Verdict.skipped ~backend:backend_name" ] );
      ( "a callback that only skips",
        {ocaml|let () = Array.iter names ~f:(fun l -> skipped (tag ^ l))|ocaml},
        [ "iterated tag ^ l" ] );
      ( "a loop body that only skips",
        {ocaml|let () = for i = 0 to 2 do Verdict.skipped ~backend:"cc" names.(i) done|ocaml},
        [ "iterated names.(i)" ] );
      ( "a branch running two literal skips",
        {ocaml|let () =
  if on_cpu then (p "a" x; p "b" y) else (skipped "a"; skipped "b")|ocaml},
        [ "run 2 a" ] );
      ( "a match case running two skips through a per-test partial skip",
        {ocaml|let skipped = Verdict.skipped ~backend:backend_name
let () = match cap with `Withheld -> skipped first; skipped second | `Advertised -> run ()|ocaml},
        [ "run 2 first" ] );
      ( "claims written once in a leg",
        {ocaml|let () =
  let leg = Verdict.leg ~when_:on_cpu ~on:backend_name () in
  in_leg ~leg "a" (fun label -> p label x);
  in_leg ~leg "b" (fun label -> p label y)|ocaml},
        [] );
      ( "a loop whose callback evaluates on one side of its gate and skips on the other",
        {ocaml|let () =
  List.iter cases ~f:(fun c -> if on_gpu then p (case_label c) (passes c) else skipped (case_label c))|ocaml},
        [] );
      ("one skip per branch", {ocaml|let () = if on_gpu then p "a" x else skipped "a"|ocaml}, []);
      ( "a local function that only shares the name",
        {ocaml|let () =
  let skipped s = s > 2 in
  if a then (ignore (skipped 1); ignore (skipped 2)) else ();
  List.iter xs ~f:(fun x -> ignore (skipped x))|ocaml},
        [] );
      ( "a parameter that only shares the name",
        {ocaml|let () = ignore (List.filter_map skips ~f:(fun skipped -> pair skipped))|ocaml},
        [] );
      ( "a nested branch, judged on its own",
        {ocaml|let () = if a then (skipped "x"; if b then skipped "y" else p "y" ok) else p "x" ok|ocaml},
        [] );
      ( "another module's skipped",
        {ocaml|module Other = struct let skipped x = ignore x end
let () = List.iter xs ~f:Other.skipped|ocaml},
        [] );
      ( "a skip alias a local open shadows with another skipped",
        {ocaml|module Local = struct let skipped _ = () end
let skipped = Verdict.skipped ~backend:"cc"
let () = let open Local in List.iter labels ~f:skipped|ocaml},
        [] );
      ( "a skip alias past an open of a standard library",
        {ocaml|let skipped = Verdict.skipped ~backend:"cc"
let () = let open Stdio in List.iter labels ~f:skipped|ocaml},
        [ "iterated skipped" ] );
      ( "a skip callback a collection stores rather than calls",
        {ocaml|let checks = List.singleton (fun () -> Verdict.skipped ~backend:"cc" "one")|ocaml},
        [] );
      ( "an iteration through a module alias of a collection",
        {ocaml|module L = List
let () = L.iter labels ~f:(Verdict.skipped ~backend:backend_name)|ocaml},
        [ "iterated Verdict.skipped ~backend:backend_name" ] );
      ( "a source's own List module, which only stores its callback",
        {ocaml|module List = struct let iter xs ~f = ignore (xs, f) end
let () = List.iter labels ~f:skipped|ocaml},
        [] );
      ( "a function value a loop body only builds",
        {ocaml|let () = for _i = 1 to 2 do ignore (fun () -> Verdict.skipped ~backend:"cc" "one") done|ocaml},
        [] );
      ( "a partial iteration, which stores its skip callback",
        {ocaml|let skip_labels = List.iter ~f:(Verdict.skipped ~backend:"cc")|ocaml},
        [] );
      ( "an iteration written as a pipeline",
        {ocaml|let () = labels |> List.iter ~f:(Verdict.skipped ~backend:backend_name)|ocaml},
        [ "iterated Verdict.skipped ~backend:backend_name" ] );
      ( "a lazy a loop body only builds",
        {ocaml|let () = for _i = 1 to 2 do ignore (lazy (Verdict.skipped ~backend:"cc" "one")) done|ocaml},
        [] );
      ( "an Option iteration, which runs at most once",
        {ocaml|let () = Option.iter maybe_label ~f:skipped|ocaml},
        [] );
    ]
    ~f:(fun (name, source, expected) ->
      let found =
        Attempt.attempted ~what:"skip-list source" ~name ~default:[ "<raised>" ] (fun () ->
            List.map (Scan.skip_list_census source).Scan.skip_lists ~f:(fun site ->
                (match site.Scan.skip_kind with
                  | Scan.Iterated -> "iterated "
                  | Scan.Skip_run n -> Printf.sprintf "run %d " n)
                ^ site.Scan.first_label))
      in
      if List.equal String.equal found expected then printf "ok: skip list -- %s\n" name
      else
        fail "skip list -- %s: expected [%s], found [%s]" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found));
  (* The census the ratchet leans on to tell "nothing to report" from "nothing read": on a source
     with literals it must count them, and it must place the ones an application receives. *)
  let counted = Scan.scan {ocaml|let unapplied = "c"
let () = printf "a" "b"|ocaml} in
  Verdict.p "the walk counts every string literal it passes" (counted.Scan.literals = 3);
  Verdict.p "and places the ones a named function receives" (counted.Scan.applied_literals = 2);
  (match Stdlib.Sys.argv with
  | [| _; exe |] -> scanner_refusal_controls exe
  | _ -> Verdict.fail "usage: verdict_scan_cases <verdict_ratchet.exe>");
  Test_utils.Refusal_control_manifest.print "verdict_ratchet.ml"
