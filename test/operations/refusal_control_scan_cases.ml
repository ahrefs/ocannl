(** The source reader and relationship behind gh-ocannl-800, on diagnostics and goldens the
    repository does not have to contain. *)

open Base
open Stdio
module Scan = Test_utils.Refusal_control_scan
module Manifest = Test_utils.Refusal_control_manifest

let minus = Manifest.minus

(* gh-ocannl-1088: the stanza's argument list decides which goldens answer for a source, but
   [Manifest.sources] owns which sources there are. The catalogue is compared as a sorted multiset
   against the manifest's distinct sources, so a source the argument list lacks, one it adds, and
   one it names twice are each a mismatch -- and a manifest row repeated is one of its own, which a
   catalogue repeating it too would otherwise match. The lists: the manifest's uncatalogued sources,
   the catalogue's sources outside the manifest or beyond its first, and the manifest's repeats. *)
let catalogue_mismatch ~manifest ~catalogued =
  let distinct = List.dedup_and_sort manifest ~compare:String.compare
  and catalogued = List.sort catalogued ~compare:String.compare in
  ( minus distinct catalogued,
    minus catalogued distinct,
    minus (List.sort manifest ~compare:String.compare) distinct )

let () =
  let source =
    {ocaml|
let fail = Verdict.fail

let direct () =
  Verdict.fail "a direct scanner refusal has a permanent diagnostic"

let formatted name =
  fail
    (Printf.sprintf
       "%s: the formatted refusal names the `formatted_relationship` its control exercises"
       name)

let applied name =
  fail @@ Printf.sprintf "%s is stale -- drop it from the exemption list" name

let quantified xs =
  Verdict.p_all "every refusal control remains related to its diagnostic" xs ~f:Fn.id

let formatted_claim name ok = Verdict.pf "%s refusal stays live" name ok

let paired got want = Verdict.p_all2 "paired" got want ~f:Int.equal

let concise ok = Verdict.p "valid" ok
let concise_fail () = Verdict.fail "bad key"
let exception_fail () = failwith "stopped"

let prose = "Verdict.fail \"quoted code is not an application\""
let dynamic reason = Verdict.fail reason
|ocaml}
  in
  let diagnostics = Scan.diagnostics source in
  let fragments = List.map diagnostics ~f:(fun diagnostic -> diagnostic.Scan.fragment) in
  Verdict.p_all ~min:9
    "direct, `failwith`, formatted, `@@`, quantified, `p_all2`, `pf`, and short refusal formats \
     are extracted"
    diagnostics ~f:(fun diagnostic -> not (String.is_empty diagnostic.Scan.fragment));
  Verdict.p "comments, quoted code, and a dynamic value contribute no diagnostic string constant"
    (List.length diagnostics = 9);
  Verdict.p "Printf substitutions are holes and a stable literal fragment becomes the fragment"
    (List.mem fragments "formatted_relationship" ~equal:String.equal);
  let one = List.hd_exn diagnostics in
  Verdict.p "an absent fragment is an orphan"
    (List.length (Scan.orphans ~control_text:"" [ one ]) = 1);
  Verdict.p_empty "the diagnostic's unique marker in a control golden covers it" ~over:[ one ]
    (Scan.orphans ~control_text:(Scan.marker one) [ one ]);
  let colliding_fragment =
    { one with Scan.format = one.format ^ " elsewhere"; identity = "other" }
  in
  Verdict.p "two diagnostics sharing a display fragment still require distinct controls"
    (List.length (Scan.orphans ~control_text:(Scan.marker one) [ one; colliding_fragment ]) = 1);
  Verdict.p "one control marker occurrence covers only one identical diagnostic"
    (List.length (Scan.orphans ~control_text:(Scan.marker one) [ one; one ]) = 1);
  let valid =
    List.find_exn diagnostics ~f:(fun diagnostic -> String.equal diagnostic.Scan.format "valid")
  in
  Verdict.p "one observed claim execution is consumed by only one matching diagnostic"
    (Option.value_map (Manifest.claim_exercises [ "valid" ] valid) ~default:false ~f:List.is_empty);
  Verdict.p "a second identical diagnostic cannot reuse the consumed claim execution"
    (Option.is_none
       (Option.bind (Manifest.claim_exercises [ "valid" ] valid) ~f:(fun remaining ->
            Manifest.claim_exercises remaining valid)));
  (* The stale-row report a scan's own [Manifest.print] writes on stderr: the difference it names
     and the direct failures it asks evidence for, over a row the synthetic source has outgrown. *)
  let extracted = List.map diagnostics ~f:Scan.marker in
  let first = List.hd_exn extracted and second = List.nth_exn extracted 1 in
  let reworded = "[scanner-refusal:00000000000000000000000000000000] reworded" in
  Verdict.p "a row holding the extraction in order has no difference"
    (Option.is_none (Manifest.row_difference ~registered:extracted ~extracted));
  Verdict.p "a reworded format is one marker added and one removed, with no order to report"
    (match Manifest.row_difference ~registered:(reworded :: List.tl_exn extracted) ~extracted with
    | Some { absent; no_longer; first_misplaced } ->
        List.equal String.equal absent [ first ]
        && List.equal String.equal no_longer [ reworded ]
        && Option.is_none first_misplaced
    | None -> false);
  Verdict.p "a row holding the same markers in another order names the first misplaced entry"
    (match
       Manifest.row_difference ~registered:(second :: first :: List.drop extracted 2) ~extracted
     with
    | Some { absent = []; no_longer = []; first_misplaced = Some (1, e, r) } ->
        String.equal e first && String.equal r second
    | _ -> false);
  let synthetic = "test/operations/synthetic_scan.ml" in
  let is_kind kind diagnostic = Poly.equal diagnostic.Scan.kind kind in
  let new_fail = List.find_exn diagnostics ~f:(is_kind Scan.Fail)
  and new_claim = List.find_exn diagnostics ~f:(is_kind Scan.Claim) in
  let outgrown =
    List.filter diagnostics ~f:(fun diagnostic ->
        not (phys_equal diagnostic new_fail || phys_equal diagnostic new_claim))
    |> List.map ~f:Scan.marker
  in
  Verdict.p
    "a direct failure new to the row, observed by nothing, is named for a direct-evidence entry; a \
     new claim is not"
    (List.equal String.equal
       (Manifest.unevidenced_failures ~source:synthetic ~registered:outgrown diagnostics)
       [ "synthetic_scan.ml:" ^ new_fail.Scan.identity ]);
  Verdict.p "a second occurrence of a listed direct failure is new to the row, and is named"
    (List.equal String.equal
       (Manifest.unevidenced_failures ~source:synthetic ~registered:extracted
          (diagnostics @ [ new_fail ]))
       [ "synthetic_scan.ml:" ^ new_fail.Scan.identity ]);
  Verdict.p_empty "a direct failure the row already lists asks for no new evidence"
    ~over:(List.filter diagnostics ~f:(is_kind Scan.Fail))
    (Manifest.unevidenced_failures ~source:synthetic ~registered:extracted diagnostics);
  (* gh-ocannl-1207: what counts as executing a direct failure. A control claim mapped to the
     refusal counts only together with the refusal's own line from a refused child run; either alone
     -- an accepted fixture that passed, or a refusal nothing asserted on -- does not, and neither
     does any printed line, which [Manifest.standing] does not even take. *)
  let key = "synthetic_scan.ml:" ^ new_fail.Scan.identity in
  let control = "the malformed fixture is refused" in
  let refusing_line =
    String.substr_replace_all new_fail.Scan.format ~pattern:"%s" ~with_:"fixture.ml"
  in
  let standing ?(diagnostic = new_fail) ?(direct_evidence = [ (key, control ^ ": true") ])
      ?(catalogue_only = []) ?(observed = false) ?(refused = [ refusing_line ])
      ?(rivals = [ new_fail.Scan.format ]) ?(passed_labels = [ control ]) () =
    Manifest.standing ~direct_evidence ~catalogue_only ~observed ~refused ~rivals ~passed_labels
      ~key diagnostic
  in
  Verdict.p "a passed control claim and the refusal's line from a refused run exercise it"
    (Poly.equal (standing ()) Manifest.Exercised);
  Verdict.p "a passed control claim alone, with no refused run printing the refusal, does not"
    (Poly.equal (standing ~refused:[] ()) Manifest.Unexercised);
  Verdict.p "a refused run printing a different refusal does not either"
    (Poly.equal (standing ~refused:[ "an unrelated refusal" ] ()) Manifest.Unexercised);
  Verdict.p "the refusal's line alone, with the mapped claim not passed, does not"
    (Poly.equal (standing ~passed_labels:[] ()) Manifest.Unexercised);
  Verdict.p "an observation from the caught branch exercises it with no mapping"
    (Poly.equal (standing ~direct_evidence:[] ~refused:[] ~observed:true ()) Manifest.Exercised);
  (* A line belongs to the most specific format that matches it: a general refusal does not live on
     the output of a sibling whose format fixes more of the same text. *)
  let general = { new_fail with Scan.format = "keys missing from %s: %s"; identity = "general" }
  and specific =
    { new_fail with Scan.format = "keys missing from the registry: %s"; identity = "specific" }
  in
  let rivals = [ general.Scan.format; specific.Scan.format ] in
  Verdict.p "a line a more specific sibling format also matches does not exercise the general one"
    (Poly.equal
       (standing ~diagnostic:general ~rivals ~refused:[ "keys missing from the registry: k" ] ())
       Manifest.Unexercised);
  Verdict.p "the general refusal's own line still exercises it beside the specific sibling"
    (Poly.equal
       (standing ~diagnostic:general ~rivals ~refused:[ "keys missing from reference: k" ] ())
       Manifest.Exercised);
  Verdict.p "and the specific sibling owns the line both formats match"
    (Poly.equal
       (standing ~diagnostic:specific ~rivals ~refused:[ "keys missing from the registry: k" ] ())
       Manifest.Exercised);
  (* A catalogue-only key never has a mapping -- the audit below refuses that -- so whether it ran
     is read from the lines and the caught branch alone. *)
  let catalogue_only = [ (key, "unreachable") ] in
  Verdict.p "a catalogue-only refusal nothing executes stays catalogue-only"
    (Poly.equal
       (standing ~direct_evidence:[] ~catalogue_only ~refused:[] ())
       Manifest.Catalogue_only);
  Verdict.p "a catalogue-only refusal a refused child run prints is reported as executed"
    (Poly.equal
       (standing ~direct_evidence:[] ~catalogue_only ())
       Manifest.Executed_yet_catalogue_only);
  Verdict.p "and so is one its caught branch observes"
    (Poly.equal
       (standing ~direct_evidence:[] ~catalogue_only ~refused:[] ~observed:true ())
       Manifest.Executed_yet_catalogue_only);
  let manifest = [ "b.ml"; "a.ml"; "c.ml" ] in
  Verdict.p "a manifest source missing from the argument list is named as uncatalogued"
    (Poly.equal (catalogue_mismatch ~manifest ~catalogued:[ "c.ml"; "a.ml" ]) ([ "b.ml" ], [], []));
  Verdict.p "an argument-list source outside the manifest is named as extra"
    (Poly.equal
       (catalogue_mismatch ~manifest ~catalogued:[ "a.ml"; "d.ml"; "b.ml"; "c.ml" ])
       ([], [ "d.ml" ], []));
  Verdict.p "a source catalogued twice is a mismatch, not a match"
    (Poly.equal
       (catalogue_mismatch ~manifest ~catalogued:[ "a.ml"; "b.ml"; "c.ml"; "a.ml" ])
       ([], [ "a.ml" ], []));
  Verdict.p "a manifest row repeated is a mismatch even where the argument list repeats it too"
    (Poly.equal
       (catalogue_mismatch ~manifest:("a.ml" :: manifest)
          ~catalogued:[ "a.ml"; "b.ml"; "c.ml"; "a.ml" ])
       ([], [ "a.ml" ], [ "a.ml" ]));
  Verdict.p "the same sources in another order match"
    (Poly.equal (catalogue_mismatch ~manifest ~catalogued:[ "c.ml"; "a.ml"; "b.ml" ]) ([], [], []));
  let arguments = Array.to_list (Array.subo Stdlib.Sys.argv ~pos:1) in
  let rec pairs = function
    | source :: control :: rest -> (source, control) :: pairs rest
    | [] -> []
    | [ dangling ] ->
        Verdict.fail (Printf.sprintf "scanner source %s has no assigned control golden" dangling);
        []
  in
  let pairs = pairs arguments in
  let uncatalogued, extra, repeated =
    catalogue_mismatch ~manifest:Manifest.sources
      ~catalogued:(List.map pairs ~f:(fun (source, _) -> "test/operations/" ^ source))
  in
  List.iter uncatalogued ~f:(fun source ->
      eprintf
        "%s: in Refusal_control_manifest.sources but not catalogued -- add it and its control \
         goldens to the refusal_control_scan_cases stanza in test/operations/dune (not part of the \
         golden)\n"
        source);
  List.iter extra ~f:(fun source ->
      eprintf
        "%s: catalogued by the refusal_control_scan_cases stanza %s (not part of the golden)\n"
        source
        (if List.mem Manifest.sources source ~equal:String.equal then "more than once"
         else "but absent from Refusal_control_manifest.sources"));
  List.iter repeated ~f:(fun source ->
      eprintf
        "%s: has more than one row in `raw_entries` in test/support/refusal_control_manifest.ml \
         (not part of the golden)\n"
        source);
  Verdict.p_empty
    "the stanza's scanner sources are exactly Refusal_control_manifest.sources, each once"
    ~over:Manifest.sources
    (uncatalogued @ extra @ repeated);
  printf "\nScanner refusal formats and the permanent control suite assigned to their source:\n";
  let catalogued = Hashtbl.create (module String) in
  pairs
  |> List.iter ~f:(fun (source, control) ->
      let controls = String.split control ~on:',' in
      let control_text = controls |> List.map ~f:In_channel.read_all |> String.concat ~sep:"\n" in
      let diagnostics = Scan.diagnostics (In_channel.read_all source) in
      let extracted = List.map diagnostics ~f:Scan.marker in
      let source_key = "test/operations/" ^ source in
      let coverage = Scan.coverage ~control_text diagnostics in
      Hashtbl.set catalogued ~key:source_key ~data:diagnostics;
      let registered = Manifest.markers source_key in
      let row_holds = List.equal String.equal extracted registered in
      Verdict.p
        (Printf.sprintf "%s has exactly the explicitly assigned refusal controls" source)
        row_holds;
      (* A reworded format changes its marker: name both sides of the difference and the row to
         paste, so the author never has to recover a digest from another failure line. *)
      Option.iter
        (Manifest.row_difference ~registered ~extracted)
        ~f:(Manifest.eprint_row_difference ~source diagnostics);
      List.iter2_exn diagnostics coverage ~f:(fun diagnostic covered ->
          Verdict.p
            (Printf.sprintf "%s: %s (%s) is catalogued beside %s" source (Scan.marker diagnostic)
               (match diagnostic.Scan.kind with
               | Scan.Fail -> "direct failure"
               | Scan.Claim -> "claim")
               (String.concat ~sep:", " controls))
            covered));
  let stale = Manifest.stale_classifications ~diagnostics_of:(Hashtbl.find catalogued) in
  List.iter stale ~f:(fun key ->
      eprintf
        "%s: `raw_direct_evidence` or `raw_catalogue_only` key answers to no current \
         direct-failure diagnostic of its source; re-key it from the row difference above or drop \
         it (not part of the golden)\n"
        (Option.value (String.chop_prefix key ~prefix:"test/operations/") ~default:key));
  Verdict.p_empty
    "every `raw_direct_evidence` and `raw_catalogue_only` key names a current direct failure of a \
     catalogued scanner source"
    ~over:(Manifest.direct_evidence @ Manifest.catalogue_only)
    stale;
  List.iter Manifest.doubly_classified
    ~f:
      (eprintf
         "%s: in both `raw_direct_evidence` and `raw_catalogue_only` (not part of the golden)\n");
  Verdict.p_empty
    "no direct failure is both answered by a control and catalogued as answered by none"
    ~over:Manifest.direct_evidence Manifest.doubly_classified
