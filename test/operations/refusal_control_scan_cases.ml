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
  let stale = Manifest.stale_direct_evidence ~diagnostics_of:(Hashtbl.find catalogued) in
  List.iter stale ~f:(fun key ->
      eprintf
        "%s: `raw_direct_evidence` key answers to no current direct-failure diagnostic of its \
         source; re-key it from the row difference above or drop it (not part of the golden)\n"
        (Option.value (String.chop_prefix key ~prefix:"test/operations/") ~default:key));
  Verdict.p_empty
    "every `raw_direct_evidence` key names a current direct failure of a catalogued scanner source"
    ~over:Manifest.direct_evidence stale
