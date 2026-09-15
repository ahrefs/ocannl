(** The source reader and relationship behind gh-ocannl-800, on diagnostics and goldens the
    repository does not have to contain. *)

open Base
open Stdio
module Scan = Test_utils.Refusal_control_scan
module Manifest = Test_utils.Refusal_control_manifest

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
  let arguments = Array.to_list (Array.subo Stdlib.Sys.argv ~pos:1) in
  let rec pairs = function
    | source :: control :: rest -> (source, control) :: pairs rest
    | [] -> []
    | [ dangling ] ->
        Verdict.fail (Printf.sprintf "scanner source %s has no assigned control golden" dangling);
        []
  in
  printf "\nScanner refusal formats and the permanent control suite assigned to their source:\n";
  let catalogued = Hashtbl.create (module String) in
  pairs arguments
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
      if not row_holds then (
        (* Markers repeat when formats do, so each side is a multiset: [minus xs ys] removes one
           occurrence per [ys] element. A move within the source leaves both sides empty and differs
           only in order, reported at its first differing position. *)
        let minus xs ys =
          List.fold ys ~init:xs ~f:(fun remaining y ->
              let before, after = List.split_while remaining ~f:(Fn.non (String.equal y)) in
              before @ Option.value (List.tl after) ~default:[])
        in
        eprintf "%s: refusal-control row differs from extraction (not part of the golden):\n" source;
        let absent = minus extracted registered and no_longer = minus registered extracted in
        List.iter absent ~f:(eprintf "  extracted, absent from the row: %s\n");
        List.iter no_longer ~f:(eprintf "  in the row, no longer extracted: %s\n");
        (if List.is_empty absent && List.is_empty no_longer then
           match
             List.findi (List.zip_exn extracted registered) ~f:(fun _ (e, r) ->
                 not (String.equal e r))
           with
           | Some (position, (e, r)) ->
               eprintf
                 "  same markers in a different order; first difference at entry %d: extracted %s, \
                  the row has %s\n"
                 (position + 1) e r
           | None -> ());
        eprintf
          "  replace its `raw_entries` row in test/support/refusal_control_manifest.ml with:\n%s"
          (Manifest.row ~source diagnostics));
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
