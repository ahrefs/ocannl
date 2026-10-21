(* tools/refusal-manifest-sync.sh's engine (gh-ocannl-1429) on a manifest and scanners the
   repository does not have to contain: what a rewording determines is rewritten, what it does not
   is left unresolved, and the text reader agrees with the compiled tables of the real manifest. *)

open Base
open Stdio
open Verdict.Claims
module Sync = Test_utils.Refusal_manifest_sync
module Scan = Test_utils.Refusal_control_scan
module Manifest = Test_utils.Refusal_control_manifest

let markers source = List.map (Scan.diagnostics source) ~f:Scan.marker

(* A synthetic scanner before and after a rewording of its second direct failure and of the control
   claim the evidence entry names; the first failure and the last claim keep their wording. *)
let before =
  {ocaml|
let check name ok =
  if not ok then Verdict.fail (Printf.sprintf "%s keeps the first refusal" name);
  Verdict.fail "the second refusal reads the old way";
  Verdict.p "the malformed tree is refused" ok;
  Verdict.p "the last claim keeps its words" ok
|ocaml}

let after =
  {ocaml|
let check name ok =
  if not ok then Verdict.fail (Printf.sprintf "%s keeps the first refusal" name);
  Verdict.fail "the second refusal reads a new way";
  Verdict.p "a malformed tree is refused" ok;
  Verdict.p "the last claim keeps its words" ok
|ocaml}

let identity_of source format =
  List.find_map_exn (Scan.diagnostics source) ~f:(fun d ->
      Option.some_if (String.equal d.Scan.format format) d.Scan.identity)

let quote = Sync.quote

(* A manifest text in the real one's layout, with [rows], [evidence] and [catalogue] entries. *)
let manifest ~rows ~evidence ~catalogue =
  let entries entries =
    String.concat
      (List.map entries ~f:(fun (key, value) ->
           Printf.sprintf "    ( %s,\n      %s );\n" (quote key) (quote value)))
  in
  String.concat
    [
      "open Base\n\nlet raw_entries =\n  [\n";
      String.concat
        (List.map rows ~f:(fun (source, markers) ->
             Printf.sprintf "    ( %s,\n      [\n%s      ] );\n" (quote source)
               (String.concat
                  (List.map markers ~f:(fun m -> Printf.sprintf "        %s;\n" (quote m))))));
      "  ]\n\nlet raw_direct_evidence =\n  [\n";
      entries evidence;
      "  ]\n\nlet raw_catalogue_only : (string * string) list =\n  [\n";
      entries catalogue;
      "  ]\n";
    ]

let () =
  (* [align]: kept elements anchor the comparison; equal-length runs between them pair in order,
     unequal ones pair nothing. *)
  let align before after = Sync.align ~equal:String.equal ~before ~after in
  p "a list aligned with itself pairs, drops and inserts nothing"
    (Poly.equal (align [ "a"; "b"; "c" ] [ "a"; "b"; "c" ]) ([], [], []));
  p "an element replaced between two kept ones pairs with its replacement"
    (Poly.equal (align [ "a"; "b"; "c" ] [ "a"; "B"; "c" ]) ([ ("b", "B") ], [], []));
  p "two adjacent replacements pair in order"
    (Poly.equal
       (align [ "a"; "b"; "c"; "d" ] [ "a"; "B"; "C"; "d" ])
       ([ ("b", "B"); ("c", "C") ], [], []));
  p "a replacement beside an insertion pairs nothing: which replaced which is not determined"
    (Poly.equal (align [ "a"; "b"; "c" ] [ "a"; "B"; "x"; "c" ]) ([], [ "b" ], [ "B"; "x" ]));
  p "a replacement at either end still pairs"
    (Poly.equal (align [ "a"; "b"; "c" ] [ "A"; "b"; "C" ]) ([ ("a", "A"); ("c", "C") ], [], []));
  (* [plan] over a synthetic manifest whose row is [before]'s extraction. *)
  let second_before = identity_of before "the second refusal reads the old way"
  and second_after = identity_of after "the second refusal reads a new way"
  and first = identity_of before "%s keeps the first refusal" in
  let text =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers before) ]
      ~evidence:
        [
          ("synthetic_scan.ml:" ^ second_before, "the malformed tree is refused: true");
          ("synthetic_scan.ml:" ^ first, "the last claim keeps its words: true");
        ]
      ~catalogue:[]
  in
  let diagnostics_of source =
    Option.some_if (String.equal source "synthetic_scan.ml") (Scan.diagnostics after)
  in
  let plan = Sync.plan ~text ~diagnostics_of ~renamed_claims:[] () in
  p_empty "a rewording the row pairs one for one leaves nothing unresolved" ~over:plan.changes
    plan.unresolved;
  let synced = Sync.read_tables (Sync.apply text plan.edits) in
  p "the rewritten row is the new extraction, in order"
    (match synced.rows with
    | [ { source = "synthetic_scan.ml"; markers = row; _ } ] ->
        List.equal String.equal row (markers after)
    | _ -> false);
  p "the reworded direct failure's evidence key follows it, its unchanged sibling's stays"
    (List.equal String.equal
       (List.map synced.direct_evidence ~f:(fun e -> e.key.value))
       [ "synthetic_scan.ml:" ^ second_after; "synthetic_scan.ml:" ^ first ]);
  p "the reworded control claim's label follows it, an unchanged claim's stays"
    (List.equal String.equal
       (List.map synced.direct_evidence ~f:(fun e -> e.value.value))
       [ "a malformed tree is refused: true"; "the last claim keeps its words: true" ]);
  p_empty "a synced manifest plans nothing more" ~over:plan.edits
    (Sync.plan ~text:(Sync.apply text plan.edits) ~diagnostics_of ~renamed_claims:[] ()).edits;
  p "the plan names the row, the re-key and the relabel"
    (match plan.changes with
    | [ Row { source = "synthetic_scan.ml"; added; removed }; Rekeyed _; Relabeled _ ] ->
        List.length added = 2 && List.length removed = 2
    | _ -> false);
  (* The goldens: each replaced marker swapped, each reworded claim's row relabeled, nothing
     else. *)
  let golden_before =
    String.concat ~sep:"\n"
      ([ "the malformed tree is refused: true"; "the malformed tree is refused but not here: true" ]
      @ List.map (markers before) ~f:(fun m -> "  " ^ m)
      @ [ "End of synthetic controls."; "" ])
  in
  let golden_after =
    String.concat ~sep:"\n"
      ([ "a malformed tree is refused: true"; "the malformed tree is refused but not here: true" ]
      @ List.map (markers after) ~f:(fun m -> "  " ^ m)
      @ [ "End of synthetic controls."; "" ])
  in
  p "a golden's replaced markers and reworded claim rows are rewritten, and nothing else"
    (String.equal (Sync.rewrite_golden plan golden_before) golden_after);
  p "a golden holding none of them is unchanged"
    (String.equal (Sync.rewrite_golden plan "unrelated: true\n") "unrelated: true\n");
  (* What a rewording does not determine stays a hand edit. *)
  let unresolved ~after ?(evidence = []) ?(catalogue = []) () =
    let diagnostics_of source =
      Option.some_if (String.equal source "synthetic_scan.ml") (Scan.diagnostics after)
    in
    let text = manifest ~rows:[ ("synthetic_scan.ml", markers before) ] ~evidence ~catalogue in
    (Sync.plan ~text ~diagnostics_of ~renamed_claims:[] ()).unresolved
  in
  let inserted =
    String.substr_replace_first after ~pattern:"  Verdict.p \"a malformed"
      ~with_:"  Verdict.fail \"an inserted refusal\";\n  Verdict.p \"a malformed"
  in
  p "a key whose failure was reworded beside an insertion is unresolved, not guessed"
    (List.length
       (unresolved ~after:inserted
          ~catalogue:[ ("synthetic_scan.ml:" ^ second_before, "reason") ]
          ())
    = 1);
  let into_claim =
    String.substr_replace_first after ~pattern:"Verdict.fail \"the second refusal reads a new way\""
      ~with_:"Verdict.p \"the second refusal reads a new way\" ok"
  in
  p "a direct failure reworded into a claim leaves its key unresolved"
    (List.length
       (unresolved ~after:into_claim
          ~evidence:[ ("synthetic_scan.ml:" ^ second_before, "x: true") ]
          ())
    = 1);
  let same_words =
    String.substr_replace_first before
      ~pattern:"Verdict.fail \"the second refusal reads the old way\""
      ~with_:"Verdict.p \"the second refusal reads the old way\" ok"
  in
  p "a direct failure turned into a claim of the same words leaves its key unresolved"
    (List.length
       (unresolved ~after:same_words
          ~evidence:[ ("synthetic_scan.ml:" ^ second_before, "x: true") ]
          ())
    = 1);
  let formatted =
    String.substr_replace_first after ~pattern:"Verdict.p \"a malformed tree is refused\" ok"
      ~with_:"Verdict.pf \"a malformed %s is refused\" name ok"
  in
  p "a control claim reworded into a format with substitutions leaves its label unresolved"
    (List.length
       (unresolved ~after:formatted
          ~evidence:[ ("synthetic_scan.ml:" ^ first, "the malformed tree is refused: true") ]
          ())
    = 1);
  p "a key that answers to no current failure and no rewording is unresolved"
    (List.length
       (unresolved ~after:before
          ~evidence:[ ("synthetic_scan.ml:00000000000000000000000000000000", "x: true") ]
          ())
    = 1);
  (* A control claim outside the rows, reworded in its own file: [claim_renames] pairs it. *)
  let renamed_claims, unpaired_claims = Sync.claim_renames ~before ~after in
  p "a control file's reworded claim is paired old to new, its unchanged one is not"
    (Poly.equal renamed_claims [ ("the malformed tree is refused", "a malformed tree is refused") ]);
  p_empty "and a one-for-one rewording leaves no old claim unpaired" ~over:renamed_claims
    unpaired_claims;
  let outside =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers after) ]
      ~evidence:[ ("synthetic_scan.ml:" ^ first, "the malformed tree is refused: true") ]
      ~catalogue:[]
  in
  let outside_plan = Sync.plan ~text:outside ~diagnostics_of ~renamed_claims () in
  (* Not followed by text: the same label may back another scanner's evidence from another file. *)
  p
    "an evidence label naming a claim reworded outside the scanner sources is unresolved, with the \
     new text as its hint"
    (match outside_plan.unresolved with
    | [ message ] -> String.is_substring message ~substring:"a malformed tree is refused"
    | _ -> false);
  let outside_evidence = (Sync.read_tables outside).direct_evidence in
  p_empty "and is not relabeled" ~over:outside_evidence outside_plan.edits;
  (* The same control file with a claim inserted beside the rewording: nothing pairs the old claim,
     so the label naming it is unresolved rather than left stale behind "nothing to sync". *)
  let beside =
    String.substr_replace_first after ~pattern:"  Verdict.p \"a malformed"
      ~with_:"  Verdict.p \"an inserted claim\" ok;\n  Verdict.p \"a malformed"
  in
  let renamed_claims, unpaired_claims = Sync.claim_renames ~before ~after:beside in
  let beside_plan = Sync.plan ~text:outside ~diagnostics_of ~renamed_claims ~unpaired_claims () in
  p "a control file's claim reworded beside an insertion leaves the label naming it unresolved"
    (List.length beside_plan.unresolved = 1);
  p_empty "and relabels nothing" ~over:outside_evidence beside_plan.edits;
  (* One format at two sites, its occurrences reworded differently: no single successor. *)
  let twice =
    {ocaml|
let check () =
  Verdict.fail "a refusal at two sites";
  Verdict.fail "a refusal at two sites"
|ocaml}
  in
  let twice_after =
    {ocaml|
let check () =
  Verdict.fail "a refusal at the first site";
  Verdict.fail "a refusal at the second site"
|ocaml}
  in
  let twice_key = "synthetic_scan.ml:" ^ identity_of twice "a refusal at two sites" in
  let twice_text =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers twice) ]
      ~evidence:[ (twice_key, "x: true") ]
      ~catalogue:[]
  in
  let twice_plan =
    Sync.plan ~text:twice_text
      ~diagnostics_of:(fun _ -> Some (Scan.diagnostics twice_after))
      ~renamed_claims:[] ()
  in
  p "a key whose format's two sites were reworded differently is unresolved, as is its marker"
    (List.length twice_plan.unresolved = 2);
  p_none ~min:1 "and no entry is re-keyed to either successor (the row is still rewritten)"
    twice_plan.changes ~f:(function
    | Sync.Rekeyed _ -> true
    | Row _ | Relabeled _ -> false);
  p "and no golden swaps its marker for either successor"
    (let golden = String.concat (List.map (markers twice) ~f:(fun m -> m ^ "\n")) in
     String.equal (Sync.rewrite_golden twice_plan golden) golden);
  (* One site of a claim reworded while the same claim survives at another: the label still names
     the surviving site, so nothing is relabeled or swapped, and the row's reading is unresolved. *)
  let two_sites =
    {ocaml|
let check ok =
  Verdict.fail "a refusal that stays";
  Verdict.p "the malformed tree is refused" ok;
  Verdict.p "the malformed tree is refused" ok
|ocaml}
  in
  let one_site_reworded =
    String.substr_replace_first two_sites ~pattern:"the malformed tree is refused\" ok;"
      ~with_:"a malformed tree is refused\" ok;"
  in
  let survivor_text =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers two_sites) ]
      ~evidence:
        [
          ( "synthetic_scan.ml:" ^ identity_of two_sites "a refusal that stays",
            "the malformed tree is refused: true" );
        ]
      ~catalogue:[]
  in
  let survivor_plan =
    Sync.plan ~text:survivor_text
      ~diagnostics_of:(fun _ -> Some (Scan.diagnostics one_site_reworded))
      ~renamed_claims:[] ()
  in
  p "a claim reworded at one site while it survives at another is unresolved"
    (List.length survivor_plan.unresolved = 1);
  p_none ~min:1 "and no evidence label follows it (the row is still rewritten)"
    survivor_plan.changes ~f:(function
    | Sync.Relabeled _ -> true
    | Row _ | Rekeyed _ -> false);
  p_empty "nor does a control file's rename of it: the label names the surviving site"
    ~over:(Scan.diagnostics two_sites)
    (fst (Sync.claim_renames ~before:two_sites ~after:one_site_reworded));
  p_all ~min:2 "a control file deleted since the base leaves every old claim unpaired"
    (List.filter (Scan.diagnostics before) ~f:(fun d -> Poly.equal d.Scan.kind Scan.Claim))
    ~f:(fun d ->
      List.mem (snd (Sync.claim_renames ~before ~after:"")) d.Scan.format ~equal:String.equal);
  (* A formatted control claim reworded: the evidence label is what it RENDERED, no format's
     identity, so it is matched against the retired format and left a hand edit. *)
  let formatted_before =
    {ocaml|
let check name ok = Verdict.pf "the %s tree is refused" name ok
|ocaml}
  in
  let formatted_after =
    {ocaml|
let check name ok = Verdict.pf "a %s tree is refused" name ok
|ocaml}
  in
  let renamed_claims, unpaired_claims =
    Sync.claim_renames ~before:formatted_before ~after:formatted_after
  in
  let rendered_plan = Sync.plan ~text:outside ~diagnostics_of ~renamed_claims ~unpaired_claims () in
  p "an evidence label a reworded formatted claim rendered is unresolved, not left stale"
    (List.length rendered_plan.unresolved = 1);
  p_empty "and is not rewritten" ~over:outside_evidence rendered_plan.edits;
  (* A claim the row leaves unpaired -- reworded beside an inserted failure -- is not renamed by the
     claims-only pairing of the same source, which never sees the failure. *)
  let beside_failure =
    String.substr_replace_first after ~pattern:"  Verdict.p \"a malformed"
      ~with_:"  Verdict.fail \"an inserted refusal\";\n  Verdict.p \"a malformed"
  in
  let renamed_claims, unpaired_claims = Sync.claim_renames ~before ~after:beside_failure in
  p "the claims-only pairing alone does pair the reworded claim" (List.length renamed_claims = 1);
  let row_text =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers before) ]
      ~evidence:[ ("synthetic_scan.ml:" ^ first, "the malformed tree is refused: true") ]
      ~catalogue:[]
  in
  let row_plan =
    Sync.plan ~text:row_text
      ~diagnostics_of:(fun _ -> Some (Scan.diagnostics beside_failure))
      ~renamed_claims ~unpaired_claims ()
  in
  p_none ~min:1 "but the row's alignment outranks it: the evidence label is not relabeled"
    row_plan.changes ~f:(function
    | Sync.Relabeled _ -> true
    | Row _ | Rekeyed _ -> false);
  p "and the label naming it is unresolved"
    (List.exists row_plan.unresolved ~f:(String.is_substring ~substring:"nothing pairs it"));
  (* Two claims that only swap places: the one the alignment drops still occurs, so nothing is
     unpaired. *)
  let swapped =
    {ocaml|
let check ok =
  Verdict.p "the last claim keeps its words" ok;
  Verdict.p "the malformed tree is refused" ok
|ocaml}
  and in_order =
    {ocaml|
let check ok =
  Verdict.p "the malformed tree is refused" ok;
  Verdict.p "the last claim keeps its words" ok
|ocaml}
  in
  p_empty "two claims that only swap places leave nothing unpaired"
    ~over:(Scan.diagnostics in_order)
    (snd (Sync.claim_renames ~before:in_order ~after:swapped));
  (* A claim two scanner sources share, reworded in one: a relabel or golden swap by text would move
     the other's too. *)
  let other_scan = {ocaml|
let check ok = Verdict.p "the malformed tree is refused" ok
|ocaml} in
  let shared_text =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers before); ("other_scan.ml", markers other_scan) ]
      ~evidence:[ ("synthetic_scan.ml:" ^ second_before, "the malformed tree is refused: true") ]
      ~catalogue:[]
  in
  let shared_plan =
    Sync.plan ~text:shared_text
      ~diagnostics_of:(fun source ->
        Some (Scan.diagnostics (if String.equal source "other_scan.ml" then other_scan else after)))
      ~renamed_claims:[] ()
  in
  p "a claim another scanner source still makes, reworded in one, is unresolved"
    (List.exists shared_plan.unresolved ~f:(String.is_substring ~substring:"also another scanner"));
  p_none ~min:1 "and no evidence label follows it (the row is still rewritten)" shared_plan.changes
    ~f:(function
    | Sync.Relabeled _ -> true
    | Row _ | Rekeyed _ -> false);
  (* The text reader against the real manifest's compiled tables: every row, key and label it would
     rewrite is the one the tests read. *)
  let live_text = In_channel.read_all "../support/refusal_control_manifest.ml" in
  let live = Sync.read_tables live_text in
  p "the text reader reads the real manifest's rows as compiled"
    (Poly.equal (List.map live.rows ~f:(fun r -> (r.source, r.markers))) Manifest.raw_entries);
  let pairs entries = List.map entries ~f:(fun (e : Sync.entry) -> (e.key.value, e.value.value)) in
  p "and its direct-evidence entries"
    (Poly.equal (pairs live.direct_evidence) Manifest.raw_direct_evidence);
  p "and its catalogue-only entries"
    (Poly.equal (pairs live.catalogue_only) Manifest.raw_catalogue_only);
  p_all ~min:20 "every row's marker list spans a bracketed list of the manifest text" live.rows
    ~f:(fun r ->
      Char.equal live_text.[r.list_start] '[' && Char.equal live_text.[r.list_stop - 1] ']')
