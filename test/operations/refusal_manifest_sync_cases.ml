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
  let plan = Sync.plan ~text ~diagnostics_of ~renamed_claims:[] in
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
  p "a synced manifest plans nothing more"
    (List.is_empty
       (Sync.plan ~text:(Sync.apply text plan.edits) ~diagnostics_of ~renamed_claims:[]).edits);
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
    (Sync.plan ~text ~diagnostics_of ~renamed_claims:[]).unresolved
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
  let renamed_claims = Sync.claim_renames ~before ~after in
  p "a control file's reworded claim is paired old to new, its unchanged one is not"
    (Poly.equal renamed_claims [ ("the malformed tree is refused", "a malformed tree is refused") ]);
  let outside =
    manifest
      ~rows:[ ("synthetic_scan.ml", markers after) ]
      ~evidence:[ ("synthetic_scan.ml:" ^ first, "the malformed tree is refused: true") ]
      ~catalogue:[]
  in
  let outside_plan = Sync.plan ~text:outside ~diagnostics_of ~renamed_claims in
  p "an evidence label naming a claim reworded outside the rows follows it"
    (List.equal String.equal
       (List.map (Sync.read_tables (Sync.apply outside outside_plan.edits)).direct_evidence
          ~f:(fun e -> e.value.value))
       [ "a malformed tree is refused: true" ]);
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
